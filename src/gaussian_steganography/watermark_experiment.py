from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from gaussian2d import Gaussian2DRenderer
from run_fit import build_optimizer, build_scheduler, compute_psnr, initialize_renderer
from stego_ideas_proposal import JointParityStegoCoder, MuStegoCoder, ScaleTierStegoCoder, ThetaStegoCoder
from stego_protocol import (
    COST_STEGO_V1_PROTOCOL,
    COST_STEGO_V2_PROTOCOL,
    MU_JITTER_PROTOCOL,
    JOINT_PARITY_PROTOCOL,
    SCALE_TIER_PROTOCOL,
    THETA_DISTRIBUTION_PROTOCOL,
    DEFAULT_SECURITY_MODEL,
    DEFAULT_STEGO_PROTOCOL,
    RunLevelStegaDetector,
    active_channel_indices,
    adapt_wet_mask_for_capacity,
    assignment_bit_tensor,
    build_channel_costs,
    build_cost_code_assignments,
    build_generic_cost_code_assignments,
    build_greedy_nokey_cost_code_assignments,
    build_random_cost_code_assignments,
    build_random_polarity_cost_code_assignments,
    build_density_reference,
    build_detector_feature_vector,
    build_soft_selection_field,
    build_unkeyed_selection_field,
    carrier_margin_loss,
    capacity_failure_report,
    carrier_ewc_penalty,
    cost_code_action_margin_loss,
    cost_code_action_target_loss,
    cost_code_block_probabilities,
    covariance_and_circular_shift,
    decode_cost_code,
    decode_image_cost_code,
    decode_parity_blocks,
    default_spread_factor,
    density_reference_loss,
    detector_embed_loss,
    detector_feature_names,
    generic_cost_code_probabilities,
    grouped_pooled_detection_statistics,
    mmd_distribution_loss,
    modified_gaussian_count,
    parity_block_logits,
    pooled_detection_statistics,
    rgb_to_ycbcr,
    selection_channel_penalty,
    select_compensation_neighbors,
    select_keyed_parity_blocks,
    serializable_block_assignments,
    sliced_wasserstein_loss,
    train_detector_steps,
    wrong_key_contrastive_loss,
    ycbcr_to_rgb,
)
from utils import canonicalize_gaussian_geometry, compute_gaussian_importance, export_parameter_table, get_best_device, save_comparison_figure, save_json, save_metrics_card, save_raw_tensor_image, set_seed
from watermark_utils import (
    LOG_ANISOTROPY_MODE,
    LEGACY_LOGRATIO_MODE,
    THETA_MAJOR_MODE,
    anchor_loss_from_scales,
    bits_to_string,
    build_bits,
    build_clean_distribution_reference,
    build_renderer_from_export,
    carrier_sensitivity_field,
    carrier_values_from_canonical,
    decode_bits,
    detectability_auc_from_bootstrap,
    distribution_distance_summary,
    evaluate_renderer,
    experiment_config_from_export,
    hamming74_decode,
    hamming74_encode,
    key_to_hash,
    key_to_seed,
    keyed_adjustment_selection,
    load_exported_run,
    load_target_tensor,
    make_grid_from_run,
    matched_moment_loss,
    message_loss,
    mode_uses_scale_carrier,
    compute_ssim,
    select_low_importance_carriers,
    select_sensitivity_candidates,
    select_topk_carriers,
    summarize_run_features,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
VALID_MODES = (LOG_ANISOTROPY_MODE, THETA_MAJOR_MODE, LEGACY_LOGRATIO_MODE)
VALID_CARRIER_POOLS = ("topk", "low_importance", "sensitivity_topk")
VALID_STAGES = ("postfit", "joint")
VALID_ECC = ("none", "hamming74")
VALID_CARRIER_SCORE = ("importance", "sensitivity_v2")
PROTOTYPE_PROTOCOLS = (
    THETA_DISTRIBUTION_PROTOCOL,
    MU_JITTER_PROTOCOL,
    SCALE_TIER_PROTOCOL,
    JOINT_PARITY_PROTOCOL,
)
VALID_PROTOCOLS = ("legacy_wm", COST_STEGO_V1_PROTOCOL, COST_STEGO_V2_PROTOCOL, *PROTOTYPE_PROTOCOLS)
VALID_CODING_MODES = (
    "hamming74",
    "parity_v1",
    "cost_code_v1",
    "theta_distribution_v1",
    "mu_jitter_v1",
    "scale_tier_v1",
    "joint_parity_v1",
)
VALID_BASELINE_TYPES = (
    "native",
    "native_matched_random",
    "native_greedy_nokey",
    "native_random_polarity",
    "pixel",
    "render_refit",
    "theta_distribution",
    "mu_jitter",
    "scale_tier",
    "joint_parity",
)
VALID_NATIVE_ABLATIONS = ("full", "no_detector", "no_wrong_key", "no_attack_training")
VALID_CLAIM_TRACKS = ("full", "no_detector")
MAIN_CLAIM_LABEL = "main_native_8"
HEURISTIC_PARAMETER_BASELINES = {"native_greedy_nokey", "native_random_polarity"}
KEYED_PARAMETER_BASELINES = {"native", "native_matched_random"}
NATIVE_PARAMETER_BASELINES = KEYED_PARAMETER_BASELINES | HEURISTIC_PARAMETER_BASELINES
PROTOTYPE_BASELINES = {"theta_distribution", "mu_jitter", "scale_tier", "joint_parity"}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed a distribution-aware white-box watermark into exported Gaussian parameters.")
    parser.add_argument("--source-run", type=str, required=True, help="Path to a clean exported run directory.")
    parser.add_argument("--protocol", choices=VALID_PROTOCOLS, default="legacy_wm", help="Embedding protocol to run.")
    parser.add_argument("--coding-mode", choices=VALID_CODING_MODES, default="hamming74", help="Coding rule used by the selected protocol.")
    parser.add_argument("--mode", choices=VALID_MODES, default=LOG_ANISOTROPY_MODE, help="Carrier variable and decoding rule.")
    parser.add_argument("--carrier-pool", choices=VALID_CARRIER_POOLS, default="topk", help="Carrier pool selection strategy.")
    parser.add_argument("--stage", choices=VALID_STAGES, default="postfit", help="Optimization stage used for watermark embedding.")
    parser.add_argument("--baseline-type", choices=VALID_BASELINE_TYPES, default="native", help="Which steganography baseline variant to export.")
    parser.add_argument("--native-ablation", choices=VALID_NATIVE_ABLATIONS, default="full", help="Native cost_stego_v2 ablation variant used by the study runner.")
    parser.add_argument(
        "--claim-track",
        choices=VALID_CLAIM_TRACKS,
        default="full",
        help="Which native track is allowed to carry the main paper claim. "
             "'full' keeps the original detector-enabled line; 'no_detector' promotes the detector-free line.",
    )
    parser.add_argument("--payload-bits", type=positive_int, default=16, help="Number of raw message bits to embed.")
    parser.add_argument("--message", type=str, default=None, help="Optional explicit bitstring.")
    parser.add_argument("--message-seed", type=int, default=0, help="Seed used when generating random bits.")
    parser.add_argument("--msg-strength", type=positive_float, default=0.01, help="Weight of the message loss.")
    parser.add_argument("--dist-weight", type=float, default=0.0, help="Weight of the matched-moments distribution loss.")
    parser.add_argument("--anchor-weight", type=float, default=0.0, help="Weight of the sensitivity-weighted anchor loss.")
    parser.add_argument("--key", type=str, default="ucl-2d-watermark", help="Key used for deterministic carrier permutation.")
    parser.add_argument("--ecc", choices=VALID_ECC, default="none", help="Optional error-correcting code applied before embedding.")
    parser.add_argument("--candidate-factor", type=positive_int, default=3, help="Multiplier used when building the sensitivity-aware candidate pool.")
    parser.add_argument("--carrier-score", choices=VALID_CARRIER_SCORE, default="sensitivity_v2", help="Score used by sensitivity-aware carrier selection.")
    parser.add_argument("--tune-steps", type=positive_int, default=200, help="Post-fit fine-tuning steps.")
    parser.add_argument("--tune-lr", type=positive_float, default=0.01, help="Learning rate for watermark fine-tuning.")
    parser.add_argument("--joint-steps", type=positive_int, default=None, help="Optional override for the number of joint-optimization steps.")
    parser.add_argument("--target-magnitude", type=positive_float, default=0.35, help="Absolute bin target used by the message loss.")
    parser.add_argument("--theta-delta", type=positive_float, default=0.08, help="Per-pass rotation nudge used by theta-distribution prototypes.")
    parser.add_argument("--theta-target-spread", type=positive_float, default=0.78539816339, help="Angular separation used by theta-distribution prototypes.")
    parser.add_argument("--prototype-max-passes", type=positive_int, default=4, help="Maximum encode-strengthening passes used by prototype protocols that iterate decode feedback.")
    parser.add_argument("--mu-epsilon", type=positive_float, default=0.02, help="Maximum center displacement used by mu-jitter prototypes.")
    parser.add_argument("--scale-tier-delta-scale", type=positive_float, default=0.06, help="Multiplicative scale boost used by scale-tier prototypes.")
    parser.add_argument("--mmd-weight", type=float, default=None, help="Differentiable MMD covertness weight used by cost_stego_v1.")
    parser.add_argument("--detector-weight", type=float, default=None, help="Adversarial detector weight used by cost_stego_v1.")
    parser.add_argument("--alpha-weight", type=float, default=0.5, help="Relative contribution of the alpha channel in cost_stego_v1.")
    parser.add_argument("--ewc-weight", type=float, default=None, help="Carrier stabilization weight used by cost_stego_v1.")
    parser.add_argument("--wasserstein-weight", type=float, default=0.05, help="Sliced Wasserstein covertness weight used by cost_stego_v2.")
    parser.add_argument("--selection-penalty-weight", type=float, default=0.05, help="Selection-channel penalty weight used by cost_stego_v2.")
    parser.add_argument("--selection-temperature", type=positive_float, default=0.5, help="Temperature used by keyed Gumbel-top-k sampling.")
    parser.add_argument("--max-modification-ratio", type=float, default=None, help="Maximum fraction of Gaussians that cost_stego_v2 may modify.")
    parser.add_argument(
        "--adaptive-native-budget",
        action="store_true",
        help="Enable a cover-adaptive cost_stego_v2 budget rule for flat, broad parameter covers. "
             "This is intended for fragile cases such as stripe-like synthetic fits without changing easy covers.",
    )
    parser.add_argument("--wrong-key-samples", type=positive_int, default=4, help="Number of wrong keys sampled by the v2 contrastive loss.")
    parser.add_argument("--beam-width", type=positive_int, default=64, help="Beam width used by cost_code_v1 block solving.")
    parser.add_argument("--covertness-weight", type=float, default=1.0, help="Global multiplier applied to covertness losses.")
    parser.add_argument("--wet-threshold", type=float, default=0.20, help="Quantile threshold used by the wet mask rules.")
    # S0-C: Cost weight parameters for ablation studies.  Defaults match the original design.
    parser.add_argument("--cost-sensitivity-weight", type=float, default=0.40, help="Weight for the rendering-sensitivity component of the channel cost model.")
    parser.add_argument("--cost-density-weight", type=float, default=0.30, help="Weight for the distribution-density component of the channel cost model.")
    parser.add_argument("--cost-visual-weight", type=float, default=0.20, help="Weight for the visual-importance component of the channel cost model.")
    parser.add_argument("--cost-detector-weight", type=float, default=0.10, help="Weight for the detector-tail component of the channel cost model.")
    parser.add_argument("--spread-factor", type=positive_int, default=None, help="How many carrier updates each coded bit is spread across in cost_stego_v1.")
    parser.add_argument("--parity-block-size", type=positive_int, default=1, help="Logical block size for keyed parity embedding.")
    parser.add_argument("--cost-model", choices=("default_v1",), default="default_v1", help="Named cost model for cost_stego_v1.")
    parser.add_argument("--dataset-name", type=str, default="ad_hoc", help="Dataset identifier recorded in reports.")
    parser.add_argument("--image-id", type=str, default=None, help="Stable image identifier used by grouped detectability summaries.")
    parser.add_argument("--crop-id", type=str, default="center", help="Crop identifier used by benchmark manifests.")
    parser.add_argument("--fit-seed", type=int, default=0, help="Seed that created the clean source run.")
    parser.add_argument("--actor-id", type=str, default=None, help="Optional actor/source-cluster identifier.")
    parser.add_argument("--label", type=str, default=None, help="Optional study label recorded in the watermark report.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional explicit watermark directory. Defaults to outputs/watermark_<timestamp>/.")
    return parser.parse_args(argv)


def compute_ber(reference_bits: torch.Tensor, decoded_bits: torch.Tensor) -> float:
    return float((reference_bits.to(torch.int64) != decoded_bits.to(torch.int64)).float().mean().item())


def _format_float_slug(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p") or "0"


def default_watermark_output_dir(source_run_name: str, args: argparse.Namespace) -> Path:
    if getattr(args, "protocol", "legacy_wm") == COST_STEGO_V2_PROTOCOL:
        suffix_parts = [
            args.protocol,
            args.baseline_type,
            f"{args.payload_bits:03d}b",
            f"key_{key_to_hash(args.key)[:8]}",
            f"steps_{int(args.tune_steps)}",
            f"lr_{_format_float_slug(args.tune_lr)}",
            f"msg_{_format_float_slug(args.msg_strength)}",
            f"mmd_{_format_float_slug(float(args.mmd_weight if args.mmd_weight is not None else 0.0))}",
            f"ws_{_format_float_slug(float(args.wasserstein_weight))}",
            f"sel_{_format_float_slug(float(args.selection_penalty_weight))}",
            f"temp_{_format_float_slug(float(args.selection_temperature))}",
            f"mods_{_format_float_slug(float(args.max_modification_ratio if args.max_modification_ratio is not None else 0.0))}",
        ]
        if getattr(args, "native_ablation", "full") != "full":
            suffix_parts.append(f"abl_{args.native_ablation}")
        suffix = "_".join(suffix_parts)
        return OUTPUT_ROOT / f"watermark_{source_run_name}_{suffix}"
    if getattr(args, "protocol", "legacy_wm") == DEFAULT_STEGO_PROTOCOL:
        suffix = "_".join(
            [
                args.protocol,
                f"{args.payload_bits:03d}b",
                f"key_{key_to_hash(args.key)[:8]}",
                f"steps_{int(args.tune_steps)}",
                f"lr_{_format_float_slug(args.tune_lr)}",
                f"msg_{_format_float_slug(args.msg_strength)}",
                f"mmd_{_format_float_slug(float(args.mmd_weight if args.mmd_weight is not None else 0.0))}",
                f"det_{_format_float_slug(float(args.detector_weight if args.detector_weight is not None else 0.0))}",
                f"ewc_{_format_float_slug(float(args.ewc_weight if args.ewc_weight is not None else 0.0))}",
                f"cover_{_format_float_slug(float(args.covertness_weight))}",
                f"wet_{_format_float_slug(float(args.wet_threshold))}",
                f"spread_{int(args.spread_factor if args.spread_factor is not None else default_spread_factor(args.payload_bits))}",
            ]
        )
        return OUTPUT_ROOT / f"watermark_{source_run_name}_{suffix}"
    suffix = "_".join(
        [
            args.mode,
            args.carrier_pool,
            args.stage,
            args.ecc,
            args.carrier_score,
            f"{args.payload_bits:03d}b",
            f"cand_{int(args.candidate_factor)}",
            f"key_{key_to_hash(args.key)[:8]}",
            f"target_{_format_float_slug(args.target_magnitude)}",
            f"steps_{int(args.tune_steps)}",
            f"lr_{_format_float_slug(args.tune_lr)}",
            f"msg_{_format_float_slug(args.msg_strength)}",
            f"dist_{_format_float_slug(args.dist_weight)}",
            f"anchor_{_format_float_slug(args.anchor_weight)}",
        ]
    )
    return OUTPUT_ROOT / f"watermark_{source_run_name}_{suffix}"


def _detectability_display_value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _metric_display_value(value: float | None, *, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _parameter_metrics_applicable(baseline_type: str) -> bool:
    return baseline_type != "pixel"


def _uses_native_parameter_embedding(baseline_type: str) -> bool:
    return baseline_type in NATIVE_PARAMETER_BASELINES


def _supports_keyed_assignment_family(baseline_type: str) -> bool:
    return baseline_type in KEYED_PARAMETER_BASELINES


def _main_claim_eligible(
    *,
    baseline_type: str,
    native_ablation: str,
    claim_eligible: bool,
    label: str | None = None,
    claim_track: str = "full",
) -> bool:
    expected_ablation = "no_detector" if claim_track == "no_detector" else "full"
    return bool(
        baseline_type == "native"
        and native_ablation == expected_ablation
        and claim_eligible
        and label == MAIN_CLAIM_LABEL
    )


def _baseline_role(
    *,
    baseline_type: str,
    native_ablation: str = "full",
    claim_eligible: bool,
    label: str | None = None,
    claim_track: str = "full",
) -> str:
    if baseline_type in {"pixel", "render_refit"}:
        return "transfer_baseline"
    if baseline_type in PROTOTYPE_BASELINES:
        return "prototype_baseline"
    if baseline_type == "native_matched_random":
        return "matched_random_baseline"
    if baseline_type in HEURISTIC_PARAMETER_BASELINES:
        return "heuristic_nonkey_baseline"
    if _main_claim_eligible(
        baseline_type=baseline_type,
        native_ablation=native_ablation,
        claim_eligible=claim_eligible,
        label=label,
        claim_track=claim_track,
    ):
        return "main_native"
    if baseline_type == "native" and native_ablation != "full":
        return "ablations"
    return "failure_analysis"


def _claim_eligible(
    *,
    baseline_type: str,
    native_ablation: str,
    qualified_cover: bool | None,
    detector_training_enabled: bool,
    wrong_key_training_enabled: bool,
    capacity_failure: bool,
    claim_track: str = "full",
) -> bool:
    expected_ablation = "no_detector" if claim_track == "no_detector" else "full"
    detector_requirement = True if claim_track == "full" else False
    return bool(
        baseline_type == "native"
        and native_ablation == expected_ablation
        and qualified_cover is True
        and (detector_training_enabled if detector_requirement else True)
        and wrong_key_training_enabled
        and not capacity_failure
    )


def _qualified_cover_diagnostics(
    *,
    clean_clamped_psnr: float,
    gaussian_count: int,
    effective_action_budget: int,
    payload_bits: int,
) -> dict[str, object]:
    psnr_threshold = 30.0
    gaussian_threshold = 192
    action_budget_threshold = int(math.ceil(2.5 * float(payload_bits)))
    failed_checks: list[str] = []
    if clean_clamped_psnr < psnr_threshold:
        failed_checks.append("clean_clamped_psnr")
    if gaussian_count < gaussian_threshold:
        failed_checks.append("gaussian_count")
    if effective_action_budget < action_budget_threshold:
        failed_checks.append("effective_action_budget")
    return {
        "qualified_cover": len(failed_checks) == 0,
        "reason": "passed" if not failed_checks else ",".join(failed_checks),
        "failed_checks": failed_checks,
        "clean_clamped_psnr": float(clean_clamped_psnr),
        "clean_clamped_psnr_threshold": float(psnr_threshold),
        "gaussian_count": int(gaussian_count),
        "gaussian_count_threshold": int(gaussian_threshold),
        "effective_action_budget": int(effective_action_budget),
        "effective_action_budget_threshold": int(action_budget_threshold),
        "payload_bits": int(payload_bits),
    }


def _attack_family_for_case(name: str, *, baseline_type: str) -> str:
    if name.startswith("adaptive_point_refit_"):
        return "topology_changing"
    if baseline_type in {"pixel", "render_refit"} and name == "clean_watermarked":
        return "image_transfer"
    return "carrier_preserving"


def _wrong_key_comparable_for_case(
    name: str,
    *,
    baseline_type: str,
    wrong_key_status: str,
    protocol: str,
) -> bool:
    if _attack_family_for_case(name, baseline_type=baseline_type) == "topology_changing":
        return False
    if protocol not in {DEFAULT_STEGO_PROTOCOL, COST_STEGO_V2_PROTOCOL, COST_STEGO_V1_PROTOCOL, JOINT_PARITY_PROTOCOL}:
        return False
    return wrong_key_status not in {
        "capacity_failure",
        "not_supported_legacy_protocol",
        "not_supported_prototype_protocol",
        "not_supported_joint_parity",
        "no_wrong_key_assignments",
    }


def _human_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _protocol_capability_note(
    *,
    protocol: str,
    baseline_type: str,
    detector_training_enabled: bool,
    wrong_key_training_enabled: bool,
) -> str:
    if protocol == THETA_DISTRIBUTION_PROTOCOL:
        return (
            "theta_distribution_v1 is an exploratory rotation-distribution prototype that biases keyed group-level "
            "circular means of theta; it does not yet share the cost-aware covertness budget or detector training used by cost_stego_v2."
        )
    if protocol == MU_JITTER_PROTOCOL:
        return (
            "mu_jitter_v1 is an exploratory position-jitter prototype that encodes bits through keyed signed center displacements; "
            "it currently uses white-box offset references and does not yet share the full cost-aware covertness machinery."
        )
    if protocol == SCALE_TIER_PROTOCOL:
        return (
            "scale_tier_v1 is an exploratory scale-hierarchy prototype that encodes bits through keyed within-tier mean shifts; "
            "it currently relies on clean-reference evaluation and remains outside the headline covertness budget."
        )
    if protocol == JOINT_PARITY_PROTOCOL:
        return (
            "joint_parity_v1 is an exploratory multi-parameter prototype that encodes payload bits across "
            "log-anisotropy, opacity, and rotation; it does not yet share the full cost-aware covertness "
            "budget, wrong-key training, or detector regularization used by cost_stego_v2."
        )
    if protocol == COST_STEGO_V2_PROTOCOL:
        if baseline_type == "native":
            capabilities = [
                "keyed cost-coded signed parameter actions",
                "soft action selection",
            ]
            if wrong_key_training_enabled:
                capabilities.append("wrong-key contrastive training")
            if detector_training_enabled:
                capabilities.append("adversarial detector regularization")
            capabilities.append("multi-metric covertness regularization")
            return f"cost_stego_v2 native embedding uses {_human_join(capabilities)}."
        if baseline_type == "native_matched_random":
            return (
                "cost_stego_v2 native_matched_random uses the same modifiable parameter pool and tune budget as the native method, "
                "but replaces keyed cost-based assignment with uniformly sampled random assignments."
            )
        if baseline_type == "native_greedy_nokey":
            return (
                "cost_stego_v2 native_greedy_nokey uses the same modifiable parameter pool and tune budget as the native method, "
                "but replaces keyed sampling with a deterministic no-key lowest-cost greedy candidate policy."
            )
        if baseline_type == "native_random_polarity":
            return (
                "cost_stego_v2 native_random_polarity uses the same modifiable parameter pool and tune budget as the native method, "
                "but keeps carrier selection unkeyed and assigns candidate action polarities randomly."
            )
        if baseline_type == "render_refit":
            return (
                "cost_stego_v2 render_refit uses keyed image-domain cost coding followed by Gaussian refitting; "
                "it is a transfer baseline, not direct native parameter embedding."
            )
        return "cost_stego_v2 pixel uses keyed image-domain cost coding on the rendered Y channel; it is a pixel-domain baseline."
    if protocol == DEFAULT_STEGO_PROTOCOL:
        return "cost_stego_v1 uses keyed parity blocks over log-anisotropy and alpha with covertness regularization."
    return "legacy_wm is a keyed watermark baseline rather than a cost-coded steganography protocol."


def _steganography_claim_scope(
    *,
    protocol: str,
    baseline_type: str,
    capacity_failure: bool,
    qualified_cover: bool | None,
) -> str:
    if protocol not in {DEFAULT_STEGO_PROTOCOL, COST_STEGO_V1_PROTOCOL, COST_STEGO_V2_PROTOCOL, *PROTOTYPE_PROTOCOLS}:
        return "legacy_watermark_baseline"
    if baseline_type == "pixel":
        return "pixel_domain_baseline_only"
    if baseline_type == "render_refit":
        return "render_then_refit_baseline"
    if baseline_type in PROTOTYPE_BASELINES:
        if capacity_failure:
            return "capacity_failure"
        if qualified_cover is False:
            return "parameter_domain_prototype_unqualified_cover"
        return "parameter_domain_prototype"
    if baseline_type == "native_matched_random":
        return "matched_random_parameter_baseline"
    if baseline_type in HEURISTIC_PARAMETER_BASELINES:
        return "heuristic_parameter_baseline"
    if capacity_failure:
        return "capacity_failure"
    if qualified_cover is False:
        return "parameter_domain_prototype_unqualified_cover"
    if qualified_cover is None:
        return "parameter_domain_prototype"
    return "qualified_parameter_domain_steganography"


def _steganography_claim_note(
    *,
    protocol: str,
    baseline_type: str,
    capacity_failure: bool,
    qualified_cover: bool | None,
) -> str:
    scope = _steganography_claim_scope(
        protocol=protocol,
        baseline_type=baseline_type,
        capacity_failure=capacity_failure,
        qualified_cover=qualified_cover,
    )
    if scope == "qualified_parameter_domain_steganography":
        return "This run falls inside the intended keyed white-box parameter-domain steganography regime."
    if scope == "parameter_domain_prototype_unqualified_cover":
        return (
            "This run remains a keyed parameter-domain embedding result, but it did not meet the qualified-cover "
            "threshold; treat it as prototype evidence rather than a strong steganography claim."
        )
    if scope == "parameter_domain_prototype":
        return "This run is parameter-domain and keyed, but its claim scope is still prototype-only."
    if scope == "render_then_refit_baseline":
        return "This run is a render-then-refit transfer baseline; it is not direct parameter-domain steganography evidence."
    if scope == "pixel_domain_baseline_only":
        return "This run is a pixel-domain baseline and should not be cited as parameter-domain steganography evidence."
    if scope == "parameter_domain_prototype":
        return "This run is an exploratory parameter-domain prototype and should only be used as appendix-level evidence."
    if scope == "matched_random_parameter_baseline":
        return "This run is a matched random parameter-domain baseline and is only evidence for fairness comparisons, not the main steganography claim."
    if scope == "heuristic_parameter_baseline":
        return "This run is a no-key heuristic parameter-domain baseline and is only evidence for medium-strength fairness comparisons, not the main steganography claim."
    if scope == "capacity_failure":
        return "The requested payload exceeded the available budget under the declared constraints, so no valid steganographic payload was embedded."
    return "This run belongs to the legacy watermark baseline path rather than the cost-coded steganography path."


def _wrong_key_status(
    *,
    protocol: str,
    capacity_failure: bool,
    wrong_key_accuracy: float | None,
    wrong_key_groups_present: bool,
    wrong_key_sample_failures: int = 0,
) -> str:
    if protocol in {THETA_DISTRIBUTION_PROTOCOL, MU_JITTER_PROTOCOL, SCALE_TIER_PROTOCOL}:
        if capacity_failure:
            return "capacity_failure"
        if wrong_key_accuracy is not None:
            return "computed"
        return "not_supported_prototype_protocol"
    if protocol == JOINT_PARITY_PROTOCOL:
        if capacity_failure:
            return "capacity_failure"
        if wrong_key_accuracy is not None:
            return "computed"
        return "not_supported_joint_parity"
    if protocol not in {DEFAULT_STEGO_PROTOCOL, COST_STEGO_V1_PROTOCOL, COST_STEGO_V2_PROTOCOL}:
        return "not_supported_legacy_protocol"
    if capacity_failure:
        return "capacity_failure"
    if wrong_key_accuracy is not None:
        return "computed"
    if wrong_key_groups_present:
        return "decode_failed"
    if wrong_key_sample_failures > 0:
        return "sampling_failed"
    return "no_wrong_key_assignments"


def _flatten_block_probabilities(
    block_probabilities: list[list[float]],
    *,
    raw_length: int,
) -> list[float]:
    flat = [float(value) for row in block_probabilities for value in row]
    return flat[:raw_length]


def _cost_code_decode_diagnostics(
    *,
    raw_bits: torch.Tensor,
    assignments: list[dict[str, object]],
    block_probabilities: list[list[float]],
    action_scores: list[list[float]],
    wrong_key_block_probabilities: list[list[list[float]]],
) -> dict[str, object]:
    per_bit_probabilities = _flatten_block_probabilities(block_probabilities, raw_length=int(raw_bits.numel()))
    if not per_bit_probabilities:
        return {
            "true_key_margin": None,
            "wrong_key_margin": None,
            "per_bit_confidence": [],
            "wrong_key_per_bit_confidence": [],
            "selected_action_count_per_bit": [],
            "action_polarity_mismatch": {
                "selected_action_count": 0,
                "mismatch_count": 0,
                "mismatch_rate": None,
                "per_bit_mismatch_count": [],
            },
        }
    target = raw_bits.to(torch.float32)[: len(per_bit_probabilities)]
    target_probs = torch.tensor(per_bit_probabilities, dtype=torch.float32)
    aligned = torch.where(target > 0.5, target_probs, 1.0 - target_probs)
    per_bit_confidence = (2.0 * (aligned - 0.5)).clamp(min=0.0, max=1.0)

    wrong_key_margin: float | None = None
    wrong_key_per_bit_confidence: list[float] = []
    if wrong_key_block_probabilities:
        wrong_confidences: list[torch.Tensor] = []
        wrong_margins: list[float] = []
        for wrong_probs in wrong_key_block_probabilities:
            flattened = _flatten_block_probabilities(wrong_probs, raw_length=len(per_bit_probabilities))
            if len(flattened) != len(per_bit_probabilities):
                continue
            wrong_tensor = torch.tensor(flattened, dtype=torch.float32)
            wrong_aligned = torch.where(target > 0.5, wrong_tensor, 1.0 - wrong_tensor)
            wrong_confidences.append((2.0 * (wrong_aligned - 0.5)).clamp(min=0.0, max=1.0))
            wrong_margins.append(float((wrong_aligned - 0.5).mean().item()))
        if wrong_confidences:
            wrong_key_margin = float(sum(wrong_margins) / len(wrong_margins))
            wrong_key_per_bit_confidence = [
                float(value)
                for value in torch.stack(wrong_confidences, dim=0).mean(dim=0).detach().cpu().tolist()
            ]

    selected_action_count_per_bit: list[int] = []
    per_bit_mismatch_count: list[int] = []
    mismatch_count = 0
    selected_action_count = 0
    for assignment, block_action_scores in zip(assignments, action_scores):
        selected_indices = [
            index
            for index, value in enumerate(assignment.get("selected_mask", []))
            if float(value) > 0.5
        ]
        block_selected_count = len(selected_indices)
        block_mismatch_count = sum(
            1
            for index in selected_indices
            if index >= len(block_action_scores) or float(block_action_scores[index]) <= 0.0
        )
        selected_action_count += block_selected_count
        mismatch_count += block_mismatch_count
        block_bits = assignment.get("message_bits", [])
        block_length = min(4, int(raw_bits.numel()) - len(selected_action_count_per_bit))
        if not block_bits:
            block_length = min(block_length, 4)
        selected_action_count_per_bit.extend([block_selected_count] * max(0, block_length))
        per_bit_mismatch_count.extend([block_mismatch_count] * max(0, block_length))
        if len(selected_action_count_per_bit) >= int(raw_bits.numel()):
            break

    return {
        "true_key_margin": float((aligned - 0.5).mean().item()),
        "wrong_key_margin": wrong_key_margin,
        "per_bit_confidence": [float(value) for value in per_bit_confidence.detach().cpu().tolist()],
        "wrong_key_per_bit_confidence": wrong_key_per_bit_confidence,
        "selected_action_count_per_bit": selected_action_count_per_bit[: int(raw_bits.numel())],
        "action_polarity_mismatch": {
            "selected_action_count": int(selected_action_count),
            "mismatch_count": int(mismatch_count),
            "mismatch_rate": None if selected_action_count == 0 else float(mismatch_count / selected_action_count),
            "per_bit_mismatch_count": per_bit_mismatch_count[: int(raw_bits.numel())],
        },
    }


def _cost_stego_defaults(args: argparse.Namespace) -> dict[str, float]:
    if args.payload_bits <= 8:
        return {
            "msg_strength": 0.010 if args.msg_strength is None else args.msg_strength,
            "mmd_weight": 0.100 if args.mmd_weight is None else float(args.mmd_weight),
            "moment_weight": 0.010 if args.dist_weight == 0.0 else float(args.dist_weight),
            "detector_weight": 0.050 if args.detector_weight is None else float(args.detector_weight),
            "ewc_weight": 0.010 if args.ewc_weight is None else float(args.ewc_weight),
            "compensation_weight": 0.020,
            "wrong_key_weight": 0.030,
        }
    return {
        "msg_strength": 0.012 if args.msg_strength is None else args.msg_strength,
        "mmd_weight": 0.120 if args.mmd_weight is None else float(args.mmd_weight),
        "moment_weight": 0.010 if args.dist_weight == 0.0 else float(args.dist_weight),
        "detector_weight": 0.060 if args.detector_weight is None else float(args.detector_weight),
        "ewc_weight": 0.020 if args.ewc_weight is None else float(args.ewc_weight),
        "compensation_weight": 0.030,
        "wrong_key_weight": 0.015,
    }


def _cost_stego_v2_defaults(args: argparse.Namespace) -> dict[str, float]:
    if args.payload_bits <= 8:
        return {
            "msg_strength": float(args.msg_strength),
            "mmd_weight": 0.08 if args.mmd_weight is None else float(args.mmd_weight),
            "wasserstein_weight": float(args.wasserstein_weight),
            "detector_weight": 0.04 if args.detector_weight is None else float(args.detector_weight),
            "selection_weight": float(args.selection_penalty_weight),
            "moment_weight": 0.01 if args.dist_weight == 0.0 else float(args.dist_weight),
            "ewc_weight": 0.01 if args.ewc_weight is None else float(args.ewc_weight),
            "compensation_weight": 0.02,
            "wrong_key_weight": 0.06,
            "native_margin_weight": 0.10,
            "native_wrong_key_specificity_weight": 0.12,
        }
    return {
        "msg_strength": max(0.012, float(args.msg_strength)),
        "mmd_weight": 0.10 if args.mmd_weight is None else float(args.mmd_weight),
        "wasserstein_weight": float(args.wasserstein_weight),
        "detector_weight": 0.05 if args.detector_weight is None else float(args.detector_weight),
        "selection_weight": float(args.selection_penalty_weight),
        "moment_weight": 0.01 if args.dist_weight == 0.0 else float(args.dist_weight),
        "ewc_weight": 0.015 if args.ewc_weight is None else float(args.ewc_weight),
        "compensation_weight": 0.03,
        "wrong_key_weight": 0.05,
        "native_margin_weight": 0.08,
        "native_wrong_key_specificity_weight": 0.10,
    }


def _alpha_delta_from_target(target_magnitude: float) -> float:
    return float(min(0.08, max(0.03, 0.16 * target_magnitude)))


def _v2_max_modification_ratio(args: argparse.Namespace) -> float:
    if args.max_modification_ratio is not None:
        return float(args.max_modification_ratio)
    if args.payload_bits <= 8:
        return 0.20
    return 0.30


def _top_importance_share(values: torch.Tensor, topk: int) -> float:
    flattened = values.reshape(-1).to(torch.float32)
    if flattened.numel() == 0:
        return 0.0
    sorted_values = torch.sort(flattened, descending=True).values
    return float(sorted_values[: min(int(topk), int(sorted_values.numel()))].sum() / sorted_values.sum().clamp_min(1e-12))


def _adaptive_v2_cover_profile(
    *,
    clean_params: dict[str, torch.Tensor],
    clean_importance: torch.Tensor,
    clean_sensitivity: dict[str, torch.Tensor],
) -> dict[str, object]:
    canonical = canonicalize_gaussian_geometry(
        clean_params["scales"],
        clean_params["thetas"],
    )
    importance_top32_share = _top_importance_share(clean_importance, 32)
    log_area = canonical["log_area"].reshape(-1).to(torch.float32)
    alpha_sensitivity_top32_share = _top_importance_share(clean_sensitivity["alpha"].abs(), 32)
    log_sensitivity_top32_share = _top_importance_share(clean_sensitivity["log_anisotropy"].abs(), 32)
    sensitivity_focus_top32_share = max(alpha_sensitivity_top32_share, log_sensitivity_top32_share)
    mean_log_area = float(log_area.mean().item()) if log_area.numel() > 0 else float("-inf")
    std_log_area = float(log_area.std(unbiased=False).item()) if log_area.numel() > 1 else 0.0
    if importance_top32_share < 0.26 and mean_log_area > -5.5:
        profile = "flat_broad_focus_cover" if sensitivity_focus_top32_share > 0.45 else "flat_broad_cover"
    else:
        profile = "default_cover"
    return {
        "profile": profile,
        "importance_top32_share": float(importance_top32_share),
        "mean_log_area": float(mean_log_area),
        "std_log_area": float(std_log_area),
        "alpha_sensitivity_top32_share": float(alpha_sensitivity_top32_share),
        "log_sensitivity_top32_share": float(log_sensitivity_top32_share),
        "sensitivity_focus_top32_share": float(sensitivity_focus_top32_share),
    }


def _adaptive_v2_embedding_controls(
    *,
    args: argparse.Namespace,
    clean_params: dict[str, torch.Tensor],
    clean_importance: torch.Tensor,
    clean_sensitivity: dict[str, torch.Tensor],
) -> dict[str, object]:
    base_target_magnitude = float(args.target_magnitude)
    base_max_modification_ratio = _v2_max_modification_ratio(args)
    cover_profile = _adaptive_v2_cover_profile(
        clean_params=clean_params,
        clean_importance=clean_importance,
        clean_sensitivity=clean_sensitivity,
    )
    if not bool(getattr(args, "adaptive_native_budget", False)):
        return {
            "enabled": False,
            "profile": str(cover_profile["profile"]),
            "profile_stats": cover_profile,
            "target_magnitude": float(base_target_magnitude),
            "alpha_delta": float(_alpha_delta_from_target(base_target_magnitude)),
            "max_modification_ratio": float(base_max_modification_ratio),
        }

    effective_target_magnitude = float(base_target_magnitude)
    effective_max_modification_ratio = float(base_max_modification_ratio)
    if str(cover_profile["profile"]) == "flat_broad_focus_cover" and int(args.payload_bits) <= 8:
        # Concentrate the payload onto fewer actions for broad, evenly-distributed covers.
        # The local stripe_circle proxy shows that globally shrinking the magnitude hurts,
        # so the adaptive rule only boosts covers whose channel sensitivity is still sufficiently focused.
        effective_target_magnitude = float(max(base_target_magnitude * 1.10, 0.38))
        effective_max_modification_ratio = float(min(base_max_modification_ratio, 0.14))

    return {
        "enabled": True,
        "profile": str(cover_profile["profile"]),
        "profile_stats": cover_profile,
        "target_magnitude": float(effective_target_magnitude),
        "alpha_delta": float(_alpha_delta_from_target(effective_target_magnitude)),
        "max_modification_ratio": float(effective_max_modification_ratio),
    }


def _v2_selected_indices(assignments: list[dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor]:
    log_indices: set[int] = set()
    alpha_indices: set[int] = set()
    for assignment in assignments:
        selected_mask = assignment["selected_mask"]
        for action, selected in zip(assignment["candidate_actions"], selected_mask):
            if float(selected) <= 0.5:
                continue
            if str(action["channel"]) == "log":
                log_indices.add(int(action["gaussian_index"]))
            else:
                alpha_indices.add(int(action["gaussian_index"]))
    return torch.tensor(sorted(log_indices), dtype=torch.long), torch.tensor(sorted(alpha_indices), dtype=torch.long)


def _v2_wrong_key_auxiliary_indices(
    correct_assignments: list[dict[str, object]],
    wrong_assignments: list[list[dict[str, object]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    correct_log, correct_alpha = _v2_selected_indices(correct_assignments)
    correct_log_set = {int(index) for index in correct_log.tolist()}
    correct_alpha_set = {int(index) for index in correct_alpha.tolist()}
    aux_log: set[int] = set()
    aux_alpha: set[int] = set()
    for assignment_group in wrong_assignments:
        wrong_log, wrong_alpha = _v2_selected_indices(assignment_group)
        aux_log.update(int(index) for index in wrong_log.tolist() if int(index) not in correct_log_set)
        aux_alpha.update(int(index) for index in wrong_alpha.tolist() if int(index) not in correct_alpha_set)
    return torch.tensor(sorted(aux_log), dtype=torch.long), torch.tensor(sorted(aux_alpha), dtype=torch.long)


def _collect_wrong_key_assignment_groups(
    *,
    requested_samples: int,
    builder: Callable[[int], list[dict[str, object]]],
    max_attempt_multiplier: int = 8,
) -> tuple[list[list[dict[str, object]]], int]:
    wrong_key_assignments: list[list[dict[str, object]]] = []
    if requested_samples <= 0:
        return wrong_key_assignments, 0
    max_attempts = max(requested_samples, requested_samples * max(1, int(max_attempt_multiplier)))
    attempts = 0
    while len(wrong_key_assignments) < requested_samples and attempts < max_attempts:
        try:
            wrong_key_assignments.append(builder(attempts))
        except RuntimeError:
            pass
        attempts += 1
    return wrong_key_assignments, max(0, attempts - len(wrong_key_assignments))


def _native_detector_requested(args: argparse.Namespace) -> bool:
    return str(getattr(args, "native_ablation", "full")) != "no_detector"


def _native_wrong_key_requested(args: argparse.Namespace) -> bool:
    return str(getattr(args, "native_ablation", "full")) != "no_wrong_key"


def _native_attack_training_requested(args: argparse.Namespace) -> bool:
    return str(getattr(args, "native_ablation", "full")) != "no_attack_training"


def _straight_through_quantize(tensor: torch.Tensor, *, step: float, min_value: float | None = None, max_value: float | None = None) -> torch.Tensor:
    quantized = torch.round(tensor / step) * step
    if min_value is not None or max_value is not None:
        quantized = torch.clamp(quantized, min=min_value if min_value is not None else -float("inf"), max=max_value if max_value is not None else float("inf"))
    return tensor + (quantized - tensor).detach()


def _carrier_preserving_attack_params(
    params: dict[str, torch.Tensor],
    *,
    attack_name: str,
    assignments: list[dict[str, object]],
    small_noise_log: torch.Tensor | None = None,
    small_noise_alpha: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    attacked = {key: value for key, value in params.items()}
    if attack_name == "quantize_step_0.01":
        attacked = {
            **attacked,
            "offsets": _straight_through_quantize(attacked["offsets"], step=0.01),
            "scales": _straight_through_quantize(attacked["scales"], step=0.01, min_value=1e-6),
            "thetas": _straight_through_quantize(attacked["thetas"], step=0.01),
            "colors": _straight_through_quantize(attacked["colors"], step=0.01, min_value=0.0, max_value=1.0),
            "opacities": _straight_through_quantize(attacked["opacities"], step=0.01, min_value=0.0, max_value=1.0),
        }
    elif attack_name == "small_parameter_noise_0.01":
        attacked = {key: value.clone() for key, value in attacked.items()}
        log_indices, alpha_indices = _v2_selected_indices(assignments)
        if log_indices.numel() > 0 and small_noise_log is not None:
            indices = log_indices.to(attacked["scales"].device)
            attacked["scales"][indices] = torch.clamp(attacked["scales"][indices] + small_noise_log.to(attacked["scales"].device), min=1e-6)
        if alpha_indices.numel() > 0 and small_noise_alpha is not None:
            indices = alpha_indices.to(attacked["opacities"].device)
            attacked["opacities"][indices] = torch.clamp(attacked["opacities"][indices] + small_noise_alpha.to(attacked["opacities"].device), 0.0, 1.0)
    elif attack_name == "importance_pruning_10pct":
        attacked = {key: value.clone() for key, value in attacked.items()}
        drop_count = max(1, int(round(0.10 * float(attacked["offsets"].shape[0]))))
        importance = compute_gaussian_importance(attacked["scales"], attacked["opacities"])
        drop_indices = torch.argsort(importance, descending=False)[:drop_count]
        attacked["opacities"][drop_indices, 0] = 0.0
    else:
        raise ValueError(f"unsupported carrier-preserving attack: {attack_name}")
    attacked["importance"] = compute_gaussian_importance(attacked["scales"], attacked["opacities"])
    return attacked


def _v2_assignments_for_compensation(assignments: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for assignment in assignments:
        log_indices: list[int] = []
        alpha_indices: list[int] = []
        for action, selected in zip(assignment["candidate_actions"], assignment["selected_mask"]):
            if float(selected) <= 0.5:
                continue
            if str(action["channel"]) == "log":
                log_indices.append(int(action["gaussian_index"]))
            else:
                alpha_indices.append(int(action["gaussian_index"]))
        output.append(
            {
                "log_indices": log_indices,
                "alpha_indices": alpha_indices,
            }
        )
    return output


def _serialize_cost_code_assignments(assignments: list[dict[str, object]]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for assignment in assignments:
        candidate_actions: list[dict[str, object]] = []
        for action in assignment["candidate_actions"]:
            if "gaussian_index" in action:
                candidate_actions.append(
                    {
                        "gaussian_index": int(action["gaussian_index"]),
                        "channel": str(action["channel"]),
                        "direction": int(action["direction"]),
                        "selection_probability": float(action["selection_probability"]),
                    }
                )
            else:
                candidate_actions.append(
                    {
                        "candidate_id": int(action["candidate_id"]),
                        "action_type": str(action["action_type"]),
                        "direction": int(action["direction"]),
                        "strength_scale": float(action.get("strength_scale", 1.0)),
                        "selection_probability": float(action["selection_probability"]),
                    }
                )
        serialized.append(
            {
                "block_index": int(assignment["block_index"]),
                "message_bits": [int(value) for value in assignment["message_bits"]],
                "candidate_count": int(assignment["candidate_count"]),
                "candidate_actions": candidate_actions,
                "candidate_costs": [float(value) for value in assignment["candidate_costs"]],
                "parity_matrix": [[int(value) for value in row] for row in assignment["parity_matrix"]],
                "selected_mask": [float(value) for value in assignment["selected_mask"]],
                "selected_count": int(assignment["selected_count"]),
            }
        )
    return serialized


def _serialize_wrong_key_assignments(protocol: str, assignments: object) -> object:
    if protocol == COST_STEGO_V2_PROTOCOL:
        return [_serialize_cost_code_assignments(list(group)) for group in list(assignments)]
    return serializable_block_assignments(list(assignments))


def _pixel_delta() -> float:
    return 1.0 / 255.0


def _baseline_pixel_delta(args: argparse.Namespace) -> float:
    delta = _pixel_delta()
    if args.baseline_type == "render_refit":
        multiplier = 2.0 if int(args.payload_bits) <= 8 else 3.0
        return float(multiplier * delta)
    return delta


def _image_saturation_fraction(image: torch.Tensor) -> float:
    clamped = image.detach().clamp(0.0, 1.0)
    saturated = (clamped <= 1e-6) | (clamped >= 1.0 - 1e-6)
    return float(saturated.float().mean().item())


def _apply_reconstruction_override(
    evaluation: dict[str, object],
    *,
    target: torch.Tensor,
    reconstruction: torch.Tensor,
) -> dict[str, object]:
    output = dict(evaluation)
    clamped_reconstruction = reconstruction.detach().clamp(0.0, 1.0).cpu()
    clamped_target = target.detach().clamp(0.0, 1.0).cpu()
    output["reconstruction"] = clamped_reconstruction
    output["clamped_psnr"] = float(compute_psnr(clamped_target, clamped_reconstruction))
    output["ssim"] = float(compute_ssim(clamped_target, clamped_reconstruction))
    output["saturated_value_fraction"] = _image_saturation_fraction(clamped_reconstruction)
    return output


def _pixel_candidate_field(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    y = rgb_to_ycbcr(image.clamp(0.0, 1.0))[..., 0]
    padded = F.pad(y.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    kernel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(padded, kernel_x)[0, 0]
    grad_y = F.conv2d(padded, kernel_y)[0, 0]
    texture = (grad_x.abs() + grad_y.abs()).clamp_min(1e-4)
    local_mean = F.avg_pool2d(padded, kernel_size=3, stride=1)[0, 0]
    local_var = F.avg_pool2d((padded - F.pad(local_mean.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")) ** 2, kernel_size=3, stride=1)[0, 0]
    base_cost = 1.0 / (0.05 + texture + local_var.clamp_min(0.0))
    delta = _pixel_delta()
    plus_cost = base_cost + torch.relu((y + delta) - 1.0) * 50.0
    minus_cost = base_cost + torch.relu(delta - y) * 50.0
    candidate_ids = torch.arange(y.numel(), dtype=torch.long, device=y.device)
    border_mask = torch.zeros_like(y, dtype=torch.bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    selection_logits = -torch.minimum(plus_cost, minus_cost)
    selection_logits = selection_logits.reshape(-1)
    selection_logits[border_mask.reshape(-1)] = -1e9
    return candidate_ids.reshape(-1), plus_cost.reshape(-1), minus_cost.reshape(-1), selection_logits


def _apply_pixel_cost_code_assignments(
    image: torch.Tensor,
    *,
    assignments: list[dict[str, object]],
    delta: float,
) -> torch.Tensor:
    ycbcr = rgb_to_ycbcr(image.clamp(0.0, 1.0)).clone()
    y_values = ycbcr[..., 0].reshape(-1)
    for assignment in assignments:
        for action, selected in zip(assignment["candidate_actions"], assignment["selected_mask"]):
            if float(selected) <= 0.5:
                continue
            candidate_id = int(action["candidate_id"])
            action_delta = float(delta) * float(action.get("strength_scale", 1.0))
            y_values[candidate_id] = torch.clamp(
                y_values[candidate_id] + float(action["direction"]) * action_delta,
                0.0,
                1.0,
            )
    ycbcr[..., 0] = y_values.view_as(ycbcr[..., 0])
    return ycbcr_to_rgb(ycbcr)


def _selected_pixel_candidate_ids(assignments: list[dict[str, object]]) -> torch.Tensor:
    selected_ids: set[int] = set()
    for assignment in assignments:
        for action, selected in zip(assignment["candidate_actions"], assignment["selected_mask"]):
            if float(selected) <= 0.5:
                continue
            selected_ids.add(int(action["candidate_id"]))
    return torch.tensor(sorted(selected_ids), dtype=torch.long)


def _render_refit_code_losses(
    reconstruction: torch.Tensor,
    *,
    clean_image: torch.Tensor,
    target_image: torch.Tensor,
    assignments: list[dict[str, object]],
    wrong_key_assignments: list[list[dict[str, object]]] | None,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not assignments:
        zero = torch.tensor(0.0, dtype=reconstruction.dtype, device=reconstruction.device)
        return zero, zero, zero
    clean_y = rgb_to_ycbcr(clean_image.clamp(0.0, 1.0))[..., 0].reshape(-1)
    target_y = rgb_to_ycbcr(target_image.clamp(0.0, 1.0))[..., 0].reshape(-1)
    reconstruction_y = rgb_to_ycbcr(reconstruction.clamp(0.0, 1.0))[..., 0].reshape(-1)
    selected_ids = _selected_pixel_candidate_ids(assignments).to(reconstruction.device)
    if selected_ids.numel() > 0:
        pixel_loss = F.mse_loss(reconstruction_y[selected_ids], target_y[selected_ids])
    else:
        pixel_loss = torch.tensor(0.0, dtype=reconstruction.dtype, device=reconstruction.device)

    def action_score_provider(assignment: dict[str, object]) -> torch.Tensor:
        scores: list[torch.Tensor] = []
        for action in assignment["candidate_actions"]:
            candidate_id = int(action["candidate_id"])
            action_delta = max(float(delta) * float(action.get("strength_scale", 1.0)), 1e-6)
            signed_delta = (reconstruction_y[candidate_id] - clean_y[candidate_id]) / action_delta
            action_strength = float(action["direction"]) * signed_delta
            scores.append(torch.clamp(2.0 * action_strength - 1.0, min=-1.0, max=1.5))
        return torch.stack(scores) if scores else torch.empty(0, dtype=reconstruction.dtype, device=reconstruction.device)

    block_probabilities, _ = generic_cost_code_probabilities(
        assignments=assignments,
        action_score_provider=action_score_provider,
        beta=4.0,
    )
    if block_probabilities.numel() == 0:
        block_loss = torch.tensor(0.0, dtype=reconstruction.dtype, device=reconstruction.device)
        correct_confidence = torch.tensor(0.0, dtype=reconstruction.dtype, device=reconstruction.device)
    else:
        target_bits = torch.tensor(
            [assignment["message_bits"] for assignment in assignments],
            dtype=block_probabilities.dtype,
            device=block_probabilities.device,
        )
        block_margin = torch.relu(0.35 - (2.0 * target_bits - 1.0) * (2.0 * block_probabilities - 1.0)).mean()
        block_loss = F.binary_cross_entropy(block_probabilities, target_bits) + 0.25 * block_margin
        correct_confidence = torch.abs(block_probabilities - 0.5).mean() * 2.0
    wrong_key_loss = torch.tensor(0.0, dtype=reconstruction.dtype, device=reconstruction.device)
    if wrong_key_assignments:
        confusion_terms: list[torch.Tensor] = []
        wrong_confidences: list[torch.Tensor] = []
        for wrong_group in wrong_key_assignments:
            wrong_probs, _ = generic_cost_code_probabilities(
                assignments=list(wrong_group),
                action_score_provider=action_score_provider,
                beta=4.0,
            )
            if wrong_probs.numel() == 0:
                continue
            confusion_terms.append(
                F.binary_cross_entropy(
                    wrong_probs,
                    torch.full_like(wrong_probs, 0.5),
                )
            )
            wrong_confidences.append(torch.abs(wrong_probs - 0.5).mean() * 2.0)
        if confusion_terms:
            wrong_key_loss = torch.stack(confusion_terms).mean()
            wrong_mean_confidence = torch.stack(wrong_confidences).mean()
            wrong_key_loss = wrong_key_loss + torch.relu(
                torch.tensor(0.25, dtype=correct_confidence.dtype, device=correct_confidence.device)
                - (correct_confidence - wrong_mean_confidence)
            )
    return pixel_loss, block_loss, wrong_key_loss


def _refit_renderer_to_image(
    *,
    clean_renderer: Gaussian2DRenderer,
    target_image: torch.Tensor,
    grid: torch.Tensor,
    steps: int,
    lr: float,
    clean_image: torch.Tensor | None = None,
    assignments: list[dict[str, object]] | None = None,
    delta: float | None = None,
    selected_pixel_weight: float = 24.0,
    code_weight: float = 2.0,
    wrong_key_weight: float = 0.5,
    anchor_weight: float = 0.01,
    geometry_trainable: bool = False,
    wrong_key_assignments: list[list[dict[str, object]]] | None = None,
) -> Gaussian2DRenderer:
    renderer = deepcopy(clean_renderer)
    for parameter in renderer.parameters():
        parameter.requires_grad_(False)
    trainable_parameters = [renderer.raw_colors, renderer.raw_opacities]
    renderer.raw_colors.requires_grad_(True)
    renderer.raw_opacities.requires_grad_(True)
    if geometry_trainable:
        renderer.raw_offsets.requires_grad_(True)
        renderer.raw_scales.requires_grad_(True)
        renderer.raw_thetas.requires_grad_(True)
        trainable_parameters.extend([renderer.raw_offsets, renderer.raw_scales, renderer.raw_thetas])
    optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
    assignments = [] if assignments is None else assignments
    clean_image = None if clean_image is None else clean_image.to(next(clean_renderer.parameters()).device)
    total_steps = max(1, int(steps))
    for step_index in range(total_steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = renderer(grid[..., 0], grid[..., 1], clamp_output=False)
        loss = F.mse_loss(reconstruction, target_image)
        loss = loss + float(anchor_weight) * (
            F.mse_loss(renderer.raw_colors, clean_renderer.raw_colors.detach().to(renderer.raw_colors.device))
            + F.mse_loss(renderer.raw_opacities, clean_renderer.raw_opacities.detach().to(renderer.raw_opacities.device))
        )
        if geometry_trainable:
            loss = loss + 0.5 * float(anchor_weight) * (
                F.mse_loss(renderer.raw_offsets, clean_renderer.raw_offsets.detach().to(renderer.raw_offsets.device))
                + F.mse_loss(renderer.raw_scales, clean_renderer.raw_scales.detach().to(renderer.raw_scales.device))
                + F.mse_loss(renderer.raw_thetas, clean_renderer.raw_thetas.detach().to(renderer.raw_thetas.device))
            )
        if clean_image is not None and assignments and delta is not None:
            selected_pixel_loss, block_loss, wrong_key_loss = _render_refit_code_losses(
                reconstruction,
                clean_image=clean_image,
                target_image=target_image,
                assignments=assignments,
                wrong_key_assignments=wrong_key_assignments,
                delta=float(delta),
            )
            progress = 1.0 if total_steps <= 1 else float(step_index) / float(total_steps - 1)
            code_scale = max(0.0, (progress - 0.35) / 0.65)
            loss = loss + code_scale * (
                float(selected_pixel_weight) * selected_pixel_loss
                + float(code_weight) * block_loss
                + float(wrong_key_weight) * wrong_key_loss
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
        optimizer.step()
    return renderer


def build_joint_renderer(source_run, target, grid, device: torch.device) -> Gaussian2DRenderer:
    renderer = Gaussian2DRenderer(num_gaussians=source_run.offsets.shape[0]).to(device)
    config = experiment_config_from_export(source_run, device=device)
    initialize_renderer(renderer, target, grid[..., 0], grid[..., 1], config)
    return renderer


def build_payload(raw_bits: torch.Tensor, ecc_mode: str) -> tuple[torch.Tensor, dict[str, int]]:
    if ecc_mode == "none":
        return raw_bits.clone(), {"pad_bits": 0, "corrected_blocks": 0}
    if ecc_mode == "hamming74":
        coded_bits, pad_bits = hamming74_encode(raw_bits)
        return coded_bits, {"pad_bits": pad_bits, "corrected_blocks": 0}
    raise ValueError(f"unsupported ecc mode: {ecc_mode}")


def decode_payload(decoded_coded_bits: torch.Tensor, *, raw_length: int, ecc_mode: str) -> tuple[torch.Tensor, dict[str, int]]:
    if ecc_mode == "none":
        return decoded_coded_bits[:raw_length].clone(), {"corrected_blocks": 0}
    if ecc_mode == "hamming74":
        decoded_raw, corrected_blocks = hamming74_decode(decoded_coded_bits, raw_length=raw_length)
        return decoded_raw, {"corrected_blocks": corrected_blocks}
    raise ValueError(f"unsupported ecc mode: {ecc_mode}")


def select_carriers(
    *,
    source_run,
    coded_bits: torch.Tensor,
    mode: str,
    carrier_pool: str,
    carrier_score_mode: str,
    target_magnitude: float,
    candidate_factor: int,
    key: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    coded_payload_bits = int(coded_bits.numel())
    if coded_payload_bits > source_run.importance.numel():
        raise ValueError("coded payload bits cannot exceed the number of Gaussians in the source run")
    if carrier_pool == "topk":
        carrier_indices = select_topk_carriers(source_run.importance, payload_bits=coded_payload_bits)
        return carrier_indices, {"candidate_count": coded_payload_bits, "carrier_score_mode": "importance"}
    if carrier_pool == "low_importance":
        carrier_indices = select_low_importance_carriers(source_run.importance, payload_bits=coded_payload_bits)
        return carrier_indices, {"candidate_count": coded_payload_bits, "carrier_score_mode": "importance"}

    if source_run.sensitivity is None:
        raise ValueError("sensitivity-aware carrier selection requires sensitivity export in the source run")

    carrier_values = carrier_values_from_canonical(source_run.canonical, mode=mode)
    sensitivity_field = carrier_sensitivity_field(mode)
    if carrier_score_mode == "importance":
        candidate_count = min(source_run.importance.numel(), max(coded_payload_bits, candidate_factor * coded_payload_bits))
        candidate_indices = torch.topk(source_run.importance, k=candidate_count).indices
        candidate_scores = source_run.importance
    else:
        candidate_indices, candidate_scores = select_sensitivity_candidates(
            importance=source_run.importance,
            sensitivity_values=source_run.sensitivity[sensitivity_field],
            carrier_values=carrier_values,
            coded_payload_bits=coded_payload_bits,
            candidate_factor=candidate_factor,
            mode=mode,
            target_magnitude=target_magnitude,
        )
        candidate_count = int(candidate_indices.numel())
    carrier_indices = keyed_adjustment_selection(
        candidate_indices=candidate_indices,
        carrier_values=carrier_values,
        preference_scores=candidate_scores,
        coded_bits=coded_bits,
        target_magnitude=target_magnitude,
        key=key,
        mode=mode,
    )
    return carrier_indices, {"candidate_count": candidate_count, "carrier_score_mode": carrier_score_mode}


def run_postfit_embedding(
    *,
    clean_renderer: Gaussian2DRenderer,
    target: torch.Tensor,
    grid: torch.Tensor,
    carrier_indices: torch.Tensor,
    coded_bits: torch.Tensor,
    mode: str,
    msg_strength: float,
    dist_weight: float,
    anchor_weight: float,
    target_magnitude: float,
    tune_steps: int,
    tune_lr: float,
    clean_reference: dict[str, torch.Tensor],
    clean_scales: torch.Tensor,
    sensitivity_weights: torch.Tensor,
) -> Gaussian2DRenderer:
    renderer = deepcopy(clean_renderer)
    for parameter in renderer.parameters():
        parameter.requires_grad_(False)

    if mode_uses_scale_carrier(mode):
        renderer.raw_scales.requires_grad_(True)
        trainable_tensor = renderer.raw_scales
    else:
        renderer.raw_thetas.requires_grad_(True)
        trainable_tensor = renderer.raw_thetas
    optimizer = torch.optim.Adam([trainable_tensor], lr=tune_lr)
    carrier_mask = torch.zeros(trainable_tensor.shape[0], dtype=torch.bool, device=trainable_tensor.device)
    carrier_mask[carrier_indices] = True
    expand_dims = [1] * (trainable_tensor.ndim - 1)
    grad_mask = carrier_mask.view(-1, *expand_dims)

    for _ in range(tune_steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = renderer(grid[..., 0], grid[..., 1], clamp_output=False)
        params = renderer.decode_parameters()
        total_loss = F.mse_loss(reconstruction, target)
        total_loss = total_loss + msg_strength * message_loss(
            params,
            carrier_indices,
            coded_bits,
            mode=mode,
            target_magnitude=target_magnitude,
        )
        if dist_weight > 0.0:
            total_loss = total_loss + dist_weight * matched_moment_loss(params, clean_reference)
        if anchor_weight > 0.0 and mode_uses_scale_carrier(mode):
            total_loss = total_loss + anchor_weight * anchor_loss_from_scales(
                renderer=renderer,
                clean_scales=clean_scales,
                carrier_indices=carrier_indices,
                sensitivity_weights=sensitivity_weights,
            )
        total_loss.backward()
        if trainable_tensor.grad is not None:
            trainable_tensor.grad.mul_(grad_mask)
        torch.nn.utils.clip_grad_norm_([trainable_tensor], max_norm=1.0)
        optimizer.step()
    return renderer


def run_joint_embedding(
    *,
    source_run,
    target: torch.Tensor,
    grid: torch.Tensor,
    device: torch.device,
    carrier_indices: torch.Tensor,
    coded_bits: torch.Tensor,
    mode: str,
    msg_strength: float,
    dist_weight: float,
    anchor_weight: float,
    target_magnitude: float,
    clean_reference: dict[str, torch.Tensor],
    clean_scales: torch.Tensor,
    sensitivity_weights: torch.Tensor,
    joint_steps: int | None,
) -> tuple[Gaussian2DRenderer, int]:
    config = experiment_config_from_export(source_run, device=device, steps=joint_steps)
    set_seed(config.seed, deterministic=True)
    renderer = build_joint_renderer(source_run, target, grid, device)
    optimizer = build_optimizer(renderer, config)
    scheduler = build_scheduler(optimizer, config)

    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = renderer(grid[..., 0], grid[..., 1], clamp_output=False)
        params = renderer.decode_parameters()
        total_loss = F.mse_loss(reconstruction, target)
        total_loss = total_loss + msg_strength * message_loss(
            params,
            carrier_indices,
            coded_bits,
            mode=mode,
            target_magnitude=target_magnitude,
        )
        if dist_weight > 0.0:
            total_loss = total_loss + dist_weight * matched_moment_loss(params, clean_reference)
        if anchor_weight > 0.0 and mode_uses_scale_carrier(mode):
            total_loss = total_loss + anchor_weight * anchor_loss_from_scales(
                renderer=renderer,
                clean_scales=clean_scales,
                carrier_indices=carrier_indices,
                sensitivity_weights=sensitivity_weights,
            )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(renderer.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return renderer, config.steps


def _row_mask_like(parameter: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(parameter.shape[0], dtype=torch.bool, device=parameter.device)
    if row_indices.numel() > 0:
        mask[row_indices.to(parameter.device)] = True
    expand_dims = [1] * (parameter.ndim - 1)
    return mask.view(-1, *expand_dims)


def _masked_rows_step(
    parameter: torch.Tensor,
    row_indices: torch.Tensor,
) -> None:
    if parameter.grad is None:
        return
    parameter.grad.mul_(_row_mask_like(parameter, row_indices))


def _decode_cost_stego_payload(
    params: dict[str, torch.Tensor],
    *,
    clean_params: dict[str, torch.Tensor],
    assignments: list[dict[str, object]],
    payload_bits: int,
    ecc_mode: str,
    log_delta: float,
    alpha_delta: float,
    alpha_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], torch.Tensor]:
    decoded_coded_bits, soft_scores = decode_parity_blocks(
        params,
        clean_params=clean_params,
        assignments=assignments,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
        alpha_weight=alpha_weight,
    )
    decoded_raw_bits, decode_meta = decode_payload(decoded_coded_bits, raw_length=payload_bits, ecc_mode=ecc_mode)
    return decoded_coded_bits, decoded_raw_bits, decode_meta, soft_scores


def run_cost_stego_embedding(
    *,
    clean_renderer: Gaussian2DRenderer,
    clean_params: dict[str, torch.Tensor],
    clean_importance: torch.Tensor,
    clean_sensitivity: dict[str, torch.Tensor],
    target: torch.Tensor,
    grid: torch.Tensor,
    coded_bits: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[Gaussian2DRenderer, dict[str, object]]:
    if args.stage != "postfit":
        raise ValueError("cost_stego_v1 currently supports only --stage postfit")

    defaults = _cost_stego_defaults(args)
    requested_spread_factor = int(args.spread_factor if args.spread_factor is not None else default_spread_factor(args.payload_bits))
    alpha_delta = _alpha_delta_from_target(args.target_magnitude)
    cost_bundle = build_channel_costs(
        clean_params=clean_params,
        importance=clean_importance.to(clean_params["offsets"].device),
        sensitivity={key: value.to(clean_params["offsets"].device) for key, value in clean_sensitivity.items()},
        wet_threshold=float(args.wet_threshold),
        log_delta=float(args.target_magnitude),
        alpha_delta=alpha_delta,
        cost_sensitivity_w=float(getattr(args, "cost_sensitivity_weight", 0.40)),
        cost_density_w=float(getattr(args, "cost_density_weight", 0.30)),
        cost_visual_w=float(getattr(args, "cost_visual_weight", 0.20)),
        cost_detector_w=float(getattr(args, "cost_detector_weight", 0.10)),
    )
    effective_wet_mask, spread_factor, relaxed_wet_indices, dry_count = adapt_wet_mask_for_capacity(
        cost_bundle=cost_bundle,
        coded_bits=coded_bits,
        requested_spread_factor=requested_spread_factor,
    )
    capacity_scale = max(1.0, float(requested_spread_factor) / float(max(1, spread_factor)))
    relaxation_scale = 1.0 + 0.5 * (float(len(relaxed_wet_indices)) / float(max(1, dry_count)))
    if spread_factor <= 0:
        return deepcopy(clean_renderer), {
            **capacity_failure_report(payload_bits=args.payload_bits, coded_payload_bits=int(coded_bits.numel())),
            "assignments": [],
            "wrong_key_assignments": [],
            "spread_factor": requested_spread_factor,
            "requested_spread_factor": requested_spread_factor,
            "alpha_delta": alpha_delta,
            "log_delta": float(args.target_magnitude),
            "compensation_indices": [],
            "detector_feature_names": detector_feature_names(),
            "pooled_detectability_auc": None,
            "false_alarm_rate": None,
            "security_model": DEFAULT_SECURITY_MODEL,
            "stego_protocol": DEFAULT_STEGO_PROTOCOL,
            "relaxed_wet_indices": [],
            "relaxed_wet_count": 0,
            "requested_wet_ratio": float(cost_bundle.wet_mask.float().mean().item()),
            "dry_count_before_relaxation": dry_count,
        }
    assignments, capacity_failure, total_embed_cost = select_keyed_parity_blocks(
        coded_bits=coded_bits,
        cost_bundle=cost_bundle,
        key=args.key,
        spread_factor=spread_factor,
        wet_mask_override=effective_wet_mask,
    )
    wrong_key_assignments, _, _ = select_keyed_parity_blocks(
        coded_bits=coded_bits,
        cost_bundle=cost_bundle,
        key=f"{args.key}__wrong",
        spread_factor=spread_factor,
        wet_mask_override=effective_wet_mask,
    )
    if capacity_failure:
        return deepcopy(clean_renderer), {
            **capacity_failure_report(payload_bits=args.payload_bits, coded_payload_bits=int(coded_bits.numel())),
            "assignments": [],
            "wrong_key_assignments": [],
            "spread_factor": spread_factor,
            "requested_spread_factor": requested_spread_factor,
            "alpha_delta": alpha_delta,
            "log_delta": float(args.target_magnitude),
            "compensation_indices": [],
            "detector_feature_names": detector_feature_names(),
            "pooled_detectability_auc": None,
            "false_alarm_rate": None,
            "security_model": DEFAULT_SECURITY_MODEL,
            "stego_protocol": DEFAULT_STEGO_PROTOCOL,
            "relaxed_wet_indices": relaxed_wet_indices,
            "relaxed_wet_count": len(relaxed_wet_indices),
            "requested_wet_ratio": float(cost_bundle.wet_mask.float().mean().item()),
            "dry_count_before_relaxation": dry_count,
        }

    renderer = deepcopy(clean_renderer)
    for parameter in renderer.parameters():
        parameter.requires_grad_(False)
    renderer.raw_scales.requires_grad_(True)
    renderer.raw_opacities.requires_grad_(True)
    optimizer = torch.optim.Adam([renderer.raw_scales, renderer.raw_opacities], lr=args.tune_lr)

    detector_clean_features = build_detector_feature_vector(
        clean_params,
        importance=clean_importance.to(clean_params["offsets"].device),
    )
    detector = RunLevelStegaDetector(input_dim=int(detector_clean_features.numel())).to(clean_params["offsets"].device)
    detector_optimizer = torch.optim.Adam(detector.parameters(), lr=0.01)

    sensitivity_device = {key: value.to(clean_params["offsets"].device) for key, value in clean_sensitivity.items()}
    density_reference = build_density_reference(clean_params)
    clean_moment_reference = build_clean_distribution_reference(clean_params)
    compensation_indices = select_compensation_neighbors(
        clean_params["offsets"],
        assignments=assignments,
        wet_mask=effective_wet_mask,
        per_active=2,
    ).to(clean_params["offsets"].device)
    active_log_indices, active_alpha_indices = active_channel_indices(assignments)
    active_log_indices = active_log_indices.to(clean_params["offsets"].device)
    active_alpha_indices = active_alpha_indices.to(clean_params["offsets"].device)
    embed_target_bits = assignment_bit_tensor(assignments, field="embed_bit").to(clean_params["offsets"].device).to(torch.float32)

    for step in range(args.tune_steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = renderer(grid[..., 0], grid[..., 1], clamp_output=False)
        params = renderer.decode_parameters()
        importance = compute_gaussian_importance(params["scales"], params["opacities"])
        block_logits = parity_block_logits(
            params,
            clean_params=clean_params,
            assignments=assignments,
            log_delta=float(args.target_magnitude),
            alpha_delta=alpha_delta,
            alpha_weight=float(args.alpha_weight),
        )
        wrong_block_logits = parity_block_logits(
            params,
            clean_params=clean_params,
            assignments=wrong_key_assignments,
            log_delta=float(args.target_magnitude),
            alpha_delta=alpha_delta,
            alpha_weight=float(args.alpha_weight),
        )
        target_bits = embed_target_bits.to(block_logits.device)
        recon_loss = F.mse_loss(reconstruction, target)
        msg_loss = F.binary_cross_entropy_with_logits(block_logits, target_bits)
        signed_logits = block_logits * (2.0 * target_bits - 1.0)
        msg_margin_loss = torch.relu(0.75 - signed_logits).mean()
        carrier_direction_loss = carrier_margin_loss(
            params,
            clean_params=clean_params,
            assignments=assignments,
            log_delta=float(args.target_magnitude),
            alpha_delta=alpha_delta,
            margin=0.75,
        )
        msg_loss = msg_loss + 0.25 * msg_margin_loss + 0.25 * carrier_direction_loss
        wrong_key_loss = wrong_block_logits.abs().mean() if wrong_block_logits.numel() > 0 else torch.tensor(0.0, device=block_logits.device)
        moment_loss = matched_moment_loss(params, clean_moment_reference)
        mmd_loss = mmd_distribution_loss(clean_params, params)
        detector_features = build_detector_feature_vector(params, importance=importance)
        detector_loss = detector_embed_loss(detector, detector_features)
        ewc_loss = carrier_ewc_penalty(
            renderer,
            clean_renderer,
            assignments=assignments,
            sensitivity=sensitivity_device,
            alpha_weight=float(args.alpha_weight),
        )
        compensation_loss = density_reference_loss(params, density_reference)
        total_loss = recon_loss
        total_loss = total_loss + defaults["msg_strength"] * capacity_scale * msg_loss
        total_loss = total_loss + float(args.covertness_weight) * (
            defaults["mmd_weight"] * mmd_loss
            + defaults["moment_weight"] * moment_loss
            + defaults["detector_weight"] * detector_loss
            + defaults["compensation_weight"] * compensation_loss
        )
        total_loss = total_loss + defaults["wrong_key_weight"] * wrong_key_loss
        total_loss = total_loss + defaults["ewc_weight"] * relaxation_scale * ewc_loss
        total_loss.backward()
        _masked_rows_step(renderer.raw_scales, active_log_indices)
        _masked_rows_step(renderer.raw_opacities, active_alpha_indices)
        torch.nn.utils.clip_grad_norm_([renderer.raw_scales, renderer.raw_opacities], max_norm=1.0)
        optimizer.step()
        train_detector_steps(
            detector,
            detector_optimizer,
            clean_features=detector_clean_features,
            wm_features=detector_features,
            steps=3,
        )

        if compensation_indices.numel() > 0 and (step + 1) % 10 == 0:
            for _ in range(5):
                optimizer.zero_grad(set_to_none=True)
                reconstruction = renderer(grid[..., 0], grid[..., 1], clamp_output=False)
                params = renderer.decode_parameters()
                importance = compute_gaussian_importance(params["scales"], params["opacities"])
                detector_features = build_detector_feature_vector(params, importance=importance)
                compensation_objective = F.mse_loss(reconstruction, target)
                compensation_objective = compensation_objective + float(args.covertness_weight) * (
                    defaults["mmd_weight"] * mmd_distribution_loss(clean_params, params)
                    + defaults["detector_weight"] * detector_embed_loss(detector, detector_features)
                    + defaults["compensation_weight"] * density_reference_loss(params, density_reference)
                )
                compensation_objective.backward()
                _masked_rows_step(renderer.raw_scales, compensation_indices)
                _masked_rows_step(renderer.raw_opacities, compensation_indices)
                torch.nn.utils.clip_grad_norm_([renderer.raw_scales, renderer.raw_opacities], max_norm=1.0)
                optimizer.step()
                train_detector_steps(
                    detector,
                    detector_optimizer,
                    clean_features=detector_clean_features,
                    wm_features=detector_features,
                    steps=1,
                )

    return renderer, {
        "capacity_failure": False,
        "assignments": assignments,
        "wrong_key_assignments": wrong_key_assignments,
        "spread_factor": spread_factor,
        "requested_spread_factor": requested_spread_factor,
        "alpha_delta": alpha_delta,
        "log_delta": float(args.target_magnitude),
        "wet_ratio": float(effective_wet_mask.float().mean().item()),
        "requested_wet_ratio": float(cost_bundle.wet_mask.float().mean().item()),
        "modified_gaussian_count": modified_gaussian_count(assignments),
        "total_embed_cost": float(total_embed_cost),
        "compensation_indices": [int(index) for index in compensation_indices.detach().cpu().tolist()],
        "detector_feature_names": detector_feature_names(),
        "pooled_detectability_auc": None,
        "false_alarm_rate": None,
        "security_model": DEFAULT_SECURITY_MODEL,
        "stego_protocol": DEFAULT_STEGO_PROTOCOL,
        "relaxed_wet_indices": relaxed_wet_indices,
        "relaxed_wet_count": len(relaxed_wet_indices),
        "dry_count_before_relaxation": dry_count,
    }


def run_cost_stego_v2_embedding(
    *,
    clean_renderer: Gaussian2DRenderer,
    clean_params: dict[str, torch.Tensor],
    clean_importance: torch.Tensor,
    clean_sensitivity: dict[str, torch.Tensor],
    clean_clamped_psnr: float,
    target: torch.Tensor,
    grid: torch.Tensor,
    raw_bits: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[Gaussian2DRenderer, dict[str, object]]:
    if args.stage != "postfit":
        raise ValueError("cost_stego_v2 currently supports only --stage postfit")

    defaults = _cost_stego_v2_defaults(args)
    adaptive_controls = _adaptive_v2_embedding_controls(
        args=args,
        clean_params=clean_params,
        clean_importance=clean_importance,
        clean_sensitivity=clean_sensitivity,
    )
    effective_target_magnitude = float(adaptive_controls["target_magnitude"])
    alpha_delta = float(adaptive_controls["alpha_delta"])
    cost_bundle = build_channel_costs(
        clean_params=clean_params,
        importance=clean_importance.to(clean_params["offsets"].device),
        sensitivity={key: value.to(clean_params["offsets"].device) for key, value in clean_sensitivity.items()},
        wet_threshold=float(args.wet_threshold),
        log_delta=float(effective_target_magnitude),
        alpha_delta=alpha_delta,
        cost_sensitivity_w=float(getattr(args, "cost_sensitivity_weight", 0.40)),
        cost_density_w=float(getattr(args, "cost_density_weight", 0.30)),
        cost_visual_w=float(getattr(args, "cost_visual_weight", 0.20)),
        cost_detector_w=float(getattr(args, "cost_detector_weight", 0.10)),
    )
    max_modification_ratio = float(adaptive_controls["max_modification_ratio"])
    assignment_builder = {
        "native": build_cost_code_assignments,
        "native_matched_random": build_random_cost_code_assignments,
        "native_greedy_nokey": build_greedy_nokey_cost_code_assignments,
        "native_random_polarity": build_random_polarity_cost_code_assignments,
    }.get(args.baseline_type, build_cost_code_assignments)
    selection_field_builder = (
        build_soft_selection_field
        if _supports_keyed_assignment_family(args.baseline_type)
        else build_unkeyed_selection_field
    )
    detector_training_requested = _uses_native_parameter_embedding(args.baseline_type) and _native_detector_requested(args)
    wrong_key_training_requested = _supports_keyed_assignment_family(args.baseline_type) and _native_wrong_key_requested(args)
    attack_training_requested = _uses_native_parameter_embedding(args.baseline_type) and _native_attack_training_requested(args)
    try:
        assignments, assignment_meta = assignment_builder(
            raw_bits=raw_bits.detach().cpu(),
            clean_params=clean_params,
            importance=clean_importance.to(clean_params["offsets"].device),
            sensitivity={key: value.to(clean_params["offsets"].device) for key, value in clean_sensitivity.items()},
            cost_bundle=cost_bundle,
            key=args.key,
            candidate_count=24,
            fallback_candidate_count=16,
            beam_width=int(args.beam_width),
            selection_temperature=float(args.selection_temperature),
            max_modification_ratio=max_modification_ratio,
        )
    except RuntimeError:
        return deepcopy(clean_renderer), {
            **capacity_failure_report(payload_bits=args.payload_bits, coded_payload_bits=int(raw_bits.numel())),
            "coding_mode": "cost_code_v1",
            "assignments": [],
            "wrong_key_assignments": [],
            "spread_factor": None,
            "requested_spread_factor": None,
            "alpha_delta": alpha_delta,
            "log_delta": float(effective_target_magnitude),
            "wet_ratio": None,
            "requested_wet_ratio": None,
            "relaxed_wet_count": 0,
            "relaxed_wet_indices": [],
            "dry_count_before_relaxation": None,
            "compensation_indices": [],
            "detector_feature_names": detector_feature_names(),
            "pooled_detectability_auc": None,
            "false_alarm_rate": None,
            "same_source_auc": None,
            "cross_source_auc": None,
            "actor_holdout_auc": None,
            "security_model": DEFAULT_SECURITY_MODEL,
            "stego_protocol": COST_STEGO_V2_PROTOCOL,
            "selection_entropy": None,
            "effective_action_budget": 0,
            "qualified_cover": False,
            "qualified_cover_reason": "capacity_failure_assignment_builder",
            "qualified_cover_failed_checks": ["capacity_failure"],
            "qualified_cover_diagnostics": {
                "qualified_cover": False,
                "reason": "capacity_failure_assignment_builder",
                "failed_checks": ["capacity_failure"],
                "clean_clamped_psnr": float(clean_clamped_psnr),
                "clean_clamped_psnr_threshold": 30.0,
                "gaussian_count": int(clean_params["offsets"].shape[0]),
                "gaussian_count_threshold": 192,
                "effective_action_budget": 0,
                "effective_action_budget_threshold": int(math.ceil(2.5 * float(raw_bits.numel()))),
                "payload_bits": int(raw_bits.numel()),
            },
            "baseline_type": args.baseline_type,
            "native_ablation": args.native_ablation,
            "detector_training_enabled": False,
            "detector_training_requested": detector_training_requested,
            "effective_detector_weight": 0.0,
            "wrong_key_training_enabled": False,
            "wrong_key_training_requested": wrong_key_training_requested,
            "attack_training_enabled": False,
            "attack_training_requested": attack_training_requested,
            "trained_wrong_key_groups": 0,
            "wrong_key_sample_failures": 0,
            "adaptive_budget_enabled": bool(adaptive_controls["enabled"]),
            "adaptive_cover_profile": str(adaptive_controls["profile"]),
            "adaptive_profile_stats": dict(adaptive_controls["profile_stats"]),
            "effective_target_magnitude": float(effective_target_magnitude),
            "effective_max_modification_ratio": float(max_modification_ratio),
        }
    detector_training_enabled = detector_training_requested and int(assignment_meta["effective_action_budget"]) >= 128
    effective_detector_weight = defaults["detector_weight"] if detector_training_enabled else 0.0
    wrong_key_training_enabled = wrong_key_training_requested and int(assignment_meta["effective_action_budget"]) >= 128
    attack_training_enabled = attack_training_requested and int(assignment_meta["effective_action_budget"]) >= 128

    def build_assignment_group_for_key(key_suffix: str) -> list[dict[str, object]]:
        return assignment_builder(
            raw_bits=raw_bits.detach().cpu(),
            clean_params=clean_params,
            importance=clean_importance.to(clean_params["offsets"].device),
            sensitivity={key: value.to(clean_params["offsets"].device) for key, value in clean_sensitivity.items()},
            cost_bundle=cost_bundle,
            key=key_suffix,
            candidate_count=24,
            fallback_candidate_count=16,
            beam_width=int(args.beam_width),
            selection_temperature=float(args.selection_temperature),
            max_modification_ratio=max_modification_ratio,
        )[0]

    training_wrong_key_assignments: list[list[dict[str, object]]] = []
    if wrong_key_training_enabled:
        for wrong_index in range(int(args.wrong_key_samples)):
            try:
                training_wrong_key_assignments.append(build_assignment_group_for_key(f"{args.key}::wrong::{wrong_index}"))
            except RuntimeError:
                continue
    if _supports_keyed_assignment_family(args.baseline_type):
        wrong_key_assignments, wrong_key_sample_failures = _collect_wrong_key_assignment_groups(
            requested_samples=int(args.wrong_key_samples),
            builder=lambda wrong_index: build_assignment_group_for_key(f"{args.key}::wrong::{wrong_index}"),
        )
    else:
        wrong_key_assignments = []
        wrong_key_sample_failures = 0

    renderer = deepcopy(clean_renderer)
    for parameter in renderer.parameters():
        parameter.requires_grad_(False)
    renderer.raw_scales.requires_grad_(True)
    renderer.raw_opacities.requires_grad_(True)
    optimizer = torch.optim.Adam([renderer.raw_scales, renderer.raw_opacities], lr=args.tune_lr)

    detector_clean_features = build_detector_feature_vector(
        clean_params,
        importance=clean_importance.to(clean_params["offsets"].device),
    )
    detector = RunLevelStegaDetector(input_dim=int(detector_clean_features.numel())).to(clean_params["offsets"].device)
    detector_optimizer = torch.optim.Adam(detector.parameters(), lr=0.01)

    density_reference = build_density_reference(clean_params)
    clean_moment_reference = build_clean_distribution_reference(clean_params)
    selection_kwargs = {
        "clean_params": clean_params,
        "importance": clean_importance.to(clean_params["offsets"].device),
        "sensitivity": {key: value.to(clean_params["offsets"].device) for key, value in clean_sensitivity.items()},
        "cost_bundle": cost_bundle,
        "selection_temperature": float(args.selection_temperature),
    }
    if selection_field_builder is build_soft_selection_field:
        selection_bundle = selection_field_builder(key=args.key, **selection_kwargs)
    else:
        selection_bundle = selection_field_builder(**selection_kwargs)
    selected_log_indices, selected_alpha_indices = _v2_selected_indices(assignments)
    selected_log_indices = selected_log_indices.to(clean_params["offsets"].device)
    selected_alpha_indices = selected_alpha_indices.to(clean_params["offsets"].device)
    compensation_indices = select_compensation_neighbors(
        clean_params["offsets"],
        assignments=_v2_assignments_for_compensation(assignments),
        wet_mask=selection_bundle.hard_forbid_mask,
        per_active=2,
    ).to(clean_params["offsets"].device)
    raw_targets = raw_bits.to(clean_params["offsets"].device).to(torch.float32)
    if raw_targets.numel() % 4 != 0:
        raw_targets = torch.cat((raw_targets, torch.zeros((-raw_targets.numel()) % 4, dtype=raw_targets.dtype, device=raw_targets.device)))
    block_targets = raw_targets.view(-1, 4).to(clean_params["offsets"].device)
    attack_schedule = [
        "quantize_step_0.01",
        "small_parameter_noise_0.01",
        "importance_pruning_10pct",
    ]
    small_noise_generator = torch.Generator(device="cpu")
    small_noise_generator.manual_seed(key_to_seed(f"{args.key}::attack_noise::{args.message_seed}::{args.fit_seed}") + 17)
    attack_noise_log = (
        torch.empty((selected_log_indices.numel(), 2), dtype=clean_params["scales"].dtype).uniform_(-0.01, 0.01, generator=small_noise_generator)
        if selected_log_indices.numel() > 0
        else None
    )
    attack_noise_alpha = (
        torch.empty((selected_alpha_indices.numel(), 1), dtype=clean_params["opacities"].dtype).uniform_(-0.01, 0.01, generator=small_noise_generator)
        if selected_alpha_indices.numel() > 0
        else None
    )
    native_main_method = str(args.baseline_type) == "native"

    def compute_true_key_margin_terms(
        block_probs: torch.Tensor,
        action_scores: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_sign = 2.0 * block_targets - 1.0
        bit_alignment = target_sign * (2.0 * block_probs - 1.0)
        true_key_margin = bit_alignment.mean()
        bit_margin_penalty = torch.relu(
            torch.tensor(0.65, dtype=block_probs.dtype, device=block_probs.device) - bit_alignment
        ).mean()
        mean_margin_penalty = torch.relu(
            torch.tensor(0.55, dtype=block_probs.dtype, device=block_probs.device) - true_key_margin
        )
        action_losses: list[torch.Tensor] = []
        for scores, assignment in zip(action_scores, assignments):
            if scores.numel() == 0:
                continue
            selected = torch.tensor(assignment["selected_mask"], dtype=torch.float32, device=scores.device) > 0.5
            selected_loss = (
                torch.relu(torch.tensor(0.85, dtype=scores.dtype, device=scores.device) - scores[selected]).mean()
                if bool(selected.any())
                else torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
            )
            unselected_loss = (
                torch.relu(scores[~selected] + torch.tensor(0.05, dtype=scores.dtype, device=scores.device)).mean()
                if bool((~selected).any())
                else torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
            )
            action_losses.append(selected_loss + 0.50 * unselected_loss)
        action_polarity_penalty = (
            torch.stack(action_losses).mean()
            if action_losses
            else torch.tensor(0.0, dtype=block_probs.dtype, device=block_probs.device)
        )
        return true_key_margin, bit_margin_penalty, mean_margin_penalty, action_polarity_penalty

    def compute_wrong_key_terms(
        params: dict[str, torch.Tensor],
        correct_block_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.tensor(0.0, dtype=correct_block_probs.dtype, device=correct_block_probs.device)
        if not training_wrong_key_assignments:
            return zero, zero, zero, zero
        if native_main_method:
            contrastive_loss, contrastive_gap_penalty = wrong_key_contrastive_loss(
                params,
                clean_params=clean_params,
                raw_bits=block_targets.to(torch.int64).flatten(),
                correct_assignments=assignments,
                wrong_assignments=training_wrong_key_assignments,
                log_delta=float(effective_target_magnitude),
                alpha_delta=alpha_delta,
                beta=4.0,
                margin=0.45,
            )
            correct_alignment = (2.0 * block_targets - 1.0) * (2.0 * correct_block_probs - 1.0)
            correct_margin_penalty = torch.relu(
                torch.tensor(0.65, dtype=correct_block_probs.dtype, device=correct_block_probs.device) - correct_alignment
            ).mean()
            wrong_uniform_losses: list[torch.Tensor] = []
            wrong_margin_caps: list[torch.Tensor] = []
            wrong_gap_penalties: list[torch.Tensor] = []
            for wrong_assignment_group in training_wrong_key_assignments:
                wrong_probs, _ = cost_code_block_probabilities(
                    params,
                    clean_params=clean_params,
                    assignments=wrong_assignment_group,
                    log_delta=float(effective_target_magnitude),
                    alpha_delta=alpha_delta,
                    beta=4.0,
                )
                wrong_alignment = torch.abs((2.0 * block_targets - 1.0) * (2.0 * wrong_probs - 1.0))
                wrong_uniform_losses.append(
                    F.binary_cross_entropy(wrong_probs, torch.full_like(wrong_probs, 0.5))
                )
                wrong_margin_caps.append(
                    torch.relu(
                        wrong_alignment - torch.tensor(0.08, dtype=wrong_probs.dtype, device=wrong_probs.device)
                    ).mean()
                )
                wrong_gap_penalties.append(
                    torch.relu(
                        torch.tensor(0.40, dtype=wrong_probs.dtype, device=wrong_probs.device)
                        - (correct_alignment - wrong_alignment)
                    ).mean()
                )
            return (
                contrastive_loss,
                correct_margin_penalty,
                torch.stack(wrong_uniform_losses).mean(),
                contrastive_gap_penalty + torch.stack(wrong_margin_caps).mean() + torch.stack(wrong_gap_penalties).mean(),
            )
        correct_confidence = torch.abs(correct_block_probs - 0.5).mean() * 2.0
        wrong_losses: list[torch.Tensor] = []
        wrong_confidences: list[torch.Tensor] = []
        for wrong_assignment_group in training_wrong_key_assignments:
            wrong_probs, _ = cost_code_block_probabilities(
                params,
                clean_params=clean_params,
                assignments=wrong_assignment_group,
                log_delta=float(effective_target_magnitude),
                alpha_delta=alpha_delta,
                beta=4.0,
            )
            wrong_losses.append(
                F.binary_cross_entropy(
                    wrong_probs,
                    torch.full_like(wrong_probs, 0.5),
                )
            )
            wrong_confidences.append(torch.abs(wrong_probs - 0.5).mean() * 2.0)
        wrong_key_loss = torch.stack(wrong_losses).mean()
        wrong_mean_confidence = torch.stack(wrong_confidences).mean()
        correct_confidence_penalty = torch.relu(
            torch.tensor(0.35, dtype=correct_block_probs.dtype, device=correct_block_probs.device)
            - correct_confidence
        )
        wrong_confidence_penalty = torch.relu(
            wrong_mean_confidence
            - torch.tensor(0.10, dtype=correct_block_probs.dtype, device=correct_block_probs.device)
        )
        wrong_key_gap_penalty = torch.relu(
            torch.tensor(0.20, dtype=correct_block_probs.dtype, device=correct_block_probs.device)
            - (correct_confidence - wrong_mean_confidence)
        )
        return wrong_key_loss, correct_confidence_penalty, wrong_confidence_penalty, wrong_key_gap_penalty

    for step in range(args.tune_steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = renderer(grid[..., 0], grid[..., 1], clamp_output=False)
        params = renderer.decode_parameters()
        importance = compute_gaussian_importance(params["scales"], params["opacities"])
        progress = 1.0 if int(args.tune_steps) <= 1 else float(step) / float(int(args.tune_steps) - 1)
        block_probs, action_scores = cost_code_block_probabilities(
            params,
            clean_params=clean_params,
            assignments=assignments,
            log_delta=float(effective_target_magnitude),
            alpha_delta=alpha_delta,
            beta=4.0,
        )
        action_target_loss = cost_code_action_target_loss(
            params,
            clean_params=clean_params,
            assignments=assignments,
            log_delta=float(effective_target_magnitude),
            alpha_delta=alpha_delta,
            beta=4.0,
        )
        action_margin_loss = cost_code_action_margin_loss(
            params,
            clean_params=clean_params,
            assignments=assignments,
            log_delta=float(effective_target_magnitude),
            alpha_delta=alpha_delta,
            decision_threshold=0.5,
            margin=0.25,
        )
        block_margin_loss = torch.relu(0.35 - (2.0 * block_targets - 1.0) * (2.0 * block_probs - 1.0)).mean()
        clean_block_bce = F.binary_cross_entropy(block_probs, block_targets)
        msg_loss = (
            clean_block_bce
            + 0.25 * action_target_loss
            + 0.25 * action_margin_loss
            + 0.25 * block_margin_loss
        )
        clean_true_key_push_loss = torch.tensor(0.0, dtype=block_probs.dtype, device=block_probs.device)
        clean_true_key_bit_penalty = torch.tensor(0.0, dtype=block_probs.dtype, device=block_probs.device)
        clean_true_key_mean_penalty = torch.tensor(0.0, dtype=block_probs.dtype, device=block_probs.device)
        clean_action_polarity_penalty = torch.tensor(0.0, dtype=block_probs.dtype, device=block_probs.device)
        if native_main_method:
            clean_true_key_margin, clean_true_key_bit_penalty, clean_true_key_mean_penalty, clean_action_polarity_penalty = (
                compute_true_key_margin_terms(block_probs, action_scores)
            )
            clean_true_key_push_loss = torch.clamp(1.0 - clean_true_key_margin, min=0.0)
        moment_loss = matched_moment_loss(params, clean_moment_reference)
        mmd_loss = mmd_distribution_loss(clean_params, params)
        wasserstein_loss = sliced_wasserstein_loss(clean_params, params, projections=32, seed=args.message_seed + step)
        selection_penalty = selection_channel_penalty(
            selection_probs=selection_bundle.selection_probs.to(params["offsets"].device),
            clean_params=clean_params,
            importance=clean_importance.to(params["offsets"].device),
            sensitivity={key: value.to(params["offsets"].device) for key, value in clean_sensitivity.items()},
        )
        detector_features = build_detector_feature_vector(params, importance=importance)
        detector_loss = detector_embed_loss(detector, detector_features)
        if not detector_training_enabled:
            detector_loss = torch.tensor(0.0, dtype=detector_features.dtype, device=detector_features.device)
        ewc_loss = carrier_ewc_penalty(
            renderer,
            clean_renderer,
            assignments=_v2_assignments_for_compensation(assignments),
            sensitivity={key: value.to(params["offsets"].device) for key, value in clean_sensitivity.items()},
            alpha_weight=float(args.alpha_weight),
        )
        compensation_loss = density_reference_loss(params, density_reference)
        clean_wrong_key_loss = torch.tensor(0.0, dtype=params["offsets"].dtype, device=params["offsets"].device)
        clean_correct_penalty = torch.tensor(0.0, dtype=params["offsets"].dtype, device=params["offsets"].device)
        clean_wrong_penalty = torch.tensor(0.0, dtype=params["offsets"].dtype, device=params["offsets"].device)
        clean_gap_penalty = torch.tensor(0.0, dtype=params["offsets"].dtype, device=params["offsets"].device)
        if wrong_key_training_enabled:
            clean_wrong_key_loss, clean_correct_penalty, clean_wrong_penalty, clean_gap_penalty = compute_wrong_key_terms(params, block_probs)
        total_loss = F.mse_loss(reconstruction, target)
        total_loss = total_loss + defaults["msg_strength"] * msg_loss
        total_loss = total_loss + float(args.covertness_weight) * (
            defaults["mmd_weight"] * mmd_loss
            + defaults["wasserstein_weight"] * wasserstein_loss
            + defaults["selection_weight"] * selection_penalty
            + defaults["moment_weight"] * moment_loss
            + effective_detector_weight * detector_loss
            + defaults["compensation_weight"] * compensation_loss
        )
        total_loss = total_loss + defaults["ewc_weight"] * ewc_loss
        if native_main_method:
            total_loss = total_loss + defaults["native_margin_weight"] * (
                clean_true_key_push_loss
                + clean_true_key_bit_penalty
                + clean_true_key_mean_penalty
                + 0.5 * clean_action_polarity_penalty
            )
        if wrong_key_training_enabled:
            if native_main_method:
                total_loss = total_loss + defaults["wrong_key_weight"] * clean_wrong_key_loss
                total_loss = total_loss + defaults["native_wrong_key_specificity_weight"] * (
                    clean_correct_penalty
                    + clean_wrong_penalty
                    + clean_gap_penalty
                )
            else:
                total_loss = total_loss + defaults["wrong_key_weight"] * (
                    clean_wrong_key_loss
                    + clean_correct_penalty
                    + clean_wrong_penalty
                    + clean_gap_penalty
                )
        progress = 1.0 if int(args.tune_steps) <= 1 else float(step) / float(int(args.tune_steps) - 1)
        if attack_training_enabled and progress >= 0.60:
            attack_name = attack_schedule[(step - int(0.60 * max(1, int(args.tune_steps)))) % len(attack_schedule)]
            attacked_params = _carrier_preserving_attack_params(
                params,
                attack_name=attack_name,
                assignments=assignments,
                small_noise_log=attack_noise_log,
                small_noise_alpha=attack_noise_alpha,
            )
            attacked_block_probs, attacked_action_scores = cost_code_block_probabilities(
                attacked_params,
                clean_params=clean_params,
                assignments=assignments,
                log_delta=float(effective_target_magnitude),
                alpha_delta=alpha_delta,
                beta=4.0,
            )
            attacked_block_bce = F.binary_cross_entropy(attacked_block_probs, block_targets)
            attacked_block_margin = torch.relu(0.35 - (2.0 * block_targets - 1.0) * (2.0 * attacked_block_probs - 1.0)).mean()
            total_loss = total_loss + 0.5 * defaults["msg_strength"] * (attacked_block_bce + 0.25 * attacked_block_margin)
            if native_main_method:
                (
                    attacked_true_key_margin,
                    attacked_true_key_bit_penalty,
                    attacked_true_key_mean_penalty,
                    attacked_action_polarity_penalty,
                ) = compute_true_key_margin_terms(attacked_block_probs, attacked_action_scores)
                attacked_true_key_push_loss = torch.clamp(1.0 - attacked_true_key_margin, min=0.0)
                total_loss = total_loss + 0.5 * defaults["native_margin_weight"] * (
                    attacked_true_key_push_loss
                    + attacked_true_key_bit_penalty
                    + attacked_true_key_mean_penalty
                    + 0.5 * attacked_action_polarity_penalty
                )
            if wrong_key_training_enabled:
                attacked_wrong_key_loss, attacked_correct_penalty, attacked_wrong_penalty, attacked_gap_penalty = compute_wrong_key_terms(
                    attacked_params,
                    attacked_block_probs,
                )
                if native_main_method:
                    total_loss = total_loss + 0.5 * defaults["wrong_key_weight"] * attacked_wrong_key_loss
                    total_loss = total_loss + 0.5 * defaults["native_wrong_key_specificity_weight"] * (
                        attacked_correct_penalty
                        + attacked_wrong_penalty
                        + attacked_gap_penalty
                    )
                else:
                    total_loss = total_loss + 0.5 * defaults["wrong_key_weight"] * (
                        attacked_wrong_key_loss
                        + attacked_correct_penalty
                        + attacked_wrong_penalty
                        + attacked_gap_penalty
                    )
        total_loss.backward()
        if selected_log_indices.numel() > 0:
            _masked_rows_step(renderer.raw_scales, selected_log_indices)
        if selected_alpha_indices.numel() > 0:
            _masked_rows_step(renderer.raw_opacities, selected_alpha_indices)
        torch.nn.utils.clip_grad_norm_([renderer.raw_scales, renderer.raw_opacities], max_norm=1.0)
        optimizer.step()
        if detector_training_enabled:
            train_detector_steps(
                detector,
                detector_optimizer,
                clean_features=detector_clean_features,
                wm_features=detector_features,
                steps=2,
            )
        if compensation_indices.numel() > 0 and (step + 1) % 8 == 0:
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                reconstruction = renderer(grid[..., 0], grid[..., 1], clamp_output=False)
                params = renderer.decode_parameters()
                importance = compute_gaussian_importance(params["scales"], params["opacities"])
                detector_features = build_detector_feature_vector(params, importance=importance)
                selection_penalty = selection_channel_penalty(
                    selection_probs=selection_bundle.selection_probs.to(params["offsets"].device),
                    clean_params=clean_params,
                    importance=clean_importance.to(params["offsets"].device),
                    sensitivity={key: value.to(params["offsets"].device) for key, value in clean_sensitivity.items()},
                )
                compensation_objective = F.mse_loss(reconstruction, target)
                compensation_objective = compensation_objective + float(args.covertness_weight) * (
                    defaults["mmd_weight"] * mmd_distribution_loss(clean_params, params)
                    + defaults["wasserstein_weight"] * sliced_wasserstein_loss(clean_params, params, projections=16, seed=step)
                    + defaults["selection_weight"] * selection_penalty
                    + effective_detector_weight * detector_embed_loss(detector, detector_features)
                    + defaults["compensation_weight"] * density_reference_loss(params, density_reference)
                )
                compensation_objective.backward()
                _masked_rows_step(renderer.raw_scales, compensation_indices)
                _masked_rows_step(renderer.raw_opacities, compensation_indices)
                torch.nn.utils.clip_grad_norm_([renderer.raw_scales, renderer.raw_opacities], max_norm=1.0)
                optimizer.step()

    qualified_cover_meta = _qualified_cover_diagnostics(
        clean_clamped_psnr=float(clean_clamped_psnr),
        gaussian_count=int(clean_params["offsets"].shape[0]),
        effective_action_budget=int(assignment_meta["effective_action_budget"]),
        payload_bits=int(raw_bits.numel()),
    )
    qualified_cover = bool(qualified_cover_meta["qualified_cover"])
    return renderer, {
        "capacity_failure": False,
        "coding_mode": "cost_code_v1",
        "assignments": assignments,
        "wrong_key_assignments": wrong_key_assignments,
        "spread_factor": None,
        "requested_spread_factor": None,
        "alpha_delta": alpha_delta,
        "log_delta": float(effective_target_magnitude),
        "wet_ratio": float(selection_bundle.hard_forbid_mask.float().mean().item()),
        "requested_wet_ratio": float(selection_bundle.hard_forbid_mask.float().mean().item()),
        "relaxed_wet_count": 0,
        "relaxed_wet_indices": [],
        "dry_count_before_relaxation": None,
        "modified_gaussian_count": int(sum(int(assignment["selected_count"]) for assignment in assignments)),
        "total_embed_cost": float(sum(sum(float(cost) * float(mask) for cost, mask in zip(assignment["candidate_costs"], assignment["selected_mask"])) for assignment in assignments)),
        "compensation_indices": [int(index) for index in compensation_indices.detach().cpu().tolist()],
        "detector_feature_names": detector_feature_names(),
        "pooled_detectability_auc": None,
        "false_alarm_rate": None,
        "same_source_auc": None,
        "cross_source_auc": None,
        "actor_holdout_auc": None,
        "security_model": DEFAULT_SECURITY_MODEL,
        "stego_protocol": COST_STEGO_V2_PROTOCOL,
        "selection_entropy": float(assignment_meta["selection_entropy"]),
        "effective_action_budget": int(assignment_meta["effective_action_budget"]),
        "qualified_cover": qualified_cover,
        "qualified_cover_reason": str(qualified_cover_meta["reason"]),
        "qualified_cover_failed_checks": list(qualified_cover_meta["failed_checks"]),
        "qualified_cover_diagnostics": qualified_cover_meta,
        "baseline_type": args.baseline_type,
        "native_ablation": args.native_ablation,
        "detector_training_enabled": detector_training_enabled,
        "detector_training_requested": detector_training_requested,
        "effective_detector_weight": float(effective_detector_weight),
        "wrong_key_training_enabled": wrong_key_training_enabled,
        "wrong_key_training_requested": wrong_key_training_requested,
        "attack_training_enabled": attack_training_enabled,
        "attack_training_requested": attack_training_requested,
        "trained_wrong_key_groups": len(training_wrong_key_assignments),
        "wrong_key_sample_failures": wrong_key_sample_failures,
        "adaptive_budget_enabled": bool(adaptive_controls["enabled"]),
        "adaptive_cover_profile": str(adaptive_controls["profile"]),
        "adaptive_profile_stats": dict(adaptive_controls["profile_stats"]),
        "effective_target_magnitude": float(effective_target_magnitude),
        "effective_max_modification_ratio": float(max_modification_ratio),
        "hard_forbid_ratio": float(assignment_meta["hard_forbid_ratio"]),
        "selection_probs": list(assignment_meta["selection_probs"]),
        "hard_forbid_mask": list(assignment_meta["hard_forbid_mask"]),
    }


def run_pixel_baseline_embedding(
    *,
    clean_renderer: Gaussian2DRenderer,
    clean_image: torch.Tensor,
    grid: torch.Tensor,
    raw_bits: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[Gaussian2DRenderer, dict[str, object]]:
    pixel_delta = _baseline_pixel_delta(args)
    candidate_ids, plus_costs, minus_costs, selection_logits = _pixel_candidate_field(clean_image)
    selection_probs = torch.softmax(selection_logits, dim=0)
    max_modification_count = max(1, int(math.floor(_v2_max_modification_ratio(args) * float(clean_image.shape[0] * clean_image.shape[1]))))
    try:
        assignments, assignment_meta = build_generic_cost_code_assignments(
            raw_bits=raw_bits.detach().cpu(),
            candidate_ids=candidate_ids.detach().cpu(),
            selection_logits=selection_logits.detach().cpu(),
            selection_probs=selection_probs.detach().cpu(),
            plus_costs=plus_costs.detach().cpu(),
            minus_costs=minus_costs.detach().cpu(),
            key=args.key,
            action_type="pixel_y",
            candidate_count=24,
            fallback_candidate_count=16,
            beam_width=int(args.beam_width),
            max_modification_count=max_modification_count,
        )
    except RuntimeError:
        return deepcopy(clean_renderer), {
            **capacity_failure_report(payload_bits=args.payload_bits, coded_payload_bits=int(raw_bits.numel())),
            "coding_mode": "cost_code_v1",
            "assignments": [],
            "wrong_key_assignments": [],
            "spread_factor": None,
            "requested_spread_factor": None,
            "alpha_delta": None,
            "log_delta": None,
            "wet_ratio": 0.0,
            "requested_wet_ratio": 0.0,
            "relaxed_wet_count": 0,
            "relaxed_wet_indices": [],
            "dry_count_before_relaxation": None,
            "compensation_indices": [],
            "detector_feature_names": detector_feature_names(),
            "pooled_detectability_auc": None,
            "false_alarm_rate": None,
            "same_source_auc": None,
            "cross_source_auc": None,
            "actor_holdout_auc": None,
            "security_model": DEFAULT_SECURITY_MODEL,
            "stego_protocol": COST_STEGO_V2_PROTOCOL,
            "selection_entropy": None,
            "effective_action_budget": 0,
            "qualified_cover": False,
            "baseline_type": args.baseline_type,
            "pixel_delta": pixel_delta,
            "decode_domain": "image_y",
            "reference_image_tensor": clean_image.detach().cpu(),
            "clean_image_tensor": clean_image.detach().cpu(),
            "wrong_key_assignments_image": [],
            "detector_training_enabled": False,
            "effective_detector_weight": 0.0,
            "wrong_key_training_enabled": False,
            "trained_wrong_key_groups": 0,
            "wrong_key_sample_failures": 0,
        }
    wrong_key_assignments, wrong_key_sample_failures = _collect_wrong_key_assignment_groups(
        requested_samples=int(args.wrong_key_samples),
        builder=lambda wrong_index: build_generic_cost_code_assignments(
            raw_bits=raw_bits.detach().cpu(),
            candidate_ids=candidate_ids.detach().cpu(),
            selection_logits=selection_logits.detach().cpu(),
            selection_probs=selection_probs.detach().cpu(),
            plus_costs=plus_costs.detach().cpu(),
            minus_costs=minus_costs.detach().cpu(),
            key=f"{args.key}::wrong::{wrong_index}",
            action_type="pixel_y",
            candidate_count=24,
            fallback_candidate_count=16,
            beam_width=int(args.beam_width),
            max_modification_count=max_modification_count,
        )[0],
    )
    training_wrong_key_assignments = wrong_key_assignments[: min(4, len(wrong_key_assignments))]
    pixel_wm_image = _apply_pixel_cost_code_assignments(clean_image, assignments=assignments, delta=pixel_delta)
    renderer = deepcopy(clean_renderer)
    if args.baseline_type == "render_refit":
        renderer = _refit_renderer_to_image(
            clean_renderer=clean_renderer,
            target_image=pixel_wm_image.to(next(clean_renderer.parameters()).device),
            grid=grid,
            steps=max(300, int(args.tune_steps) * 20),
            lr=max(float(args.tune_lr), 0.005),
            clean_image=clean_image,
            assignments=assignments,
            delta=pixel_delta,
            selected_pixel_weight=8.0 if int(raw_bits.numel()) <= 8 else 12.0,
            code_weight=0.75 if int(raw_bits.numel()) <= 8 else 1.25,
            wrong_key_weight=0.08 if int(raw_bits.numel()) <= 8 else 0.12,
            wrong_key_assignments=training_wrong_key_assignments,
        )
    return renderer, {
        "capacity_failure": False,
        "coding_mode": "cost_code_v1",
        "assignments": assignments,
        "wrong_key_assignments": wrong_key_assignments,
        "wrong_key_assignments_image": wrong_key_assignments,
        "spread_factor": None,
        "requested_spread_factor": None,
        "alpha_delta": None,
        "log_delta": None,
        "wet_ratio": 0.0,
        "requested_wet_ratio": 0.0,
        "relaxed_wet_count": 0,
        "relaxed_wet_indices": [],
        "dry_count_before_relaxation": None,
        "modified_gaussian_count": int(assignment_meta["total_selected_actions"]),
        "total_embed_cost": float(sum(sum(float(cost) * float(mask) for cost, mask in zip(assignment["candidate_costs"], assignment["selected_mask"])) for assignment in assignments)),
        "compensation_indices": [],
        "detector_feature_names": detector_feature_names(),
        "pooled_detectability_auc": None,
        "false_alarm_rate": None,
        "same_source_auc": None,
        "cross_source_auc": None,
        "actor_holdout_auc": None,
        "security_model": DEFAULT_SECURITY_MODEL,
        "stego_protocol": COST_STEGO_V2_PROTOCOL,
        "selection_entropy": float(assignment_meta["selection_entropy"]),
        "effective_action_budget": int(assignment_meta["effective_action_budget"]),
        "qualified_cover": True,
        "baseline_type": args.baseline_type,
        "pixel_delta": pixel_delta,
        "decode_domain": "image_y",
        "reference_image_tensor": pixel_wm_image.detach().cpu(),
        "clean_image_tensor": clean_image.detach().cpu(),
        "detector_training_enabled": False,
        "effective_detector_weight": 0.0,
        "wrong_key_training_enabled": True,
        "trained_wrong_key_groups": len(wrong_key_assignments),
        "wrong_key_sample_failures": wrong_key_sample_failures,
    }


def run_joint_parity_embedding(
    *,
    clean_renderer: Gaussian2DRenderer,
    clean_params: dict[str, torch.Tensor],
    clean_sensitivity: dict[str, torch.Tensor],
    raw_bits: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[Gaussian2DRenderer, dict[str, object]]:
    """
    S1-A: Embed bits using JointParityStegoCoder (multi-channel GF(2) parity).

    Selects the top-k Gaussians by log_anisotropy sensitivity, then calls
    JointParityStegoCoder.encode() to write bits jointly across scales, thetas,
    and opacities. The modified tensors are written back via set_physical_parameters().
    """
    n_bits = int(raw_bits.numel())
    bits_list: list[int] = [int(b) for b in raw_bits.detach().cpu().tolist()]

    # This prototype currently operates on the full Gaussian set using its own
    # keyed carrier assignment logic. It is intentionally not treated as a fair
    # headline baseline against cost_stego_v2 until it shares the same explicit
    # covertness budget and training constraints.
    scales_cpu = clean_params["scales"].detach().cpu()
    thetas_cpu = clean_params["thetas"].detach().cpu()
    opacities_cpu = clean_params["opacities"].detach().cpu()
    offsets_cpu = clean_params["offsets"].detach().cpu()
    colors_cpu = clean_params["colors"].detach().cpu()

    coder = JointParityStegoCoder(
        log_delta=float(getattr(args, "target_magnitude", 0.35)),
        alpha_delta=0.05,
        theta_delta=0.10,
    )
    new_scales, new_thetas, new_opacities = coder.encode(
        bits_list,
        scales_cpu,
        thetas_cpu,
        opacities_cpu,
        key=str(args.key),
    )

    # Count actually modified Gaussians (scales, thetas, or opacities changed).
    log_ratio_before = torch.log(scales_cpu[:, 0].clamp_min(1e-8) / scales_cpu[:, 1].clamp_min(1e-8))
    log_ratio_after = torch.log(new_scales[:, 0].clamp_min(1e-8) / new_scales[:, 1].clamp_min(1e-8))
    log_changed = (log_ratio_after - log_ratio_before).abs() > 1e-6
    alpha_changed = (new_opacities[:, 0] - opacities_cpu[:, 0]).abs() > 1e-6
    theta_changed = (new_thetas[:, 0] - thetas_cpu[:, 0]).abs() > 1e-6
    n_modified = int((log_changed | alpha_changed | theta_changed).sum().item())

    watermarked_renderer = deepcopy(clean_renderer)
    device = next(watermarked_renderer.parameters()).device
    watermarked_renderer.set_physical_parameters(
        offsets=offsets_cpu.to(device),
        scales=new_scales.to(device),
        thetas=new_thetas.to(device),
        colors=colors_cpu.to(device),
        opacities=new_opacities.to(device),
    )

    return watermarked_renderer, {
        "capacity_failure": False,
        "coding_mode": "joint_parity_v1",
        "assignments": [],
        "wrong_key_assignments": [],
        "spread_factor": None,
        "requested_spread_factor": None,
        "alpha_delta": 0.05,
        "log_delta": float(getattr(args, "target_magnitude", 0.35)),
        "theta_delta": 0.10,
        "wet_ratio": float(n_modified) / max(1, scales_cpu.shape[0]),
        "requested_wet_ratio": None,
        "relaxed_wet_count": 0,
        "relaxed_wet_indices": [],
        "dry_count_before_relaxation": None,
        "modified_gaussian_count": n_modified,
        "total_embed_cost": None,
        "compensation_indices": [],
        "detector_feature_names": detector_feature_names(),
        "pooled_detectability_auc": None,
        "false_alarm_rate": None,
        "same_source_auc": None,
        "cross_source_auc": None,
        "actor_holdout_auc": None,
        "security_model": DEFAULT_SECURITY_MODEL,
        "stego_protocol": JOINT_PARITY_PROTOCOL,
        "selection_entropy": None,
        "effective_action_budget": n_bits * 3,
        "qualified_cover": False,
        "baseline_type": args.baseline_type,
        "pixel_delta": None,
        "decode_domain": "parameter",
        "clean_image_tensor": None,
        "reference_image_tensor": None,
        "detector_training_enabled": False,
        "effective_detector_weight": 0.0,
        "wrong_key_training_enabled": False,
        "trained_wrong_key_groups": 0,
        "wrong_key_sample_failures": 0,
    }


def run_theta_distribution_embedding(
    *,
    clean_renderer: Gaussian2DRenderer,
    clean_params: dict[str, torch.Tensor],
    raw_bits: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[Gaussian2DRenderer, dict[str, object]]:
    bits_list: list[int] = [int(bit) for bit in raw_bits.detach().cpu().tolist()]
    offsets_cpu = clean_params["offsets"].detach().cpu()
    scales_cpu = clean_params["scales"].detach().cpu()
    thetas_cpu = clean_params["thetas"].detach().cpu()
    opacities_cpu = clean_params["opacities"].detach().cpu()
    colors_cpu = clean_params["colors"].detach().cpu()
    coder = ThetaStegoCoder(
        n_groups=max(1, len(bits_list)),
        target_spread=float(getattr(args, "theta_target_spread", math.pi / 4.0)),
    )
    current_thetas = thetas_cpu.clone()
    passes_used = 0
    max_passes = max(1, int(getattr(args, "prototype_max_passes", 4)))
    for pass_index in range(max_passes):
        passes_used = pass_index + 1
        current_thetas = coder.encode(
            bits_list,
            offsets_cpu,
            current_thetas,
            key=str(args.key),
            delta=float(getattr(args, "theta_delta", 0.08)),
        )
        if coder.decode(offsets_cpu, current_thetas, key=str(args.key))[: len(bits_list)] == bits_list:
            break
    changed_mask = (current_thetas[:, 0] - thetas_cpu[:, 0]).abs() > 1e-6
    n_modified = int(changed_mask.sum().item())
    watermarked_renderer = deepcopy(clean_renderer)
    device = next(watermarked_renderer.parameters()).device
    watermarked_renderer.set_physical_parameters(
        offsets=offsets_cpu.to(device),
        scales=scales_cpu.to(device),
        thetas=current_thetas.to(device),
        colors=colors_cpu.to(device),
        opacities=opacities_cpu.to(device),
    )
    return watermarked_renderer, {
        "capacity_failure": False,
        "coding_mode": "theta_distribution_v1",
        "assignments": [],
        "wrong_key_assignments": [],
        "spread_factor": None,
        "requested_spread_factor": None,
        "alpha_delta": None,
        "log_delta": None,
        "theta_delta": float(getattr(args, "theta_delta", 0.08)),
        "wet_ratio": float(n_modified) / max(1, scales_cpu.shape[0]),
        "requested_wet_ratio": None,
        "relaxed_wet_count": 0,
        "relaxed_wet_indices": [],
        "dry_count_before_relaxation": None,
        "modified_gaussian_count": n_modified,
        "total_embed_cost": None,
        "compensation_indices": [],
        "detector_feature_names": detector_feature_names(),
        "pooled_detectability_auc": None,
        "false_alarm_rate": None,
        "same_source_auc": None,
        "cross_source_auc": None,
        "actor_holdout_auc": None,
        "security_model": DEFAULT_SECURITY_MODEL,
        "stego_protocol": THETA_DISTRIBUTION_PROTOCOL,
        "selection_entropy": None,
        "effective_action_budget": len(bits_list),
        "qualified_cover": False,
        "baseline_type": args.baseline_type,
        "pixel_delta": None,
        "decode_domain": "parameter",
        "clean_image_tensor": None,
        "reference_image_tensor": None,
        "detector_training_enabled": False,
        "effective_detector_weight": 0.0,
        "wrong_key_training_enabled": False,
        "trained_wrong_key_groups": 0,
        "wrong_key_sample_failures": 0,
        "prototype_metadata": {
            "group_count": len(bits_list),
            "target_spread": float(getattr(args, "theta_target_spread", math.pi / 4.0)),
            "max_passes": max_passes,
            "passes_used": passes_used,
        },
    }


def run_mu_jitter_embedding(
    *,
    clean_renderer: Gaussian2DRenderer,
    clean_params: dict[str, torch.Tensor],
    raw_bits: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[Gaussian2DRenderer, dict[str, object]]:
    bits_list: list[int] = [int(bit) for bit in raw_bits.detach().cpu().tolist()]
    offsets_cpu = clean_params["offsets"].detach().cpu()
    scales_cpu = clean_params["scales"].detach().cpu()
    thetas_cpu = clean_params["thetas"].detach().cpu()
    opacities_cpu = clean_params["opacities"].detach().cpu()
    colors_cpu = clean_params["colors"].detach().cpu()
    coder = MuStegoCoder(base_epsilon=float(getattr(args, "mu_epsilon", 0.02)))
    new_offsets = coder.encode(bits_list, offsets_cpu, key=str(args.key))
    changed_mask = (new_offsets - offsets_cpu).abs().sum(dim=1) > 1e-8
    n_modified = int(changed_mask.sum().item())
    watermarked_renderer = deepcopy(clean_renderer)
    device = next(watermarked_renderer.parameters()).device
    watermarked_renderer.set_physical_parameters(
        offsets=new_offsets.to(device),
        scales=scales_cpu.to(device),
        thetas=thetas_cpu.to(device),
        colors=colors_cpu.to(device),
        opacities=opacities_cpu.to(device),
    )
    return watermarked_renderer, {
        "capacity_failure": False,
        "coding_mode": "mu_jitter_v1",
        "assignments": [],
        "wrong_key_assignments": [],
        "spread_factor": None,
        "requested_spread_factor": None,
        "alpha_delta": None,
        "log_delta": None,
        "theta_delta": None,
        "wet_ratio": float(n_modified) / max(1, offsets_cpu.shape[0]),
        "requested_wet_ratio": None,
        "relaxed_wet_count": 0,
        "relaxed_wet_indices": [],
        "dry_count_before_relaxation": None,
        "modified_gaussian_count": n_modified,
        "total_embed_cost": None,
        "compensation_indices": [],
        "detector_feature_names": detector_feature_names(),
        "pooled_detectability_auc": None,
        "false_alarm_rate": None,
        "same_source_auc": None,
        "cross_source_auc": None,
        "actor_holdout_auc": None,
        "security_model": DEFAULT_SECURITY_MODEL,
        "stego_protocol": MU_JITTER_PROTOCOL,
        "selection_entropy": None,
        "effective_action_budget": len(bits_list),
        "qualified_cover": False,
        "baseline_type": args.baseline_type,
        "pixel_delta": None,
        "decode_domain": "parameter",
        "clean_image_tensor": None,
        "reference_image_tensor": None,
        "detector_training_enabled": False,
        "effective_detector_weight": 0.0,
        "wrong_key_training_enabled": False,
        "trained_wrong_key_groups": 0,
        "wrong_key_sample_failures": 0,
        "prototype_metadata": {
            "epsilon": float(getattr(args, "mu_epsilon", 0.02)),
            "carrier_count": len(bits_list),
        },
    }


def run_scale_tier_embedding(
    *,
    clean_renderer: Gaussian2DRenderer,
    clean_params: dict[str, torch.Tensor],
    raw_bits: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[Gaussian2DRenderer, dict[str, object]]:
    bits_list: list[int] = [int(bit) for bit in raw_bits.detach().cpu().tolist()]
    offsets_cpu = clean_params["offsets"].detach().cpu()
    scales_cpu = clean_params["scales"].detach().cpu()
    thetas_cpu = clean_params["thetas"].detach().cpu()
    opacities_cpu = clean_params["opacities"].detach().cpu()
    colors_cpu = clean_params["colors"].detach().cpu()
    coder = ScaleTierStegoCoder(
        n_tiers=max(1, len(bits_list)),
        delta_scale=float(getattr(args, "scale_tier_delta_scale", 0.06)),
    )
    new_scales = coder.encode(bits_list, scales_cpu, thetas_cpu, key=str(args.key))
    changed_mask = (new_scales - scales_cpu).abs().sum(dim=1) > 1e-8
    n_modified = int(changed_mask.sum().item())
    watermarked_renderer = deepcopy(clean_renderer)
    device = next(watermarked_renderer.parameters()).device
    watermarked_renderer.set_physical_parameters(
        offsets=offsets_cpu.to(device),
        scales=new_scales.to(device),
        thetas=thetas_cpu.to(device),
        colors=colors_cpu.to(device),
        opacities=opacities_cpu.to(device),
    )
    return watermarked_renderer, {
        "capacity_failure": False,
        "coding_mode": "scale_tier_v1",
        "assignments": [],
        "wrong_key_assignments": [],
        "spread_factor": None,
        "requested_spread_factor": None,
        "alpha_delta": None,
        "log_delta": None,
        "theta_delta": None,
        "wet_ratio": float(n_modified) / max(1, scales_cpu.shape[0]),
        "requested_wet_ratio": None,
        "relaxed_wet_count": 0,
        "relaxed_wet_indices": [],
        "dry_count_before_relaxation": None,
        "modified_gaussian_count": n_modified,
        "total_embed_cost": None,
        "compensation_indices": [],
        "detector_feature_names": detector_feature_names(),
        "pooled_detectability_auc": None,
        "false_alarm_rate": None,
        "same_source_auc": None,
        "cross_source_auc": None,
        "actor_holdout_auc": None,
        "security_model": DEFAULT_SECURITY_MODEL,
        "stego_protocol": SCALE_TIER_PROTOCOL,
        "selection_entropy": None,
        "effective_action_budget": len(bits_list),
        "qualified_cover": False,
        "baseline_type": args.baseline_type,
        "pixel_delta": None,
        "decode_domain": "parameter",
        "clean_image_tensor": None,
        "reference_image_tensor": None,
        "detector_training_enabled": False,
        "effective_detector_weight": 0.0,
        "wrong_key_training_enabled": False,
        "trained_wrong_key_groups": 0,
        "wrong_key_sample_failures": 0,
        "prototype_metadata": {
            "tier_count": len(bits_list),
            "delta_scale": float(getattr(args, "scale_tier_delta_scale", 0.06)),
        },
    }


def _legacy_report_feature_vector(
    params: dict[str, torch.Tensor],
    *,
    importance: torch.Tensor,
) -> tuple[torch.Tensor, list[str]]:
    return build_detector_feature_vector(params, importance=importance), detector_feature_names()


def main() -> None:
    args = parse_args()
    set_seed(key_to_seed(f"{args.key}::wm::{args.message_seed}::{args.fit_seed}"), deterministic=True)
    source_run = load_exported_run(args.source_run)
    device = get_best_device()
    target = load_target_tensor(source_run, device=device)
    grid = make_grid_from_run(source_run, device=device)

    clean_renderer = build_renderer_from_export(source_run, device=device)
    clean_eval = evaluate_renderer(clean_renderer, target, grid, include_sensitivity=True)
    clean_params_cpu = clean_eval["params"]
    clean_params_device = {
        key: value.detach()
        for key, value in clean_renderer.decode_parameters().items()
    }
    clean_reference = build_clean_distribution_reference(clean_params_cpu)
    clean_importance_cpu = source_run.importance
    clean_sensitivity_cpu = source_run.sensitivity if source_run.sensitivity is not None else clean_eval.get("sensitivity")
    if clean_sensitivity_cpu is None:
        raise ValueError("watermark experiments require sensitivity export or evaluable sensitivity measurements")

    raw_message_bits = build_bits(args.payload_bits, message=args.message, seed=args.message_seed)
    if args.protocol in {COST_STEGO_V2_PROTOCOL, *PROTOTYPE_PROTOCOLS}:
        effective_ecc_mode = "none"
    elif args.protocol == DEFAULT_STEGO_PROTOCOL:
        effective_ecc_mode = "hamming74"
    else:
        effective_ecc_mode = args.ecc
    coded_bits, ecc_meta = build_payload(raw_message_bits, effective_ecc_mode)
    carrier_meta: dict[str, object] = {
        "candidate_count": 0,
        "carrier_score_mode": args.carrier_score,
    }
    carrier_indices = torch.empty(0, dtype=torch.long)
    stego_meta: dict[str, object] = {
        "capacity_failure": False,
        "assignments": [],
        "wrong_key_assignments": [],
        "spread_factor": None,
        "requested_spread_factor": None,
        "alpha_delta": None,
        "log_delta": float(args.target_magnitude),
        "wet_ratio": None,
        "requested_wet_ratio": None,
        "modified_gaussian_count": 0,
        "total_embed_cost": None,
        "compensation_indices": [],
        "detector_feature_names": detector_feature_names(),
        "pooled_detectability_auc": None,
        "false_alarm_rate": None,
        "security_model": DEFAULT_SECURITY_MODEL,
        "stego_protocol": "legacy_wm",
        "relaxed_wet_indices": [],
        "relaxed_wet_count": 0,
        "dry_count_before_relaxation": None,
        "coding_mode": args.coding_mode,
        "same_source_auc": None,
        "cross_source_auc": None,
        "actor_holdout_auc": None,
        "effective_action_budget": None,
        "qualified_cover": None,
        "selection_entropy": None,
        "baseline_type": args.baseline_type,
        "decode_domain": "parameter",
        "pixel_delta": None,
        "clean_image_tensor": None,
        "reference_image_tensor": None,
    }
    sensitivity_weights = torch.ones(1, dtype=torch.float32)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else default_watermark_output_dir(source_run.run_dir.name, args)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.protocol == COST_STEGO_V2_PROTOCOL:
        if _uses_native_parameter_embedding(args.baseline_type):
            watermarked_renderer, stego_meta = run_cost_stego_v2_embedding(
                clean_renderer=clean_renderer,
                clean_params=clean_params_device,
                clean_importance=clean_importance_cpu,
                clean_sensitivity=clean_sensitivity_cpu,
                clean_clamped_psnr=float(clean_eval["clamped_psnr"]),
                target=target,
                grid=grid,
                raw_bits=raw_message_bits.to(device),
                args=args,
            )
        else:
            watermarked_renderer, stego_meta = run_pixel_baseline_embedding(
                clean_renderer=clean_renderer,
                clean_image=clean_eval["reconstruction"].clamp(0.0, 1.0),
                grid=grid,
                raw_bits=raw_message_bits.to(device),
                args=args,
            )
        effective_steps = args.tune_steps
    elif args.protocol == THETA_DISTRIBUTION_PROTOCOL:
        watermarked_renderer, stego_meta = run_theta_distribution_embedding(
            clean_renderer=clean_renderer,
            clean_params=clean_params_device,
            raw_bits=raw_message_bits,
            args=args,
        )
        effective_steps = 0
    elif args.protocol == MU_JITTER_PROTOCOL:
        watermarked_renderer, stego_meta = run_mu_jitter_embedding(
            clean_renderer=clean_renderer,
            clean_params=clean_params_device,
            raw_bits=raw_message_bits,
            args=args,
        )
        effective_steps = 0
    elif args.protocol == SCALE_TIER_PROTOCOL:
        watermarked_renderer, stego_meta = run_scale_tier_embedding(
            clean_renderer=clean_renderer,
            clean_params=clean_params_device,
            raw_bits=raw_message_bits,
            args=args,
        )
        effective_steps = 0
    elif args.protocol == JOINT_PARITY_PROTOCOL:
        watermarked_renderer, stego_meta = run_joint_parity_embedding(
            clean_renderer=clean_renderer,
            clean_params=clean_params_device,
            clean_sensitivity=clean_sensitivity_cpu,
            raw_bits=raw_message_bits,
            args=args,
        )
        effective_steps = 0
    elif args.protocol == DEFAULT_STEGO_PROTOCOL:
        watermarked_renderer, stego_meta = run_cost_stego_embedding(
            clean_renderer=clean_renderer,
            clean_params=clean_params_device,
            clean_importance=clean_importance_cpu,
            clean_sensitivity=clean_sensitivity_cpu,
            target=target,
            grid=grid,
            coded_bits=coded_bits.to(device),
            args=args,
        )
        effective_steps = args.tune_steps
    else:
        carrier_indices, carrier_meta = select_carriers(
            source_run=source_run,
            coded_bits=coded_bits,
            mode=args.mode,
            carrier_pool=args.carrier_pool,
            carrier_score_mode=args.carrier_score,
            target_magnitude=args.target_magnitude,
            candidate_factor=args.candidate_factor,
            key=args.key,
        )
        if carrier_indices.numel() != coded_bits.numel():
            raise RuntimeError("carrier selection failed to allocate one carrier per coded bit")
        carrier_indices_device = carrier_indices.to(device)
        coded_bits_device = coded_bits.to(device)
        sensitivity_field = carrier_sensitivity_field(args.mode)
        sensitivity_weights = clean_sensitivity_cpu[sensitivity_field][carrier_indices]
        sensitivity_weights = sensitivity_weights / sensitivity_weights.mean().clamp_min(1e-6)
        clean_scales = clean_params_cpu["scales"].detach().cpu()

        if args.stage == "postfit":
            watermarked_renderer = run_postfit_embedding(
                clean_renderer=clean_renderer,
                target=target,
                grid=grid,
                carrier_indices=carrier_indices_device,
                coded_bits=coded_bits_device,
                mode=args.mode,
                msg_strength=float(args.msg_strength),
                dist_weight=float(args.dist_weight),
                anchor_weight=float(args.anchor_weight),
                target_magnitude=args.target_magnitude,
                tune_steps=args.tune_steps,
                tune_lr=args.tune_lr,
                clean_reference=clean_reference,
                clean_scales=clean_scales,
                sensitivity_weights=sensitivity_weights.to(device),
            )
            effective_steps = args.tune_steps
        else:
            watermarked_renderer, effective_steps = run_joint_embedding(
                source_run=source_run,
                target=target,
                grid=grid,
                device=device,
                carrier_indices=carrier_indices_device,
                coded_bits=coded_bits_device,
                mode=args.mode,
                msg_strength=float(args.msg_strength),
                dist_weight=float(args.dist_weight),
                anchor_weight=float(args.anchor_weight),
                target_magnitude=args.target_magnitude,
                clean_reference=clean_reference,
                clean_scales=clean_scales,
                sensitivity_weights=sensitivity_weights.to(device),
                joint_steps=args.joint_steps,
            )
        stego_meta = {
            **stego_meta,
            "security_model": DEFAULT_SECURITY_MODEL,
            "stego_protocol": "legacy_wm",
        }

    watermarked_eval = evaluate_renderer(watermarked_renderer, target, grid, include_sensitivity=True)
    if args.protocol == COST_STEGO_V2_PROTOCOL and args.baseline_type == "pixel" and stego_meta.get("reference_image_tensor") is not None:
        watermarked_eval = _apply_reconstruction_override(
            watermarked_eval,
            target=target,
            reconstruction=stego_meta["reference_image_tensor"],
        )
    embedded_coded_bits = assignment_bit_tensor(list(stego_meta["assignments"]), field="embed_bit") if args.protocol == DEFAULT_STEGO_PROTOCOL and stego_meta["assignments"] else coded_bits.clone()
    if args.protocol == COST_STEGO_V2_PROTOCOL:
        if bool(stego_meta["capacity_failure"]):
            decoded_coded_bits = torch.zeros_like(raw_message_bits)
            decoded_raw_bits = torch.zeros_like(raw_message_bits)
            decode_meta = {"corrected_blocks": 0}
            decode_scores = torch.empty(0, dtype=torch.float32)
            wrong_key_accuracy = None
            v2_block_probs: list[list[float]] = []
            v2_action_scores: list[list[float]] = []
            wrong_key_block_probabilities: list[list[list[float]]] = []
        else:
            if _uses_native_parameter_embedding(args.baseline_type):
                decoded_raw_bits, v2_block_probs, v2_action_scores = decode_cost_code(
                    watermarked_eval["params"],
                    clean_params=clean_params_cpu,
                    assignments=list(stego_meta["assignments"]),
                    raw_length=args.payload_bits,
                    log_delta=float(stego_meta["log_delta"]),
                    alpha_delta=float(stego_meta["alpha_delta"]),
                    beta=4.0,
                )
            else:
                decode_image = (
                    stego_meta["reference_image_tensor"]
                    if args.baseline_type == "pixel"
                    else watermarked_eval["reconstruction"]
                )
                decoded_raw_bits, v2_block_probs, v2_action_scores = decode_image_cost_code(
                    decode_image,
                    clean_image=stego_meta["clean_image_tensor"],
                    assignments=list(stego_meta["assignments"]),
                    raw_length=args.payload_bits,
                    delta=float(stego_meta["pixel_delta"]),
                    beta=4.0,
                )
            decoded_coded_bits = decoded_raw_bits.clone()
            decode_meta = {"corrected_blocks": 0}
            decode_scores = torch.tensor(
                [value for row in v2_block_probs for value in row],
                dtype=torch.float32,
            )
            wrong_key_block_probabilities = []
            if stego_meta["wrong_key_assignments"]:
                wrong_scores = []
                for wrong_assignment_group in list(stego_meta["wrong_key_assignments"]):
                    if _uses_native_parameter_embedding(args.baseline_type):
                        wrong_raw_bits, wrong_block_probs, _ = decode_cost_code(
                            watermarked_eval["params"],
                            clean_params=clean_params_cpu,
                            assignments=list(wrong_assignment_group),
                            raw_length=args.payload_bits,
                            log_delta=float(stego_meta["log_delta"]),
                            alpha_delta=float(stego_meta["alpha_delta"]),
                            beta=4.0,
                        )
                    else:
                        decode_image = (
                            stego_meta["reference_image_tensor"]
                            if args.baseline_type == "pixel"
                            else watermarked_eval["reconstruction"]
                        )
                        wrong_raw_bits, wrong_block_probs, _ = decode_image_cost_code(
                            decode_image,
                            clean_image=stego_meta["clean_image_tensor"],
                            assignments=list(wrong_assignment_group),
                            raw_length=args.payload_bits,
                            delta=float(stego_meta["pixel_delta"]),
                            beta=4.0,
                        )
                    wrong_key_block_probabilities.append(wrong_block_probs)
                    wrong_scores.append(1.0 - compute_ber(raw_message_bits, wrong_raw_bits))
                wrong_key_accuracy = float(sum(wrong_scores) / len(wrong_scores)) if wrong_scores else None
            else:
                wrong_key_accuracy = None
        if _uses_native_parameter_embedding(args.baseline_type) and stego_meta["assignments"]:
            carrier_indices = torch.unique(torch.cat(_v2_selected_indices(list(stego_meta["assignments"])), dim=0), sorted=True)
        else:
            carrier_indices = torch.empty(0, dtype=torch.long)
    elif args.protocol == DEFAULT_STEGO_PROTOCOL:
        if bool(stego_meta["capacity_failure"]):
            decoded_coded_bits = torch.zeros_like(coded_bits)
            decoded_raw_bits = torch.zeros_like(raw_message_bits)
            decode_meta = {"corrected_blocks": 0}
            decode_scores = torch.empty(0, dtype=torch.float32)
            wrong_key_accuracy = None
            wrong_key_block_probabilities = []
        else:
            decoded_coded_bits, decoded_raw_bits, decode_meta, decode_scores = _decode_cost_stego_payload(
                watermarked_eval["params"],
                clean_params=clean_params_cpu,
                assignments=list(stego_meta["assignments"]),
                payload_bits=args.payload_bits,
                ecc_mode=effective_ecc_mode,
                log_delta=float(stego_meta["log_delta"]),
                alpha_delta=float(stego_meta["alpha_delta"]),
                alpha_weight=float(args.alpha_weight),
            )
            if stego_meta["wrong_key_assignments"]:
                _, wrong_key_raw_bits, _, _ = _decode_cost_stego_payload(
                    watermarked_eval["params"],
                    clean_params=clean_params_cpu,
                    assignments=list(stego_meta["wrong_key_assignments"]),
                    payload_bits=args.payload_bits,
                    ecc_mode=effective_ecc_mode,
                    log_delta=float(stego_meta["log_delta"]),
                    alpha_delta=float(stego_meta["alpha_delta"]),
                    alpha_weight=float(args.alpha_weight),
                )
                wrong_key_accuracy = 1.0 - compute_ber(raw_message_bits, wrong_key_raw_bits)
            else:
                wrong_key_accuracy = None
            wrong_key_block_probabilities = []
        active_log_indices, active_alpha_indices = active_channel_indices(list(stego_meta["assignments"]))
        carrier_indices = torch.unique(torch.cat((active_log_indices, active_alpha_indices), dim=0), sorted=True)
    elif args.protocol == THETA_DISTRIBUTION_PROTOCOL:
        coder = ThetaStegoCoder(
            n_groups=int(stego_meta.get("prototype_metadata", {}).get("group_count", args.payload_bits)),
            target_spread=float(stego_meta.get("prototype_metadata", {}).get("target_spread", getattr(args, "theta_target_spread", math.pi / 4.0))),
        )
        decoded_raw_bits = torch.tensor(
            coder.decode(
                clean_params_cpu["offsets"],
                watermarked_eval["params"]["thetas"].detach().cpu(),
                key=str(args.key),
            )[: args.payload_bits],
            dtype=torch.int64,
        )
        decoded_coded_bits = decoded_raw_bits.clone()
        decode_meta = {"corrected_blocks": 0}
        decode_scores = torch.empty(0, dtype=torch.float32)
        wrong_key_accuracy = None
        wrong_key_block_probabilities = []
    elif args.protocol == MU_JITTER_PROTOCOL:
        coder = MuStegoCoder(base_epsilon=float(stego_meta.get("prototype_metadata", {}).get("epsilon", getattr(args, "mu_epsilon", 0.02))))
        decoded_raw_bits = torch.tensor(
            coder.decode(
                clean_params_cpu["offsets"],
                watermarked_eval["params"]["offsets"].detach().cpu(),
                key=str(args.key),
                n_bits=args.payload_bits,
            ),
            dtype=torch.int64,
        )
        decoded_coded_bits = decoded_raw_bits.clone()
        decode_meta = {"corrected_blocks": 0}
        decode_scores = torch.empty(0, dtype=torch.float32)
        wrong_key_accuracy = None
        wrong_key_block_probabilities = []
    elif args.protocol == SCALE_TIER_PROTOCOL:
        coder = ScaleTierStegoCoder(
            n_tiers=int(stego_meta.get("prototype_metadata", {}).get("tier_count", args.payload_bits)),
            delta_scale=float(stego_meta.get("prototype_metadata", {}).get("delta_scale", getattr(args, "scale_tier_delta_scale", 0.06))),
        )
        decoded_raw_bits = torch.tensor(
            coder.decode(
                watermarked_eval["params"]["scales"].detach().cpu(),
                watermarked_eval["params"]["thetas"].detach().cpu(),
                key=str(args.key),
                clean_scales=clean_params_cpu["scales"],
                clean_thetas=clean_params_cpu["thetas"],
            )[: args.payload_bits],
            dtype=torch.int64,
        )
        decoded_coded_bits = decoded_raw_bits.clone()
        decode_meta = {"corrected_blocks": 0}
        decode_scores = torch.empty(0, dtype=torch.float32)
        wrong_key_accuracy = None
        wrong_key_block_probabilities = []
    elif args.protocol == JOINT_PARITY_PROTOCOL:
        coder = JointParityStegoCoder(
            log_delta=float(stego_meta["log_delta"]),
            alpha_delta=float(stego_meta["alpha_delta"]),
            theta_delta=float(stego_meta.get("theta_delta", 0.10)),
        )
        decoded_raw_bits = torch.tensor(
            coder.decode(
                watermarked_eval["params"]["scales"].detach().cpu(),
                watermarked_eval["params"]["thetas"].detach().cpu(),
                watermarked_eval["params"]["opacities"].detach().cpu(),
                clean_params_cpu["scales"],
                clean_params_cpu["thetas"],
                clean_params_cpu["opacities"],
                key=str(args.key),
                n_bits=args.payload_bits,
            ),
            dtype=torch.int64,
        )
        decoded_coded_bits = decoded_raw_bits.clone()
        decode_meta = {"corrected_blocks": 0}
        decode_scores = torch.empty(0, dtype=torch.float32)
        wrong_key_accuracy = None
        wrong_key_block_probabilities = []
    else:
        decoded_coded_bits = decode_bits(
            watermarked_eval["params"],
            carrier_indices,
            mode=args.mode,
            target_magnitude=args.target_magnitude,
        )
        decoded_raw_bits, decode_meta = decode_payload(decoded_coded_bits, raw_length=args.payload_bits, ecc_mode=effective_ecc_mode)
        decode_scores = torch.empty(0, dtype=torch.float32)
        wrong_key_accuracy = None
        wrong_key_block_probabilities = []
    raw_ber = compute_ber(raw_message_bits, decoded_raw_bits)
    coded_reference = raw_message_bits if args.protocol == COST_STEGO_V2_PROTOCOL else coded_bits
    coded_ber = compute_ber(coded_reference, decoded_coded_bits)
    parameter_metrics_applicable = _parameter_metrics_applicable(args.baseline_type)
    wrong_key_status = _wrong_key_status(
        protocol=args.protocol,
        capacity_failure=bool(stego_meta["capacity_failure"]),
        wrong_key_accuracy=wrong_key_accuracy,
        wrong_key_groups_present=bool(stego_meta["wrong_key_assignments"]),
        wrong_key_sample_failures=int(stego_meta.get("wrong_key_sample_failures", 0)),
    )
    claim_eligible = _claim_eligible(
        baseline_type=str(stego_meta["baseline_type"]),
        native_ablation=str(stego_meta.get("native_ablation", args.native_ablation)),
        qualified_cover=stego_meta.get("qualified_cover"),
        detector_training_enabled=bool(stego_meta.get("detector_training_enabled", False)),
        wrong_key_training_enabled=bool(stego_meta.get("wrong_key_training_enabled", False)),
        capacity_failure=bool(stego_meta["capacity_failure"]),
        claim_track=str(getattr(args, "claim_track", "full")),
    )
    main_claim_eligible = _main_claim_eligible(
        baseline_type=str(stego_meta["baseline_type"]),
        native_ablation=str(stego_meta.get("native_ablation", args.native_ablation)),
        claim_eligible=claim_eligible,
        label=args.label,
        claim_track=str(getattr(args, "claim_track", "full")),
    )
    baseline_role = _baseline_role(
        baseline_type=str(stego_meta["baseline_type"]),
        native_ablation=str(stego_meta.get("native_ablation", args.native_ablation)),
        claim_eligible=claim_eligible,
        label=args.label,
        claim_track=str(getattr(args, "claim_track", "full")),
    )
    claim_scope = _steganography_claim_scope(
        protocol=str(stego_meta["stego_protocol"]),
        baseline_type=str(stego_meta["baseline_type"]),
        capacity_failure=bool(stego_meta["capacity_failure"]),
        qualified_cover=stego_meta.get("qualified_cover"),
    )
    claim_note = _steganography_claim_note(
        protocol=str(stego_meta["stego_protocol"]),
        baseline_type=str(stego_meta["baseline_type"]),
        capacity_failure=bool(stego_meta["capacity_failure"]),
        qualified_cover=stego_meta.get("qualified_cover"),
    )
    protocol_note = _protocol_capability_note(
        protocol=str(stego_meta["stego_protocol"]),
        baseline_type=str(stego_meta["baseline_type"]),
        detector_training_enabled=bool(stego_meta.get("detector_training_enabled", False)),
        wrong_key_training_enabled=bool(stego_meta.get("wrong_key_training_enabled", False)),
    )
    decode_diagnostics = (
        _cost_code_decode_diagnostics(
            raw_bits=raw_message_bits,
            assignments=list(stego_meta.get("assignments", [])),
            block_probabilities=v2_block_probs,
            action_scores=v2_action_scores,
            wrong_key_block_probabilities=wrong_key_block_probabilities,
        )
        if args.protocol == COST_STEGO_V2_PROTOCOL and list(stego_meta.get("assignments", []))
        else {
            "true_key_margin": None,
            "wrong_key_margin": None,
            "per_bit_confidence": [],
            "wrong_key_per_bit_confidence": [],
            "selected_action_count_per_bit": [],
            "action_polarity_mismatch": {
                "selected_action_count": 0,
                "mismatch_count": 0,
                "mismatch_rate": None,
                "per_bit_mismatch_count": [],
            },
        }
    )

    clean_clamped_psnr = float(clean_eval["clamped_psnr"])
    clean_ssim = float(clean_eval["ssim"])
    watermarked_clamped_psnr = float(watermarked_eval["clamped_psnr"])
    watermarked_ssim = float(watermarked_eval["ssim"])
    clean_saturation = float(clean_eval["saturated_value_fraction"])
    watermarked_saturation = float(watermarked_eval["saturated_value_fraction"])
    psnr_drop = clean_clamped_psnr - watermarked_clamped_psnr
    ssim_drop = clean_ssim - watermarked_ssim
    off_canvas_delta = int(watermarked_eval["off_canvas_count"]) - int(clean_eval["off_canvas_count"])
    distribution_distance = distribution_distance_summary(
        clean_params_cpu,
        watermarked_eval["params"],
        clean_metrics=clean_eval,
        watermarked_metrics=watermarked_eval,
    )
    distribution_distance.update(covariance_and_circular_shift(clean_params_cpu, watermarked_eval["params"]))
    reported_distribution_distance = (
        distribution_distance
        if parameter_metrics_applicable
        else {
            **distribution_distance,
            "moment_shift": None,
            "MMD_rbf": None,
        }
    )
    reported_distribution_distance_extra = (
        {
            "covariance_shift": float(distribution_distance["covariance_shift"]),
            "circular_shift": float(distribution_distance["circular_shift"]),
            "theta_resultant_shift": float(distribution_distance["theta_resultant_shift"]),
        }
        if parameter_metrics_applicable
        else {
            "covariance_shift": None,
            "circular_shift": None,
            "theta_resultant_shift": None,
        }
    )
    watermarked_importance = compute_gaussian_importance(
        watermarked_eval["params"]["scales"],
        watermarked_eval["params"]["opacities"],
    )
    clean_run_features, run_feature_names = _legacy_report_feature_vector(
        clean_params_cpu,
        importance=clean_importance_cpu,
    )
    watermarked_run_features: torch.Tensor | None = None
    if parameter_metrics_applicable:
        watermarked_run_features, _ = _legacy_report_feature_vector(
            watermarked_eval["params"],
            importance=watermarked_importance,
        )
        single_run_detection = pooled_detection_statistics(clean_run_features, watermarked_run_features, seed=args.message_seed)
        detectability_auc = single_run_detection["auc"]
        false_alarm_rate = single_run_detection["false_alarm_rate"]
    else:
        detectability_auc = None
        false_alarm_rate = None
    if args.protocol == COST_STEGO_V2_PROTOCOL and _uses_native_parameter_embedding(args.baseline_type) and stego_meta["assignments"]:
        log_indices, alpha_indices = _v2_selected_indices(list(stego_meta["assignments"]))
        sensitivity_samples = []
        if log_indices.numel() > 0:
            sensitivity_samples.append(clean_sensitivity_cpu["log_anisotropy"][log_indices])
        if alpha_indices.numel() > 0:
            sensitivity_samples.append(clean_sensitivity_cpu["alpha"][alpha_indices])
        sensitivity_weights = torch.cat(sensitivity_samples, dim=0) if sensitivity_samples else torch.ones(1, dtype=torch.float32)
    elif args.protocol == COST_STEGO_V2_PROTOCOL:
        sensitivity_weights = torch.ones(1, dtype=torch.float32)
    elif args.protocol == DEFAULT_STEGO_PROTOCOL and stego_meta["assignments"]:
        log_indices, alpha_indices = active_channel_indices(list(stego_meta["assignments"]))
        sensitivity_samples: list[torch.Tensor] = []
        if log_indices.numel() > 0:
            sensitivity_samples.append(clean_sensitivity_cpu["log_anisotropy"][log_indices])
        if alpha_indices.numel() > 0:
            sensitivity_samples.append(clean_sensitivity_cpu["alpha"][alpha_indices])
        sensitivity_weights = torch.cat(sensitivity_samples, dim=0) if sensitivity_samples else torch.ones(1, dtype=torch.float32)
    elif args.protocol == DEFAULT_STEGO_PROTOCOL:
        sensitivity_weights = torch.ones(1, dtype=torch.float32)

    save_raw_tensor_image(target.detach().cpu(), output_dir / "target_image.png")
    save_raw_tensor_image(clean_eval["reconstruction"].clamp(0.0, 1.0), output_dir / "clean_reconstruction.png")
    save_raw_tensor_image(watermarked_eval["reconstruction"].clamp(0.0, 1.0), output_dir / "watermarked_reconstruction.png")
    save_comparison_figure(target.detach().cpu(), watermarked_eval["reconstruction"], output_dir / "watermarked_comparison.png")
    save_metrics_card(
        title="Steganography Metrics" if args.protocol in {DEFAULT_STEGO_PROTOCOL, COST_STEGO_V2_PROTOCOL, *PROTOTYPE_PROTOCOLS} else "Watermark Metrics",
        metric_rows=[
            ("Source run", source_run.run_dir.name),
            ("Protocol", str(stego_meta["stego_protocol"])),
            ("Mode", args.mode),
            ("Carrier pool", args.carrier_pool if args.protocol != DEFAULT_STEGO_PROTOCOL else "keyed_parity_blocks"),
            ("Carrier score", str(carrier_meta["carrier_score_mode"])),
            ("Stage", args.stage),
            ("Claim eligible", str(claim_eligible).lower()),
            ("Main claim eligible", str(main_claim_eligible).lower()),
            ("Baseline role", baseline_role),
            ("Claim scope", claim_scope),
            ("ECC", effective_ecc_mode),
            ("Raw payload bits", str(args.payload_bits)),
            ("Coded payload bits", str(int(coded_bits.numel()))),
            ("Raw BER", f"{raw_ber:.6f}"),
            ("Coded BER", f"{coded_ber:.6f}"),
            ("Detectability AUC", _detectability_display_value(detectability_auc)),
            ("False alarm rate", _detectability_display_value(false_alarm_rate)),
            ("Clean clamped PSNR", f"{clean_clamped_psnr:.3f} dB"),
            ("Watermarked clamped PSNR", f"{watermarked_clamped_psnr:.3f} dB"),
            ("PSNR drop", f"{psnr_drop:.3f} dB"),
            ("SSIM drop", f"{ssim_drop:.6f}"),
            ("MMD_rbf", _metric_display_value(float(distribution_distance["MMD_rbf"]) if parameter_metrics_applicable else None)),
            (
                "Covariance shift",
                _metric_display_value(float(distribution_distance["covariance_shift"]) if parameter_metrics_applicable else None),
            ),
            (
                "Circular shift",
                _metric_display_value(float(distribution_distance["circular_shift"]) if parameter_metrics_applicable else None),
            ),
            ("Wrong-key accuracy", _detectability_display_value(wrong_key_accuracy)),
            ("Wrong-key status", wrong_key_status),
            ("True-key margin", _metric_display_value(decode_diagnostics["true_key_margin"])),
            ("Wrong-key margin", _metric_display_value(decode_diagnostics["wrong_key_margin"])),
            (
                "Action polarity mismatch",
                _metric_display_value(decode_diagnostics["action_polarity_mismatch"]["mismatch_rate"]),
            ),
            ("Wet ratio", _detectability_display_value(stego_meta["wet_ratio"])),
            ("Modified Gaussians", str(int(stego_meta["modified_gaussian_count"]))),
            ("Saturation delta", f"{100.0 * (watermarked_saturation - clean_saturation):.3f} pp"),
            ("Off-canvas delta", str(off_canvas_delta)),
        ],
        notes=(
            "White-box parameter-domain extraction only; no blind render-only extraction claim.",
            protocol_note,
            claim_note,
            "Single-run reports export run-level feature vectors but do not claim a valid detectability AUC on their own.",
            "Pure pixel baselines do not expose comparable parameter-domain covertness metrics; those fields are reported as n/a.",
            "This artifact is intended as a bounded 2D research baseline rather than a broad robustness claim.",
        ),
        path=output_dir / "watermark_metrics_card.png",
    )

    torch.save(
        {
            "schema_version": 5,
            "artifact_type": "watermarked_export",
            "source_run_dir": str(source_run.run_dir),
            "metrics_kind": "evaluation_metrics",
            "config": source_run.config,
            "metrics": {key: value for key, value in watermarked_eval.items() if key not in {"reconstruction", "params", "sensitivity"}},
            "watermark": {
                "label": args.label,
                "source_run_dir": str(source_run.run_dir),
                "protocol": str(stego_meta["stego_protocol"]),
                "security_model": str(stego_meta["security_model"]),
                "mode": args.mode,
                "carrier_pool": args.carrier_pool,
                "stage": args.stage,
                "ecc_mode": effective_ecc_mode,
                "payload_bits": args.payload_bits,
                "coded_payload_bits": int(coded_bits.numel()),
                "message_bits": bits_to_string(raw_message_bits),
                "coded_message_bits": bits_to_string(coded_bits),
                "embedded_coded_bits": bits_to_string(embedded_coded_bits),
                "decoded_bits": bits_to_string(decoded_raw_bits),
                "decoded_coded_bits": bits_to_string(decoded_coded_bits),
                "carrier_indices": [int(index) for index in carrier_indices.tolist()],
                "coding_mode": str(stego_meta["coding_mode"]),
                "prototype_metadata": stego_meta.get("prototype_metadata", {}),
                "block_assignments": (
                    _serialize_cost_code_assignments(list(stego_meta["assignments"]))
                    if args.protocol == COST_STEGO_V2_PROTOCOL
                    else serializable_block_assignments(list(stego_meta["assignments"]))
                ),
                "wrong_key_assignments": _serialize_wrong_key_assignments(args.protocol, stego_meta["wrong_key_assignments"]),
                "spread_factor": stego_meta["spread_factor"],
                "requested_spread_factor": stego_meta["requested_spread_factor"],
                "wet_ratio": stego_meta["wet_ratio"],
                "requested_wet_ratio": stego_meta["requested_wet_ratio"],
                "relaxed_wet_count": stego_meta["relaxed_wet_count"],
                "relaxed_wet_indices": list(stego_meta["relaxed_wet_indices"]),
                "modified_gaussian_count": stego_meta["modified_gaussian_count"],
                "total_embed_cost": stego_meta["total_embed_cost"],
                "capacity_failure": stego_meta["capacity_failure"],
                "alpha_delta": stego_meta["alpha_delta"],
                "log_delta": stego_meta["log_delta"],
                "alpha_weight": float(args.alpha_weight),
                "key_hash": key_to_hash(args.key),
                "selection_entropy": stego_meta["selection_entropy"],
                "effective_action_budget": stego_meta["effective_action_budget"],
                "adaptive_budget_enabled": bool(stego_meta.get("adaptive_budget_enabled", False)),
                "adaptive_cover_profile": stego_meta.get("adaptive_cover_profile"),
                "adaptive_profile_stats": stego_meta.get("adaptive_profile_stats"),
                "effective_target_magnitude": float(
                    stego_meta.get("effective_target_magnitude")
                    if stego_meta.get("effective_target_magnitude") is not None
                    else stego_meta.get("log_delta")
                    if stego_meta.get("log_delta") is not None
                    else args.target_magnitude
                ),
                "effective_max_modification_ratio": float(
                    stego_meta.get(
                        "effective_max_modification_ratio",
                        _v2_max_modification_ratio(args) if args.protocol == COST_STEGO_V2_PROTOCOL else (args.max_modification_ratio or 0.0),
                    )
                ),
                "qualified_cover": stego_meta["qualified_cover"],
                "qualified_cover_reason": stego_meta.get("qualified_cover_reason"),
                "qualified_cover_failed_checks": stego_meta.get("qualified_cover_failed_checks"),
                "qualified_cover_diagnostics": stego_meta.get("qualified_cover_diagnostics"),
                "cost_sensitivity_weight": float(getattr(args, "cost_sensitivity_weight", 0.40)),
                "cost_density_weight": float(getattr(args, "cost_density_weight", 0.30)),
                "cost_visual_weight": float(getattr(args, "cost_visual_weight", 0.20)),
                "cost_detector_weight": float(getattr(args, "cost_detector_weight", 0.10)),
                "selection_probs": stego_meta.get("selection_probs"),
                "hard_forbid_mask": stego_meta.get("hard_forbid_mask"),
                "claim_eligible": claim_eligible,
                "main_claim_eligible": main_claim_eligible,
                "claim_track": str(getattr(args, "claim_track", "full")),
                "baseline_role": baseline_role,
                "steganography_claim_scope": claim_scope,
                "steganography_claim_note": claim_note,
                "protocol_capability_note": protocol_note,
                "baseline_type": stego_meta["baseline_type"],
                "native_ablation": str(stego_meta.get("native_ablation", args.native_ablation)),
                "decode_domain": stego_meta.get("decode_domain", "parameter"),
                "pixel_delta": stego_meta.get("pixel_delta"),
                "detector_training_enabled": bool(stego_meta.get("detector_training_enabled", False)),
                "detector_training_requested": bool(stego_meta.get("detector_training_requested", False)),
                "effective_detector_weight": float(stego_meta.get("effective_detector_weight", 0.0)),
                "wrong_key_training_enabled": bool(stego_meta.get("wrong_key_training_enabled", False)),
                "wrong_key_training_requested": bool(stego_meta.get("wrong_key_training_requested", False)),
                "attack_training_enabled": bool(stego_meta.get("attack_training_enabled", False)),
                "attack_training_requested": bool(stego_meta.get("attack_training_requested", False)),
                "trained_wrong_key_groups": int(stego_meta.get("trained_wrong_key_groups", 0)),
                "wrong_key_sample_failures": int(stego_meta.get("wrong_key_sample_failures", 0)),
                "wrong_key_status": wrong_key_status,
                "decode_diagnostics": decode_diagnostics,
            },
            "run_level_feature_names": run_feature_names,
            "run_level_features": None if watermarked_run_features is None else [float(value.item()) for value in watermarked_run_features],
            "parameters": {
                **{key: value.detach().cpu() for key, value in watermarked_eval["params"].items()},
                "importance": watermarked_importance.detach().cpu(),
            },
            "canonical_parameters": {
                key: value.detach().cpu()
                for key, value in canonicalize_gaussian_geometry(
                    watermarked_eval["params"]["scales"],
                    watermarked_eval["params"]["thetas"],
                ).items()
            },
            "sensitivity": {
                key: value.detach().cpu()
                for key, value in watermarked_eval["sensitivity"].items()
            },
            "watermark_images": {
                "clean_image": stego_meta["clean_image_tensor"].detach().cpu()
                if stego_meta.get("clean_image_tensor") is not None
                else clean_eval["reconstruction"].detach().cpu(),
                "reference_image": stego_meta["reference_image_tensor"].detach().cpu()
                if stego_meta.get("reference_image_tensor") is not None
                else watermarked_eval["reconstruction"].detach().cpu(),
            },
        },
        data_dir / "watermarked_params.pt",
    )
    export_parameter_table(
        offsets=watermarked_eval["params"]["offsets"],
        scales=watermarked_eval["params"]["scales"],
        thetas=watermarked_eval["params"]["thetas"],
        colors=watermarked_eval["params"]["colors"],
        opacities=watermarked_eval["params"]["opacities"],
        importance=watermarked_importance,
        height=int(source_run.config["height"]),
        width=int(source_run.config["width"]),
        path=data_dir / "watermarked_params.csv",
        sensitivity=watermarked_eval["sensitivity"],
    )

    export_payload = {
        "schema_version": 5,
        "label": args.label,
        "source_run_dir": str(source_run.run_dir),
        "protocol": str(stego_meta["stego_protocol"]),
        "stego_protocol": str(stego_meta["stego_protocol"]),
        "security_model": str(stego_meta["security_model"]),
        "mode": args.mode,
        "carrier_semantics": (
            "canonical_log_anisotropy"
            if args.mode == LOG_ANISOTROPY_MODE
            else "legacy_signed_logratio"
            if args.mode == LEGACY_LOGRATIO_MODE
            else "canonical_theta_major"
        ),
        "carrier_pool": args.carrier_pool,
        "carrier_score_mode": carrier_meta["carrier_score_mode"],
        "stage": args.stage,
        "payload_bits": args.payload_bits,
        "coded_payload_bits": int(coded_bits.numel()),
        "ecc_mode": effective_ecc_mode,
        "key_hash": key_to_hash(args.key),
        "embed_key": args.key,
        "message_bits": bits_to_string(raw_message_bits),
        "coded_message_bits": bits_to_string(coded_bits),
        "embedded_coded_bits": bits_to_string(embedded_coded_bits),
        "decoded_bits": bits_to_string(decoded_raw_bits),
        "decoded_coded_bits": bits_to_string(decoded_coded_bits),
        "carrier_indices": [int(index) for index in carrier_indices.tolist()],
        "coding_mode": str(stego_meta["coding_mode"]),
        "prototype_metadata": stego_meta.get("prototype_metadata", {}),
        "block_assignments": (
            _serialize_cost_code_assignments(list(stego_meta["assignments"]))
            if args.protocol == COST_STEGO_V2_PROTOCOL
            else serializable_block_assignments(list(stego_meta["assignments"]))
        ),
        "wrong_key_assignments": _serialize_wrong_key_assignments(args.protocol, stego_meta["wrong_key_assignments"]),
        "msg_strength": args.msg_strength,
        "dist_weight": args.dist_weight,
        "anchor_weight": args.anchor_weight,
        "mmd_weight": args.mmd_weight,
        "detector_weight": args.detector_weight,
        "effective_detector_weight": float(stego_meta.get("effective_detector_weight", 0.0)),
        "alpha_weight": args.alpha_weight,
        "ewc_weight": args.ewc_weight,
        "wasserstein_weight": args.wasserstein_weight,
        "selection_penalty_weight": args.selection_penalty_weight,
        "selection_temperature": args.selection_temperature,
        "cost_sensitivity_weight": float(getattr(args, "cost_sensitivity_weight", 0.40)),
        "cost_density_weight": float(getattr(args, "cost_density_weight", 0.30)),
        "cost_visual_weight": float(getattr(args, "cost_visual_weight", 0.20)),
        "cost_detector_weight": float(getattr(args, "cost_detector_weight", 0.10)),
        "max_modification_ratio": (
            float(stego_meta.get("effective_max_modification_ratio"))
            if args.protocol == COST_STEGO_V2_PROTOCOL and stego_meta.get("effective_max_modification_ratio") is not None
            else _v2_max_modification_ratio(args)
            if args.protocol == COST_STEGO_V2_PROTOCOL
            else args.max_modification_ratio
        ),
        "adaptive_budget_enabled": bool(stego_meta.get("adaptive_budget_enabled", False)),
        "adaptive_cover_profile": stego_meta.get("adaptive_cover_profile"),
        "adaptive_profile_stats": stego_meta.get("adaptive_profile_stats"),
        "wrong_key_samples": args.wrong_key_samples,
        "native_ablation": str(stego_meta.get("native_ablation", args.native_ablation)),
        "detector_training_enabled": bool(stego_meta.get("detector_training_enabled", False)),
        "detector_training_requested": bool(stego_meta.get("detector_training_requested", False)),
        "wrong_key_training_enabled": bool(stego_meta.get("wrong_key_training_enabled", False)),
        "wrong_key_training_requested": bool(stego_meta.get("wrong_key_training_requested", False)),
        "attack_training_enabled": bool(stego_meta.get("attack_training_enabled", False)),
        "attack_training_requested": bool(stego_meta.get("attack_training_requested", False)),
        "trained_wrong_key_groups": int(stego_meta.get("trained_wrong_key_groups", 0)),
        "wrong_key_sample_failures": int(stego_meta.get("wrong_key_sample_failures", 0)),
        "beam_width": args.beam_width,
        "covertness_weight": args.covertness_weight,
        "wet_threshold": args.wet_threshold,
        "spread_factor": stego_meta["spread_factor"],
        "requested_spread_factor": stego_meta["requested_spread_factor"],
        "log_delta": stego_meta["log_delta"],
        "alpha_delta": stego_meta["alpha_delta"],
        "theta_delta": stego_meta.get("theta_delta"),
        "parity_block_size": args.parity_block_size,
        "cost_model": args.cost_model,
        "effective_steps": effective_steps,
        "tune_steps": args.tune_steps,
        "joint_steps": args.joint_steps,
        "tune_lr": args.tune_lr,
        "target_magnitude": float(
            stego_meta.get("effective_target_magnitude")
            if stego_meta.get("effective_target_magnitude") is not None
            else stego_meta.get("log_delta")
            if stego_meta.get("log_delta") is not None
            else args.target_magnitude
        ),
        "candidate_count": int(carrier_meta["candidate_count"]),
        "dataset_name": args.dataset_name,
        "image_id": args.image_id or source_run.run_dir.name,
        "crop_id": args.crop_id,
        "fit_seed": args.fit_seed,
        "actor_id": args.actor_id,
        "baseline_type": stego_meta["baseline_type"],
        "decode_domain": stego_meta.get("decode_domain", "parameter"),
        "pixel_delta": stego_meta.get("pixel_delta"),
        "clean_metrics": {key: value for key, value in clean_eval.items() if key not in {"reconstruction", "params", "sensitivity"}},
        "watermarked_metrics": {key: value for key, value in watermarked_eval.items() if key not in {"reconstruction", "params", "sensitivity"}},
        "distribution_distance_vs_clean": reported_distribution_distance,
        "parameter_metrics_applicable": parameter_metrics_applicable,
        "sensitivity_stats": {
            "carrier_mean": float(sensitivity_weights.mean().item()),
            "carrier_min": float(sensitivity_weights.min().item()),
            "carrier_max": float(sensitivity_weights.max().item()),
        },
        "ecc_metadata": {**ecc_meta, **decode_meta},
        "raw_ber": raw_ber,
        "coded_ber": coded_ber,
        "psnr_drop": psnr_drop,
        "ssim_drop": ssim_drop,
        "saturation_delta": watermarked_saturation - clean_saturation,
        "off_canvas_delta": off_canvas_delta,
        "detectability_auc": detectability_auc,
        "pooled_detectability_auc": stego_meta["pooled_detectability_auc"],
        "false_alarm_rate": false_alarm_rate if false_alarm_rate is not None else stego_meta["false_alarm_rate"],
        "detectability_protocol": "run_level_cross_validated_linear_probe" if parameter_metrics_applicable else None,
        "detectability_status": (
            "not_applicable_pixel_domain"
            if not parameter_metrics_applicable
            else "requires_multi_run_aggregation"
            if detectability_auc is None
            else "computed"
        ),
        "run_level_feature_names": run_feature_names,
        "clean_run_level_features": [float(value.item()) for value in clean_run_features],
        "watermarked_run_level_features": None if watermarked_run_features is None else [float(value.item()) for value in watermarked_run_features],
        "decode_block_scores": [float(value) for value in decode_scores.detach().cpu().tolist()],
        "wrong_key_accuracy": wrong_key_accuracy,
        "capacity_failure": bool(stego_meta["capacity_failure"]),
        "wet_ratio": stego_meta["wet_ratio"],
        "requested_wet_ratio": stego_meta["requested_wet_ratio"],
        "relaxed_wet_count": stego_meta["relaxed_wet_count"],
        "relaxed_wet_indices": list(stego_meta["relaxed_wet_indices"]),
        "dry_count_before_relaxation": stego_meta["dry_count_before_relaxation"],
        "modified_gaussian_count": stego_meta["modified_gaussian_count"],
        "total_embed_cost": stego_meta["total_embed_cost"],
        "selection_entropy": stego_meta["selection_entropy"],
        "effective_action_budget": stego_meta["effective_action_budget"],
        "qualified_cover": stego_meta["qualified_cover"],
        "qualified_cover_reason": stego_meta.get("qualified_cover_reason"),
        "qualified_cover_failed_checks": stego_meta.get("qualified_cover_failed_checks"),
        "qualified_cover_diagnostics": stego_meta.get("qualified_cover_diagnostics"),
        "selection_probs": stego_meta.get("selection_probs"),
        "hard_forbid_mask": stego_meta.get("hard_forbid_mask"),
        "claim_eligible": claim_eligible,
        "main_claim_eligible": main_claim_eligible,
        "claim_track": str(getattr(args, "claim_track", "full")),
        "baseline_role": baseline_role,
        "steganography_claim_scope": claim_scope,
        "steganography_claim_note": claim_note,
        "protocol_capability_note": protocol_note,
        "same_source_auc": stego_meta["same_source_auc"],
        "cross_source_auc": stego_meta["cross_source_auc"],
        "actor_holdout_auc": stego_meta["actor_holdout_auc"],
        "compensation_indices": list(stego_meta["compensation_indices"]),
        "distribution_distance_extra": reported_distribution_distance_extra,
        "wrong_key_status": wrong_key_status,
        "true_key_margin": decode_diagnostics["true_key_margin"],
        "wrong_key_margin": decode_diagnostics["wrong_key_margin"],
        "per_bit_confidence": decode_diagnostics["per_bit_confidence"],
        "wrong_key_per_bit_confidence": decode_diagnostics["wrong_key_per_bit_confidence"],
        "selected_action_count_per_bit": decode_diagnostics["selected_action_count_per_bit"],
        "action_polarity_mismatch": decode_diagnostics["action_polarity_mismatch"],
        "decode_diagnostics": decode_diagnostics,
    }
    save_json(export_payload, data_dir / "watermark_report.json")
    (data_dir / "message_bits.txt").write_text(export_payload["message_bits"] + "\n", encoding="utf-8")
    (data_dir / "coded_message_bits.txt").write_text(export_payload["coded_message_bits"] + "\n", encoding="utf-8")
    (data_dir / "decoded_bits.txt").write_text(export_payload["decoded_bits"] + "\n", encoding="utf-8")
    (data_dir / "decoded_coded_bits.txt").write_text(export_payload["decoded_coded_bits"] + "\n", encoding="utf-8")

    print(f"Watermark directory: {output_dir}")
    print(f"Watermark report: {data_dir / 'watermark_report.json'}")


if __name__ == "__main__":
    main()
