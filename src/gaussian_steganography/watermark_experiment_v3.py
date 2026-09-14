"""Fit-dependent parameter embedding and checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

# ── Shared infrastructure (unchanged from v2.0) ──────────────────────────────
from gaussian2d import Gaussian2DRenderer
from run_fit import build_optimizer, build_scheduler, compute_psnr, initialize_renderer
from utils import (
    canonicalize_gaussian_geometry,
    compute_gaussian_importance,
    export_parameter_table,
    get_best_device,
    save_comparison_figure,
    save_json,
    save_metrics_card,
    save_raw_tensor_image,
    set_seed,
)

# ── v2.0 protocol helpers (all unchanged) ────────────────────────────────────
from stego_protocol import (
    ChannelCostBundle,
    DEFAULT_SECURITY_MODEL,
    DEFAULT_STEGO_PROTOCOL,
    RunLevelStegaDetector,
    assignment_bit_tensor,
    build_density_reference,
    build_detector_feature_vector,
    build_soft_selection_field,
    build_unkeyed_selection_field,
    carrier_margin_loss,
    capacity_failure_report,
    carrier_ewc_penalty,
    cost_code_action_margin_loss,
    cost_code_action_target_loss,
    covariance_and_circular_shift,
    decode_image_cost_code,
    decode_parity_blocks,
    density_reference_loss,
    detector_embed_loss,
    detector_feature_names,
    grouped_pooled_detection_statistics,
    mmd_distribution_loss,
    pooled_detection_statistics,
    rgb_to_ycbcr,
    selection_channel_penalty,
    select_compensation_neighbors,
    sliced_wasserstein_loss,
    train_detector_steps,
    ycbcr_to_rgb,
)
from watermark_utils import (
    LOG_ANISOTROPY_MODE,
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
    renderer_from_params,
    select_low_importance_carriers,
    select_sensitivity_candidates,
    select_topk_carriers,
    summarize_run_features,
)

# ── Import a subset of shared helpers from the v2.0 experiment module ─────────
# We use a conditional import so that watermark_experiment_v3 can be used
# even before all v2.0 imports are available.
try:
    from watermark_experiment import (
        _cost_stego_v2_defaults,
        _adaptive_v2_embedding_controls,
        _v2_selected_indices,
        _v2_assignments_for_compensation,
        _carrier_preserving_attack_params,
        _collect_wrong_key_assignment_groups,
        _native_detector_requested,
        _native_wrong_key_requested,
        _native_attack_training_requested,
        _straight_through_quantize,
        _uses_native_parameter_embedding,
        _supports_keyed_assignment_family,
        _masked_rows_step,
        _qualified_cover_diagnostics,
        MAIN_CLAIM_LABEL,
        VALID_NATIVE_ABLATIONS,
        VALID_CLAIM_TRACKS,
        KEYED_PARAMETER_BASELINES,
        HEURISTIC_PARAMETER_BASELINES,
        NATIVE_PARAMETER_BASELINES,
        default_watermark_output_dir,
    )
    _V2_EXPERIMENT_AVAILABLE = True
except ImportError:
    _V2_EXPERIMENT_AVAILABLE = False

# ── v3.0 protocol (new) ───────────────────────────────────────────────────────
from stego_protocol_v3 import (
    COST_STEGO_V3_PROTOCOL,
    COST_STEGO_V3_1_PROTOCOL,
    V3ChannelCostBundle,
    _build_v3_selection_proxy_bundle,
    PerturbationHardenedDecoder,
    SensitivityAdaptiveEncoder,
    SinkhornCovertnessTerm,
    build_v3_cost_code_assignments,
    build_v3_random_cost_code_assignments,
    build_v3_greedy_nokey_cost_code_assignments,
    build_v3_random_polarity_cost_code_assignments,
    build_v3_channel_costs,
    compute_v3_channel_weights,
    cost_code_block_probabilities_v3,
    decode_cost_code_v3,
    enforce_assignment_coverage_constraints,
    estimate_embedding_capacity,
    extract_modified_carriers,
    cost_stego_v3_defaults,
    parity_block_logits_v3,
    prune_assignments_by_snr,
    theta_channel_ber,
    color_lum_channel_ber,
    v3_per_channel_action_counts,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"

VALID_CARRIER_CHANNELS_V3 = ("log_anisotropy", "alpha", "theta", "color_lum")
COST_STEGO_V3_PROTOCOL_ID = COST_STEGO_V3_1_PROTOCOL
V3_MAIN_CLAIM_LABEL = "v3_main_native_8"
V3_1_FALLBACK_PAYLOAD_BITS = 8
V3_1_RELIABLE_CARRIER_FLOOR = 384
GREEDY_ATTACKER_PUBLIC_SEED = "public-greedy-attacker-v1"
GREEDY_NOKEY_PUBLIC_SEED = "public-greedy-nokey-v1"


def _primary_assignment_key(args: argparse.Namespace) -> str:
    if str(getattr(args, "baseline_type", "native")) == "native_greedy_nokey":
        return GREEDY_NOKEY_PUBLIC_SEED
    return str(args.key)
_WARNED_RUNTIME_EVENTS: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_RUNTIME_EVENTS:
        return
    _WARNED_RUNTIME_EVENTS.add(key)
    print(message)


def _v3_claim_eligible(
    *,
    baseline_type: str,
    native_ablation: str,
    claim_track: str,
    qualified_cover: bool | None,
    detector_training_enabled: bool,
    wrong_key_training_enabled: bool,
    capacity_failure: bool,
) -> bool:
    return bool(
        baseline_type == "native"
        and native_ablation == claim_track
        and qualified_cover is True
        and wrong_key_training_enabled
        and (claim_track != "full" or detector_training_enabled)
        and not capacity_failure
    )


def _v3_main_claim_eligible(
    *,
    label: str | None,
    baseline_type: str,
    native_ablation: str,
    claim_track: str,
    claim_eligible: bool,
    requested_payload_bits: int | None = None,
    effective_payload_bits: int | None = None,
    threat_model: str = "benign_carrier",
    profile_policy: str = "legacy",
) -> bool:
    eligible = bool(
        claim_eligible
        and baseline_type == "native"
        and native_ablation == claim_track
        and label == V3_MAIN_CLAIM_LABEL
    )
    if not eligible:
        return False
    if str(profile_policy) == "publishable_2d_v1":
        return bool(
            str(threat_model) == "benign_carrier"
            and int(requested_payload_bits or 0) == 8
            and int(effective_payload_bits or 0) == 8
        )
    return True


def _v3_baseline_role(
    *,
    label: str | None,
    baseline_type: str,
    native_ablation: str,
    claim_track: str,
    claim_eligible: bool,
    main_claim_eligible: bool,
) -> str:
    if baseline_type in {"pixel", "render_refit"}:
        return "transfer_baseline"
    if baseline_type == "native_matched_random":
        return "matched_random_baseline"
    if baseline_type in HEURISTIC_PARAMETER_BASELINES:
        return "heuristic_nonkey_baseline"
    if baseline_type == "native" and native_ablation != claim_track:
        return "ablations"
    if main_claim_eligible:
        return "main_native"
    if claim_eligible:
        return "supporting_native"
    if label is not None and ("ablation" in label or "stress" in label or "inno_" in label):
        return "ablations"
    return "failure_analysis"


# ============================================================================
# CLI argument parsing (v3 extensions)
# ============================================================================

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


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def parse_args_v3(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments for watermark_experiment_v3.

    All v2.0 arguments are accepted plus v3-specific additions.
    """
    parser = argparse.ArgumentParser(
        description="v3.0 Gaussian parameter-domain steganography experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Mandatory ────────────────────────────────────────────────────────────
    parser.add_argument("--source-run", type=str, required=True,
                        help="Path to a clean exported run directory.")

    # ── Protocol & mode ──────────────────────────────────────────────────────
    parser.add_argument("--protocol", type=str, default=COST_STEGO_V3_1_PROTOCOL,
                        help="Embedding protocol. Use 'cost_stego_v3_1' for v3.1.")
    parser.add_argument("--baseline-type", type=str, default="native",
                        choices=["native", "native_matched_random", "native_greedy_nokey",
                                 "native_random_polarity"],
                        help="Baseline variant.")
    parser.add_argument("--native-ablation", type=str, default="no_detector",
                        choices=list(VALID_NATIVE_ABLATIONS) if _V2_EXPERIMENT_AVAILABLE
                                else ["full", "no_detector", "no_wrong_key", "no_attack_training"],
                        help="Ablation to run (v3.0 default: no_detector).")
    parser.add_argument("--claim-track", type=str, default="no_detector",
                        choices=["full", "no_detector"],
                        help="Which track carries the main paper claim.")

    # ── Payload ──────────────────────────────────────────────────────────────
    parser.add_argument("--payload-bits", type=positive_int, default=8,
                        help="Number of raw payload bits (used if --auto-payload-bits is off).")
    parser.add_argument("--auto-payload-bits", action="store_true",
                        help="[Innovation 5] Automatically estimate payload capacity from "
                             "the V3ChannelCostBundle before embedding.")
    parser.add_argument("--message", type=str, default=None,
                        help="Optional explicit bitstring.")
    parser.add_argument("--message-seed", type=int, default=0,
                        help="Seed for random bit generation.")

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument("--tune-steps", type=positive_int, default=200,
                        help="Embedding optimisation steps.")
    parser.add_argument("--tune-lr", type=positive_float, default=0.01,
                        help="Learning rate for embedding fine-tuning.")
    parser.add_argument("--msg-strength", type=positive_float, default=0.01,
                        help="Message loss weight.")
    parser.add_argument("--wasserstein-weight", type=float, default=0.05,
                        help="Sliced-Wasserstein covertness weight (v2.0 baseline).")
    parser.add_argument("--selection-penalty-weight", type=float, default=0.05,
                        help="Selection-channel penalty weight.")
    parser.add_argument("--selection-temperature", type=positive_float, default=0.5,
                        help="Temperature for Gumbel-top-k carrier sampling.")
    parser.add_argument("--beam-width", type=positive_int, default=64,
                        help="Beam width for cost-code block solving.")
    parser.add_argument("--min-selected-count", type=non_negative_int, default=0,
                        help="Optional preferred floor for selected actions per 4-bit parity block.")
    parser.add_argument("--covertness-weight", type=float, default=1.0,
                        help="Global covertness loss multiplier.")
    parser.add_argument("--wrong-key-samples", type=positive_int, default=4,
                        help="Wrong-key samples per contrastive step.")
    parser.add_argument("--alpha-weight", type=float, default=0.5,
                        help="Alpha channel relative contribution.")
    parser.add_argument("--wet-threshold", type=float, default=0.20,
                        help="Quantile threshold for the wet-carrier mask.")
    parser.add_argument("--max-modification-ratio", type=float, default=None,
                        help="Max fraction of Gaussians that may be modified.")
    parser.add_argument("--target-magnitude", type=positive_float, default=0.35,
                        help="Embedding magnitude for log_anisotropy channel.")

    # ── Adaptive budget ───────────────────────────────────────────────────────
    parser.add_argument("--adaptive-native-budget", action="store_true",
                        help="Enable cover-adaptive embedding budget heuristic.")

    # ── v3 carrier channels (Innovation 2) ───────────────────────────────────
    parser.add_argument("--carrier-channels", type=str,
                        default="log_anisotropy,alpha,color_lum",
                        help="Comma-separated list of active carrier channels. "
                             "Supported: log_anisotropy, alpha, theta, color_lum.")
    parser.add_argument("--decoder-variant", type=str, default="v3_full",
                        choices=["v2_compat", "v3_full"],
                        help="Which decoder/action-score path to use for training and evaluation.")
    parser.add_argument("--coding-strategy", type=str, default="snr_prune",
                        choices=["fixed74", "snr_prune"],
                        help="Payload coding strategy for v3.1.")
    parser.add_argument("--structure-profile", type=str, default="auto",
                        choices=["auto", "generic", "periodic"],
                        help="Structure-aware embedding profile. 'auto' infers from the target.")
    parser.add_argument("--threat-model", type=str, default="benign_carrier",
                        choices=["benign_carrier", "active_refit"],
                        help="Threat model declaration for claim/audit partitioning.")
    parser.add_argument("--profile-policy", type=str, default="legacy",
                        choices=["legacy", "publishable_2d_v1"],
                        help="2D profile policy. publishable_2d_v1 enables 2D periodic zoning and balanced coverage.")
    parser.add_argument("--wavelet-band-loss-weight", type=float, default=None,
                        help="Optional override for 2D Haar detail-band consistency weight.")
    parser.add_argument("--stripe-luminance-loss-weight", type=float, default=None,
                        help="Optional override for 2D stripe-luminance smoothness weight.")
    parser.add_argument("--disable-region-policy", action="store_true",
                        help="Disable the 2D region-zone policy even under publishable_2d_v1.")
    parser.add_argument("--disable-coverage-balancing", action="store_true",
                        help="Disable stripe-balanced bin coverage while keeping other v3.2 losses active.")
    parser.add_argument("--periodic-lowfreq-carrier-bias", action="store_true",
                        help="Bias periodic covers away from periodic-core carriers and toward transition/generic carriers.")
    parser.add_argument("--periodic-bias-mode", type=str, default="legacy",
                        choices=[
                            "legacy",
                            "transition_alpha_loganis_v1",
                            "transition_alpha_loganis_strict_v1",
                            "transition_alpha_phase_guard_v1",
                            "transition_alpha_only_v2",
                            "transition_alpha_dispersion_v1",
                            "carrier_compensator_split_v1",
                            "quantize_consistency_v1",
                        ],
                        help="Periodic carrier-bias mode. 'legacy' keeps per-Gaussian scaling; "
                             "the other modes apply per-candidate region/channel scaling.")
    parser.add_argument("--periodic-bias-strength", type=positive_float, default=0.35,
                        help="Strength of the periodic low-frequency carrier bias multiplier.")
    parser.add_argument("--periodic-extra-compensation-per-active", type=non_negative_int, default=0,
                        help="Extra periodic compensators to add per active carrier during the compensation pass.")
    parser.add_argument("--publishable-stage-lock", type=str, default="auto",
                        choices=["auto", "P1", "P2", "G0", "G1", "G2"],
                        help="Optional publishable ladder stage lock for matched local experiments.")
    parser.add_argument("--periodic-min-stage", type=str, default="auto",
                        choices=["auto", "P1", "P2"],
                        help="Minimum publishable ladder stage for the periodic profile. "
                             "'auto' (default): proxy can select among P0/P1/P2 as usual. "
                             "'P1': drop the aggressive P0 from the periodic ladder. "
                             "'P2': drop both P0 and P1, leaving only the most conservative stage. "
                             "Non-periodic profiles are unaffected.")
    parser.add_argument("--theta-delta", type=positive_float, default=0.05,
                        help="Embedding step for theta_major channel (radians).")
    parser.add_argument("--lum-delta", type=positive_float, default=0.008,
                        help="Embedding step for color_luminance channel (pixel-space).")

    # ── Perturbation-hardened training (Innovation 1) ────────────────────────
    parser.add_argument("--hardening-sigma-ratio", type=float, default=0.30,
                        help="Stage-B noise std as fraction of embedding_delta. "
                             "Set 0.0 to disable perturbation hardening.")
    parser.add_argument("--hardening-sigma-ratio-c", type=float, default=0.50,
                        help="Stage-C noise std as fraction of embedding_delta.")
    parser.add_argument("--hardening-start-iter", type=int, default=100,
                        help="Iteration at which Stage B (hardening) begins.")
    parser.add_argument("--hardening-end-iter", type=int, default=180,
                        help="Iteration at which Stage C (max hardening) begins.")
    parser.add_argument("--mixed-distortion-hardening", action="store_true",
                        help="Augment noise hardening with quantization/noise decode surrogates and a blur-consistency term.")
    parser.add_argument("--mixed-distortion-hardening-weight", type=positive_float, default=0.35,
                        help="Relative weight of the mixed-distortion hardening auxiliary loss.")
    parser.add_argument("--hardening-quantize-step", type=positive_float, default=0.01,
                        help="Quantization step used by mixed-distortion hardening.")
    parser.add_argument("--hardening-small-noise-step", type=positive_float, default=0.01,
                        help="Uniform carrier-parameter jitter magnitude used by mixed-distortion hardening.")
    parser.add_argument("--hardening-blur-sigma", type=positive_float, default=1.0,
                        help="Gaussian blur sigma used by the render-space blur surrogate in mixed-distortion hardening.")

    # ── Sinkhorn OT covertness (Innovation 3) ────────────────────────────────
    parser.add_argument("--ot-covertness-weight", type=float, default=0.15,
                        help="Sinkhorn divergence covertness weight. "
                             "Set 0.0 to disable (falls back to sliced-Wasserstein only).")
    parser.add_argument("--sinkhorn-blur", type=positive_float, default=0.05,
                        help="Sinkhorn regularisation blur parameter (ε = blur²).")

    # ── Adaptive coding (Innovation 4) ───────────────────────────────────────
    parser.add_argument("--adaptive-coding", action="store_true",
                        help="[Innovation 4] Enable per-bit adaptive Hamming block size "
                             "selection based on per-carrier SNR estimates.")
    parser.add_argument("--refit-adversary-steps", type=non_negative_int, default=0,
                        help="Number of truncated reconstruction-only inner steps for the v3.1 refit adversary.")
    parser.add_argument("--refit-adversary-weight", type=float, default=0.0,
                        help="Weight of the refit-aware decode and margin penalties.")

    # ── v2.0 cost weights (kept for ablation) ────────────────────────────────
    parser.add_argument("--cost-sensitivity-weight", type=float, default=0.40)
    parser.add_argument("--cost-density-weight",     type=float, default=0.30)
    parser.add_argument("--cost-visual-weight",      type=float, default=0.20)
    parser.add_argument("--cost-detector-weight",    type=float, default=0.10)
    parser.add_argument("--native-margin-weight", type=float, default=None,
                        help="Override the correct-key parity-margin loss weight "
                             "(default: 0.10 for <=8 bits, 0.08 otherwise). "
                             "Set 0.0 to disable the margin term (P0-2 objective-symmetry ablation).")
    parser.add_argument("--wrong-key-weight", type=float, default=None,
                        help="Override the wrong-key chance loss weight.")
    parser.add_argument("--native-wrong-key-specificity-weight", type=float, default=None,
                        help="Override the native wrong-key specificity loss weight.")
    parser.add_argument("--mmd-weight",  type=float, default=None)
    parser.add_argument("--detector-weight", type=float, default=None)
    parser.add_argument("--ewc-weight",  type=float, default=None)
    parser.add_argument("--dist-weight", type=float, default=0.0)

    # ── Study metadata ────────────────────────────────────────────────────────
    parser.add_argument("--key",         type=str, default="ucl-2d-watermark-v3")
    parser.add_argument("--dataset-name", type=str, default="ad_hoc")
    parser.add_argument("--image-id",   type=str, default=None)
    parser.add_argument("--crop-id",    type=str, default="center")
    parser.add_argument("--fit-seed",   type=int, default=0)
    parser.add_argument("--actor-id",   type=str, default=None)
    parser.add_argument("--label",      type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-wrong-keys", type=positive_int, default=32,
                        help="Wrong-key pairs to report (not used in training; "
                             "use --wrong-key-samples for training samples).")
    parser.add_argument("--detectability-bootstrap-rounds", type=positive_int, default=1000)
    return parser.parse_args(argv)


# ============================================================================
# V3 defaults helper
# ============================================================================

def _cost_stego_v3_experiment_defaults(args: argparse.Namespace) -> dict:
    """Merge v2.0 defaults with v3 overrides for the experiment."""
    # Start from the v3 protocol defaults
    d = cost_stego_v3_defaults()

    # Override with explicit CLI arguments
    d["msg_strength"]        = float(args.msg_strength)
    d["wasserstein_weight"]  = float(args.wasserstein_weight)
    d["selection_weight"]    = float(args.selection_penalty_weight)
    d["ot_covertness_weight"]= float(args.ot_covertness_weight)
    d["sinkhorn_blur"]       = float(args.sinkhorn_blur)

    # Hard-coded weights matching v2.0 (only changes are Sinkhorn & hardening)
    if args.payload_bits <= 8:
        d["mmd_weight"]       = 0.08 if args.mmd_weight is None else float(args.mmd_weight)
        d["detector_weight"]  = 0.04 if args.detector_weight is None else float(args.detector_weight)
        d["moment_weight"]    = 0.01 if args.dist_weight == 0.0 else float(args.dist_weight)
        d["ewc_weight"]       = 0.01 if args.ewc_weight is None else float(args.ewc_weight)
        d["compensation_weight"]   = 0.02
        d["wrong_key_weight"]      = 0.06
        d["native_margin_weight"]  = 0.10
        d["native_wrong_key_specificity_weight"] = 0.12
    else:
        d["mmd_weight"]       = 0.10 if args.mmd_weight is None else float(args.mmd_weight)
        d["detector_weight"]  = 0.05 if args.detector_weight is None else float(args.detector_weight)
        d["moment_weight"]    = 0.01 if args.dist_weight == 0.0 else float(args.dist_weight)
        d["ewc_weight"]       = 0.015 if args.ewc_weight is None else float(args.ewc_weight)
        d["compensation_weight"]   = 0.03
        d["wrong_key_weight"]      = 0.05
        d["native_margin_weight"]  = 0.08
        d["native_wrong_key_specificity_weight"] = 0.10
    # P0-2 ablation hook: explicit override of the correct-key margin weight.
    # Default (None) preserves the original behavior above; --native-margin-weight 0 disables it.
    if getattr(args, "native_margin_weight", None) is not None:
        d["native_margin_weight"] = float(args.native_margin_weight)
    if getattr(args, "wrong_key_weight", None) is not None:
        d["wrong_key_weight"] = float(args.wrong_key_weight)
    if getattr(args, "native_wrong_key_specificity_weight", None) is not None:
        d["native_wrong_key_specificity_weight"] = float(args.native_wrong_key_specificity_weight)
    return d


def _active_carrier_channels(args: argparse.Namespace) -> list[str]:
    """Parse the --carrier-channels argument into a list of channel names."""
    raw = str(getattr(args, "carrier_channels", "log_anisotropy,alpha,color_lum"))
    channels = [c.strip() for c in raw.split(",") if c.strip()]
    valid = set(VALID_CARRIER_CHANNELS_V3)
    for ch in channels:
        if ch not in valid:
            raise ValueError(f"Unknown carrier channel: {ch!r}. Valid: {sorted(valid)}")
    return channels


def _resolve_coding_strategy(args: argparse.Namespace) -> str:
    strategy = str(getattr(args, "coding_strategy", "snr_prune"))
    if bool(getattr(args, "adaptive_coding", False)) and strategy == "fixed74":
        strategy = "snr_prune"
    return strategy


def _resize_payload_bits(raw_bits: torch.Tensor, effective_payload_bits: int) -> torch.Tensor:
    effective_payload_bits = int(effective_payload_bits)
    if raw_bits.numel() > effective_payload_bits:
        return raw_bits[:effective_payload_bits]
    if raw_bits.numel() < effective_payload_bits:
        return torch.cat(
            [
                raw_bits,
                torch.zeros(
                    effective_payload_bits - raw_bits.numel(),
                    dtype=raw_bits.dtype,
                    device=raw_bits.device,
                ),
            ]
        )
    return raw_bits


def _resolve_payload_fallback(
    *,
    requested_payload_bits: int,
    effective_payload_bits: int,
    estimated_capacity: int,
    reliable_carriers: int,
    structure_profile: str,
) -> tuple[int, bool, list[str]]:
    requested_payload_bits = int(requested_payload_bits)
    effective_payload_bits = int(effective_payload_bits)
    reasons: list[str] = []
    if requested_payload_bits >= 16:
        if structure_profile == "periodic":
            reasons.append("periodic_profile")
        if estimated_capacity < requested_payload_bits:
            reasons.append("capacity_estimate_below_requested")
        if reliable_carriers < V3_1_RELIABLE_CARRIER_FLOOR:
            reasons.append("reliable_carriers_below_floor")
    if reasons:
        return min(effective_payload_bits, V3_1_FALLBACK_PAYLOAD_BITS), True, reasons
    return effective_payload_bits, False, reasons


def _gaussian_edge_scores(target: torch.Tensor, clean_params: dict[str, torch.Tensor]) -> torch.Tensor:
    gray = (target[..., :3] * torch.tensor([0.299, 0.587, 0.114], device=target.device, dtype=target.dtype)).sum(dim=-1)
    dx = torch.zeros_like(gray)
    dy = torch.zeros_like(gray)
    dx[:, 1:] = torch.abs(gray[:, 1:] - gray[:, :-1])
    dy[1:, :] = torch.abs(gray[1:, :] - gray[:-1, :])
    edge = torch.sqrt(dx.pow(2) + dy.pow(2))
    h, w = gray.shape
    x = ((clean_params["offsets"][:, 0] + 1.0) * 0.5 * max(w - 1, 1)).round().to(torch.long).clamp(0, max(w - 1, 0))
    y = ((clean_params["offsets"][:, 1] + 1.0) * 0.5 * max(h - 1, 1)).round().to(torch.long).clamp(0, max(h - 1, 0))
    return edge[y, x]


def _dominant_periodic_frequency(gray: torch.Tensor) -> dict[str, float]:
    spectrum = torch.fft.rfft2(gray - gray.mean())
    power = spectrum.abs().pow(2)
    h, w = gray.shape
    fy = torch.fft.fftfreq(h, device=gray.device).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=gray.device).view(1, -1)
    radial = torch.sqrt(fx.abs().pow(2) + fy.abs().pow(2))
    band_mask = (radial > 0.08) & (radial < 0.30)
    if not bool(band_mask.any().item()):
        return {"fx": 0.0, "fy": 0.0, "power_ratio": 0.0}
    masked_power = power.masked_fill(~band_mask, -1.0)
    peak_index = int(torch.argmax(masked_power).item())
    row = peak_index // masked_power.shape[1]
    col = peak_index % masked_power.shape[1]
    total_power = float(power.sum().item() + 1e-6)
    return {
        "fx": float(fx[0, col].item()),
        "fy": float(fy[row, 0].item()),
        "power_ratio": float(power[row, col].item() / total_power),
    }


def _classify_structure_profile(
    target: torch.Tensor,
    clean_params: dict[str, torch.Tensor],
    *,
    requested: str,
) -> tuple[str, dict[str, float]]:
    gray = (target[..., :3] * torch.tensor([0.299, 0.587, 0.114], device=target.device, dtype=target.dtype)).sum(dim=-1)
    spectrum = torch.fft.rfft2(gray - gray.mean())
    power = spectrum.abs().pow(2)
    h, w = gray.shape
    fy = torch.fft.fftfreq(h, device=gray.device).abs().view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=gray.device).abs().view(1, -1)
    radial = torch.sqrt(fx.pow(2) + fy.pow(2))
    high_mask = radial > 0.20
    band_mask = (radial > 0.08) & (radial < 0.30)
    total_power = float(power.sum().item() + 1e-6)
    high_freq_ratio = float(power[high_mask].sum().item() / total_power)
    stripe_energy = float(power[band_mask].sum().item() / total_power)
    dominant_frequency = _dominant_periodic_frequency(gray)
    canonical = _canonicalize_geometry_local(clean_params)
    theta = canonical["theta_major"][:, 0]
    theta_concentration = float(torch.sqrt(torch.mean(torch.cos(2.0 * theta)) ** 2 + torch.mean(torch.sin(2.0 * theta)) ** 2).item())
    edge_scores = _gaussian_edge_scores(target, clean_params)
    edge_ratio = float((edge_scores >= edge_scores.quantile(0.75)).float().mean().item())

    if requested == "periodic":
        profile = "periodic"
        decision_rule = "requested_override_periodic"
    elif requested == "generic":
        profile = "generic"
        decision_rule = "requested_override_generic"
    else:
        aligned_periodic = stripe_energy >= 0.22 and theta_concentration >= 0.55 and edge_ratio >= 0.15
        dominant_power_ratio = float(dominant_frequency["power_ratio"])
        frequency_periodic = (
            stripe_energy >= 0.09
            and dominant_power_ratio >= 0.006
            and edge_ratio >= 0.20
            and (
                high_freq_ratio <= 0.06
                or (
                    stripe_energy >= 0.12
                    and dominant_power_ratio >= 0.006
                    and high_freq_ratio <= 0.20
                )
            )
        )
        if aligned_periodic:
            profile = "periodic"
            decision_rule = "aligned_periodic"
        elif frequency_periodic:
            profile = "periodic"
            decision_rule = "frequency_periodic"
        else:
            profile = "generic"
            decision_rule = "generic_fallback"
    return profile, {
        "stripe_energy": stripe_energy,
        "high_freq_ratio": high_freq_ratio,
        "theta_concentration": theta_concentration,
        "edge_ratio": edge_ratio,
        "dominant_frequency_fx": float(dominant_frequency["fx"]),
        "dominant_frequency_fy": float(dominant_frequency["fy"]),
        "dominant_frequency_power_ratio": float(dominant_frequency["power_ratio"]),
        "profile_decision_rule": decision_rule,
    }


def _periodic_edge_wet_mask(
    target: torch.Tensor,
    clean_params: dict[str, torch.Tensor],
    clean_importance: torch.Tensor,
    clean_sensitivity: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    gray = (target[..., :3] * torch.tensor([0.299, 0.587, 0.114], device=target.device, dtype=target.dtype)).sum(dim=-1)
    dominant_frequency = _dominant_periodic_frequency(gray)
    edge_scores = _gaussian_edge_scores(target, clean_params)
    edge_mask = edge_scores >= edge_scores.quantile(0.60)
    importance_mask = clean_importance >= clean_importance.quantile(0.60)

    canonical = _canonicalize_geometry_local(clean_params)
    theta = canonical["theta_major"][:, 0]
    dominant_theta = 0.5 * torch.atan2(torch.mean(torch.sin(2.0 * theta)), torch.mean(torch.cos(2.0 * theta)))
    theta_alignment = torch.cos(2.0 * (theta - dominant_theta))
    theta_alignment_mask = theta_alignment >= theta_alignment.quantile(0.65)

    h, w = gray.shape
    x = ((clean_params["offsets"][:, 0] + 1.0) * 0.5 * max(w - 1, 1)).to(clean_params["offsets"].dtype)
    y = ((clean_params["offsets"][:, 1] + 1.0) * 0.5 * max(h - 1, 1)).to(clean_params["offsets"].dtype)
    phase = x * dominant_frequency["fx"] + y * dominant_frequency["fy"]
    repetition_strength = torch.abs(torch.cos((2.0 * math.pi) * phase))
    if dominant_frequency["power_ratio"] > 0.01:
        repetition_mask = repetition_strength >= repetition_strength.quantile(0.60)
    else:
        repetition_mask = torch.zeros_like(edge_mask)

    if clean_sensitivity:
        log_sens = clean_sensitivity["log_anisotropy"].to(clean_importance.device)
        alpha_sens = clean_sensitivity.get("alpha", log_sens).to(clean_importance.device)
        sensitivity_score = 0.5 * log_sens + 0.5 * alpha_sens
        sensitivity_mask = sensitivity_score >= sensitivity_score.quantile(0.60)
    else:
        sensitivity_mask = torch.zeros_like(edge_mask)

    return (
        edge_mask
        | importance_mask
        | (repetition_mask & theta_alignment_mask)
        | (edge_mask & sensitivity_mask)
        | (repetition_mask & sensitivity_mask)
    )


def _target_luminance(target: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], device=target.device, dtype=target.dtype)
    return (target[..., :3] * weights).sum(dim=-1)


def _build_2d_region_policy(
    *,
    target: torch.Tensor,
    clean_params: dict[str, torch.Tensor],
    clean_importance: torch.Tensor,
    clean_sensitivity: dict[str, torch.Tensor],
    active_channels: list[str],
    structure_profile: str,
    profile_policy: str,
    disable_region_policy: bool,
    disable_coverage_balancing: bool,
    stripe_bin_rows: int = 6,
    stripe_bin_cols: int = 4,
) -> dict[str, object]:
    n = int(clean_params["offsets"].shape[0])
    generic_channels = [channel for channel in active_channels if channel != "theta"]
    region_policy = "legacy"
    gaussian_region_labels = ["generic_region"] * n
    gaussian_bin_ids = [f"bin_r0_c0" for _ in range(n)]
    region_allowed_channels = {"generic_region": list(generic_channels)}
    region_strength_scales = {
        "generic_region": {
            "log_anisotropy": 1.0,
            "alpha": 1.0,
            "color_lum": 1.0,
        }
    }
    dominant_frequency = _dominant_periodic_frequency(_target_luminance(target))
    stripe_normal = torch.tensor(
        [dominant_frequency["fx"], dominant_frequency["fy"]],
        dtype=clean_params["offsets"].dtype,
        device=clean_params["offsets"].device,
    )
    if float(torch.norm(stripe_normal).item()) < 1e-6:
        stripe_normal = torch.tensor([1.0, 0.0], dtype=clean_params["offsets"].dtype, device=clean_params["offsets"].device)
    stripe_normal = stripe_normal / stripe_normal.norm().clamp_min(1e-6)
    stripe_tangent = torch.stack((-stripe_normal[1], stripe_normal[0]))
    default_region_metrics = {
        "gaussian_local_periodicity_scores": [0.0] * n,
        "gaussian_repetition_strength": [0.0] * n,
        "gaussian_edge_norm": [0.0] * n,
        "gaussian_normalized_stripe_normal": [0.0] * n,
        "gaussian_normalized_stripe_tangent": [0.0] * n,
        "gaussian_normal_bin_ids": ["n0"] * n,
        "gaussian_tangent_bin_ids": ["t0"] * n,
    }

    if str(profile_policy) != "publishable_2d_v1":
        return {
            "region_policy": region_policy,
            "resolved_region_policy_variant": "legacy",
            "gaussian_region_labels": gaussian_region_labels,
            "gaussian_bin_ids": gaussian_bin_ids,
            "region_allowed_channels": region_allowed_channels,
            "region_strength_scales": region_strength_scales,
            "coverage_balancing": False,
            "minimum_block_bin_coverage": 1,
            "target_max_bin_share_cap": 1.0,
            "periodic_core_action_cap_ratio": 1.0,
            "stripe_normal": [float(value) for value in stripe_normal.detach().cpu().tolist()],
            "stripe_tangent": [float(value) for value in stripe_tangent.detach().cpu().tolist()],
            "stripe_bin_shape": [int(stripe_bin_rows), int(stripe_bin_cols)],
            **default_region_metrics,
            "stats": {
                "dominant_frequency_fx": float(dominant_frequency["fx"]),
                "dominant_frequency_fy": float(dominant_frequency["fy"]),
                "dominant_frequency_power_ratio": float(dominant_frequency["power_ratio"]),
            },
        }

    if str(structure_profile) != "periodic":
        region_policy = "publishable_2d_v1"
        gaussian_region_labels = ["generic_region"] * n
        offsets = clean_params["offsets"]
        offset_x = offsets[:, 0]
        offset_y = offsets[:, 1]
        x_min = float(offset_x.min().item())
        x_span = float((offset_x.max() - offset_x.min()).item()) + 1e-6
        y_min = float(offset_y.min().item())
        y_span = float((offset_y.max() - offset_y.min()).item()) + 1e-6
        gaussian_bin_ids = []
        gaussian_normal_bin_ids = []
        gaussian_tangent_bin_ids = []
        for x_value, y_value in zip(offset_x.tolist(), offset_y.tolist()):
            row = min(1, max(0, int(((y_value - y_min) / y_span) * 2.0)))
            col = min(1, max(0, int(((x_value - x_min) / x_span) * 2.0)))
            gaussian_bin_ids.append(f"bin_r{row}_c{col}")
            gaussian_normal_bin_ids.append(f"n{row}")
            gaussian_tangent_bin_ids.append(f"t{col}")
        generic_strengths = {channel: 1.0 for channel in generic_channels}
        return {
            "region_policy": region_policy,
            "resolved_region_policy_variant": (
                "publishable_no_region_generic"
                if disable_region_policy
                else "publishable_generic"
            ),
            "gaussian_region_labels": gaussian_region_labels,
            "gaussian_bin_ids": gaussian_bin_ids,
            "region_allowed_channels": {"generic_region": list(generic_channels)},
            "region_strength_scales": {"generic_region": dict(generic_strengths)},
            "coverage_balancing": not disable_coverage_balancing,
            "minimum_block_bin_coverage": 2,
            "target_max_bin_share_cap": 0.50,
            "periodic_core_action_cap_ratio": 1.0,
            "stripe_normal": [float(value) for value in stripe_normal.detach().cpu().tolist()],
            "stripe_tangent": [float(value) for value in stripe_tangent.detach().cpu().tolist()],
            "stripe_bin_shape": [2, 2],
            **default_region_metrics,
            "gaussian_normal_bin_ids": gaussian_normal_bin_ids,
            "gaussian_tangent_bin_ids": gaussian_tangent_bin_ids,
            "stats": {
                "dominant_frequency_fx": float(dominant_frequency["fx"]),
                "dominant_frequency_fy": float(dominant_frequency["fy"]),
                "dominant_frequency_power_ratio": float(dominant_frequency["power_ratio"]),
                "generic_region_fraction": 1.0,
            },
        }

    region_policy = "publishable_2d_v1"
    offsets = clean_params["offsets"]
    projected_normal = offsets @ stripe_normal
    projected_tangent = offsets @ stripe_tangent
    normal_min = float(projected_normal.min().item())
    normal_span = float((projected_normal.max() - projected_normal.min()).item()) + 1e-6
    tangent_min = float(projected_tangent.min().item())
    tangent_span = float((projected_tangent.max() - projected_tangent.min()).item()) + 1e-6
    normalized_stripe_normal = ((projected_normal - normal_min) / normal_span).clamp(0.0, 1.0)
    normalized_stripe_tangent = ((projected_tangent - tangent_min) / tangent_span).clamp(0.0, 1.0)
    gaussian_bin_ids = []
    gaussian_normal_bin_ids = []
    gaussian_tangent_bin_ids = []
    for normal_value, tangent_value in zip(projected_normal.tolist(), projected_tangent.tolist()):
        row = min(stripe_bin_rows - 1, max(0, int(((normal_value - normal_min) / normal_span) * stripe_bin_rows)))
        col = min(stripe_bin_cols - 1, max(0, int(((tangent_value - tangent_min) / tangent_span) * stripe_bin_cols)))
        gaussian_bin_ids.append(f"bin_r{row}_c{col}")
        gaussian_normal_bin_ids.append(f"n{row}")
        gaussian_tangent_bin_ids.append(f"t{col}")

    if disable_region_policy:
        generic_strengths = {channel: 1.0 for channel in generic_channels}
        return {
            "region_policy": region_policy,
            "resolved_region_policy_variant": "publishable_no_region_periodic",
            "gaussian_region_labels": ["generic_region"] * n,
            "gaussian_bin_ids": gaussian_bin_ids,
            "region_allowed_channels": {"generic_region": list(generic_channels)},
            "region_strength_scales": {"generic_region": dict(generic_strengths)},
            "coverage_balancing": not disable_coverage_balancing,
            "minimum_block_bin_coverage": 4,
            "target_max_bin_share_cap": 0.25,
            "periodic_core_action_cap_ratio": 1.0,
            "stripe_normal": [float(value) for value in stripe_normal.detach().cpu().tolist()],
            "stripe_tangent": [float(value) for value in stripe_tangent.detach().cpu().tolist()],
            "stripe_bin_shape": [int(stripe_bin_rows), int(stripe_bin_cols)],
            **default_region_metrics,
            "gaussian_normal_bin_ids": gaussian_normal_bin_ids,
            "gaussian_tangent_bin_ids": gaussian_tangent_bin_ids,
            "gaussian_normalized_stripe_normal": [float(value) for value in normalized_stripe_normal.detach().cpu().tolist()],
            "gaussian_normalized_stripe_tangent": [float(value) for value in normalized_stripe_tangent.detach().cpu().tolist()],
            "stats": {
                "dominant_frequency_fx": float(dominant_frequency["fx"]),
                "dominant_frequency_fy": float(dominant_frequency["fy"]),
                "dominant_frequency_power_ratio": float(dominant_frequency["power_ratio"]),
                "generic_region_fraction": 1.0,
            },
        }

    edge_scores = _gaussian_edge_scores(target, clean_params)
    edge_norm = edge_scores / edge_scores.max().clamp_min(1e-6)
    canonical = _canonicalize_geometry_local(clean_params)
    theta = canonical["theta_major"][:, 0]
    dominant_theta = torch.atan2(stripe_normal[1], stripe_normal[0])
    theta_alignment = 0.5 * (torch.cos(2.0 * (theta - dominant_theta)) + 1.0)
    phase = offsets[:, 0] * float(dominant_frequency["fx"]) + offsets[:, 1] * float(dominant_frequency["fy"])
    repetition_strength = torch.abs(torch.cos((2.0 * math.pi) * phase))
    local_periodicity_score = repetition_strength * theta_alignment * edge_norm
    core_threshold = float(torch.quantile(local_periodicity_score, 0.75).item())
    transition_threshold = float(torch.quantile(local_periodicity_score, 0.50).item())
    periodic_core = local_periodicity_score >= core_threshold
    transition_band = ((local_periodicity_score >= transition_threshold) | ((edge_norm >= 0.65) & (repetition_strength >= 0.45))) & ~periodic_core
    generic_region = ~(periodic_core | transition_band)
    gaussian_region_labels = []
    for index in range(n):
        if bool(periodic_core[index].item()):
            gaussian_region_labels.append("periodic_core")
        elif bool(transition_band[index].item()):
            gaussian_region_labels.append("transition_band")
        else:
            gaussian_region_labels.append("generic_region")

    region_allowed_channels = {
        "periodic_core": ["color_lum"],
        "transition_band": [channel for channel in ("alpha", "color_lum") if channel in generic_channels],
        "generic_region": list(generic_channels),
    }
    region_strength_scales = {
        "periodic_core": {"color_lum": 0.50},
        "transition_band": {"alpha": 0.50, "color_lum": 0.75},
        "generic_region": {"log_anisotropy": 1.0, "alpha": 1.0, "color_lum": 1.0},
    }
    return {
        "region_policy": region_policy,
        "resolved_region_policy_variant": "publishable_periodic",
        "gaussian_region_labels": gaussian_region_labels,
        "gaussian_bin_ids": gaussian_bin_ids,
        "gaussian_normal_bin_ids": gaussian_normal_bin_ids,
        "gaussian_tangent_bin_ids": gaussian_tangent_bin_ids,
        "region_allowed_channels": region_allowed_channels,
        "region_strength_scales": region_strength_scales,
        "coverage_balancing": not disable_coverage_balancing,
        "minimum_block_bin_coverage": 4,
        "target_max_bin_share_cap": 0.25,
        "periodic_core_action_cap_ratio": 0.20,
        "stripe_normal": [float(value) for value in stripe_normal.detach().cpu().tolist()],
        "stripe_tangent": [float(value) for value in stripe_tangent.detach().cpu().tolist()],
        "stripe_bin_shape": [int(stripe_bin_rows), int(stripe_bin_cols)],
        "gaussian_local_periodicity_scores": [float(value) for value in local_periodicity_score.detach().cpu().tolist()],
        "gaussian_repetition_strength": [float(value) for value in repetition_strength.detach().cpu().tolist()],
        "gaussian_edge_norm": [float(value) for value in edge_norm.detach().cpu().tolist()],
        "gaussian_normalized_stripe_normal": [float(value) for value in normalized_stripe_normal.detach().cpu().tolist()],
        "gaussian_normalized_stripe_tangent": [float(value) for value in normalized_stripe_tangent.detach().cpu().tolist()],
        "stats": {
            "dominant_frequency_fx": float(dominant_frequency["fx"]),
            "dominant_frequency_fy": float(dominant_frequency["fy"]),
            "dominant_frequency_power_ratio": float(dominant_frequency["power_ratio"]),
            "periodic_core_fraction": float(periodic_core.float().mean().item()),
            "transition_band_fraction": float(transition_band.float().mean().item()),
            "generic_region_fraction": float(generic_region.float().mean().item()),
        },
    }


def _summarize_assignment_regions(
    assignments: list[dict],
    *,
    target_max_bin_share_cap: float | None = None,
    periodic_core_action_cap_ratio: float | None = None,
) -> dict[str, object]:
    selected_bin_counts: dict[str, int] = {}
    zone_action_counts: dict[str, int] = {}
    total_selected = 0
    periodic_core_selected = 0
    for assignment in assignments:
        for action, selected in zip(assignment.get("candidate_actions", []), assignment.get("selected_mask", [])):
            if float(selected) <= 0.5:
                continue
            total_selected += 1
            bin_id = action.get("bin_id")
            if bin_id is not None:
                key = str(bin_id)
                selected_bin_counts[key] = selected_bin_counts.get(key, 0) + 1
            region_label = str(action.get("region_label", "generic_region"))
            zone_action_counts[region_label] = zone_action_counts.get(region_label, 0) + 1
            if region_label == "periodic_core":
                periodic_core_selected += 1
    max_share = 0.0
    if total_selected > 0 and selected_bin_counts:
        max_share = max(selected_bin_counts.values()) / float(total_selected)
    periodic_core_share = (
        float(periodic_core_selected) / float(total_selected)
        if total_selected > 0
        else 0.0
    )
    coverage_constraint_satisfied = True
    if target_max_bin_share_cap is not None:
        coverage_constraint_satisfied = float(max_share) <= float(target_max_bin_share_cap) + 1e-9
    periodic_core_constraint_satisfied = True
    if periodic_core_action_cap_ratio is not None:
        periodic_core_constraint_satisfied = float(periodic_core_share) <= float(periodic_core_action_cap_ratio) + 1e-9
    return {
        "selected_bin_counts": dict(sorted(selected_bin_counts.items())),
        "realized_selected_max_bin_share": float(max_share),
        "target_max_bin_share_cap": target_max_bin_share_cap,
        "realized_periodic_core_action_share": float(periodic_core_share),
        "coverage_constraint_satisfied": bool(
            coverage_constraint_satisfied and periodic_core_constraint_satisfied
        ),
        "periodic_core_constraint_satisfied": bool(periodic_core_constraint_satisfied),
        "periodic_core_action_cap_ratio": periodic_core_action_cap_ratio,
        "zone_action_counts": dict(sorted(zone_action_counts.items())),
    }


def _gaussian_blur_image(image: torch.Tensor, *, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return image
    if image.ndim != 3:
        raise ValueError(f"expected HxWxC image, got shape {tuple(image.shape)}")
    radius = max(1, int(math.ceil(2.0 * float(sigma))))
    coords = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (coords / float(sigma)).pow(2))
    kernel = kernel / kernel.sum().clamp_min(1e-6)
    channels = int(image.shape[-1])
    work = image.permute(2, 0, 1).unsqueeze(0)
    kernel_x = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    work = F.pad(work, (radius, radius, 0, 0), mode="reflect")
    work = F.conv2d(work, kernel_x, groups=channels)
    work = F.pad(work, (0, 0, radius, radius), mode="reflect")
    work = F.conv2d(work, kernel_y, groups=channels)
    return work.squeeze(0).permute(1, 2, 0)


def _mixed_distortion_phase_scale(*, current_iter: int, start_iter: int, end_iter: int) -> float:
    if current_iter < start_iter:
        return 0.0
    if current_iter < end_iter:
        return 0.5
    return 1.0


def _periodic_lowfreq_cost_scale_for_region(region_label: str, *, bias_strength: float) -> float:
    if region_label == "periodic_core":
        return 1.0 + float(bias_strength)
    if region_label == "transition_band":
        return max(0.60, 1.0 - 0.50 * float(bias_strength))
    if region_label == "generic_region":
        return max(0.75, 1.0 - 0.15 * float(bias_strength))
    return 1.0


def _build_periodic_lowfreq_carrier_scales(
    *,
    gaussian_region_labels: list[str],
    structure_profile: str,
    enabled: bool,
    bias_strength: float,
    bias_mode: str = "legacy",
    device: torch.device,
) -> torch.Tensor:
    scales = torch.ones(len(gaussian_region_labels), dtype=torch.float32, device=device)
    if not enabled or structure_profile != "periodic" or str(bias_mode) != "legacy":
        return scales
    for index, region_label in enumerate(gaussian_region_labels):
        scales[index] = _periodic_lowfreq_cost_scale_for_region(
            str(region_label),
            bias_strength=float(bias_strength),
        )
    return scales


def _normalize_periodic_bias_mode(mode: str | None) -> str:
    normalized = "legacy" if mode is None else str(mode).strip()
    aliases = {
        "transition_alpha_only_v1": "transition_alpha_only_v2",
    }
    normalized = aliases.get(normalized, normalized)
    valid = {
        "legacy",
        "transition_alpha_loganis_v1",
        "transition_alpha_loganis_strict_v1",
        "transition_alpha_phase_guard_v1",
        "transition_alpha_only_v2",
        "transition_alpha_dispersion_v1",
        "carrier_compensator_split_v1",
        "quantize_consistency_v1",
    }
    if normalized not in valid:
        raise ValueError(f"unsupported periodic bias mode: {mode!r}")
    return normalized


def _periodic_candidate_cost_scale_for_action(
    *,
    region_label: str,
    channel: str,
    bias_strength: float,
    bias_mode: str,
) -> float:
    resolved_mode = _normalize_periodic_bias_mode(bias_mode)
    if resolved_mode == "legacy":
        return _periodic_lowfreq_cost_scale_for_region(
            region_label,
            bias_strength=float(bias_strength),
        )
    if resolved_mode == "transition_alpha_loganis_strict_v1":
        lookup = {
            ("periodic_core", "color_lum"): 1.32,
            ("transition_band", "alpha"): 0.84,
            ("transition_band", "color_lum"): 1.08,
            ("generic_region", "log_anisotropy"): 0.90,
            ("generic_region", "alpha"): 1.00,
            ("generic_region", "color_lum"): 1.03,
        }
    elif resolved_mode in {
        "transition_alpha_loganis_v1",
        "transition_alpha_phase_guard_v1",
        "quantize_consistency_v1",
    }:
        lookup = {
            ("periodic_core", "color_lum"): 1.20,
            ("transition_band", "alpha"): 0.88,
            ("transition_band", "color_lum"): (
                0.98
                if resolved_mode != "transition_alpha_phase_guard_v1"
                else (0.98 * 1.06)
            ),
            ("generic_region", "log_anisotropy"): 0.94,
            ("generic_region", "alpha"): 1.00,
            ("generic_region", "color_lum"): 1.02,
        }
    else:
        lookup = {
            ("periodic_core", "color_lum"): 1.25,
            ("transition_band", "alpha"): 0.86,
            ("transition_band", "color_lum"): 1.00,
            ("generic_region", "log_anisotropy"): 1.00,
            ("generic_region", "alpha"): 1.00,
            ("generic_region", "color_lum"): 1.03,
        }
    default_scale = 1.0
    base_scale = float(lookup.get((str(region_label), str(channel)), default_scale))
    return max(1e-6, float(base_scale))


def _periodic_candidate_dispersion_scale(
    *,
    region_label: str,
    channel: str,
    normal_bin_id: str | None,
    tangent_bin_id: str | None,
    block_normal_bin_counts: dict[str, int] | None,
    block_tangent_bin_counts: dict[str, int] | None,
    bias_mode: str,
) -> float:
    resolved_mode = _normalize_periodic_bias_mode(bias_mode)
    if resolved_mode == "transition_alpha_phase_guard_v1":
        if str(region_label) != "transition_band" or str(channel) != "color_lum":
            return 1.0
        scale = 1.0
        tangent_occupancy = 0 if tangent_bin_id is None or not block_tangent_bin_counts else int(
            block_tangent_bin_counts.get(str(tangent_bin_id), 0)
        )
        normal_occupancy = 0 if normal_bin_id is None or not block_normal_bin_counts else int(
            block_normal_bin_counts.get(str(normal_bin_id), 0)
        )
        if tangent_occupancy > 0:
            scale *= 1.08
        if normal_occupancy > 0:
            scale *= 1.04
        return float(scale)
    if resolved_mode not in {
        "transition_alpha_dispersion_v1",
        "carrier_compensator_split_v1",
    }:
        return 1.0
    if normal_bin_id is None or not block_normal_bin_counts:
        return 1.0
    occupancy = int(block_normal_bin_counts.get(str(normal_bin_id), 0))
    if occupancy <= 0:
        return 1.0
    scale = 1.08
    if str(region_label) == "periodic_core":
        scale *= 1.12
    return float(scale)


def _build_periodic_carrier_cost_scale_fn(
    *,
    gaussian_region_labels: list[str],
    gaussian_normal_bin_ids: list[str],
    structure_profile: str,
    enabled: bool,
    bias_strength: float,
    bias_mode: str,
) -> Callable[[int, str, str, dict[str, object]], float] | None:
    resolved_mode = _normalize_periodic_bias_mode(bias_mode)
    if not enabled or structure_profile != "periodic" or resolved_mode == "legacy":
        return None

    def _carrier_cost_scale_fn(
        gaussian_index: int,
        region_label: str,
        channel: str,
        context: dict[str, object],
    ) -> float:
        fallback_region_label = "generic_region"
        if 0 <= int(gaussian_index) < len(gaussian_region_labels):
            fallback_region_label = str(gaussian_region_labels[int(gaussian_index)])
        normal_bin_id = context.get("normal_bin_id")
        tangent_bin_id = context.get("tangent_bin_id")
        if normal_bin_id is None and 0 <= int(gaussian_index) < len(gaussian_normal_bin_ids):
            normal_bin_id = gaussian_normal_bin_ids[int(gaussian_index)]
        base_scale = _periodic_candidate_cost_scale_for_action(
            region_label=str(region_label or fallback_region_label),
            channel=str(channel),
            bias_strength=float(bias_strength),
            bias_mode=resolved_mode,
        )
        dispersion_scale = _periodic_candidate_dispersion_scale(
            region_label=str(region_label or fallback_region_label),
            channel=str(channel),
            normal_bin_id=None if normal_bin_id is None else str(normal_bin_id),
            tangent_bin_id=None if tangent_bin_id is None else str(tangent_bin_id),
            block_normal_bin_counts=context.get("block_normal_bin_counts"),  # type: ignore[arg-type]
            block_tangent_bin_counts=context.get("block_tangent_bin_counts"),  # type: ignore[arg-type]
            bias_mode=resolved_mode,
        )
        return float(base_scale * dispersion_scale)

    return _carrier_cost_scale_fn


def _build_pack_selection_proxy_bundle(
    *,
    pack: dict[str, object],
    periodic_lowfreq_carrier_bias_enabled: bool,
    periodic_bias_strength: float,
) -> ChannelCostBundle:
    v3_bundle_local = pack["v3_bundle"]
    if not isinstance(v3_bundle_local, V3ChannelCostBundle):
        raise TypeError("pack must carry a V3ChannelCostBundle for proxy selection")
    region_policy_context = dict(pack.get("region_policy_context", {}))
    periodic_carrier_scales_local = pack.get("periodic_carrier_scales", [])
    gaussian_cost_scales = None
    if periodic_carrier_scales_local:
        gaussian_cost_scales = torch.as_tensor(
            periodic_carrier_scales_local,
            dtype=v3_bundle_local.log_rho_plus.dtype,
            device=v3_bundle_local.log_rho_plus.device,
        )
    periodic_bias_mode = str(pack.get("periodic_bias_mode", "legacy"))
    carrier_cost_scale_fn = _build_periodic_carrier_cost_scale_fn(
        gaussian_region_labels=list(region_policy_context.get("gaussian_region_labels", [])),
        gaussian_normal_bin_ids=list(region_policy_context.get("gaussian_normal_bin_ids", [])),
        structure_profile=str(pack.get("structure_profile", "generic")),
        enabled=bool(periodic_lowfreq_carrier_bias_enabled or periodic_bias_mode != "legacy"),
        bias_strength=float(periodic_bias_strength),
        bias_mode=periodic_bias_mode,
    )
    return _build_v3_selection_proxy_bundle(
        v3_bundle_local,
        active_channels=list(pack.get("active_channels", [])),
        structure_profile=str(pack.get("structure_profile", "generic")),
        gaussian_region_labels=list(region_policy_context.get("gaussian_region_labels", [])),
        gaussian_normal_bin_ids=list(region_policy_context.get("gaussian_normal_bin_ids", [])),
        gaussian_tangent_bin_ids=list(region_policy_context.get("gaussian_tangent_bin_ids", [])),
        region_allowed_channels=dict(region_policy_context.get("region_allowed_channels", {}) or {}),
        region_strength_scales=dict(region_policy_context.get("region_strength_scales", {}) or {}),
        gaussian_cost_scales=gaussian_cost_scales,
        carrier_cost_scale_fn=carrier_cost_scale_fn,
    )


def _periodic_bias_uses_split_compensators(bias_mode: str) -> bool:
    return _normalize_periodic_bias_mode(bias_mode) == "carrier_compensator_split_v1"


def _periodic_bias_uses_quantize_consistency(bias_mode: str) -> bool:
    return _normalize_periodic_bias_mode(bias_mode) == "quantize_consistency_v1"


def _periodic_bias_uses_phase_guard(bias_mode: str) -> bool:
    return _normalize_periodic_bias_mode(bias_mode) == "transition_alpha_phase_guard_v1"


def _normalize_publishable_stage_lock(requested_stage_lock: str | None) -> str:
    normalized = "auto" if requested_stage_lock is None else str(requested_stage_lock).strip()
    valid = {"auto", "P1", "P2", "G0", "G1", "G2"}
    if normalized not in valid:
        raise ValueError(f"unsupported publishable stage lock: {requested_stage_lock!r}")
    return normalized


PUBLISHABLE_WRONG_KEY_TARGET = 0.5
PUBLISHABLE_WRONG_KEY_TOLERANCE = 0.15
PUBLISHABLE_REPORT_WRONG_KEY_TOLERANCE = 0.10


def _safe_float(value: object, *, default: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def _wrong_key_chance_gap(value: object, *, target: float = PUBLISHABLE_WRONG_KEY_TARGET) -> float:
    wrong_key_value = _safe_float(value, default=float("inf"))
    return abs(wrong_key_value - float(target))


def _wrong_key_near_chance(
    value: object,
    *,
    tolerance: float = PUBLISHABLE_WRONG_KEY_TOLERANCE,
) -> bool:
    return bool(_wrong_key_chance_gap(value) <= float(tolerance) + 1e-9)


def _select_publishable_stage_result(
    stage_results: list[dict[str, object]],
    *,
    requested_stage_lock: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if not stage_results:
        raise RuntimeError("no_viable_policy_assignments")
    auto_result = min(
        stage_results,
        key=lambda result: _publishable_stage_rank(
            coverage_constraint_satisfied=bool(result["coverage_constraint_satisfied"]),
            clean_ber=result.get("proxy_clean_ber"),
            clean_psnr_drop=result.get("proxy_clean_psnr_drop"),
            wrong_key_mean_ber=result.get("proxy_wrong_key_mean_ber"),
            resolved_log_delta=float(result["resolved_log_delta"]),
        ),
    )
    if requested_stage_lock == "auto":
        return auto_result, auto_result
    locked_result = next(
        (result for result in stage_results if str(result.get("stage_name")) == requested_stage_lock),
        None,
    )
    if locked_result is None:
        available = ", ".join(str(result.get("stage_name")) for result in stage_results)
        raise ValueError(
            f"requested publishable stage lock {requested_stage_lock!r} is unavailable; "
            f"available stages: {available}"
        )
    return locked_result, auto_result


def _build_assignment_audit(
    *,
    assignments: list[dict],
    source_policy: dict[str, object],
    compensation_actions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    zone_action_counts: dict[str, int] = {}
    selected_region_channel_counts: dict[str, int] = {}
    carrier_region_channel_counts: dict[str, int] = {}
    compensator_region_channel_counts: dict[str, int] = {}
    selected_normal_bin_counts: dict[str, int] = {}
    selected_tangent_bin_counts: dict[str, int] = {}
    selected_scales: list[float] = []
    selected_action_count = 0
    selected_carrier_actions: list[dict[str, object]] = []

    def _record_action(
        *,
        action: dict[str, object],
        region_channel_counts: dict[str, int],
        include_scale: bool,
    ) -> None:
        nonlocal selected_action_count
        selected_action_count += 1
        region_label = str(action.get("region_label", "generic_region"))
        channel = str(action.get("channel", "unknown"))
        zone_action_counts[region_label] = zone_action_counts.get(region_label, 0) + 1
        region_channel_key = f"{region_label}:{channel}"
        selected_region_channel_counts[region_channel_key] = (
            selected_region_channel_counts.get(region_channel_key, 0) + 1
        )
        region_channel_counts[region_channel_key] = (
            region_channel_counts.get(region_channel_key, 0) + 1
        )
        normal_bin_id = action.get("normal_bin_id")
        tangent_bin_id = action.get("tangent_bin_id")
        if normal_bin_id is not None:
            key = str(normal_bin_id)
            selected_normal_bin_counts[key] = selected_normal_bin_counts.get(key, 0) + 1
        if tangent_bin_id is not None:
            key = str(tangent_bin_id)
            selected_tangent_bin_counts[key] = selected_tangent_bin_counts.get(key, 0) + 1
        if include_scale:
            scale = action.get("carrier_cost_scale")
            if scale is not None:
                selected_scales.append(float(scale))
            selected_carrier_actions.append(dict(action))

    for assignment in assignments:
        for action, selected in zip(assignment.get("candidate_actions", []), assignment.get("selected_mask", [])):
            if float(selected) <= 0.5:
                continue
            _record_action(
                action=dict(action),
                region_channel_counts=carrier_region_channel_counts,
                include_scale=True,
            )
    for action in compensation_actions or []:
        _record_action(
            action=dict(action),
            region_channel_counts=compensator_region_channel_counts,
            include_scale=False,
        )
    periodic_bias_mode = str(source_policy.get("periodic_bias_mode", "legacy"))
    phase_guard_mode = "inactive"
    phase_guard_penalty_histogram: dict[str, int] = {}
    phase_guard_penalty_count = 0
    if _periodic_bias_uses_phase_guard(periodic_bias_mode):
        phase_guard_mode = "transition_alpha_phase_guard_v1"
        for action in selected_carrier_actions:
            if str(action.get("region_label", "")) != "transition_band":
                continue
            if str(action.get("channel", "")) != "color_lum":
                continue
            phase_guard_penalty_histogram["base_transition_band_color_lum"] = (
                phase_guard_penalty_histogram.get("base_transition_band_color_lum", 0) + 1
            )
            tangent_bin_id = action.get("tangent_bin_id")
            if tangent_bin_id is not None and selected_tangent_bin_counts.get(str(tangent_bin_id), 0) > 1:
                phase_guard_penalty_histogram["same_tangent_bin"] = (
                    phase_guard_penalty_histogram.get("same_tangent_bin", 0) + 1
                )
            normal_bin_id = action.get("normal_bin_id")
            if normal_bin_id is not None and selected_normal_bin_counts.get(str(normal_bin_id), 0) > 1:
                phase_guard_penalty_histogram["same_normal_bin"] = (
                    phase_guard_penalty_histogram.get("same_normal_bin", 0) + 1
                )
        phase_guard_penalty_count = int(sum(phase_guard_penalty_histogram.values()))

    return {
        "zone_action_counts": dict(sorted(zone_action_counts.items())),
        "selected_region_channel_counts": dict(sorted(selected_region_channel_counts.items())),
        "selected_action_count": int(selected_action_count),
        "selected_normal_bin_counts": dict(sorted(selected_normal_bin_counts.items())),
        "selected_tangent_bin_counts": dict(sorted(selected_tangent_bin_counts.items())),
        "carrier_action_count": int(sum(carrier_region_channel_counts.values())),
        "compensator_action_count": int(sum(compensator_region_channel_counts.values())),
        "carrier_region_channel_counts": dict(sorted(carrier_region_channel_counts.items())),
        "compensator_region_channel_counts": dict(sorted(compensator_region_channel_counts.items())),
        "mean_selected_carrier_cost_scale": (
            None if not selected_scales else float(sum(selected_scales) / len(selected_scales))
        ),
        "max_selected_carrier_cost_scale": (
            None if not selected_scales else float(max(selected_scales))
        ),
        "requested_publishable_stage_lock": str(
            source_policy.get("requested_publishable_stage_lock", "auto")
        ),
        "selected_publishable_stage": source_policy.get("selected_publishable_stage"),
        "auto_selected_publishable_stage": source_policy.get("auto_selected_publishable_stage"),
        "publishable_stage_lock_applied": bool(
            source_policy.get("publishable_stage_lock_applied", False)
        ),
        "phase_guard_mode": phase_guard_mode,
        "phase_guard_penalty_count": int(phase_guard_penalty_count),
        "phase_guard_penalty_histogram": dict(sorted(phase_guard_penalty_histogram.items())),
    }


def _bin_id_to_index(bin_id: str | None, *, prefix: str) -> int | None:
    if bin_id is None:
        return None
    value = str(bin_id)
    if not value.startswith(prefix):
        return None
    try:
        return int(value[len(prefix):])
    except ValueError:
        return None


def _select_periodic_compensator_actions(
    offsets: torch.Tensor,
    *,
    assignments_local: list[dict],
    wet_mask: torch.Tensor,
    region_policy_context: dict[str, object],
    max_per_carrier: int = 1,
) -> list[dict[str, object]]:
    if max_per_carrier <= 0:
        return []
    gaussian_region_labels = list(region_policy_context.get("gaussian_region_labels", []))
    gaussian_normal_bin_ids = list(region_policy_context.get("gaussian_normal_bin_ids", []))
    gaussian_tangent_bin_ids = list(region_policy_context.get("gaussian_tangent_bin_ids", []))
    selected_carrier_actions = [
        dict(action)
        for assignment in assignments_local
        for action, selected in zip(assignment.get("candidate_actions", []), assignment.get("selected_mask", []))
        if float(selected) > 0.5
    ]
    if not selected_carrier_actions:
        return []
    selected_carrier_indices = {int(action["gaussian_index"]) for action in selected_carrier_actions}
    used_compensator_indices: set[int] = set()
    compensation_actions: list[dict[str, object]] = []
    offsets_cpu = offsets.detach().cpu()
    for carrier_action in selected_carrier_actions:
        carrier_index = int(carrier_action["gaussian_index"])
        carrier_normal_bin_id = carrier_action.get("normal_bin_id")
        if carrier_normal_bin_id is None and 0 <= carrier_index < len(gaussian_normal_bin_ids):
            carrier_normal_bin_id = gaussian_normal_bin_ids[carrier_index]
        carrier_tangent_bin_id = carrier_action.get("tangent_bin_id")
        if carrier_tangent_bin_id is None and 0 <= carrier_index < len(gaussian_tangent_bin_ids):
            carrier_tangent_bin_id = gaussian_tangent_bin_ids[carrier_index]
        carrier_normal_idx = _bin_id_to_index(
            None if carrier_normal_bin_id is None else str(carrier_normal_bin_id),
            prefix="n",
        )
        carrier_tangent_idx = _bin_id_to_index(
            None if carrier_tangent_bin_id is None else str(carrier_tangent_bin_id),
            prefix="t",
        )
        eligible: list[tuple[object, ...]] = []
        for candidate_index, region_label in enumerate(gaussian_region_labels):
            if candidate_index in selected_carrier_indices or candidate_index in used_compensator_indices:
                continue
            if bool(wet_mask[candidate_index].item()):
                continue
            if str(region_label) not in {"transition_band", "generic_region"}:
                continue
            normal_bin_id = (
                gaussian_normal_bin_ids[candidate_index]
                if candidate_index < len(gaussian_normal_bin_ids)
                else None
            )
            tangent_bin_id = (
                gaussian_tangent_bin_ids[candidate_index]
                if candidate_index < len(gaussian_tangent_bin_ids)
                else None
            )
            normal_idx = _bin_id_to_index(
                None if normal_bin_id is None else str(normal_bin_id),
                prefix="n",
            )
            tangent_idx = _bin_id_to_index(
                None if tangent_bin_id is None else str(tangent_bin_id),
                prefix="t",
            )
            if (
                carrier_normal_idx is not None
                and normal_idx is not None
                and abs(int(normal_idx) - int(carrier_normal_idx)) > 1
            ):
                continue
            if (
                carrier_tangent_idx is not None
                and tangent_idx is not None
                and int(tangent_idx) == int(carrier_tangent_idx)
            ):
                continue
            distance = float(
                torch.norm(offsets_cpu[candidate_index] - offsets_cpu[carrier_index], p=2).item()
            )
            region_priority = 0 if str(region_label) == "transition_band" else 1
            normal_gap = (
                0
                if carrier_normal_idx is None or normal_idx is None
                else abs(int(normal_idx) - int(carrier_normal_idx))
            )
            eligible.append(
                (
                    int(region_priority),
                    int(normal_gap),
                    float(distance),
                    int(candidate_index),
                    str(region_label),
                    normal_bin_id,
                    tangent_bin_id,
                )
            )
        if not eligible:
            continue
        eligible.sort()
        for rank, (_, _, _, candidate_index, region_label, normal_bin_id, tangent_bin_id) in enumerate(eligible):
            if rank >= max_per_carrier:
                break
            used_compensator_indices.add(int(candidate_index))
            compensation_actions.append(
                {
                    "gaussian_index": int(candidate_index),
                    "region_label": str(region_label),
                    "channel": "alpha",
                    "normal_bin_id": normal_bin_id,
                    "tangent_bin_id": tangent_bin_id,
                    "anchor_gaussian_index": int(carrier_index),
                    "anchor_channel": str(carrier_action.get("channel", "unknown")),
                    "role": "compensator",
                }
            )
    return compensation_actions


def _build_compensation_plan(
    offsets: torch.Tensor,
    *,
    assignments_local: list[dict],
    wet_mask: torch.Tensor,
    region_policy_context: dict[str, object],
    structure_profile: str,
    periodic_bias_mode: str,
    periodic_extra_compensation_per_active: int,
) -> tuple[torch.Tensor, list[dict[str, object]], bool]:
    resolved_mode = _normalize_periodic_bias_mode(periodic_bias_mode)
    if structure_profile == "periodic" and _periodic_bias_uses_split_compensators(resolved_mode):
        compensation_actions = _select_periodic_compensator_actions(
            offsets,
            assignments_local=assignments_local,
            wet_mask=wet_mask,
            region_policy_context=region_policy_context,
            max_per_carrier=1,
        )
        if not compensation_actions:
            return torch.empty(0, dtype=torch.long, device=offsets.device), [], True
        compensation_indices = torch.tensor(
            sorted({int(action["gaussian_index"]) for action in compensation_actions}),
            dtype=torch.long,
            device=offsets.device,
        )
        return compensation_indices, compensation_actions, True

    compensation_indices = select_compensation_neighbors(
        offsets,
        assignments=_v2_assignments_for_compensation(assignments_local),
        wet_mask=wet_mask,
        per_active=2,
    ).to(offsets.device)
    if structure_profile == "periodic" and periodic_extra_compensation_per_active > 0:
        extra_compensation_indices = _periodic_extra_compensation_indices(
            offsets,
            assignments_local=assignments_local,
            wet_mask=wet_mask,
            gaussian_region_labels=list(region_policy_context.get("gaussian_region_labels", [])),
            extra_per_active=int(periodic_extra_compensation_per_active),
        ).to(offsets.device)
        if extra_compensation_indices.numel() > 0:
            compensation_indices = torch.unique(torch.cat((compensation_indices, extra_compensation_indices)))
    return compensation_indices, [], False


def _selected_gaussian_indices(assignments_local: list[dict]) -> torch.Tensor:
    selected_indices = sorted(
        {
            int(action["gaussian_index"])
            for assignment in assignments_local
            for action, mask in zip(assignment.get("candidate_actions", []), assignment.get("selected_mask", []))
            if float(mask) > 0.5
        }
    )
    if not selected_indices:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(selected_indices, dtype=torch.long)


def _periodic_extra_compensation_indices(
    offsets: torch.Tensor,
    *,
    assignments_local: list[dict],
    wet_mask: torch.Tensor,
    gaussian_region_labels: list[str],
    extra_per_active: int,
) -> torch.Tensor:
    if extra_per_active <= 0:
        return torch.empty(0, dtype=torch.long)
    active_indices = _selected_gaussian_indices(assignments_local)
    if active_indices.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    active_set = {int(index) for index in active_indices.tolist()}
    preferred_candidates = [
        index
        for index, region_label in enumerate(gaussian_region_labels)
        if str(region_label) in {"periodic_core", "transition_band"}
        and not bool(wet_mask[index].item())
        and index not in active_set
    ]
    if not preferred_candidates:
        return torch.empty(0, dtype=torch.long)
    candidate_tensor = torch.tensor(sorted(preferred_candidates), dtype=torch.long)
    distances = torch.cdist(
        offsets[active_indices].detach().cpu(),
        offsets[candidate_tensor].detach().cpu(),
    )
    region_priority = [
        0 if str(gaussian_region_labels[int(index)]) == "periodic_core" else 1
        for index in candidate_tensor.tolist()
    ]
    neighbors: set[int] = set()
    for row in range(distances.shape[0]):
        ranked = sorted(
            range(distances.shape[1]),
            key=lambda column: (
                float(distances[row, column].item()),
                int(region_priority[column]),
            ),
        )
        for column in ranked[: min(int(extra_per_active), len(ranked))]:
            neighbors.add(int(candidate_tensor[column].item()))
    if not neighbors:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(sorted(neighbors), dtype=torch.long)


def _haar_detail_bands(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    luminance = _target_luminance(image)
    even_even = luminance[0::2, 0::2]
    even_odd = luminance[0::2, 1::2]
    odd_even = luminance[1::2, 0::2]
    odd_odd = luminance[1::2, 1::2]
    lh = 0.5 * (even_even - even_odd + odd_even - odd_odd)
    hl = 0.5 * (even_even + even_odd - odd_even - odd_odd)
    hh = 0.5 * (even_even - even_odd - odd_even + odd_odd)
    return lh, hl, hh


def _wavelet_band_consistency_loss(
    clean_render: torch.Tensor,
    watermarked_render: torch.Tensor,
) -> torch.Tensor:
    clean_lh, clean_hl, clean_hh = _haar_detail_bands(clean_render)
    wm_lh, wm_hl, wm_hh = _haar_detail_bands(watermarked_render)
    return (
        F.l1_loss(wm_lh, clean_lh)
        + F.l1_loss(wm_hl, clean_hl)
        + F.l1_loss(wm_hh, clean_hh)
    ) / 3.0


def _stripe_luminance_smoothness_loss(
    clean_render: torch.Tensor,
    watermarked_render: torch.Tensor,
    *,
    stripe_normal: list[float] | tuple[float, float],
    bins: int = 32,
) -> torch.Tensor:
    delta = _target_luminance(watermarked_render) - _target_luminance(clean_render)
    height, width = delta.shape
    xs = torch.linspace(-1.0, 1.0, width, device=delta.device, dtype=delta.dtype)
    ys = torch.linspace(-1.0, 1.0, height, device=delta.device, dtype=delta.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    normal = torch.tensor(stripe_normal, device=delta.device, dtype=delta.dtype)
    if float(torch.norm(normal).item()) < 1e-6:
        return delta.new_zeros(())
    normal = normal / normal.norm().clamp_min(1e-6)
    coord = grid_x * normal[0] + grid_y * normal[1]
    coord_min = coord.min()
    coord_span = (coord.max() - coord.min()).clamp_min(1e-6)
    coord_index = ((coord - coord_min) / coord_span * float(max(bins - 1, 1))).round().to(torch.long)
    profile = torch.zeros(bins, device=delta.device, dtype=delta.dtype)
    counts = torch.zeros(bins, device=delta.device, dtype=delta.dtype)
    profile.scatter_add_(0, coord_index.reshape(-1), delta.reshape(-1))
    counts.scatter_add_(0, coord_index.reshape(-1), torch.ones_like(delta).reshape(-1))
    profile = profile / counts.clamp_min(1.0)
    if profile.numel() < 3:
        return delta.new_zeros(())
    second_diff = profile[2:] - 2.0 * profile[1:-1] + profile[:-2]
    return second_diff.abs().mean()


def _default_wavelet_band_loss_weight(
    *,
    structure_profile: str,
    profile_policy: str,
    explicit_weight: float | None = None,
) -> float:
    if explicit_weight is not None:
        return float(explicit_weight)
    if profile_policy == "publishable_2d_v1" and structure_profile == "periodic":
        return 0.20
    if profile_policy == "publishable_2d_v1":
        return 0.05
    return 0.0


def _default_stripe_luminance_loss_weight(
    *,
    structure_profile: str,
    profile_policy: str,
    explicit_weight: float | None = None,
) -> float:
    if explicit_weight is not None:
        return float(explicit_weight)
    if profile_policy == "publishable_2d_v1" and structure_profile == "periodic":
        return 0.12
    if profile_policy == "publishable_2d_v1":
        return 0.03
    return 0.0


def _render_grid_from_target(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = target.shape[:2]
    xs = torch.linspace(-1.0, 1.0, width, device=target.device, dtype=target.dtype)
    ys = torch.linspace(-1.0, 1.0, height, device=target.device, dtype=target.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def _memory_safe_render_chunk_size(num_gaussians: int, grid_x: torch.Tensor) -> int | None:
    """Bound temporary renderer memory for dense, high-resolution fits."""
    if int(num_gaussians) * int(grid_x.numel()) >= 64_000_000:
        return 256
    return None


def _render_from_physical_params(
    params: dict[str, torch.Tensor],
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
) -> torch.Tensor:
    offsets = params["offsets"]
    scales = params["scales"].clamp_min(1e-6)
    thetas = params["thetas"]
    colors = params["colors"].clamp(0.0, 1.0)
    opacities = params["opacities"].clamp(0.0, 1.0)
    cos_theta = torch.cos(thetas[:, 0])
    sin_theta = torch.sin(thetas[:, 0])
    inv_sx2 = 1.0 / (scales[:, 0] * scales[:, 0])
    inv_sy2 = 1.0 / (scales[:, 1] * scales[:, 1])
    chunk_size = _memory_safe_render_chunk_size(int(offsets.shape[0]), grid_x)
    if chunk_size is None:
        dx = grid_x.unsqueeze(0) - offsets[:, 0].view(-1, 1, 1)
        dy = grid_y.unsqueeze(0) - offsets[:, 1].view(-1, 1, 1)
        local_x = cos_theta.view(-1, 1, 1) * dx + sin_theta.view(-1, 1, 1) * dy
        local_y = -sin_theta.view(-1, 1, 1) * dx + cos_theta.view(-1, 1, 1) * dy
        quadratic = inv_sx2.view(-1, 1, 1) * local_x.pow(2) + inv_sy2.view(-1, 1, 1) * local_y.pow(2)
        gaussian_response = torch.exp(-0.5 * quadratic)
        weights = gaussian_response * opacities[:, 0].view(-1, 1, 1)
        return torch.sum(weights.unsqueeze(-1) * colors.view(-1, 1, 1, 3), dim=0)

    rendered = torch.zeros((*grid_x.shape, 3), dtype=grid_x.dtype, device=grid_x.device)
    for start in range(0, int(offsets.shape[0]), chunk_size):
        end = min(start + chunk_size, int(offsets.shape[0]))
        dx = grid_x.unsqueeze(0) - offsets[start:end, 0].view(-1, 1, 1)
        dy = grid_y.unsqueeze(0) - offsets[start:end, 1].view(-1, 1, 1)
        local_x = cos_theta[start:end].view(-1, 1, 1) * dx + sin_theta[start:end].view(-1, 1, 1) * dy
        local_y = -sin_theta[start:end].view(-1, 1, 1) * dx + cos_theta[start:end].view(-1, 1, 1) * dy
        quadratic = (
            inv_sx2[start:end].view(-1, 1, 1) * local_x.pow(2)
            + inv_sy2[start:end].view(-1, 1, 1) * local_y.pow(2)
        )
        gaussian_response = torch.exp(-0.5 * quadratic)
        weights = gaussian_response * opacities[start:end, 0].view(-1, 1, 1)
        rendered = rendered + torch.sum(
            weights.unsqueeze(-1) * colors[start:end].view(-1, 1, 1, 3),
            dim=0,
        )
    return rendered


def _apply_v3_selected_actions_to_params(
    clean_params: dict[str, torch.Tensor],
    assignments: list[dict],
    *,
    log_delta: float,
    alpha_delta: float,
    theta_delta: float,
    lum_delta: float,
) -> dict[str, torch.Tensor]:
    params = {key: value.clone() for key, value in clean_params.items()}
    luma_w = torch.tensor([0.299, 0.587, 0.114], device=params["colors"].device, dtype=params["colors"].dtype)
    luma_denom = float((luma_w * luma_w).sum().item())
    for assignment in assignments:
        for action, selected in zip(assignment.get("candidate_actions", []), assignment.get("selected_mask", [])):
            if float(selected) <= 0.5:
                continue
            idx = int(action["gaussian_index"])
            direction = float(action.get("direction", 1.0))
            strength_scale = float(action.get("strength_scale", 1.0))
            channel = str(action.get("channel", "log_anisotropy"))
            if channel == "log_anisotropy":
                sx = float(params["scales"][idx, 0].item())
                sy = float(params["scales"][idx, 1].item())
                delta = math.exp(direction * log_delta * strength_scale * 0.5)
                inv_delta = math.exp(-direction * log_delta * strength_scale * 0.5)
                if sx >= sy:
                    params["scales"][idx, 0] = params["scales"][idx, 0] * delta
                    params["scales"][idx, 1] = params["scales"][idx, 1] * inv_delta
                else:
                    params["scales"][idx, 0] = params["scales"][idx, 0] * inv_delta
                    params["scales"][idx, 1] = params["scales"][idx, 1] * delta
            elif channel == "alpha":
                params["opacities"][idx, 0] = torch.clamp(
                    params["opacities"][idx, 0] + direction * alpha_delta * strength_scale,
                    0.0,
                    1.0,
                )
            elif channel == "theta":
                params["thetas"][idx, 0] = params["thetas"][idx, 0] + direction * theta_delta * strength_scale
            elif channel == "color_lum":
                delta_rgb = direction * lum_delta * strength_scale * luma_w / max(luma_denom, 1e-6)
                params["colors"][idx] = torch.clamp(params["colors"][idx] + delta_rgb, 0.0, 1.0)
    return params


def _pilot_retry_config(
    *,
    profile: str,
    selection_temperature: float,
    max_modification_ratio: float,
    keep_fraction: float,
    active_channels: list[str],
    allow_profile_change: bool = True,
) -> dict[str, object]:
    if profile == "periodic":
        return {
            "profile": "periodic",
            "active_channels": [channel for channel in active_channels if channel != "theta"],
            "selection_temperature": min(selection_temperature, 0.30),
            "max_modification_ratio": max_modification_ratio * 0.8,
            "keep_fraction": min(keep_fraction, 0.55),
        }
    if not allow_profile_change:
        return {
            "profile": "generic",
            "active_channels": list(active_channels),
            "selection_temperature": min(selection_temperature, 0.40),
            "max_modification_ratio": max_modification_ratio * 0.85,
            "keep_fraction": min(keep_fraction, 0.85),
        }
    return {
        "profile": "periodic",
        "active_channels": [channel for channel in active_channels if channel != "theta"],
        "selection_temperature": min(selection_temperature, 0.35),
        "max_modification_ratio": max_modification_ratio * 0.8,
        "keep_fraction": min(keep_fraction, 0.70),
    }


def _pilot_clean_ber(
    *,
    clean_params: dict[str, torch.Tensor],
    assignments: list[dict],
    channel_weights: dict[str, float],
    raw_bits: torch.Tensor,
    log_delta: float,
    alpha_delta: float,
    theta_delta: float,
    lum_delta: float,
) -> float:
    pilot_params = _apply_v3_selected_actions_to_params(
        clean_params,
        assignments,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
        theta_delta=theta_delta,
        lum_delta=lum_delta,
    )
    decoded_raw_bits, _, _ = decode_cost_code_v3(
        pilot_params,
        clean_params=clean_params,
        assignments=assignments,
        raw_length=int(raw_bits.numel()),
        channel_weights=channel_weights,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
        theta_delta=theta_delta,
        lum_delta=lum_delta,
        beta=4.0,
    )
    return float((decoded_raw_bits != raw_bits.to(decoded_raw_bits.device)).float().mean().item())


def _pilot_clean_psnr_drop(
    *,
    target: torch.Tensor,
    clean_params: dict[str, torch.Tensor],
    assignments: list[dict],
    log_delta: float,
    alpha_delta: float,
    theta_delta: float,
    lum_delta: float,
) -> float:
    pilot_params = _apply_v3_selected_actions_to_params(
        clean_params,
        assignments,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
        theta_delta=theta_delta,
        lum_delta=lum_delta,
    )
    grid_x, grid_y = _render_grid_from_target(target)
    clean_render = _render_from_physical_params(clean_params, grid_x, grid_y).clamp(0.0, 1.0)
    pilot_render = _render_from_physical_params(pilot_params, grid_x, grid_y).clamp(0.0, 1.0)
    clean_psnr = compute_psnr(clean_render, target)
    pilot_psnr = compute_psnr(pilot_render, target)
    clean_psnr_value = float(clean_psnr.item() if hasattr(clean_psnr, "item") else clean_psnr)
    pilot_psnr_value = float(pilot_psnr.item() if hasattr(pilot_psnr, "item") else pilot_psnr)
    return float(max(0.0, clean_psnr_value - pilot_psnr_value))


def _pilot_candidate_is_publishable(
    *,
    pilot_clean_ber: float,
    pilot_clean_psnr_drop: float,
) -> bool:
    try:
        clean_ber = float(pilot_clean_ber)
        clean_psnr_drop = float(pilot_clean_psnr_drop)
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(clean_ber)
        and math.isfinite(clean_psnr_drop)
        and clean_ber <= 0.25 + 1e-9
        and clean_psnr_drop <= 3.0 + 1e-9
    )


def _load_renderer_from_physical_params(
    renderer: Gaussian2DRenderer,
    params: dict[str, torch.Tensor],
) -> None:
    renderer.set_physical_parameters(
        offsets=params["offsets"],
        scales=params["scales"],
        thetas=params["thetas"],
        colors=params["colors"],
        opacities=params["opacities"],
    )


def _policy_knobs_for_profile(
    *,
    base_profile: str,
    selection_temperature: float,
    max_modification_ratio: float,
) -> dict[str, float]:
    keep_fraction = 1.0
    resolved_selection_temperature = float(selection_temperature)
    resolved_max_modification_ratio = float(max_modification_ratio)
    if base_profile == "periodic":
        resolved_selection_temperature = min(resolved_selection_temperature, 0.35)
        resolved_max_modification_ratio = resolved_max_modification_ratio * 0.8
        keep_fraction = 0.70
    return {
        "selection_temperature": float(resolved_selection_temperature),
        "max_modification_ratio": float(resolved_max_modification_ratio),
        "keep_fraction": float(keep_fraction),
    }


def _source_policy_signature(policy: dict[str, object]) -> str:
    payload = {
        "policy_resolution_mode": str(policy.get("policy_resolution_mode", "baseline_local")),
        "resolved_structure_profile": str(policy.get("resolved_structure_profile", "generic")),
        "resolved_region_policy": str(policy.get("resolved_region_policy", "legacy")),
        "resolved_region_policy_variant": str(policy.get("resolved_region_policy_variant", "legacy")),
        "resolved_active_channels": list(policy.get("resolved_active_channels", [])),
        "resolved_selection_temperature": float(policy.get("resolved_selection_temperature", 0.0) or 0.0),
        "resolved_max_modification_ratio": float(policy.get("resolved_max_modification_ratio", 0.0) or 0.0),
        "resolved_keep_fraction": float(policy.get("resolved_keep_fraction", 1.0) or 1.0),
        "resolved_log_delta": float(policy.get("resolved_log_delta", 0.0) or 0.0),
        "resolved_alpha_delta": float(policy.get("resolved_alpha_delta", 0.0) or 0.0),
        "resolved_lum_delta": float(policy.get("resolved_lum_delta", 0.0) or 0.0),
        "resolved_wavelet_band_loss_weight": float(policy.get("resolved_wavelet_band_loss_weight", 0.0) or 0.0),
        "resolved_stripe_luminance_loss_weight": float(
            policy.get("resolved_stripe_luminance_loss_weight", 0.0) or 0.0
        ),
        "selected_publishable_stage": policy.get("selected_publishable_stage"),
        "requested_publishable_stage_lock": str(policy.get("requested_publishable_stage_lock", "auto")),
        "auto_selected_publishable_stage": policy.get("auto_selected_publishable_stage"),
        "proxy_tune_steps": int(policy.get("proxy_tune_steps", 0) or 0),
        "proxy_clean_ber": None if policy.get("proxy_clean_ber") is None else float(policy.get("proxy_clean_ber", 0.0) or 0.0),
        "proxy_clean_psnr_drop": None if policy.get("proxy_clean_psnr_drop") is None else float(policy.get("proxy_clean_psnr_drop", 0.0) or 0.0),
        "proxy_wrong_key_mean_ber": None if policy.get("proxy_wrong_key_mean_ber") is None else float(policy.get("proxy_wrong_key_mean_ber", 0.0) or 0.0),
        "resolved_msg_strength_scale": float(policy.get("resolved_msg_strength_scale", 1.0) or 1.0),
        "resolved_native_margin_scale": float(policy.get("resolved_native_margin_scale", 1.0) or 1.0),
        "resolved_wrong_key_scale": float(policy.get("resolved_wrong_key_scale", 1.0) or 1.0),
        "resolved_attack_start_frac": float(policy.get("resolved_attack_start_frac", 0.60) or 0.60),
        "policy_retry_used": bool(policy.get("policy_retry_used", False)),
        "quality_retry_used": bool(policy.get("quality_retry_used", False)),
        "pilot_clean_psnr_drop": float(policy.get("pilot_clean_psnr_drop", 0.0) or 0.0),
        "publishable_quality_constraint_satisfied": bool(
            policy.get("publishable_quality_constraint_satisfied", False)
        ),
        "profile_policy": str(policy.get("profile_policy", "legacy")),
        "periodic_bias_mode": str(policy.get("periodic_bias_mode", "legacy")),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _payload_signature_from_bits(raw_bits_local: torch.Tensor) -> str:
    bitstring = "".join(
        "1" if float(value) >= 0.5 else "0"
        for value in raw_bits_local.detach().flatten().cpu().tolist()
    )
    return hashlib.sha256(bitstring.encode("ascii")).hexdigest()[:24]


def _source_policy_cache_key(
    args: argparse.Namespace,
    *,
    payload_signature: str,
) -> str:
    """Compute a stable cache key for the source-level resolved proxy policy.

    Keyed on policy-determining parameters only; intentionally excludes
    ``baseline_type`` so that ``native`` and ``native_matched_random`` share
    the same cache entry under ``publishable_2d_v1`` for the same payload.
    """
    key_data: dict[str, object] = {
        "policy_cache_version": "publishable_ckpt_v5_no_region_singlevar",
        "source_run": str(getattr(args, "source_run", "")),
        "payload_signature": str(payload_signature),
        "payload_bits": int(getattr(args, "payload_bits", 8)),
        "profile_policy": str(getattr(args, "profile_policy", "legacy")),
        "carrier_channels": str(getattr(args, "carrier_channels", "")),
        "structure_profile": str(getattr(args, "structure_profile", "auto")),
        "tune_steps": int(getattr(args, "tune_steps", 200)),
        "msg_strength": float(getattr(args, "msg_strength", 0.01)),
        "wrong_key_samples": int(getattr(args, "wrong_key_samples", 4)),
        "hardening_sigma_ratio": float(getattr(args, "hardening_sigma_ratio", 0.3)),
        "ot_covertness_weight": float(getattr(args, "ot_covertness_weight", 0.15)),
        "disable_region_policy": bool(getattr(args, "disable_region_policy", False)),
        "disable_coverage_balancing": bool(getattr(args, "disable_coverage_balancing", False)),
        "periodic_lowfreq_carrier_bias": bool(getattr(args, "periodic_lowfreq_carrier_bias", False)),
        "periodic_bias_mode": str(getattr(args, "periodic_bias_mode", "legacy")),
        "periodic_bias_strength": float(getattr(args, "periodic_bias_strength", 0.35)),
        "publishable_stage_lock": str(getattr(args, "publishable_stage_lock", "auto")),
        "periodic_min_stage": str(getattr(args, "periodic_min_stage", "auto")),
        "min_selected_count": int(getattr(args, "min_selected_count", 0)),
        "wavelet_band_loss_weight": getattr(args, "wavelet_band_loss_weight", None),
        "stripe_luminance_loss_weight": getattr(args, "stripe_luminance_loss_weight", None),
    }
    raw = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _load_source_policy_cache(
    args: argparse.Namespace,
    *,
    payload_signature: str,
) -> "dict[str, object] | None":
    """Try to load the cached resolved source policy. Returns None on any miss."""
    try:
        source_run_dir = Path(str(getattr(args, "source_run", "."))).resolve()
        cache_file = (
            source_run_dir
            / "publishable_policy_cache"
            / f"{_source_policy_cache_key(args, payload_signature=payload_signature)}.json"
        )
        if not cache_file.exists():
            return None
        with open(cache_file) as fh:
            cached_policy = json.load(fh)
        if str(cached_policy.get("payload_signature", "")) != str(payload_signature):
            return None
        return cached_policy
    except Exception as exc:
        _warn_once("source_policy_cache_load", f"[v3-cache-warn] source policy cache load failed: {exc}")
        return None


def _save_source_policy_cache(
    source_policy: "dict[str, object]",
    args: argparse.Namespace,
    *,
    payload_signature: str,
) -> None:
    """Persist the resolved source policy to disk; no-op if file already exists."""
    try:
        source_run_dir = Path(str(getattr(args, "source_run", "."))).resolve()
        cache_dir = source_run_dir / "publishable_policy_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{_source_policy_cache_key(args, payload_signature=payload_signature)}.json"
        if cache_file.exists():
            return
        with open(cache_file, "w") as fh:
            json.dump(source_policy, fh, default=str, indent=2)
    except Exception as exc:
        _warn_once("source_policy_cache_save", f"[v3-cache-warn] source policy cache save failed: {exc}")


def _build_v3_pack_from_policy(
    *,
    target: torch.Tensor,
    clean_params: dict[str, torch.Tensor],
    clean_importance: torch.Tensor,
    clean_sensitivity: dict[str, torch.Tensor],
    raw_bits_local: torch.Tensor,
    args: argparse.Namespace,
    key: str,
    assignment_builder: Callable[..., tuple[list[dict], dict[str, object]]],
    active_channels_local: list[str],
    structure_profile: str,
    profile_policy: str,
    selection_temperature: float,
    max_modification_ratio: float,
    keep_fraction: float,
    log_delta: float,
    alpha_delta: float,
    theta_delta: float,
    lum_delta: float,
    coding_strategy: str,
) -> dict[str, object]:
    resolved_active_channels = list(active_channels_local)
    device = clean_params["offsets"].device
    periodic_bias_mode = _normalize_periodic_bias_mode(getattr(args, "periodic_bias_mode", "legacy"))
    importance_device = clean_importance.to(device)
    sensitivity_device = {k: v.to(device) for k, v in clean_sensitivity.items()}
    region_policy_local = _build_2d_region_policy(
        target=target,
        clean_params=clean_params,
        clean_importance=importance_device,
        clean_sensitivity=sensitivity_device,
        active_channels=resolved_active_channels,
        structure_profile=structure_profile,
        profile_policy=profile_policy,
        disable_region_policy=bool(getattr(args, "disable_region_policy", False)),
        disable_coverage_balancing=bool(getattr(args, "disable_coverage_balancing", False)),
    )
    extra_wet_mask = None
    if structure_profile == "periodic":
        resolved_active_channels = [channel for channel in resolved_active_channels if channel != "theta"]
        if profile_policy != "publishable_2d_v1":
            extra_wet_mask = _periodic_edge_wet_mask(
                target,
                clean_params,
                importance_device,
                sensitivity_device,
            )
    periodic_carrier_scales = _build_periodic_lowfreq_carrier_scales(
        gaussian_region_labels=list(region_policy_local["gaussian_region_labels"]),
        structure_profile=structure_profile,
        enabled=bool(getattr(args, "periodic_lowfreq_carrier_bias", False)),
        bias_strength=float(getattr(args, "periodic_bias_strength", 0.35)),
        bias_mode=periodic_bias_mode,
        device=clean_params["offsets"].device,
    )
    carrier_cost_scale_fn = _build_periodic_carrier_cost_scale_fn(
        gaussian_region_labels=list(region_policy_local["gaussian_region_labels"]),
        gaussian_normal_bin_ids=list(region_policy_local.get("gaussian_normal_bin_ids", [])),
        structure_profile=structure_profile,
        enabled=bool(getattr(args, "periodic_lowfreq_carrier_bias", False)),
        bias_strength=float(getattr(args, "periodic_bias_strength", 0.35)),
        bias_mode=periodic_bias_mode,
    )
    v3_bundle_local = build_v3_channel_costs(
        clean_params=clean_params,
        importance=importance_device,
        sensitivity=sensitivity_device,
        wet_threshold=float(args.wet_threshold),
        log_delta=float(log_delta),
        alpha_delta=alpha_delta,
        theta_delta=theta_delta,
        lum_delta=lum_delta,
        cost_sensitivity_w=float(getattr(args, "cost_sensitivity_weight", 0.40)),
        cost_density_w=float(getattr(args, "cost_density_weight", 0.30)),
        cost_visual_w=float(getattr(args, "cost_visual_weight", 0.20)),
        cost_detector_w=float(getattr(args, "cost_detector_weight", 0.10)),
        extra_wet_mask=extra_wet_mask,
    )
    assignments_local, assignment_meta_local = assignment_builder(
        raw_bits=raw_bits_local.detach().cpu(),
        clean_params=clean_params,
        importance=importance_device,
        sensitivity=sensitivity_device,
        v3_bundle=v3_bundle_local,
        key=key,
        candidate_count=24,
        fallback_candidate_count=16,
        beam_width=int(args.beam_width),
        selection_temperature=float(selection_temperature),
        max_modification_ratio=float(max_modification_ratio),
        active_channels=resolved_active_channels,
        structure_profile=structure_profile,
        gaussian_region_labels=list(region_policy_local["gaussian_region_labels"]),
        gaussian_bin_ids=list(region_policy_local["gaussian_bin_ids"]),
        gaussian_normal_bin_ids=list(region_policy_local.get("gaussian_normal_bin_ids", [])),
        gaussian_tangent_bin_ids=list(region_policy_local.get("gaussian_tangent_bin_ids", [])),
        region_allowed_channels=dict(region_policy_local["region_allowed_channels"]),
        region_strength_scales=dict(region_policy_local["region_strength_scales"]),
        gaussian_cost_scales=periodic_carrier_scales,
        carrier_cost_scale_fn=carrier_cost_scale_fn,
        coverage_balancing=bool(region_policy_local["coverage_balancing"]),
        minimum_block_bin_coverage=int(region_policy_local["minimum_block_bin_coverage"]),
        max_bin_share=float(region_policy_local["target_max_bin_share_cap"]),
        periodic_core_action_cap_ratio=float(region_policy_local["periodic_core_action_cap_ratio"]),
        min_selected_count=(
            int(getattr(args, "min_selected_count", 0))
            if int(getattr(args, "min_selected_count", 0)) > 0
            else None
        ),
    )
    channel_weights_local = compute_v3_channel_weights(
        assignments_local,
        active_channels=resolved_active_channels,
        structure_profile=structure_profile,
    )
    assignments_local = _annotate_v3_assignments(assignments_local, channel_weights_local)
    delta_map = {
        "log_anisotropy": float(log_delta),
        "alpha": alpha_delta,
        "theta": theta_delta,
        "color_lum": lum_delta,
    }
    adaptive_encoder = SensitivityAdaptiveEncoder()
    snr_scores_local = adaptive_encoder.estimate_block_snr(
        assignments_local,
        channel_weights=channel_weights_local,
        delta_map=delta_map,
    )
    if coding_strategy == "snr_prune":
        assignments_local = prune_assignments_by_snr(
            assignments_local,
            snr_scores=snr_scores_local,
            keep_fraction=float(keep_fraction),
            beam_width=int(args.beam_width),
        )
        if bool(region_policy_local["coverage_balancing"]):
            assignments_local, post_prune_coverage = enforce_assignment_coverage_constraints(
                assignments_local,
                beam_width=int(args.beam_width),
                modification_budget=int(
                    assignment_meta_local.get("max_modification_count", clean_params["offsets"].shape[0])
                ),
                max_bin_share=float(assignment_meta_local.get("target_max_bin_share_cap", 1.0) or 1.0),
                periodic_core_action_cap_ratio=float(
                    assignment_meta_local.get("periodic_core_action_cap_ratio", 1.0) or 1.0
                ),
            )
        else:
            post_prune_coverage = {
                **_summarize_assignment_regions(
                    assignments_local,
                    target_max_bin_share_cap=assignment_meta_local.get("target_max_bin_share_cap"),
                    periodic_core_action_cap_ratio=assignment_meta_local.get("periodic_core_action_cap_ratio"),
                ),
                "coverage_feasible": True,
                "coverage_repair_used": False,
            }
        channel_weights_local = compute_v3_channel_weights(
            assignments_local,
            active_channels=resolved_active_channels,
            structure_profile=structure_profile,
        )
        assignments_local = _annotate_v3_assignments(assignments_local, channel_weights_local)
        snr_scores_local = adaptive_encoder.estimate_block_snr(
            assignments_local,
            channel_weights=channel_weights_local,
            delta_map=delta_map,
        )
        assignment_meta_local = {
            **assignment_meta_local,
            **post_prune_coverage,
        }
    assignment_meta_local = {
        **assignment_meta_local,
        **_summarize_assignment_regions(
            assignments_local,
            target_max_bin_share_cap=assignment_meta_local.get("target_max_bin_share_cap"),
            periodic_core_action_cap_ratio=assignment_meta_local.get("periodic_core_action_cap_ratio"),
        ),
    }
    return {
        "v3_bundle": v3_bundle_local,
        "assignments": assignments_local,
        "assignment_meta": assignment_meta_local,
        "channel_weights": channel_weights_local,
        "snr_scores": snr_scores_local,
        "active_channels": resolved_active_channels,
        "structure_profile": structure_profile,
        "selection_temperature": float(selection_temperature),
        "max_modification_ratio": float(max_modification_ratio),
        "keep_fraction": float(keep_fraction),
        "region_policy": str(region_policy_local["region_policy"]),
        "region_policy_context": region_policy_local,
        "periodic_carrier_scales": periodic_carrier_scales.detach().cpu().tolist(),
        "periodic_bias_mode": periodic_bias_mode,
    }


def _normalize_periodic_min_stage(requested: str | None) -> str:
    normalized = "auto" if requested is None else str(requested).strip()
    valid = {"auto", "P1", "P2"}
    if normalized not in valid:
        raise ValueError(f"unsupported periodic_min_stage: {requested!r}")
    return normalized


def _filter_periodic_stage_ladder(
    ladder: list[dict[str, float | str]],
    *,
    periodic_min_stage: str,
) -> list[dict[str, float | str]]:
    """Restrict the periodic ladder to stages >= periodic_min_stage.

    The periodic ladder is ordered [P0, P1, P2] (most to least aggressive).
    Returns the input ladder unchanged when periodic_min_stage == 'auto'.
    """
    normalized = _normalize_periodic_min_stage(periodic_min_stage)
    if normalized == "auto":
        return list(ladder)
    if normalized == "P1":
        return [stage for stage in ladder if str(stage["stage_name"]) != "P0"]
    if normalized == "P2":
        return [stage for stage in ladder if str(stage["stage_name"]) == "P2"]
    return list(ladder)


def _publishable_policy_ladder(
    structure_profile: str,
    *,
    periodic_min_stage: str = "auto",
) -> list[dict[str, float | str]]:
    if structure_profile == "periodic":
        periodic_ladder: list[dict[str, float | str]] = [
            {
                "stage_name": "P0",
                "log_delta": 0.12,
                "alpha_delta": 0.028,
                "lum_delta": 0.0050,
                "max_modification_ratio": 0.10,
                "keep_fraction": 0.55,
                "selection_temperature": 0.32,
                "wavelet_band_loss_weight": 0.36,
                "stripe_luminance_loss_weight": 0.24,
                "resolved_msg_strength_scale": 0.85,
                "resolved_native_margin_scale": 0.80,
                "resolved_wrong_key_scale": 0.70,
                "resolved_attack_start_frac": 0.80,
            },
            {
                "stage_name": "P1",
                "log_delta": 0.10,
                "alpha_delta": 0.025,
                "lum_delta": 0.0045,
                "max_modification_ratio": 0.09,
                "keep_fraction": 0.50,
                "selection_temperature": 0.30,
                "wavelet_band_loss_weight": 0.40,
                "stripe_luminance_loss_weight": 0.26,
                "resolved_msg_strength_scale": 0.80,
                "resolved_native_margin_scale": 0.80,
                "resolved_wrong_key_scale": 0.70,
                "resolved_attack_start_frac": 0.80,
            },
            {
                "stage_name": "P2",
                "log_delta": 0.09,
                "alpha_delta": 0.022,
                "lum_delta": 0.0040,
                "max_modification_ratio": 0.08,
                "keep_fraction": 0.45,
                "selection_temperature": 0.28,
                "wavelet_band_loss_weight": 0.44,
                "stripe_luminance_loss_weight": 0.28,
                "resolved_msg_strength_scale": 0.75,
                "resolved_native_margin_scale": 0.80,
                "resolved_wrong_key_scale": 0.70,
                "resolved_attack_start_frac": 0.80,
            },
        ]
        return _filter_periodic_stage_ladder(
            periodic_ladder,
            periodic_min_stage=periodic_min_stage,
        )
    return [
        {
            "stage_name": "G0",
            "log_delta": 0.22,
            "alpha_delta": 0.048,
            "lum_delta": 0.0085,
            "max_modification_ratio": 0.22,
            "keep_fraction": 1.00,
            "selection_temperature": 0.45,
            "wavelet_band_loss_weight": 0.02,
            "stripe_luminance_loss_weight": 0.01,
            "resolved_msg_strength_scale": 1.25,
            "resolved_native_margin_scale": 1.15,
            "resolved_wrong_key_scale": 1.00,
            "resolved_attack_start_frac": 0.60,
        },
        {
            "stage_name": "G1",
            "log_delta": 0.20,
            "alpha_delta": 0.045,
            "lum_delta": 0.0080,
            "max_modification_ratio": 0.20,
            "keep_fraction": 0.95,
            "selection_temperature": 0.42,
            "wavelet_band_loss_weight": 0.04,
            "stripe_luminance_loss_weight": 0.02,
            "resolved_msg_strength_scale": 1.20,
            "resolved_native_margin_scale": 1.15,
            "resolved_wrong_key_scale": 1.00,
            "resolved_attack_start_frac": 0.60,
        },
        {
            "stage_name": "G2",
            "log_delta": 0.16,
            "alpha_delta": 0.040,
            "lum_delta": 0.0070,
            "max_modification_ratio": 0.18,
            "keep_fraction": 0.90,
            "selection_temperature": 0.40,
            "wavelet_band_loss_weight": 0.06,
            "stripe_luminance_loss_weight": 0.03,
            "resolved_msg_strength_scale": 1.15,
            "resolved_native_margin_scale": 1.15,
            "resolved_wrong_key_scale": 1.00,
            "resolved_attack_start_frac": 0.60,
        },
    ]


def _publishable_proxy_tune_steps(structure_profile: str) -> int:
    return 64 if structure_profile == "periodic" else 48


def _publishable_checkpoint_schedule_mode(
    structure_profile: str,
    *,
    proxy: bool = False,
) -> str:
    if proxy:
        return "legacy_fixed_16"
    return "periodic_tail_dense" if structure_profile == "periodic" else "generic_early_dense_20"


def _should_evaluate_publishable_checkpoint(
    *,
    step: int,
    total_steps: int,
    structure_profile: str,
    proxy: bool = False,
) -> bool:
    current_step = int(step)
    if current_step <= 0:
        return False
    if current_step >= int(total_steps):
        return True
    if proxy:
        return (current_step % 16) == 0
    # The optimizer starts from the signed pilot action update. Check every
    # early step so a short-lived valid post-step state is not missed.
    if current_step <= 20:
        return True
    if structure_profile != "periodic":
        return (current_step % 20) == 0
    if current_step <= 80:
        return (current_step % 5) == 0
    if current_step <= 140:
        return (current_step % 10) == 0
    return (current_step % 5) == 0


def _publishable_checkpoint_gate_metrics(
    *,
    clean_ber: float | None,
    clean_psnr_drop: float | None,
    wrong_key_mean_ber: float | None,
    quantized_clean_ber: float | None = None,
    quantized_clean_psnr_drop: float | None = None,
    quantized_wrong_key_mean_ber: float | None = None,
    quantize_consistency: bool = False,
) -> tuple[float, float, float]:
    clean_ber_value = float("inf") if clean_ber is None or math.isnan(float(clean_ber)) else float(clean_ber)
    clean_psnr_value = float("inf") if clean_psnr_drop is None or math.isnan(float(clean_psnr_drop)) else float(clean_psnr_drop)
    wrong_key_value = float("-inf") if wrong_key_mean_ber is None or math.isnan(float(wrong_key_mean_ber)) else float(wrong_key_mean_ber)
    if not quantize_consistency:
        return clean_ber_value, clean_psnr_value, wrong_key_value
    quantized_clean_ber_value = (
        clean_ber_value
        if quantized_clean_ber is None or math.isnan(float(quantized_clean_ber))
        else float(quantized_clean_ber)
    )
    quantized_clean_psnr_value = (
        clean_psnr_value
        if quantized_clean_psnr_drop is None or math.isnan(float(quantized_clean_psnr_drop))
        else float(quantized_clean_psnr_drop)
    )
    quantized_wrong_key_value = (
        wrong_key_value
        if quantized_wrong_key_mean_ber is None or math.isnan(float(quantized_wrong_key_mean_ber))
        else float(quantized_wrong_key_mean_ber)
    )
    return (
        quantized_clean_ber_value,
        quantized_clean_psnr_value,
        quantized_wrong_key_value,
    )


def _publishable_checkpoint_is_admissible(
    *,
    clean_ber: float | None,
    clean_psnr_drop: float | None,
    wrong_key_mean_ber: float | None,
    quantized_clean_ber: float | None = None,
    quantized_clean_psnr_drop: float | None = None,
    quantized_wrong_key_mean_ber: float | None = None,
    quantize_consistency: bool = False,
) -> bool:
    clean_ber_value, clean_psnr_value, wrong_key_value = _publishable_checkpoint_gate_metrics(
        clean_ber=clean_ber,
        clean_psnr_drop=clean_psnr_drop,
        wrong_key_mean_ber=wrong_key_mean_ber,
        quantized_clean_ber=quantized_clean_ber,
        quantized_clean_psnr_drop=quantized_clean_psnr_drop,
        quantized_wrong_key_mean_ber=quantized_wrong_key_mean_ber,
        quantize_consistency=quantize_consistency,
    )
    wrong_key_gap = _wrong_key_chance_gap(wrong_key_value)
    return bool(
        math.isfinite(clean_ber_value)
        and math.isfinite(clean_psnr_value)
        and math.isfinite(wrong_key_value)
        and clean_ber_value <= 0.25 + 1e-9
        and clean_psnr_value <= 3.0 + 1e-9
        and wrong_key_gap <= PUBLISHABLE_WRONG_KEY_TOLERANCE + 1e-9
    )


def _publishable_stage_rank(
    *,
    coverage_constraint_satisfied: bool,
    clean_ber: float | None,
    clean_psnr_drop: float | None,
    wrong_key_mean_ber: float | None,
    resolved_log_delta: float,
) -> tuple[object, ...]:
    clean_ber_value = float("inf") if clean_ber is None or math.isnan(float(clean_ber)) else float(clean_ber)
    clean_psnr_value = float("inf") if clean_psnr_drop is None or math.isnan(float(clean_psnr_drop)) else float(clean_psnr_drop)
    wrong_key_value = float("-inf") if wrong_key_mean_ber is None or math.isnan(float(wrong_key_mean_ber)) else float(wrong_key_mean_ber)
    wrong_key_gap = _wrong_key_chance_gap(wrong_key_value)
    return (
        0 if bool(coverage_constraint_satisfied) else 1,
        0 if clean_ber_value <= 0.25 + 1e-9 else 1,
        0 if clean_psnr_value <= 3.0 + 1e-9 else 1,
        0 if wrong_key_gap <= PUBLISHABLE_WRONG_KEY_TOLERANCE + 1e-9 else 1,
        clean_ber_value,
        0 if wrong_key_gap <= PUBLISHABLE_REPORT_WRONG_KEY_TOLERANCE + 1e-9 else 1,
        clean_psnr_value,
        wrong_key_gap,
        float(resolved_log_delta),
    )


def _selected_checkpoint_satisfies_publishable_quality(
    *,
    publishable_checkpoint_enabled: bool,
    best_checkpoint_selected: bool,
    clean_ber: float | None,
    clean_psnr_drop: float | None,
    wrong_key_mean_ber: float | None,
    quantized_clean_ber: float | None = None,
    quantized_clean_psnr_drop: float | None = None,
    quantized_wrong_key_mean_ber: float | None = None,
    quantize_consistency: bool = False,
) -> bool:
    """Report the quality gate for the checkpoint that was actually restored."""
    return bool(
        publishable_checkpoint_enabled
        and best_checkpoint_selected
        and _publishable_checkpoint_is_admissible(
            clean_ber=clean_ber,
            clean_psnr_drop=clean_psnr_drop,
            wrong_key_mean_ber=wrong_key_mean_ber,
            quantized_clean_ber=quantized_clean_ber,
            quantized_clean_psnr_drop=quantized_clean_psnr_drop,
            quantized_wrong_key_mean_ber=quantized_wrong_key_mean_ber,
            quantize_consistency=quantize_consistency,
        )
        and _wrong_key_near_chance(
            wrong_key_mean_ber,
            tolerance=PUBLISHABLE_REPORT_WRONG_KEY_TOLERANCE,
        )
    )


def _publishable_checkpoint_rank(
    *,
    clean_ber: float | None,
    clean_psnr_drop: float | None,
    wrong_key_mean_ber: float | None,
    quantized_clean_ber: float | None = None,
    quantized_clean_psnr_drop: float | None = None,
    quantized_wrong_key_mean_ber: float | None = None,
    quantize_consistency: bool = False,
    step: int,
) -> tuple[object, ...]:
    clean_ber_value = float("inf") if clean_ber is None or math.isnan(float(clean_ber)) else float(clean_ber)
    clean_psnr_value = float("inf") if clean_psnr_drop is None or math.isnan(float(clean_psnr_drop)) else float(clean_psnr_drop)
    wrong_key_value = float("-inf") if wrong_key_mean_ber is None or math.isnan(float(wrong_key_mean_ber)) else float(wrong_key_mean_ber)
    quantized_clean_ber_value = (
        clean_ber_value
        if quantized_clean_ber is None or math.isnan(float(quantized_clean_ber))
        else float(quantized_clean_ber)
    )
    quantized_clean_psnr_value = (
        clean_psnr_value
        if quantized_clean_psnr_drop is None or math.isnan(float(quantized_clean_psnr_drop))
        else float(quantized_clean_psnr_drop)
    )
    quantized_wrong_key_value = (
        wrong_key_value
        if quantized_wrong_key_mean_ber is None or math.isnan(float(quantized_wrong_key_mean_ber))
        else float(quantized_wrong_key_mean_ber)
    )
    if quantize_consistency:
        quantized_wrong_key_gap = _wrong_key_chance_gap(quantized_wrong_key_value)
        wrong_key_gap = _wrong_key_chance_gap(wrong_key_value)
        return (
            0 if quantized_clean_ber_value <= 0.25 + 1e-9 else 1,
            0 if quantized_clean_psnr_value <= 3.0 + 1e-9 else 1,
            0 if quantized_wrong_key_gap <= PUBLISHABLE_WRONG_KEY_TOLERANCE + 1e-9 else 1,
            quantized_clean_ber_value,
            0 if quantized_wrong_key_gap <= PUBLISHABLE_REPORT_WRONG_KEY_TOLERANCE + 1e-9 else 1,
            quantized_clean_psnr_value,
            quantized_wrong_key_gap,
            clean_ber_value,
            clean_psnr_value,
            wrong_key_gap,
            int(step),
        )
    wrong_key_gap = _wrong_key_chance_gap(wrong_key_value)
    return (
        0 if clean_ber_value <= 0.25 + 1e-9 else 1,
        0 if clean_psnr_value <= 3.0 + 1e-9 else 1,
        0 if wrong_key_gap <= PUBLISHABLE_WRONG_KEY_TOLERANCE + 1e-9 else 1,
        clean_ber_value,
        0 if wrong_key_gap <= PUBLISHABLE_REPORT_WRONG_KEY_TOLERANCE + 1e-9 else 1,
        clean_psnr_value,
        wrong_key_gap,
        int(step),
    )


def _publishable_exact_clean_checkpoint_rank(
    *,
    clean_psnr_drop: float | None,
    wrong_key_mean_ber: float | None,
    step: int,
) -> tuple[object, ...]:
    clean_psnr_value = (
        float("inf")
        if clean_psnr_drop is None or math.isnan(float(clean_psnr_drop))
        else float(clean_psnr_drop)
    )
    wrong_key_value = (
        float("-inf")
        if wrong_key_mean_ber is None or math.isnan(float(wrong_key_mean_ber))
        else float(wrong_key_mean_ber)
    )
    wrong_key_gap = _wrong_key_chance_gap(wrong_key_value)
    return (
        0 if wrong_key_gap <= PUBLISHABLE_WRONG_KEY_TOLERANCE + 1e-9 else 1,
        0 if wrong_key_gap <= PUBLISHABLE_REPORT_WRONG_KEY_TOLERANCE + 1e-9 else 1,
        0 if clean_psnr_value <= 3.0 + 1e-9 else 1,
        wrong_key_gap,
        clean_psnr_value,
        int(step),
    )


def _resolve_publishable_source_policy(
    *,
    target: torch.Tensor,
    clean_params: dict[str, torch.Tensor],
    clean_importance: torch.Tensor,
    clean_sensitivity: dict[str, torch.Tensor],
    raw_bits: torch.Tensor,
    args: argparse.Namespace,
    initial_active_channels: list[str],
    theta_delta: float,
    coding_strategy: str,
    base_profile: str,
    structure_profile_stats: dict[str, float],
    reference_builder: Callable[..., tuple[list[dict], dict[str, object]]],
    policy_resolution_mode: str,
    profile_policy: str,
    proxy_stage_scorer: Callable[[dict[str, object], dict[str, object]], dict[str, object]] | None = None,
) -> dict[str, object]:
    requested_stage_lock = _normalize_publishable_stage_lock(
        getattr(args, "publishable_stage_lock", "auto")
    )
    periodic_min_stage = _normalize_periodic_min_stage(
        getattr(args, "periodic_min_stage", "auto")
    )
    stage_results: list[dict[str, object]] = []
    for stage_index, stage in enumerate(
        _publishable_policy_ladder(base_profile, periodic_min_stage=periodic_min_stage)
    ):
        pack_local = _build_v3_pack_from_policy(
            target=target,
            clean_params=clean_params,
            clean_importance=clean_importance,
            clean_sensitivity=clean_sensitivity,
            raw_bits_local=raw_bits,
            args=args,
            key=str(getattr(args, "key", "ucl-2d-watermark-v3")),
            assignment_builder=reference_builder,
            active_channels_local=initial_active_channels,
            structure_profile=base_profile,
            profile_policy=profile_policy,
            selection_temperature=float(stage["selection_temperature"]),
            max_modification_ratio=float(stage["max_modification_ratio"]),
            keep_fraction=float(stage["keep_fraction"]),
            log_delta=float(stage["log_delta"]),
            alpha_delta=float(stage["alpha_delta"]),
            theta_delta=theta_delta,
            lum_delta=float(stage["lum_delta"]),
            coding_strategy=coding_strategy,
        )
        if not pack_local.get("assignments") or int(pack_local["assignment_meta"].get("effective_action_budget", 0)) <= 0:
            raise RuntimeError("no_viable_policy_assignments")
        pilot_clean_ber = _pilot_clean_ber(
            clean_params=clean_params,
            assignments=pack_local["assignments"],
            channel_weights=pack_local["channel_weights"],
            raw_bits=raw_bits,
            log_delta=float(stage["log_delta"]),
            alpha_delta=float(stage["alpha_delta"]),
            theta_delta=theta_delta,
            lum_delta=float(stage["lum_delta"]),
        )
        pilot_clean_psnr_drop = _pilot_clean_psnr_drop(
            target=target,
            clean_params=clean_params,
            assignments=pack_local["assignments"],
            log_delta=float(stage["log_delta"]),
            alpha_delta=float(stage["alpha_delta"]),
            theta_delta=theta_delta,
            lum_delta=float(stage["lum_delta"]),
        )
        coverage_constraint_satisfied = bool(pack_local["assignment_meta"].get("coverage_constraint_satisfied", True))
        publishable_quality_constraint_satisfied = bool(
            coverage_constraint_satisfied
            and pilot_clean_ber <= 0.25 + 1e-9
            and pilot_clean_psnr_drop <= 3.0 + 1e-9
        )
        proxy_metrics = (
            proxy_stage_scorer(pack_local, dict(stage))
            if proxy_stage_scorer is not None
            else {}
        )
        proxy_clean_ber = proxy_metrics.get("proxy_clean_ber")
        proxy_clean_psnr_drop = proxy_metrics.get("proxy_clean_psnr_drop")
        proxy_wrong_key_mean_ber = proxy_metrics.get("proxy_wrong_key_mean_ber")
        proxy_wrong_key_decode_attempts = proxy_metrics.get("proxy_wrong_key_decode_attempts")
        proxy_wrong_key_decode_failures = proxy_metrics.get("proxy_wrong_key_decode_failures")
        proxy_tune_steps = int(proxy_metrics.get("proxy_tune_steps", 0) or 0)
        quality_clean_ber = proxy_clean_ber if proxy_clean_ber is not None else pilot_clean_ber
        quality_clean_psnr_drop = (
            proxy_clean_psnr_drop if proxy_clean_psnr_drop is not None else pilot_clean_psnr_drop
        )
        quality_wrong_key_mean_ber = proxy_wrong_key_mean_ber
        publishable_quality_constraint_satisfied = bool(
            coverage_constraint_satisfied
            and quality_clean_ber <= 0.25 + 1e-9
            and quality_clean_psnr_drop <= 3.0 + 1e-9
            and quality_wrong_key_mean_ber is not None
            and _wrong_key_near_chance(quality_wrong_key_mean_ber)
        )
        result = {
            "stage_index": int(stage_index),
            "stage_name": str(stage["stage_name"]),
            "pack": pack_local,
            "pilot_clean_ber": float(pilot_clean_ber),
            "pilot_clean_psnr_drop": float(pilot_clean_psnr_drop),
            "proxy_clean_ber": None if proxy_clean_ber is None else float(proxy_clean_ber),
            "proxy_clean_psnr_drop": None if proxy_clean_psnr_drop is None else float(proxy_clean_psnr_drop),
            "proxy_wrong_key_mean_ber": None if proxy_wrong_key_mean_ber is None else float(proxy_wrong_key_mean_ber),
            "proxy_wrong_key_decode_attempts": proxy_wrong_key_decode_attempts,
            "proxy_wrong_key_decode_failures": proxy_wrong_key_decode_failures,
            "proxy_tune_steps": int(proxy_tune_steps),
            "coverage_constraint_satisfied": bool(coverage_constraint_satisfied),
            "publishable_quality_constraint_satisfied": bool(publishable_quality_constraint_satisfied),
            "resolved_log_delta": float(stage["log_delta"]),
            "resolved_alpha_delta": float(stage["alpha_delta"]),
            "resolved_lum_delta": float(stage["lum_delta"]),
            "resolved_selection_temperature": float(stage["selection_temperature"]),
            "resolved_max_modification_ratio": float(stage["max_modification_ratio"]),
            "resolved_keep_fraction": float(stage["keep_fraction"]),
            "resolved_wavelet_band_loss_weight": float(stage["wavelet_band_loss_weight"]),
            "resolved_stripe_luminance_loss_weight": float(stage["stripe_luminance_loss_weight"]),
            "resolved_msg_strength_scale": float(stage["resolved_msg_strength_scale"]),
            "resolved_native_margin_scale": float(stage["resolved_native_margin_scale"]),
            "resolved_wrong_key_scale": float(stage["resolved_wrong_key_scale"]),
            "resolved_attack_start_frac": float(stage["resolved_attack_start_frac"]),
            "resolved_region_policy_variant": str(
                pack_local["region_policy_context"].get("resolved_region_policy_variant", "legacy")
            ),
        }
        stage_results.append(result)
    chosen_result, auto_result = _select_publishable_stage_result(
        stage_results,
        requested_stage_lock=requested_stage_lock,
    )
    pack_local = chosen_result["pack"]
    resolved_policy = {
        "policy_resolution_mode": policy_resolution_mode,
        "resolved_structure_profile": str(pack_local["structure_profile"]),
        "resolved_region_policy": str(pack_local["region_policy"]),
        "resolved_region_policy_variant": str(chosen_result["resolved_region_policy_variant"]),
        "resolved_active_channels": list(pack_local["active_channels"]),
        "resolved_selection_temperature": float(chosen_result["resolved_selection_temperature"]),
        "resolved_max_modification_ratio": float(chosen_result["resolved_max_modification_ratio"]),
        "resolved_keep_fraction": float(chosen_result["resolved_keep_fraction"]),
        "resolved_log_delta": float(chosen_result["resolved_log_delta"]),
        "resolved_alpha_delta": float(chosen_result["resolved_alpha_delta"]),
        "resolved_theta_delta": float(theta_delta),
        "resolved_lum_delta": float(chosen_result["resolved_lum_delta"]),
        "resolved_wavelet_band_loss_weight": float(chosen_result["resolved_wavelet_band_loss_weight"]),
        "resolved_stripe_luminance_loss_weight": float(
            chosen_result["resolved_stripe_luminance_loss_weight"]
        ),
        "resolved_msg_strength_scale": float(chosen_result["resolved_msg_strength_scale"]),
        "resolved_native_margin_scale": float(chosen_result["resolved_native_margin_scale"]),
        "resolved_wrong_key_scale": float(chosen_result["resolved_wrong_key_scale"]),
        "resolved_attack_start_frac": float(chosen_result["resolved_attack_start_frac"]),
        "requested_publishable_stage_lock": requested_stage_lock,
        "auto_selected_publishable_stage": str(auto_result["stage_name"]),
        "selected_publishable_stage": str(chosen_result["stage_name"]),
        "publishable_stage_lock_applied": bool(requested_stage_lock != "auto"),
        "periodic_min_stage": periodic_min_stage,
        "periodic_min_stage_applied": bool(
            base_profile == "periodic" and periodic_min_stage != "auto"
        ),
        "proxy_tune_steps": int(chosen_result["proxy_tune_steps"]),
        "proxy_clean_ber": chosen_result.get("proxy_clean_ber"),
        "proxy_clean_psnr_drop": chosen_result.get("proxy_clean_psnr_drop"),
        "proxy_wrong_key_mean_ber": chosen_result.get("proxy_wrong_key_mean_ber"),
        "proxy_wrong_key_decode_attempts": chosen_result.get("proxy_wrong_key_decode_attempts"),
        "proxy_wrong_key_decode_failures": chosen_result.get("proxy_wrong_key_decode_failures"),
        "proxy_checkpoint_schedule_mode": str(
            chosen_result.get(
                "proxy_checkpoint_schedule_mode",
                _publishable_checkpoint_schedule_mode(
                    str(pack_local["structure_profile"]),
                    proxy=True,
                ),
            )
        ),
        "policy_retry_used": bool(int(chosen_result["stage_index"]) > 0),
        "quality_retry_used": bool(int(chosen_result["stage_index"]) > 0),
        "publishable_quality_constraint_satisfied": bool(
            chosen_result["publishable_quality_constraint_satisfied"]
        ),
        "profile_policy": profile_policy,
        "periodic_bias_mode": str(getattr(args, "periodic_bias_mode", "legacy")),
        "pilot_clean_ber": float(chosen_result["pilot_clean_ber"]),
        "pilot_clean_psnr_drop": float(chosen_result["pilot_clean_psnr_drop"]),
    }
    resolved_policy["source_policy_signature"] = _source_policy_signature(resolved_policy)
    return {
        **resolved_policy,
        "structure_profile_stats": {
            **structure_profile_stats,
            "resolved_structure_profile": str(resolved_policy["resolved_structure_profile"]),
            "resolved_region_policy": str(resolved_policy["resolved_region_policy"]),
            "resolved_region_policy_variant": str(resolved_policy["resolved_region_policy_variant"]),
            "policy_resolution_mode": policy_resolution_mode,
            "policy_retry_used": bool(resolved_policy["policy_retry_used"]),
            "quality_retry_used": bool(resolved_policy["quality_retry_used"]),
            "requested_publishable_stage_lock": requested_stage_lock,
            "auto_selected_publishable_stage": str(auto_result["stage_name"]),
            "selected_publishable_stage": str(chosen_result["stage_name"]),
        },
    }


def resolve_source_policy(
    *,
    target: torch.Tensor,
    clean_params: dict[str, torch.Tensor],
    clean_importance: torch.Tensor,
    clean_sensitivity: dict[str, torch.Tensor],
    raw_bits: torch.Tensor,
    args: argparse.Namespace,
    initial_active_channels: list[str],
    log_delta: float,
    alpha_delta: float,
    theta_delta: float,
    lum_delta: float,
    coding_strategy: str,
    max_modification_ratio: float,
    proxy_stage_scorer: Callable[[dict[str, object], dict[str, object]], dict[str, object]] | None = None,
) -> dict[str, object]:
    requested_profile = str(getattr(args, "structure_profile", "auto"))
    profile_policy = str(getattr(args, "profile_policy", "legacy"))
    baseline_type = str(getattr(args, "baseline_type", "native"))
    base_profile, structure_profile_stats = _classify_structure_profile(
        target,
        clean_params,
        requested=requested_profile,
    )
    policy_knobs = _policy_knobs_for_profile(
        base_profile=base_profile,
        selection_temperature=float(args.selection_temperature),
        max_modification_ratio=float(max_modification_ratio),
    )
    shared_reference = profile_policy == "publishable_2d_v1" and baseline_type in {"native", "native_matched_random"}
    requested_stage_lock = _normalize_publishable_stage_lock(
        getattr(args, "publishable_stage_lock", "auto")
    )
    if requested_stage_lock != "auto" and not shared_reference:
        raise ValueError(
            "publishable stage lock requires a shared publishable reference "
            "(profile_policy=publishable_2d_v1 with baseline_type native/native_matched_random)"
        )
    reference_builder = build_v3_cost_code_assignments if shared_reference else {
        "native": build_v3_cost_code_assignments,
        "native_matched_random": build_v3_random_cost_code_assignments,
        "native_greedy_nokey": build_v3_greedy_nokey_cost_code_assignments,
        "native_random_polarity": build_v3_random_polarity_cost_code_assignments,
    }.get(baseline_type, build_v3_cost_code_assignments)
    policy_resolution_mode = "shared_publishable_reference" if shared_reference else "baseline_local"
    explicit_profile_override = requested_profile in {"generic", "periodic"}

    if shared_reference:
        return _resolve_publishable_source_policy(
            target=target,
            clean_params=clean_params,
            clean_importance=clean_importance,
            clean_sensitivity=clean_sensitivity,
            raw_bits=raw_bits,
            args=args,
            initial_active_channels=initial_active_channels,
            theta_delta=theta_delta,
            coding_strategy=coding_strategy,
            base_profile=base_profile,
            structure_profile_stats=structure_profile_stats,
            reference_builder=reference_builder,
            policy_resolution_mode=policy_resolution_mode,
            profile_policy=profile_policy,
            proxy_stage_scorer=proxy_stage_scorer,
        )

    policy_pack = _build_v3_pack_from_policy(
        target=target,
        clean_params=clean_params,
        clean_importance=clean_importance,
        clean_sensitivity=clean_sensitivity,
        raw_bits_local=raw_bits,
        args=args,
        key=_primary_assignment_key(args),
        assignment_builder=reference_builder,
        active_channels_local=initial_active_channels,
        structure_profile=base_profile,
        profile_policy=profile_policy,
        selection_temperature=float(policy_knobs["selection_temperature"]),
        max_modification_ratio=float(policy_knobs["max_modification_ratio"]),
        keep_fraction=float(policy_knobs["keep_fraction"]),
        log_delta=log_delta,
        alpha_delta=alpha_delta,
        theta_delta=theta_delta,
        lum_delta=lum_delta,
        coding_strategy=coding_strategy,
    )
    if not policy_pack.get("assignments") or int(policy_pack["assignment_meta"].get("effective_action_budget", 0)) <= 0:
        raise RuntimeError("no_viable_policy_assignments")

    policy_retry_used = False
    pilot_clean_ber = _pilot_clean_ber(
        clean_params=clean_params,
        assignments=policy_pack["assignments"],
        channel_weights=policy_pack["channel_weights"],
        raw_bits=raw_bits,
        log_delta=float(log_delta),
        alpha_delta=alpha_delta,
        theta_delta=theta_delta,
        lum_delta=lum_delta,
    )
    pilot_clean_psnr_drop = _pilot_clean_psnr_drop(
        target=target,
        clean_params=clean_params,
        assignments=policy_pack["assignments"],
        log_delta=float(log_delta),
        alpha_delta=alpha_delta,
        theta_delta=theta_delta,
        lum_delta=lum_delta,
    )
    if pilot_clean_ber > 0.05:
        policy_retry_used = True
        retry_config = _pilot_retry_config(
            profile=str(policy_pack["structure_profile"]),
            selection_temperature=float(policy_pack["selection_temperature"]),
            max_modification_ratio=float(policy_pack["max_modification_ratio"]),
            keep_fraction=float(policy_pack["keep_fraction"]),
            active_channels=list(policy_pack["active_channels"]),
            allow_profile_change=not explicit_profile_override,
        )
        policy_pack = _build_v3_pack_from_policy(
            target=target,
            clean_params=clean_params,
            clean_importance=clean_importance,
            clean_sensitivity=clean_sensitivity,
            raw_bits_local=raw_bits,
            args=args,
            key=_primary_assignment_key(args),
            assignment_builder=reference_builder,
            active_channels_local=list(retry_config["active_channels"]),
            structure_profile=str(retry_config["profile"]),
            profile_policy=profile_policy,
            selection_temperature=float(retry_config["selection_temperature"]),
            max_modification_ratio=float(retry_config["max_modification_ratio"]),
            keep_fraction=float(retry_config["keep_fraction"]),
            log_delta=log_delta,
            alpha_delta=alpha_delta,
            theta_delta=theta_delta,
            lum_delta=lum_delta,
            coding_strategy=coding_strategy,
        )
        if not policy_pack.get("assignments") or int(policy_pack["assignment_meta"].get("effective_action_budget", 0)) <= 0:
            raise RuntimeError("no_viable_policy_assignments_after_retry")
        pilot_clean_ber = _pilot_clean_ber(
            clean_params=clean_params,
            assignments=policy_pack["assignments"],
            channel_weights=policy_pack["channel_weights"],
            raw_bits=raw_bits,
            log_delta=float(log_delta),
            alpha_delta=alpha_delta,
            theta_delta=theta_delta,
            lum_delta=lum_delta,
        )
        pilot_clean_psnr_drop = _pilot_clean_psnr_drop(
            target=target,
            clean_params=clean_params,
            assignments=policy_pack["assignments"],
            log_delta=float(log_delta),
            alpha_delta=alpha_delta,
            theta_delta=theta_delta,
            lum_delta=lum_delta,
        )

    resolved_policy = {
        "policy_resolution_mode": policy_resolution_mode,
        "resolved_structure_profile": str(policy_pack["structure_profile"]),
        "resolved_region_policy": str(policy_pack["region_policy"]),
        "resolved_region_policy_variant": str(
            policy_pack["region_policy_context"].get("resolved_region_policy_variant", "legacy")
        ),
        "resolved_active_channels": list(policy_pack["active_channels"]),
        "resolved_selection_temperature": float(policy_pack["selection_temperature"]),
        "resolved_max_modification_ratio": float(policy_pack["max_modification_ratio"]),
        "resolved_keep_fraction": float(policy_pack["keep_fraction"]),
        "resolved_log_delta": float(log_delta),
        "resolved_alpha_delta": float(alpha_delta),
        "resolved_theta_delta": float(theta_delta),
        "resolved_lum_delta": float(lum_delta),
        "resolved_wavelet_band_loss_weight": _default_wavelet_band_loss_weight(
            structure_profile=str(policy_pack["structure_profile"]),
            profile_policy=profile_policy,
            explicit_weight=getattr(args, "wavelet_band_loss_weight", None),
        ),
        "resolved_stripe_luminance_loss_weight": _default_stripe_luminance_loss_weight(
            structure_profile=str(policy_pack["structure_profile"]),
            profile_policy=profile_policy,
            explicit_weight=getattr(args, "stripe_luminance_loss_weight", None),
        ),
        "resolved_msg_strength_scale": 1.0,
        "resolved_native_margin_scale": 1.0,
        "resolved_wrong_key_scale": 1.0,
        "resolved_attack_start_frac": 0.60,
        "requested_publishable_stage_lock": requested_stage_lock,
        "auto_selected_publishable_stage": None,
        "selected_publishable_stage": None,
        "publishable_stage_lock_applied": False,
        "proxy_tune_steps": 0,
        "proxy_clean_ber": None,
        "proxy_clean_psnr_drop": None,
        "proxy_wrong_key_mean_ber": None,
        "policy_retry_used": bool(policy_retry_used),
        "quality_retry_used": False,
        "publishable_quality_constraint_satisfied": bool(
            pilot_clean_ber <= 0.25 + 1e-9 and pilot_clean_psnr_drop <= 3.0 + 1e-9
        ),
        "profile_policy": profile_policy,
        "periodic_bias_mode": str(getattr(args, "periodic_bias_mode", "legacy")),
    }
    resolved_policy["source_policy_signature"] = _source_policy_signature(resolved_policy)
    return {
        **resolved_policy,
        "pilot_clean_ber": float(pilot_clean_ber),
        "pilot_clean_psnr_drop": float(pilot_clean_psnr_drop),
        "structure_profile_stats": {
            **structure_profile_stats,
            "resolved_structure_profile": str(resolved_policy["resolved_structure_profile"]),
            "resolved_region_policy": str(resolved_policy["resolved_region_policy"]),
            "resolved_region_policy_variant": str(resolved_policy["resolved_region_policy_variant"]),
            "policy_resolution_mode": policy_resolution_mode,
            "policy_retry_used": bool(policy_retry_used),
            "quality_retry_used": False,
            "requested_publishable_stage_lock": requested_stage_lock,
        },
    }


def _annotate_v3_assignments(
    assignments: list[dict],
    channel_weights: dict[str, float],
) -> list[dict]:
    for assignment in assignments:
        for action in assignment.get("candidate_actions", []):
            channel = str(action.get("channel", "log_anisotropy"))
            action["channel_weight"] = float(channel_weights.get(channel, 1.0))
    return assignments


def _v3_per_channel_ber_summary(
    clean_params: dict[str, torch.Tensor],
    wm_params: dict[str, torch.Tensor],
    assignments: list[dict],
    *,
    active_channels: list[str],
    theta_delta: float,
    lum_delta: float,
) -> dict[str, float | None]:
    summary: dict[str, float | None] = {
        "log_anisotropy": None,
        "alpha": None,
        "theta": None,
        "color_lum": None,
    }
    if "theta" in active_channels:
        summary["theta"] = theta_channel_ber(
            _canonicalize_geometry_local(clean_params)["theta_major"][:, 0],
            _canonicalize_geometry_local(wm_params)["theta_major"][:, 0],
            assignments,
        )
    if "color_lum" in active_channels:
        summary["color_lum"] = color_lum_channel_ber(clean_params["colors"], wm_params["colors"], assignments, lum_delta)
    return summary


def _truncated_refit_attack(
    params: dict[str, torch.Tensor],
    *,
    target: torch.Tensor,
    grid: torch.Tensor,
    steps: int,
    lr: float = 0.02,
) -> dict[str, torch.Tensor]:
    attacked = {
        "offsets": params["offsets"],
        "thetas": params["thetas"],
        "scales": params["scales"],
        "opacities": params["opacities"],
        "colors": params["colors"],
    }
    scales = attacked["scales"]
    opacities = attacked["opacities"]
    colors = attacked["colors"]
    for _ in range(max(1, int(steps))):
        inner_params = {
            "offsets": attacked["offsets"],
            "thetas": attacked["thetas"],
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
        }
        recon = _render_from_physical_params(inner_params, grid[..., 0], grid[..., 1])
        loss = F.mse_loss(recon, target)
        grad_scales, grad_opacities, grad_colors = torch.autograd.grad(
            loss,
            [scales, opacities, colors],
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        scales = torch.clamp(scales - lr * grad_scales, min=1e-6)
        opacities = torch.clamp(opacities - lr * grad_opacities, 0.0, 1.0)
        colors = torch.clamp(colors - lr * grad_colors, 0.0, 1.0)
    attacked["scales"] = scales
    attacked["opacities"] = opacities
    attacked["colors"] = colors
    return attacked


# ============================================================================
# Core v3 embedding function
# ============================================================================

def run_cost_stego_v3_embedding(
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
) -> tuple[Gaussian2DRenderer, dict]:
    """Embed a v3.1 watermark with a true 4-channel decode path."""
    defaults = _cost_stego_v3_experiment_defaults(args)
    initial_active_channels = _active_carrier_channels(args)
    coding_strategy = _resolve_coding_strategy(args)
    threat_model = str(getattr(args, "threat_model", "benign_carrier"))
    profile_policy = str(getattr(args, "profile_policy", "legacy"))
    capacity_failure_reason: str | None = None

    adaptive_controls = _adaptive_v2_embedding_controls(
        args=args,
        clean_params=clean_params,
        clean_importance=clean_importance,
        clean_sensitivity=clean_sensitivity,
    )
    effective_target_magnitude = float(adaptive_controls["target_magnitude"])
    alpha_delta = float(adaptive_controls["alpha_delta"])
    theta_delta = float(getattr(args, "theta_delta", 0.05))
    lum_delta = float(getattr(args, "lum_delta", 0.008))

    base_max_modification_ratio = float(adaptive_controls["max_modification_ratio"])
    render_chunk_size = _memory_safe_render_chunk_size(clean_renderer.num_gaussians, grid[..., 0])
    clean_render_reference = clean_renderer(
        grid[..., 0],
        grid[..., 1],
        clamp_output=False,
        chunk_size=render_chunk_size,
    ).detach()

    detector_training_requested = (
        _uses_native_parameter_embedding(args.baseline_type)
        and _native_detector_requested(args)
    )
    wrong_key_training_requested = (
        _supports_keyed_assignment_family(args.baseline_type)
        and _native_wrong_key_requested(args)
    )
    attack_training_requested = (
        _uses_native_parameter_embedding(args.baseline_type)
        and _native_attack_training_requested(args)
    )
    native_main_method = str(args.baseline_type) == "native"
    adaptive_encoder = SensitivityAdaptiveEncoder()
    selection_field_builder = (
        build_soft_selection_field
        if _supports_keyed_assignment_family(args.baseline_type)
        else build_unkeyed_selection_field
    )
    assignment_builder = {
        "native": build_v3_cost_code_assignments,
        "native_matched_random": build_v3_random_cost_code_assignments,
        "native_greedy_nokey": build_v3_greedy_nokey_cost_code_assignments,
        "native_random_polarity": build_v3_random_polarity_cost_code_assignments,
    }.get(args.baseline_type, build_v3_cost_code_assignments)

    def _selected_indices_by_channel(assignments_local: list[dict]) -> dict[str, torch.Tensor]:
        selected: dict[str, set[int]] = {
            "log_anisotropy": set(),
            "alpha": set(),
            "theta": set(),
            "color_lum": set(),
        }
        for assignment in assignments_local:
            for action, mask in zip(assignment.get("candidate_actions", []), assignment.get("selected_mask", [])):
                if float(mask) <= 0.5:
                    continue
                channel = str(action.get("channel", "log_anisotropy"))
                if channel in selected:
                    selected[channel].add(int(action["gaussian_index"]))
        return {
            key: torch.tensor(sorted(indices), dtype=torch.long)
            for key, indices in selected.items()
        }

    def _action_losses(
        action_scores: list[torch.Tensor],
        assignments_local: list[dict],
        *,
        beta: float = 4.0,
        decision_threshold: float = 0.5,
        margin: float = 0.25,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        losses: list[torch.Tensor] = []
        margins: list[torch.Tensor] = []
        for scores, assignment in zip(action_scores, assignments_local):
            if scores.numel() == 0:
                continue
            selected = torch.tensor(
                assignment["selected_mask"],
                dtype=scores.dtype,
                device=scores.device,
            )
            losses.append(
                F.binary_cross_entropy_with_logits(
                    beta * (scores - float(decision_threshold)),
                    selected,
                )
            )
            signed_scores = (2.0 * selected - 1.0) * (scores - float(decision_threshold))
            margins.append(torch.relu(float(margin) - signed_scores).mean())
        zero = clean_params["offsets"].new_zeros(1).squeeze()
        return (
            torch.stack(losses).mean() if losses else zero,
            torch.stack(margins).mean() if margins else zero,
        )

    def _decoder_channel_weights(weights: dict[str, float]) -> dict[str, float]:
        if str(getattr(args, "decoder_variant", "v3_full")) == "v3_full":
            return dict(weights)
        compat = {channel: 0.0 for channel in VALID_CARRIER_CHANNELS_V3}
        compat["log_anisotropy"] = float(weights.get("log_anisotropy", 1.0))
        compat["alpha"] = float(weights.get("alpha", 1.0))
        return compat

    def _decode_probs(
        params_local: dict[str, torch.Tensor],
        assignments_local: list[dict],
        weights_local: dict[str, float],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return cost_code_block_probabilities_v3(
            params_local,
            clean_params=clean_params,
            assignments=assignments_local,
            channel_weights=_decoder_channel_weights(weights_local),
            log_delta=float(effective_target_magnitude),
            alpha_delta=alpha_delta,
            theta_delta=theta_delta,
            lum_delta=lum_delta,
            beta=4.0,
        )

    def _build_v3_pack(
        *,
        key: str,
        raw_bits_local: torch.Tensor,
        structure_profile: str,
        active_channels_local: list[str],
        selection_temperature: float,
        max_modification_ratio: float,
        keep_fraction: float,
        log_delta: float,
        alpha_delta_local: float,
        theta_delta_local: float,
        lum_delta_local: float,
    ) -> dict[str, object]:
        return _build_v3_pack_from_policy(
            target=target,
            clean_params=clean_params,
            clean_importance=clean_importance,
            clean_sensitivity=clean_sensitivity,
            raw_bits_local=raw_bits_local,
            args=args,
            key=key,
            assignment_builder=assignment_builder,
            active_channels_local=active_channels_local,
            structure_profile=structure_profile,
            profile_policy=profile_policy,
            selection_temperature=selection_temperature,
            max_modification_ratio=max_modification_ratio,
            keep_fraction=keep_fraction,
            log_delta=log_delta,
            alpha_delta=alpha_delta_local,
            theta_delta=theta_delta_local,
            lum_delta=lum_delta_local,
            coding_strategy=coding_strategy,
        )

    prototype_bundle = build_v3_channel_costs(
        clean_params=clean_params,
        importance=clean_importance.to(clean_params["offsets"].device),
        sensitivity={k: v.to(clean_params["offsets"].device) for k, v in clean_sensitivity.items()},
        wet_threshold=float(args.wet_threshold),
        log_delta=float(effective_target_magnitude),
        alpha_delta=alpha_delta,
        theta_delta=theta_delta,
        lum_delta=lum_delta,
        cost_sensitivity_w=float(getattr(args, "cost_sensitivity_weight", 0.40)),
        cost_density_w=float(getattr(args, "cost_density_weight", 0.30)),
        cost_visual_w=float(getattr(args, "cost_visual_weight", 0.20)),
        cost_detector_w=float(getattr(args, "cost_detector_weight", 0.10)),
    )
    requested_payload_bits = int(raw_bits.numel())
    estimated_capacity = (
        int(estimate_embedding_capacity(prototype_bundle))
        if bool(getattr(args, "auto_payload_bits", False))
        else requested_payload_bits
    )
    reliable_carriers = int(prototype_bundle.n_carriers_below_cost_threshold(0.50))
    effective_payload_bits = estimated_capacity if bool(getattr(args, "auto_payload_bits", False)) else requested_payload_bits
    base_profile_for_fallback = _classify_structure_profile(
        target,
        clean_params,
        requested=str(getattr(args, "structure_profile", "auto")),
    )[0]
    effective_payload_bits, payload_fallback_triggered, payload_fallback_reasons = _resolve_payload_fallback(
        requested_payload_bits=requested_payload_bits,
        effective_payload_bits=effective_payload_bits,
        estimated_capacity=estimated_capacity,
        reliable_carriers=reliable_carriers,
        structure_profile=base_profile_for_fallback,
    )

    requested_raw_bits = raw_bits.clone()

    def _set_effective_payload_bits(payload_bits: int) -> torch.Tensor:
        nonlocal effective_payload_bits
        effective_payload_bits = int(payload_bits)
        return _resize_payload_bits(requested_raw_bits, effective_payload_bits)

    def _mark_payload_fallback(reason: str) -> bool:
        nonlocal payload_fallback_triggered
        if requested_payload_bits < 16 or effective_payload_bits <= V3_1_FALLBACK_PAYLOAD_BITS:
            return False
        payload_fallback_triggered = True
        if reason not in payload_fallback_reasons:
            payload_fallback_reasons.append(reason)
        return True

    def _evaluate_publishable_checkpoint(
        *,
        params_local: dict[str, torch.Tensor],
        raw_bits_local: torch.Tensor,
        assignments_local: list[dict],
        channel_weights_local: dict[str, float],
        log_delta_local: float,
        alpha_delta_local: float,
        theta_delta_local: float,
        lum_delta_local: float,
        wrong_key_groups_local: list[list[dict]] | None = None,
        quantize_consistency: bool = False,
        quantize_step: float = 0.01,
    ) -> dict[str, object]:
        def _decode_clean_ber(eval_params: dict[str, torch.Tensor]) -> float | None:
            try:
                decoded_raw_bits, _, _ = decode_cost_code_v3(
                    eval_params,
                    clean_params=clean_params,
                    assignments=assignments_local,
                    raw_length=int(raw_bits_local.numel()),
                    channel_weights=channel_weights_local,
                    log_delta=float(log_delta_local),
                    alpha_delta=float(alpha_delta_local),
                    theta_delta=float(theta_delta_local),
                    lum_delta=float(lum_delta_local),
                )
            except Exception:
                return None
            return float((decoded_raw_bits != raw_bits_local.to(decoded_raw_bits.device)).float().mean().item())

        def _decode_wrong_key_stats(eval_params: dict[str, torch.Tensor]) -> tuple[float | None, int, int]:
            wrong_key_mean_ber_local: float | None = None
            wrong_key_attempts_local = 0
            wrong_key_failures_local = 0
            if wrong_key_groups_local:
                wrong_key_bers: list[float] = []
                for wrong_group in wrong_key_groups_local:
                    wrong_key_attempts_local += 1
                    try:
                        wrong_raw_bits, _, _ = decode_cost_code_v3(
                            eval_params,
                            clean_params=clean_params,
                            assignments=wrong_group,
                            raw_length=int(raw_bits_local.numel()),
                            channel_weights=channel_weights_local,
                            log_delta=float(log_delta_local),
                            alpha_delta=float(alpha_delta_local),
                            theta_delta=float(theta_delta_local),
                            lum_delta=float(lum_delta_local),
                        )
                    except Exception:
                        wrong_key_failures_local += 1
                        continue
                    wrong_key_bers.append(
                        float((wrong_raw_bits != raw_bits_local.to(wrong_raw_bits.device)).float().mean().item())
                    )
                if wrong_key_bers:
                    wrong_key_mean_ber_local = float(sum(wrong_key_bers) / len(wrong_key_bers))
                elif wrong_key_attempts_local > 0:
                    _warn_once(
                        "wrong_key_decode_all_failed",
                        "[v3-decode-warn] all wrong-key checkpoint decodes failed; "
                        "wrong-key BER will be recorded as missing for this metric.",
                    )
            return wrong_key_mean_ber_local, wrong_key_attempts_local, wrong_key_failures_local

        clean_ber = _decode_clean_ber(params_local)

        rendered = _render_from_physical_params(params_local, grid[..., 0], grid[..., 1]).clamp(0.0, 1.0)
        rendered_psnr = compute_psnr(rendered, target)
        rendered_psnr_value = float(rendered_psnr.item() if hasattr(rendered_psnr, "item") else rendered_psnr)
        clean_psnr_drop = float(max(0.0, float(clean_clamped_psnr) - rendered_psnr_value))

        wrong_key_mean_ber, wrong_key_attempts, wrong_key_failures = _decode_wrong_key_stats(params_local)

        quantized_clean_ber: float | None = None
        quantized_clean_psnr_drop: float | None = None
        quantized_wrong_key_mean_ber: float | None = None
        quantized_wrong_key_attempts: int | None = None
        quantized_wrong_key_failures: int | None = None
        if quantize_consistency:
            quantized_params = {
                **params_local,
                "offsets": _straight_through_quantize(params_local["offsets"], step=quantize_step),
                "scales": _straight_through_quantize(params_local["scales"], step=quantize_step, min_value=1e-6),
                "thetas": _straight_through_quantize(params_local["thetas"], step=quantize_step),
                "colors": _straight_through_quantize(
                    params_local["colors"],
                    step=quantize_step,
                    min_value=0.0,
                    max_value=1.0,
                ),
                "opacities": _straight_through_quantize(
                    params_local["opacities"],
                    step=quantize_step,
                    min_value=0.0,
                    max_value=1.0,
                ),
            }
            quantized_clean_ber = _decode_clean_ber(quantized_params)
            (
                quantized_wrong_key_mean_ber,
                quantized_wrong_key_attempts,
                quantized_wrong_key_failures,
            ) = _decode_wrong_key_stats(quantized_params)
            quantized_rendered = _render_from_physical_params(
                quantized_params,
                grid[..., 0],
                grid[..., 1],
            ).clamp(0.0, 1.0)
            quantized_rendered_psnr = compute_psnr(quantized_rendered, target)
            quantized_rendered_psnr_value = float(
                quantized_rendered_psnr.item()
                if hasattr(quantized_rendered_psnr, "item")
                else quantized_rendered_psnr
            )
            quantized_clean_psnr_drop = float(
                max(0.0, float(clean_clamped_psnr) - quantized_rendered_psnr_value)
            )

        return {
            "clean_ber": None if clean_ber is None else float(clean_ber),
            "clean_psnr_drop": float(clean_psnr_drop),
            "wrong_key_mean_ber": None if wrong_key_mean_ber is None else float(wrong_key_mean_ber),
            "wrong_key_decode_attempts": int(wrong_key_attempts),
            "wrong_key_decode_failures": int(wrong_key_failures),
            "quantized_clean_ber": None if quantized_clean_ber is None else float(quantized_clean_ber),
            "quantized_clean_psnr_drop": (
                None if quantized_clean_psnr_drop is None else float(quantized_clean_psnr_drop)
            ),
            "quantized_wrong_key_mean_ber": (
                None
                if quantized_wrong_key_mean_ber is None
                else float(quantized_wrong_key_mean_ber)
            ),
            "quantized_wrong_key_decode_attempts": quantized_wrong_key_attempts,
            "quantized_wrong_key_decode_failures": quantized_wrong_key_failures,
        }

    def _make_proxy_stage_scorer(
        raw_bits_local: torch.Tensor,
    ) -> Callable[[dict[str, object], dict[str, object]], dict[str, object]]:
        def _score_stage(pack_local: dict[str, object], stage_local: dict[str, object]) -> dict[str, object]:
            proxy_steps = int(_publishable_proxy_tune_steps(str(pack_local["structure_profile"])))
            if proxy_steps <= 0:
                return {
                    "proxy_tune_steps": 0,
                    "proxy_clean_ber": None,
                    "proxy_clean_psnr_drop": None,
                    "proxy_wrong_key_mean_ber": None,
                }

            proxy_renderer = deepcopy(clean_renderer)
            for parameter in proxy_renderer.parameters():
                parameter.requires_grad_(False)
            proxy_renderer.raw_scales.requires_grad_(True)
            proxy_renderer.raw_opacities.requires_grad_(True)
            proxy_has_theta = hasattr(proxy_renderer, "raw_thetas") and ("theta" in list(pack_local["active_channels"]))
            proxy_has_colors = hasattr(proxy_renderer, "raw_colors") and ("color_lum" in list(pack_local["active_channels"]))
            proxy_optimizer_params = [proxy_renderer.raw_scales, proxy_renderer.raw_opacities]
            if proxy_has_theta:
                proxy_renderer.raw_thetas.requires_grad_(True)
                proxy_optimizer_params.append(proxy_renderer.raw_thetas)
            if proxy_has_colors:
                proxy_renderer.raw_colors.requires_grad_(True)
                proxy_optimizer_params.append(proxy_renderer.raw_colors)
            proxy_optimizer = torch.optim.Adam(proxy_optimizer_params, lr=float(args.tune_lr))

            proxy_assignments = list(pack_local["assignments"])
            proxy_channel_weights = dict(pack_local["channel_weights"])
            proxy_structure_profile = str(pack_local["structure_profile"])
            proxy_active_channels = list(pack_local["active_channels"])
            proxy_assignment_meta = dict(pack_local["assignment_meta"])
            proxy_detector_clean_features = build_detector_feature_vector(
                clean_params,
                importance=clean_importance.to(clean_params["offsets"].device),
            )
            proxy_wavelet = float(stage_local["wavelet_band_loss_weight"])
            proxy_stripe = float(stage_local["stripe_luminance_loss_weight"])
            proxy_log_delta = float(stage_local["log_delta"])
            proxy_alpha_delta = float(stage_local["alpha_delta"])
            proxy_lum_delta = float(stage_local["lum_delta"])
            proxy_msg_strength_scale = float(stage_local["resolved_msg_strength_scale"])
            proxy_native_margin_scale = float(stage_local["resolved_native_margin_scale"])
            proxy_wrong_key_scale = float(stage_local["resolved_wrong_key_scale"])
            proxy_attack_start_frac = float(stage_local["resolved_attack_start_frac"])
            proxy_effective_action_budget = int(proxy_assignment_meta.get("effective_action_budget", 0))

            proxy_detector_training_enabled = detector_training_requested and proxy_effective_action_budget >= 128
            proxy_wrong_key_training_enabled = wrong_key_training_requested and proxy_effective_action_budget >= 128
            proxy_attack_training_enabled = attack_training_requested and proxy_effective_action_budget >= 128
            proxy_pilot_clean_ber = _pilot_clean_ber(
                clean_params=clean_params,
                assignments=proxy_assignments,
                channel_weights=proxy_channel_weights,
                raw_bits=raw_bits_local,
                log_delta=float(proxy_log_delta),
                alpha_delta=float(proxy_alpha_delta),
                theta_delta=float(theta_delta),
                lum_delta=float(proxy_lum_delta),
            )
            proxy_pilot_clean_psnr_drop = _pilot_clean_psnr_drop(
                target=target,
                clean_params=clean_params,
                assignments=proxy_assignments,
                log_delta=float(proxy_log_delta),
                alpha_delta=float(proxy_alpha_delta),
                theta_delta=float(theta_delta),
                lum_delta=float(proxy_lum_delta),
            )
            proxy_pilot_warm_start_used = False
            if _pilot_candidate_is_publishable(
                pilot_clean_ber=float(proxy_pilot_clean_ber),
                pilot_clean_psnr_drop=float(proxy_pilot_clean_psnr_drop),
            ):
                proxy_pilot_params = _apply_v3_selected_actions_to_params(
                    clean_params,
                    proxy_assignments,
                    log_delta=float(proxy_log_delta),
                    alpha_delta=float(proxy_alpha_delta),
                    theta_delta=float(theta_delta),
                    lum_delta=float(proxy_lum_delta),
                )
                _load_renderer_from_physical_params(proxy_renderer, proxy_pilot_params)
                proxy_pilot_warm_start_used = True

            proxy_density_reference = build_density_reference(clean_params)
            proxy_clean_moment_ref = build_clean_distribution_reference(clean_params)
            proxy_selection_kwargs = {
                "clean_params": clean_params,
                "importance": clean_importance.to(clean_params["offsets"].device),
                "sensitivity": {k: v.to(clean_params["offsets"].device) for k, v in clean_sensitivity.items()},
                "cost_bundle": _build_pack_selection_proxy_bundle(
                    pack=pack_local,
                    periodic_lowfreq_carrier_bias_enabled=bool(
                        getattr(args, "periodic_lowfreq_carrier_bias", False)
                    ),
                    periodic_bias_strength=float(getattr(args, "periodic_bias_strength", 0.35)),
                ),
                "selection_temperature": float(pack_local["selection_temperature"]),
            }
            if selection_field_builder is build_soft_selection_field:
                proxy_selection_bundle = selection_field_builder(key=args.key, **proxy_selection_kwargs)
            else:
                proxy_selection_bundle = selection_field_builder(**proxy_selection_kwargs)

            proxy_selected_by_channel = _selected_indices_by_channel(proxy_assignments)
            proxy_selected_log_indices = proxy_selected_by_channel["log_anisotropy"].to(clean_params["offsets"].device)
            proxy_selected_alpha_indices = proxy_selected_by_channel["alpha"].to(clean_params["offsets"].device)
            proxy_selected_theta_indices = proxy_selected_by_channel["theta"].to(clean_params["offsets"].device)
            proxy_selected_lum_indices = proxy_selected_by_channel["color_lum"].to(clean_params["offsets"].device)
            proxy_compensation_indices, _, proxy_compensation_alpha_only = _build_compensation_plan(
                clean_params["offsets"],
                assignments_local=proxy_assignments,
                wet_mask=pack_local["v3_bundle"].wet_mask.to(clean_params["offsets"].device),
                region_policy_context=dict(pack_local["region_policy_context"]),
                structure_profile=proxy_structure_profile,
                periodic_bias_mode=str(pack_local.get("periodic_bias_mode", "legacy")),
                periodic_extra_compensation_per_active=int(
                    getattr(args, "periodic_extra_compensation_per_active", 0)
                ),
            )

            proxy_raw_targets = raw_bits_local.to(clean_params["offsets"].device).to(torch.float32)
            if proxy_raw_targets.numel() % 4 != 0:
                proxy_raw_targets = torch.cat(
                    [
                        proxy_raw_targets,
                        torch.zeros(
                            (-proxy_raw_targets.numel()) % 4,
                            dtype=proxy_raw_targets.dtype,
                            device=proxy_raw_targets.device,
                        ),
                    ]
                )
            proxy_block_targets = proxy_raw_targets.view(-1, 4).to(clean_params["offsets"].device)

            proxy_detector = RunLevelStegaDetector(input_dim=int(proxy_detector_clean_features.numel())).to(
                clean_params["offsets"].device
            )
            proxy_detector_optimizer = torch.optim.Adam(proxy_detector.parameters(), lr=0.01)
            proxy_effective_detector_weight = defaults["detector_weight"] if proxy_detector_training_enabled else 0.0
            proxy_sigma_ratio_b = float(getattr(args, "hardening_sigma_ratio", 0.30))
            proxy_sigma_ratio_c = float(getattr(args, "hardening_sigma_ratio_c", 0.50))
            if proxy_effective_action_budget < 32:
                proxy_sigma_ratio_c = proxy_sigma_ratio_b
            proxy_start_iter = int(getattr(args, "hardening_start_iter", 100))
            proxy_end_iter = int(getattr(args, "hardening_end_iter", 180))
            proxy_hardening_enabled = proxy_sigma_ratio_b > 0.0
            proxy_hardened_decoder = PerturbationHardenedDecoder(
                sigma_ratio_b=proxy_sigma_ratio_b,
                sigma_ratio_c=proxy_sigma_ratio_c,
                start_iter=proxy_start_iter,
                end_iter=proxy_end_iter,
                min_block_prob=0.55,
            )

            proxy_ot_weight = float(getattr(args, "ot_covertness_weight", 0.15))
            proxy_sinkhorn_blur = float(getattr(args, "sinkhorn_blur", 0.05))
            proxy_sinkhorn_enabled = proxy_ot_weight > 0.0 and proxy_effective_action_budget >= 32
            proxy_sinkhorn_term = SinkhornCovertnessTerm(
                blur=proxy_sinkhorn_blur,
                weight=proxy_ot_weight,
                n_iter=50,
                max_carriers=512,
            )

            proxy_wrong_key_groups: list[list[dict]] = []
            if proxy_wrong_key_training_enabled:
                for proxy_index in range(min(8, int(args.wrong_key_samples))):
                    try:
                        proxy_wrong_key_groups.append(
                            _build_v3_pack_from_policy(
                                target=target,
                                clean_params=clean_params,
                                clean_importance=clean_importance,
                                clean_sensitivity=clean_sensitivity,
                                raw_bits_local=raw_bits_local,
                                args=args,
                                key=f"{args.key}::proxy_wrong::{proxy_index}",
                                assignment_builder=build_v3_cost_code_assignments,
                                active_channels_local=proxy_active_channels,
                                structure_profile=proxy_structure_profile,
                                profile_policy=profile_policy,
                                selection_temperature=float(stage_local["selection_temperature"]),
                                max_modification_ratio=float(stage_local["max_modification_ratio"]),
                                keep_fraction=float(stage_local["keep_fraction"]),
                                log_delta=proxy_log_delta,
                                alpha_delta=proxy_alpha_delta,
                                theta_delta=theta_delta,
                                lum_delta=proxy_lum_delta,
                                coding_strategy=coding_strategy,
                            )["assignments"]
                        )
                    except RuntimeError:
                        continue

            proxy_attack_schedule = ["quantize_step_0.01", "small_parameter_noise_0.01", "importance_pruning_10pct"]
            proxy_small_noise_gen = torch.Generator(device="cpu")
            proxy_small_noise_gen.manual_seed(
                key_to_seed(f"{args.key}::proxy_attack_noise::{args.message_seed}::{args.fit_seed}") + 17
            )
            proxy_attack_noise_log = (
                torch.empty((proxy_selected_log_indices.numel(), 2), dtype=clean_params["scales"].dtype)
                .uniform_(-0.01, 0.01, generator=proxy_small_noise_gen)
                if proxy_selected_log_indices.numel() > 0 else None
            )
            proxy_attack_noise_alpha = (
                torch.empty((proxy_selected_alpha_indices.numel(), 1), dtype=clean_params["opacities"].dtype)
                .uniform_(-0.01, 0.01, generator=proxy_small_noise_gen)
                if proxy_selected_alpha_indices.numel() > 0 else None
            )
            proxy_attack_noise_theta = (
                torch.empty((proxy_selected_theta_indices.numel(), 1), dtype=clean_params["thetas"].dtype)
                .uniform_(-0.01, 0.01, generator=proxy_small_noise_gen)
                if proxy_selected_theta_indices.numel() > 0 else None
            )
            proxy_attack_noise_lum = (
                torch.empty((proxy_selected_lum_indices.numel(), 1), dtype=clean_params["colors"].dtype)
                .uniform_(-0.01, 0.01, generator=proxy_small_noise_gen)
                if proxy_selected_lum_indices.numel() > 0 else None
            )
            proxy_luma_w = torch.tensor(
                [0.299, 0.587, 0.114],
                device=clean_params["offsets"].device,
                dtype=clean_params["offsets"].dtype,
            )
            proxy_luma_w = proxy_luma_w / max(float((proxy_luma_w * proxy_luma_w).sum().item()), 1e-6)

            def _proxy_decode_probs(
                params_local: dict[str, torch.Tensor],
                assignments_local: list[dict],
                weights_local: dict[str, float],
            ) -> tuple[torch.Tensor, list[torch.Tensor]]:
                return cost_code_block_probabilities_v3(
                    params_local,
                    clean_params=clean_params,
                    assignments=assignments_local,
                    channel_weights=_decoder_channel_weights(weights_local),
                    log_delta=float(proxy_log_delta),
                    alpha_delta=float(proxy_alpha_delta),
                    theta_delta=float(theta_delta),
                    lum_delta=float(proxy_lum_delta),
                    beta=4.0,
                )

            def _proxy_true_key_margin_terms(
                block_probs_local: torch.Tensor,
                action_scores_local: list[torch.Tensor],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                target_sign = 2.0 * proxy_block_targets - 1.0
                bit_alignment = target_sign * (2.0 * block_probs_local - 1.0)
                true_key_margin = bit_alignment.mean()
                bit_margin_penalty = torch.relu(0.65 - bit_alignment).mean()
                mean_margin_penalty = torch.relu(
                    torch.tensor(0.55, dtype=block_probs_local.dtype, device=block_probs_local.device)
                    - true_key_margin
                )
                action_losses: list[torch.Tensor] = []
                for scores, assignment in zip(action_scores_local, proxy_assignments):
                    if scores.numel() == 0:
                        continue
                    selected = torch.tensor(
                        assignment["selected_mask"],
                        dtype=torch.float32,
                        device=scores.device,
                    ) > 0.5
                    selected_loss = (
                        torch.relu(0.85 - scores[selected]).mean()
                        if selected.any()
                        else scores.new_zeros(1).squeeze()
                    )
                    unselected_loss = (
                        torch.relu(scores[~selected] + 0.05).mean()
                        if (~selected).any()
                        else scores.new_zeros(1).squeeze()
                    )
                    action_losses.append(selected_loss + 0.50 * unselected_loss)
                action_penalty = (
                    torch.stack(action_losses).mean()
                    if action_losses
                    else block_probs_local.new_zeros(1).squeeze()
                )
                return true_key_margin, bit_margin_penalty, mean_margin_penalty, action_penalty

            def _proxy_wrong_key_terms(
                params_local: dict[str, torch.Tensor],
                correct_block_probs: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                zero = params_local["offsets"].new_zeros(1).squeeze()
                if not proxy_wrong_key_groups:
                    return zero, zero, zero, zero
                correct_alignment = (2.0 * proxy_block_targets - 1.0) * (2.0 * correct_block_probs - 1.0)
                correct_conf = torch.abs(correct_block_probs - 0.5).mean() * 2.0
                wrong_uniform_losses: list[torch.Tensor] = []
                wrong_margin_caps: list[torch.Tensor] = []
                wrong_gap_penalties: list[torch.Tensor] = []
                wrong_confs: list[torch.Tensor] = []
                for wrong_group in proxy_wrong_key_groups:
                    wrong_probs, _ = _proxy_decode_probs(params_local, wrong_group, proxy_channel_weights)
                    wrong_align = torch.abs((2.0 * proxy_block_targets - 1.0) * (2.0 * wrong_probs - 1.0))
                    wrong_uniform_losses.append(F.binary_cross_entropy(wrong_probs, torch.full_like(wrong_probs, 0.5)))
                    wrong_margin_caps.append(torch.relu(wrong_align - 0.08).mean())
                    wrong_gap_penalties.append(torch.relu(0.40 - (correct_alignment - wrong_align)).mean())
                    wrong_confs.append(torch.abs(wrong_probs - 0.5).mean() * 2.0)
                wrong_key_loss = torch.stack(wrong_uniform_losses).mean()
                wrong_conf = torch.stack(wrong_confs).mean()
                if native_main_method:
                    return (
                        wrong_key_loss,
                        torch.relu(0.65 - correct_alignment).mean(),
                        torch.stack(wrong_margin_caps).mean(),
                        torch.stack(wrong_gap_penalties).mean(),
                    )
                return (
                    wrong_key_loss,
                    torch.relu(0.35 - correct_conf),
                    torch.relu(wrong_conf - 0.10),
                    torch.relu(0.20 - (correct_conf - wrong_conf)),
                )

            def _proxy_attack_params(
                params_local: dict[str, torch.Tensor],
                *,
                attack_name: str,
            ) -> dict[str, torch.Tensor]:
                attacked = {key: value for key, value in params_local.items()}
                if attack_name == "quantize_step_0.01":
                    attacked = {
                        **attacked,
                        "offsets": _straight_through_quantize(attacked["offsets"], step=0.01),
                        "scales": _straight_through_quantize(attacked["scales"], step=0.01, min_value=1e-6),
                        "thetas": _straight_through_quantize(attacked["thetas"], step=0.01),
                        "colors": _straight_through_quantize(
                            attacked["colors"],
                            step=0.01,
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "opacities": _straight_through_quantize(
                            attacked["opacities"],
                            step=0.01,
                            min_value=0.0,
                            max_value=1.0,
                        ),
                    }
                elif attack_name == "small_parameter_noise_0.01":
                    attacked = {key: value.clone() for key, value in attacked.items()}
                    if proxy_selected_log_indices.numel() > 0 and proxy_attack_noise_log is not None:
                        attacked["scales"][proxy_selected_log_indices] = torch.clamp(
                            attacked["scales"][proxy_selected_log_indices]
                            + proxy_attack_noise_log.to(attacked["scales"].device),
                            min=1e-6,
                        )
                    if proxy_selected_alpha_indices.numel() > 0 and proxy_attack_noise_alpha is not None:
                        attacked["opacities"][proxy_selected_alpha_indices] = torch.clamp(
                            attacked["opacities"][proxy_selected_alpha_indices]
                            + proxy_attack_noise_alpha.to(attacked["opacities"].device),
                            0.0,
                            1.0,
                        )
                    if proxy_selected_theta_indices.numel() > 0 and proxy_attack_noise_theta is not None:
                        attacked["thetas"][proxy_selected_theta_indices] = (
                            attacked["thetas"][proxy_selected_theta_indices]
                            + proxy_attack_noise_theta.to(attacked["thetas"].device)
                        )
                    if proxy_selected_lum_indices.numel() > 0 and proxy_attack_noise_lum is not None:
                        lum_step = proxy_attack_noise_lum.to(attacked["colors"].device) * proxy_luma_w.view(1, 3)
                        attacked["colors"][proxy_selected_lum_indices] = torch.clamp(
                            attacked["colors"][proxy_selected_lum_indices] + lum_step,
                            0.0,
                            1.0,
                        )
                elif attack_name == "importance_pruning_10pct":
                    attacked = {key: value.clone() for key, value in attacked.items()}
                    drop_count = max(1, int(round(0.10 * float(attacked["offsets"].shape[0]))))
                    proxy_importance_local = compute_gaussian_importance(attacked["scales"], attacked["opacities"])
                    drop_indices = torch.argsort(proxy_importance_local, descending=False)[:drop_count]
                    attacked["opacities"][drop_indices, 0] = 0.0
                else:
                    raise ValueError(f"unsupported carrier-preserving attack: {attack_name}")
                attacked["importance"] = compute_gaussian_importance(attacked["scales"], attacked["opacities"])
                return attacked

            checkpoint_quantize_consistency = _periodic_bias_uses_quantize_consistency(
                str(pack_local.get("periodic_bias_mode", "legacy"))
            )
            proxy_checkpoint_schedule_mode = _publishable_checkpoint_schedule_mode(
                str(pack_local["structure_profile"]),
                proxy=True,
            )
            best_proxy_metrics: dict[str, float | None] | None = None
            best_proxy_rank: tuple[object, ...] | None = None
            proxy_attack_offset = int(proxy_attack_start_frac * max(1, proxy_steps))

            def _consider_proxy_checkpoint(
                *,
                step_value: int,
                metrics: dict[str, float | None],
            ) -> None:
                nonlocal best_proxy_metrics
                nonlocal best_proxy_rank

                proxy_rank = _publishable_checkpoint_rank(
                    clean_ber=metrics["clean_ber"],
                    clean_psnr_drop=metrics["clean_psnr_drop"],
                    wrong_key_mean_ber=metrics["wrong_key_mean_ber"],
                    quantized_clean_ber=metrics.get("quantized_clean_ber"),
                    quantized_clean_psnr_drop=metrics.get("quantized_clean_psnr_drop"),
                    quantized_wrong_key_mean_ber=metrics.get("quantized_wrong_key_mean_ber"),
                    quantize_consistency=checkpoint_quantize_consistency,
                    step=step_value,
                )
                if best_proxy_rank is None or proxy_rank < best_proxy_rank:
                    best_proxy_rank = proxy_rank
                    best_proxy_metrics = dict(metrics)

            pilot_proxy_metrics: dict[str, object] | None = None
            if proxy_pilot_warm_start_used:
                pilot_proxy_metrics = _evaluate_publishable_checkpoint(
                    params_local=proxy_renderer.decode_parameters(),
                    raw_bits_local=raw_bits_local,
                    assignments_local=proxy_assignments,
                    channel_weights_local=proxy_channel_weights,
                    log_delta_local=proxy_log_delta,
                    alpha_delta_local=proxy_alpha_delta,
                    theta_delta_local=float(theta_delta),
                    lum_delta_local=proxy_lum_delta,
                    wrong_key_groups_local=proxy_wrong_key_groups,
                    quantize_consistency=checkpoint_quantize_consistency,
                    quantize_step=0.01,
                )

            for proxy_step in range(proxy_steps):
                proxy_optimizer.zero_grad(set_to_none=True)
                proxy_reconstruction = proxy_renderer(
                    grid[..., 0],
                    grid[..., 1],
                    clamp_output=False,
                    chunk_size=render_chunk_size,
                )
                proxy_params = proxy_renderer.decode_parameters()
                proxy_importance = compute_gaussian_importance(proxy_params["scales"], proxy_params["opacities"])
                proxy_progress = (0.0 if proxy_steps <= 1 else float(proxy_step) / float(proxy_steps - 1))

                proxy_block_probs, proxy_action_scores = _proxy_decode_probs(
                    proxy_params,
                    proxy_assignments,
                    proxy_channel_weights,
                )
                proxy_action_target_loss, proxy_action_margin_loss = _action_losses(
                    proxy_action_scores,
                    proxy_assignments,
                )
                proxy_block_margin_loss = torch.relu(
                    0.35 - (2.0 * proxy_block_targets - 1.0) * (2.0 * proxy_block_probs - 1.0)
                ).mean()
                proxy_clean_block_bce = F.binary_cross_entropy(proxy_block_probs, proxy_block_targets)
                proxy_msg_loss = (
                    proxy_clean_block_bce
                    + 0.25 * proxy_action_target_loss
                    + 0.25 * proxy_action_margin_loss
                    + 0.25 * proxy_block_margin_loss
                )

                proxy_tk_push = proxy_tk_bit_pen = proxy_tk_mean_pen = proxy_tk_action_pen = (
                    proxy_params["offsets"].new_zeros(1).squeeze()
                )
                if native_main_method:
                    proxy_true_key_margin, proxy_tk_bit_pen, proxy_tk_mean_pen, proxy_tk_action_pen = _proxy_true_key_margin_terms(
                        proxy_block_probs,
                        proxy_action_scores,
                    )
                    proxy_tk_push = torch.clamp(1.0 - proxy_true_key_margin, min=0.0)

                proxy_hardened_loss = proxy_params["offsets"].new_zeros(1).squeeze()
                if proxy_hardening_enabled:
                    proxy_hardened_loss = proxy_hardened_decoder.hardened_loss_v3(
                        params=proxy_params,
                        clean_params=clean_params,
                        assignments=proxy_assignments,
                        channel_weights=_decoder_channel_weights(proxy_channel_weights),
                        log_delta=float(proxy_log_delta),
                        alpha_delta=float(proxy_alpha_delta),
                        theta_delta=float(theta_delta),
                        lum_delta=float(proxy_lum_delta),
                        current_iter=proxy_step,
                    )

                proxy_moment_loss = matched_moment_loss(proxy_params, proxy_clean_moment_ref)
                proxy_mmd_loss = mmd_distribution_loss(clean_params, proxy_params)
                proxy_wass_loss = sliced_wasserstein_loss(
                    clean_params,
                    proxy_params,
                    projections=32,
                    seed=args.message_seed + proxy_step,
                )
                proxy_sel_penalty = selection_channel_penalty(
                    selection_probs=proxy_selection_bundle.selection_probs.to(proxy_params["offsets"].device),
                    clean_params=clean_params,
                    importance=clean_importance.to(proxy_params["offsets"].device),
                    sensitivity={k: v.to(proxy_params["offsets"].device) for k, v in clean_sensitivity.items()},
                )
                proxy_det_features = build_detector_feature_vector(proxy_params, importance=proxy_importance)
                proxy_det_loss = detector_embed_loss(proxy_detector, proxy_det_features)
                if not proxy_detector_training_enabled:
                    proxy_det_loss = proxy_det_features.new_zeros(1).squeeze()
                proxy_ewc_loss = carrier_ewc_penalty(
                    proxy_renderer,
                    clean_renderer,
                    assignments=_v2_assignments_for_compensation(proxy_assignments),
                    sensitivity={k: v.to(proxy_params["offsets"].device) for k, v in clean_sensitivity.items()},
                    alpha_weight=float(args.alpha_weight),
                )
                proxy_comp_loss = density_reference_loss(proxy_params, proxy_density_reference)

                proxy_sinkhorn_loss = proxy_params["offsets"].new_zeros(1).squeeze()
                if proxy_sinkhorn_enabled:
                    try:
                        clean_feat, wm_feat = extract_modified_carriers(clean_params, proxy_params, pack_local["v3_bundle"])
                        if clean_feat.shape[0] >= 4:
                            proxy_sinkhorn_loss = proxy_sinkhorn_term(clean_feat, wm_feat)
                    except Exception as exc:
                        _warn_once("proxy_sinkhorn_loss", f"[v3-loss-warn] proxy Sinkhorn loss disabled after failure: {exc}")

                proxy_wavelet_loss = proxy_params["offsets"].new_zeros(1).squeeze()
                if proxy_wavelet > 0.0:
                    proxy_wavelet_loss = _wavelet_band_consistency_loss(clean_render_reference, proxy_reconstruction)
                proxy_stripe_loss = proxy_params["offsets"].new_zeros(1).squeeze()
                if proxy_stripe > 0.0:
                    proxy_stripe_loss = _stripe_luminance_smoothness_loss(
                        clean_render_reference,
                        proxy_reconstruction,
                        stripe_normal=list(pack_local["region_policy_context"].get("stripe_normal", [1.0, 0.0])),
                    )

                proxy_wk_loss = proxy_corr_pen = proxy_wrong_pen = proxy_gap_pen = (
                    proxy_params["offsets"].new_zeros(1).squeeze()
                )
                if proxy_wrong_key_training_enabled:
                    proxy_wk_loss, proxy_corr_pen, proxy_wrong_pen, proxy_gap_pen = _proxy_wrong_key_terms(
                        proxy_params,
                        proxy_block_probs,
                    )

                proxy_total_loss = F.mse_loss(proxy_reconstruction, target)
                proxy_total_loss = proxy_total_loss + defaults["msg_strength"] * proxy_msg_strength_scale * proxy_msg_loss
                proxy_total_loss = proxy_total_loss + float(args.covertness_weight) * (
                    defaults["mmd_weight"] * proxy_mmd_loss
                    + defaults["wasserstein_weight"] * proxy_wass_loss
                    + defaults["selection_weight"] * proxy_sel_penalty
                    + defaults["moment_weight"] * proxy_moment_loss
                    + proxy_effective_detector_weight * proxy_det_loss
                    + defaults["compensation_weight"] * proxy_comp_loss
                    + proxy_sinkhorn_loss
                    + proxy_wavelet * proxy_wavelet_loss
                    + proxy_stripe * proxy_stripe_loss
                )
                proxy_total_loss = proxy_total_loss + defaults["ewc_weight"] * proxy_ewc_loss
                if native_main_method:
                    proxy_total_loss = proxy_total_loss + defaults["native_margin_weight"] * proxy_native_margin_scale * (
                        proxy_tk_push + proxy_tk_bit_pen + proxy_tk_mean_pen + 0.5 * proxy_tk_action_pen
                    )
                if proxy_wrong_key_training_enabled:
                    if native_main_method:
                        proxy_total_loss = proxy_total_loss + defaults["wrong_key_weight"] * proxy_wrong_key_scale * proxy_wk_loss
                        proxy_total_loss = proxy_total_loss + defaults["native_wrong_key_specificity_weight"] * proxy_wrong_key_scale * (
                            proxy_corr_pen + proxy_wrong_pen + proxy_gap_pen
                        )
                    else:
                        proxy_total_loss = proxy_total_loss + defaults["wrong_key_weight"] * proxy_wrong_key_scale * (
                            proxy_wk_loss + proxy_corr_pen + proxy_wrong_pen + proxy_gap_pen
                        )
                if proxy_hardening_enabled:
                    proxy_total_loss = proxy_total_loss + 0.5 * defaults["msg_strength"] * proxy_msg_strength_scale * proxy_hardened_loss

                if proxy_attack_training_enabled and proxy_progress >= proxy_attack_start_frac:
                    proxy_attack_name = proxy_attack_schedule[(proxy_step - proxy_attack_offset) % len(proxy_attack_schedule)]
                    proxy_attacked_params = _proxy_attack_params(
                        proxy_params,
                        attack_name=proxy_attack_name,
                    )
                    proxy_attacked_probs, proxy_attacked_scores = _proxy_decode_probs(
                        proxy_attacked_params,
                        proxy_assignments,
                        proxy_channel_weights,
                    )
                    proxy_attacked_bce = F.binary_cross_entropy(proxy_attacked_probs, proxy_block_targets)
                    proxy_attacked_margin = torch.relu(
                        0.35 - (2.0 * proxy_block_targets - 1.0) * (2.0 * proxy_attacked_probs - 1.0)
                    ).mean()
                    proxy_total_loss = proxy_total_loss + 0.5 * defaults["msg_strength"] * proxy_msg_strength_scale * (
                        proxy_attacked_bce + 0.25 * proxy_attacked_margin
                    )
                    if native_main_method:
                        proxy_atk_margin, proxy_atk_bp, proxy_atk_mp, proxy_atk_ap = _proxy_true_key_margin_terms(
                            proxy_attacked_probs,
                            proxy_attacked_scores,
                        )
                        proxy_atk_push = torch.clamp(1.0 - proxy_atk_margin, min=0.0)
                        proxy_total_loss = proxy_total_loss + 0.5 * defaults["native_margin_weight"] * proxy_native_margin_scale * (
                            proxy_atk_push + proxy_atk_bp + proxy_atk_mp + 0.5 * proxy_atk_ap
                        )
                    if proxy_wrong_key_training_enabled:
                        proxy_a_wk, proxy_a_cp, proxy_a_wp, proxy_a_gp = _proxy_wrong_key_terms(
                            proxy_attacked_params,
                            proxy_attacked_probs,
                        )
                        if native_main_method:
                            proxy_total_loss = proxy_total_loss + 0.5 * defaults["wrong_key_weight"] * proxy_wrong_key_scale * proxy_a_wk
                            proxy_total_loss = proxy_total_loss + 0.5 * defaults["native_wrong_key_specificity_weight"] * proxy_wrong_key_scale * (
                                proxy_a_cp + proxy_a_wp + proxy_a_gp
                            )
                        else:
                            proxy_total_loss = proxy_total_loss + 0.5 * defaults["wrong_key_weight"] * proxy_wrong_key_scale * (
                                proxy_a_wk + proxy_a_cp + proxy_a_wp + proxy_a_gp
                            )

                proxy_total_loss.backward()
                if proxy_selected_log_indices.numel() > 0:
                    _masked_rows_step(proxy_renderer.raw_scales, proxy_selected_log_indices)
                if proxy_selected_alpha_indices.numel() > 0:
                    _masked_rows_step(proxy_renderer.raw_opacities, proxy_selected_alpha_indices)
                if proxy_has_theta and proxy_selected_theta_indices.numel() > 0:
                    _masked_rows_step(proxy_renderer.raw_thetas, proxy_selected_theta_indices)
                if proxy_has_colors and proxy_selected_lum_indices.numel() > 0:
                    _masked_rows_step(proxy_renderer.raw_colors, proxy_selected_lum_indices)
                proxy_grad_params = [parameter for parameter in proxy_optimizer_params if parameter.grad is not None]
                if proxy_grad_params:
                    torch.nn.utils.clip_grad_norm_(proxy_grad_params, max_norm=1.0)
                proxy_optimizer.step()

                if proxy_detector_training_enabled:
                    train_detector_steps(
                        proxy_detector,
                        proxy_detector_optimizer,
                        clean_features=proxy_detector_clean_features,
                        wm_features=proxy_det_features,
                        steps=2,
                    )

                if proxy_compensation_indices.numel() > 0 and (proxy_step + 1) % 8 == 0:
                    for _ in range(4):
                        proxy_optimizer.zero_grad(set_to_none=True)
                        proxy_recon2 = proxy_renderer(
                            grid[..., 0],
                            grid[..., 1],
                            clamp_output=False,
                            chunk_size=render_chunk_size,
                        )
                        proxy_params2 = proxy_renderer.decode_parameters()
                        proxy_imp2 = compute_gaussian_importance(proxy_params2["scales"], proxy_params2["opacities"])
                        proxy_det_features2 = build_detector_feature_vector(proxy_params2, importance=proxy_imp2)
                        proxy_selection_penalty2 = selection_channel_penalty(
                            selection_probs=proxy_selection_bundle.selection_probs.to(proxy_params2["offsets"].device),
                            clean_params=clean_params,
                            importance=clean_importance.to(proxy_params2["offsets"].device),
                            sensitivity={k: v.to(proxy_params2["offsets"].device) for k, v in clean_sensitivity.items()},
                        )
                        proxy_comp_obj = F.mse_loss(proxy_recon2, target) + float(args.covertness_weight) * (
                            defaults["mmd_weight"] * mmd_distribution_loss(clean_params, proxy_params2)
                            + defaults["wasserstein_weight"] * sliced_wasserstein_loss(
                                clean_params,
                                proxy_params2,
                                projections=16,
                                seed=proxy_step,
                            )
                            + defaults["selection_weight"] * proxy_selection_penalty2
                            + proxy_effective_detector_weight * detector_embed_loss(proxy_detector, proxy_det_features2)
                            + defaults["compensation_weight"] * density_reference_loss(
                                proxy_params2,
                                proxy_density_reference,
                            )
                        )
                        proxy_comp_obj.backward()
                        _masked_rows_step(proxy_renderer.raw_opacities, proxy_compensation_indices)
                        if not proxy_compensation_alpha_only:
                            _masked_rows_step(proxy_renderer.raw_scales, proxy_compensation_indices)
                            grad_targets = [proxy_renderer.raw_scales, proxy_renderer.raw_opacities]
                        else:
                            grad_targets = [proxy_renderer.raw_opacities]
                        torch.nn.utils.clip_grad_norm_(grad_targets, max_norm=1.0)
                        proxy_optimizer.step()

                if _should_evaluate_publishable_checkpoint(
                    step=proxy_step + 1,
                    total_steps=proxy_steps,
                    structure_profile=str(pack_local["structure_profile"]),
                    proxy=True,
                ):
                    proxy_metrics = _evaluate_publishable_checkpoint(
                        params_local=proxy_renderer.decode_parameters(),
                        raw_bits_local=raw_bits_local,
                        assignments_local=proxy_assignments,
                        channel_weights_local=proxy_channel_weights,
                        log_delta_local=proxy_log_delta,
                        alpha_delta_local=proxy_alpha_delta,
                        theta_delta_local=float(theta_delta),
                        lum_delta_local=proxy_lum_delta,
                        wrong_key_groups_local=proxy_wrong_key_groups,
                        quantize_consistency=checkpoint_quantize_consistency,
                        quantize_step=0.01,
                    )
                    _consider_proxy_checkpoint(
                        step_value=proxy_step + 1,
                        metrics=proxy_metrics,
                    )

            return {
                "proxy_tune_steps": int(proxy_steps),
                "proxy_clean_ber": None if best_proxy_metrics is None else best_proxy_metrics["clean_ber"],
                "proxy_clean_psnr_drop": None if best_proxy_metrics is None else best_proxy_metrics["clean_psnr_drop"],
                "proxy_wrong_key_mean_ber": None if best_proxy_metrics is None else best_proxy_metrics["wrong_key_mean_ber"],
                "proxy_wrong_key_decode_attempts": None if best_proxy_metrics is None else best_proxy_metrics.get("wrong_key_decode_attempts"),
                "proxy_wrong_key_decode_failures": None if best_proxy_metrics is None else best_proxy_metrics.get("wrong_key_decode_failures"),
                "proxy_checkpoint_schedule_mode": proxy_checkpoint_schedule_mode,
            }

        return _score_stage

    def _prepare_pack_for_payload(raw_bits_local: torch.Tensor) -> tuple[dict[str, object], dict[str, object], str, float]:
        # --- Source-level proxy cache for shared publishable policy ---
        # Under publishable_2d_v1, native and native_matched_random share the
        # same resolved stage/scales so the fair-pairing comparison_signature
        # matches. The cache is still payload-specific because proxy scoring
        # depends on the message bits used to resolve the source policy.
        _profile_policy_local = str(getattr(args, "profile_policy", "legacy"))
        _baseline_type_local = str(getattr(args, "baseline_type", "native"))
        _payload_signature_local = _payload_signature_from_bits(raw_bits_local)
        _use_cache = (
            _profile_policy_local == "publishable_2d_v1"
            and _baseline_type_local in {"native", "native_matched_random"}
        )
        _cached_policy = (
            _load_source_policy_cache(args, payload_signature=_payload_signature_local)
            if _use_cache
            else None
        )

        if _cached_policy is not None:
            # Cache hit: reuse the resolved stage/scales from native; rebuild
            # the pack with the current run's message bits so assignments are
            # correct for this condition's payload.
            source_policy_local = _cached_policy
            structure_profile_local = str(source_policy_local["resolved_structure_profile"])
            pack_local = _build_v3_pack(
                key=_primary_assignment_key(args),
                raw_bits_local=raw_bits_local,
                structure_profile=structure_profile_local,
                active_channels_local=list(source_policy_local["resolved_active_channels"]),
                selection_temperature=float(source_policy_local["resolved_selection_temperature"]),
                max_modification_ratio=float(source_policy_local["resolved_max_modification_ratio"]),
                keep_fraction=float(source_policy_local["resolved_keep_fraction"]),
                log_delta=float(source_policy_local["resolved_log_delta"]),
                alpha_delta_local=float(source_policy_local["resolved_alpha_delta"]),
                theta_delta_local=float(source_policy_local.get("resolved_theta_delta", theta_delta)),
                lum_delta_local=float(source_policy_local["resolved_lum_delta"]),
            )
            if not pack_local.get("assignments") or int(pack_local["assignment_meta"].get("effective_action_budget", 0)) <= 0:
                raise RuntimeError("no_viable_assignments")
            return pack_local, source_policy_local, structure_profile_local, float(source_policy_local.get("pilot_clean_ber", 0.0))

        # Cache miss: run full proxy tuning to resolve the policy.
        source_policy_local = resolve_source_policy(
            target=target,
            clean_params=clean_params,
            clean_importance=clean_importance,
            clean_sensitivity=clean_sensitivity,
            raw_bits=raw_bits_local,
            args=args,
            initial_active_channels=initial_active_channels,
            log_delta=effective_target_magnitude,
            alpha_delta=alpha_delta,
            theta_delta=theta_delta,
            lum_delta=lum_delta,
            coding_strategy=coding_strategy,
            max_modification_ratio=base_max_modification_ratio,
            proxy_stage_scorer=_make_proxy_stage_scorer(raw_bits_local),
        )
        source_policy_local["payload_signature"] = _payload_signature_local
        # Persist the resolved policy so that matched_random can reuse it.
        if _use_cache and _baseline_type_local == "native":
            _save_source_policy_cache(
                source_policy_local,
                args,
                payload_signature=_payload_signature_local,
            )

        structure_profile_local = str(source_policy_local["resolved_structure_profile"])
        pack_local = _build_v3_pack(
            key=_primary_assignment_key(args),
            raw_bits_local=raw_bits_local,
            structure_profile=structure_profile_local,
            active_channels_local=list(source_policy_local["resolved_active_channels"]),
            selection_temperature=float(source_policy_local["resolved_selection_temperature"]),
            max_modification_ratio=float(source_policy_local["resolved_max_modification_ratio"]),
            keep_fraction=float(source_policy_local["resolved_keep_fraction"]),
            log_delta=float(source_policy_local["resolved_log_delta"]),
            alpha_delta_local=float(source_policy_local["resolved_alpha_delta"]),
            theta_delta_local=float(source_policy_local.get("resolved_theta_delta", theta_delta)),
            lum_delta_local=float(source_policy_local["resolved_lum_delta"]),
        )
        if not pack_local.get("assignments") or int(pack_local["assignment_meta"].get("effective_action_budget", 0)) <= 0:
            raise RuntimeError("no_viable_assignments")
        return pack_local, source_policy_local, structure_profile_local, float(source_policy_local["pilot_clean_ber"])

    source_policy: dict[str, object] = {
        "policy_resolution_mode": "unresolved",
        "resolved_structure_profile": str(base_profile_for_fallback),
        "resolved_region_policy": "legacy",
        "resolved_region_policy_variant": "legacy",
        "resolved_active_channels": list(initial_active_channels),
        "resolved_selection_temperature": float(args.selection_temperature),
        "resolved_max_modification_ratio": float(base_max_modification_ratio),
        "resolved_keep_fraction": 1.0,
        "resolved_log_delta": float(effective_target_magnitude),
        "resolved_alpha_delta": float(alpha_delta),
        "resolved_theta_delta": float(theta_delta),
        "resolved_lum_delta": float(lum_delta),
        "resolved_wavelet_band_loss_weight": _default_wavelet_band_loss_weight(
            structure_profile=str(base_profile_for_fallback),
            profile_policy=profile_policy,
            explicit_weight=getattr(args, "wavelet_band_loss_weight", None),
        ),
        "resolved_stripe_luminance_loss_weight": _default_stripe_luminance_loss_weight(
            structure_profile=str(base_profile_for_fallback),
            profile_policy=profile_policy,
            explicit_weight=getattr(args, "stripe_luminance_loss_weight", None),
        ),
        "resolved_msg_strength_scale": 1.0,
        "resolved_native_margin_scale": 1.0,
        "resolved_wrong_key_scale": 1.0,
        "resolved_attack_start_frac": 0.60,
        "requested_publishable_stage_lock": _normalize_publishable_stage_lock(
            getattr(args, "publishable_stage_lock", "auto")
        ),
        "auto_selected_publishable_stage": None,
        "selected_publishable_stage": None,
        "publishable_stage_lock_applied": False,
        "proxy_tune_steps": 0,
        "proxy_clean_ber": None,
        "proxy_clean_psnr_drop": None,
        "proxy_wrong_key_mean_ber": None,
        "policy_retry_used": False,
        "quality_retry_used": False,
        "publishable_quality_constraint_satisfied": False,
        "periodic_bias_mode": str(getattr(args, "periodic_bias_mode", "legacy")),
        "source_policy_signature": _source_policy_signature(
            {
                "policy_resolution_mode": "unresolved",
                "resolved_structure_profile": str(base_profile_for_fallback),
                "resolved_region_policy": "legacy",
                "resolved_region_policy_variant": "legacy",
                "resolved_active_channels": list(initial_active_channels),
                "resolved_selection_temperature": float(args.selection_temperature),
                "resolved_max_modification_ratio": float(base_max_modification_ratio),
                "resolved_keep_fraction": 1.0,
                "resolved_log_delta": float(effective_target_magnitude),
                "resolved_alpha_delta": float(alpha_delta),
                "resolved_lum_delta": float(lum_delta),
                "resolved_wavelet_band_loss_weight": _default_wavelet_band_loss_weight(
                    structure_profile=str(base_profile_for_fallback),
                    profile_policy=profile_policy,
                    explicit_weight=getattr(args, "wavelet_band_loss_weight", None),
                ),
                "resolved_stripe_luminance_loss_weight": _default_stripe_luminance_loss_weight(
                    structure_profile=str(base_profile_for_fallback),
                    profile_policy=profile_policy,
                    explicit_weight=getattr(args, "stripe_luminance_loss_weight", None),
                ),
                "resolved_msg_strength_scale": 1.0,
                "resolved_native_margin_scale": 1.0,
                "resolved_wrong_key_scale": 1.0,
                "resolved_attack_start_frac": 0.60,
                "requested_publishable_stage_lock": _normalize_publishable_stage_lock(
                    getattr(args, "publishable_stage_lock", "auto")
                ),
                "auto_selected_publishable_stage": None,
                "selected_publishable_stage": None,
                "proxy_tune_steps": 0,
                "proxy_clean_ber": None,
                "proxy_clean_psnr_drop": None,
                "proxy_wrong_key_mean_ber": None,
                "policy_retry_used": False,
                "quality_retry_used": False,
                "publishable_quality_constraint_satisfied": False,
                "profile_policy": profile_policy,
                "periodic_bias_mode": str(getattr(args, "periodic_bias_mode", "legacy")),
            }
        ),
        "structure_profile_stats": {
            **_classify_structure_profile(
                target,
                clean_params,
                requested=str(getattr(args, "structure_profile", "auto")),
            )[1],
            "resolved_structure_profile": str(base_profile_for_fallback),
            "resolved_region_policy": "legacy",
            "resolved_region_policy_variant": "legacy",
            "policy_resolution_mode": "unresolved",
            "policy_retry_used": False,
            "quality_retry_used": False,
            "requested_publishable_stage_lock": _normalize_publishable_stage_lock(
                getattr(args, "publishable_stage_lock", "auto")
            ),
        },
        "pilot_clean_ber": float("nan"),
        "pilot_clean_psnr_drop": float("nan"),
    }

    def _capacity_failure_meta() -> dict[str, object]:
        return {
            **capacity_failure_report(
                payload_bits=effective_payload_bits,
                coded_payload_bits=int(raw_bits.numel()),
            ),
            "protocol": COST_STEGO_V3_1_PROTOCOL,
            "stego_protocol": COST_STEGO_V3_1_PROTOCOL,
            "coding_mode": "cost_code_v3_1",
            "v3_protocol": True,
            "decoder_variant": str(getattr(args, "decoder_variant", "v3_full")),
            "coding_strategy": coding_strategy,
            "active_channels": list(initial_active_channels),
            "channel_weights": {},
            "per_channel_action_counts": {},
            "per_channel_ber": {},
            "auto_payload_bits_used": bool(getattr(args, "auto_payload_bits", False)),
            "requested_payload_bits": requested_payload_bits,
            "effective_payload_bits": effective_payload_bits,
            "payload_fallback_triggered": payload_fallback_triggered,
            "payload_fallback_reason": ",".join(payload_fallback_reasons) if payload_fallback_reasons else None,
            "prototype_capacity_estimate": estimated_capacity,
            "prototype_reliable_carriers": reliable_carriers,
            "capacity_failure_reason": capacity_failure_reason,
            "policy_resolution_mode": str(source_policy.get("policy_resolution_mode", "unresolved")),
            "resolved_structure_profile": str(source_policy.get("resolved_structure_profile", structure_profile)),
            "resolved_region_policy": str(source_policy.get("resolved_region_policy", "legacy")),
            "resolved_region_policy_variant": str(source_policy.get("resolved_region_policy_variant", "legacy")),
            "resolved_active_channels": list(source_policy.get("resolved_active_channels", initial_active_channels)),
            "resolved_selection_temperature": float(
                source_policy.get("resolved_selection_temperature", float(args.selection_temperature))
            ),
            "resolved_max_modification_ratio": float(
                source_policy.get("resolved_max_modification_ratio", base_max_modification_ratio)
            ),
            "resolved_keep_fraction": float(source_policy.get("resolved_keep_fraction", 1.0)),
            "resolved_log_delta": float(source_policy.get("resolved_log_delta", effective_target_magnitude)),
            "resolved_alpha_delta": float(source_policy.get("resolved_alpha_delta", alpha_delta)),
            "resolved_theta_delta": float(source_policy.get("resolved_theta_delta", theta_delta)),
            "resolved_lum_delta": float(source_policy.get("resolved_lum_delta", lum_delta)),
            "resolved_wavelet_band_loss_weight": float(
                source_policy.get(
                    "resolved_wavelet_band_loss_weight",
                    _default_wavelet_band_loss_weight(
                        structure_profile=structure_profile,
                        profile_policy=profile_policy,
                        explicit_weight=getattr(args, "wavelet_band_loss_weight", None),
                    ),
                )
            ),
            "resolved_stripe_luminance_loss_weight": float(
                source_policy.get(
                    "resolved_stripe_luminance_loss_weight",
                    _default_stripe_luminance_loss_weight(
                        structure_profile=structure_profile,
                        profile_policy=profile_policy,
                        explicit_weight=getattr(args, "stripe_luminance_loss_weight", None),
                    ),
                )
            ),
            "resolved_msg_strength_scale": float(source_policy.get("resolved_msg_strength_scale", 1.0)),
            "resolved_native_margin_scale": float(source_policy.get("resolved_native_margin_scale", 1.0)),
            "resolved_wrong_key_scale": float(source_policy.get("resolved_wrong_key_scale", 1.0)),
            "resolved_attack_start_frac": float(source_policy.get("resolved_attack_start_frac", 0.60)),
            "requested_publishable_stage_lock": str(
                source_policy.get("requested_publishable_stage_lock", "auto")
            ),
            "auto_selected_publishable_stage": source_policy.get("auto_selected_publishable_stage"),
            "selected_publishable_stage": source_policy.get("selected_publishable_stage"),
            "publishable_stage_lock_applied": bool(
                source_policy.get("publishable_stage_lock_applied", False)
            ),
            "proxy_tune_steps": int(source_policy.get("proxy_tune_steps", 0) or 0),
            "proxy_clean_ber": source_policy.get("proxy_clean_ber"),
            "proxy_clean_psnr_drop": source_policy.get("proxy_clean_psnr_drop"),
            "proxy_wrong_key_mean_ber": source_policy.get("proxy_wrong_key_mean_ber"),
            "proxy_wrong_key_decode_attempts": source_policy.get("proxy_wrong_key_decode_attempts"),
            "proxy_wrong_key_decode_failures": source_policy.get("proxy_wrong_key_decode_failures"),
            "policy_retry_used": bool(source_policy.get("policy_retry_used", False)),
            "quality_retry_used": bool(source_policy.get("quality_retry_used", False)),
            "publishable_quality_constraint_satisfied": bool(
                source_policy.get("publishable_quality_constraint_satisfied", False)
            ),
            "periodic_bias_mode": str(source_policy.get("periodic_bias_mode", "legacy")),
            "source_policy_signature": source_policy.get("source_policy_signature"),
            "best_checkpoint_step": None,
            "best_checkpoint_clean_ber": None,
            "best_checkpoint_psnr_drop": None,
            "best_checkpoint_wrong_key_mean_ber": None,
            "best_checkpoint_attacker_greedy_ber": None,
            "best_checkpoint_attacker_greedy_chance_gap": None,
            "best_checkpoint_selected": False,
            "attacker_greedy_reference": None,
            "attacker_greedy_public_seed": None,
            "attacker_greedy_assignment_conditioned_on_true_payload": False,
            "attacker_greedy_decode_attempts": 0,
            "attacker_greedy_decode_failures": 0,
            "attacker_greedy_assignment_count": 0,
            "attacker_greedy_error": None,
            "pilot_retry_used": pilot_retry_used,
            "pilot_clean_ber": float(source_policy.get("pilot_clean_ber", float("nan"))),
            "pilot_clean_psnr_drop": float(source_policy.get("pilot_clean_psnr_drop", float("nan"))),
            "structure_profile": structure_profile,
            "structure_profile_stats": source_policy.get("structure_profile_stats"),
            "wavelet_band_loss_weight": float(
                source_policy.get(
                    "resolved_wavelet_band_loss_weight",
                    _default_wavelet_band_loss_weight(
                        structure_profile=structure_profile,
                        profile_policy=profile_policy,
                        explicit_weight=getattr(args, "wavelet_band_loss_weight", None),
                    ),
                )
            ),
            "stripe_luminance_loss_weight": float(
                source_policy.get(
                    "resolved_stripe_luminance_loss_weight",
                    _default_stripe_luminance_loss_weight(
                        structure_profile=structure_profile,
                        profile_policy=profile_policy,
                        explicit_weight=getattr(args, "stripe_luminance_loss_weight", None),
                    ),
                )
            ),
            "assignments": [],
            "wrong_key_assignments": [],
            "baseline_type": args.baseline_type,
            "native_ablation": args.native_ablation,
            "detector_training_enabled": False,
            "detector_training_requested": detector_training_requested,
            "wrong_key_training_enabled": False,
            "wrong_key_training_requested": wrong_key_training_requested,
            "attack_training_enabled": False,
            "attack_training_requested": attack_training_requested,
            "hardening_enabled": False,
            "sinkhorn_enabled": False,
            "adaptive_coding_enabled": coding_strategy == "snr_prune",
            "hardening_sigma_ratio_b": float(getattr(args, "hardening_sigma_ratio", 0.0)),
            "hardening_sigma_ratio_c": float(getattr(args, "hardening_sigma_ratio_c", 0.0)),
            "assignment_audit": _build_assignment_audit(assignments=[], source_policy=source_policy),
        }

    pilot_retry_used = False
    structure_profile = str(base_profile_for_fallback)
    pilot_clean_ber = float("nan")
    raw_bits = _set_effective_payload_bits(effective_payload_bits)
    try:
        pack, source_policy, structure_profile, pilot_clean_ber = _prepare_pack_for_payload(raw_bits)
        pilot_retry_used = bool(source_policy["policy_retry_used"])
    except RuntimeError as exc:
        capacity_failure_reason = f"prepare_pack:{type(exc).__name__}:{exc}"
        if _mark_payload_fallback("prepare_pack_runtime_error"):
            raw_bits = _set_effective_payload_bits(V3_1_FALLBACK_PAYLOAD_BITS)
            try:
                pack, source_policy, structure_profile, pilot_clean_ber = _prepare_pack_for_payload(raw_bits)
                pilot_retry_used = bool(source_policy["policy_retry_used"])
            except RuntimeError as fallback_exc:
                capacity_failure_reason = (
                    f"{capacity_failure_reason};fallback_prepare_pack:"
                    f"{type(fallback_exc).__name__}:{fallback_exc}"
                )
                return deepcopy(clean_renderer), _capacity_failure_meta()
        else:
            return deepcopy(clean_renderer), _capacity_failure_meta()
    if (
        requested_payload_bits >= 16
        and not payload_fallback_triggered
        and int(pack["assignment_meta"]["effective_action_budget"]) < V3_1_RELIABLE_CARRIER_FLOOR
        and _mark_payload_fallback("effective_action_budget_below_floor")
    ):
        raw_bits = _set_effective_payload_bits(V3_1_FALLBACK_PAYLOAD_BITS)
        try:
            pack, source_policy, structure_profile, pilot_clean_ber = _prepare_pack_for_payload(raw_bits)
            pilot_retry_used = bool(source_policy["policy_retry_used"])
        except RuntimeError as fallback_exc:
            capacity_failure_reason = (
                f"effective_budget_fallback_prepare_pack:"
                f"{type(fallback_exc).__name__}:{fallback_exc}"
            )
            return deepcopy(clean_renderer), _capacity_failure_meta()

    if not pack.get("assignments"):
        capacity_failure_reason = "empty_assignment_pack"
        return deepcopy(clean_renderer), _capacity_failure_meta()

    effective_target_magnitude = float(source_policy.get("resolved_log_delta", effective_target_magnitude))
    alpha_delta = float(source_policy.get("resolved_alpha_delta", alpha_delta))
    theta_delta = float(source_policy.get("resolved_theta_delta", theta_delta))
    lum_delta = float(source_policy.get("resolved_lum_delta", lum_delta))
    v3_bundle = pack["v3_bundle"]
    assignments = pack["assignments"]
    assignment_meta = pack["assignment_meta"]
    channel_weights = pack["channel_weights"]
    active_channels = pack["active_channels"]
    block_snr_scores = pack["snr_scores"]
    selection_temperature = float(pack["selection_temperature"])
    max_modification_ratio = float(pack["max_modification_ratio"])
    snr_keep_fraction = float(pack["keep_fraction"])
    wavelet_band_loss_weight = float(
        source_policy.get(
            "resolved_wavelet_band_loss_weight",
            _default_wavelet_band_loss_weight(
                structure_profile=structure_profile,
                profile_policy=profile_policy,
                explicit_weight=getattr(args, "wavelet_band_loss_weight", None),
            ),
        )
    )
    stripe_luminance_loss_weight = float(
        source_policy.get(
            "resolved_stripe_luminance_loss_weight",
            _default_stripe_luminance_loss_weight(
                structure_profile=structure_profile,
                profile_policy=profile_policy,
                explicit_weight=getattr(args, "stripe_luminance_loss_weight", None),
            ),
        )
    )
    msg_strength_scale = float(source_policy.get("resolved_msg_strength_scale", 1.0) or 1.0)
    native_margin_scale = float(source_policy.get("resolved_native_margin_scale", 1.0) or 1.0)
    wrong_key_scale = float(source_policy.get("resolved_wrong_key_scale", 1.0) or 1.0)
    attack_start_frac = float(source_policy.get("resolved_attack_start_frac", 0.60) or 0.60)
    publishable_checkpoint_enabled = (
        str(source_policy.get("resolved_profile_policy", profile_policy)) == "publishable_2d_v1"
    )

    effective_action_budget = int(assignment_meta["effective_action_budget"])
    detector_training_enabled = detector_training_requested and effective_action_budget >= 128
    wrong_key_training_enabled = wrong_key_training_requested and effective_action_budget >= 128
    attack_training_enabled = attack_training_requested and effective_action_budget >= 128

    def _build_for_key(key_suffix: str) -> list[dict]:
        return _build_v3_pack(
            key=key_suffix,
            raw_bits_local=raw_bits,
            structure_profile=structure_profile,
            active_channels_local=list(source_policy["resolved_active_channels"]),
            selection_temperature=float(source_policy["resolved_selection_temperature"]),
            max_modification_ratio=float(source_policy["resolved_max_modification_ratio"]),
            keep_fraction=float(source_policy["resolved_keep_fraction"]),
            log_delta=float(source_policy["resolved_log_delta"]),
            alpha_delta_local=float(source_policy["resolved_alpha_delta"]),
            theta_delta_local=float(source_policy.get("resolved_theta_delta", theta_delta)),
            lum_delta_local=float(source_policy["resolved_lum_delta"]),
        )["assignments"]

    training_wrong_key_assignments: list[list[dict]] = []
    if wrong_key_training_enabled:
        for wi in range(int(args.wrong_key_samples)):
            try:
                training_wrong_key_assignments.append(_build_for_key(f"{args.key}::wrong::{wi}"))
            except RuntimeError:
                continue
    if _supports_keyed_assignment_family(args.baseline_type):
        wrong_key_assignments, wrong_key_sample_failures = _collect_wrong_key_assignment_groups(
            requested_samples=int(args.wrong_key_samples),
            builder=lambda wi: _build_for_key(f"{args.key}::wrong::{wi}"),
        )
    else:
        wrong_key_assignments, wrong_key_sample_failures = [], 0

    attacker_greedy_assignments: list[dict] = []
    attacker_greedy_channel_weights: dict[str, float] = {}
    attacker_greedy_error: str | None = None

    adaptive_coding_enabled = coding_strategy == "snr_prune"
    sigma_ratio_b = float(getattr(args, "hardening_sigma_ratio", 0.30))
    sigma_ratio_c = float(getattr(args, "hardening_sigma_ratio_c", 0.50))
    if effective_action_budget < 32:
        sigma_ratio_c = sigma_ratio_b
    start_iter = int(getattr(args, "hardening_start_iter", 100))
    end_iter = int(getattr(args, "hardening_end_iter", 180))
    hardening_enabled = sigma_ratio_b > 0.0
    mixed_distortion_hardening_enabled = hardening_enabled and bool(
        getattr(args, "mixed_distortion_hardening", False)
    )
    mixed_distortion_hardening_weight = float(getattr(args, "mixed_distortion_hardening_weight", 0.35))
    hardening_quantize_step = float(getattr(args, "hardening_quantize_step", 0.01))
    hardening_small_noise_step = float(getattr(args, "hardening_small_noise_step", 0.01))
    hardening_blur_sigma = float(getattr(args, "hardening_blur_sigma", 1.0))
    hardened_decoder = PerturbationHardenedDecoder(
        sigma_ratio_b=sigma_ratio_b,
        sigma_ratio_c=sigma_ratio_c,
        start_iter=start_iter,
        end_iter=end_iter,
        min_block_prob=0.55,
    )

    ot_weight = float(getattr(args, "ot_covertness_weight", 0.15))
    sinkhorn_blur = float(getattr(args, "sinkhorn_blur", 0.05))
    sinkhorn_enabled = ot_weight > 0.0 and effective_action_budget >= 32
    sinkhorn_term = SinkhornCovertnessTerm(
        blur=sinkhorn_blur,
        weight=ot_weight,
        n_iter=50,
        max_carriers=512,
    )

    renderer = deepcopy(clean_renderer)
    for parameter in renderer.parameters():
        parameter.requires_grad_(False)
    renderer.raw_scales.requires_grad_(True)
    renderer.raw_opacities.requires_grad_(True)
    has_theta = hasattr(renderer, "raw_thetas") and ("theta" in active_channels)
    has_colors = hasattr(renderer, "raw_colors") and ("color_lum" in active_channels)
    optimizer_params = [renderer.raw_scales, renderer.raw_opacities]
    if has_theta:
        renderer.raw_thetas.requires_grad_(True)
        optimizer_params.append(renderer.raw_thetas)
    if has_colors:
        renderer.raw_colors.requires_grad_(True)
        optimizer_params.append(renderer.raw_colors)
    pilot_clean_psnr_drop = float(source_policy.get("pilot_clean_psnr_drop", float("nan")))
    pilot_warm_start_used = False
    if _pilot_candidate_is_publishable(
        pilot_clean_ber=float(pilot_clean_ber),
        pilot_clean_psnr_drop=float(pilot_clean_psnr_drop),
    ):
        pilot_params = _apply_v3_selected_actions_to_params(
            clean_params,
            assignments,
            log_delta=float(effective_target_magnitude),
            alpha_delta=float(alpha_delta),
            theta_delta=float(theta_delta),
            lum_delta=float(lum_delta),
        )
        _load_renderer_from_physical_params(renderer, pilot_params)
        pilot_warm_start_used = True
    optimizer = torch.optim.Adam(optimizer_params, lr=args.tune_lr)

    effective_detector_weight = defaults["detector_weight"] if detector_training_enabled else 0.0
    detector_clean_features = build_detector_feature_vector(
        clean_params,
        importance=clean_importance.to(clean_params["offsets"].device),
    )
    detector = RunLevelStegaDetector(input_dim=int(detector_clean_features.numel())).to(clean_params["offsets"].device)
    detector_optimizer = torch.optim.Adam(detector.parameters(), lr=0.01)

    density_reference = build_density_reference(clean_params)
    clean_moment_ref = build_clean_distribution_reference(clean_params)
    selection_kwargs = {
        "clean_params": clean_params,
        "importance": clean_importance.to(clean_params["offsets"].device),
        "sensitivity": {k: v.to(clean_params["offsets"].device) for k, v in clean_sensitivity.items()},
        "cost_bundle": _build_pack_selection_proxy_bundle(
            pack=pack,
            periodic_lowfreq_carrier_bias_enabled=bool(
                getattr(args, "periodic_lowfreq_carrier_bias", False)
            ),
            periodic_bias_strength=float(getattr(args, "periodic_bias_strength", 0.35)),
        ),
        "selection_temperature": selection_temperature,
    }
    if selection_field_builder is build_soft_selection_field:
        selection_bundle = selection_field_builder(key=args.key, **selection_kwargs)
    else:
        selection_bundle = selection_field_builder(**selection_kwargs)

    selected_by_channel = _selected_indices_by_channel(assignments)
    selected_log_indices = selected_by_channel["log_anisotropy"].to(clean_params["offsets"].device)
    selected_alpha_indices = selected_by_channel["alpha"].to(clean_params["offsets"].device)
    selected_theta_indices = selected_by_channel["theta"].to(clean_params["offsets"].device)
    selected_lum_indices = selected_by_channel["color_lum"].to(clean_params["offsets"].device)
    compensation_indices, compensation_actions, compensation_alpha_only = _build_compensation_plan(
        clean_params["offsets"],
        assignments_local=assignments,
        wet_mask=v3_bundle.wet_mask.to(clean_params["offsets"].device),
        region_policy_context=dict(pack["region_policy_context"]),
        structure_profile=structure_profile,
        periodic_bias_mode=str(pack.get("periodic_bias_mode", "legacy")),
        periodic_extra_compensation_per_active=int(
            getattr(args, "periodic_extra_compensation_per_active", 0)
        ),
    )

    raw_targets = raw_bits.to(clean_params["offsets"].device).to(torch.float32)
    if raw_targets.numel() % 4 != 0:
        raw_targets = torch.cat(
            [
                raw_targets,
                torch.zeros((-raw_targets.numel()) % 4, dtype=raw_targets.dtype, device=raw_targets.device),
            ]
        )
    block_targets = raw_targets.view(-1, 4).to(clean_params["offsets"].device)

    attack_schedule = ["quantize_step_0.01", "small_parameter_noise_0.01", "importance_pruning_10pct"]
    small_noise_gen = torch.Generator(device="cpu")
    small_noise_gen.manual_seed(
        key_to_seed(f"{args.key}::attack_noise::{args.message_seed}::{args.fit_seed}") + 42
    )
    attack_noise_log = (
        torch.empty((selected_log_indices.numel(), 2), dtype=clean_params["scales"].dtype)
        .uniform_(-0.01, 0.01, generator=small_noise_gen)
        if selected_log_indices.numel() > 0 else None
    )
    attack_noise_alpha = (
        torch.empty((selected_alpha_indices.numel(), 1), dtype=clean_params["opacities"].dtype)
        .uniform_(-0.01, 0.01, generator=small_noise_gen)
        if selected_alpha_indices.numel() > 0 else None
    )
    attack_noise_theta = (
        torch.empty((selected_theta_indices.numel(), 1), dtype=clean_params["thetas"].dtype)
        .uniform_(-0.01, 0.01, generator=small_noise_gen)
        if selected_theta_indices.numel() > 0 else None
    )
    attack_noise_lum = (
        torch.empty((selected_lum_indices.numel(), 1), dtype=clean_params["colors"].dtype)
        .uniform_(-0.01, 0.01, generator=small_noise_gen)
        if selected_lum_indices.numel() > 0 else None
    )
    mixed_noise_gen = torch.Generator(device="cpu")
    mixed_noise_gen.manual_seed(
        key_to_seed(f"{args.key}::mixed_hardening_noise::{args.message_seed}::{args.fit_seed}") + 91
    )
    mixed_noise_log = (
        torch.empty((selected_log_indices.numel(), 2), dtype=clean_params["scales"].dtype)
        .uniform_(-hardening_small_noise_step, hardening_small_noise_step, generator=mixed_noise_gen)
        if selected_log_indices.numel() > 0 else None
    )
    mixed_noise_alpha = (
        torch.empty((selected_alpha_indices.numel(), 1), dtype=clean_params["opacities"].dtype)
        .uniform_(-hardening_small_noise_step, hardening_small_noise_step, generator=mixed_noise_gen)
        if selected_alpha_indices.numel() > 0 else None
    )
    mixed_noise_theta = (
        torch.empty((selected_theta_indices.numel(), 1), dtype=clean_params["thetas"].dtype)
        .uniform_(-hardening_small_noise_step, hardening_small_noise_step, generator=mixed_noise_gen)
        if selected_theta_indices.numel() > 0 else None
    )
    mixed_noise_lum = (
        torch.empty((selected_lum_indices.numel(), 1), dtype=clean_params["colors"].dtype)
        .uniform_(-hardening_small_noise_step, hardening_small_noise_step, generator=mixed_noise_gen)
        if selected_lum_indices.numel() > 0 else None
    )
    luma_w = torch.tensor([0.299, 0.587, 0.114], device=clean_params["offsets"].device, dtype=clean_params["offsets"].dtype)
    luma_w = luma_w / max(float((luma_w * luma_w).sum().item()), 1e-6)

    def _carrier_preserving_v3_attack_params(
        params_local: dict[str, torch.Tensor],
        *,
        attack_name: str,
    ) -> dict[str, torch.Tensor]:
        attacked = {key: value for key, value in params_local.items()}
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
            if selected_log_indices.numel() > 0 and attack_noise_log is not None:
                attacked["scales"][selected_log_indices] = torch.clamp(
                    attacked["scales"][selected_log_indices] + attack_noise_log.to(attacked["scales"].device),
                    min=1e-6,
                )
            if selected_alpha_indices.numel() > 0 and attack_noise_alpha is not None:
                attacked["opacities"][selected_alpha_indices] = torch.clamp(
                    attacked["opacities"][selected_alpha_indices] + attack_noise_alpha.to(attacked["opacities"].device),
                    0.0,
                    1.0,
                )
            if selected_theta_indices.numel() > 0 and attack_noise_theta is not None:
                attacked["thetas"][selected_theta_indices] = (
                    attacked["thetas"][selected_theta_indices]
                    + attack_noise_theta.to(attacked["thetas"].device)
                )
            if selected_lum_indices.numel() > 0 and attack_noise_lum is not None:
                lum_step = attack_noise_lum.to(attacked["colors"].device) * luma_w.view(1, 3)
                attacked["colors"][selected_lum_indices] = torch.clamp(
                    attacked["colors"][selected_lum_indices] + lum_step,
                    0.0,
                    1.0,
                )
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

    def _mixed_distortion_hardening_terms(
        params_local: dict[str, torch.Tensor],
        reconstruction_local: torch.Tensor,
        *,
        current_iter: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = params_local["offsets"].new_zeros(1).squeeze()
        if not mixed_distortion_hardening_enabled:
            return zero, zero
        phase_scale = _mixed_distortion_phase_scale(
            current_iter=current_iter,
            start_iter=start_iter,
            end_iter=end_iter,
        )
        if phase_scale <= 0.0:
            return zero, zero

        quantized_params = {
            **params_local,
            "offsets": _straight_through_quantize(params_local["offsets"], step=hardening_quantize_step),
            "scales": _straight_through_quantize(params_local["scales"], step=hardening_quantize_step, min_value=1e-6),
            "thetas": _straight_through_quantize(params_local["thetas"], step=hardening_quantize_step),
            "colors": _straight_through_quantize(params_local["colors"], step=hardening_quantize_step, min_value=0.0, max_value=1.0),
            "opacities": _straight_through_quantize(params_local["opacities"], step=hardening_quantize_step, min_value=0.0, max_value=1.0),
        }
        quantized_probs, _ = _decode_probs(quantized_params, assignments, channel_weights)
        quantized_margin = torch.relu(
            0.35 - (2.0 * block_targets - 1.0) * (2.0 * quantized_probs - 1.0)
        ).mean()

        noisy_params = {key: value.clone() for key, value in params_local.items()}
        if selected_log_indices.numel() > 0 and mixed_noise_log is not None:
            noisy_params["scales"][selected_log_indices] = torch.clamp(
                noisy_params["scales"][selected_log_indices] + mixed_noise_log.to(noisy_params["scales"].device),
                min=1e-6,
            )
        if selected_alpha_indices.numel() > 0 and mixed_noise_alpha is not None:
            noisy_params["opacities"][selected_alpha_indices] = torch.clamp(
                noisy_params["opacities"][selected_alpha_indices] + mixed_noise_alpha.to(noisy_params["opacities"].device),
                0.0,
                1.0,
            )
        if selected_theta_indices.numel() > 0 and mixed_noise_theta is not None:
            noisy_params["thetas"][selected_theta_indices] = (
                noisy_params["thetas"][selected_theta_indices]
                + mixed_noise_theta.to(noisy_params["thetas"].device)
            )
        if selected_lum_indices.numel() > 0 and mixed_noise_lum is not None:
            noisy_lum_step = mixed_noise_lum.to(noisy_params["colors"].device) * luma_w.view(1, 3)
            noisy_params["colors"][selected_lum_indices] = torch.clamp(
                noisy_params["colors"][selected_lum_indices] + noisy_lum_step,
                0.0,
                1.0,
            )
        noisy_probs, _ = _decode_probs(noisy_params, assignments, channel_weights)
        noisy_margin = torch.relu(
            0.35 - (2.0 * block_targets - 1.0) * (2.0 * noisy_probs - 1.0)
        ).mean()

        decode_loss = 0.5 * (
            F.binary_cross_entropy(quantized_probs, block_targets)
            + F.binary_cross_entropy(noisy_probs, block_targets)
        )
        margin_loss = 0.5 * (quantized_margin + noisy_margin)
        mixed_decode_loss = float(phase_scale) * (decode_loss + 0.25 * margin_loss)

        blur_loss = zero
        if hardening_blur_sigma > 0.0:
            blur_loss = float(phase_scale) * F.mse_loss(
                _gaussian_blur_image(reconstruction_local, sigma=hardening_blur_sigma),
                _gaussian_blur_image(clean_render_reference, sigma=hardening_blur_sigma),
            )
        return mixed_decode_loss, blur_loss

    def _true_key_margin_terms(
        block_probs: torch.Tensor,
        action_scores: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_sign = 2.0 * block_targets - 1.0
        bit_alignment = target_sign * (2.0 * block_probs - 1.0)
        true_key_margin = bit_alignment.mean()
        bit_margin_penalty = torch.relu(0.65 - bit_alignment).mean()
        mean_margin_penalty = torch.relu(
            torch.tensor(0.55, dtype=block_probs.dtype, device=block_probs.device) - true_key_margin
        )
        action_losses: list[torch.Tensor] = []
        for scores, assignment in zip(action_scores, assignments):
            if scores.numel() == 0:
                continue
            selected = torch.tensor(assignment["selected_mask"], dtype=torch.float32, device=scores.device) > 0.5
            selected_loss = torch.relu(0.85 - scores[selected]).mean() if selected.any() else scores.new_zeros(1).squeeze()
            unselected_loss = torch.relu(scores[~selected] + 0.05).mean() if (~selected).any() else scores.new_zeros(1).squeeze()
            action_losses.append(selected_loss + 0.50 * unselected_loss)
        action_penalty = torch.stack(action_losses).mean() if action_losses else block_probs.new_zeros(1).squeeze()
        return true_key_margin, bit_margin_penalty, mean_margin_penalty, action_penalty

    def _wrong_key_terms(
        params_local: dict[str, torch.Tensor],
        correct_block_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = params_local["offsets"].new_zeros(1).squeeze()
        if not training_wrong_key_assignments:
            return zero, zero, zero, zero
        correct_alignment = (2.0 * block_targets - 1.0) * (2.0 * correct_block_probs - 1.0)
        correct_conf = torch.abs(correct_block_probs - 0.5).mean() * 2.0
        wrong_uniform_losses: list[torch.Tensor] = []
        wrong_margin_caps: list[torch.Tensor] = []
        wrong_gap_penalties: list[torch.Tensor] = []
        wrong_confs: list[torch.Tensor] = []
        for wrong_group in training_wrong_key_assignments:
            wrong_probs, _ = _decode_probs(params_local, wrong_group, channel_weights)
            wrong_align = torch.abs((2.0 * block_targets - 1.0) * (2.0 * wrong_probs - 1.0))
            wrong_uniform_losses.append(F.binary_cross_entropy(wrong_probs, torch.full_like(wrong_probs, 0.5)))
            wrong_margin_caps.append(torch.relu(wrong_align - 0.08).mean())
            wrong_gap_penalties.append(torch.relu(0.40 - (correct_alignment - wrong_align)).mean())
            wrong_confs.append(torch.abs(wrong_probs - 0.5).mean() * 2.0)
        wrong_key_loss = torch.stack(wrong_uniform_losses).mean()
        wrong_conf = torch.stack(wrong_confs).mean()
        if native_main_method:
            return (
                wrong_key_loss,
                torch.relu(0.65 - correct_alignment).mean(),
                torch.stack(wrong_margin_caps).mean(),
                torch.stack(wrong_gap_penalties).mean(),
            )
        return (
            wrong_key_loss,
            torch.relu(0.35 - correct_conf),
            torch.relu(wrong_conf - 0.10),
            torch.relu(0.20 - (correct_conf - wrong_conf)),
        )

    refit_adversary_steps = int(getattr(args, "refit_adversary_steps", 0))
    refit_adversary_weight = float(getattr(args, "refit_adversary_weight", 0.0))
    checkpoint_quantize_consistency = _periodic_bias_uses_quantize_consistency(
        str(pack.get("periodic_bias_mode", "legacy"))
    )
    checkpoint_schedule_mode = _publishable_checkpoint_schedule_mode(
        structure_profile,
        proxy=False,
    )
    proxy_checkpoint_schedule_mode = str(
        source_policy.get(
            "proxy_checkpoint_schedule_mode",
            _publishable_checkpoint_schedule_mode(structure_profile, proxy=True),
        )
    )
    checkpoint_trace_enabled = bool(
        publishable_checkpoint_enabled and structure_profile == "periodic"
    )
    checkpoint_trace: list[dict[str, object]] = []
    attack_start_offset = int(attack_start_frac * max(1, int(args.tune_steps)))
    best_checkpoint_state: dict[str, torch.Tensor] | None = None
    best_checkpoint_step: int | None = None
    best_checkpoint_clean_ber: float | None = None
    best_checkpoint_psnr_drop: float | None = None
    best_checkpoint_wrong_key_mean_ber: float | None = None
    best_checkpoint_selected = False
    best_fallback_checkpoint_state: dict[str, torch.Tensor] | None = None
    best_fallback_checkpoint_rank: tuple[object, ...] | None = None
    best_fallback_checkpoint_step: int | None = None
    best_fallback_checkpoint_clean_ber: float | None = None
    best_fallback_checkpoint_psnr_drop: float | None = None
    best_fallback_checkpoint_wrong_key_mean_ber: float | None = None
    best_fallback_checkpoint_trace_idx: int | None = None
    best_admissible_checkpoint_state: dict[str, torch.Tensor] | None = None
    best_admissible_checkpoint_rank: tuple[object, ...] | None = None
    best_admissible_checkpoint_step: int | None = None
    best_admissible_checkpoint_clean_ber: float | None = None
    best_admissible_checkpoint_psnr_drop: float | None = None
    best_admissible_checkpoint_wrong_key_mean_ber: float | None = None
    best_admissible_checkpoint_trace_idx: int | None = None
    best_exact_clean_checkpoint_state: dict[str, torch.Tensor] | None = None
    best_exact_clean_checkpoint_rank: tuple[object, ...] | None = None
    best_exact_clean_checkpoint_step: int | None = None
    best_exact_clean_checkpoint_clean_ber: float | None = None
    best_exact_clean_checkpoint_psnr_drop: float | None = None
    best_exact_clean_checkpoint_wrong_key_mean_ber: float | None = None
    best_exact_clean_checkpoint_trace_idx: int | None = None

    def _consider_checkpoint_candidate(
        *,
        step_value: int,
        metrics: dict[str, float | None],
    ) -> None:
        nonlocal best_fallback_checkpoint_state
        nonlocal best_fallback_checkpoint_rank
        nonlocal best_fallback_checkpoint_step
        nonlocal best_fallback_checkpoint_clean_ber
        nonlocal best_fallback_checkpoint_psnr_drop
        nonlocal best_fallback_checkpoint_wrong_key_mean_ber
        nonlocal best_fallback_checkpoint_trace_idx
        nonlocal best_admissible_checkpoint_state
        nonlocal best_admissible_checkpoint_rank
        nonlocal best_admissible_checkpoint_step
        nonlocal best_admissible_checkpoint_clean_ber
        nonlocal best_admissible_checkpoint_psnr_drop
        nonlocal best_admissible_checkpoint_wrong_key_mean_ber
        nonlocal best_admissible_checkpoint_trace_idx
        nonlocal best_exact_clean_checkpoint_state
        nonlocal best_exact_clean_checkpoint_rank
        nonlocal best_exact_clean_checkpoint_step
        nonlocal best_exact_clean_checkpoint_clean_ber
        nonlocal best_exact_clean_checkpoint_psnr_drop
        nonlocal best_exact_clean_checkpoint_wrong_key_mean_ber
        nonlocal best_exact_clean_checkpoint_trace_idx

        checkpoint_rank = _publishable_checkpoint_rank(
            clean_ber=metrics["clean_ber"],
            clean_psnr_drop=metrics["clean_psnr_drop"],
            wrong_key_mean_ber=metrics["wrong_key_mean_ber"],
            quantized_clean_ber=metrics.get("quantized_clean_ber"),
            quantized_clean_psnr_drop=metrics.get("quantized_clean_psnr_drop"),
            quantized_wrong_key_mean_ber=metrics.get("quantized_wrong_key_mean_ber"),
            quantize_consistency=checkpoint_quantize_consistency,
            step=step_value,
        )
        admissible = _publishable_checkpoint_is_admissible(
            clean_ber=metrics["clean_ber"],
            clean_psnr_drop=metrics["clean_psnr_drop"],
            wrong_key_mean_ber=metrics["wrong_key_mean_ber"],
            quantized_clean_ber=metrics.get("quantized_clean_ber"),
            quantized_clean_psnr_drop=metrics.get("quantized_clean_psnr_drop"),
            quantized_wrong_key_mean_ber=metrics.get("quantized_wrong_key_mean_ber"),
            quantize_consistency=checkpoint_quantize_consistency,
        )
        trace_idx: int | None = None
        if checkpoint_trace_enabled:
            checkpoint_trace.append(
                {
                    "step": int(step_value),
                    "clean_ber": metrics["clean_ber"],
                    "clean_psnr_drop": metrics["clean_psnr_drop"],
                    "wrong_key_mean_ber": metrics["wrong_key_mean_ber"],
                    "wrong_key_decode_attempts": metrics.get("wrong_key_decode_attempts"),
                    "wrong_key_decode_failures": metrics.get("wrong_key_decode_failures"),
                    "quantized_clean_ber": metrics.get("quantized_clean_ber"),
                    "quantized_clean_psnr_drop": metrics.get("quantized_clean_psnr_drop"),
                    "quantized_wrong_key_mean_ber": metrics.get("quantized_wrong_key_mean_ber"),
                    "quantized_wrong_key_decode_attempts": metrics.get("quantized_wrong_key_decode_attempts"),
                    "quantized_wrong_key_decode_failures": metrics.get("quantized_wrong_key_decode_failures"),
                    "admissible": bool(admissible),
                    "selected_as_best_admissible": False,
                    "selected_as_best_fallback": False,
                    "selected_as_best_exact_clean": False,
                }
            )
            trace_idx = len(checkpoint_trace) - 1

        fallback_better = (
            best_fallback_checkpoint_rank is None
            or checkpoint_rank < best_fallback_checkpoint_rank
        )
        admissible_better = bool(
            admissible
            and (
                best_admissible_checkpoint_rank is None
                or checkpoint_rank < best_admissible_checkpoint_rank
            )
        )
        exact_clean = bool(
            metrics["clean_ber"] is not None
            and not math.isnan(float(metrics["clean_ber"]))
            and float(metrics["clean_ber"]) <= 1e-9
        )
        exact_clean_rank: tuple[object, ...] | None = None
        exact_clean_better = False
        if exact_clean:
            exact_clean_rank = _publishable_exact_clean_checkpoint_rank(
                clean_psnr_drop=metrics["clean_psnr_drop"],
                wrong_key_mean_ber=metrics["wrong_key_mean_ber"],
                step=step_value,
            )
            exact_clean_better = bool(
                best_exact_clean_checkpoint_rank is None
                or exact_clean_rank < best_exact_clean_checkpoint_rank
            )
        state_copy: dict[str, torch.Tensor] | None = None
        if fallback_better or admissible_better or exact_clean_better:
            state_copy = deepcopy(renderer.state_dict())

        if fallback_better:
            if checkpoint_trace_enabled and best_fallback_checkpoint_trace_idx is not None:
                checkpoint_trace[best_fallback_checkpoint_trace_idx][
                    "selected_as_best_fallback"
                ] = False
            best_fallback_checkpoint_rank = checkpoint_rank
            best_fallback_checkpoint_state = state_copy
            best_fallback_checkpoint_step = int(step_value)
            best_fallback_checkpoint_clean_ber = metrics["clean_ber"]
            best_fallback_checkpoint_psnr_drop = metrics["clean_psnr_drop"]
            best_fallback_checkpoint_wrong_key_mean_ber = metrics["wrong_key_mean_ber"]
            best_fallback_checkpoint_trace_idx = trace_idx
            if checkpoint_trace_enabled and trace_idx is not None:
                checkpoint_trace[trace_idx]["selected_as_best_fallback"] = True

        if admissible_better:
            if checkpoint_trace_enabled and best_admissible_checkpoint_trace_idx is not None:
                checkpoint_trace[best_admissible_checkpoint_trace_idx][
                    "selected_as_best_admissible"
                ] = False
            best_admissible_checkpoint_rank = checkpoint_rank
            best_admissible_checkpoint_state = state_copy
            best_admissible_checkpoint_step = int(step_value)
            best_admissible_checkpoint_clean_ber = metrics["clean_ber"]
            best_admissible_checkpoint_psnr_drop = metrics["clean_psnr_drop"]
            best_admissible_checkpoint_wrong_key_mean_ber = metrics["wrong_key_mean_ber"]
            best_admissible_checkpoint_trace_idx = trace_idx
            if checkpoint_trace_enabled and trace_idx is not None:
                checkpoint_trace[trace_idx]["selected_as_best_admissible"] = True

        if exact_clean_better:
            if checkpoint_trace_enabled and best_exact_clean_checkpoint_trace_idx is not None:
                checkpoint_trace[best_exact_clean_checkpoint_trace_idx][
                    "selected_as_best_exact_clean"
                ] = False
            best_exact_clean_checkpoint_rank = exact_clean_rank
            best_exact_clean_checkpoint_state = state_copy
            best_exact_clean_checkpoint_step = int(step_value)
            best_exact_clean_checkpoint_clean_ber = metrics["clean_ber"]
            best_exact_clean_checkpoint_psnr_drop = metrics["clean_psnr_drop"]
            best_exact_clean_checkpoint_wrong_key_mean_ber = metrics["wrong_key_mean_ber"]
            best_exact_clean_checkpoint_trace_idx = trace_idx
            if checkpoint_trace_enabled and trace_idx is not None:
                checkpoint_trace[trace_idx]["selected_as_best_exact_clean"] = True

    pilot_checkpoint_metrics: dict[str, object] | None = None
    if publishable_checkpoint_enabled and pilot_warm_start_used:
        pilot_checkpoint_metrics = _evaluate_publishable_checkpoint(
            params_local=renderer.decode_parameters(),
            raw_bits_local=raw_bits,
            assignments_local=assignments,
            channel_weights_local=channel_weights,
            log_delta_local=float(effective_target_magnitude),
            alpha_delta_local=float(alpha_delta),
            theta_delta_local=float(theta_delta),
            lum_delta_local=float(lum_delta),
            wrong_key_groups_local=wrong_key_assignments,
            quantize_consistency=checkpoint_quantize_consistency,
            quantize_step=0.01,
        )

    for step in range(int(args.tune_steps)):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = renderer(
            grid[..., 0],
            grid[..., 1],
            clamp_output=False,
            chunk_size=render_chunk_size,
        )
        params = renderer.decode_parameters()
        importance = compute_gaussian_importance(params["scales"], params["opacities"])
        progress = (0.0 if int(args.tune_steps) <= 1 else float(step) / float(int(args.tune_steps) - 1))

        block_probs, action_scores = _decode_probs(params, assignments, channel_weights)
        action_target_loss, action_margin_loss = _action_losses(action_scores, assignments)
        block_margin_loss = torch.relu(0.35 - (2.0 * block_targets - 1.0) * (2.0 * block_probs - 1.0)).mean()
        clean_block_bce = F.binary_cross_entropy(block_probs, block_targets)
        msg_loss = clean_block_bce + 0.25 * action_target_loss + 0.25 * action_margin_loss + 0.25 * block_margin_loss

        tk_push = tk_bit_pen = tk_mean_pen = tk_action_pen = params["offsets"].new_zeros(1).squeeze()
        if native_main_method:
            true_key_margin, tk_bit_pen, tk_mean_pen, tk_action_pen = _true_key_margin_terms(block_probs, action_scores)
            tk_push = torch.clamp(1.0 - true_key_margin, min=0.0)

        hardened_loss_value = params["offsets"].new_zeros(1).squeeze()
        if hardening_enabled:
            hardened_loss_value = hardened_decoder.hardened_loss_v3(
                params=params,
                clean_params=clean_params,
                assignments=assignments,
                channel_weights=_decoder_channel_weights(channel_weights),
                log_delta=float(effective_target_magnitude),
                alpha_delta=alpha_delta,
                theta_delta=theta_delta,
                lum_delta=lum_delta,
                current_iter=step,
            )
        mixed_distortion_loss_value = params["offsets"].new_zeros(1).squeeze()
        mixed_blur_loss_value = params["offsets"].new_zeros(1).squeeze()
        if mixed_distortion_hardening_enabled:
            mixed_distortion_loss_value, mixed_blur_loss_value = _mixed_distortion_hardening_terms(
                params,
                reconstruction,
                current_iter=step,
            )

        moment_loss = matched_moment_loss(params, clean_moment_ref)
        mmd_loss = mmd_distribution_loss(clean_params, params)
        wass_loss = sliced_wasserstein_loss(clean_params, params, projections=32, seed=args.message_seed + step)
        sel_penalty = selection_channel_penalty(
            selection_probs=selection_bundle.selection_probs.to(params["offsets"].device),
            clean_params=clean_params,
            importance=clean_importance.to(params["offsets"].device),
            sensitivity={k: v.to(params["offsets"].device) for k, v in clean_sensitivity.items()},
        )
        det_features = build_detector_feature_vector(params, importance=importance)
        det_loss = detector_embed_loss(detector, det_features)
        if not detector_training_enabled:
            det_loss = det_features.new_zeros(1).squeeze()
        ewc_loss = carrier_ewc_penalty(
            renderer,
            clean_renderer,
            assignments=_v2_assignments_for_compensation(assignments),
            sensitivity={k: v.to(params["offsets"].device) for k, v in clean_sensitivity.items()},
            alpha_weight=float(args.alpha_weight),
        )
        comp_loss = density_reference_loss(params, density_reference)

        sinkhorn_loss_value = params["offsets"].new_zeros(1).squeeze()
        if sinkhorn_enabled:
            try:
                clean_feat, wm_feat = extract_modified_carriers(clean_params, params, v3_bundle)
                if clean_feat.shape[0] >= 4:
                    sinkhorn_loss_value = sinkhorn_term(clean_feat, wm_feat)
            except Exception as exc:
                _warn_once("main_sinkhorn_loss", f"[v3-loss-warn] Sinkhorn loss disabled after failure: {exc}")

        wavelet_loss_value = params["offsets"].new_zeros(1).squeeze()
        if wavelet_band_loss_weight > 0.0:
            wavelet_loss_value = _wavelet_band_consistency_loss(clean_render_reference, reconstruction)
        stripe_luminance_loss_value = params["offsets"].new_zeros(1).squeeze()
        if stripe_luminance_loss_weight > 0.0:
            stripe_luminance_loss_value = _stripe_luminance_smoothness_loss(
                clean_render_reference,
                reconstruction,
                stripe_normal=list(pack["region_policy_context"].get("stripe_normal", [1.0, 0.0])),
            )

        wk_loss = corr_pen = wrong_pen = gap_pen = params["offsets"].new_zeros(1).squeeze()
        if wrong_key_training_enabled:
            wk_loss, corr_pen, wrong_pen, gap_pen = _wrong_key_terms(params, block_probs)

        total_loss = F.mse_loss(reconstruction, target)
        total_loss = total_loss + defaults["msg_strength"] * msg_strength_scale * msg_loss
        total_loss = total_loss + float(args.covertness_weight) * (
            defaults["mmd_weight"] * mmd_loss
            + defaults["wasserstein_weight"] * wass_loss
            + defaults["selection_weight"] * sel_penalty
            + defaults["moment_weight"] * moment_loss
            + effective_detector_weight * det_loss
            + defaults["compensation_weight"] * comp_loss
            + sinkhorn_loss_value
            + wavelet_band_loss_weight * wavelet_loss_value
            + stripe_luminance_loss_weight * stripe_luminance_loss_value
        )
        total_loss = total_loss + defaults["ewc_weight"] * ewc_loss
        if native_main_method:
            total_loss = total_loss + defaults["native_margin_weight"] * native_margin_scale * (
                tk_push + tk_bit_pen + tk_mean_pen + 0.5 * tk_action_pen
            )
        if wrong_key_training_enabled:
            if native_main_method:
                total_loss = total_loss + defaults["wrong_key_weight"] * wrong_key_scale * wk_loss
                total_loss = total_loss + defaults["native_wrong_key_specificity_weight"] * wrong_key_scale * (
                    corr_pen + wrong_pen + gap_pen
                )
            else:
                total_loss = total_loss + defaults["wrong_key_weight"] * wrong_key_scale * (
                    wk_loss + corr_pen + wrong_pen + gap_pen
                )
        if hardening_enabled:
            total_loss = total_loss + 0.5 * defaults["msg_strength"] * msg_strength_scale * hardened_loss_value
        if mixed_distortion_hardening_enabled:
            total_loss = total_loss + mixed_distortion_hardening_weight * defaults["msg_strength"] * msg_strength_scale * mixed_distortion_loss_value
            total_loss = total_loss + 0.25 * mixed_distortion_hardening_weight * float(args.covertness_weight) * mixed_blur_loss_value

        if attack_training_enabled and progress >= attack_start_frac:
            attack_name = attack_schedule[(step - attack_start_offset) % len(attack_schedule)]
            attacked_params = _carrier_preserving_v3_attack_params(params, attack_name=attack_name)
            attacked_block_probs, attacked_action_scores = _decode_probs(attacked_params, assignments, channel_weights)
            attacked_bce = F.binary_cross_entropy(attacked_block_probs, block_targets)
            attacked_margin = torch.relu(
                0.35 - (2.0 * block_targets - 1.0) * (2.0 * attacked_block_probs - 1.0)
            ).mean()
            total_loss = total_loss + 0.5 * defaults["msg_strength"] * msg_strength_scale * (
                attacked_bce + 0.25 * attacked_margin
            )
            if native_main_method:
                atk_margin, atk_bp_pen, atk_mp_pen, atk_ap_pen = _true_key_margin_terms(attacked_block_probs, attacked_action_scores)
                atk_push = torch.clamp(1.0 - atk_margin, min=0.0)
                total_loss = total_loss + 0.5 * defaults["native_margin_weight"] * native_margin_scale * (
                    atk_push + atk_bp_pen + atk_mp_pen + 0.5 * atk_ap_pen
                )
            if wrong_key_training_enabled:
                a_wk, a_cp, a_wp, a_gp = _wrong_key_terms(attacked_params, attacked_block_probs)
                if native_main_method:
                    total_loss = total_loss + 0.5 * defaults["wrong_key_weight"] * wrong_key_scale * a_wk
                    total_loss = total_loss + 0.5 * defaults["native_wrong_key_specificity_weight"] * wrong_key_scale * (
                        a_cp + a_wp + a_gp
                    )
                else:
                    total_loss = total_loss + 0.5 * defaults["wrong_key_weight"] * wrong_key_scale * (
                        a_wk + a_cp + a_wp + a_gp
                    )

        if refit_adversary_steps > 0 and refit_adversary_weight > 0.0 and progress >= 0.70:
            refit_params = _truncated_refit_attack(
                params,
                target=target,
                grid=grid,
                steps=refit_adversary_steps,
            )
            refit_block_probs, refit_action_scores = _decode_probs(refit_params, assignments, channel_weights)
            refit_decode_loss = F.binary_cross_entropy(refit_block_probs, block_targets)
            refit_margin_loss = torch.relu(
                0.35 - (2.0 * block_targets - 1.0) * (2.0 * refit_block_probs - 1.0)
            ).mean()
            total_loss = total_loss + refit_adversary_weight * defaults["msg_strength"] * msg_strength_scale * (
                refit_decode_loss + 0.5 * refit_margin_loss
            )
            if native_main_method:
                refit_margin, refit_bp_pen, refit_mp_pen, refit_ap_pen = _true_key_margin_terms(
                    refit_block_probs,
                    refit_action_scores,
                )
                refit_push = torch.clamp(1.0 - refit_margin, min=0.0)
                total_loss = total_loss + refit_adversary_weight * defaults["native_margin_weight"] * native_margin_scale * (
                    refit_push + refit_bp_pen + refit_mp_pen + 0.5 * refit_ap_pen
                )

        total_loss.backward()
        if selected_log_indices.numel() > 0:
            _masked_rows_step(renderer.raw_scales, selected_log_indices)
        if selected_alpha_indices.numel() > 0:
            _masked_rows_step(renderer.raw_opacities, selected_alpha_indices)
        if has_theta and selected_theta_indices.numel() > 0:
            _masked_rows_step(renderer.raw_thetas, selected_theta_indices)
        if has_colors and selected_lum_indices.numel() > 0:
            _masked_rows_step(renderer.raw_colors, selected_lum_indices)
        grad_params = [parameter for parameter in optimizer_params if parameter.grad is not None]
        if grad_params:
            torch.nn.utils.clip_grad_norm_(grad_params, max_norm=1.0)
        optimizer.step()

        if detector_training_enabled:
            train_detector_steps(
                detector,
                detector_optimizer,
                clean_features=detector_clean_features,
                wm_features=det_features,
                steps=2,
            )

        if compensation_indices.numel() > 0 and (step + 1) % 8 == 0:
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                recon2 = renderer(
                    grid[..., 0],
                    grid[..., 1],
                    clamp_output=False,
                    chunk_size=render_chunk_size,
                )
                params2 = renderer.decode_parameters()
                imp2 = compute_gaussian_importance(params2["scales"], params2["opacities"])
                det_features2 = build_detector_feature_vector(params2, importance=imp2)
                selection_penalty2 = selection_channel_penalty(
                    selection_probs=selection_bundle.selection_probs.to(params2["offsets"].device),
                    clean_params=clean_params,
                    importance=clean_importance.to(params2["offsets"].device),
                    sensitivity={k: v.to(params2["offsets"].device) for k, v in clean_sensitivity.items()},
                )
                comp_obj = F.mse_loss(recon2, target) + float(args.covertness_weight) * (
                    defaults["mmd_weight"] * mmd_distribution_loss(clean_params, params2)
                    + defaults["wasserstein_weight"] * sliced_wasserstein_loss(clean_params, params2, projections=16, seed=step)
                    + defaults["selection_weight"] * selection_penalty2
                    + effective_detector_weight * detector_embed_loss(detector, det_features2)
                    + defaults["compensation_weight"] * density_reference_loss(params2, density_reference)
                )
                comp_obj.backward()
                _masked_rows_step(renderer.raw_opacities, compensation_indices)
                if not compensation_alpha_only:
                    _masked_rows_step(renderer.raw_scales, compensation_indices)
                    grad_targets = [renderer.raw_scales, renderer.raw_opacities]
                else:
                    grad_targets = [renderer.raw_opacities]
                torch.nn.utils.clip_grad_norm_(grad_targets, max_norm=1.0)
                optimizer.step()

        if publishable_checkpoint_enabled and _should_evaluate_publishable_checkpoint(
            step=step + 1,
            total_steps=int(args.tune_steps),
            structure_profile=structure_profile,
            proxy=False,
        ):
            checkpoint_metrics = _evaluate_publishable_checkpoint(
                params_local=renderer.decode_parameters(),
                raw_bits_local=raw_bits,
                assignments_local=assignments,
                channel_weights_local=channel_weights,
                log_delta_local=float(effective_target_magnitude),
                alpha_delta_local=float(alpha_delta),
                theta_delta_local=float(theta_delta),
                lum_delta_local=float(lum_delta),
                wrong_key_groups_local=wrong_key_assignments,
                quantize_consistency=checkpoint_quantize_consistency,
                quantize_step=0.01,
            )
            _consider_checkpoint_candidate(step_value=step + 1, metrics=checkpoint_metrics)

    exact_clean_override = bool(
        best_exact_clean_checkpoint_state is not None
        and best_exact_clean_checkpoint_psnr_drop is not None
        and float(best_exact_clean_checkpoint_psnr_drop) <= 3.0 + 1e-9
        and best_exact_clean_checkpoint_wrong_key_mean_ber is not None
        and _wrong_key_near_chance(best_exact_clean_checkpoint_wrong_key_mean_ber)
        and (
            best_admissible_checkpoint_state is None
            or best_admissible_checkpoint_clean_ber is None
            or float(best_admissible_checkpoint_clean_ber) > 1e-9
        )
    )

    if exact_clean_override:
        best_checkpoint_state = best_exact_clean_checkpoint_state
        best_checkpoint_step = best_exact_clean_checkpoint_step
        best_checkpoint_clean_ber = best_exact_clean_checkpoint_clean_ber
        best_checkpoint_psnr_drop = best_exact_clean_checkpoint_psnr_drop
        best_checkpoint_wrong_key_mean_ber = best_exact_clean_checkpoint_wrong_key_mean_ber
        best_checkpoint_selected = True
        best_checkpoint_selection_mode = "exact_clean_override"
    elif best_admissible_checkpoint_state is not None:
        best_checkpoint_state = best_admissible_checkpoint_state
        best_checkpoint_step = best_admissible_checkpoint_step
        best_checkpoint_clean_ber = best_admissible_checkpoint_clean_ber
        best_checkpoint_psnr_drop = best_admissible_checkpoint_psnr_drop
        best_checkpoint_wrong_key_mean_ber = best_admissible_checkpoint_wrong_key_mean_ber
        best_checkpoint_selected = True
        best_checkpoint_selection_mode = "admissible"
    else:
        best_checkpoint_state = best_fallback_checkpoint_state
        best_checkpoint_step = best_fallback_checkpoint_step
        best_checkpoint_clean_ber = best_fallback_checkpoint_clean_ber
        best_checkpoint_psnr_drop = best_fallback_checkpoint_psnr_drop
        best_checkpoint_wrong_key_mean_ber = best_fallback_checkpoint_wrong_key_mean_ber
        best_checkpoint_selected = bool(best_checkpoint_state is not None)
        best_checkpoint_selection_mode = (
            "fallback" if best_checkpoint_selected else "none"
        )

    if publishable_checkpoint_enabled and best_checkpoint_state is not None:
        renderer.load_state_dict(best_checkpoint_state)

    final_params = renderer.decode_parameters()
    selected_checkpoint_is_post_step = bool(
        best_checkpoint_step is not None and int(best_checkpoint_step) > 0
    )
    pilot_checkpoint_selected = bool(best_checkpoint_step is not None and int(best_checkpoint_step) == 0)
    pilot_only_selected = bool(pilot_checkpoint_selected and not selected_checkpoint_is_post_step)
    best_checkpoint_wrong_key_chance_gap = (
        None
        if best_checkpoint_wrong_key_mean_ber is None
        else float(_wrong_key_chance_gap(best_checkpoint_wrong_key_mean_ber))
    )
    best_checkpoint_attacker_greedy_ber: float | None = None
    best_checkpoint_attacker_greedy_chance_gap: float | None = None
    attacker_greedy_decode_attempts = 0
    attacker_greedy_decode_failures = 0
    # Build the attacker assignment after checkpoint selection so this
    # diagnostic cannot consume randomness used by embedding optimization.
    if str(args.baseline_type) == "native":
        try:
            attacker_greedy_pack = _build_v3_pack_from_policy(
                target=target,
                clean_params=clean_params,
                clean_importance=clean_importance,
                clean_sensitivity=clean_sensitivity,
                raw_bits_local=raw_bits,
                args=args,
                key=GREEDY_ATTACKER_PUBLIC_SEED,
                assignment_builder=build_v3_greedy_nokey_cost_code_assignments,
                active_channels_local=list(source_policy["resolved_active_channels"]),
                structure_profile=structure_profile,
                profile_policy=profile_policy,
                selection_temperature=float(source_policy["resolved_selection_temperature"]),
                max_modification_ratio=float(source_policy["resolved_max_modification_ratio"]),
                keep_fraction=float(source_policy["resolved_keep_fraction"]),
                log_delta=float(source_policy["resolved_log_delta"]),
                alpha_delta=float(source_policy["resolved_alpha_delta"]),
                theta_delta=float(source_policy.get("resolved_theta_delta", theta_delta)),
                lum_delta=float(source_policy["resolved_lum_delta"]),
                coding_strategy=coding_strategy,
            )
            attacker_greedy_assignments = list(attacker_greedy_pack.get("assignments", []))
            attacker_greedy_channel_weights = dict(attacker_greedy_pack.get("channel_weights", {}))
        except Exception as exc:
            attacker_greedy_error = str(exc)
    if attacker_greedy_assignments:
        attacker_greedy_decode_attempts = 1
        try:
            attacker_raw_bits, _, _ = decode_cost_code_v3(
                final_params,
                clean_params=clean_params,
                assignments=attacker_greedy_assignments,
                raw_length=int(raw_bits.numel()),
                channel_weights=attacker_greedy_channel_weights or channel_weights,
                log_delta=float(effective_target_magnitude),
                alpha_delta=float(alpha_delta),
                theta_delta=float(theta_delta),
                lum_delta=float(lum_delta),
            )
            best_checkpoint_attacker_greedy_ber = float(
                (attacker_raw_bits != raw_bits.to(attacker_raw_bits.device)).float().mean().item()
            )
            best_checkpoint_attacker_greedy_chance_gap = float(
                abs(best_checkpoint_attacker_greedy_ber - 0.5)
            )
        except Exception as exc:
            attacker_greedy_decode_failures = 1
            attacker_greedy_error = str(exc)
    final_publishable_quality_constraint_satisfied = (
        _selected_checkpoint_satisfies_publishable_quality(
            publishable_checkpoint_enabled=publishable_checkpoint_enabled,
            best_checkpoint_selected=best_checkpoint_selected,
            clean_ber=best_checkpoint_clean_ber,
            clean_psnr_drop=best_checkpoint_psnr_drop,
            wrong_key_mean_ber=best_checkpoint_wrong_key_mean_ber,
        )
    )
    optimized_checkpoint_valid = bool(
        final_publishable_quality_constraint_satisfied and selected_checkpoint_is_post_step
    )
    qc_meta = _qualified_cover_diagnostics(
        clean_clamped_psnr=float(clean_clamped_psnr),
        gaussian_count=int(clean_params["offsets"].shape[0]),
        effective_action_budget=effective_action_budget,
        payload_bits=effective_payload_bits,
    )

    return renderer, {
        "capacity_failure": False,
        "protocol": COST_STEGO_V3_1_PROTOCOL,
        "stego_protocol": COST_STEGO_V3_1_PROTOCOL,
        "coding_mode": "cost_code_v3_1",
        "v3_protocol": True,
        "decoder_variant": str(getattr(args, "decoder_variant", "v3_full")),
        "coding_strategy": coding_strategy,
        "threat_model": threat_model,
        "profile_policy": profile_policy,
        "region_policy": str(pack["region_policy"]),
        "active_channels": list(active_channels),
        "channel_weights": {key: float(value) for key, value in channel_weights.items()},
        "per_channel_action_counts": v3_per_channel_action_counts(assignments),
        "per_channel_ber": _v3_per_channel_ber_summary(
            clean_params,
            final_params,
            assignments,
            active_channels=active_channels,
            theta_delta=theta_delta,
            lum_delta=lum_delta,
        ),
        "auto_payload_bits_used": bool(getattr(args, "auto_payload_bits", False)),
        "requested_payload_bits": requested_payload_bits,
        "effective_payload_bits": effective_payload_bits,
        "payload_fallback_triggered": payload_fallback_triggered,
        "payload_fallback_reason": ",".join(payload_fallback_reasons) if payload_fallback_reasons else None,
        "prototype_capacity_estimate": estimated_capacity,
        "prototype_reliable_carriers": reliable_carriers,
        "policy_resolution_mode": str(source_policy["policy_resolution_mode"]),
        "resolved_structure_profile": str(source_policy["resolved_structure_profile"]),
        "resolved_region_policy": str(source_policy["resolved_region_policy"]),
        "resolved_region_policy_variant": str(source_policy.get("resolved_region_policy_variant", "legacy")),
        "resolved_active_channels": list(source_policy["resolved_active_channels"]),
        "resolved_selection_temperature": float(source_policy["resolved_selection_temperature"]),
        "resolved_max_modification_ratio": float(source_policy["resolved_max_modification_ratio"]),
        "resolved_keep_fraction": float(source_policy["resolved_keep_fraction"]),
        "resolved_log_delta": float(source_policy["resolved_log_delta"]),
        "resolved_alpha_delta": float(source_policy["resolved_alpha_delta"]),
        "resolved_theta_delta": float(source_policy.get("resolved_theta_delta", theta_delta)),
        "resolved_lum_delta": float(source_policy["resolved_lum_delta"]),
        "resolved_wavelet_band_loss_weight": float(source_policy["resolved_wavelet_band_loss_weight"]),
        "resolved_stripe_luminance_loss_weight": float(
            source_policy["resolved_stripe_luminance_loss_weight"]
        ),
        "resolved_msg_strength_scale": float(source_policy.get("resolved_msg_strength_scale", 1.0)),
        "resolved_native_margin_scale": float(source_policy.get("resolved_native_margin_scale", 1.0)),
        "resolved_wrong_key_scale": float(source_policy.get("resolved_wrong_key_scale", 1.0)),
        "resolved_attack_start_frac": float(source_policy.get("resolved_attack_start_frac", 0.60)),
        "requested_publishable_stage_lock": str(
            source_policy.get("requested_publishable_stage_lock", "auto")
        ),
        "auto_selected_publishable_stage": source_policy.get("auto_selected_publishable_stage"),
        "selected_publishable_stage": source_policy.get("selected_publishable_stage"),
        "publishable_stage_lock_applied": bool(
            source_policy.get("publishable_stage_lock_applied", False)
        ),
        "proxy_tune_steps": int(source_policy.get("proxy_tune_steps", 0) or 0),
        "proxy_clean_ber": source_policy.get("proxy_clean_ber"),
        "proxy_clean_psnr_drop": source_policy.get("proxy_clean_psnr_drop"),
        "proxy_wrong_key_mean_ber": source_policy.get("proxy_wrong_key_mean_ber"),
        "proxy_wrong_key_decode_attempts": source_policy.get("proxy_wrong_key_decode_attempts"),
        "proxy_wrong_key_decode_failures": source_policy.get("proxy_wrong_key_decode_failures"),
        "policy_retry_used": bool(source_policy["policy_retry_used"]),
        "quality_retry_used": bool(source_policy.get("quality_retry_used", False)),
        "publishable_quality_constraint_satisfied": bool(
            optimized_checkpoint_valid
            if publishable_checkpoint_enabled
            else source_policy.get("publishable_quality_constraint_satisfied", False)
        ),
        "best_checkpoint_publishable_quality_constraint_satisfied": bool(
            final_publishable_quality_constraint_satisfied
        ),
        "source_policy_signature": str(source_policy["source_policy_signature"]),
        "best_checkpoint_step": best_checkpoint_step,
        "best_checkpoint_clean_ber": best_checkpoint_clean_ber,
        "best_checkpoint_psnr_drop": best_checkpoint_psnr_drop,
        "best_checkpoint_wrong_key_mean_ber": best_checkpoint_wrong_key_mean_ber,
        "best_checkpoint_wrong_key_chance_gap": best_checkpoint_wrong_key_chance_gap,
        "best_checkpoint_attacker_greedy_ber": best_checkpoint_attacker_greedy_ber,
        "best_checkpoint_attacker_greedy_chance_gap": best_checkpoint_attacker_greedy_chance_gap,
        "best_checkpoint_selected": bool(best_checkpoint_selected),
        "attacker_greedy_reference": "clean_fit_given" if str(args.baseline_type) == "native" else None,
        "attacker_greedy_public_seed": (
            GREEDY_ATTACKER_PUBLIC_SEED if str(args.baseline_type) == "native" else None
        ),
        "attacker_greedy_assignment_conditioned_on_true_payload": bool(
            str(args.baseline_type) == "native"
        ),
        "attacker_greedy_decode_attempts": attacker_greedy_decode_attempts,
        "attacker_greedy_decode_failures": attacker_greedy_decode_failures,
        "attacker_greedy_assignment_count": len(attacker_greedy_assignments),
        "attacker_greedy_error": attacker_greedy_error,
        "pilot_checkpoint_metrics": pilot_checkpoint_metrics,
        "pilot_checkpoint_selected": pilot_checkpoint_selected,
        "pilot_only_selected": pilot_only_selected,
        "post_optimization_checkpoint_selected": selected_checkpoint_is_post_step,
        "selected_checkpoint_is_post_step": selected_checkpoint_is_post_step,
        "optimized_checkpoint_valid": optimized_checkpoint_valid,
        "best_admissible_checkpoint_step": best_admissible_checkpoint_step,
        "best_admissible_checkpoint_clean_ber": best_admissible_checkpoint_clean_ber,
        "best_admissible_checkpoint_wrong_key_mean_ber": best_admissible_checkpoint_wrong_key_mean_ber,
        "best_exact_clean_checkpoint_step": best_exact_clean_checkpoint_step,
        "best_exact_clean_checkpoint_clean_ber": best_exact_clean_checkpoint_clean_ber,
        "best_exact_clean_checkpoint_wrong_key_mean_ber": best_exact_clean_checkpoint_wrong_key_mean_ber,
        "best_fallback_checkpoint_step": best_fallback_checkpoint_step,
        "best_fallback_checkpoint_clean_ber": best_fallback_checkpoint_clean_ber,
        "best_fallback_checkpoint_wrong_key_mean_ber": best_fallback_checkpoint_wrong_key_mean_ber,
        "best_checkpoint_selection_mode": best_checkpoint_selection_mode,
        "checkpoint_schedule_mode": checkpoint_schedule_mode,
        "proxy_checkpoint_schedule_mode": proxy_checkpoint_schedule_mode,
        "checkpoint_trace": checkpoint_trace if checkpoint_trace_enabled else [],
        "pilot_retry_used": pilot_retry_used,
        "pilot_warm_start_used": bool(pilot_warm_start_used),
        "structure_profile": structure_profile,
        "structure_profile_stats": source_policy.get("structure_profile_stats", {}),
        "selected_bin_counts": assignment_meta.get("selected_bin_counts", {}),
        "realized_selected_max_bin_share": assignment_meta.get("realized_selected_max_bin_share"),
        "realized_periodic_core_action_share": assignment_meta.get("realized_periodic_core_action_share"),
        "target_max_bin_share_cap": assignment_meta.get("target_max_bin_share_cap"),
        "coverage_constraint_satisfied": assignment_meta.get("coverage_constraint_satisfied", True),
        "periodic_core_constraint_satisfied": assignment_meta.get("periodic_core_constraint_satisfied", True),
        "coverage_feasible": assignment_meta.get("coverage_feasible", True),
        "coverage_repair_used": assignment_meta.get("coverage_repair_used", False),
        "zone_action_counts": assignment_meta.get("zone_action_counts", {}),
        "coverage_balancing_enabled": assignment_meta.get("coverage_balancing_enabled", False),
        "minimum_block_bin_coverage": assignment_meta.get("minimum_block_bin_coverage", 1),
        "periodic_core_action_cap_ratio": assignment_meta.get("periodic_core_action_cap_ratio", 1.0),
        "gaussian_region_labels": pack["region_policy_context"].get("gaussian_region_labels", []),
        "gaussian_bin_ids": pack["region_policy_context"].get("gaussian_bin_ids", []),
        "gaussian_normal_bin_ids": pack["region_policy_context"].get("gaussian_normal_bin_ids", []),
        "gaussian_tangent_bin_ids": pack["region_policy_context"].get("gaussian_tangent_bin_ids", []),
        "region_allowed_channels": pack["region_policy_context"].get("region_allowed_channels", {}),
        "region_strength_scales": pack["region_policy_context"].get("region_strength_scales", {}),
        "gaussian_local_periodicity_scores": pack["region_policy_context"].get(
            "gaussian_local_periodicity_scores",
            [],
        ),
        "gaussian_repetition_strength": pack["region_policy_context"].get(
            "gaussian_repetition_strength",
            [],
        ),
        "gaussian_edge_norm": pack["region_policy_context"].get("gaussian_edge_norm", []),
        "gaussian_normalized_stripe_normal": pack["region_policy_context"].get(
            "gaussian_normalized_stripe_normal",
            [],
        ),
        "gaussian_normalized_stripe_tangent": pack["region_policy_context"].get(
            "gaussian_normalized_stripe_tangent",
            [],
        ),
        "wavelet_band_loss_weight": float(wavelet_band_loss_weight),
        "stripe_luminance_loss_weight": float(stripe_luminance_loss_weight),
        "alpha_delta": alpha_delta,
        "log_delta": float(effective_target_magnitude),
        "theta_delta": theta_delta,
        "lum_delta": lum_delta,
        "block_snr_scores": [float(value) for value in block_snr_scores],
        "adaptive_block_sizes": [7] * len(assignments),
        "assignments": assignments,
        "wrong_key_assignments": wrong_key_assignments,
        "spread_factor": None,
        "requested_spread_factor": None,
        "wet_ratio": float(v3_bundle.wet_mask.float().mean().item()),
        "requested_wet_ratio": float(v3_bundle.wet_mask.float().mean().item()),
        "relaxed_wet_count": 0,
        "relaxed_wet_indices": [],
        "dry_count_before_relaxation": None,
        "hard_forbid_mask": [bool(value) for value in v3_bundle.wet_mask.detach().cpu().tolist()],
        "modified_gaussian_count": int(sum(int(assignment["selected_count"]) for assignment in assignments)),
        "total_embed_cost": float(
            sum(
                sum(float(cost) * float(mask) for cost, mask in zip(assignment["candidate_costs"], assignment["selected_mask"]))
                for assignment in assignments
            )
        ),
        "compensation_indices": [int(index) for index in compensation_indices.detach().cpu().tolist()],
        "baseline_type": args.baseline_type,
        "native_ablation": args.native_ablation,
        "detector_feature_names": detector_feature_names(),
        "detector_training_enabled": detector_training_enabled,
        "detector_training_requested": detector_training_requested,
        "effective_detector_weight": float(effective_detector_weight),
        "wrong_key_training_enabled": wrong_key_training_enabled,
        "wrong_key_training_requested": wrong_key_training_requested,
        "wrong_key_samples_requested": int(args.wrong_key_samples),
        "attack_training_enabled": attack_training_enabled,
        "attack_training_requested": attack_training_requested,
        "trained_wrong_key_groups": len(training_wrong_key_assignments),
        "wrong_key_sample_failures": wrong_key_sample_failures,
        "hardening_enabled": hardening_enabled,
        "hardening_sigma_ratio_b": sigma_ratio_b,
        "hardening_sigma_ratio_c": sigma_ratio_c,
        "hardening_start_iter": start_iter,
        "hardening_end_iter": end_iter,
        "mixed_distortion_hardening_enabled": mixed_distortion_hardening_enabled,
        "mixed_distortion_hardening_weight": mixed_distortion_hardening_weight,
        "hardening_quantize_step": hardening_quantize_step,
        "hardening_small_noise_step": hardening_small_noise_step,
        "hardening_blur_sigma": hardening_blur_sigma,
        "sinkhorn_enabled": sinkhorn_enabled,
        "ot_covertness_weight": ot_weight,
        "sinkhorn_blur": sinkhorn_blur,
        "adaptive_coding_enabled": adaptive_coding_enabled,
        "pooled_detectability_auc": None,
        "false_alarm_rate": None,
        "same_source_auc": None,
        "cross_source_auc": None,
        "actor_holdout_auc": None,
        "security_model": DEFAULT_SECURITY_MODEL,
        "selection_entropy": float(assignment_meta["selection_entropy"]),
        "effective_action_budget": effective_action_budget,
        "qualified_cover": bool(qc_meta["qualified_cover"]),
        "qualified_cover_reason": str(qc_meta["reason"]),
        "qualified_cover_failed_checks": list(qc_meta["failed_checks"]),
        "qualified_cover_diagnostics": qc_meta,
        "adaptive_budget_enabled": bool(adaptive_controls["enabled"]),
        "adaptive_cover_profile": str(adaptive_controls["profile"]),
        "adaptive_profile_stats": dict(adaptive_controls["profile_stats"]),
        "periodic_lowfreq_carrier_bias_enabled": bool(getattr(args, "periodic_lowfreq_carrier_bias", False)),
        "periodic_bias_mode": str(getattr(args, "periodic_bias_mode", "legacy")),
        "periodic_bias_strength": float(getattr(args, "periodic_bias_strength", 0.35)),
        "periodic_extra_compensation_per_active": int(getattr(args, "periodic_extra_compensation_per_active", 0)),
        "periodic_carrier_scales": pack.get("periodic_carrier_scales", []),
        "assignment_audit": _build_assignment_audit(
            assignments=assignments,
            source_policy=source_policy,
            compensation_actions=compensation_actions,
        ),
        "effective_target_magnitude": float(effective_target_magnitude),
        "effective_max_modification_ratio": float(max_modification_ratio),
        "selection_temperature": float(selection_temperature),
        "snr_keep_fraction": float(snr_keep_fraction),
        "pilot_clean_ber": float(pilot_clean_ber),
        "pilot_clean_psnr_drop": float(source_policy.get("pilot_clean_psnr_drop", float("nan"))),
        "refit_adversary_steps": refit_adversary_steps,
        "refit_adversary_weight": refit_adversary_weight,
    }


# ── Small utility: canonicalise geometry without importing internals ──────────

def _canonicalize_geometry_local(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Wrapper around utils.canonicalize_gaussian_geometry returning a dict."""
    result = canonicalize_gaussian_geometry(params["scales"], params["thetas"])
    if isinstance(result, dict):
        return result
    # If it returns a tuple: (s_major, s_minor, theta_major, log_area, log_anisotropy)
    s_major, s_minor, theta_major, log_area, log_anisotropy = result
    return {
        "s_major": s_major,
        "s_minor": s_minor,
        "theta_major": theta_major,
        "log_area": log_area,
        "log_anisotropy": log_anisotropy,
    }


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    args = parse_args_v3()

    device = get_best_device()
    set_seed(args.message_seed)

    source_run_dir = Path(args.source_run)
    exported = load_exported_run(source_run_dir)
    clean_renderer = build_renderer_from_export(exported, device=device)
    # Detach clean_params: they're reference values, not to be optimised.
    # Without detach(), nn.Parameter leaves have requires_grad=True and
    # total_loss.backward() on step 0 frees the graph; step 1 then crashes.
    clean_params   = {k: v.detach() for k, v in clean_renderer.decode_parameters().items()}

    target = load_target_tensor(exported, device=device)
    grid   = make_grid_from_run(exported, device=device)

    clean_importance = compute_gaussian_importance(clean_params["scales"], clean_params["opacities"])
    # Compute per-Gaussian sensitivity dict (keys: log_anisotropy, alpha, theta_major, …)
    _clean_sensitivity_src = getattr(exported, "sensitivity", None)
    if _clean_sensitivity_src is None:
        _clean_eval = evaluate_renderer(clean_renderer, target, grid, include_sensitivity=True)
        _clean_sensitivity_src = _clean_eval.get("sensitivity")
    if _clean_sensitivity_src is None:
        raise ValueError("Cannot obtain sensitivity — evaluate_renderer returned no sensitivity dict")
    clean_sensitivity = {k: v.cpu() for k, v in _clean_sensitivity_src.items()}
    clean_reconstruction = clean_renderer(
        grid[..., 0],
        grid[..., 1],
        chunk_size=_memory_safe_render_chunk_size(clean_renderer.num_gaussians, grid[..., 0]),
    )
    _psnr_val = compute_psnr(clean_reconstruction.clamp(0.0, 1.0), target)
    clean_clamped_psnr = float(_psnr_val.item() if hasattr(_psnr_val, "item") else _psnr_val)

    # Auto or fixed payload
    if bool(getattr(args, "auto_payload_bits", False)):
        # Build a quick bundle just for capacity estimation
        _tmp_bundle = build_v3_channel_costs(
            clean_params=clean_params,
            importance=clean_importance.to(device),
            sensitivity={k: v.to(device) for k, v in clean_sensitivity.items()},
            log_delta=float(args.target_magnitude),
            alpha_delta=float(min(0.08, max(0.03, 0.16 * float(args.target_magnitude)))),
        )
        effective_payload_bits = estimate_embedding_capacity(_tmp_bundle)
        print(f"[v3] Auto capacity probe → {effective_payload_bits} payload bits")
    else:
        effective_payload_bits = int(args.payload_bits)

    raw_bits = build_bits(
        payload_bits=effective_payload_bits,
        message=args.message,
        seed=args.message_seed,
    )

    wm_renderer, meta = run_cost_stego_v3_embedding(
        clean_renderer=clean_renderer,
        clean_params=clean_params,
        clean_importance=clean_importance,
        clean_sensitivity=clean_sensitivity,
        clean_clamped_psnr=clean_clamped_psnr,
        target=target,
        grid=grid,
        raw_bits=raw_bits,
        args=args,
    )
    effective_payload_bits = int(meta.get("effective_payload_bits", effective_payload_bits))
    raw_bits = _resize_payload_bits(raw_bits, effective_payload_bits)

    # ── Output directory ──────────────────────────────────────────────────────
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_ROOT / (
            f"wm_v3_{source_run_dir.name}"
            f"_{int(effective_payload_bits):03d}b"
            f"_key_{key_to_hash(args.key)[:8]}"
            f"_steps_{int(args.tune_steps)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect watermarked parameters for evaluate_watermark.py compatibility ─
    wm_params = wm_renderer.decode_parameters()
    assignments = meta.get("assignments", [])
    # Gather carrier indices from all block assignments
    carrier_idx_set: set[int] = set()
    for asgn in assignments:
        for action in asgn.get("candidate_actions", []):
            if action.get("selected_mask", action.get("selected", False)):
                carrier_idx_set.add(int(action["gaussian_index"]))
        # fallback: selected_mask list
        for idx, (action, sel) in enumerate(
            zip(asgn.get("candidate_actions", []),
                asgn.get("selected_mask", []))
        ):
            if sel:
                carrier_idx_set.add(int(action["gaussian_index"]))
    # Always include all candidate indices (evaluate_watermark re-decodes anyway)
    for asgn in assignments:
        for action in asgn.get("candidate_actions", []):
            carrier_idx_set.add(int(action["gaussian_index"]))
    carrier_indices_list = sorted(carrier_idx_set)

    message_bits_str = "".join(str(int(b.item())) for b in raw_bits)
    claim_track = str(getattr(args, "claim_track", "full"))
    label = getattr(args, "label", None)
    claim_eligible = _v3_claim_eligible(
        baseline_type=str(meta.get("baseline_type", args.baseline_type)),
        native_ablation=str(meta.get("native_ablation", args.native_ablation)),
        claim_track=claim_track,
        qualified_cover=meta.get("qualified_cover"),
        detector_training_enabled=bool(meta.get("detector_training_enabled", False)),
        wrong_key_training_enabled=bool(meta.get("wrong_key_training_enabled", False)),
        capacity_failure=bool(meta.get("capacity_failure", False)),
    )
    main_claim_eligible = _v3_main_claim_eligible(
        label=label,
        baseline_type=str(meta.get("baseline_type", args.baseline_type)),
        native_ablation=str(meta.get("native_ablation", args.native_ablation)),
        claim_track=claim_track,
        claim_eligible=claim_eligible,
        requested_payload_bits=int(meta.get("requested_payload_bits", int(args.payload_bits))),
        effective_payload_bits=int(meta.get("effective_payload_bits", effective_payload_bits)),
        threat_model=str(meta.get("threat_model", getattr(args, "threat_model", "benign_carrier"))),
        profile_policy=str(meta.get("profile_policy", getattr(args, "profile_policy", "legacy"))),
    )
    baseline_role = _v3_baseline_role(
        label=label,
        baseline_type=str(meta.get("baseline_type", args.baseline_type)),
        native_ablation=str(meta.get("native_ablation", args.native_ablation)),
        claim_track=claim_track,
        claim_eligible=claim_eligible,
        main_claim_eligible=main_claim_eligible,
    )

    # Add evaluate_watermark.py-compatible fields to report
    meta["source_run_dir"]        = str(source_run_dir.resolve())
    meta["mode"]                  = "log_anisotropy"
    meta["protocol"]              = COST_STEGO_V3_1_PROTOCOL
    meta["stego_protocol"]        = COST_STEGO_V3_1_PROTOCOL
    meta["carrier_indices"]       = carrier_indices_list
    meta["block_assignments"]     = assignments   # alias for evaluate_watermark.py
    meta["message_bits"]          = message_bits_str
    meta["coded_message_bits"]    = message_bits_str   # no ECC in v3 cost_stego
    meta["payload_bits"]          = int(effective_payload_bits)
    meta["coded_payload_bits"]    = int(effective_payload_bits)  # same (no ECC)
    meta["requested_payload_bits"] = int(meta.get("requested_payload_bits", int(args.payload_bits)))
    meta["effective_payload_bits"] = int(meta.get("effective_payload_bits", effective_payload_bits))
    meta["payload_fallback_triggered"] = bool(meta.get("payload_fallback_triggered", False))
    meta["payload_fallback_reason"] = meta.get("payload_fallback_reason")
    meta["prototype_capacity_estimate"] = meta.get("prototype_capacity_estimate")
    meta["prototype_reliable_carriers"] = meta.get("prototype_reliable_carriers")
    meta["ecc_mode"]              = "none"
    meta["label"]                 = label
    meta["dataset_name"]          = args.dataset_name
    meta["image_id"]              = args.image_id or source_run_dir.name
    meta["crop_id"]               = args.crop_id
    meta["fit_seed"]              = int(args.fit_seed)
    meta["actor_id"]              = args.actor_id
    meta["threat_model"]          = str(meta.get("threat_model", getattr(args, "threat_model", "benign_carrier")))
    meta["profile_policy"]        = str(meta.get("profile_policy", getattr(args, "profile_policy", "legacy")))
    meta["region_policy"]         = str(meta.get("region_policy", "legacy"))
    meta["policy_resolution_mode"] = str(meta.get("policy_resolution_mode", "baseline_local"))
    meta["resolved_structure_profile"] = str(meta.get("resolved_structure_profile", meta.get("structure_profile", getattr(args, "structure_profile", "auto"))))
    meta["resolved_region_policy"] = str(meta.get("resolved_region_policy", meta.get("region_policy", "legacy")))
    meta["resolved_region_policy_variant"] = str(meta.get("resolved_region_policy_variant", "legacy"))
    meta["resolved_active_channels"] = list(meta.get("resolved_active_channels", meta.get("active_channels", [])))
    meta["resolved_selection_temperature"] = float(meta.get("resolved_selection_temperature", meta.get("selection_temperature", getattr(args, "selection_temperature", 0.5))))
    meta["resolved_max_modification_ratio"] = float(meta.get("resolved_max_modification_ratio", meta.get("effective_max_modification_ratio", getattr(args, "max_modification_ratio", 0.2) or 0.2)))
    meta["resolved_keep_fraction"] = float(meta.get("resolved_keep_fraction", meta.get("snr_keep_fraction", 1.0)))
    meta["resolved_log_delta"] = float(meta.get("resolved_log_delta", meta.get("log_delta", meta.get("effective_target_magnitude", args.target_magnitude))) or 0.0)
    meta["resolved_alpha_delta"] = float(meta.get("resolved_alpha_delta", meta.get("alpha_delta", min(0.08, max(0.03, 0.16 * float(args.target_magnitude))))) or 0.0)
    meta["resolved_theta_delta"] = float(meta.get("resolved_theta_delta", meta.get("theta_delta", 0.05)) or 0.0)
    meta["resolved_lum_delta"] = float(meta.get("resolved_lum_delta", meta.get("lum_delta", 0.008)) or 0.0)
    meta["resolved_wavelet_band_loss_weight"] = float(
        meta.get(
            "resolved_wavelet_band_loss_weight",
            meta.get("wavelet_band_loss_weight", 0.0),
        )
        or 0.0
    )
    meta["resolved_stripe_luminance_loss_weight"] = float(
        meta.get(
            "resolved_stripe_luminance_loss_weight",
            meta.get("stripe_luminance_loss_weight", 0.0),
        )
        or 0.0
    )
    meta["resolved_msg_strength_scale"] = float(meta.get("resolved_msg_strength_scale", 1.0) or 1.0)
    meta["resolved_native_margin_scale"] = float(meta.get("resolved_native_margin_scale", 1.0) or 1.0)
    meta["resolved_wrong_key_scale"] = float(meta.get("resolved_wrong_key_scale", 1.0) or 1.0)
    meta["resolved_attack_start_frac"] = float(meta.get("resolved_attack_start_frac", 0.60) or 0.60)
    meta["requested_publishable_stage_lock"] = str(meta.get("requested_publishable_stage_lock", "auto"))
    meta["auto_selected_publishable_stage"] = meta.get("auto_selected_publishable_stage")
    meta["selected_publishable_stage"] = meta.get("selected_publishable_stage")
    meta["publishable_stage_lock_applied"] = bool(meta.get("publishable_stage_lock_applied", False))
    meta["periodic_min_stage"] = str(
        meta.get("periodic_min_stage", getattr(args, "periodic_min_stage", "auto"))
    )
    meta["periodic_min_stage_applied"] = bool(
        meta.get(
            "periodic_min_stage_applied",
            (
                str(getattr(args, "periodic_min_stage", "auto")) != "auto"
                and str(meta.get("structure_profile", "")) == "periodic"
            ),
        )
    )
    meta["proxy_tune_steps"] = int(meta.get("proxy_tune_steps", 0) or 0)
    meta["proxy_clean_ber"] = meta.get("proxy_clean_ber")
    meta["proxy_clean_psnr_drop"] = meta.get("proxy_clean_psnr_drop")
    meta["proxy_wrong_key_mean_ber"] = meta.get("proxy_wrong_key_mean_ber")
    meta["proxy_wrong_key_decode_attempts"] = meta.get("proxy_wrong_key_decode_attempts")
    meta["proxy_wrong_key_decode_failures"] = meta.get("proxy_wrong_key_decode_failures")
    meta["policy_retry_used"] = bool(meta.get("policy_retry_used", meta.get("pilot_retry_used", False)))
    meta["quality_retry_used"] = bool(meta.get("quality_retry_used", False))
    meta["publishable_quality_constraint_satisfied"] = bool(
        meta.get("publishable_quality_constraint_satisfied", False)
    )
    meta["periodic_bias_mode"] = str(meta.get("periodic_bias_mode", getattr(args, "periodic_bias_mode", "legacy")))
    meta["pilot_clean_psnr_drop"] = meta.get("pilot_clean_psnr_drop")
    meta["best_checkpoint_step"] = meta.get("best_checkpoint_step")
    meta["best_checkpoint_clean_ber"] = meta.get("best_checkpoint_clean_ber")
    meta["best_checkpoint_psnr_drop"] = meta.get("best_checkpoint_psnr_drop")
    meta["best_checkpoint_wrong_key_mean_ber"] = meta.get("best_checkpoint_wrong_key_mean_ber")
    meta["best_checkpoint_wrong_key_chance_gap"] = meta.get(
        "best_checkpoint_wrong_key_chance_gap",
        None
        if meta.get("best_checkpoint_wrong_key_mean_ber") is None
        else float(_wrong_key_chance_gap(meta.get("best_checkpoint_wrong_key_mean_ber"))),
    )
    meta["best_checkpoint_selected"] = bool(meta.get("best_checkpoint_selected", False))
    best_checkpoint_step_value = meta.get("best_checkpoint_step")
    try:
        best_checkpoint_step_int = None if best_checkpoint_step_value is None else int(best_checkpoint_step_value)
    except (TypeError, ValueError):
        best_checkpoint_step_int = None
    meta["pilot_checkpoint_metrics"] = meta.get("pilot_checkpoint_metrics")
    meta["pilot_checkpoint_selected"] = bool(
        meta.get("pilot_checkpoint_selected", best_checkpoint_step_int == 0)
    )
    meta["pilot_only_selected"] = bool(
        meta.get(
            "pilot_only_selected",
            meta["pilot_checkpoint_selected"] and not (best_checkpoint_step_int is not None and best_checkpoint_step_int > 0),
        )
    )
    meta["post_optimization_checkpoint_selected"] = bool(
        meta.get(
            "post_optimization_checkpoint_selected",
            best_checkpoint_step_int is not None and best_checkpoint_step_int > 0,
        )
    )
    meta["selected_checkpoint_is_post_step"] = bool(
        meta.get("selected_checkpoint_is_post_step", meta["post_optimization_checkpoint_selected"])
    )
    meta["optimized_checkpoint_valid"] = bool(
        meta.get(
            "optimized_checkpoint_valid",
            bool(meta.get("publishable_quality_constraint_satisfied", False))
            and meta["selected_checkpoint_is_post_step"],
        )
    )
    meta["best_checkpoint_publishable_quality_constraint_satisfied"] = bool(
        meta.get(
            "best_checkpoint_publishable_quality_constraint_satisfied",
            meta.get("publishable_quality_constraint_satisfied", False),
        )
    )
    meta["best_admissible_checkpoint_step"] = meta.get("best_admissible_checkpoint_step")
    meta["best_admissible_checkpoint_clean_ber"] = meta.get("best_admissible_checkpoint_clean_ber")
    meta["best_admissible_checkpoint_wrong_key_mean_ber"] = meta.get("best_admissible_checkpoint_wrong_key_mean_ber")
    meta["best_exact_clean_checkpoint_step"] = meta.get("best_exact_clean_checkpoint_step")
    meta["best_exact_clean_checkpoint_clean_ber"] = meta.get("best_exact_clean_checkpoint_clean_ber")
    meta["best_exact_clean_checkpoint_wrong_key_mean_ber"] = meta.get("best_exact_clean_checkpoint_wrong_key_mean_ber")
    meta["best_fallback_checkpoint_step"] = meta.get("best_fallback_checkpoint_step")
    meta["best_fallback_checkpoint_clean_ber"] = meta.get("best_fallback_checkpoint_clean_ber")
    meta["best_fallback_checkpoint_wrong_key_mean_ber"] = meta.get("best_fallback_checkpoint_wrong_key_mean_ber")
    meta["best_checkpoint_selection_mode"] = str(meta.get("best_checkpoint_selection_mode", "none"))
    meta["checkpoint_schedule_mode"] = str(
        meta.get(
            "checkpoint_schedule_mode",
            _publishable_checkpoint_schedule_mode(
                str(meta.get("resolved_structure_profile", meta.get("structure_profile", "generic"))),
                proxy=False,
            ),
        )
    )
    meta["proxy_checkpoint_schedule_mode"] = str(
        meta.get(
            "proxy_checkpoint_schedule_mode",
            _publishable_checkpoint_schedule_mode(
                str(meta.get("resolved_structure_profile", meta.get("structure_profile", "generic"))),
                proxy=True,
            ),
        )
    )
    meta["checkpoint_trace"] = meta.get("checkpoint_trace", [])
    if meta.get("source_policy_signature") is None:
        meta["source_policy_signature"] = _source_policy_signature(
            {
                "policy_resolution_mode": meta["policy_resolution_mode"],
                "resolved_structure_profile": meta["resolved_structure_profile"],
                "resolved_region_policy": meta["resolved_region_policy"],
                "resolved_region_policy_variant": meta["resolved_region_policy_variant"],
                "resolved_active_channels": meta["resolved_active_channels"],
                "resolved_selection_temperature": meta["resolved_selection_temperature"],
                "resolved_max_modification_ratio": meta["resolved_max_modification_ratio"],
                "resolved_keep_fraction": meta["resolved_keep_fraction"],
                "resolved_log_delta": meta["resolved_log_delta"],
                "resolved_alpha_delta": meta["resolved_alpha_delta"],
                "resolved_lum_delta": meta["resolved_lum_delta"],
                "resolved_wavelet_band_loss_weight": meta["resolved_wavelet_band_loss_weight"],
                "resolved_stripe_luminance_loss_weight": meta["resolved_stripe_luminance_loss_weight"],
                "resolved_msg_strength_scale": meta["resolved_msg_strength_scale"],
                "resolved_native_margin_scale": meta["resolved_native_margin_scale"],
                "resolved_wrong_key_scale": meta["resolved_wrong_key_scale"],
                "resolved_attack_start_frac": meta["resolved_attack_start_frac"],
                "requested_publishable_stage_lock": meta["requested_publishable_stage_lock"],
                "auto_selected_publishable_stage": meta["auto_selected_publishable_stage"],
                "selected_publishable_stage": meta["selected_publishable_stage"],
                "proxy_tune_steps": meta["proxy_tune_steps"],
                "proxy_clean_ber": meta["proxy_clean_ber"],
                "proxy_clean_psnr_drop": meta["proxy_clean_psnr_drop"],
                "proxy_wrong_key_mean_ber": meta["proxy_wrong_key_mean_ber"],
                "proxy_wrong_key_decode_attempts": meta["proxy_wrong_key_decode_attempts"],
                "proxy_wrong_key_decode_failures": meta["proxy_wrong_key_decode_failures"],
                "policy_retry_used": meta["policy_retry_used"],
                "quality_retry_used": meta["quality_retry_used"],
                "pilot_clean_psnr_drop": 0.0 if meta["pilot_clean_psnr_drop"] is None else float(meta["pilot_clean_psnr_drop"]),
                "publishable_quality_constraint_satisfied": meta["publishable_quality_constraint_satisfied"],
                "profile_policy": meta["profile_policy"],
                "periodic_bias_mode": meta["periodic_bias_mode"],
            }
        )
    meta["selected_bin_counts"]   = meta.get("selected_bin_counts", {})
    meta["realized_selected_max_bin_share"] = meta.get("realized_selected_max_bin_share")
    meta["realized_periodic_core_action_share"] = meta.get("realized_periodic_core_action_share")
    meta["target_max_bin_share_cap"] = meta.get("target_max_bin_share_cap")
    meta["coverage_constraint_satisfied"] = bool(meta.get("coverage_constraint_satisfied", True))
    meta["periodic_core_constraint_satisfied"] = bool(meta.get("periodic_core_constraint_satisfied", True))
    meta["coverage_feasible"] = bool(meta.get("coverage_feasible", True))
    meta["coverage_repair_used"] = bool(meta.get("coverage_repair_used", False))
    meta["zone_action_counts"]    = meta.get("zone_action_counts", {})
    meta["wavelet_band_loss_weight"] = float(meta.get("wavelet_band_loss_weight", 0.0) or 0.0)
    meta["stripe_luminance_loss_weight"] = float(meta.get("stripe_luminance_loss_weight", 0.0) or 0.0)
    meta["gaussian_region_labels"] = meta.get("gaussian_region_labels", [])
    meta["gaussian_bin_ids"]      = meta.get("gaussian_bin_ids", [])
    meta["gaussian_normal_bin_ids"] = meta.get("gaussian_normal_bin_ids", [])
    meta["gaussian_tangent_bin_ids"] = meta.get("gaussian_tangent_bin_ids", [])
    meta["region_allowed_channels"] = meta.get("region_allowed_channels", {})
    meta["region_strength_scales"] = meta.get("region_strength_scales", {})
    meta["gaussian_local_periodicity_scores"] = meta.get("gaussian_local_periodicity_scores", [])
    meta["gaussian_repetition_strength"] = meta.get("gaussian_repetition_strength", [])
    meta["gaussian_edge_norm"] = meta.get("gaussian_edge_norm", [])
    meta["gaussian_normalized_stripe_normal"] = meta.get("gaussian_normalized_stripe_normal", [])
    meta["gaussian_normalized_stripe_tangent"] = meta.get("gaussian_normalized_stripe_tangent", [])
    meta["assignment_audit"] = meta.get(
        "assignment_audit",
        _build_assignment_audit(
            assignments=assignments,
            source_policy={
                "requested_publishable_stage_lock": meta["requested_publishable_stage_lock"],
                "auto_selected_publishable_stage": meta["auto_selected_publishable_stage"],
                "selected_publishable_stage": meta["selected_publishable_stage"],
                "publishable_stage_lock_applied": meta["publishable_stage_lock_applied"],
            },
            compensation_actions=[],
        ),
    )
    meta["carrier_pool"]          = "v3_cost_code"
    meta["key_hash"]              = key_to_hash(args.key)
    meta["embed_key"]             = args.key
    meta["tune_steps"]            = int(args.tune_steps)
    meta["tune_lr"]               = float(args.tune_lr)
    meta["claim_track"]           = claim_track
    meta["claim_eligible"]        = claim_eligible
    meta["main_claim_eligible"]   = main_claim_eligible
    meta["baseline_role"]         = baseline_role
    meta["v3_protocol"]           = True   # flag to distinguish from pure v2

    # Save updated report
    save_json(meta, data_dir / "watermark_report.json")
    (data_dir / "message_bits.txt").write_text(message_bits_str + "\n", encoding="utf-8")

    # Save watermarked_params.pt (required by evaluate_watermark.py)
    # Export in Gaussian chunks so high-resolution runs do not materialize the
    # full N x H x W renderer workspace after optimization.
    with torch.no_grad():
        wm_reconstruction = (
            wm_renderer(
                grid[..., 0],
                grid[..., 1],
                chunk_size=_memory_safe_render_chunk_size(wm_renderer.num_gaussians, grid[..., 0]),
            )
            .clamp(0.0, 1.0)
            .cpu()
        )
        clean_reconstruction_cpu = (
            clean_renderer(
                grid[..., 0],
                grid[..., 1],
                chunk_size=_memory_safe_render_chunk_size(clean_renderer.num_gaussians, grid[..., 0]),
            )
            .clamp(0.0, 1.0)
            .cpu()
        )
    torch.save(
        {
            "schema_version": 5,
            "artifact_type": "watermarked_export",
            "source_run_dir": str(source_run_dir.resolve()),
            "config": exported.config,
            "v3_protocol": True,
            "parameters": {
                "offsets":    wm_params["offsets"].detach().cpu(),
                "scales":     wm_params["scales"].detach().cpu(),
                "thetas":     wm_params["thetas"].detach().cpu(),
                "colors":     wm_params["colors"].detach().cpu(),
                "opacities":  wm_params["opacities"].detach().cpu(),
            },
            "watermark_images": {
                "clean_image":     clean_reconstruction_cpu,
                "reference_image": wm_reconstruction,
            },
        },
        data_dir / "watermarked_params.pt",
    )

    # Also save clean-vs-watermarked images for visual inspection
    save_raw_tensor_image(clean_reconstruction_cpu, output_dir / "clean_reconstruction.png")
    save_raw_tensor_image(wm_reconstruction, output_dir / "watermarked_reconstruction.png")
    save_raw_tensor_image(target.detach().cpu(), output_dir / "target_image.png")

    # Quick inline BER check (no attack)
    with torch.no_grad():
        clean_params_cpu = {k: v.cpu() for k, v in clean_params.items()}
        wm_params_cpu    = {k: v.cpu() for k, v in wm_params.items()}
        try:
            decoded_raw_bits, _, _ = decode_cost_code_v3(
                wm_params_cpu,
                clean_params=clean_params_cpu,
                assignments=assignments,
                raw_length=int(effective_payload_bits),
                channel_weights=dict(meta.get("channel_weights", {})),
                log_delta=float(meta.get("log_delta", meta.get("effective_target_magnitude", args.target_magnitude))),
                alpha_delta=float(meta.get("alpha_delta", min(0.08, max(0.03, 0.16 * float(args.target_magnitude))))),
                theta_delta=float(meta.get("theta_delta", 0.05)),
                lum_delta=float(meta.get("lum_delta", 0.008)),
                beta=4.0,
            )
            raw_bits_tensor = torch.tensor(
                [int(b) for b in message_bits_str], dtype=torch.int64
            )
            inline_ber = float((decoded_raw_bits != raw_bits_tensor).float().mean().item())
        except Exception as exc:
            inline_ber = float("nan")
            print(f"[v3] Warning: inline BER check failed: {exc}")

    print(f"[v3] Watermark output: {output_dir}")
    print(f"[v3] Capacity failure: {meta.get('capacity_failure', False)}")
    if meta.get("capacity_failure_reason"):
        print(f"[v3] Capacity failure reason: {meta['capacity_failure_reason']}")
    print(f"[v3] Inline clean BER: {inline_ber:.4f}")
    print(f"[v3] Hardening: {meta.get('hardening_enabled', False)}, "
          f"Sinkhorn: {meta.get('sinkhorn_enabled', False)}")


if __name__ == "__main__":
    main()
