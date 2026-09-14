from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from run_fit import compute_saturation_statistics
from utils import compute_gaussian_importance
from watermark_utils import hamming74_decode, hamming74_encode, key_to_seed


# ---------------------------------------------------------------------------
# Literature positioning
# ---------------------------------------------------------------------------
# This module implements white-box parameter-domain steganography for 2D
# Gaussian Splatting.  Relevant prior work for comparison and positioning:
#
#   HiDDeN [Zhu et al., ECCV 2018]   — learned pixel-domain encoder/decoder;
#       our render-refit transfer baseline approximates the threat model.
#       https://arxiv.org/abs/1807.09937
#
#   StegaStamp [Tancik et al., CVPR 2020] — spatially-uniform pixel stego
#       resistant to print-scan distortions; our pixel baseline is the
#       closest analogue in the 2D Gaussian setting.
#       https://arxiv.org/abs/1904.05343
#
#   WatermarkGS [Li et al., 2024] — concurrent 3DGS watermarking in the
#       rendered RGB domain; does NOT embed in raw Gaussian parameters.
#       Our native parameter-domain approach is architecturally distinct.
#       [Cite when available; check CVPR/SIGGRAPH 2024 proceedings]
#
#   HUGO [Pevný et al., IH 2010] — cost-based steganography in images using
#       Heuristic-Objective Unified Embedding; conceptually analogous to our
#       sensitivity-weighted carrier selection.
#       https://doi.org/10.1007/978-3-642-16435-4_2
#
#   STC [Filler, Judas & Fridrich, IEEE TIFS 2011] — Syndrome Trellis Coding
#       for near-optimal cost-based embedding; our parity-block code is a
#       lightweight non-trellis analogue targeting the 2D Gaussian parameter
#       space (see build_cost_code_assignments).
#       https://doi.org/10.1109/TIFS.2011.2134094
#
#   Spread-spectrum watermarking [Cox et al., IEEE TITS 1997] — multi-carrier
#       redundancy for robustness; multi_cover_round_trip implements a
#       round-robin shard analogue for the multi-cover setting.
#       https://doi.org/10.1109/83.650120
# ---------------------------------------------------------------------------

DEFAULT_SECURITY_MODEL = "white_box_parameter_receiver"
DEFAULT_STEGO_PROTOCOL = "cost_stego_v1"
COST_STEGO_V1_PROTOCOL = DEFAULT_STEGO_PROTOCOL
COST_STEGO_V2_PROTOCOL = "cost_stego_v2"
# Prototype-only steganography ideas derived from the 6-parameter analysis.
THETA_DISTRIBUTION_PROTOCOL = "theta_distribution_v1"
MU_JITTER_PROTOCOL = "mu_jitter_v1"
SCALE_TIER_PROTOCOL = "scale_tier_v1"
# S1-A: Joint-parity multi-channel steganography protocol (Idea 4 from stego_ideas_proposal.py).
# Encodes each bit jointly across three Gaussian parameter channels (log_anisotropy, alpha, theta)
# via a GF(2) parity constraint, reducing per-channel modification depth by ~3×.
JOINT_PARITY_PROTOCOL = "joint_parity_v1"


@dataclass
class ChannelCostBundle:
    wet_mask: Tensor
    log_rho_plus: Tensor
    log_rho_minus: Tensor
    alpha_rho_plus: Tensor
    alpha_rho_minus: Tensor
    log_delta: float
    alpha_delta: float
    density_reference: dict[str, Tensor]


@dataclass
class CostCodeV2Bundle:
    hard_forbid_mask: Tensor
    selection_logits: Tensor
    selection_probs: Tensor
    selection_entropy: float
    effective_action_budget: int
    log_delta: float
    alpha_delta: float


class RunLevelStegaDetector(nn.Module):
    """A small differentiable run-level detector used as an adversarial covertness probe.

    Design analogous to the discriminator in HiDDeN [Zhu et al., ECCV 2018] which
    trains an end-to-end encoder/decoder with a pixel-level adversarial loss.  Here
    the detector operates on the Gaussian *parameter* feature vector rather than the
    rendered image, serving as an online covertness regulariser during embedding.
    Ref: https://arxiv.org/abs/1807.09937
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.LeakyReLU(0.2),
            nn.Linear(48, 24),
            nn.LeakyReLU(0.2),
            nn.Linear(24, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


def _safe_corr(x: Tensor, y: Tensor) -> Tensor:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt(torch.sum(x_centered * x_centered) * torch.sum(y_centered * y_centered)).clamp_min(1e-12)
    return torch.sum(x_centered * y_centered) / denom


def _binary_auc(scores: Tensor, labels: Tensor) -> float:
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.5
    pairwise = (pos_scores[:, None] > neg_scores[None, :]).float()
    ties = (pos_scores[:, None] == neg_scores[None, :]).float()
    return float((pairwise.mean() + 0.5 * ties.mean()).item())


def _robust_scale(values: Tensor) -> Tensor:
    median = values.median()
    mad = (values - median).abs().median()
    return (1.4826 * mad).clamp_min(1e-6)


def robust_z_scores(values: Tensor) -> Tensor:
    median = values.median()
    scale = _robust_scale(values)
    return (values - median) / scale


def _canonicalize_geometry(scales: Tensor, thetas: Tensor) -> dict[str, Tensor]:
    sx = scales[:, 0]
    sy = scales[:, 1]
    theta = thetas[:, 0]
    keep_order = sx >= sy
    s_major = torch.where(keep_order, sx, sy)
    s_minor = torch.where(keep_order, sy, sx)
    theta_major = torch.where(keep_order, theta, theta + (math.pi / 2.0))
    theta_major = torch.remainder(theta_major, math.pi)
    return {
        "s_major": s_major,
        "s_minor": s_minor,
        "theta_major": theta_major.unsqueeze(-1),
        "log_area": torch.log(s_major * s_minor),
        "log_anisotropy": torch.log(s_major / s_minor),
        "signed_logratio": torch.log(sx / sy),
        "axis_swapped": (~keep_order).to(scales.dtype),
    }


def cover_feature_matrix(params: dict[str, Tensor]) -> Tensor:
    canonical = _canonicalize_geometry(params["scales"], params["thetas"])
    alpha = params["opacities"][:, 0]
    return torch.stack(
        [
            params["offsets"][:, 0],
            params["offsets"][:, 1],
            canonical["log_area"],
            canonical["log_anisotropy"],
            canonical["theta_major"][:, 0],
            alpha,
        ],
        dim=1,
    )


def build_density_reference(params: dict[str, Tensor]) -> dict[str, Tensor]:
    features = cover_feature_matrix(params)
    canonical = _canonicalize_geometry(params["scales"], params["thetas"])
    theta_major = canonical["theta_major"][:, 0]
    alpha = params["opacities"][:, 0]
    centered = features - features.mean(dim=0, keepdim=True)
    covariance = centered.t() @ centered / max(1, features.shape[0])
    phase = 2.0 * theta_major
    return {
        "mean": features.mean(dim=0).detach(),
        "std": features.std(dim=0, unbiased=False).clamp_min(1e-6).detach(),
        "covariance": covariance.detach(),
        "theta_resultant_length": torch.sqrt(torch.cos(phase).mean() ** 2 + torch.sin(phase).mean() ** 2).detach(),
        "theta_mean_phase_cos": torch.cos(phase).mean().detach(),
        "theta_mean_phase_sin": torch.sin(phase).mean().detach(),
        "alpha_log_area_corr": _safe_corr(alpha, canonical["log_area"]).detach(),
    }


def density_reference_loss(params: dict[str, Tensor], reference: dict[str, Tensor]) -> Tensor:
    features = cover_feature_matrix(params)
    canonical = _canonicalize_geometry(params["scales"], params["thetas"])
    theta_major = canonical["theta_major"][:, 0]
    alpha = params["opacities"][:, 0]
    centered = features - features.mean(dim=0, keepdim=True)
    covariance = centered.t() @ centered / max(1, features.shape[0])
    phase = 2.0 * theta_major
    theta_resultant = torch.sqrt(torch.cos(phase).mean() ** 2 + torch.sin(phase).mean() ** 2)
    losses = [
        (features.mean(dim=0) - reference["mean"]).pow(2).mean(),
        (features.std(dim=0, unbiased=False) - reference["std"]).pow(2).mean(),
        (covariance - reference["covariance"]).pow(2).mean(),
        (theta_resultant - reference["theta_resultant_length"]).pow(2),
        (_safe_corr(alpha, canonical["log_area"]) - reference["alpha_log_area_corr"]).pow(2),
    ]
    return torch.stack(losses).mean()


def covariance_and_circular_shift(clean_params: dict[str, Tensor], compare_params: dict[str, Tensor]) -> dict[str, float]:
    clean_features = cover_feature_matrix(clean_params).detach().cpu()
    compare_features = cover_feature_matrix(compare_params).detach().cpu()
    clean_centered = clean_features - clean_features.mean(dim=0, keepdim=True)
    compare_centered = compare_features - compare_features.mean(dim=0, keepdim=True)
    clean_cov = clean_centered.t() @ clean_centered / max(1, clean_features.shape[0])
    compare_cov = compare_centered.t() @ compare_centered / max(1, compare_features.shape[0])

    clean_theta = _canonicalize_geometry(clean_params["scales"], clean_params["thetas"])["theta_major"][:, 0].detach().cpu()
    compare_theta = _canonicalize_geometry(compare_params["scales"], compare_params["thetas"])["theta_major"][:, 0].detach().cpu()
    clean_phase = 2.0 * clean_theta
    compare_phase = 2.0 * compare_theta
    clean_resultant = torch.sqrt(torch.cos(clean_phase).mean() ** 2 + torch.sin(clean_phase).mean() ** 2)
    compare_resultant = torch.sqrt(torch.cos(compare_phase).mean() ** 2 + torch.sin(compare_phase).mean() ** 2)
    clean_mean = 0.5 * torch.atan2(torch.sin(clean_phase).mean(), torch.cos(clean_phase).mean())
    compare_mean = 0.5 * torch.atan2(torch.sin(compare_phase).mean(), torch.cos(compare_phase).mean())
    circular_shift = 0.5 * torch.abs(torch.atan2(torch.sin(2.0 * (compare_mean - clean_mean)), torch.cos(2.0 * (compare_mean - clean_mean))))
    return {
        "covariance_shift": float((clean_cov - compare_cov).pow(2).mean().sqrt().item()),
        "circular_shift": float(circular_shift.item()),
        "theta_resultant_shift": float(abs(float(clean_resultant.item()) - float(compare_resultant.item()))),
    }


def mmd_distribution_loss(clean_params: dict[str, Tensor], compare_params: dict[str, Tensor]) -> Tensor:
    # Maximum Mean Discrepancy (MMD) with an RBF kernel.
    # Used as a distribution-preservation penalty during embedding to keep the
    # modified Gaussian parameter distribution close to the clean distribution.
    # Related to the statistical detectability framework in
    #   Pevný et al. HUGO [IH 2010] (cost computed from local statistics) and
    #   Gretton et al. "A Kernel Two-Sample Test" [JMLR 2012] for the kernel choice.
    # In our setting, the feature space is the Gaussian parameter manifold rather
    # than pixel space or a DCT domain.
    clean_x = cover_feature_matrix(clean_params)
    compare_x = cover_feature_matrix(compare_params)
    pairwise = torch.cdist(clean_x.detach(), clean_x.detach(), p=2) ** 2
    positive = pairwise[pairwise > 0]
    sigma2 = positive.median().clamp_min(1e-3) if positive.numel() > 0 else torch.tensor(1.0, device=clean_x.device)

    def kernel(a: Tensor, b: Tensor) -> Tensor:
        return torch.exp(-torch.cdist(a, b, p=2) ** 2 / (2.0 * sigma2))

    k_xx = kernel(clean_x, clean_x).mean()
    k_yy = kernel(compare_x, compare_x).mean()
    k_xy = kernel(clean_x, compare_x).mean()
    return k_xx + k_yy - 2.0 * k_xy


def sliced_wasserstein_loss(
    clean_params: dict[str, Tensor],
    compare_params: dict[str, Tensor],
    *,
    projections: int = 32,
    seed: int = 0,
) -> Tensor:
    # Sliced Wasserstein distance as a complementary distribution-shift penalty.
    # Unlike MMD it avoids the kernel bandwidth choice and is more sensitive to
    # tail behaviour — particularly relevant for the heavy-tailed opacity (alpha)
    # and log-anisotropy distributions in Gaussian Splatting.
    # Conceptually related to frequency-domain detectability metrics used in
    # DCT-domain image steganography (Holub & Fridrich [IEEE TIFS 2014]:
    # "Universal Distortion Function for Steganography in an Arbitrary Domain"):
    # both measure the shift between clean and stego distributions as an
    # embedding cost.  https://doi.org/10.1109/TIFS.2013.2286692
    clean_x = cover_feature_matrix(clean_params)
    compare_x = cover_feature_matrix(compare_params)
    if clean_x.shape[0] == 0 or compare_x.shape[0] == 0:
        return torch.tensor(0.0, dtype=clean_x.dtype, device=clean_x.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 991)
    losses: list[Tensor] = []
    for _ in range(max(1, projections)):
        direction = torch.randn(clean_x.shape[1], generator=generator, dtype=clean_x.dtype)
        direction = direction.to(clean_x.device)
        direction = direction / direction.norm().clamp_min(1e-6)
        clean_proj = torch.sort(clean_x @ direction)[0]
        compare_proj = torch.sort(compare_x @ direction)[0]
        count = min(clean_proj.numel(), compare_proj.numel())
        clean_slice = clean_proj[:count]
        compare_slice = compare_proj[:count]
        losses.append(torch.mean((clean_slice - compare_slice) ** 2))
    return torch.stack(losses).mean()


def _smooth_tail_penalty(z_scores: Tensor, threshold: float = 2.5) -> Tensor:
    return torch.sigmoid(4.0 * (z_scores.abs() - threshold))


def _channel_boundary_penalty(values: Tensor, lower: float, upper: float, delta: float) -> Tensor:
    lower_margin = (values - lower) / max(delta, 1e-6)
    upper_margin = (upper - values) / max(delta, 1e-6)
    return torch.relu(1.0 - torch.minimum(lower_margin, upper_margin))


def _top_fraction_mask(values: Tensor, fraction: float) -> Tensor:
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    count = max(1, int(math.ceil(float(fraction) * float(values.numel()))))
    count = min(count, int(values.numel()))
    indices = torch.topk(values, k=count, largest=True).indices
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[indices] = True
    return mask


def build_channel_costs(
    *,
    clean_params: dict[str, Tensor],
    importance: Tensor,
    sensitivity: dict[str, Tensor],
    wet_threshold: float = 0.20,
    log_delta: float = 0.12,
    alpha_delta: float = 0.04,
    # S0-C: Cost weight parameters (configurable for ablation studies).
    # Default values (0.40/0.30/0.20/0.10) were chosen to weight rendering-geometry
    # sensitivity most heavily, followed by distribution-shift, visual importance,
    # and the tail-detector penalty.  Pass different values to generate ablation data.
    cost_sensitivity_w: float = 0.40,
    cost_density_w: float = 0.30,
    cost_visual_w: float = 0.20,
    cost_detector_w: float = 0.10,
) -> ChannelCostBundle:
    features = cover_feature_matrix(clean_params)
    canonical = _canonicalize_geometry(clean_params["scales"], clean_params["thetas"])
    alpha = clean_params["opacities"][:, 0]
    importance_norm = importance / importance.mean().clamp_min(1e-6)
    density_reference = build_density_reference(clean_params)

    z_scores = torch.stack([robust_z_scores(features[:, index]) for index in range(features.shape[1])], dim=1)
    outlier_mask = (z_scores.abs() > 2.5).any(dim=1)
    importance_mask = _top_fraction_mask(importance_norm, 0.10)
    log_sensitivity_mask = _top_fraction_mask(sensitivity["log_anisotropy"], wet_threshold)
    alpha_sensitivity_mask = _top_fraction_mask(sensitivity["alpha"], wet_threshold)
    boundary_mask = (
        (alpha < alpha_delta)
        | (alpha > 1.0 - alpha_delta)
        | (clean_params["scales"].min(dim=1).values < 0.03)
    )
    wet_mask = (
        outlier_mask
        | importance_mask
        | log_sensitivity_mask
        | alpha_sensitivity_mask
        | boundary_mask
    )

    log_values = canonical["log_anisotropy"]
    alpha_values = alpha
    log_mean = density_reference["mean"][3]
    log_std = density_reference["std"][3]
    alpha_mean = density_reference["mean"][5]
    alpha_std = density_reference["std"][5]

    log_plus_density = torch.abs((log_values + log_delta - log_mean) / log_std)
    log_minus_density = torch.abs((log_values - log_delta - log_mean) / log_std)
    alpha_plus_density = torch.abs((alpha_values + alpha_delta - alpha_mean) / alpha_std)
    alpha_minus_density = torch.abs((alpha_values - alpha_delta - alpha_mean) / alpha_std)

    log_visual = 0.6 * importance_norm + 0.4 * _smooth_tail_penalty(z_scores[:, 3])
    alpha_visual = 0.6 * importance_norm + 0.4 * _channel_boundary_penalty(alpha_values, 0.0, 1.0, alpha_delta)
    log_detector = 0.5 * _smooth_tail_penalty(z_scores[:, 3]) + 0.5 * _smooth_tail_penalty(z_scores[:, 2])
    alpha_detector = 0.5 * _smooth_tail_penalty(z_scores[:, 5]) + 0.5 * _smooth_tail_penalty(z_scores[:, 2])
    log_sens = sensitivity["log_anisotropy"] / sensitivity["log_anisotropy"].mean().clamp_min(1e-6)
    alpha_sens = sensitivity["alpha"] / sensitivity["alpha"].mean().clamp_min(1e-6)

    log_rho_plus = cost_sensitivity_w * log_sens + cost_density_w * log_plus_density + cost_visual_w * log_visual + cost_detector_w * log_detector
    log_rho_minus = cost_sensitivity_w * log_sens + cost_density_w * log_minus_density + cost_visual_w * log_visual + cost_detector_w * log_detector
    alpha_rho_plus = cost_sensitivity_w * alpha_sens + cost_density_w * alpha_plus_density + cost_visual_w * alpha_visual + cost_detector_w * alpha_detector
    alpha_rho_minus = cost_sensitivity_w * alpha_sens + cost_density_w * alpha_minus_density + cost_visual_w * alpha_visual + cost_detector_w * alpha_detector

    return ChannelCostBundle(
        wet_mask=wet_mask,
        log_rho_plus=log_rho_plus,
        log_rho_minus=log_rho_minus,
        alpha_rho_plus=alpha_rho_plus,
        alpha_rho_minus=alpha_rho_minus,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
        density_reference=density_reference,
    )


def _minimum_carrier_scale(num_gaussians: int) -> float:
    """Return the scale floor used to exclude degenerate primitives."""
    count = max(1, int(num_gaussians))
    return min(0.03, 0.5 / math.sqrt(float(count)))


def build_hard_forbid_mask(
    *,
    clean_params: dict[str, Tensor],
    importance: Tensor,
    sensitivity: dict[str, Tensor],
) -> Tensor:
    features = cover_feature_matrix(clean_params)
    alpha = clean_params["opacities"][:, 0]
    outlier_mask = (torch.stack([robust_z_scores(features[:, index]) for index in range(features.shape[1])], dim=1).abs() > 3.0).any(dim=1)
    importance_mask = _top_fraction_mask(importance / importance.mean().clamp_min(1e-6), 0.05)
    minimum_scale = _minimum_carrier_scale(int(clean_params["scales"].shape[0]))
    boundary_mask = (
        (alpha < 0.03)
        | (alpha > 0.97)
        | (clean_params["scales"].min(dim=1).values < minimum_scale)
    )
    sensitivity_floor = (
        _top_fraction_mask(sensitivity["log_anisotropy"], 0.05)
        | _top_fraction_mask(sensitivity["alpha"], 0.05)
    )
    return outlier_mask | importance_mask | boundary_mask | sensitivity_floor


def _binary_entropy(probabilities: Tensor) -> Tensor:
    probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
    return -(probabilities * probabilities.log() + (1.0 - probabilities) * (1.0 - probabilities).log())


def build_soft_selection_field(
    *,
    clean_params: dict[str, Tensor],
    importance: Tensor,
    sensitivity: dict[str, Tensor],
    cost_bundle: ChannelCostBundle,
    key: str,
    selection_temperature: float,
    usage_penalty: Tensor | None = None,
) -> CostCodeV2Bundle:
    hard_forbid = build_hard_forbid_mask(
        clean_params=clean_params,
        importance=importance,
        sensitivity=sensitivity,
    )
    min_cost = torch.minimum(
        torch.minimum(cost_bundle.log_rho_plus, cost_bundle.log_rho_minus),
        torch.minimum(cost_bundle.alpha_rho_plus, cost_bundle.alpha_rho_minus),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(key_to_seed(key) + 19)
    keyed_jitter = torch.rand(min_cost.numel(), generator=generator, dtype=min_cost.dtype).to(min_cost.device)
    keyed_jitter = torch.log(keyed_jitter.clamp_min(1e-6)) - torch.log1p(-keyed_jitter.clamp(max=1.0 - 1e-6))
    logits = -min_cost / max(float(selection_temperature), 1e-6)
    logits = logits + 0.15 * keyed_jitter
    if usage_penalty is not None:
        logits = logits - usage_penalty.to(logits.device)
    logits = torch.where(hard_forbid, torch.full_like(logits, -1e9), logits)
    probs = torch.softmax(logits, dim=0)
    probs = torch.where(hard_forbid, torch.zeros_like(probs), probs)
    entropy = float(_binary_entropy(probs.clamp(0.0, 1.0)).mean().item())
    effective_action_budget = int((~hard_forbid).sum().item()) * 2
    return CostCodeV2Bundle(
        hard_forbid_mask=hard_forbid,
        selection_logits=logits,
        selection_probs=probs,
        selection_entropy=entropy,
        effective_action_budget=effective_action_budget,
        log_delta=cost_bundle.log_delta,
        alpha_delta=cost_bundle.alpha_delta,
    )


def build_unkeyed_selection_field(
    *,
    clean_params: dict[str, Tensor],
    importance: Tensor,
    sensitivity: dict[str, Tensor],
    cost_bundle: ChannelCostBundle,
    selection_temperature: float,
    usage_penalty: Tensor | None = None,
) -> CostCodeV2Bundle:
    hard_forbid = build_hard_forbid_mask(
        clean_params=clean_params,
        importance=importance,
        sensitivity=sensitivity,
    )
    min_cost = torch.minimum(
        torch.minimum(cost_bundle.log_rho_plus, cost_bundle.log_rho_minus),
        torch.minimum(cost_bundle.alpha_rho_plus, cost_bundle.alpha_rho_minus),
    )
    logits = -min_cost / max(float(selection_temperature), 1e-6)
    if usage_penalty is not None:
        logits = logits - usage_penalty.to(logits.device)
    logits = torch.where(hard_forbid, torch.full_like(logits, -1e9), logits)
    probs = torch.softmax(logits, dim=0)
    probs = torch.where(hard_forbid, torch.zeros_like(probs), probs)
    entropy = float(_binary_entropy(probs.clamp(0.0, 1.0)).mean().item())
    effective_action_budget = int((~hard_forbid).sum().item()) * 2
    return CostCodeV2Bundle(
        hard_forbid_mask=hard_forbid,
        selection_logits=logits,
        selection_probs=probs,
        selection_entropy=entropy,
        effective_action_budget=effective_action_budget,
        log_delta=cost_bundle.log_delta,
        alpha_delta=cost_bundle.alpha_delta,
    )


def selection_channel_penalty(
    *,
    selection_probs: Tensor,
    clean_params: dict[str, Tensor],
    importance: Tensor,
    sensitivity: dict[str, Tensor],
) -> Tensor:
    if selection_probs.numel() == 0:
        return torch.tensor(0.0, dtype=clean_params["offsets"].dtype, device=clean_params["offsets"].device)
    canonical = cover_feature_matrix(clean_params)
    penalties: list[Tensor] = []
    for column in range(canonical.shape[1]):
        penalties.append(_safe_corr(selection_probs, canonical[:, column]).abs())
    penalties.append(_safe_corr(selection_probs, importance).abs())
    penalties.append(_safe_corr(selection_probs, sensitivity["log_anisotropy"]).abs())
    penalties.append(_safe_corr(selection_probs, sensitivity["alpha"]).abs())
    entropy = -(selection_probs.clamp_min(1e-6) * selection_probs.clamp_min(1e-6).log()).sum()
    max_entropy = torch.log(torch.tensor(float(selection_probs.numel()), dtype=selection_probs.dtype, device=selection_probs.device)).clamp_min(1e-6)
    penalties.append(torch.relu(0.80 - entropy / max_entropy))
    return torch.stack(penalties).mean()


def default_spread_factor(payload_bits: int) -> int:
    return 4 if payload_bits <= 8 else 6


def _channel_counts(spread_factor: int) -> tuple[int, int]:
    alpha_count = max(1, int(round(spread_factor / 3.0)))
    log_count = max(1, spread_factor - alpha_count)
    return log_count, alpha_count


def _serializable_weights(weights: Tensor) -> list[float]:
    return [float(value.item()) for value in weights.detach().cpu()]


def _sample_keyed_candidates(
    remaining: set[int],
    costs: Tensor,
    *,
    count: int,
    generator: torch.Generator,
    pool_factor: int = 8,
) -> list[int]:
    if count <= 0:
        return []
    if len(remaining) < count:
        return []
    ordered = sorted(remaining, key=lambda index: float(costs[index].item()))
    pool_size = min(len(ordered), max(count, pool_factor * count))
    pool = ordered[:pool_size]
    # Rank-based weights preserve cost preference while letting the key change the actual subset.
    rank_weights = torch.linspace(float(pool_size), 1.0, steps=pool_size, dtype=torch.float32)
    sampled = torch.multinomial(rank_weights, num_samples=count, replacement=False, generator=generator)
    return [int(pool[index]) for index in sampled.tolist()]


def _required_available_per_channel(coded_bit_count: int, spread_factor: int) -> int:
    log_count, alpha_count = _channel_counts(spread_factor)
    return max(int(coded_bit_count) * log_count, int(coded_bit_count) * alpha_count)


def adapt_wet_mask_for_capacity(
    *,
    cost_bundle: ChannelCostBundle,
    coded_bits: Tensor,
    requested_spread_factor: int,
) -> tuple[Tensor, int, list[int], int]:
    coded_bit_count = int(coded_bits.numel())
    total_available = int(cost_bundle.wet_mask.numel())
    dry_count = int((~cost_bundle.wet_mask).sum().item())
    effective_spread_factor = 0
    for spread_factor in range(int(requested_spread_factor), 0, -1):
        if _required_available_per_channel(coded_bit_count, spread_factor) <= total_available:
            effective_spread_factor = spread_factor
            break
    if effective_spread_factor <= 0:
        return cost_bundle.wet_mask.clone(), 0, [], dry_count

    required_available = _required_available_per_channel(coded_bit_count, effective_spread_factor)
    relaxed_needed = max(0, required_available - dry_count)
    if relaxed_needed == 0:
        return cost_bundle.wet_mask.clone(), effective_spread_factor, [], dry_count

    wet_indices = torch.where(cost_bundle.wet_mask)[0]
    if int(wet_indices.numel()) < relaxed_needed:
        return cost_bundle.wet_mask.clone(), 0, [], dry_count

    combined_cost = torch.minimum(
        torch.minimum(cost_bundle.log_rho_plus, cost_bundle.log_rho_minus),
        torch.minimum(cost_bundle.alpha_rho_plus, cost_bundle.alpha_rho_minus),
    )
    ordered_relaxed = wet_indices[torch.argsort(combined_cost[wet_indices], descending=False)[:relaxed_needed]]
    effective_wet_mask = cost_bundle.wet_mask.clone()
    effective_wet_mask[ordered_relaxed] = False
    return (
        effective_wet_mask,
        effective_spread_factor,
        [int(index) for index in ordered_relaxed.detach().cpu().tolist()],
        dry_count,
    )


def select_keyed_parity_blocks(
    *,
    coded_bits: Tensor,
    cost_bundle: ChannelCostBundle,
    key: str,
    spread_factor: int,
    wet_mask_override: Tensor | None = None,
) -> tuple[list[dict[str, object]], bool, float]:
    coded_bits_cpu = coded_bits.detach().cpu().to(torch.int64)
    log_count, alpha_count = _channel_counts(spread_factor)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(key_to_seed(key))
    wet_mask = cost_bundle.wet_mask if wet_mask_override is None else wet_mask_override.detach().to(torch.bool)
    available_log = (~wet_mask).nonzero(as_tuple=False)[:, 0].detach().cpu()
    available_alpha = (~wet_mask).nonzero(as_tuple=False)[:, 0].detach().cpu()
    required_log = int(coded_bits_cpu.numel()) * log_count
    required_alpha = int(coded_bits_cpu.numel()) * alpha_count
    if available_log.numel() < required_log or available_alpha.numel() < required_alpha:
        return [], True, float("inf")

    selected_blocks: list[dict[str, object]] = []
    total_embed_cost = 0.0
    remaining_log = set(int(index) for index in available_log.tolist())
    remaining_alpha = set(int(index) for index in available_alpha.tolist())
    for bit_index, bit in enumerate(coded_bits_cpu.tolist()):
        mask_bit = int(torch.randint(0, 2, size=(1,), generator=generator, dtype=torch.int64).item())
        embed_bit = int(bit) ^ mask_bit
        polarity_sign = 1 if int(torch.randint(0, 2, size=(1,), generator=generator, dtype=torch.int64).item()) > 0 else -1
        use_plus = ((1 if embed_bit > 0 else -1) * polarity_sign) > 0
        log_costs = cost_bundle.log_rho_plus if use_plus else cost_bundle.log_rho_minus
        alpha_costs = cost_bundle.alpha_rho_plus if use_plus else cost_bundle.alpha_rho_minus
        log_candidates = _sample_keyed_candidates(
            remaining_log,
            log_costs,
            count=log_count,
            generator=generator,
        )
        alpha_candidates = _sample_keyed_candidates(
            remaining_alpha,
            alpha_costs,
            count=alpha_count,
            generator=generator,
        )
        if len(log_candidates) < log_count or len(alpha_candidates) < alpha_count:
            return [], True, float("inf")
        for index in log_candidates:
            remaining_log.remove(index)
            total_embed_cost += float(log_costs[index].item())
        for index in alpha_candidates:
            remaining_alpha.remove(index)
            total_embed_cost += float(alpha_costs[index].item())
        log_weights = 1.0 / torch.tensor(
            [float(log_costs[index].item()) for index in log_candidates],
            dtype=torch.float32,
        ).clamp_min(1e-6)
        alpha_weights = 1.0 / torch.tensor(
            [float(alpha_costs[index].item()) for index in alpha_candidates],
            dtype=torch.float32,
        ).clamp_min(1e-6)
        log_weights = log_weights / log_weights.sum().clamp_min(1e-6)
        alpha_weights = alpha_weights / alpha_weights.sum().clamp_min(1e-6)
        selected_blocks.append(
            {
                "block_index": bit_index,
                "bit": int(bit),
                "mask_bit": mask_bit,
                "embed_bit": embed_bit,
                "polarity_sign": polarity_sign,
                "log_indices": [int(index) for index in log_candidates],
                "alpha_indices": [int(index) for index in alpha_candidates],
                "log_weights": _serializable_weights(log_weights),
                "alpha_weights": _serializable_weights(alpha_weights),
            }
        )
    return selected_blocks, False, total_embed_cost


def serializable_block_assignments(assignments: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "block_index": int(item["block_index"]),
            "bit": int(item["bit"]),
            "mask_bit": int(item.get("mask_bit", 0)),
            "embed_bit": int(item.get("embed_bit", int(item["bit"]))),
            "polarity_sign": int(item.get("polarity_sign", 1)),
            "log_indices": [int(index) for index in item["log_indices"]],
            "alpha_indices": [int(index) for index in item["alpha_indices"]],
            "log_weights": [float(weight) for weight in item["log_weights"]],
            "alpha_weights": [float(weight) for weight in item["alpha_weights"]],
        }
        for item in assignments
    ]


def assignment_bit_tensor(assignments: list[dict[str, object]], *, field: str = "bit") -> Tensor:
    return torch.tensor([int(item.get(field, item["bit"])) for item in assignments], dtype=torch.int64)


def _unique_active_indices(assignments: list[dict[str, object]], key: str) -> Tensor:
    indices = sorted({int(index) for assignment in assignments for index in assignment[key]})
    return torch.tensor(indices, dtype=torch.long)


def active_channel_indices(assignments: list[dict[str, object]]) -> tuple[Tensor, Tensor]:
    return _unique_active_indices(assignments, "log_indices"), _unique_active_indices(assignments, "alpha_indices")


def modified_gaussian_count(assignments: list[dict[str, object]]) -> int:
    return int(
        len(
            {
                int(index)
                for assignment in assignments
                for index in [*assignment["log_indices"], *assignment["alpha_indices"]]
            }
        )
    )


def _device_block_weights(assignments: list[dict[str, object]], *, key: str, device: torch.device) -> list[Tensor]:
    return [torch.tensor(item[key], dtype=torch.float32, device=device) for item in assignments]


def parity_block_logits(
    params: dict[str, Tensor],
    *,
    clean_params: dict[str, Tensor],
    assignments: list[dict[str, object]],
    log_delta: float,
    alpha_delta: float,
    alpha_weight: float,
) -> Tensor:
    canonical = _canonicalize_geometry(params["scales"], params["thetas"])
    clean_canonical = _canonicalize_geometry(clean_params["scales"], clean_params["thetas"])
    current_log = canonical["log_anisotropy"]
    clean_log = clean_canonical["log_anisotropy"].to(current_log.device)
    current_alpha = params["opacities"][:, 0]
    clean_alpha = clean_params["opacities"][:, 0].to(current_alpha.device)

    logits: list[Tensor] = []
    for item in assignments:
        log_indices = torch.tensor(item["log_indices"], dtype=torch.long, device=current_log.device)
        alpha_indices = torch.tensor(item["alpha_indices"], dtype=torch.long, device=current_alpha.device)
        log_weights = torch.tensor(item["log_weights"], dtype=torch.float32, device=current_log.device)
        alpha_weights = torch.tensor(item["alpha_weights"], dtype=torch.float32, device=current_alpha.device)
        log_score = torch.sum(log_weights * ((current_log[log_indices] - clean_log[log_indices]) / max(log_delta, 1e-6)))
        alpha_score = torch.sum(alpha_weights * ((current_alpha[alpha_indices] - clean_alpha[alpha_indices]) / max(alpha_delta, 1e-6)))
        logits.append(float(item.get("polarity_sign", 1)) * (log_score + alpha_weight * alpha_score))
    if not logits:
        return torch.empty(0, dtype=torch.float32, device=clean_params["offsets"].device)
    return torch.stack(logits)


def carrier_margin_loss(
    params: dict[str, Tensor],
    *,
    clean_params: dict[str, Tensor],
    assignments: list[dict[str, object]],
    log_delta: float,
    alpha_delta: float,
    margin: float = 0.75,
) -> Tensor:
    canonical = _canonicalize_geometry(params["scales"], params["thetas"])
    clean_canonical = _canonicalize_geometry(clean_params["scales"], clean_params["thetas"])
    current_log = canonical["log_anisotropy"]
    clean_log = clean_canonical["log_anisotropy"].to(current_log.device)
    current_alpha = params["opacities"][:, 0]
    clean_alpha = clean_params["opacities"][:, 0].to(current_alpha.device)
    losses: list[Tensor] = []
    for item in assignments:
        effective_sign = float(item.get("polarity_sign", 1)) * (1.0 if int(item.get("embed_bit", item["bit"])) > 0 else -1.0)
        log_indices = torch.tensor(item["log_indices"], dtype=torch.long, device=current_log.device)
        alpha_indices = torch.tensor(item["alpha_indices"], dtype=torch.long, device=current_alpha.device)
        if log_indices.numel() > 0:
            log_weights = torch.tensor(item["log_weights"], dtype=torch.float32, device=current_log.device)
            log_signed = effective_sign * ((current_log[log_indices] - clean_log[log_indices]) / max(log_delta, 1e-6))
            losses.append(torch.sum(log_weights * torch.relu(float(margin) - log_signed)))
        if alpha_indices.numel() > 0:
            alpha_weights = torch.tensor(item["alpha_weights"], dtype=torch.float32, device=current_alpha.device)
            alpha_signed = effective_sign * ((current_alpha[alpha_indices] - clean_alpha[alpha_indices]) / max(alpha_delta, 1e-6))
            losses.append(torch.sum(alpha_weights * torch.relu(float(margin) - alpha_signed)))
    if not losses:
        return torch.tensor(0.0, dtype=clean_params["offsets"].dtype, device=clean_params["offsets"].device)
    return torch.stack(losses).mean()


def decode_parity_blocks(
    params: dict[str, Tensor],
    *,
    clean_params: dict[str, Tensor],
    assignments: list[dict[str, object]],
    log_delta: float,
    alpha_delta: float,
    alpha_weight: float,
) -> tuple[Tensor, Tensor]:
    logits = parity_block_logits(
        params,
        clean_params=clean_params,
        assignments=assignments,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
        alpha_weight=alpha_weight,
    )
    if logits.numel() == 0:
        return torch.empty(0, dtype=torch.int64), logits.detach().cpu()
    embed_bits = (logits > 0.0).to(torch.int64)
    mask_bits = torch.tensor(
        [int(item.get("mask_bit", 0)) for item in assignments],
        dtype=torch.int64,
        device=embed_bits.device,
    )
    decoded = torch.remainder(embed_bits + mask_bits, 2)
    return decoded, logits.detach().cpu()


def _bits_to_state(bits: Iterable[int]) -> int:
    state = 0
    for index, value in enumerate(bits):
        state |= (int(value) & 1) << index
    return state


def _state_to_bits(state: int, width: int = 4) -> list[int]:
    return [int((state >> index) & 1) for index in range(width)]


def _gf2_rank(matrix: Tensor) -> int:
    mat = matrix.to(torch.int64).clone()
    rank = 0
    rows, cols = mat.shape
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if int(mat[row, col].item()) == 1:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            tmp = mat[rank].clone()
            mat[rank] = mat[pivot]
            mat[pivot] = tmp
        for row in range(rows):
            if row != rank and int(mat[row, col].item()) == 1:
                mat[row] = torch.remainder(mat[row] + mat[rank], 2)
        rank += 1
        if rank == rows:
            break
    return rank


def _make_parity_matrix(num_actions: int, *, generator: torch.Generator) -> Tensor:
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")
    while True:
        columns = torch.randint(0, 2, size=(4, num_actions), generator=generator, dtype=torch.int64)
        nonzero = columns.sum(dim=0) > 0
        if not bool(nonzero.all()):
            columns[:, ~nonzero] = 0
            replacement = torch.randint(1, 16, size=(int((~nonzero).sum().item()),), generator=generator, dtype=torch.int64)
            for offset, packed in enumerate(replacement.tolist()):
                columns[:, torch.where(~nonzero)[0][offset]] = torch.tensor(_state_to_bits(int(packed)), dtype=torch.int64)
        if _gf2_rank(columns) == 4:
            return columns


def _gumbel_topk(logits: Tensor, *, k: int, generator: torch.Generator) -> Tensor:
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=logits.device)
    uniform = torch.rand(logits.shape[0], generator=generator, dtype=logits.dtype).to(logits.device).clamp(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(uniform))
    return torch.topk(logits + gumbel, k=min(k, logits.numel()), largest=True).indices


def _cost_code_block_sort_key(
    item: tuple[float, int, tuple[int, ...]],
    *,
    min_selected_count: int | None = None,
) -> tuple[float, int, int]:
    cost, _state, selected = item
    selected_count = len(selected)
    if min_selected_count is None:
        return (cost, 0, selected_count)
    selected_floor = max(0, int(min_selected_count))
    shortage = max(0, selected_floor - selected_count)
    return (cost, shortage, -selected_count)


def _solve_cost_code_block(
    *,
    candidate_costs: Tensor,
    parity_matrix: Tensor,
    target_bits: Tensor,
    beam_width: int,
    min_selected_count: int | None = None,
) -> Tensor:
    target_state = _bits_to_state(target_bits.tolist())
    beams: list[tuple[float, int, tuple[int, ...]]] = [(0.0, 0, tuple())]
    for action_index in range(candidate_costs.numel()):
        column_state = _bits_to_state(parity_matrix[:, action_index].tolist())
        next_beams: list[tuple[float, int, tuple[int, ...]]] = []
        action_cost = float(candidate_costs[action_index].item())
        for cost, state, selected in beams:
            next_beams.append((cost, state, selected))
            next_beams.append((cost + action_cost, state ^ column_state, (*selected, action_index)))
        next_beams.sort(key=lambda item: _cost_code_block_sort_key(item, min_selected_count=min_selected_count))
        compact: list[tuple[float, int, tuple[int, ...]]] = []
        seen: set[tuple[int, tuple[int, ...]]] = set()
        for item in next_beams:
            signature = (item[1], item[2])
            if signature in seen:
                continue
            seen.add(signature)
            compact.append(item)
            if len(compact) >= max(1, int(beam_width)):
                break
        beams = compact
    valid = [item for item in beams if item[1] == target_state]
    if not valid:
        raise RuntimeError("cost_code_v1 could not satisfy the target syndrome with the current candidate set")
    _, _, selected = min(valid, key=lambda item: _cost_code_block_sort_key(item, min_selected_count=min_selected_count))
    mask = torch.zeros(candidate_costs.numel(), dtype=torch.float32)
    if selected:
        mask[torch.tensor(list(selected), dtype=torch.long)] = 1.0
    return mask


def _candidate_action_option(
    gaussian_index: int,
    *,
    cost_bundle: ChannelCostBundle,
    generator: torch.Generator,
    selection_temperature: float,
) -> tuple[str, int, float]:
    keyed_jitter = torch.rand(4, generator=generator, dtype=torch.float32)
    keyed_jitter = torch.log(keyed_jitter.clamp_min(1e-6)) - torch.log1p(-keyed_jitter.clamp(max=1.0 - 1e-6))
    costs = torch.tensor(
        [
            float(cost_bundle.log_rho_plus[gaussian_index].item()),
            float(cost_bundle.log_rho_minus[gaussian_index].item()),
            float(cost_bundle.alpha_rho_plus[gaussian_index].item()),
            float(cost_bundle.alpha_rho_minus[gaussian_index].item()),
        ],
        dtype=torch.float32,
    )
    logits = -costs / max(float(selection_temperature), 1e-6) + 0.10 * keyed_jitter
    choice = int(torch.argmax(logits).item())
    if choice == 0:
        return "log", 1, float(costs[0].item())
    if choice == 1:
        return "log", -1, float(costs[1].item())
    if choice == 2:
        return "alpha", 1, float(costs[2].item())
    return "alpha", -1, float(costs[3].item())


def _lowest_cost_action_option(
    gaussian_index: int,
    *,
    cost_bundle: ChannelCostBundle,
) -> tuple[str, int, float]:
    costs = torch.tensor(
        [
            float(cost_bundle.log_rho_plus[gaussian_index].item()),
            float(cost_bundle.log_rho_minus[gaussian_index].item()),
            float(cost_bundle.alpha_rho_plus[gaussian_index].item()),
            float(cost_bundle.alpha_rho_minus[gaussian_index].item()),
        ],
        dtype=torch.float32,
    )
    choice = int(torch.argmin(costs).item())
    if choice == 0:
        return "log", 1, float(costs[0].item())
    if choice == 1:
        return "log", -1, float(costs[1].item())
    if choice == 2:
        return "alpha", 1, float(costs[2].item())
    return "alpha", -1, float(costs[3].item())


def build_generic_cost_code_assignments(
    *,
    raw_bits: Tensor,
    candidate_ids: Tensor,
    selection_logits: Tensor,
    selection_probs: Tensor,
    plus_costs: Tensor,
    minus_costs: Tensor,
    key: str,
    action_type: str,
    candidate_count: int = 24,
    fallback_candidate_count: int = 16,
    beam_width: int = 64,
    max_modification_count: int,
    min_selected_count: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw_bits = raw_bits.to(torch.int64).flatten()
    if raw_bits.numel() % 4 != 0:
        pad = (-raw_bits.numel()) % 4
        raw_bits = torch.cat((raw_bits, torch.zeros(pad, dtype=torch.int64)))
    available_indices = candidate_ids.detach().cpu().to(torch.long)
    base_logits = selection_logits.detach().cpu().to(torch.float32)
    base_probs = selection_probs.detach().cpu().to(torch.float32)
    plus = plus_costs.detach().cpu().to(torch.float32)
    minus = minus_costs.detach().cpu().to(torch.float32)
    assignments: list[dict[str, object]] = []
    used_candidates: set[int] = set()
    total_selected = 0
    for block_index, block_bits in enumerate(raw_bits.view(-1, 4)):
        current_positions = [pos for pos, candidate_id in enumerate(available_indices.tolist()) if int(candidate_id) not in used_candidates]
        if len(current_positions) < 4:
            raise RuntimeError("generic cost_code_v1 ran out of available candidates")
        desired = candidate_count if len(current_positions) >= candidate_count else min(len(current_positions), fallback_candidate_count)
        base_seed = key_to_seed(f"{key}::generic::{block_index}") + 211
        current_logits = base_logits[torch.tensor(current_positions, dtype=torch.long)]
        selected_mask: Tensor | None = None
        actions: list[dict[str, object]] = []
        action_costs: list[float] = []
        parity_matrix: Tensor | None = None
        for retry_index in range(4):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(base_seed + 997 * retry_index)
            retry_desired = desired if retry_index == 0 else min(len(current_positions), max(4, fallback_candidate_count))
            sampled = _gumbel_topk(current_logits, k=retry_desired, generator=generator).detach().cpu()
            sampled_positions = [current_positions[int(index)] for index in sampled.tolist()]
            action_costs = []
            actions = []
            for position in sampled_positions:
                candidate_id = int(available_indices[position].item())
                jitter = torch.rand(2, generator=generator, dtype=torch.float32)
                jitter = torch.log(jitter.clamp_min(1e-6)) - torch.log1p(-jitter.clamp(max=1.0 - 1e-6))
                choose_plus = float(plus[position].item()) - 0.05 * float(jitter[0].item()) <= float(minus[position].item()) - 0.05 * float(jitter[1].item())
                direction = 1 if choose_plus else -1
                strength_seed = torch.rand(1, generator=generator, dtype=torch.float32).item()
                if strength_seed < 0.34:
                    strength_scale = 0.80
                elif strength_seed < 0.67:
                    strength_scale = 1.00
                else:
                    strength_scale = 1.20
                action_costs.append(float(plus[position].item()) if choose_plus else float(minus[position].item()))
                actions.append(
                    {
                        "candidate_id": candidate_id,
                        "action_type": action_type,
                        "direction": direction,
                        "strength_scale": strength_scale,
                        "selection_probability": float(base_probs[position].item()),
                    }
                )
            parity_matrix = _make_parity_matrix(len(actions), generator=generator)
            try:
                selected_mask = _solve_cost_code_block(
                    candidate_costs=torch.tensor(action_costs, dtype=torch.float32),
                    parity_matrix=parity_matrix,
                    target_bits=block_bits,
                    beam_width=beam_width,
                    min_selected_count=min_selected_count,
                )
                break
            except RuntimeError:
                selected_mask = None
                continue
        if selected_mask is None or parity_matrix is None:
            raise RuntimeError("generic cost_code_v1 could not satisfy the target syndrome after retrying candidate pools")
        selected_positions = torch.where(selected_mask > 0.5)[0].tolist()
        total_selected += len(selected_positions)
        if total_selected > int(max_modification_count):
            raise RuntimeError("generic cost_code_v1 exceeded the maximum modification count")
        for action in actions:
            used_candidates.add(int(action["candidate_id"]))
        assignments.append(
            {
                "block_index": int(block_index),
                "message_bits": [int(value) for value in block_bits.tolist()],
                "candidate_count": len(actions),
                "candidate_actions": actions,
                "candidate_costs": action_costs,
                "parity_matrix": parity_matrix.t().tolist(),
                "selected_mask": [float(value) for value in selected_mask.tolist()],
                "selected_count": int(len(selected_positions)),
            }
        )
    entropy = -(base_probs.clamp_min(1e-6) * base_probs.clamp_min(1e-6).log()).sum()
    max_entropy = math.log(max(1, int(base_probs.numel())))
    return assignments, {
        "selection_entropy": float((entropy / max(max_entropy, 1e-6)).item() if hasattr(entropy, "item") else entropy / max(max_entropy, 1e-6)),
        "effective_action_budget": int(available_indices.numel()),
        "total_selected_actions": int(total_selected),
    }


def generic_cost_code_probabilities(
    *,
    assignments: list[dict[str, object]],
    action_score_provider,
    beta: float = 4.0,
) -> tuple[Tensor, list[Tensor]]:
    block_probabilities: list[Tensor] = []
    action_scores_per_block: list[Tensor] = []
    for assignment in assignments:
        scores = action_score_provider(assignment)
        if scores.numel() != int(assignment["candidate_count"]):
            raise ValueError("action_score_provider returned a score vector with the wrong shape")
        action_scores_per_block.append(scores)
        action_probs = torch.sigmoid(beta * (scores - 0.5))
        parity_matrix = torch.tensor(assignment["parity_matrix"], dtype=torch.float32, device=scores.device).t()
        bit_probs: list[Tensor] = []
        for row in range(parity_matrix.shape[0]):
            row_mask = parity_matrix[row] > 0.5
            if int(row_mask.sum().item()) == 0:
                bit_probs.append(torch.tensor(0.5, dtype=scores.dtype, device=scores.device))
                continue
            selected_probs = action_probs[row_mask]
            bit_probs.append(0.5 - 0.5 * torch.prod(1.0 - 2.0 * selected_probs))
        block_probabilities.append(torch.stack(bit_probs))
    if not block_probabilities:
        return torch.empty((0, 4), dtype=torch.float32), action_scores_per_block
    return torch.stack(block_probabilities, dim=0), action_scores_per_block


def decode_generic_cost_code(
    *,
    assignments: list[dict[str, object]],
    action_score_provider,
    raw_length: int,
    beta: float = 4.0,
) -> tuple[Tensor, list[list[float]], list[list[float]]]:
    block_probs, action_scores = generic_cost_code_probabilities(
        assignments=assignments,
        action_score_provider=action_score_provider,
        beta=beta,
    )
    if block_probs.numel() == 0:
        return torch.empty(0, dtype=torch.int64), [], []
    hard_blocks: list[Tensor] = []
    for assignment, scores in zip(assignments, action_scores):
        if scores.numel() == 0:
            hard_blocks.append(torch.zeros(4, dtype=torch.int64))
            continue
        parity_matrix = torch.tensor(assignment["parity_matrix"], dtype=torch.float32, device=scores.device).t()
        hard_actions = (scores > 0.0).to(torch.float32)
        hard_bits = torch.remainder((parity_matrix @ hard_actions).round().to(torch.int64), 2)
        hard_blocks.append(hard_bits)
    decoded = torch.cat(hard_blocks, dim=0)[:raw_length]
    return (
        decoded,
        [[float(value) for value in row.detach().cpu().tolist()] for row in block_probs],
        [[float(value) for value in row.detach().cpu().tolist()] for row in action_scores],
    )


def decode_image_cost_code(
    image: Tensor,
    *,
    clean_image: Tensor,
    assignments: list[dict[str, object]],
    raw_length: int,
    delta: float,
    beta: float = 4.0,
) -> tuple[Tensor, list[list[float]], list[list[float]]]:
    wm_y = rgb_to_ycbcr(image.clamp(0.0, 1.0))[..., 0].reshape(-1)
    clean_y = rgb_to_ycbcr(clean_image.clamp(0.0, 1.0))[..., 0].reshape(-1)
    scale = max(float(delta), 1e-6)

    def action_score_provider(assignment: dict[str, object]) -> Tensor:
        scores: list[Tensor] = []
        for action in assignment["candidate_actions"]:
            candidate_id = int(action["candidate_id"])
            if candidate_id < 0 or candidate_id >= int(wm_y.numel()) or candidate_id >= int(clean_y.numel()):
                scores.append(torch.tensor(-1.0, dtype=wm_y.dtype, device=wm_y.device))
                continue
            action_scale = max(float(action.get("strength_scale", 1.0)) * scale, 1e-6)
            signed_delta = (wm_y[candidate_id] - clean_y[candidate_id]) / action_scale
            action_strength = float(action["direction"]) * signed_delta
            scores.append(torch.clamp(2.0 * action_strength - 1.0, min=-1.0, max=1.5))
        if not scores:
            return torch.empty(0, dtype=wm_y.dtype, device=wm_y.device)
        return torch.stack(scores)

    return decode_generic_cost_code(
        assignments=assignments,
        action_score_provider=action_score_provider,
        raw_length=raw_length,
        beta=beta,
    )


def rgb_to_ycbcr(image: Tensor) -> Tensor:
    if image.shape[-1] != 3:
        raise ValueError("expected an RGB image with shape [..., 3]")
    matrix = torch.tensor(
        [
            [0.2990, 0.5870, 0.1140],
            [-0.168736, -0.331264, 0.500000],
            [0.500000, -0.418688, -0.081312],
        ],
        dtype=image.dtype,
        device=image.device,
    )
    offset = torch.tensor([0.0, 0.5, 0.5], dtype=image.dtype, device=image.device)
    flat = image.reshape(-1, 3)
    ycbcr = flat @ matrix.t() + offset
    return ycbcr.reshape_as(image)


def ycbcr_to_rgb(image: Tensor) -> Tensor:
    if image.shape[-1] != 3:
        raise ValueError("expected a YCbCr image with shape [..., 3]")
    matrix = torch.tensor(
        [
            [1.0, 0.0, 1.4020],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.7720, 0.0],
        ],
        dtype=image.dtype,
        device=image.device,
    )
    offset = torch.tensor([0.0, 0.5, 0.5], dtype=image.dtype, device=image.device)
    flat = image.reshape(-1, 3)
    rgb = (flat - offset) @ matrix.t()
    return rgb.reshape_as(image).clamp(0.0, 1.0)


def build_cost_code_assignments(
    *,
    raw_bits: Tensor,
    clean_params: dict[str, Tensor],
    importance: Tensor,
    sensitivity: dict[str, Tensor],
    cost_bundle: ChannelCostBundle,
    key: str,
    candidate_count: int = 24,
    fallback_candidate_count: int = 16,
    beam_width: int = 64,
    selection_temperature: float = 0.5,
    max_modification_ratio: float = 0.30,
    min_selected_count: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    # Cost-based parity-block assignment for the cost_stego_v1 protocol.
    #
    # Conceptual lineage:
    #   STC [Filler, Judas & Fridrich, IEEE TIFS 2011] — optimal cost-based
    #       steganography via Syndrome Trellis Coding; our parity blocks are a
    #       lightweight non-trellis approximation using a beam-search solver
    #       over the key-seeded Gaussian carrier candidates.
    #       https://doi.org/10.1109/TIFS.2011.2134094
    #
    #   HUGO [Pevný et al., IH 2010] — heuristic costs from local statistics;
    #       our sensitivity scores (log_rho) play the analogous role of the
    #       distortion measure ρ(x, y) in the Gibbs-sampled embedding.
    #
    # Key-seeded assignment: the key hash seeds the carrier subset and parity
    # matrix, providing cryptographic key-specificity without a separate
    # authentication channel.  A wrong key produces a Gaussian-random parity
    # constraint, yielding BER ≈ 0.5 under the decoder.
    raw_bits = raw_bits.to(torch.int64).flatten()
    if raw_bits.numel() % 4 != 0:
        pad = (-raw_bits.numel()) % 4
        raw_bits = torch.cat((raw_bits, torch.zeros(pad, dtype=torch.int64)))
    selection_bundle = build_soft_selection_field(
        clean_params=clean_params,
        importance=importance,
        sensitivity=sensitivity,
        cost_bundle=cost_bundle,
        key=key,
        selection_temperature=selection_temperature,
    )
    allowed_indices = torch.where(~selection_bundle.hard_forbid_mask)[0]
    modification_budget = max(1, int(math.floor(float(max_modification_ratio) * float(clean_params["offsets"].shape[0]))))
    assignments: list[dict[str, object]] = []
    used_gaussians: set[int] = set()
    total_selected = 0
    selection_mass = selection_bundle.selection_probs
    for block_index, block_bits in enumerate(raw_bits.view(-1, 4)):
        available = torch.tensor(
            [int(index) for index in allowed_indices.tolist() if int(index) not in used_gaussians],
            dtype=torch.long,
        )
        if available.numel() < 8:
            raise RuntimeError("cost_code_v1 ran out of available Gaussian carriers")
        desired = candidate_count if available.numel() >= candidate_count else min(int(available.numel()), fallback_candidate_count)
        if desired <= 0:
            raise RuntimeError("cost_code_v1 could not allocate a non-empty candidate set")
        candidate_logits = selection_bundle.selection_logits[available]
        base_seed = key_to_seed(f"{key}::block::{block_index}") + 97
        selected_mask: Tensor | None = None
        candidates: list[dict[str, object]] = []
        candidate_costs: list[float] = []
        parity_matrix: Tensor | None = None
        for retry_index in range(4):
            block_generator = torch.Generator(device="cpu")
            block_generator.manual_seed(base_seed + 997 * retry_index)
            retry_desired = desired if retry_index == 0 else min(int(available.numel()), max(8, fallback_candidate_count))
            sampled = _gumbel_topk(candidate_logits.detach().cpu(), k=retry_desired, generator=block_generator)
            candidate_indices = available[sampled]
            candidates = []
            candidate_costs = []
            for gaussian_index in candidate_indices.tolist():
                channel, direction, cost = _candidate_action_option(
                    int(gaussian_index),
                    cost_bundle=cost_bundle,
                    generator=block_generator,
                    selection_temperature=selection_temperature,
                )
                candidates.append(
                    {
                        "gaussian_index": int(gaussian_index),
                        "channel": channel,
                        "direction": int(direction),
                        "selection_probability": float(selection_mass[gaussian_index].item()),
                    }
                )
                candidate_costs.append(float(cost))
            parity_matrix = _make_parity_matrix(len(candidates), generator=block_generator)
            try:
                selected_mask = _solve_cost_code_block(
                    candidate_costs=torch.tensor(candidate_costs, dtype=torch.float32),
                    parity_matrix=parity_matrix,
                    target_bits=block_bits,
                    beam_width=beam_width,
                    min_selected_count=min_selected_count,
                )
                break
            except RuntimeError:
                selected_mask = None
                continue
        if selected_mask is None or parity_matrix is None:
            raise RuntimeError("cost_code_v1 could not satisfy the target syndrome after retrying candidate pools")
        selected_indices = torch.where(selected_mask > 0.5)[0]
        total_selected += int(selected_indices.numel())
        if total_selected > modification_budget:
            raise RuntimeError("cost_code_v1 exceeded the maximum modification ratio")
        for candidate in candidates:
            used_gaussians.add(int(candidate["gaussian_index"]))
        assignments.append(
            {
                "block_index": int(block_index),
                "message_bits": [int(value) for value in block_bits.tolist()],
                "candidate_count": len(candidates),
                "candidate_actions": candidates,
                "candidate_costs": candidate_costs,
                "parity_matrix": parity_matrix.t().tolist(),
                "selected_mask": [float(value) for value in selected_mask.tolist()],
                "selected_count": int(selected_indices.numel()),
            }
        )
    meta = {
        "selection_entropy": selection_bundle.selection_entropy,
        "effective_action_budget": selection_bundle.effective_action_budget,
        "hard_forbid_ratio": float(selection_bundle.hard_forbid_mask.float().mean().item()),
        "total_selected_actions": int(total_selected),
        "max_modification_count": modification_budget,
        "qualified_cover": bool(
            clean_params["offsets"].shape[0] >= 192
            and selection_bundle.effective_action_budget >= int(math.ceil(2.5 * float(raw_bits.numel())))
        ),
        "selection_probs": [float(value) for value in selection_bundle.selection_probs.detach().cpu().tolist()],
        "hard_forbid_mask": [bool(value) for value in selection_bundle.hard_forbid_mask.detach().cpu().tolist()],
    }
    return assignments, meta


def build_random_cost_code_assignments(
    *,
    raw_bits: Tensor,
    clean_params: dict[str, Tensor],
    importance: Tensor,
    sensitivity: dict[str, Tensor],
    cost_bundle: ChannelCostBundle,
    key: str,
    candidate_count: int = 24,
    fallback_candidate_count: int = 16,
    beam_width: int = 64,
    selection_temperature: float = 0.5,
    max_modification_ratio: float = 0.30,
    min_selected_count: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    del selection_temperature
    raw_bits = raw_bits.to(torch.int64).flatten()
    if raw_bits.numel() % 4 != 0:
        pad = (-raw_bits.numel()) % 4
        raw_bits = torch.cat((raw_bits, torch.zeros(pad, dtype=torch.int64)))
    selection_bundle = build_soft_selection_field(
        clean_params=clean_params,
        importance=importance,
        sensitivity=sensitivity,
        cost_bundle=cost_bundle,
        key=f"{key}::uniform_pool",
        selection_temperature=0.5,
    )
    allowed_indices = torch.where(~selection_bundle.hard_forbid_mask)[0]
    if allowed_indices.numel() == 0:
        raise RuntimeError("random_cost_code_v1 did not find any modifiable Gaussian carriers")
    modification_budget = max(1, int(math.floor(float(max_modification_ratio) * float(clean_params["offsets"].shape[0]))))
    selection_probs = torch.zeros_like(selection_bundle.selection_probs)
    selection_probs[allowed_indices] = 1.0 / float(allowed_indices.numel())
    entropy = -(selection_probs[allowed_indices] * selection_probs[allowed_indices].clamp_min(1e-6).log()).sum()
    max_entropy = torch.log(torch.tensor(float(max(1, allowed_indices.numel())), dtype=selection_probs.dtype)).clamp_min(1e-6)

    assignments: list[dict[str, object]] = []
    used_gaussians: set[int] = set()
    total_selected = 0
    for block_index, block_bits in enumerate(raw_bits.view(-1, 4)):
        available = torch.tensor(
            [int(index) for index in allowed_indices.tolist() if int(index) not in used_gaussians],
            dtype=torch.long,
        )
        if available.numel() < 8:
            raise RuntimeError("random_cost_code_v1 ran out of available Gaussian carriers")
        desired = candidate_count if available.numel() >= candidate_count else min(int(available.numel()), fallback_candidate_count)
        if desired <= 0:
            raise RuntimeError("random_cost_code_v1 could not allocate a non-empty candidate set")
        base_seed = key_to_seed(f"{key}::random_block::{block_index}") + 211
        selected_mask: Tensor | None = None
        candidates: list[dict[str, object]] = []
        actual_candidate_costs: list[float] = []
        parity_matrix: Tensor | None = None
        for retry_index in range(4):
            block_generator = torch.Generator(device="cpu")
            block_generator.manual_seed(base_seed + 997 * retry_index)
            retry_desired = desired if retry_index == 0 else min(int(available.numel()), max(8, fallback_candidate_count))
            shuffled = torch.randperm(int(available.numel()), generator=block_generator)
            candidate_indices = available[shuffled[:retry_desired]]
            candidates = []
            actual_candidate_costs = []
            for gaussian_index in candidate_indices.tolist():
                choice = int(torch.randint(0, 4, size=(1,), generator=block_generator).item())
                if choice == 0:
                    channel = "log"
                    direction = 1
                    cost = float(cost_bundle.log_rho_plus[gaussian_index].item())
                elif choice == 1:
                    channel = "log"
                    direction = -1
                    cost = float(cost_bundle.log_rho_minus[gaussian_index].item())
                elif choice == 2:
                    channel = "alpha"
                    direction = 1
                    cost = float(cost_bundle.alpha_rho_plus[gaussian_index].item())
                else:
                    channel = "alpha"
                    direction = -1
                    cost = float(cost_bundle.alpha_rho_minus[gaussian_index].item())
                candidates.append(
                    {
                        "gaussian_index": int(gaussian_index),
                        "channel": channel,
                        "direction": int(direction),
                        "selection_probability": float(selection_probs[gaussian_index].item()),
                    }
                )
                actual_candidate_costs.append(cost)
            parity_matrix = _make_parity_matrix(len(candidates), generator=block_generator)
            try:
                selected_mask = _solve_cost_code_block(
                    candidate_costs=torch.ones(len(candidates), dtype=torch.float32),
                    parity_matrix=parity_matrix,
                    target_bits=block_bits,
                    beam_width=beam_width,
                    min_selected_count=min_selected_count,
                )
                break
            except RuntimeError:
                selected_mask = None
                continue
        if selected_mask is None or parity_matrix is None:
            raise RuntimeError("random_cost_code_v1 could not satisfy the target syndrome after retrying candidate pools")
        selected_indices = torch.where(selected_mask > 0.5)[0]
        total_selected += int(selected_indices.numel())
        if total_selected > modification_budget:
            raise RuntimeError("random_cost_code_v1 exceeded the maximum modification ratio")
        for candidate in candidates:
            used_gaussians.add(int(candidate["gaussian_index"]))
        assignments.append(
            {
                "block_index": int(block_index),
                "message_bits": [int(value) for value in block_bits.tolist()],
                "candidate_count": len(candidates),
                "candidate_actions": candidates,
                "candidate_costs": actual_candidate_costs,
                "parity_matrix": parity_matrix.t().tolist(),
                "selected_mask": [float(value) for value in selected_mask.tolist()],
                "selected_count": int(selected_indices.numel()),
            }
        )
    meta = {
        "selection_entropy": float((entropy / max_entropy).item()),
        "effective_action_budget": selection_bundle.effective_action_budget,
        "hard_forbid_ratio": float(selection_bundle.hard_forbid_mask.float().mean().item()),
        "total_selected_actions": int(total_selected),
        "max_modification_count": modification_budget,
        "qualified_cover": bool(
            clean_params["offsets"].shape[0] >= 192
            and selection_bundle.effective_action_budget >= int(math.ceil(2.5 * float(raw_bits.numel())))
        ),
        "selection_probs": [float(value) for value in selection_probs.detach().cpu().tolist()],
        "hard_forbid_mask": [bool(value) for value in selection_bundle.hard_forbid_mask.detach().cpu().tolist()],
    }
    return assignments, meta


def build_greedy_nokey_cost_code_assignments(
    *,
    raw_bits: Tensor,
    clean_params: dict[str, Tensor],
    importance: Tensor,
    sensitivity: dict[str, Tensor],
    cost_bundle: ChannelCostBundle,
    key: str,
    candidate_count: int = 24,
    fallback_candidate_count: int = 16,
    beam_width: int = 64,
    selection_temperature: float = 0.5,
    max_modification_ratio: float = 0.30,
    min_selected_count: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    del key
    raw_bits = raw_bits.to(torch.int64).flatten()
    if raw_bits.numel() % 4 != 0:
        pad = (-raw_bits.numel()) % 4
        raw_bits = torch.cat((raw_bits, torch.zeros(pad, dtype=torch.int64)))
    selection_bundle = build_unkeyed_selection_field(
        clean_params=clean_params,
        importance=importance,
        sensitivity=sensitivity,
        cost_bundle=cost_bundle,
        selection_temperature=selection_temperature,
    )
    allowed_indices = torch.where(~selection_bundle.hard_forbid_mask)[0]
    if allowed_indices.numel() == 0:
        raise RuntimeError("greedy_nokey_cost_code_v1 did not find any modifiable Gaussian carriers")
    modification_budget = max(1, int(math.floor(float(max_modification_ratio) * float(clean_params["offsets"].shape[0]))))
    assignments: list[dict[str, object]] = []
    used_gaussians: set[int] = set()
    total_selected = 0
    for block_index, block_bits in enumerate(raw_bits.view(-1, 4)):
        available = torch.tensor(
            [int(index) for index in allowed_indices.tolist() if int(index) not in used_gaussians],
            dtype=torch.long,
        )
        if available.numel() < 8:
            raise RuntimeError("greedy_nokey_cost_code_v1 ran out of available Gaussian carriers")
        desired = candidate_count if available.numel() >= candidate_count else min(int(available.numel()), fallback_candidate_count)
        if desired <= 0:
            raise RuntimeError("greedy_nokey_cost_code_v1 could not allocate a non-empty candidate set")
        priority = selection_bundle.selection_logits[available]
        # `available` is intentionally tracked on CPU for deterministic bookkeeping.
        # On MPS, argsort returns device-local indices, so move them back before indexing.
        ranked = available[torch.argsort(priority, descending=True).detach().cpu()]
        selected_mask: Tensor | None = None
        candidates: list[dict[str, object]] = []
        candidate_costs: list[float] = []
        parity_matrix: Tensor | None = None
        base_seed = key_to_seed(f"greedy_nokey_cost_code_v1::block::{block_index}") + 421
        for retry_index in range(4):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(base_seed + 997 * retry_index)
            retry_desired = desired if retry_index == 0 else min(int(ranked.numel()), max(8, fallback_candidate_count + 4 * retry_index))
            candidate_indices = ranked[:retry_desired]
            candidates = []
            candidate_costs = []
            for gaussian_index in candidate_indices.tolist():
                channel, direction, cost = _lowest_cost_action_option(int(gaussian_index), cost_bundle=cost_bundle)
                candidates.append(
                    {
                        "gaussian_index": int(gaussian_index),
                        "channel": channel,
                        "direction": int(direction),
                        "selection_probability": float(selection_bundle.selection_probs[gaussian_index].item()),
                    }
                )
                candidate_costs.append(float(cost))
            parity_matrix = _make_parity_matrix(len(candidates), generator=generator)
            try:
                selected_mask = _solve_cost_code_block(
                    candidate_costs=torch.tensor(candidate_costs, dtype=torch.float32),
                    parity_matrix=parity_matrix,
                    target_bits=block_bits,
                    beam_width=beam_width,
                    min_selected_count=min_selected_count,
                )
                break
            except RuntimeError:
                selected_mask = None
                continue
        if selected_mask is None or parity_matrix is None:
            raise RuntimeError("greedy_nokey_cost_code_v1 could not satisfy the target syndrome after retrying candidate pools")
        selected_indices = torch.where(selected_mask > 0.5)[0]
        total_selected += int(selected_indices.numel())
        if total_selected > modification_budget:
            raise RuntimeError("greedy_nokey_cost_code_v1 exceeded the maximum modification ratio")
        for candidate in candidates:
            used_gaussians.add(int(candidate["gaussian_index"]))
        assignments.append(
            {
                "block_index": int(block_index),
                "message_bits": [int(value) for value in block_bits.tolist()],
                "candidate_count": len(candidates),
                "candidate_actions": candidates,
                "candidate_costs": candidate_costs,
                "parity_matrix": parity_matrix.t().tolist(),
                "selected_mask": [float(value) for value in selected_mask.tolist()],
                "selected_count": int(selected_indices.numel()),
            }
        )
    meta = {
        "selection_entropy": selection_bundle.selection_entropy,
        "effective_action_budget": selection_bundle.effective_action_budget,
        "hard_forbid_ratio": float(selection_bundle.hard_forbid_mask.float().mean().item()),
        "total_selected_actions": int(total_selected),
        "max_modification_count": modification_budget,
        "qualified_cover": bool(
            clean_params["offsets"].shape[0] >= 192
            and selection_bundle.effective_action_budget >= int(math.ceil(2.5 * float(raw_bits.numel())))
        ),
        "selection_probs": [float(value) for value in selection_bundle.selection_probs.detach().cpu().tolist()],
        "hard_forbid_mask": [bool(value) for value in selection_bundle.hard_forbid_mask.detach().cpu().tolist()],
    }
    return assignments, meta


def build_random_polarity_cost_code_assignments(
    *,
    raw_bits: Tensor,
    clean_params: dict[str, Tensor],
    importance: Tensor,
    sensitivity: dict[str, Tensor],
    cost_bundle: ChannelCostBundle,
    key: str,
    candidate_count: int = 24,
    fallback_candidate_count: int = 16,
    beam_width: int = 64,
    selection_temperature: float = 0.5,
    max_modification_ratio: float = 0.30,
    min_selected_count: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    del key
    raw_bits = raw_bits.to(torch.int64).flatten()
    if raw_bits.numel() % 4 != 0:
        pad = (-raw_bits.numel()) % 4
        raw_bits = torch.cat((raw_bits, torch.zeros(pad, dtype=torch.int64)))
    selection_bundle = build_unkeyed_selection_field(
        clean_params=clean_params,
        importance=importance,
        sensitivity=sensitivity,
        cost_bundle=cost_bundle,
        selection_temperature=selection_temperature,
    )
    allowed_indices = torch.where(~selection_bundle.hard_forbid_mask)[0]
    if allowed_indices.numel() == 0:
        raise RuntimeError("random_polarity_cost_code_v1 did not find any modifiable Gaussian carriers")
    modification_budget = max(1, int(math.floor(float(max_modification_ratio) * float(clean_params["offsets"].shape[0]))))
    assignments: list[dict[str, object]] = []
    used_gaussians: set[int] = set()
    total_selected = 0
    for block_index, block_bits in enumerate(raw_bits.view(-1, 4)):
        available = torch.tensor(
            [int(index) for index in allowed_indices.tolist() if int(index) not in used_gaussians],
            dtype=torch.long,
        )
        if available.numel() < 8:
            raise RuntimeError("random_polarity_cost_code_v1 ran out of available Gaussian carriers")
        desired = candidate_count if available.numel() >= candidate_count else min(int(available.numel()), fallback_candidate_count)
        if desired <= 0:
            raise RuntimeError("random_polarity_cost_code_v1 could not allocate a non-empty candidate set")
        priority = selection_bundle.selection_logits[available]
        # Mirror the greedy no-key path: keep the bookkeeping tensor on CPU even when
        # the selection field lives on an accelerator.
        ranked = available[torch.argsort(priority, descending=True).detach().cpu()]
        selected_mask: Tensor | None = None
        candidates: list[dict[str, object]] = []
        candidate_costs: list[float] = []
        parity_matrix: Tensor | None = None
        base_seed = key_to_seed(f"random_polarity_cost_code_v1::block::{block_index}") + 607
        for retry_index in range(4):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(base_seed + 997 * retry_index)
            retry_desired = desired if retry_index == 0 else min(int(ranked.numel()), max(8, fallback_candidate_count + 4 * retry_index))
            candidate_indices = ranked[:retry_desired]
            candidates = []
            candidate_costs = []
            for gaussian_index in candidate_indices.tolist():
                costs = torch.tensor(
                    [
                        float(cost_bundle.log_rho_plus[gaussian_index].item()),
                        float(cost_bundle.log_rho_minus[gaussian_index].item()),
                        float(cost_bundle.alpha_rho_plus[gaussian_index].item()),
                        float(cost_bundle.alpha_rho_minus[gaussian_index].item()),
                    ],
                    dtype=torch.float32,
                )
                choice = int(torch.randint(0, 4, size=(1,), generator=generator).item())
                channel = "log" if choice < 2 else "alpha"
                direction = 1 if choice % 2 == 0 else -1
                candidates.append(
                    {
                        "gaussian_index": int(gaussian_index),
                        "channel": channel,
                        "direction": int(direction),
                        "selection_probability": float(selection_bundle.selection_probs[gaussian_index].item()),
                    }
                )
                candidate_costs.append(float(costs[choice].item()))
            parity_matrix = _make_parity_matrix(len(candidates), generator=generator)
            try:
                selected_mask = _solve_cost_code_block(
                    candidate_costs=torch.tensor(candidate_costs, dtype=torch.float32),
                    parity_matrix=parity_matrix,
                    target_bits=block_bits,
                    beam_width=beam_width,
                    min_selected_count=min_selected_count,
                )
                break
            except RuntimeError:
                selected_mask = None
                continue
        if selected_mask is None or parity_matrix is None:
            raise RuntimeError("random_polarity_cost_code_v1 could not satisfy the target syndrome after retrying candidate pools")
        selected_indices = torch.where(selected_mask > 0.5)[0]
        total_selected += int(selected_indices.numel())
        if total_selected > modification_budget:
            raise RuntimeError("random_polarity_cost_code_v1 exceeded the maximum modification ratio")
        for candidate in candidates:
            used_gaussians.add(int(candidate["gaussian_index"]))
        assignments.append(
            {
                "block_index": int(block_index),
                "message_bits": [int(value) for value in block_bits.tolist()],
                "candidate_count": len(candidates),
                "candidate_actions": candidates,
                "candidate_costs": candidate_costs,
                "parity_matrix": parity_matrix.t().tolist(),
                "selected_mask": [float(value) for value in selected_mask.tolist()],
                "selected_count": int(selected_indices.numel()),
            }
        )
    meta = {
        "selection_entropy": selection_bundle.selection_entropy,
        "effective_action_budget": selection_bundle.effective_action_budget,
        "hard_forbid_ratio": float(selection_bundle.hard_forbid_mask.float().mean().item()),
        "total_selected_actions": int(total_selected),
        "max_modification_count": modification_budget,
        "qualified_cover": bool(
            clean_params["offsets"].shape[0] >= 192
            and selection_bundle.effective_action_budget >= int(math.ceil(2.5 * float(raw_bits.numel())))
        ),
        "selection_probs": [float(value) for value in selection_bundle.selection_probs.detach().cpu().tolist()],
        "hard_forbid_mask": [bool(value) for value in selection_bundle.hard_forbid_mask.detach().cpu().tolist()],
    }
    return assignments, meta


def _cost_code_action_scores(
    params: dict[str, Tensor],
    *,
    clean_params: dict[str, Tensor],
    assignments: list[dict[str, object]],
    log_delta: float,
    alpha_delta: float,
) -> list[Tensor]:
    canonical = _canonicalize_geometry(params["scales"], params["thetas"])
    clean_canonical = _canonicalize_geometry(clean_params["scales"], clean_params["thetas"])
    current_log = canonical["log_anisotropy"]
    clean_log = clean_canonical["log_anisotropy"].to(current_log.device)
    current_alpha = params["opacities"][:, 0]
    clean_alpha = clean_params["opacities"][:, 0].to(current_alpha.device)
    outputs: list[Tensor] = []
    for assignment in assignments:
        action_scores: list[Tensor] = []
        for action in assignment["candidate_actions"]:
            gaussian_index = int(action["gaussian_index"])
            if str(action["channel"]) == "log":
                score = (current_log[gaussian_index] - clean_log[gaussian_index]) / max(log_delta, 1e-6)
            else:
                score = (current_alpha[gaussian_index] - clean_alpha[gaussian_index]) / max(alpha_delta, 1e-6)
            action_scores.append(float(action["direction"]) * score)
        outputs.append(torch.stack(action_scores))
    return outputs


def cost_code_action_target_loss(
    params: dict[str, Tensor],
    *,
    clean_params: dict[str, Tensor],
    assignments: list[dict[str, object]],
    log_delta: float,
    alpha_delta: float,
    beta: float = 4.0,
    decision_threshold: float = 0.5,
) -> Tensor:
    action_scores = _cost_code_action_scores(
        params,
        clean_params=clean_params,
        assignments=assignments,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
    )
    losses: list[Tensor] = []
    for scores, assignment in zip(action_scores, assignments):
        selected = torch.tensor(assignment["selected_mask"], dtype=torch.float32, device=scores.device)
        losses.append(F.binary_cross_entropy_with_logits(beta * (scores - float(decision_threshold)), selected))
    if not losses:
        return torch.tensor(0.0, dtype=clean_params["offsets"].dtype, device=clean_params["offsets"].device)
    return torch.stack(losses).mean()


def cost_code_action_margin_loss(
    params: dict[str, Tensor],
    *,
    clean_params: dict[str, Tensor],
    assignments: list[dict[str, object]],
    log_delta: float,
    alpha_delta: float,
    decision_threshold: float = 0.5,
    margin: float = 0.25,
) -> Tensor:
    action_scores = _cost_code_action_scores(
        params,
        clean_params=clean_params,
        assignments=assignments,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
    )
    losses: list[Tensor] = []
    for scores, assignment in zip(action_scores, assignments):
        selected = torch.tensor(assignment["selected_mask"], dtype=torch.float32, device=scores.device)
        signed_scores = (2.0 * selected - 1.0) * (scores - float(decision_threshold))
        losses.append(torch.relu(float(margin) - signed_scores).mean())
    if not losses:
        return torch.tensor(0.0, dtype=clean_params["offsets"].dtype, device=clean_params["offsets"].device)
    return torch.stack(losses).mean()


def cost_code_block_probabilities(
    params: dict[str, Tensor],
    *,
    clean_params: dict[str, Tensor],
    assignments: list[dict[str, object]],
    log_delta: float,
    alpha_delta: float,
    beta: float = 4.0,
) -> tuple[Tensor, list[Tensor]]:
    action_scores = _cost_code_action_scores(
        params,
        clean_params=clean_params,
        assignments=assignments,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
    )
    block_probabilities: list[Tensor] = []
    for scores, assignment in zip(action_scores, assignments):
        action_probs = torch.sigmoid(beta * (scores - 0.5))
        parity_matrix = torch.tensor(assignment["parity_matrix"], dtype=torch.float32, device=scores.device).t()
        bit_probs: list[Tensor] = []
        for row in range(parity_matrix.shape[0]):
            row_mask = parity_matrix[row] > 0.5
            if int(row_mask.sum().item()) == 0:
                bit_probs.append(torch.tensor(0.5, dtype=scores.dtype, device=scores.device))
                continue
            selected_probs = action_probs[row_mask]
            parity_prob = 0.5 - 0.5 * torch.prod(1.0 - 2.0 * selected_probs)
            bit_probs.append(parity_prob)
        block_probabilities.append(torch.stack(bit_probs))
    if not block_probabilities:
        return torch.empty((0, 4), dtype=torch.float32, device=clean_params["offsets"].device), action_scores
    return torch.stack(block_probabilities, dim=0), action_scores


def decode_cost_code(
    params: dict[str, Tensor],
    *,
    clean_params: dict[str, Tensor],
    assignments: list[dict[str, object]],
    raw_length: int,
    log_delta: float,
    alpha_delta: float,
    beta: float = 4.0,
) -> tuple[Tensor, list[list[float]], list[list[float]]]:
    block_probs, action_scores = cost_code_block_probabilities(
        params,
        clean_params=clean_params,
        assignments=assignments,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
        beta=beta,
    )
    if block_probs.numel() == 0:
        return torch.empty(0, dtype=torch.int64), [], []
    hard_blocks: list[Tensor] = []
    for assignment, scores in zip(assignments, action_scores):
        if scores.numel() == 0:
            hard_blocks.append(torch.zeros(4, dtype=torch.int64, device=block_probs.device))
            continue
        parity_matrix = torch.tensor(assignment["parity_matrix"], dtype=torch.float32, device=scores.device).t()
        hard_actions = (scores > 0.5).to(torch.float32)
        hard_bits = torch.remainder((parity_matrix @ hard_actions).round().to(torch.int64), 2)
        hard_blocks.append(hard_bits)
    decoded = torch.cat(hard_blocks, dim=0)[:raw_length]
    return (
        decoded,
        [[float(value) for value in row.tolist()] for row in block_probs.detach().cpu()],
        [[float(value) for value in row.detach().cpu().tolist()] for row in action_scores],
    )


def wrong_key_contrastive_loss(
    params: dict[str, Tensor],
    *,
    clean_params: dict[str, Tensor],
    raw_bits: Tensor,
    correct_assignments: list[dict[str, object]],
    wrong_assignments: list[list[dict[str, object]]],
    log_delta: float,
    alpha_delta: float,
    beta: float = 4.0,
    margin: float = 0.20,
) -> tuple[Tensor, Tensor]:
    # Contrastive wrong-key loss: pushes the embedding toward maximising BER
    # under wrong-key decoders while maintaining low BER under the correct key.
    #
    # Key-specificity is a required property of a steganography system
    # (analogous to semantic security in cryptographic watermarking).  The
    # margin-based contrastive formulation here is inspired by:
    #   Fingerprinting literature — different keys produce uncorrelated codewords
    #       so that the adversary cannot recover the payload without the correct key.
    #   Contrastive representation learning [Chen et al., ICML 2020 SimCLR] —
    #       the correct/wrong-key pair plays the role of positive/negative pairs.
    # The statistical test for key-specificity (BER sweep over N wrong keys) is
    # in evaluate_watermark.py :: compute_wrong_key_ber_sweep.
    correct_probs, _ = cost_code_block_probabilities(
        params,
        clean_params=clean_params,
        assignments=correct_assignments,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
        beta=beta,
    )
    correct_targets = raw_bits.to(correct_probs.device).to(torch.float32).view(-1, 4)
    correct_loss = F.binary_cross_entropy(correct_probs, correct_targets)
    correct_conf = torch.abs(correct_probs - 0.5).mean() * 2.0
    wrong_losses: list[Tensor] = []
    wrong_confidences: list[Tensor] = []
    for assignment_group in wrong_assignments:
        wrong_probs, _ = cost_code_block_probabilities(
            params,
            clean_params=clean_params,
            assignments=assignment_group,
            log_delta=log_delta,
            alpha_delta=alpha_delta,
            beta=beta,
        )
        wrong_losses.append(torch.mean((wrong_probs - 0.5) ** 2))
        wrong_confidences.append(torch.abs(wrong_probs - 0.5).mean() * 2.0)
    if wrong_losses:
        wrong_mean = torch.stack(wrong_losses).mean()
        wrong_conf = torch.stack(wrong_confidences).mean()
        margin_loss = torch.relu(torch.tensor(float(margin), dtype=wrong_conf.dtype, device=wrong_conf.device) - (correct_conf - wrong_conf))
        return correct_loss + wrong_mean, margin_loss
    return correct_loss, torch.tensor(0.0, dtype=correct_loss.dtype, device=correct_loss.device)


def select_compensation_neighbors(
    offsets: Tensor,
    *,
    assignments: list[dict[str, object]],
    wet_mask: Tensor,
    per_active: int = 2,
) -> Tensor:
    active = sorted(
        {
            int(index)
            for assignment in assignments
            for index in [*assignment["log_indices"], *assignment["alpha_indices"]]
        }
    )
    if not active:
        return torch.empty(0, dtype=torch.long)
    dry_indices = torch.where(~wet_mask)[0]
    active_tensor = torch.tensor(active, dtype=torch.long)
    dry_set = {int(index) for index in dry_indices.tolist()} - set(active)
    if not dry_set:
        return torch.empty(0, dtype=torch.long)
    dry_tensor = torch.tensor(sorted(dry_set), dtype=torch.long)
    distances = torch.cdist(offsets[active_tensor].detach().cpu(), offsets[dry_tensor].detach().cpu())
    neighbors: set[int] = set()
    for row in range(distances.shape[0]):
        nearest = torch.topk(distances[row], k=min(per_active, distances.shape[1]), largest=False).indices.tolist()
        for neighbor_idx in nearest:
            neighbors.add(int(dry_tensor[neighbor_idx].item()))
    return torch.tensor(sorted(neighbors), dtype=torch.long)


def carrier_ewc_penalty(
    renderer,
    clean_renderer,
    *,
    assignments: list[dict[str, object]],
    sensitivity: dict[str, Tensor],
    alpha_weight: float,
) -> Tensor:
    log_indices, alpha_indices = active_channel_indices(assignments)
    penalty_terms: list[Tensor] = []
    if log_indices.numel() > 0:
        fisher = sensitivity["log_anisotropy"][log_indices].to(renderer.raw_scales.device).clamp_min(1e-6)
        current = renderer.raw_scales[log_indices]
        target = clean_renderer.raw_scales[log_indices].detach().to(current.device)
        penalty_terms.append((fisher.unsqueeze(-1) * (current - target).pow(2)).mean())
    if alpha_indices.numel() > 0:
        fisher = (alpha_weight * sensitivity["alpha"][alpha_indices]).to(renderer.raw_opacities.device).clamp_min(1e-6)
        current = renderer.raw_opacities[alpha_indices]
        target = clean_renderer.raw_opacities[alpha_indices].detach().to(current.device)
        penalty_terms.append((fisher.unsqueeze(-1) * (current - target).pow(2)).mean())
    if not penalty_terms:
        return torch.tensor(0.0, dtype=renderer.raw_offsets.dtype, device=renderer.raw_offsets.device)
    return torch.stack(penalty_terms).mean()


def train_detector_steps(
    detector: RunLevelStegaDetector,
    optimizer: torch.optim.Optimizer,
    *,
    clean_features: Tensor,
    wm_features: Tensor,
    steps: int = 3,
) -> None:
    clean = clean_features.detach()
    watermarked = wm_features.detach()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.stack((clean, watermarked), dim=0)
        labels = torch.tensor([0.0, 1.0], dtype=torch.float32, device=inputs.device)
        logits = detector(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()


def detector_embed_loss(detector: RunLevelStegaDetector, wm_features: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(detector(wm_features), torch.zeros(1, dtype=wm_features.dtype, device=wm_features.device))


def detector_feature_names() -> list[str]:
    base_fields = ("mu_x", "mu_y", "log_area", "log_anisotropy", "alpha", "importance")
    names: list[str] = []
    for field in base_fields:
        names.extend(
            [
                f"{field}_mean",
                f"{field}_std",
                f"{field}_q25",
                f"{field}_q50",
                f"{field}_q75",
            ]
        )
    names.extend(["theta_resultant_length", "alpha_log_area_corr", "off_canvas_fraction"])
    cover_fields = ("mu_x", "mu_y", "log_area", "log_anisotropy", "theta_major", "alpha")
    for left_index, left_name in enumerate(cover_fields):
        for right_name in cover_fields[left_index + 1 :]:
            names.append(f"corr_{left_name}_{right_name}")
    for name in cover_fields:
        names.append(f"tail_{name}_gt1p5")
        names.append(f"tail_{name}_gt2p5")
    return names


def build_detector_feature_vector(params: dict[str, Tensor], *, importance: Tensor | None = None) -> Tensor:
    if importance is None:
        importance = compute_gaussian_importance(params["scales"], params["opacities"])
    canonical = _canonicalize_geometry(params["scales"], params["thetas"])
    theta_major = canonical["theta_major"][:, 0]
    fields = {
        "mu_x": params["offsets"][:, 0],
        "mu_y": params["offsets"][:, 1],
        "log_area": canonical["log_area"],
        "log_anisotropy": canonical["log_anisotropy"],
        "alpha": params["opacities"][:, 0],
        "importance": importance,
    }
    feature_values: list[Tensor] = []
    for values in fields.values():
        feature_values.extend(
            [
                values.mean(),
                values.std(unbiased=False),
                torch.quantile(values, 0.25),
                torch.quantile(values, 0.50),
                torch.quantile(values, 0.75),
            ]
        )
    phase = 2.0 * theta_major
    theta_resultant = torch.sqrt(torch.cos(phase).mean() ** 2 + torch.sin(phase).mean() ** 2)
    alpha_log_area_corr = _safe_corr(params["opacities"][:, 0], canonical["log_area"])
    off_canvas_fraction = torch.sigmoid((params["offsets"].abs().amax(dim=1) - 1.0) * 40.0).mean()
    feature_values.extend([theta_resultant, alpha_log_area_corr, off_canvas_fraction])

    cover_matrix = cover_feature_matrix(params)
    for left_index in range(cover_matrix.shape[1]):
        for right_index in range(left_index + 1, cover_matrix.shape[1]):
            feature_values.append(_safe_corr(cover_matrix[:, left_index], cover_matrix[:, right_index]))
    for feature_index in range(cover_matrix.shape[1]):
        z_scores = robust_z_scores(cover_matrix[:, feature_index])
        feature_values.append(torch.sigmoid((z_scores.abs() - 1.5) * 6.0).mean())
        feature_values.append(torch.sigmoid((z_scores.abs() - 2.5) * 6.0).mean())
    return torch.stack(feature_values)


def _leave_one_out_probe_scores(
    clean_features: Tensor,
    watermarked_features: Tensor,
    *,
    seed: int = 0,
) -> tuple[Tensor | None, Tensor | None]:
    clean_x = clean_features.detach().cpu().to(torch.float32)
    watermarked_x = watermarked_features.detach().cpu().to(torch.float32)
    if clean_x.ndim == 1:
        clean_x = clean_x.unsqueeze(0)
    if watermarked_x.ndim == 1:
        watermarked_x = watermarked_x.unsqueeze(0)
    if clean_x.shape[0] < 2 or watermarked_x.shape[0] < 2:
        return None, None

    x = torch.cat((clean_x, watermarked_x), dim=0)
    y = torch.cat((torch.zeros(clean_x.shape[0]), torch.ones(watermarked_x.shape[0])), dim=0)
    scores = torch.zeros(x.shape[0], dtype=torch.float32)
    order = torch.randperm(x.shape[0], generator=torch.Generator(device="cpu").manual_seed(seed + 313)).tolist()
    for holdout_index in order:
        train_mask = torch.ones(x.shape[0], dtype=torch.bool)
        train_mask[holdout_index] = False
        x_train = x[train_mask]
        y_train = y[train_mask]
        if int((y_train == 0).sum().item()) == 0 or int((y_train == 1).sum().item()) == 0:
            return None, None
        probe = _fit_linear_probe(x_train, y_train)
        scores[holdout_index] = _predict_linear_probe(x[holdout_index : holdout_index + 1], probe)[0]
    return scores, y


def _paired_leave_one_pair_out_probe_scores(
    clean_features: Tensor,
    watermarked_features: Tensor,
    *,
    seed: int = 0,
) -> tuple[Tensor | None, Tensor | None]:
    clean_x = clean_features.detach().cpu().to(torch.float32)
    watermarked_x = watermarked_features.detach().cpu().to(torch.float32)
    if clean_x.ndim == 1:
        clean_x = clean_x.unsqueeze(0)
    if watermarked_x.ndim == 1:
        watermarked_x = watermarked_x.unsqueeze(0)
    if clean_x.shape[0] < 2 or watermarked_x.shape[0] < 2:
        return None, None

    pair_count = clean_x.shape[0]
    score_rows: list[Tensor] = []
    label_rows: list[Tensor] = []
    order = torch.randperm(pair_count, generator=torch.Generator(device="cpu").manual_seed(seed + 419)).tolist()
    for holdout_index in order:
        train_mask = torch.ones(pair_count, dtype=torch.bool)
        train_mask[holdout_index] = False
        if int(train_mask.sum().item()) < 1:
            return None, None
        x_train = torch.cat((clean_x[train_mask], watermarked_x[train_mask]), dim=0)
        y_train = torch.cat(
            (
                torch.zeros(int(train_mask.sum().item()), dtype=torch.float32),
                torch.ones(int(train_mask.sum().item()), dtype=torch.float32),
            ),
            dim=0,
        )
        if int((y_train == 0).sum().item()) == 0 or int((y_train == 1).sum().item()) == 0:
            return None, None
        x_test = torch.cat((clean_x[holdout_index : holdout_index + 1], watermarked_x[holdout_index : holdout_index + 1]), dim=0)
        y_test = torch.tensor([0.0, 1.0], dtype=torch.float32)
        probe = _fit_linear_probe(x_train, y_train)
        score_rows.append(_predict_linear_probe(x_test, probe))
        label_rows.append(y_test)
    return torch.cat(score_rows, dim=0), torch.cat(label_rows, dim=0)


def pooled_detection_statistics(clean_features: Tensor, watermarked_features: Tensor, *, seed: int = 0) -> dict[str, float | None]:
    scores, labels = _leave_one_out_probe_scores(clean_features, watermarked_features, seed=seed)
    if scores is None or labels is None:
        return {"auc": None, "false_alarm_rate": None, "true_positive_rate": None}

    false_alarm = float((scores[labels == 0] > 0.0).float().mean().item())
    true_positive = float((scores[labels == 1] > 0.0).float().mean().item())
    return {
        "auc": _binary_auc(scores, labels.to(torch.int64)),
        "false_alarm_rate": false_alarm,
        "true_positive_rate": true_positive,
    }


def _fit_linear_probe(x_train: Tensor, y_train: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    train_mean = x_train.mean(dim=0, keepdim=True)
    train_std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    x_train_norm = (x_train - train_mean) / train_std
    weight = torch.zeros(x_train_norm.shape[1], 1, dtype=torch.float32)
    bias = torch.zeros(1, dtype=torch.float32)
    for _ in range(300):
        logits = x_train_norm @ weight + bias
        probs = torch.sigmoid(logits[:, 0])
        error = probs - y_train
        grad_w = (x_train_norm.t() @ error.unsqueeze(-1)) / x_train_norm.shape[0] + 1e-3 * weight
        grad_b = error.mean()
        weight = weight - 0.1 * grad_w
        bias = bias - 0.1 * grad_b
    return weight, bias, train_mean, train_std


def _predict_linear_probe(x: Tensor, probe: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
    weight, bias, mean_value, std_value = probe
    x_norm = (x - mean_value) / std_value
    return (x_norm @ weight + bias)[:, 0]


def _balanced_accuracy(scores: Tensor, labels: Tensor) -> float:
    predictions = (scores > 0.0).to(torch.int64)
    labels = labels.to(torch.int64)
    pos_mask = labels == 1
    neg_mask = labels == 0
    tpr = (predictions[pos_mask] == 1).float().mean() if bool(pos_mask.any()) else torch.tensor(0.5)
    tnr = (predictions[neg_mask] == 0).float().mean() if bool(neg_mask.any()) else torch.tensor(0.5)
    return float((0.5 * (tpr + tnr)).item())


def _bootstrap_auc_ci(scores: Tensor, labels: Tensor, *, seed: int = 0, rounds: int = 200) -> tuple[float, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 551)
    aucs: list[float] = []
    for _ in range(rounds):
        sample_indices = torch.randint(0, scores.numel(), size=(scores.numel(),), generator=generator)
        sampled_scores = scores[sample_indices]
        sampled_labels = labels[sample_indices]
        aucs.append(_binary_auc(sampled_scores, sampled_labels.to(torch.int64)))
    auc_tensor = torch.tensor(aucs, dtype=torch.float32)
    return float(torch.quantile(auc_tensor, 0.025).item()), float(torch.quantile(auc_tensor, 0.975).item())


def _group_bootstrap_auc_summary(
    scores: Tensor,
    labels: Tensor,
    *,
    unit_ids: list[str],
    seed: int = 0,
    rounds: int = 1000,
) -> tuple[float | None, list[float] | None, int]:
    unique_units = sorted(set(unit_ids))
    if not unique_units:
        return None, None, 0
    unit_to_indices: dict[str, Tensor] = {}
    for unit in unique_units:
        unit_to_indices[unit] = torch.tensor([index for index, value in enumerate(unit_ids) if value == unit], dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 853)
    aucs: list[float] = []
    for _ in range(max(1, int(rounds))):
        sampled_units = torch.randint(0, len(unique_units), size=(len(unique_units),), generator=generator)
        sampled_indices = torch.cat([unit_to_indices[unique_units[int(index.item())]] for index in sampled_units], dim=0)
        sampled_scores = scores[sampled_indices]
        sampled_labels = labels[sampled_indices]
        if int((sampled_labels == 0).sum().item()) == 0 or int((sampled_labels == 1).sum().item()) == 0:
            continue
        aucs.append(_binary_auc(sampled_scores, sampled_labels.to(torch.int64)))
    if not aucs:
        return None, None, len(unique_units)
    auc_tensor = torch.tensor(aucs, dtype=torch.float32)
    return (
        float(auc_tensor.mean().item()),
        [
            float(torch.quantile(auc_tensor, 0.025).item()),
            float(torch.quantile(auc_tensor, 0.975).item()),
        ],
        len(unique_units),
    )


def assign_actor_clusters(feature_matrix: Tensor, *, num_clusters: int = 4, seed: int = 0) -> Tensor:
    x = feature_matrix.detach().cpu().to(torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[0] <= num_clusters:
        return torch.arange(x.shape[0], dtype=torch.int64) % max(1, int(num_clusters))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 173)
    center_indices = torch.randperm(x.shape[0], generator=generator)[:num_clusters]
    centers = x[center_indices].clone()
    assignments = torch.zeros(x.shape[0], dtype=torch.int64)
    for _ in range(20):
        distances = torch.cdist(x, centers)
        assignments = torch.argmin(distances, dim=1)
        for cluster_index in range(num_clusters):
            mask = assignments == cluster_index
            if bool(mask.any()):
                centers[cluster_index] = x[mask].mean(dim=0)
    return assignments


def _empty_grouped_detection_statistics(
    *,
    status: str,
    bootstrap_unit_count: int = 0,
    eligible_run_count: int = 0,
) -> dict[str, float | list[float] | None | str]:
    return {
        "status": status,
        "auc": None,
        "auc_mean": None,
        "false_alarm_rate": None,
        "true_positive_rate": None,
        "balanced_accuracy": None,
        "confidence_interval": None,
        "bootstrap_unit_count": bootstrap_unit_count,
        "eligible_run_count": eligible_run_count,
    }


def grouped_pooled_detection_statistics(
    clean_features: Tensor,
    watermarked_features: Tensor,
    *,
    group_ids: list[str],
    split_mode: str,
    actor_ids: list[int] | None = None,
    seed: int = 0,
    folds: int = 5,
    bootstrap_rounds: int = 1000,
) -> dict[str, float | list[float] | None]:
    clean_x = clean_features.detach().cpu().to(torch.float32)
    watermarked_x = watermarked_features.detach().cpu().to(torch.float32)
    if clean_x.ndim == 1:
        clean_x = clean_x.unsqueeze(0)
    if watermarked_x.ndim == 1:
        watermarked_x = watermarked_x.unsqueeze(0)
    if clean_x.shape != watermarked_x.shape or clean_x.shape[0] != len(group_ids):
        raise ValueError("clean/watermarked features and group_ids must align")
    scores: list[Tensor] = []
    labels: list[Tensor] = []
    bootstrap_units: list[str] = []
    eligible_run_count = 0
    if split_mode == "same_source":
        unique_groups = sorted(set(group_ids))
        eligible_group_count = 0
        for group in unique_groups:
            group_indices = torch.tensor([index for index, value in enumerate(group_ids) if value == group], dtype=torch.long)
            if group_indices.numel() < 2:
                continue
            eligible_group_count += 1
            group_clean = clean_x[group_indices]
            group_wm = watermarked_x[group_indices]
            group_scores, group_labels = _paired_leave_one_pair_out_probe_scores(group_clean, group_wm, seed=seed + len(scores))
            if group_scores is None or group_labels is None:
                continue
            scores.append(group_scores)
            labels.append(group_labels)
            bootstrap_units.extend([str(group)] * int(group_scores.numel()))
            eligible_run_count += int(group_indices.numel())
        if eligible_group_count == 0:
            return _empty_grouped_detection_statistics(status="insufficient_same_source_replicates")
    else:
        if split_mode == "cross_source":
            unique_groups = sorted(set(group_ids))
        elif split_mode == "actor_holdout":
            if actor_ids is None:
                raise ValueError("actor_ids are required for actor_holdout split")
            unique_groups = [str(value) for value in sorted(set(actor_ids))]
        else:
            raise ValueError(f"unsupported split_mode: {split_mode}")
        if len(unique_groups) < 2:
            return _empty_grouped_detection_statistics(status="insufficient_group_count")
        if split_mode == "cross_source" and len(unique_groups) > folds:
            unique_groups = [group for fold_index, group in enumerate(unique_groups) if fold_index < len(unique_groups)]
        for group in unique_groups:
            if split_mode == "cross_source":
                test_indices = torch.tensor([index for index, value in enumerate(group_ids) if value == group], dtype=torch.long)
            else:
                test_indices = torch.tensor([index for index, value in enumerate(actor_ids or []) if str(value) == group], dtype=torch.long)
            train_mask = torch.ones(clean_x.shape[0], dtype=torch.bool)
            train_mask[test_indices] = False
            if int(train_mask.sum().item()) < 2 or test_indices.numel() == 0:
                continue
            x_train = torch.cat((clean_x[train_mask], watermarked_x[train_mask]), dim=0)
            y_train = torch.cat(
                (
                    torch.zeros(int(train_mask.sum().item()), dtype=torch.float32),
                    torch.ones(int(train_mask.sum().item()), dtype=torch.float32),
                ),
                dim=0,
            )
            x_test = torch.cat((clean_x[test_indices], watermarked_x[test_indices]), dim=0)
            y_test = torch.cat(
                (
                    torch.zeros(test_indices.numel(), dtype=torch.float32),
                    torch.ones(test_indices.numel(), dtype=torch.float32),
                ),
                dim=0,
            )
            probe = _fit_linear_probe(x_train, y_train)
            fold_scores = _predict_linear_probe(x_test, probe)
            scores.append(fold_scores)
            labels.append(y_test)
            bootstrap_units.extend([str(group)] * int(fold_scores.numel()))
            eligible_run_count += int(test_indices.numel())
    if not scores:
        return _empty_grouped_detection_statistics(
            status="insufficient_grouped_holdouts",
            eligible_run_count=eligible_run_count,
        )
    all_scores = torch.cat(scores, dim=0)
    all_labels = torch.cat(labels, dim=0)
    auc_mean, ci, unit_count = _group_bootstrap_auc_summary(
        all_scores,
        all_labels,
        unit_ids=bootstrap_units,
        seed=seed,
        rounds=bootstrap_rounds,
    )
    return {
        "status": "computed",
        "auc": _binary_auc(all_scores, all_labels.to(torch.int64)),
        "auc_mean": auc_mean,
        "false_alarm_rate": float((all_scores[all_labels == 0] > 0.0).float().mean().item()),
        "true_positive_rate": float((all_scores[all_labels == 1] > 0.0).float().mean().item()),
        "balanced_accuracy": _balanced_accuracy(all_scores, all_labels),
        "confidence_interval": ci,
        "bootstrap_unit_count": unit_count,
        "eligible_run_count": eligible_run_count,
    }


def split_coded_bits_round_robin(coded_bits: Tensor, num_covers: int) -> list[Tensor]:
    output = [[] for _ in range(num_covers)]
    for index, value in enumerate(coded_bits.to(torch.int64).flatten().tolist()):
        output[index % num_covers].append(value)
    return [torch.tensor(bits, dtype=torch.int64) for bits in output]


def parity_shard_bits(bits: Tensor, shard_bits: int = 4) -> Tensor:
    bits = bits.to(torch.int64).flatten()
    output = []
    for shard_index in range(shard_bits):
        output.append(int(torch.remainder(bits[shard_index::shard_bits].sum(), 2).item()) if bits.numel() > 0 else 0)
    return torch.tensor(output, dtype=torch.int64)


def merge_round_robin_bits(chunks: list[Tensor], total_length: int) -> Tensor:
    merged = torch.zeros(total_length, dtype=torch.int64)
    positions = [0 for _ in chunks]
    for index in range(total_length):
        chunk_index = index % len(chunks)
        merged[index] = chunks[chunk_index][positions[chunk_index]]
        positions[chunk_index] += 1
    return merged


def multi_cover_round_trip(
    raw_bits: Tensor,
    *,
    num_covers: int,
) -> dict[str, object]:
    # Proof-of-concept for multi-cover spread-spectrum steganography.
    # The payload is Hamming-coded and sharded round-robin across num_covers
    # independent Gaussian Splatting scenes; recovery requires all covers.
    #
    # Conceptual lineage:
    #   Spread-spectrum watermarking [Cox et al., IEEE TITS 1997] — distributes
    #       the watermark energy across many carriers to improve robustness.
    #       https://doi.org/10.1109/83.650120
    #
    #   Secret sharing / threshold schemes — K-of-N recovery requires at least K
    #       covers to be present; our current scheme is an all-N variant.
    #
    # Note: this function is a round-trip verification utility and does NOT
    # perform actual Gaussian parameter modification.  Full multi-cover
    # embedding is a P2 research goal (see plan).
    coded_bits, pad_bits = hamming74_encode(raw_bits)
    chunks = split_coded_bits_round_robin(coded_bits, num_covers=num_covers)
    shards = [parity_shard_bits(chunk) for chunk in chunks]
    merged = merge_round_robin_bits(chunks, total_length=coded_bits.numel())
    decoded, corrected = hamming74_decode(merged, raw_length=raw_bits.numel())
    return {
        "pad_bits": pad_bits,
        "corrected_blocks": corrected,
        "coded_chunks": [chunk.tolist() for chunk in chunks],
        "parity_shards": [shard.tolist() for shard in shards],
        "decoded_bits": decoded.tolist(),
    }


def capacity_failure_report(*, payload_bits: int, coded_payload_bits: int) -> dict[str, object]:
    return {
        "capacity_failure": True,
        "payload_bits": payload_bits,
        "coded_payload_bits": coded_payload_bits,
        "modified_gaussian_count": 0,
        "wet_ratio": 1.0,
        "total_embed_cost": None,
    }


def smooth_saturation_penalty(reconstruction: Tensor) -> Tensor:
    saturated_fraction, saturation_max_excess = compute_saturation_statistics(reconstruction)
    return torch.tensor(
        saturated_fraction + saturation_max_excess,
        dtype=reconstruction.dtype,
        device=reconstruction.device,
    )
