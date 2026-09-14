"""Carrier costs, block assignment, parameter extraction and regularization."""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# ── Import everything from the unchanged v2.0 protocol ──────────────────────
from stego_protocol import (
    ChannelCostBundle,
    _canonicalize_geometry,
    _smooth_tail_penalty,
    _top_fraction_mask,
    _channel_boundary_penalty,
    _minimum_carrier_scale,
    build_density_reference,
    build_hard_forbid_mask,
    build_soft_selection_field,
    build_unkeyed_selection_field,
    cover_feature_matrix,
    robust_z_scores,
    parity_block_logits,
    decode_cost_code,
    cost_code_block_probabilities,
    _gumbel_topk,
    _make_parity_matrix,
    _solve_cost_code_block,
)
from watermark_utils import key_to_seed

# ── Protocol identifier ──────────────────────────────────────────────────────
COST_STEGO_V3_PROTOCOL = "cost_stego_v3"
COST_STEGO_V3_1_PROTOCOL = "cost_stego_v3_1"
V3_CHANNELS = ("log_anisotropy", "alpha", "theta", "color_lum")


def _keyed_whitening_bits(key: str, block_index: int, *, width: int = 4) -> Tensor:
    """Return the deterministic affine parity offset for one keyed block."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(key_to_seed(f"{key}::v3::whiten::{int(block_index)}") + 1709)
    return torch.randint(0, 2, (int(width),), generator=generator, dtype=torch.int64)


def _assignment_whitening_bits(
    assignment: Dict[str, object],
    *,
    device: torch.device | None = None,
) -> Tensor:
    values = assignment.get("whitening_bits", [0, 0, 0, 0])
    bits = torch.tensor(values, dtype=torch.int64, device=device).flatten()
    if bits.numel() != 4:
        raise ValueError("v3 cost-code whitening mask must contain four bits")
    return bits


def _assignment_encoded_message_bits(assignment: Dict[str, object]) -> Tensor:
    values = assignment.get("encoded_message_bits", assignment["message_bits"])
    bits = torch.tensor(values, dtype=torch.int64).flatten()
    if bits.numel() != 4:
        raise ValueError("v3 cost-code block must contain four encoded bits")
    return bits

# ── Luminance weights (ITU-R BT.601) ────────────────────────────────────────
_LUMA_W = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)

# ============================================================================
# Innovation 2 – 4-Channel Cost Bundle
# ============================================================================

@dataclass
class V3ChannelCostBundle:
    """Extended cost bundle with theta_major and color_luminance channels.

    Channels
    --------
    Ch1  log_anisotropy  – identical to v2.0
    Ch2  alpha           – identical to v2.0
    Ch3  theta_major     – rotation angle in [0, π); modification ±theta_delta
    Ch4  color_lum       – luminance of mean RGB; modification ±lum_delta

    wet_mask is the union of the v2.0 mask and channel-specific exclusions.
    """

    # v2.0 fields (kept for backward compatibility)
    wet_mask: Tensor
    log_rho_plus: Tensor
    log_rho_minus: Tensor
    alpha_rho_plus: Tensor
    alpha_rho_minus: Tensor
    log_delta: float
    alpha_delta: float
    density_reference: Dict[str, Tensor]

    # New Ch3: theta
    theta_rho_plus: Tensor = field(default_factory=lambda: torch.zeros(1))
    theta_rho_minus: Tensor = field(default_factory=lambda: torch.zeros(1))
    theta_wet_mask: Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.bool))
    theta_delta: float = 0.05  # radians

    # New Ch4: color luminance
    lum_rho_plus: Tensor = field(default_factory=lambda: torch.zeros(1))
    lum_rho_minus: Tensor = field(default_factory=lambda: torch.zeros(1))
    lum_wet_mask: Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.bool))
    lum_delta: float = 0.008  # ≈2/255 pixel-level luminance shift

    def to_v2_bundle(self) -> ChannelCostBundle:
        """Downcast to a v2.0 ChannelCostBundle for code reuse."""
        return ChannelCostBundle(
            wet_mask=self.wet_mask,
            log_rho_plus=self.log_rho_plus,
            log_rho_minus=self.log_rho_minus,
            alpha_rho_plus=self.alpha_rho_plus,
            alpha_rho_minus=self.alpha_rho_minus,
            log_delta=self.log_delta,
            alpha_delta=self.alpha_delta,
            density_reference=self.density_reference,
        )

    def n_carriers_below_cost_threshold(self, threshold: float) -> int:
        """Count reliable carriers across all 4 channels at a cost threshold."""
        available = ~self.wet_mask
        n_log   = int(((self.log_rho_plus   < threshold) & available).sum().item())
        n_alpha = int(((self.alpha_rho_plus  < threshold) & available).sum().item())
        n_theta = int(((self.theta_rho_plus  < threshold) & available & ~self.theta_wet_mask).sum().item())
        n_lum   = int(((self.lum_rho_plus    < threshold) & available & ~self.lum_wet_mask).sum().item())
        return n_log + n_alpha + n_theta + n_lum


def _compute_theta_sensitivity(
    clean_params: Dict[str, Tensor],
    sensitivity: Dict[str, Tensor],
) -> Tensor:
    """Return per-Gaussian sensitivity for the theta_major channel.

    Uses the gradient of the rendered output w.r.t. theta (if present in the
    sensitivity dict), or falls back to log_anisotropy sensitivity as a proxy.
    """
    if "theta" in sensitivity:
        return sensitivity["theta"]
    if "theta_major" in sensitivity:
        return sensitivity["theta_major"]
    # Proxy: geometric mean of the two scale sensitivities
    return (sensitivity["log_anisotropy"] * 0.5 + sensitivity.get("alpha", sensitivity["log_anisotropy"]) * 0.5)


def _wrap_theta_difference(theta: Tensor) -> Tensor:
    """Map theta deltas onto the half-turn interval [-pi/2, pi/2]."""
    return 0.5 * torch.atan2(torch.sin(2.0 * theta), torch.cos(2.0 * theta))


def _luminance(colors: Tensor) -> Tensor:
    luma_w = _LUMA_W.to(colors.device, colors.dtype)
    return (colors * luma_w.view(1, 3)).sum(dim=1)


def build_v3_channel_costs(
    *,
    clean_params: Dict[str, Tensor],
    importance: Tensor,
    sensitivity: Dict[str, Tensor],
    wet_threshold: float = 0.20,
    log_delta: float = 0.12,
    alpha_delta: float = 0.04,
    theta_delta: float = 0.05,
    lum_delta: float = 0.008,
    cost_sensitivity_w: float = 0.40,
    cost_density_w: float = 0.30,
    cost_visual_w: float = 0.20,
    cost_detector_w: float = 0.10,
    extra_wet_mask: Optional[Tensor] = None,
) -> V3ChannelCostBundle:
    """Build a 4-channel cost bundle for v3.0 steganography.

    Extends build_channel_costs() (v2.0) with theta_major and color_luminance.
    The cost formula for every channel follows the same four-component design:
        ρ = sensitivity_w·sensitivity + density_w·density_shift
            + visual_w·visual_importance + detector_w·tail_penalty

    References
    ----------
    • v2.0 design: stego_protocol.build_channel_costs()
    • HILL cost model: Li et al., IEEE 2014 (texture-aware carrier selection)
    • ConcealGS: colour attribute embedding with perceptual quality guidance
    """
    features = cover_feature_matrix(clean_params)
    canonical = _canonicalize_geometry(clean_params["scales"], clean_params["thetas"])
    alpha = clean_params["opacities"][:, 0]
    importance_norm = importance / importance.mean().clamp_min(1e-6)
    density_reference = build_density_reference(clean_params)

    z_scores = torch.stack(
        [robust_z_scores(features[:, i]) for i in range(features.shape[1])], dim=1
    )

    # ── Base wet/forbid masks (same as v2.0) ────────────────────────────────
    outlier_mask       = (z_scores.abs() > 2.5).any(dim=1)
    importance_mask    = _top_fraction_mask(importance_norm, 0.10)
    log_sens_mask      = _top_fraction_mask(sensitivity["log_anisotropy"], wet_threshold)
    alpha_sens_mask    = _top_fraction_mask(sensitivity["alpha"], wet_threshold)
    minimum_scale = _minimum_carrier_scale(int(clean_params["scales"].shape[0]))
    boundary_mask      = (
        (alpha < alpha_delta) | (alpha > 1.0 - alpha_delta)
        | (clean_params["scales"].min(dim=1).values < minimum_scale)
    )
    wet_mask = outlier_mask | importance_mask | log_sens_mask | alpha_sens_mask | boundary_mask
    if extra_wet_mask is not None:
        wet_mask = wet_mask | extra_wet_mask.to(wet_mask.device, dtype=torch.bool)

    # ── Ch1: log_anisotropy (identical to v2.0) ──────────────────────────────
    log_values = canonical["log_anisotropy"]
    log_mean   = density_reference["mean"][3]
    log_std    = density_reference["std"][3]
    log_plus_density  = torch.abs((log_values + log_delta - log_mean) / log_std)
    log_minus_density = torch.abs((log_values - log_delta - log_mean) / log_std)
    log_visual   = 0.6 * importance_norm + 0.4 * _smooth_tail_penalty(z_scores[:, 3])
    log_detector = 0.5 * _smooth_tail_penalty(z_scores[:, 3]) + 0.5 * _smooth_tail_penalty(z_scores[:, 2])
    log_sens_norm = sensitivity["log_anisotropy"] / sensitivity["log_anisotropy"].mean().clamp_min(1e-6)
    log_rho_plus  = cost_sensitivity_w * log_sens_norm + cost_density_w * log_plus_density  + cost_visual_w * log_visual + cost_detector_w * log_detector
    log_rho_minus = cost_sensitivity_w * log_sens_norm + cost_density_w * log_minus_density + cost_visual_w * log_visual + cost_detector_w * log_detector

    # ── Ch2: alpha (identical to v2.0) ───────────────────────────────────────
    alpha_mean = density_reference["mean"][5]
    alpha_std  = density_reference["std"][5]
    alpha_plus_density  = torch.abs((alpha + alpha_delta - alpha_mean) / alpha_std)
    alpha_minus_density = torch.abs((alpha - alpha_delta - alpha_mean) / alpha_std)
    alpha_visual   = 0.6 * importance_norm + 0.4 * _channel_boundary_penalty(alpha, 0.0, 1.0, alpha_delta)
    alpha_detector = 0.5 * _smooth_tail_penalty(z_scores[:, 5]) + 0.5 * _smooth_tail_penalty(z_scores[:, 2])
    alpha_sens_norm   = sensitivity["alpha"] / sensitivity["alpha"].mean().clamp_min(1e-6)
    alpha_rho_plus  = cost_sensitivity_w * alpha_sens_norm + cost_density_w * alpha_plus_density  + cost_visual_w * alpha_visual + cost_detector_w * alpha_detector
    alpha_rho_minus = cost_sensitivity_w * alpha_sens_norm + cost_density_w * alpha_minus_density + cost_visual_w * alpha_visual + cost_detector_w * alpha_detector

    # ── Ch3: theta_major ─────────────────────────────────────────────────────
    theta_values  = canonical["theta_major"][:, 0]   # in [0, π)
    theta_mean    = theta_values.mean()
    theta_std     = theta_values.std(unbiased=False).clamp_min(1e-4)
    theta_sens_raw = _compute_theta_sensitivity(clean_params, sensitivity)
    theta_sens_norm = theta_sens_raw / theta_sens_raw.mean().clamp_min(1e-6)

    # Density: how much would a ±theta_delta shift the theta distribution?
    theta_plus_density  = torch.abs((theta_values + theta_delta - theta_mean) / theta_std)
    theta_minus_density = torch.abs((theta_values - theta_delta - theta_mean) / theta_std)

    # Visual: high importance Gaussians are bad carriers for theta too
    theta_visual = 0.6 * importance_norm + 0.4 * _smooth_tail_penalty(
        (theta_values - theta_mean) / theta_std.clamp_min(1e-4)
    )

    # Detector: tail penalty on theta z-score (index 4 in feature matrix)
    theta_detector = _smooth_tail_penalty(z_scores[:, 4])

    theta_rho_plus  = cost_sensitivity_w * theta_sens_norm + cost_density_w * theta_plus_density  + cost_visual_w * theta_visual + cost_detector_w * theta_detector
    theta_rho_minus = cost_sensitivity_w * theta_sens_norm + cost_density_w * theta_minus_density + cost_visual_w * theta_visual + cost_detector_w * theta_detector

    # Theta-specific wet mask: exclude near-π/2 (major/minor axis ambiguity) and near-0/π
    theta_boundary = (
        (theta_values < theta_delta * 2)
        | (theta_values > math.pi - theta_delta * 2)
        | ((theta_values > math.pi / 2 - theta_delta * 3) & (theta_values < math.pi / 2 + theta_delta * 3))
    )
    theta_wet_mask = theta_boundary | wet_mask | _top_fraction_mask(theta_sens_raw, wet_threshold)

    # ── Ch4: color luminance ─────────────────────────────────────────────────
    colors = clean_params.get("colors", None)  # (N, 3) or None if not available
    if colors is not None:
        luma_w = _LUMA_W.to(colors.device)
        lum_values = (colors * luma_w.unsqueeze(0)).sum(dim=1)   # (N,)
        lum_mean   = lum_values.mean()
        lum_std    = lum_values.std(unbiased=False).clamp_min(1e-4)
        lum_plus_density  = torch.abs((lum_values + lum_delta - lum_mean) / lum_std)
        lum_minus_density = torch.abs((lum_values - lum_delta - lum_mean) / lum_std)

        # Visual cost: very dark or saturated pixels are bad for luma embedding
        lum_visual   = 0.6 * importance_norm + 0.4 * _channel_boundary_penalty(lum_values, 0.0, 1.0, lum_delta * 5)

        # Sensitivity proxy: use alpha as a rough luminance-relevance proxy
        # (high-opacity Gaussians contribute more to pixel luminance)
        lum_sens_proxy = alpha_sens_norm
        lum_detector   = _smooth_tail_penalty((lum_values - lum_mean) / lum_std.clamp_min(1e-4))

        lum_rho_plus  = cost_sensitivity_w * lum_sens_proxy + cost_density_w * lum_plus_density  + cost_visual_w * lum_visual + cost_detector_w * lum_detector
        lum_rho_minus = cost_sensitivity_w * lum_sens_proxy + cost_density_w * lum_minus_density + cost_visual_w * lum_visual + cost_detector_w * lum_detector

        # Luminance wet mask: exclude near-black or near-white Gaussians
        lum_wet_mask = (lum_values < lum_delta * 3) | (lum_values > 1.0 - lum_delta * 3) | wet_mask
    else:
        # No color data — disable lum channel
        N = wet_mask.shape[0]
        lum_rho_plus = lum_rho_minus = torch.full((N,), float("inf"), device=wet_mask.device)
        lum_wet_mask = torch.ones(N, dtype=torch.bool, device=wet_mask.device)

    return V3ChannelCostBundle(
        wet_mask=wet_mask,
        log_rho_plus=log_rho_plus,
        log_rho_minus=log_rho_minus,
        alpha_rho_plus=alpha_rho_plus,
        alpha_rho_minus=alpha_rho_minus,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
        density_reference=density_reference,
        theta_rho_plus=theta_rho_plus,
        theta_rho_minus=theta_rho_minus,
        theta_wet_mask=theta_wet_mask,
        theta_delta=theta_delta,
        lum_rho_plus=lum_rho_plus,
        lum_rho_minus=lum_rho_minus,
        lum_wet_mask=lum_wet_mask,
        lum_delta=lum_delta,
    )


def _resolve_active_channels(
    active_channels: Optional[List[str]] = None,
    *,
    structure_profile: str = "generic",
) -> List[str]:
    raw = list(active_channels or list(V3_CHANNELS))
    resolved: List[str] = []
    seen: set[str] = set()
    for channel in raw:
        if channel not in V3_CHANNELS or channel in seen:
            continue
        seen.add(channel)
        resolved.append(channel)
    if "log_anisotropy" not in resolved:
        resolved.insert(0, "log_anisotropy")
    if "alpha" not in resolved:
        resolved.insert(1 if resolved else 0, "alpha")
    if structure_profile == "periodic":
        resolved = [channel for channel in resolved if channel != "theta"]
    return resolved


def _build_v3_selection_proxy_bundle(
    bundle: V3ChannelCostBundle,
    *,
    active_channels: Optional[List[str]],
    structure_profile: str = "generic",
    gaussian_region_labels: Optional[List[str]] = None,
    gaussian_normal_bin_ids: Optional[List[str]] = None,
    gaussian_tangent_bin_ids: Optional[List[str]] = None,
    region_allowed_channels: Optional[Dict[str, List[str]]] = None,
    region_strength_scales: Optional[Dict[str, Dict[str, float]]] = None,
    gaussian_cost_scales: Optional[Tensor] = None,
    carrier_cost_scale_fn: Optional[Callable[[int, str, str, Dict[str, object]], float]] = None,
) -> ChannelCostBundle:
    resolved_channels = _resolve_active_channels(
        active_channels,
        structure_profile=structure_profile,
    )
    proxy_min_cost = torch.full_like(bundle.log_rho_plus, 1e6)
    for gaussian_index in range(int(bundle.log_rho_plus.numel())):
        region_label = _gaussian_region_label(int(gaussian_index), gaussian_region_labels)
        allowed_channels = _allowed_channels_for_region(
            resolved_channels,
            region_label=region_label,
            region_allowed_channels=region_allowed_channels,
        )
        strength_scale_map = {
            channel_name: _strength_scale_for_region_channel(
                region_label=region_label,
                channel=channel_name,
                region_strength_scales=region_strength_scales,
            )
            for channel_name in allowed_channels
        }
        normal_bin_id = _gaussian_bin_id(int(gaussian_index), gaussian_normal_bin_ids)
        tangent_bin_id = _gaussian_bin_id(int(gaussian_index), gaussian_tangent_bin_ids)
        default_carrier_cost_scale = 1.0
        if gaussian_cost_scales is not None and int(gaussian_index) < int(gaussian_cost_scales.numel()):
            default_carrier_cost_scale = float(gaussian_cost_scales[int(gaussian_index)].item())
        best_cost = float("inf")
        for channel in allowed_channels:
            if not _channel_available(bundle, channel, int(gaussian_index)):
                continue
            strength_scale = max(1e-3, float(strength_scale_map.get(channel, 1.0)))
            carrier_cost_scale = float(default_carrier_cost_scale)
            if carrier_cost_scale_fn is not None:
                carrier_cost_scale = float(
                    carrier_cost_scale_fn(
                        int(gaussian_index),
                        str(region_label),
                        str(channel),
                        {
                            "block_index": -1,
                            "bin_id": None,
                            "normal_bin_id": normal_bin_id,
                            "tangent_bin_id": tangent_bin_id,
                            "block_normal_bin_counts": {},
                            "block_tangent_bin_counts": {},
                            "block_candidate_count": 0,
                        },
                    )
                )
            carrier_cost_scale = max(1e-6, float(carrier_cost_scale))
            for direction in (1, -1):
                channel_cost = _channel_action_cost(bundle, channel, int(gaussian_index), direction)
                effective_cost = max(
                    1e-6,
                    float(channel_cost) * strength_scale * carrier_cost_scale,
                )
                best_cost = min(best_cost, effective_cost)
        if math.isfinite(best_cost):
            proxy_min_cost[int(gaussian_index)] = float(best_cost)
    return ChannelCostBundle(
        wet_mask=bundle.wet_mask,
        log_rho_plus=proxy_min_cost,
        log_rho_minus=proxy_min_cost,
        alpha_rho_plus=proxy_min_cost,
        alpha_rho_minus=proxy_min_cost,
        log_delta=bundle.log_delta,
        alpha_delta=bundle.alpha_delta,
        density_reference=bundle.density_reference,
    )


def _channel_action_cost(
    bundle: V3ChannelCostBundle,
    channel: str,
    gaussian_index: int,
    direction: int,
) -> float:
    if channel == "log_anisotropy":
        tensor = bundle.log_rho_plus if direction > 0 else bundle.log_rho_minus
    elif channel == "alpha":
        tensor = bundle.alpha_rho_plus if direction > 0 else bundle.alpha_rho_minus
    elif channel == "theta":
        tensor = bundle.theta_rho_plus if direction > 0 else bundle.theta_rho_minus
    elif channel == "color_lum":
        tensor = bundle.lum_rho_plus if direction > 0 else bundle.lum_rho_minus
    else:
        raise ValueError(f"unsupported v3 channel: {channel!r}")
    return float(tensor[int(gaussian_index)].item())


def _channel_available(
    bundle: V3ChannelCostBundle,
    channel: str,
    gaussian_index: int,
) -> bool:
    idx = int(gaussian_index)
    if bool(bundle.wet_mask[idx].item()):
        return False
    if channel == "theta":
        return not bool(bundle.theta_wet_mask[idx].item())
    if channel == "color_lum":
        return not bool(bundle.lum_wet_mask[idx].item())
    return True


def _gaussian_region_label(
    gaussian_index: int,
    gaussian_region_labels: Optional[List[str]],
) -> str:
    if gaussian_region_labels is None:
        return "generic_region"
    idx = int(gaussian_index)
    if idx < 0 or idx >= len(gaussian_region_labels):
        return "generic_region"
    return str(gaussian_region_labels[idx])


def _gaussian_bin_id(
    gaussian_index: int,
    gaussian_bin_ids: Optional[List[str]],
) -> Optional[str]:
    if gaussian_bin_ids is None:
        return None
    idx = int(gaussian_index)
    if idx < 0 or idx >= len(gaussian_bin_ids):
        return None
    return str(gaussian_bin_ids[idx])


def _allowed_channels_for_region(
    resolved_channels: List[str],
    *,
    region_label: str,
    region_allowed_channels: Optional[Dict[str, List[str]]],
) -> List[str]:
    if not region_allowed_channels:
        return list(resolved_channels)
    allowed = region_allowed_channels.get(str(region_label))
    if not allowed:
        return list(resolved_channels)
    filtered = [channel for channel in resolved_channels if channel in allowed]
    return filtered or list(resolved_channels)


def _strength_scale_for_region_channel(
    *,
    region_label: str,
    channel: str,
    region_strength_scales: Optional[Dict[str, Dict[str, float]]],
) -> float:
    if not region_strength_scales:
        return 1.0
    region_scales = region_strength_scales.get(str(region_label), {})
    return float(region_scales.get(channel, 1.0))


def _coverage_select_candidate_indices(
    ranked_indices: Tensor,
    *,
    desired: int,
    gaussian_bin_ids: Optional[List[str]],
    selection_logits: Tensor,
    selected_bin_counts: Dict[str, int],
    minimum_unique_bins: int,
) -> Tensor:
    if gaussian_bin_ids is None or desired <= 0 or minimum_unique_bins <= 1 or ranked_indices.numel() <= desired:
        return ranked_indices[:desired]
    candidate_ids = [int(index) for index in ranked_indices.tolist()]
    bin_to_candidates: Dict[str, List[int]] = defaultdict(list)
    for gaussian_index in candidate_ids:
        bin_id = _gaussian_bin_id(gaussian_index, gaussian_bin_ids) or f"bin_{gaussian_index}"
        bin_to_candidates[bin_id].append(gaussian_index)
    ordered_bins = sorted(
        bin_to_candidates.keys(),
        key=lambda bin_id: (
            int(selected_bin_counts.get(bin_id, 0)),
            -float(max(selection_logits[idx].item() for idx in bin_to_candidates[bin_id])),
        ),
    )
    chosen: list[int] = []
    used: set[int] = set()
    target_unique = min(int(desired), max(int(minimum_unique_bins), 1), len(ordered_bins))
    for bin_id in ordered_bins[:target_unique]:
        best_idx = max(bin_to_candidates[bin_id], key=lambda idx: float(selection_logits[idx].item()))
        chosen.append(best_idx)
        used.add(best_idx)
    remaining = [idx for idx in candidate_ids if idx not in used]
    remaining.sort(
        key=lambda idx: (
            int(selected_bin_counts.get(_gaussian_bin_id(idx, gaussian_bin_ids) or f"bin_{idx}", 0)),
            -float(selection_logits[idx].item()),
        )
    )
    chosen.extend(remaining[: max(0, int(desired) - len(chosen))])
    return torch.tensor(chosen[:desired], dtype=torch.long)


def _v3_candidate_action_option(
    gaussian_index: int,
    *,
    bundle: V3ChannelCostBundle,
    active_channels: List[str],
    allowed_channels: Optional[List[str]],
    strength_scale_map: Optional[Dict[str, float]],
    generator: torch.Generator,
    selection_temperature: float,
    mode: str,
) -> tuple[str, int, float, float]:
    options: list[tuple[str, int, float, float]] = []
    for channel in (allowed_channels or active_channels):
        if not _channel_available(bundle, channel, gaussian_index):
            continue
        strength_scale = max(
            1e-3,
            float((strength_scale_map or {}).get(channel, 1.0)),
        )
        for direction in (1, -1):
            options.append(
                (
                    channel,
                    direction,
                    _channel_action_cost(bundle, channel, gaussian_index, direction) * strength_scale,
                    strength_scale,
                )
            )
    if not options:
        raise RuntimeError(f"v3 builder found no legal action for gaussian {gaussian_index}")
    if mode == "greedy":
        return min(options, key=lambda item: item[2])
    if mode == "uniform":
        choice = int(torch.randint(0, len(options), size=(1,), generator=generator).item())
        return options[choice]
    if mode == "random_polarity":
        direction = 1 if int(torch.randint(0, 2, size=(1,), generator=generator).item()) == 0 else -1
        subset = [item for item in options if item[1] == direction]
        if not subset:
            subset = options
        choice = int(torch.randint(0, len(subset), size=(1,), generator=generator).item())
        return subset[choice]
    best_option = None
    best_score = None
    for channel, direction, cost, strength_scale in options:
        noise = torch.rand(1, generator=generator, dtype=torch.float32).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = float((torch.log(noise) - torch.log1p(-noise)).item())
        score = -cost + max(float(selection_temperature), 1e-4) * gumbel
        if best_score is None or score > best_score:
            best_score = score
            best_option = (channel, direction, cost, strength_scale)
    assert best_option is not None
    return best_option


def _v3_candidate_canonical_sort_key(item: tuple[dict[str, object], float]) -> tuple[int, int, int, int]:
    action, _cost = item
    channel = str(action.get("channel", ""))
    channel_rank = {
        "log_anisotropy": 0,
        "alpha": 1,
        "theta": 2,
        "color_lum": 3,
    }.get(channel, 99)
    return (
        int(action.get("gaussian_index", -1)),
        int(channel_rank),
        int(action.get("direction", 0)),
        int(action.get("_candidate_order", -1)),
    )


def _canonicalize_v3_candidate_block(
    candidate_actions: List[Dict[str, object]],
    candidate_costs: List[float],
) -> tuple[List[Dict[str, object]], List[float]]:
    if len(candidate_actions) != len(candidate_costs):
        raise ValueError("candidate_actions and candidate_costs must have the same length")
    indexed: list[tuple[dict[str, object], float]] = []
    for original_index, (action, cost) in enumerate(zip(candidate_actions, candidate_costs)):
        action_local = dict(action)
        action_local["_candidate_order"] = int(original_index)
        indexed.append((action_local, float(cost)))
    indexed.sort(key=_v3_candidate_canonical_sort_key)
    canonical_actions: list[dict[str, object]] = []
    canonical_costs: list[float] = []
    for action, cost in indexed:
        action.pop("_candidate_order", None)
        canonical_actions.append(action)
        canonical_costs.append(float(cost))
    return canonical_actions, canonical_costs


def _selected_candidate_costs_by_channel(
    assignments: List[Dict],
) -> Dict[str, list[float]]:
    grouped: Dict[str, list[float]] = {channel: [] for channel in V3_CHANNELS}
    for assignment in assignments:
        candidate_actions = assignment.get("candidate_actions", [])
        selected_mask = assignment.get("selected_mask", [])
        for action, selected in zip(candidate_actions, selected_mask):
            if float(selected) <= 0.5:
                continue
            channel = str(action.get("channel", ""))
            if channel in grouped:
                grouped[channel].append(float(action.get("channel_cost", action.get("cost", 1.0))))
    return grouped


def compute_v3_channel_weights(
    assignments: List[Dict],
    *,
    active_channels: List[str],
    structure_profile: str = "generic",
) -> Dict[str, float]:
    grouped = _selected_candidate_costs_by_channel(assignments)
    inverse_medians: Dict[str, float] = {}
    for channel in V3_CHANNELS:
        if channel not in active_channels:
            inverse_medians[channel] = 0.0
            continue
        values = grouped.get(channel, [])
        if not values:
            inverse_medians[channel] = 0.0
            continue
        median_cost = float(torch.tensor(values, dtype=torch.float32).median().item())
        inverse_medians[channel] = 1.0 / max(median_cost, 1e-6)
    max_inverse = max(inverse_medians.values()) if inverse_medians else 1.0
    weights: Dict[str, float] = {}
    for channel in V3_CHANNELS:
        if channel not in active_channels or max_inverse <= 0.0:
            weights[channel] = 0.0
            continue
        weight = inverse_medians[channel] / max_inverse
        weights[channel] = float(min(1.0, max(0.25, weight)))
    if structure_profile == "periodic":
        weights["theta"] = 0.0
    return weights


def v3_per_channel_action_counts(assignments: List[Dict]) -> Dict[str, int]:
    counts = {channel: 0 for channel in V3_CHANNELS}
    for assignment in assignments:
        for action, selected in zip(assignment.get("candidate_actions", []), assignment.get("selected_mask", [])):
            if float(selected) <= 0.5:
                continue
            channel = str(action.get("channel", ""))
            if channel in counts:
                counts[channel] += 1
    return counts


def _augment_assignment_channel_views(assignments: List[Dict]) -> List[Dict]:
    for assignment in assignments:
        assignment["log_indices"] = []
        assignment["log_weights"] = []
        assignment["alpha_indices"] = []
        assignment["alpha_weights"] = []
        assignment["theta_indices"] = []
        assignment["theta_weights"] = []
        assignment["lum_indices"] = []
        assignment["lum_weights"] = []
        for action, selected in zip(assignment.get("candidate_actions", []), assignment.get("selected_mask", [])):
            if float(selected) <= 0.5:
                continue
            channel = str(action.get("channel", ""))
            gaussian_index = int(action["gaussian_index"])
            weight = float(action.get("channel_weight", 1.0))
            if channel == "log_anisotropy":
                assignment["log_indices"].append(gaussian_index)
                assignment["log_weights"].append(weight)
            elif channel == "alpha":
                assignment["alpha_indices"].append(gaussian_index)
                assignment["alpha_weights"].append(weight)
            elif channel == "theta":
                assignment["theta_indices"].append(gaussian_index)
                assignment["theta_weights"].append(weight)
            elif channel == "color_lum":
                assignment["lum_indices"].append(gaussian_index)
                assignment["lum_weights"].append(weight)
    return assignments


def _parity_bits_to_state(bits: List[int] | Tuple[int, ...]) -> int:
    state = 0
    for index, bit in enumerate(bits):
        if int(bit):
            state |= 1 << index
    return state


def _selected_assignment_coverage_summary(
    assignments: List[Dict],
    *,
    max_bin_share: float,
    periodic_core_action_cap_ratio: float,
) -> Dict[str, object]:
    selected_bin_counts: Dict[str, int] = defaultdict(int)
    zone_action_counts: Dict[str, int] = defaultdict(int)
    total_selected = 0
    periodic_core_selected = 0
    for assignment in assignments:
        for action, selected in zip(assignment.get("candidate_actions", []), assignment.get("selected_mask", [])):
            if float(selected) <= 0.5:
                continue
            total_selected += 1
            region_label = str(action.get("region_label", "generic_region"))
            zone_action_counts[region_label] += 1
            if region_label == "periodic_core":
                periodic_core_selected += 1
            bin_id = action.get("bin_id")
            if bin_id is not None:
                selected_bin_counts[str(bin_id)] += 1
    realized_selected_max_bin_share = 0.0
    if total_selected > 0 and selected_bin_counts:
        realized_selected_max_bin_share = max(selected_bin_counts.values()) / float(total_selected)
    realized_periodic_core_action_share = (
        float(periodic_core_selected) / float(total_selected)
        if total_selected > 0
        else 0.0
    )
    coverage_constraint_satisfied = float(realized_selected_max_bin_share) <= float(max_bin_share) + 1e-9
    periodic_core_constraint_satisfied = (
        float(realized_periodic_core_action_share) <= float(periodic_core_action_cap_ratio) + 1e-9
    )
    return {
        "total_selected_actions": int(total_selected),
        "selected_bin_counts": {str(key): int(value) for key, value in sorted(selected_bin_counts.items())},
        "zone_action_counts": {str(key): int(value) for key, value in sorted(zone_action_counts.items())},
        "realized_selected_max_bin_share": float(realized_selected_max_bin_share),
        "realized_periodic_core_action_share": float(realized_periodic_core_action_share),
        "target_max_bin_share_cap": float(max_bin_share),
        "periodic_core_action_cap_ratio": float(periodic_core_action_cap_ratio),
        "coverage_constraint_satisfied": bool(
            coverage_constraint_satisfied and periodic_core_constraint_satisfied
        ),
        "periodic_core_constraint_satisfied": bool(periodic_core_constraint_satisfied),
    }


def _enumerate_cost_code_block_solutions(
    assignment: Dict[str, object],
    *,
    beam_width: int,
    max_solutions: int = 16,
) -> List[Dict[str, object]]:
    candidate_actions = list(assignment.get("candidate_actions", []))
    candidate_costs = torch.tensor(assignment.get("candidate_costs", []), dtype=torch.float32)
    if candidate_costs.numel() == 0:
        return [
            {
                "selected_indices": tuple(),
                "selected_mask": [],
                "selected_count": 0,
                "total_cost": 0.0,
                "bin_counts": {},
                "periodic_core_count": 0,
            }
        ]
    parity_matrix = torch.tensor(assignment["parity_matrix"], dtype=torch.int64).t()
    target_bits = _assignment_encoded_message_bits(assignment)
    target_state = _parity_bits_to_state([int(value) for value in target_bits.tolist()])
    beam_limit = max(int(beam_width), int(max_solutions) * 8)
    beams: List[Tuple[float, int, Tuple[int, ...]]] = [(0.0, 0, tuple())]
    for action_index in range(candidate_costs.numel()):
        column_state = _parity_bits_to_state([int(value) for value in parity_matrix[:, action_index].tolist()])
        action_cost = float(candidate_costs[action_index].item())
        next_beams: List[Tuple[float, int, Tuple[int, ...]]] = []
        for cost, state, selected in beams:
            next_beams.append((cost, state, selected))
            next_beams.append((cost + action_cost, state ^ column_state, (*selected, int(action_index))))
        next_beams.sort(key=lambda item: (item[0], len(item[2])))
        beams = next_beams[:beam_limit]
    valid = sorted(
        [item for item in beams if item[1] == target_state],
        key=lambda item: (item[0], len(item[2]), item[2]),
    )
    if not valid:
        raise RuntimeError("cost_code_v3 coverage solver could not satisfy the target syndrome")
    solutions: List[Dict[str, object]] = []
    seen: set[Tuple[int, ...]] = set()
    for cost, _, selected in valid:
        if selected in seen:
            continue
        seen.add(selected)
        selected_set = set(int(index) for index in selected)
        bin_counts: Dict[str, int] = defaultdict(int)
        periodic_core_count = 0
        selected_mask: List[float] = []
        for action_index, action in enumerate(candidate_actions):
            active = action_index in selected_set
            selected_mask.append(1.0 if active else 0.0)
            if not active:
                continue
            if str(action.get("region_label", "generic_region")) == "periodic_core":
                periodic_core_count += 1
            bin_id = action.get("bin_id")
            if bin_id is not None:
                bin_counts[str(bin_id)] += 1
        solutions.append(
            {
                "selected_indices": selected,
                "selected_mask": selected_mask,
                "selected_count": int(len(selected)),
                "total_cost": float(cost),
                "bin_counts": dict(bin_counts),
                "periodic_core_count": int(periodic_core_count),
            }
        )
        if len(solutions) >= max(1, int(max_solutions)):
            break
    return solutions


def _coverage_state_still_feasible(
    *,
    bin_counts: Dict[str, int],
    periodic_core_count: int,
    total_selected: int,
    remaining_max_selected: int,
    modification_budget: int,
    max_bin_share: float,
    periodic_core_action_cap_ratio: float,
) -> bool:
    if total_selected > int(modification_budget):
        return False
    if max_bin_share >= 1.0 and periodic_core_action_cap_ratio >= 1.0:
        return True
    max_possible_total = total_selected + max(0, int(remaining_max_selected))
    if max_possible_total <= 0:
        return True
    if max_bin_share < 1.0:
        for count in bin_counts.values():
            if float(count) / float(max_possible_total) > float(max_bin_share) + 1e-9:
                return False
    if periodic_core_action_cap_ratio < 1.0 and (
        float(periodic_core_count) / float(max_possible_total)
        > float(periodic_core_action_cap_ratio) + 1e-9
    ):
        return False
    return True


def _coverage_violation_score(
    *,
    bin_counts: Dict[str, int],
    periodic_core_count: int,
    total_selected: int,
    max_bin_share: float,
    periodic_core_action_cap_ratio: float,
) -> Tuple[float, float, float]:
    if total_selected <= 0:
        return 0.0, 0.0, 0.0
    max_share = (
        max(float(count) for count in bin_counts.values()) / float(total_selected)
        if bin_counts
        else 0.0
    )
    periodic_core_share = float(periodic_core_count) / float(total_selected)
    return (
        max(0.0, max_share - float(max_bin_share)),
        max(0.0, periodic_core_share - float(periodic_core_action_cap_ratio)),
        max_share,
    )


def enforce_assignment_coverage_constraints(
    assignments: List[Dict],
    *,
    beam_width: int,
    modification_budget: int,
    max_bin_share: float,
    periodic_core_action_cap_ratio: float,
    max_solutions_per_block: int = 16,
) -> Tuple[List[Dict], Dict[str, object]]:
    if not assignments:
        summary = _selected_assignment_coverage_summary(
            assignments,
            max_bin_share=max_bin_share,
            periodic_core_action_cap_ratio=periodic_core_action_cap_ratio,
        )
        return _augment_assignment_channel_views(assignments), {
            **summary,
            "coverage_feasible": True,
            "coverage_repair_used": False,
        }
    if max_bin_share >= 1.0 and periodic_core_action_cap_ratio >= 1.0:
        summary = _selected_assignment_coverage_summary(
            assignments,
            max_bin_share=max_bin_share,
            periodic_core_action_cap_ratio=periodic_core_action_cap_ratio,
        )
        return _augment_assignment_channel_views(assignments), {
            **summary,
            "coverage_feasible": True,
            "coverage_repair_used": False,
        }

    solution_pools = [
        _enumerate_cost_code_block_solutions(
            assignment,
            beam_width=max(int(beam_width), 8),
            max_solutions=max_solutions_per_block,
        )
        for assignment in assignments
    ]
    suffix_max_selected = [0] * (len(solution_pools) + 1)
    for block_index in range(len(solution_pools) - 1, -1, -1):
        max_selected = max((int(solution["selected_count"]) for solution in solution_pools[block_index]), default=0)
        suffix_max_selected[block_index] = suffix_max_selected[block_index + 1] + max_selected

    def _search(*, require_feasible: bool) -> Dict[str, object] | None:
        beams: List[Dict[str, object]] = [
            {
                "cost": 0.0,
                "solution_indices": [],
                "bin_counts": {},
                "periodic_core_count": 0,
                "total_selected": 0,
                "changed": False,
            }
        ]
        beam_limit = max(int(beam_width) * 8, 128)
        for block_index, pool in enumerate(solution_pools):
            next_states: List[Dict[str, object]] = []
            remaining_max_selected = suffix_max_selected[block_index + 1]
            for state in beams:
                base_bin_counts = dict(state["bin_counts"])
                for solution_index, solution in enumerate(pool):
                    new_total_selected = int(state["total_selected"]) + int(solution["selected_count"])
                    if new_total_selected > int(modification_budget):
                        continue
                    new_bin_counts = dict(base_bin_counts)
                    for bin_id, count in dict(solution["bin_counts"]).items():
                        new_bin_counts[str(bin_id)] = int(new_bin_counts.get(str(bin_id), 0)) + int(count)
                    new_periodic_core_count = int(state["periodic_core_count"]) + int(solution["periodic_core_count"])
                    if require_feasible and not _coverage_state_still_feasible(
                        bin_counts=new_bin_counts,
                        periodic_core_count=new_periodic_core_count,
                        total_selected=new_total_selected,
                        remaining_max_selected=remaining_max_selected,
                        modification_budget=modification_budget,
                        max_bin_share=max_bin_share,
                        periodic_core_action_cap_ratio=periodic_core_action_cap_ratio,
                    ):
                        continue
                    bin_violation, core_violation, realized_share = _coverage_violation_score(
                        bin_counts=new_bin_counts,
                        periodic_core_count=new_periodic_core_count,
                        total_selected=new_total_selected,
                        max_bin_share=max_bin_share,
                        periodic_core_action_cap_ratio=periodic_core_action_cap_ratio,
                    )
                    next_states.append(
                        {
                            "cost": float(state["cost"]) + float(solution["total_cost"]),
                            "solution_indices": [*list(state["solution_indices"]), int(solution_index)],
                            "bin_counts": new_bin_counts,
                            "periodic_core_count": int(new_periodic_core_count),
                            "total_selected": int(new_total_selected),
                            "changed": bool(state["changed"]) or int(solution_index) != 0,
                            "bin_violation": float(bin_violation),
                            "core_violation": float(core_violation),
                            "realized_share": float(realized_share),
                        }
                    )
            if not next_states:
                return None
            if require_feasible:
                next_states.sort(
                    key=lambda item: (
                        float(item["cost"]),
                        int(item["total_selected"]),
                        bool(item["changed"]),
                    )
                )
            else:
                next_states.sort(
                    key=lambda item: (
                        float(item["bin_violation"]) + float(item["core_violation"]),
                        float(item["cost"]),
                        int(item["total_selected"]),
                        bool(item["changed"]),
                    )
                )
            beams = next_states[:beam_limit]
        finals = []
        for state in beams:
            bin_violation, core_violation, _ = _coverage_violation_score(
                bin_counts=dict(state["bin_counts"]),
                periodic_core_count=int(state["periodic_core_count"]),
                total_selected=int(state["total_selected"]),
                max_bin_share=max_bin_share,
                periodic_core_action_cap_ratio=periodic_core_action_cap_ratio,
            )
            state = dict(state)
            state["bin_violation"] = float(bin_violation)
            state["core_violation"] = float(core_violation)
            if require_feasible and (bin_violation > 1e-9 or core_violation > 1e-9):
                continue
            finals.append(state)
        if not finals:
            return None
        if require_feasible:
            finals.sort(key=lambda item: (float(item["cost"]), int(item["total_selected"]), bool(item["changed"])))
        else:
            finals.sort(
                key=lambda item: (
                    float(item["bin_violation"]) + float(item["core_violation"]),
                    float(item["cost"]),
                    int(item["total_selected"]),
                    bool(item["changed"]),
                )
            )
        return finals[0]

    best_feasible = _search(require_feasible=True)
    chosen_state = best_feasible
    coverage_feasible = True
    if chosen_state is None:
        chosen_state = _search(require_feasible=False)
        coverage_feasible = False
    if chosen_state is None:
        summary = _selected_assignment_coverage_summary(
            assignments,
            max_bin_share=max_bin_share,
            periodic_core_action_cap_ratio=periodic_core_action_cap_ratio,
        )
        return _augment_assignment_channel_views(assignments), {
            **summary,
            "coverage_feasible": False,
            "coverage_repair_used": False,
        }

    updated_assignments: List[Dict] = []
    for assignment, solution_index in zip(assignments, chosen_state["solution_indices"]):
        solution = solution_pools[len(updated_assignments)][int(solution_index)]
        new_assignment = dict(assignment)
        new_assignment["selected_mask"] = [float(value) for value in solution["selected_mask"]]
        new_assignment["selected_count"] = int(solution["selected_count"])
        selected_bin_ids = {
            str(action.get("bin_id"))
            for action, selected in zip(new_assignment.get("candidate_actions", []), new_assignment["selected_mask"])
            if float(selected) > 0.5 and action.get("bin_id") is not None
        }
        new_assignment["selected_bin_coverage_count"] = len(selected_bin_ids)
        updated_assignments.append(new_assignment)
    updated_assignments = _augment_assignment_channel_views(updated_assignments)
    summary = _selected_assignment_coverage_summary(
        updated_assignments,
        max_bin_share=max_bin_share,
        periodic_core_action_cap_ratio=periodic_core_action_cap_ratio,
    )
    return updated_assignments, {
        **summary,
        "coverage_feasible": bool(coverage_feasible),
        "coverage_repair_used": bool(chosen_state.get("changed", False)),
    }


def _build_v3_cost_code_assignments(
    *,
    raw_bits: Tensor,
    clean_params: Dict[str, Tensor],
    importance: Tensor,
    sensitivity: Dict[str, Tensor],
    v3_bundle: V3ChannelCostBundle,
    key: str,
    selection_mode: str,
    candidate_count: int,
    fallback_candidate_count: int,
    beam_width: int,
    selection_temperature: float,
    max_modification_ratio: float,
    active_channels: Optional[List[str]] = None,
    structure_profile: str = "generic",
    gaussian_region_labels: Optional[List[str]] = None,
    gaussian_bin_ids: Optional[List[str]] = None,
    gaussian_normal_bin_ids: Optional[List[str]] = None,
    gaussian_tangent_bin_ids: Optional[List[str]] = None,
    region_allowed_channels: Optional[Dict[str, List[str]]] = None,
    region_strength_scales: Optional[Dict[str, Dict[str, float]]] = None,
    gaussian_cost_scales: Optional[Tensor] = None,
    carrier_cost_scale_fn: Optional[Callable[[int, str, str, Dict[str, object]], float]] = None,
    coverage_balancing: bool = False,
    minimum_block_bin_coverage: int = 1,
    max_bin_share: float = 1.0,
    periodic_core_action_cap_ratio: float = 1.0,
    min_selected_count: int | None = None,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    raw_bits = raw_bits.to(torch.int64).flatten()
    if raw_bits.numel() % 4 != 0:
        pad = (-raw_bits.numel()) % 4
        raw_bits = torch.cat((raw_bits, torch.zeros(pad, dtype=torch.int64)))
    selection_builder = build_soft_selection_field if selection_mode in {"keyed", "uniform"} else build_unkeyed_selection_field
    resolved_channels = _resolve_active_channels(active_channels, structure_profile=structure_profile)
    selection_kwargs = dict(
        clean_params=clean_params,
        importance=importance,
        sensitivity=sensitivity,
        cost_bundle=_build_v3_selection_proxy_bundle(
            v3_bundle,
            active_channels=resolved_channels,
            structure_profile=structure_profile,
            gaussian_region_labels=gaussian_region_labels,
            gaussian_normal_bin_ids=gaussian_normal_bin_ids,
            gaussian_tangent_bin_ids=gaussian_tangent_bin_ids,
            region_allowed_channels=region_allowed_channels,
            region_strength_scales=region_strength_scales,
            gaussian_cost_scales=gaussian_cost_scales,
            carrier_cost_scale_fn=carrier_cost_scale_fn,
        ),
        selection_temperature=selection_temperature,
    )
    if selection_builder is build_soft_selection_field:
        selection_bundle = selection_builder(key=key, **selection_kwargs)
    else:
        selection_bundle = selection_builder(**selection_kwargs)
    allowed_indices = torch.where(~selection_bundle.hard_forbid_mask)[0]
    legal_indices = [
        int(index)
        for index in allowed_indices.tolist()
        if any(_channel_available(v3_bundle, channel, int(index)) for channel in resolved_channels)
    ]
    allowed_indices = torch.tensor(legal_indices, dtype=torch.long)
    if allowed_indices.numel() == 0:
        raise RuntimeError("v3 cost code builder did not find any modifiable Gaussian carriers")
    modification_budget = max(1, int(math.floor(float(max_modification_ratio) * float(clean_params["offsets"].shape[0]))))
    selection_probs = selection_bundle.selection_probs
    assignments: List[Dict[str, object]] = []
    used_gaussians: set[int] = set()
    heuristic_total_selected = 0
    heuristic_selected_bin_counts: Dict[str, int] = defaultdict(int)
    heuristic_zone_action_counts: Dict[str, int] = defaultdict(int)
    for block_index, block_bits in enumerate(raw_bits.view(-1, 4)):
        whitening_bits = _keyed_whitening_bits(key, block_index)
        encoded_block_bits = torch.remainder(block_bits + whitening_bits, 2)
        available = torch.tensor(
            [int(index) for index in allowed_indices.tolist() if int(index) not in used_gaussians],
            dtype=torch.long,
        )
        if available.numel() < 8:
            raise RuntimeError("v3 cost code builder ran out of available Gaussian carriers")
        desired = candidate_count if available.numel() >= candidate_count else min(int(available.numel()), fallback_candidate_count)
        if desired <= 0:
            raise RuntimeError("v3 cost code builder could not allocate a non-empty candidate set")
        if selection_mode == "greedy":
            ranked = available[torch.argsort(selection_bundle.selection_logits[available], descending=True).detach().cpu()]
        else:
            ranked = available
        base_seed = key_to_seed(f"{key}::v3::{selection_mode}::{block_index}") + 1301
        selected_mask: Tensor | None = None
        candidates: list[dict[str, object]] = []
        candidate_costs: list[float] = []
        parity_matrix: Tensor | None = None
        for retry_index in range(4):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(base_seed + 997 * retry_index)
            retry_desired = desired if retry_index == 0 else min(int(ranked.numel()), max(8, fallback_candidate_count + 4 * retry_index))
            if selection_mode == "greedy":
                candidate_indices = ranked[:retry_desired]
            elif selection_mode == "uniform":
                shuffled = torch.randperm(int(ranked.numel()), generator=generator)
                candidate_indices = ranked[shuffled[:retry_desired]]
            else:
                sampled = _gumbel_topk(selection_bundle.selection_logits[ranked].detach().cpu(), k=retry_desired, generator=generator)
                candidate_indices = ranked[sampled]
                if coverage_balancing:
                    candidate_indices = _coverage_select_candidate_indices(
                        candidate_indices.detach().cpu(),
                        desired=retry_desired,
                        gaussian_bin_ids=gaussian_bin_ids,
                        selection_logits=selection_bundle.selection_logits.detach().cpu(),
                        selected_bin_counts=heuristic_selected_bin_counts,
                        minimum_unique_bins=max(int(minimum_block_bin_coverage), 1),
                    )
            candidates = []
            candidate_costs = []
            block_normal_bin_counts: Dict[str, int] = defaultdict(int)
            block_tangent_bin_counts: Dict[str, int] = defaultdict(int)
            for gaussian_index in candidate_indices.tolist():
                action_mode = selection_mode
                if action_mode == "keyed":
                    action_mode = "keyed"
                elif action_mode == "greedy":
                    action_mode = "greedy"
                elif action_mode == "uniform":
                    action_mode = "uniform"
                else:
                    action_mode = "random_polarity"
                region_label = _gaussian_region_label(int(gaussian_index), gaussian_region_labels)
                allowed_channels = _allowed_channels_for_region(
                    resolved_channels,
                    region_label=region_label,
                    region_allowed_channels=region_allowed_channels,
                )
                strength_scale_map = {
                    channel_name: _strength_scale_for_region_channel(
                        region_label=region_label,
                        channel=channel_name,
                        region_strength_scales=region_strength_scales,
                    )
                    for channel_name in allowed_channels
                }
                channel, direction, cost, strength_scale = _v3_candidate_action_option(
                    int(gaussian_index),
                    bundle=v3_bundle,
                    active_channels=resolved_channels,
                    allowed_channels=allowed_channels,
                    strength_scale_map=strength_scale_map,
                    generator=generator,
                    selection_temperature=selection_temperature,
                    mode=action_mode,
                )
                bin_id = _gaussian_bin_id(int(gaussian_index), gaussian_bin_ids)
                normal_bin_id = _gaussian_bin_id(int(gaussian_index), gaussian_normal_bin_ids)
                tangent_bin_id = _gaussian_bin_id(int(gaussian_index), gaussian_tangent_bin_ids)
                occupancy_penalty = 0.0
                if coverage_balancing and bin_id is not None and heuristic_total_selected > 0:
                    current_share = float(heuristic_selected_bin_counts.get(bin_id, 0)) / max(1.0, float(heuristic_total_selected))
                    if current_share >= max_bin_share:
                        occupancy_penalty += 1.0 + current_share
                    elif current_share >= (0.5 * max_bin_share):
                        occupancy_penalty += 0.35 * current_share / max(max_bin_share, 1e-6)
                if coverage_balancing and region_label == "periodic_core" and heuristic_total_selected > 0:
                    core_share = float(heuristic_zone_action_counts.get("periodic_core", 0)) / max(1.0, float(heuristic_total_selected))
                    if core_share >= periodic_core_action_cap_ratio:
                        occupancy_penalty += 1.0 + core_share
                    elif core_share >= (0.75 * periodic_core_action_cap_ratio):
                        occupancy_penalty += 0.35
                carrier_cost_scale = 1.0
                if carrier_cost_scale_fn is not None:
                    carrier_cost_scale = float(
                        carrier_cost_scale_fn(
                            int(gaussian_index),
                            str(region_label),
                            str(channel),
                            {
                                "block_index": int(block_index),
                                "bin_id": bin_id,
                                "normal_bin_id": normal_bin_id,
                                "tangent_bin_id": tangent_bin_id,
                                "block_normal_bin_counts": dict(block_normal_bin_counts),
                                "block_tangent_bin_counts": dict(block_tangent_bin_counts),
                                "block_candidate_count": len(candidates),
                            },
                        )
                    )
                elif gaussian_cost_scales is not None and int(gaussian_index) < int(gaussian_cost_scales.numel()):
                    carrier_cost_scale = float(gaussian_cost_scales[int(gaussian_index)].item())
                carrier_cost_scale = max(1e-6, float(carrier_cost_scale))
                effective_cost = max(1e-6, float(cost) * float(carrier_cost_scale) + float(occupancy_penalty))
                candidates.append(
                    {
                        "gaussian_index": int(gaussian_index),
                        "channel": channel,
                        "direction": int(direction),
                        "strength_scale": float(strength_scale),
                        "region_label": region_label,
                        "bin_id": bin_id,
                        "normal_bin_id": normal_bin_id,
                        "tangent_bin_id": tangent_bin_id,
                        "selection_probability": float(selection_probs[gaussian_index].item()),
                        "carrier_cost_scale": float(carrier_cost_scale),
                        "channel_cost": float(effective_cost),
                    }
                )
                candidate_costs.append(float(effective_cost))
                if normal_bin_id is not None:
                    block_normal_bin_counts[str(normal_bin_id)] += 1
                if tangent_bin_id is not None:
                    block_tangent_bin_counts[str(tangent_bin_id)] += 1
            candidates, candidate_costs = _canonicalize_v3_candidate_block(
                candidates,
                candidate_costs,
            )
            parity_matrix = _make_parity_matrix(len(candidates), generator=generator)
            try:
                solve_costs = torch.tensor(candidate_costs, dtype=torch.float32)
                if selection_mode == "uniform":
                    solve_costs = torch.ones_like(solve_costs)
                selected_mask = _solve_cost_code_block(
                    candidate_costs=solve_costs,
                    parity_matrix=parity_matrix,
                    target_bits=encoded_block_bits,
                    beam_width=beam_width,
                    min_selected_count=min_selected_count,
                )
                break
            except RuntimeError:
                selected_mask = None
                continue
        if selected_mask is None or parity_matrix is None:
            raise RuntimeError("v3 cost code builder could not satisfy the target syndrome after retrying candidate pools")
        selected_indices = torch.where(selected_mask > 0.5)[0]
        selected_bin_ids: set[str] = set()
        for selected_index in selected_indices.tolist():
            action = candidates[int(selected_index)]
            region_label = str(action.get("region_label", "generic_region"))
            heuristic_zone_action_counts[region_label] += 1
            bin_id = action.get("bin_id")
            if bin_id is not None:
                heuristic_selected_bin_counts[str(bin_id)] += 1
                selected_bin_ids.add(str(bin_id))
        heuristic_total_selected += int(selected_indices.numel())
        for candidate in candidates:
            used_gaussians.add(int(candidate["gaussian_index"]))
        assignments.append(
            {
                "block_index": int(block_index),
                "message_bits": [int(value) for value in block_bits.tolist()],
                "whitening_bits": [int(value) for value in whitening_bits.tolist()],
                "encoded_message_bits": [int(value) for value in encoded_block_bits.tolist()],
                "bit": int(block_bits[0].item()) if int(block_bits.numel()) > 0 else 0,
                "candidate_count": len(candidates),
                "candidate_actions": candidates,
                "candidate_costs": candidate_costs,
                "parity_matrix": parity_matrix.t().tolist(),
                "selected_mask": [float(value) for value in selected_mask.tolist()],
                "selected_count": int(selected_indices.numel()),
                "candidate_bin_coverage_count": len(
                    {
                        str(action.get("bin_id"))
                        for action in candidates
                        if action.get("bin_id") is not None
                    }
                ),
                "selected_bin_coverage_count": len(selected_bin_ids),
            }
        )
    if coverage_balancing:
        assignments, coverage_meta = enforce_assignment_coverage_constraints(
            assignments,
            beam_width=beam_width,
            modification_budget=modification_budget,
            max_bin_share=float(max_bin_share),
            periodic_core_action_cap_ratio=float(periodic_core_action_cap_ratio),
        )
    else:
        coverage_meta = {
            **_selected_assignment_coverage_summary(
                assignments,
                max_bin_share=float(max_bin_share),
                periodic_core_action_cap_ratio=float(periodic_core_action_cap_ratio),
            ),
            "coverage_feasible": True,
            "coverage_repair_used": False,
        }
        assignments = _augment_assignment_channel_views(assignments)
    total_selected = int(coverage_meta["total_selected_actions"])
    if total_selected > modification_budget:
        raise RuntimeError("v3 cost code builder exceeded the maximum modification ratio")
    meta = {
        "selection_entropy": float(selection_bundle.selection_entropy),
        "effective_action_budget": int(selection_bundle.effective_action_budget),
        "hard_forbid_ratio": float(selection_bundle.hard_forbid_mask.float().mean().item()),
        "total_selected_actions": int(total_selected),
        "max_modification_count": modification_budget,
        "selection_probs": [float(value) for value in selection_bundle.selection_probs.detach().cpu().tolist()],
        "hard_forbid_mask": [bool(value) for value in selection_bundle.hard_forbid_mask.detach().cpu().tolist()],
        "active_channels": resolved_channels,
        "structure_profile": structure_profile,
        "coverage_balancing_enabled": bool(coverage_balancing),
        "minimum_block_bin_coverage": int(minimum_block_bin_coverage),
        "selected_bin_counts": dict(coverage_meta["selected_bin_counts"]),
        "realized_selected_max_bin_share": float(coverage_meta["realized_selected_max_bin_share"]),
        "realized_periodic_core_action_share": float(coverage_meta["realized_periodic_core_action_share"]),
        "target_max_bin_share_cap": float(coverage_meta["target_max_bin_share_cap"]),
        "coverage_constraint_satisfied": bool(coverage_meta["coverage_constraint_satisfied"]),
        "periodic_core_constraint_satisfied": bool(coverage_meta["periodic_core_constraint_satisfied"]),
        "coverage_feasible": bool(coverage_meta["coverage_feasible"]),
        "coverage_repair_used": bool(coverage_meta["coverage_repair_used"]),
        "zone_action_counts": dict(coverage_meta["zone_action_counts"]),
        "periodic_core_action_cap_ratio": float(coverage_meta["periodic_core_action_cap_ratio"]),
    }
    return assignments, meta


def build_v3_cost_code_assignments(**kwargs):
    return _build_v3_cost_code_assignments(selection_mode="keyed", **kwargs)


def build_v3_random_cost_code_assignments(**kwargs):
    return _build_v3_cost_code_assignments(selection_mode="uniform", **kwargs)


def build_v3_greedy_nokey_cost_code_assignments(**kwargs):
    return _build_v3_cost_code_assignments(selection_mode="greedy", **kwargs)


def build_v3_random_polarity_cost_code_assignments(**kwargs):
    return _build_v3_cost_code_assignments(selection_mode="random_polarity", **kwargs)


def _v3_channel_signal(
    params: Dict[str, Tensor],
    clean_params: Dict[str, Tensor],
    *,
    channel: str,
    gaussian_index: int,
    log_delta: float,
    alpha_delta: float,
    theta_delta: float,
    lum_delta: float,
) -> Tensor:
    idx = int(gaussian_index)
    canonical = _canonicalize_geometry(params["scales"], params["thetas"])
    clean_canonical = _canonicalize_geometry(clean_params["scales"], clean_params["thetas"])
    if channel == "log_anisotropy":
        diff = canonical["log_anisotropy"][idx] - clean_canonical["log_anisotropy"][idx].to(canonical["log_anisotropy"].device)
        return diff / max(log_delta, 1e-6)
    if channel == "alpha":
        diff = params["opacities"][idx, 0] - clean_params["opacities"][idx, 0].to(params["opacities"].device)
        return diff / max(alpha_delta, 1e-6)
    if channel == "theta":
        theta = canonical["theta_major"][idx, 0]
        clean_theta = clean_canonical["theta_major"][idx, 0].to(theta.device)
        diff = _wrap_theta_difference(theta - clean_theta)
        return diff / max(theta_delta, 1e-6)
    if channel == "color_lum":
        lum = _luminance(params["colors"])[idx]
        clean_lum = _luminance(clean_params["colors"].to(params["colors"].device))[idx]
        return (lum - clean_lum) / max(lum_delta, 1e-6)
    raise ValueError(f"unsupported v3 channel: {channel!r}")


def _cost_code_action_scores_v3(
    params: Dict[str, Tensor],
    *,
    clean_params: Dict[str, Tensor],
    assignments: List[Dict[str, object]],
    channel_weights: Dict[str, float],
    log_delta: float,
    alpha_delta: float,
    theta_delta: float,
    lum_delta: float,
) -> List[Tensor]:
    outputs: List[Tensor] = []
    for assignment in assignments:
        action_scores: List[Tensor] = []
        for action in assignment.get("candidate_actions", []):
            channel = str(action.get("channel", "log_anisotropy"))
            direction = float(action.get("direction", 1))
            gaussian_index = int(action.get("gaussian_index", 0))
            strength_scale = max(1e-3, float(action.get("strength_scale", 1.0)))
            signal = _v3_channel_signal(
                params,
                clean_params,
                channel=channel,
                gaussian_index=gaussian_index,
                log_delta=log_delta,
                alpha_delta=alpha_delta,
                theta_delta=theta_delta,
                lum_delta=lum_delta,
            )
            # Hard decisions should operate on the realized action magnitude,
            # not on reliability priors or region-dependent step scaling.
            action_scores.append(direction * (signal / strength_scale))
        if action_scores:
            outputs.append(torch.stack(action_scores))
        else:
            outputs.append(torch.empty(0, dtype=clean_params["offsets"].dtype, device=clean_params["offsets"].device))
    return outputs


def parity_block_logits_v3(
    params: Dict[str, Tensor],
    *,
    clean_params: Dict[str, Tensor],
    assignments: List[Dict[str, object]],
    channel_weights: Dict[str, float],
    log_delta: float,
    alpha_delta: float,
    theta_delta: float,
    lum_delta: float,
) -> Tensor:
    scores = _cost_code_action_scores_v3(
        params,
        clean_params=clean_params,
        assignments=assignments,
        channel_weights=channel_weights,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
        theta_delta=theta_delta,
        lum_delta=lum_delta,
    )
    logits: List[Tensor] = []
    for score_row, assignment in zip(scores, assignments):
        if score_row.numel() == 0:
            logits.append(torch.tensor(0.0, dtype=clean_params["offsets"].dtype, device=clean_params["offsets"].device))
            continue
        selected = torch.tensor(assignment["selected_mask"], dtype=score_row.dtype, device=score_row.device)
        logits.append((selected * score_row).sum())
    if not logits:
        return torch.empty(0, dtype=clean_params["offsets"].dtype, device=clean_params["offsets"].device)
    return torch.stack(logits)


def cost_code_block_probabilities_v3(
    params: Dict[str, Tensor],
    *,
    clean_params: Dict[str, Tensor],
    assignments: List[Dict[str, object]],
    channel_weights: Dict[str, float],
    log_delta: float,
    alpha_delta: float,
    theta_delta: float,
    lum_delta: float,
    beta: float = 4.0,
) -> tuple[Tensor, List[Tensor]]:
    action_scores = _cost_code_action_scores_v3(
        params,
        clean_params=clean_params,
        assignments=assignments,
        channel_weights=channel_weights,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
        theta_delta=theta_delta,
        lum_delta=lum_delta,
    )
    block_probabilities: List[Tensor] = []
    for scores, assignment in zip(action_scores, assignments):
        if scores.numel() == 0:
            block_probabilities.append(torch.full((4,), 0.5, dtype=clean_params["offsets"].dtype, device=clean_params["offsets"].device))
            continue
        action_probs = torch.sigmoid(beta * (scores - 0.5))
        parity_matrix = torch.tensor(assignment["parity_matrix"], dtype=torch.float32, device=scores.device).t()
        bit_probs: List[Tensor] = []
        for row in range(parity_matrix.shape[0]):
            row_mask = parity_matrix[row] > 0.5
            if int(row_mask.sum().item()) == 0:
                bit_probs.append(torch.tensor(0.5, dtype=scores.dtype, device=scores.device))
                continue
            selected_probs = action_probs[row_mask]
            parity_prob = 0.5 - 0.5 * torch.prod(1.0 - 2.0 * selected_probs)
            if int(_assignment_whitening_bits(assignment, device=scores.device)[row].item()) == 1:
                parity_prob = 1.0 - parity_prob
            bit_probs.append(parity_prob)
        block_probabilities.append(torch.stack(bit_probs))
    if not block_probabilities:
        return torch.empty((0, 4), dtype=torch.float32, device=clean_params["offsets"].device), action_scores
    return torch.stack(block_probabilities, dim=0), action_scores


def decode_cost_code_v3(
    params: Dict[str, Tensor],
    *,
    clean_params: Dict[str, Tensor],
    assignments: List[Dict[str, object]],
    raw_length: int,
    channel_weights: Dict[str, float],
    log_delta: float,
    alpha_delta: float,
    theta_delta: float,
    lum_delta: float,
    beta: float = 4.0,
) -> tuple[Tensor, List[List[float]], List[List[float]]]:
    block_probs, action_scores = cost_code_block_probabilities_v3(
        params,
        clean_params=clean_params,
        assignments=assignments,
        channel_weights=channel_weights,
        log_delta=log_delta,
        alpha_delta=alpha_delta,
        theta_delta=theta_delta,
        lum_delta=lum_delta,
        beta=beta,
    )
    if block_probs.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=clean_params["offsets"].device), [], []
    hard_blocks: List[Tensor] = []
    for assignment, scores in zip(assignments, action_scores):
        whitening_bits = _assignment_whitening_bits(assignment, device=block_probs.device)
        if scores.numel() == 0:
            hard_blocks.append(whitening_bits)
            continue
        parity_matrix = torch.tensor(assignment["parity_matrix"], dtype=torch.float32, device=scores.device).t()
        hard_actions = (scores > 0.5).to(torch.float32)
        hard_bits = torch.remainder((parity_matrix @ hard_actions).round().to(torch.int64), 2)
        hard_bits = torch.remainder(hard_bits + whitening_bits, 2)
        hard_blocks.append(hard_bits)
    decoded = torch.cat(hard_blocks, dim=0)[:raw_length]
    return (
        decoded,
        [[float(value) for value in row.tolist()] for row in block_probs.detach().cpu()],
        [[float(value) for value in row.detach().cpu().tolist()] for row in action_scores],
    )


def prune_assignments_by_snr(
    assignments: List[Dict],
    *,
    snr_scores: List[float],
    keep_fraction: float,
    beam_width: int,
) -> List[Dict]:
    if not assignments or keep_fraction >= 0.999:
        return _augment_assignment_channel_views(assignments)
    block_count = len(assignments)
    keep_top_k = max(1, int(math.ceil(block_count * keep_fraction)))
    ranked = sorted(range(block_count), key=lambda idx: snr_scores[idx], reverse=True)
    keep_full = set(ranked[:keep_top_k])
    pruned: List[Dict] = []
    for block_index, assignment in enumerate(assignments):
        if block_index in keep_full:
            pruned.append(dict(assignment))
            continue
        candidate_actions = list(assignment.get("candidate_actions", []))
        candidate_costs = list(assignment.get("candidate_costs", []))
        if len(candidate_actions) <= 4:
            pruned.append(dict(assignment))
            continue
        keep_actions = max(4, int(math.ceil(len(candidate_actions) * keep_fraction)))
        order = sorted(range(len(candidate_actions)), key=lambda idx: float(candidate_costs[idx]))
        kept_indices = order[:keep_actions]
        new_actions = [candidate_actions[idx] for idx in kept_indices]
        new_costs = [candidate_costs[idx] for idx in kept_indices]
        new_actions, new_costs = _canonicalize_v3_candidate_block(new_actions, new_costs)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(key_to_seed(f"v3_snr_prune::{block_index}") + 401)
        parity_matrix = _make_parity_matrix(len(new_actions), generator=generator)
        try:
            selected_mask = _solve_cost_code_block(
                candidate_costs=torch.tensor(new_costs, dtype=torch.float32),
                parity_matrix=parity_matrix,
                target_bits=_assignment_encoded_message_bits(assignment),
                beam_width=beam_width,
            )
            new_assignment = dict(assignment)
            new_assignment["candidate_actions"] = new_actions
            new_assignment["candidate_costs"] = new_costs
            new_assignment["parity_matrix"] = parity_matrix.t().tolist()
            new_assignment["selected_mask"] = [float(value) for value in selected_mask.tolist()]
            new_assignment["selected_count"] = int((selected_mask > 0.5).sum().item())
            new_assignment["snr_pruned"] = True
            pruned.append(new_assignment)
        except RuntimeError:
            fallback = dict(assignment)
            fallback["snr_pruned"] = False
            pruned.append(fallback)
    return _augment_assignment_channel_views(pruned)


# ============================================================================
# Innovation 3 – Sinkhorn OT Covertness Term
# ============================================================================

def _sinkhorn_divergence_pytorch(
    x: Tensor,
    y: Tensor,
    blur: float = 0.05,
    n_iter: int = 50,
) -> Tensor:
    """Pure-PyTorch Sinkhorn divergence S_ε(x, y) = W_ε(x,y) - ½W_ε(x,x) - ½W_ε(y,y).

    Uses the log-domain Sinkhorn algorithm for numerical stability.

    Parameters
    ----------
    x : (N, D) — watermarked carrier samples
    y : (M, D) — clean carrier samples
    blur : regularisation ε (larger = smoother, less accurate OT)
    n_iter : Sinkhorn iterations

    Returns
    -------
    Scalar Sinkhorn divergence (differentiable w.r.t. x).
    """
    eps = blur ** 2

    def _cost_matrix(a: Tensor, b: Tensor) -> Tensor:
        # Squared Euclidean distance (N×M)
        return torch.cdist(a, b, p=2).pow(2)

    def _sinkhorn_internal(a: Tensor, b: Tensor) -> Tensor:
        N, M = a.shape[0], b.shape[0]
        C = _cost_matrix(a, b) / eps  # (N, M)
        log_u = torch.zeros(N, device=a.device, dtype=a.dtype)
        log_v = torch.zeros(M, device=b.device, dtype=b.dtype)
        log_p = -C  # (N, M)
        for _ in range(n_iter):
            log_u = -torch.logsumexp(log_p + log_v[None, :], dim=1)
            log_v = -torch.logsumexp(log_p + log_u[:, None], dim=0)
        return eps * (torch.logsumexp(log_p + log_v[None, :] + log_u[:, None], dim=(0, 1)) - math.log(N * M))

    W_xy = _sinkhorn_internal(x, y)
    W_xx = _sinkhorn_internal(x, x.detach())
    W_yy = _sinkhorn_internal(y, y)
    return W_xy - 0.5 * W_xx - 0.5 * W_yy


class SinkhornCovertnessTerm:
    """Sinkhorn-divergence covertness regulariser for v3.0 training.

    Minimising this term pushes the distribution of *modified* carrier
    parameters toward the clean cover distribution, making watermarked
    Gaussians statistically indistinguishable from unwatermarked ones.

    References
    ----------
    • Feydy et al., "Interpolating between Optimal Transport and MMD using
      Sinkhorn Divergences", AISTATS 2019.
    • Inspired by HILL (Li et al., 2014): embed where parameter statistics
      are hard to distinguish from the cover distribution.

    Usage
    -----
    term = SinkhornCovertnessTerm(blur=0.05, weight=0.15)
    loss = term(clean_carriers, wm_carriers)   # scalar Tensor
    """

    def __init__(
        self,
        blur: float = 0.05,
        weight: float = 0.15,
        n_iter: int = 50,
        max_carriers: int = 512,
        normalise_channels: bool = True,
    ) -> None:
        self.blur = blur
        self.weight = weight
        self.n_iter = n_iter
        self.max_carriers = max_carriers
        self.normalise_channels = normalise_channels
        self._use_geomloss = self._check_geomloss()

    @staticmethod
    def _check_geomloss() -> bool:
        try:
            import geomloss  # noqa: F401
            return True
        except ImportError:
            return False

    def _normalise(self, x: Tensor, ref_mean: Tensor, ref_std: Tensor) -> Tensor:
        return (x - ref_mean) / ref_std.clamp_min(1e-6)

    def forward(
        self,
        clean_carriers: Tensor,
        wm_carriers: Tensor,
    ) -> Tensor:
        """Compute Sinkhorn divergence between watermarked and clean carriers.

        Parameters
        ----------
        clean_carriers : (N, D)  — clean parameter values of selected carriers
        wm_carriers    : (N, D)  — watermarked parameter values (requires_grad)

        Returns
        -------
        Scalar loss (Tensor, differentiable).
        """
        if clean_carriers.shape[0] < 4:
            return torch.zeros(1, device=clean_carriers.device).squeeze()

        # Per-channel normalisation to balance scale differences
        if self.normalise_channels:
            ref_mean = clean_carriers.mean(dim=0).detach()
            ref_std  = clean_carriers.std(dim=0, unbiased=False).detach()
            x = self._normalise(wm_carriers,   ref_mean, ref_std)
            y = self._normalise(clean_carriers, ref_mean, ref_std)
        else:
            x, y = wm_carriers, clean_carriers

        # Subsample if too many carriers (for speed)
        if x.shape[0] > self.max_carriers:
            idx = torch.randperm(x.shape[0], device=x.device)[: self.max_carriers]
            x = x[idx]
            y = y[: self.max_carriers]

        if self._use_geomloss:
            try:
                import geomloss
                loss_fn = geomloss.SamplesLoss("sinkhorn", blur=self.blur, scaling=0.5)
                return self.weight * loss_fn(x.unsqueeze(0), y.detach().unsqueeze(0)).squeeze()
            except Exception:
                pass  # fall through to PyTorch implementation

        return self.weight * _sinkhorn_divergence_pytorch(x, y.detach(), blur=self.blur, n_iter=self.n_iter)

    def __call__(self, clean_carriers: Tensor, wm_carriers: Tensor) -> Tensor:
        return self.forward(clean_carriers, wm_carriers)


def extract_modified_carriers(
    clean_params: Dict[str, Tensor],
    wm_params: Dict[str, Tensor],
    v3_bundle: V3ChannelCostBundle,
) -> Tuple[Tensor, Tensor]:
    """Extract (clean, watermarked) carrier feature matrices for Sinkhorn loss.

    Collects the 4-channel carrier values for Gaussians that were actually
    modified (i.e., not in wet_mask and within the active selection).

    Returns
    -------
    clean_feat : (K, 4)
    wm_feat    : (K, 4)
    """
    canonical_c = _canonicalize_geometry(clean_params["scales"], clean_params["thetas"])
    canonical_w = _canonicalize_geometry(wm_params["scales"],   wm_params["thetas"])
    alpha_c = clean_params["opacities"][:, 0]
    alpha_w = wm_params["opacities"][:, 0]

    # Log anisotropy
    log_c = canonical_c["log_anisotropy"]
    log_w = canonical_w["log_anisotropy"]

    # Theta major
    theta_c = canonical_c["theta_major"][:, 0]
    theta_w = canonical_w["theta_major"][:, 0]

    # Color luminance
    if "colors" in clean_params and "colors" in wm_params:
        luma_w_vec = _LUMA_W.to(clean_params["colors"].device)
        lum_c = (clean_params["colors"] * luma_w_vec).sum(dim=1)
        lum_w = (wm_params["colors"]   * luma_w_vec).sum(dim=1)
    else:
        lum_c = alpha_c.clone()
        lum_w = alpha_w.clone()

    active = ~v3_bundle.wet_mask
    clean_feat = torch.stack([log_c[active], alpha_c[active], theta_c[active], lum_c[active]], dim=1)
    wm_feat    = torch.stack([log_w[active], alpha_w[active], theta_w[active], lum_w[active]], dim=1)
    return clean_feat, wm_feat


# ============================================================================
# Innovation 1 – Perturbation-Hardened Decoder
# ============================================================================

class PerturbationHardenedDecoder:
    """Noisy-decode loss for perturbation-hardened training.

    Simulates the effect of small parameter perturbations (quantisation
    noise, refit jitter) by adding Gaussian noise to the embedded parameter
    values before decoding, then penalising block probabilities below a
    fixed one-sided floor.

    References
    ----------
    • CompMarkGS (arXiv 2503.12836): quantisation distortion layer in training.
    • Certified watermarking via randomised smoothing (ICML 2022).

    Training schedule (3-stage, see watermark_experiment_v3.py)
    ----------------------------------------------------------
    Stage A  (iter  0..start_iter):       sigma_ratio = 0.0  (normal embedding)
    Stage B  (iter  start..end_iter):     sigma_ratio = sigma_ratio_b
    Stage C  (iter  end_iter..total):     sigma_ratio = sigma_ratio_c
    """

    def __init__(
        self,
        sigma_ratio_b: float = 0.30,
        sigma_ratio_c: float = 0.50,
        start_iter: int = 100,
        end_iter: int = 180,
        min_block_prob: float = 0.55,
    ) -> None:
        self.sigma_ratio_b = sigma_ratio_b
        self.sigma_ratio_c = sigma_ratio_c
        self.start_iter    = start_iter
        self.end_iter      = end_iter
        self.min_block_prob = min_block_prob

    def _sigma_at(self, current_iter: int, embedding_delta: float) -> float:
        """Return the noise standard deviation for the current training iteration."""
        if current_iter < self.start_iter:
            return 0.0
        elif current_iter < self.end_iter:
            return self.sigma_ratio_b * embedding_delta
        else:
            return self.sigma_ratio_c * embedding_delta

    def sigma_schedule(
        self,
        *,
        current_iter: int,
        log_delta: float,
        alpha_delta: float,
        theta_delta: float,
        lum_delta: float,
    ) -> Dict[str, float]:
        return {
            "log_anisotropy": self._sigma_at(current_iter, log_delta),
            "alpha": self._sigma_at(current_iter, alpha_delta),
            "theta": self._sigma_at(current_iter, theta_delta),
            "color_lum": self._sigma_at(current_iter, lum_delta),
        }

    def hardened_loss(
        self,
        wm_log_anisotropy: Tensor,
        wm_alpha: Tensor,
        assignments: List[Dict],
        embedding_delta: float,
        current_iter: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Compute the hardened decode loss.

        Adds Gaussian noise to the two main carrier channels and recomputes
        block probabilities, then penalises any probability below
        `min_block_prob`.

        Parameters
        ----------
        wm_log_anisotropy : (N,) — current embedded log-anisotropy values
        wm_alpha          : (N,) — current embedded alpha values
        assignments       : list of block assignment dicts (from v2.0 protocol)
        embedding_delta   : the per-channel embedding magnitude
        current_iter      : current training step (0-indexed)

        Returns
        -------
        Scalar Tensor (hardened decode loss, ≥ 0).
        """
        sigma = self._sigma_at(current_iter, embedding_delta)
        if sigma <= 0.0:
            return torch.zeros(1, device=wm_log_anisotropy.device).squeeze()

        noise_log   = torch.randn_like(wm_log_anisotropy) * sigma
        noise_alpha = torch.randn_like(wm_alpha)           * (sigma * 0.5)  # alpha smaller scale

        noisy_log   = wm_log_anisotropy + noise_log
        noisy_alpha = wm_alpha          + noise_alpha

        # Recompute block probabilities with noisy parameters
        # Uses the same parity_block_logits function from v2.0
        try:
            noisy_logits = parity_block_logits(
                assignments,
                log_anisotropy=noisy_log,
                alpha=noisy_alpha,
                log_delta=embedding_delta,
                alpha_delta=embedding_delta * 0.33,
            )
        except Exception:
            return torch.zeros(1, device=wm_log_anisotropy.device).squeeze()

        # Implemented legacy objective: apply a one-sided probability floor.
        noisy_probs = torch.sigmoid(noisy_logits)
        penalties = F.relu(self.min_block_prob - noisy_probs)
        return penalties.mean()

    def hardened_loss_v3(
        self,
        *,
        params: Dict[str, Tensor],
        clean_params: Dict[str, Tensor],
        assignments: List[Dict],
        channel_weights: Dict[str, float],
        log_delta: float,
        alpha_delta: float,
        theta_delta: float,
        lum_delta: float,
        current_iter: int,
    ) -> Tensor:
        sigma = self.sigma_schedule(
            current_iter=current_iter,
            log_delta=log_delta,
            alpha_delta=alpha_delta,
            theta_delta=theta_delta,
            lum_delta=lum_delta,
        )
        if max(sigma.values()) <= 0.0:
            return torch.zeros(1, device=params["offsets"].device).squeeze()
        noisy = {key: value.clone() for key, value in params.items()}
        if sigma["log_anisotropy"] > 0.0:
            log_noise = torch.randn_like(noisy["scales"]) * sigma["log_anisotropy"]
            noisy["scales"] = torch.clamp(noisy["scales"] + log_noise, min=1e-6)
        if sigma["alpha"] > 0.0:
            alpha_noise = torch.randn_like(noisy["opacities"]) * sigma["alpha"]
            noisy["opacities"] = torch.clamp(noisy["opacities"] + alpha_noise, 0.0, 1.0)
        if sigma["theta"] > 0.0:
            theta_noise = torch.randn_like(noisy["thetas"]) * sigma["theta"]
            noisy["thetas"] = noisy["thetas"] + theta_noise
        if sigma["color_lum"] > 0.0 and "colors" in noisy:
            color_noise = torch.randn_like(noisy["colors"]) * sigma["color_lum"]
            noisy["colors"] = torch.clamp(noisy["colors"] + color_noise, 0.0, 1.0)
        noisy_probs, _ = cost_code_block_probabilities_v3(
            noisy,
            clean_params=clean_params,
            assignments=assignments,
            channel_weights=channel_weights,
            log_delta=log_delta,
            alpha_delta=alpha_delta,
            theta_delta=theta_delta,
            lum_delta=lum_delta,
            beta=4.0,
        )
        penalties = F.relu(self.min_block_prob - noisy_probs)
        return penalties.mean()


# ============================================================================
# Innovation 4 – Sensitivity-Adaptive Variable-Rate Encoder
# ============================================================================

class SensitivityAdaptiveEncoder:
    """Variable-rate Hamming encoder guided by per-carrier SNR estimates.

    Rather than applying a uniform Hamming(7,4) code to all payload bits,
    this encoder estimates the reliability of each bit's carrier assignment
    and selects a matching Hamming block size:

        SNR > high_threshold  →  Hamming(4,3)   rate = 3/4  (fewer parity bits)
        SNR ∈ [low, high]     →  Hamming(7,4)   rate = 4/7  (v2.0 default)
        SNR < low_threshold   →  repetition×3   rate = 1/3  (more redundancy)

    SNR per bit is estimated as (embedding_delta)² / mean_cost(assigned carriers).

    References
    ----------
    • Filler, Judas, Fridrich: "Minimizing embedding impact in steganography
      using trellis-coded quantization", IEEE 2010.
    • Variable-Rate STC: Lerch-Hosemann et al., Springer 2021.
    """

    def __init__(
        self,
        snr_high_threshold: float = 0.8,
        snr_low_threshold:  float = 0.4,
    ) -> None:
        self.snr_high = snr_high_threshold
        self.snr_low  = snr_low_threshold

    def estimate_block_snr(
        self,
        assignments: List[Dict],
        *,
        channel_weights: Optional[Dict[str, float]] = None,
        delta_map: Optional[Dict[str, float]] = None,
        embedding_delta: Optional[float] = None,
    ) -> List[float]:
        snr_list: List[float] = []
        if delta_map is None:
            base_delta = float(embedding_delta if embedding_delta is not None else 0.12)
            delta_map = {
                "log_anisotropy": base_delta,
                "alpha": base_delta * 0.33,
                "theta": base_delta,
                "color_lum": base_delta * 0.067,
            }
        channel_weights = dict(channel_weights or {})
        for asgn in assignments:
            per_action = []
            for action, selected in zip(asgn.get("candidate_actions", []), asgn.get("selected_mask", [])):
                if float(selected) <= 0.5:
                    continue
                channel = str(action.get("channel", "log_anisotropy"))
                delta = float(delta_map.get(channel, float(embedding_delta or 0.12)))
                weight = float(action.get("channel_weight", channel_weights.get(channel, 1.0)))
                cost = float(action.get("channel_cost", 1.0))
                per_action.append(((weight * delta) ** 2) / (cost + 1e-6))
            if not per_action:
                snr_list.append(0.0)
                continue
            snr_list.append(float(sum(per_action) / len(per_action)))
        return snr_list

    def estimate_per_bit_snr(
        self,
        assignments: List[Dict],
        embedding_delta: float,
    ) -> List[float]:
        return self.estimate_block_snr(assignments, embedding_delta=embedding_delta)

    def recommended_block_sizes(
        self,
        assignments: List[Dict],
        embedding_delta: float,
    ) -> List[int]:
        """Return recommended Hamming block size for each payload bit.

        Returns
        -------
        List of ints: 4 (high-rate), 7 (standard), or 15 (high-redundancy).
        """
        snr_list = self.estimate_per_bit_snr(assignments, embedding_delta)
        block_sizes = []
        for snr in snr_list:
            if snr >= self.snr_high:
                block_sizes.append(4)   # Hamming(4,3) – rate 3/4
            elif snr >= self.snr_low:
                block_sizes.append(7)   # Hamming(7,4) – rate 4/7 (v2.0 default)
            else:
                block_sizes.append(15)  # Repeat×3     – rate 1/3
        return block_sizes

    def ranked_block_indices(
        self,
        assignments: List[Dict],
        *,
        channel_weights: Optional[Dict[str, float]] = None,
        delta_map: Optional[Dict[str, float]] = None,
        embedding_delta: Optional[float] = None,
    ) -> List[int]:
        snr_scores = self.estimate_block_snr(
            assignments,
            channel_weights=channel_weights,
            delta_map=delta_map,
            embedding_delta=embedding_delta,
        )
        return sorted(range(len(assignments)), key=lambda idx: snr_scores[idx], reverse=True)

    def adaptive_coded_bit_count(
        self,
        payload_bits: int,
        assignments: List[Dict],
        embedding_delta: float,
    ) -> int:
        """Estimate total coded bits needed for the payload under adaptive coding."""
        block_sizes = self.recommended_block_sizes(assignments[:payload_bits], embedding_delta)
        return sum(block_sizes)


# ============================================================================
# Innovation 5 – Pre-Embedding Capacity Probe
# ============================================================================

def estimate_embedding_capacity(
    v3_bundle: V3ChannelCostBundle,
    *,
    target_ber: float = 0.05,
    target_psnr_drop_db: float = 3.0,
    coding_rate: float = 4.0 / 7.0,
    cost_threshold: float = 0.50,
    safety_factor: float = 0.60,
) -> int:
    """Estimate how many payload bits can be reliably embedded in this cover.

    Performs a lightweight analysis of the carrier cost distribution without
    doing any actual embedding.  The recommendation is conservative (uses
    safety_factor < 1.0) to leave headroom for the parity overhead.

    Parameters
    ----------
    v3_bundle          : V3ChannelCostBundle from build_v3_channel_costs()
    target_ber         : maximum acceptable decoded BER (default 0.05)
    target_psnr_drop_db: maximum acceptable PSNR drop in dB
    coding_rate        : effective coding rate (4/7 for Hamming(7,4))
    cost_threshold     : maximum carrier cost to count as "reliable"
    safety_factor      : apply this factor to the raw capacity estimate

    Returns
    -------
    Recommended payload bits as one of {4, 8, 12, 16}.
    """
    n_reliable = v3_bundle.n_carriers_below_cost_threshold(cost_threshold)
    raw_capacity = int(n_reliable * coding_rate * safety_factor)

    # Map to nearest supported payload size
    if raw_capacity >= 16:
        return 16
    elif raw_capacity >= 12:
        return 12
    elif raw_capacity >= 8:
        return 8
    elif raw_capacity >= 4:
        return 4
    else:
        return 4  # minimum even if cover is marginal


# ============================================================================
# Colour luminance embedding helpers
# ============================================================================

def embed_luminance_bit(
    colors: Tensor,
    indices: Tensor,
    weights: Tensor,
    polarity: int,
    lum_delta: float,
) -> Tensor:
    """Modify the luminance of selected Gaussians to embed one bit.

    Only the Y (luma) component is modified; Cb and Cr remain unchanged,
    which means the hue and saturation of the Gaussian colours are preserved.

    Parameters
    ----------
    colors   : (N, 3) RGB colour tensor (requires_grad for training)
    indices  : (K,) selected Gaussian indices
    weights  : (K,) assignment weights (used for fractional embedding)
    polarity : ±1 — direction of luminance modification
    lum_delta: per-unit luminance step

    Returns
    -------
    Modified (N, 3) colour tensor.
    """
    luma_w = _LUMA_W.to(colors.device)
    colors_new = colors.clone()
    for idx_i, w_i in zip(indices.tolist(), weights.tolist()):
        rgb_i = colors_new[idx_i]          # (3,)
        lum_i = (rgb_i * luma_w).sum()    # scalar luma
        delta  = float(polarity) * lum_delta * float(w_i)
        # Shift luma only (project delta back onto the YCbCr Y axis)
        # rgb_new = rgb + delta * luma_w / ||luma_w||²
        luma_w_sq = (luma_w * luma_w).sum().clamp_min(1e-9)
        colors_new[idx_i] = (rgb_i + delta * luma_w / luma_w_sq).clamp(0.0, 1.0)
    return colors_new


def decode_luminance_channel(
    clean_colors: Tensor,
    wm_colors: Tensor,
    indices: Tensor,
    weights: Tensor,
    lum_delta: float,
) -> float:
    """Decode a luminance-channel bit from the difference of luma values.

    Returns the raw action score in [−1, +1]; sign → decoded bit.
    """
    luma_w = _LUMA_W.to(clean_colors.device)
    score = 0.0
    total_w = 0.0
    for idx_i, w_i in zip(indices.tolist(), weights.tolist()):
        diff_lum = ((wm_colors[idx_i] - clean_colors[idx_i]) * luma_w).sum().item()
        score   += w_i * diff_lum / (lum_delta + 1e-9)
        total_w += w_i
    return score / max(total_w, 1e-6)


# ============================================================================
# Convenience: per-channel BER computation for the new channels
# ============================================================================

def theta_channel_ber(
    clean_thetas: Tensor,
    wm_thetas: Tensor,
    assignments_theta: List[Dict],
) -> float:
    """Compute BER for the theta channel specifically.

    For each assigned bit, derive the detected polarity from the theta delta
    and compare with the embedded bit.
    """
    if not assignments_theta:
        return float("nan")
    correct = 0
    total   = 0
    for asgn in assignments_theta:
        indices = asgn.get("theta_indices", [])
        weights = asgn.get("theta_weights", [])
        if not indices:
            continue
        score = 0.0
        wt    = 0.0
        for idx_i, w_i in zip(indices, weights):
            score += w_i * float((wm_thetas[idx_i] - clean_thetas[idx_i]).item())
            wt    += w_i
        detected = 1 if (score / max(wt, 1e-6)) > 0 else 0
        embedded = asgn.get("bit", 0)
        correct += int(detected == embedded)
        total   += 1
    return 1.0 - correct / max(total, 1) if total > 0 else float("nan")


def color_lum_channel_ber(
    clean_colors: Tensor,
    wm_colors: Tensor,
    assignments_lum: List[Dict],
    lum_delta: float,
) -> float:
    """Compute BER for the color luminance channel specifically."""
    if not assignments_lum:
        return float("nan")
    correct = 0
    total   = 0
    for asgn in assignments_lum:
        indices = torch.tensor(asgn.get("lum_indices", []), dtype=torch.long)
        weights = torch.tensor(asgn.get("lum_weights", []), dtype=torch.float32)
        if indices.numel() == 0:
            continue
        score = decode_luminance_channel(clean_colors, wm_colors, indices, weights, lum_delta)
        detected = 1 if score > 0 else 0
        embedded = asgn.get("bit", 0)
        correct += int(detected == embedded)
        total   += 1
    return 1.0 - correct / max(total, 1) if total > 0 else float("nan")


# ============================================================================
# V3 default hyperparameters
# ============================================================================

def cost_stego_v3_defaults() -> Dict:
    """Return default hyperparameters for the v3.0 embedding protocol.

    Extends v2.0 defaults with new v3-specific parameters.
    """
    return {
        # ── Carrier channels ────────────────────────────────────────────────
        "carrier_channels": ["log_anisotropy", "alpha", "theta", "color_lum"],

        # ── Embedding deltas ────────────────────────────────────────────────
        "log_delta":   0.12,
        "alpha_delta": 0.04,
        "theta_delta": 0.05,    # radians
        "lum_delta":   0.008,   # ≈2/255 pixel luminance

        # ── Cost weights (same as v2.0 for Ch1 & Ch2) ───────────────────────
        "cost_sensitivity_w": 0.40,
        "cost_density_w":     0.30,
        "cost_visual_w":      0.20,
        "cost_detector_w":    0.10,

        # ── Innovation 1: Perturbation hardening ────────────────────────────
        "hardening_sigma_ratio_b": 0.30,
        "hardening_sigma_ratio_c": 0.50,
        "hardening_start_iter":    100,
        "hardening_end_iter":      180,

        # ── Innovation 3: Sinkhorn covertness ───────────────────────────────
        "ot_covertness_weight": 0.15,   # 3× stronger than v2.0 wasserstein_weight
        "sinkhorn_blur":        0.05,

        # ── Innovation 4: Adaptive coding ───────────────────────────────────
        "adaptive_coding": True,
        "snr_high_threshold": 0.8,
        "snr_low_threshold":  0.4,

        # ── Payload ─────────────────────────────────────────────────────────
        "auto_payload_bits": False,  # if True, use estimate_embedding_capacity()
        "default_payload_bits": 8,
    }
