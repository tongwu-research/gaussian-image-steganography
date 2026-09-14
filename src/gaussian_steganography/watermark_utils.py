from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from gaussian2d import DEFAULT_SCALE_EPS, Gaussian2DRenderer
from run_fit import ExperimentConfig, compute_psnr, compute_saturation_statistics
from utils import (
    canonicalize_gaussian_geometry,
    compute_gaussian_importance,
    compute_parameter_diagnostics,
    compute_parameter_sensitivity_summary,
    create_grid,
    load_target_image,
    render_gaussians_from_parameters,
)

LOG_ANISOTROPY_MODE = "log_anisotropy_bin"
LEGACY_LOGRATIO_MODE = "logratio_bin"
THETA_MAJOR_MODE = "theta_bin"


@dataclass
class ExportedRun:
    run_dir: Path
    config: dict[str, object]
    metrics: dict[str, object]
    offsets: Tensor
    scales: Tensor
    thetas: Tensor
    colors: Tensor
    opacities: Tensor
    importance: Tensor
    canonical: dict[str, Tensor]
    sensitivity: dict[str, Tensor] | None


def resolve_run_dir(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser().resolve()
    if path.is_dir():
        return path
    raise FileNotFoundError(f"expected a run directory, got: {path_like}")


def _safe_corr(x: Tensor, y: Tensor) -> Tensor:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt(torch.sum(x_centered * x_centered) * torch.sum(y_centered * y_centered)).clamp_min(1e-12)
    return torch.sum(x_centered * y_centered) / denom


def key_to_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_to_seed(key: str) -> int:
    return int(key_to_hash(key)[:16], 16) % (2**31 - 1)


def _strict_scale_floor(reference: Tensor) -> float:
    floor = torch.tensor(DEFAULT_SCALE_EPS, dtype=reference.dtype, device=reference.device)
    strict_floor = torch.nextafter(
        floor,
        torch.tensor(float("inf"), dtype=reference.dtype, device=reference.device),
    )
    return float(strict_floor.item())


def clamp_scales_strictly_above_eps(scales: Tensor) -> Tensor:
    return torch.clamp(scales, min=_strict_scale_floor(scales))


def sanitize_renderer_params(params: dict[str, Tensor]) -> dict[str, Tensor]:
    sanitized = {key: value.detach().clone() for key, value in params.items() if key != "importance"}
    sanitized["scales"] = clamp_scales_strictly_above_eps(sanitized["scales"])
    sanitized["colors"] = torch.clamp(sanitized["colors"], 0.0, 1.0)
    sanitized["opacities"] = torch.clamp(sanitized["opacities"], 0.0, 1.0)
    sanitized["importance"] = compute_gaussian_importance(sanitized["scales"], sanitized["opacities"])
    return sanitized


def normalize_sensitivity_payload(sensitivity: dict[str, Tensor] | None) -> dict[str, Tensor] | None:
    if sensitivity is None:
        return None
    log_anisotropy = sensitivity.get("logratio")
    if log_anisotropy is None:
        log_anisotropy = sensitivity.get("log_anisotropy")
    if log_anisotropy is None:
        raise KeyError("sensitivity payload is missing the log-anisotropy sensitivity field")
    return {
        "mu_x": sensitivity["mu_x"],
        "mu_y": sensitivity["mu_y"],
        "log_anisotropy": log_anisotropy,
        "log_area": sensitivity["log_area"],
        "theta_major": sensitivity["theta_major"],
        "alpha": sensitivity["alpha"],
    }


def carrier_sensitivity_field(mode: str) -> str:
    if mode in {LOG_ANISOTROPY_MODE, LEGACY_LOGRATIO_MODE}:
        return "log_anisotropy"
    if mode == THETA_MAJOR_MODE:
        return "theta_major"
    raise ValueError(f"unsupported watermark mode: {mode}")


def mode_uses_scale_carrier(mode: str) -> bool:
    return mode in {LOG_ANISOTROPY_MODE, LEGACY_LOGRATIO_MODE}


def carrier_values_from_canonical(canonical: dict[str, Tensor], *, mode: str) -> Tensor:
    if mode == LOG_ANISOTROPY_MODE:
        return canonical["log_anisotropy"]
    if mode == LEGACY_LOGRATIO_MODE:
        return canonical["signed_logratio"]
    if mode == THETA_MAJOR_MODE:
        return canonical["theta_major"][:, 0]
    raise ValueError(f"unsupported watermark mode: {mode}")


def carrier_values_from_params(params: dict[str, Tensor], *, mode: str) -> Tensor:
    canonical = canonicalize_gaussian_geometry(params["scales"], params["thetas"])
    return carrier_values_from_canonical(canonical, mode=mode)


def _theta_major_distance(values: Tensor, target: float) -> Tensor:
    delta = values - target
    return 0.5 * torch.abs(torch.atan2(torch.sin(2.0 * delta), torch.cos(2.0 * delta)))


def _theta_major_distance_scalar(value: float, target: float) -> float:
    delta = value - target
    return 0.5 * abs(math.atan2(math.sin(2.0 * delta), math.cos(2.0 * delta)))


def carrier_target_centers(mode: str, *, target_magnitude: float) -> tuple[float, float]:
    if mode == LOG_ANISOTROPY_MODE:
        return 0.5 * target_magnitude, 1.5 * target_magnitude
    if mode == LEGACY_LOGRATIO_MODE:
        return -target_magnitude, target_magnitude
    if mode == THETA_MAJOR_MODE:
        return math.pi / 4.0, 3.0 * math.pi / 4.0
    raise ValueError(f"unsupported watermark mode: {mode}")


def carrier_distance(values: Tensor, *, target: float | Tensor, mode: str) -> Tensor:
    if mode == THETA_MAJOR_MODE:
        return _theta_major_distance(values, target)
    return torch.abs(values - target)


def carrier_stability_weights(values: Tensor, *, mode: str, target_magnitude: float) -> Tensor:
    low_target, high_target = carrier_target_centers(mode, target_magnitude=target_magnitude)
    nearest = torch.minimum(
        carrier_distance(values, target=low_target, mode=mode),
        carrier_distance(values, target=high_target, mode=mode),
    )
    return 1.0 / (0.05 + nearest)


def load_exported_run(path_like: str | Path) -> ExportedRun:
    run_dir = resolve_run_dir(path_like)
    export_path = run_dir / "data" / "learned_params.pt"
    if not export_path.is_file():
        raise FileNotFoundError(f"missing exported parameter file: {export_path}")
    payload = torch.load(export_path, map_location="cpu")
    params = payload["parameters"]
    canonical = payload.get("canonical_parameters")
    if canonical is None:
        canonical = canonicalize_gaussian_geometry(params["scales"], params["thetas"])
    sensitivity = normalize_sensitivity_payload(payload.get("sensitivity"))
    return ExportedRun(
        run_dir=run_dir,
        config=dict(payload.get("config", {})),
        metrics=dict(payload.get("metrics", {})),
        offsets=params["offsets"].detach().cpu(),
        scales=params["scales"].detach().cpu(),
        thetas=params["thetas"].detach().cpu(),
        colors=params["colors"].detach().cpu(),
        opacities=params["opacities"].detach().cpu(),
        importance=params["importance"].detach().cpu(),
        canonical={key: value.detach().cpu() for key, value in canonical.items()},
        sensitivity=None if sensitivity is None else {key: value.detach().cpu() for key, value in sensitivity.items()},
    )


def build_renderer_from_export(run: ExportedRun, device: torch.device) -> Gaussian2DRenderer:
    renderer = Gaussian2DRenderer(num_gaussians=run.offsets.shape[0]).to(device)
    renderer.set_physical_parameters(
        offsets=run.offsets.to(device),
        scales=run.scales.to(device),
        thetas=run.thetas.to(device),
        colors=run.colors.to(device),
        opacities=run.opacities.to(device),
    )
    return renderer


def load_target_tensor(run: ExportedRun, device: torch.device) -> Tensor:
    target_image_path = run.run_dir / "target_image.png"
    if not target_image_path.is_file():
        raise FileNotFoundError(f"missing target image export: {target_image_path}")
    height = int(run.config["height"])
    width = int(run.config["width"])
    return load_target_image(target_image_path, height=height, width=width, device=device)


def build_bits(payload_bits: int, message: str | None = None, seed: int = 0) -> Tensor:
    if message is not None:
        if any(character not in {"0", "1"} for character in message):
            raise ValueError("message must contain only '0' and '1'")
        if len(message) != payload_bits:
            raise ValueError("message length must match payload_bits")
        return torch.tensor([int(character) for character in message], dtype=torch.int64)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randint(0, 2, size=(payload_bits,), generator=generator, dtype=torch.int64)


def bits_to_string(bits: Tensor) -> str:
    return "".join(str(int(value)) for value in bits.detach().cpu().tolist())


def char_to_bits(c: str) -> str:
    """Convert a single ASCII character to an 8-bit binary string (MSB first)."""
    if len(c) != 1 or ord(c) >= 128:
        raise ValueError(f"Expected a single ASCII character, got: {c!r}")
    return format(ord(c), '08b')


def bits_to_char(bits: str) -> str:
    """Convert an 8-bit binary string back to an ASCII character."""
    if len(bits) != 8 or any(b not in '01' for b in bits):
        raise ValueError(f"Expected an 8-character binary string, got: {bits!r}")
    return chr(int(bits, 2))


def hamming74_encode(raw_bits: Tensor) -> tuple[Tensor, int]:
    raw_bits = raw_bits.to(torch.int64).flatten()
    pad_bits = (-raw_bits.numel()) % 4
    if pad_bits > 0:
        raw_bits = torch.cat((raw_bits, torch.zeros(pad_bits, dtype=torch.int64)))
    blocks = raw_bits.view(-1, 4)
    d1, d2, d3, d4 = [blocks[:, index] for index in range(4)]
    p1 = torch.remainder(d1 + d2 + d4, 2)
    p2 = torch.remainder(d1 + d3 + d4, 2)
    p3 = torch.remainder(d2 + d3 + d4, 2)
    coded = torch.stack((p1, p2, d1, p3, d2, d3, d4), dim=1).reshape(-1)
    return coded.to(torch.int64), pad_bits


def hamming74_decode(coded_bits: Tensor, raw_length: int) -> tuple[Tensor, int]:
    coded_bits = coded_bits.to(torch.int64).flatten()
    if coded_bits.numel() % 7 != 0:
        raise ValueError("coded_bits length must be divisible by 7")
    blocks = coded_bits.view(-1, 7).clone()
    s1 = torch.remainder(blocks[:, 0] + blocks[:, 2] + blocks[:, 4] + blocks[:, 6], 2)
    s2 = torch.remainder(blocks[:, 1] + blocks[:, 2] + blocks[:, 5] + blocks[:, 6], 2)
    s3 = torch.remainder(blocks[:, 3] + blocks[:, 4] + blocks[:, 5] + blocks[:, 6], 2)
    syndrome = s1 + 2 * s2 + 4 * s3
    corrected_blocks = blocks.clone()
    corrected_count = 0
    for index, value in enumerate(syndrome.tolist()):
        if value > 0:
            corrected_blocks[index, value - 1] = 1 - corrected_blocks[index, value - 1]
            corrected_count += 1
    decoded = corrected_blocks[:, [2, 4, 5, 6]].reshape(-1)[:raw_length]
    return decoded.to(torch.int64), corrected_count


def select_topk_carriers(importance: Tensor, payload_bits: int) -> Tensor:
    if payload_bits <= 0:
        raise ValueError("payload_bits must be positive")
    if payload_bits > importance.numel():
        raise ValueError("payload_bits cannot exceed the number of Gaussians")
    return torch.topk(importance, k=payload_bits).indices


def select_low_importance_carriers(importance: Tensor, payload_bits: int) -> Tensor:
    if payload_bits <= 0:
        raise ValueError("payload_bits must be positive")
    if payload_bits > importance.numel():
        raise ValueError("payload_bits cannot exceed the number of Gaussians")
    return torch.topk(importance, k=payload_bits, largest=False).indices


def select_sensitivity_candidates(
    *,
    importance: Tensor,
    sensitivity_values: Tensor,
    carrier_values: Tensor,
    coded_payload_bits: int,
    candidate_factor: int,
    mode: str,
    target_magnitude: float,
) -> tuple[Tensor, Tensor]:
    candidate_count = min(importance.numel(), max(coded_payload_bits, 2 * candidate_factor * coded_payload_bits))
    stability = carrier_stability_weights(carrier_values, mode=mode, target_magnitude=target_magnitude)
    score = importance / (sensitivity_values + 1e-8) * stability
    candidate_indices = torch.topk(score, k=candidate_count).indices
    return candidate_indices, score


def keyed_adjustment_selection(
    *,
    candidate_indices: Tensor,
    carrier_values: Tensor,
    preference_scores: Tensor | None,
    coded_bits: Tensor,
    target_magnitude: float,
    key: str,
    mode: str,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(key_to_seed(key))
    permuted_candidates = candidate_indices[torch.randperm(candidate_indices.numel(), generator=generator)]
    assignment_order = torch.randperm(coded_bits.numel(), generator=generator).tolist()
    available = permuted_candidates.tolist()
    selected = [-1] * int(coded_bits.numel())
    target_zero, target_one = carrier_target_centers(mode, target_magnitude=target_magnitude)
    for bit_index in assignment_order:
        if not available:
            continue
        bit_value = int(coded_bits[bit_index].item())
        target_value = target_one if bit_value == 1 else target_zero
        best_index = min(
            available,
            key=lambda gaussian_index: (
                (
                    _theta_major_distance_scalar(float(carrier_values[gaussian_index].item()), target_value)
                    if mode == THETA_MAJOR_MODE
                    else abs(float(carrier_values[gaussian_index].item()) - target_value)
                ),
                -float(preference_scores[gaussian_index].item()) if preference_scores is not None else 0.0,
            ),
        )
        selected[bit_index] = int(best_index)
        available.remove(best_index)
    if any(index < 0 for index in selected):
        raise RuntimeError("keyed carrier assignment failed to allocate one carrier per coded bit")
    return torch.tensor(selected, dtype=torch.long)


def message_loss(
    params: dict[str, Tensor],
    carrier_indices: Tensor,
    bits: Tensor,
    *,
    mode: str,
    target_magnitude: float,
) -> Tensor:
    carrier_values = carrier_values_from_params(params, mode=mode)[carrier_indices]
    target_zero, target_one = carrier_target_centers(mode, target_magnitude=target_magnitude)
    targets = torch.where(bits > 0, torch.full_like(carrier_values, target_one), torch.full_like(carrier_values, target_zero))
    return carrier_distance(carrier_values, target=targets, mode=mode).pow(2).mean()


def decode_bits(
    params: dict[str, Tensor],
    carrier_indices: Tensor,
    *,
    mode: str,
    target_magnitude: float = 0.35,
) -> Tensor:
    carrier_values = carrier_values_from_params(params, mode=mode)[carrier_indices]
    target_zero, target_one = carrier_target_centers(mode, target_magnitude=target_magnitude)
    dist_zero = carrier_distance(carrier_values, target=target_zero, mode=mode)
    dist_one = carrier_distance(carrier_values, target=target_one, mode=mode)
    return (dist_one < dist_zero).to(torch.int64)


def build_clean_distribution_reference(params: dict[str, Tensor]) -> dict[str, Tensor]:
    canonical = canonicalize_gaussian_geometry(params["scales"], params["thetas"])
    alpha = params["opacities"][:, 0]
    return {
        "log_anisotropy_mean": canonical["log_anisotropy"].mean().detach(),
        "log_anisotropy_std": canonical["log_anisotropy"].std(unbiased=False).detach(),
        "log_area_mean": canonical["log_area"].mean().detach(),
        "log_area_std": canonical["log_area"].std(unbiased=False).detach(),
        "alpha_mean": alpha.mean().detach(),
        "alpha_std": alpha.std(unbiased=False).detach(),
        "alpha_area_corr": _safe_corr(alpha, canonical["log_area"]).detach(),
    }


def matched_moment_loss(params: dict[str, Tensor], reference: dict[str, Tensor]) -> Tensor:
    canonical = canonicalize_gaussian_geometry(params["scales"], params["thetas"])
    alpha = params["opacities"][:, 0]
    losses = [
        (canonical["log_anisotropy"].mean() - reference["log_anisotropy_mean"]) ** 2,
        (canonical["log_anisotropy"].std(unbiased=False) - reference["log_anisotropy_std"]) ** 2,
        (canonical["log_area"].mean() - reference["log_area_mean"]) ** 2,
        (canonical["log_area"].std(unbiased=False) - reference["log_area_std"]) ** 2,
        (alpha.mean() - reference["alpha_mean"]) ** 2,
        (alpha.std(unbiased=False) - reference["alpha_std"]) ** 2,
        (_safe_corr(alpha, canonical["log_area"]) - reference["alpha_area_corr"]) ** 2,
    ]
    return torch.stack(losses).mean()


def anchor_loss_from_scales(
    *,
    renderer: Gaussian2DRenderer,
    clean_scales: Tensor,
    carrier_indices: Tensor,
    sensitivity_weights: Tensor,
) -> Tensor:
    current_scales = renderer.decode_parameters()["scales"][carrier_indices]
    clean_scales_device = clean_scales.to(current_scales.device)
    weighted_error = (current_scales - clean_scales_device[carrier_indices]) ** 2
    return (weighted_error.mean(dim=-1) * sensitivity_weights.to(current_scales.device)).mean()


def compute_ssim(target: Tensor, reconstruction: Tensor) -> float:
    target = target.detach().cpu()
    reconstruction = reconstruction.detach().cpu().clamp(0.0, 1.0)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    per_channel_scores = []
    for channel in range(target.shape[-1]):
        x = target[..., channel]
        y = reconstruction[..., channel]
        mu_x = float(x.mean().item())
        mu_y = float(y.mean().item())
        sigma_x = float(x.var(unbiased=False).item())
        sigma_y = float(y.var(unbiased=False).item())
        sigma_xy = float(((x - mu_x) * (y - mu_y)).mean().item())
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        per_channel_scores.append(numerator / max(denominator, 1e-12))
    return float(sum(per_channel_scores) / len(per_channel_scores))


@torch.no_grad()
def evaluate_renderer(
    renderer: Gaussian2DRenderer,
    target: Tensor,
    grid: Tensor,
    *,
    include_sensitivity: bool = False,
) -> dict[str, object]:
    reconstruction = renderer(grid[..., 0], grid[..., 1], clamp_output=False).detach().cpu()
    clamped_reconstruction = reconstruction.clamp(0.0, 1.0)
    params = renderer.decode_parameters()
    diagnostics = compute_parameter_diagnostics(params["offsets"], params["scales"])
    saturated_value_fraction, saturation_max_excess = compute_saturation_statistics(reconstruction)
    payload = {
        "raw_psnr": compute_psnr(target.detach().cpu(), reconstruction),
        "clamped_psnr": compute_psnr(target.detach().cpu(), clamped_reconstruction),
        "ssim": compute_ssim(target.detach().cpu(), clamped_reconstruction),
        "saturated_value_fraction": saturated_value_fraction,
        "saturation_max_excess": saturation_max_excess,
        "off_canvas_count": int(diagnostics["off_canvas_count"]),
        "large_scale_count": int(diagnostics["large_scale_count"]),
        "max_abs_offset": float(diagnostics["max_abs_offset"]),
        "max_scale": float(diagnostics["max_scale"]),
        "reconstruction": reconstruction,
        "params": {key: value.detach().cpu() for key, value in params.items()},
    }
    if include_sensitivity:
        with torch.enable_grad():
            sensitivity = compute_parameter_sensitivity_summary(
                target=target,
                grid_x=grid[..., 0],
                grid_y=grid[..., 1],
                offsets=params["offsets"],
                scales=params["scales"],
                thetas=params["thetas"],
                colors=params["colors"],
                opacities=params["opacities"],
            )
        payload["sensitivity"] = {key: value.detach().cpu() for key, value in sensitivity.items()}
    return payload


def _feature_percentiles(values: Tensor) -> list[Tensor]:
    quantiles = torch.tensor([0.25, 0.50, 0.75], dtype=values.dtype, device=values.device)
    return [entry.detach().cpu() for entry in torch.quantile(values, quantiles)]


def canonical_feature_matrix(params: dict[str, Tensor], importance: Tensor | None = None) -> Tensor:
    if importance is None:
        importance = compute_gaussian_importance(params["scales"], params["opacities"])
    canonical = canonicalize_gaussian_geometry(params["scales"], params["thetas"])
    theta_major = canonical["theta_major"][:, 0]
    return torch.stack(
        [
            params["offsets"][:, 0],
            params["offsets"][:, 1],
            canonical["log_area"],
            canonical["log_anisotropy"],
            params["opacities"][:, 0],
            importance,
            torch.cos(2.0 * theta_major),
            torch.sin(2.0 * theta_major),
        ],
        dim=1,
    )


def compute_mmd_rbf(x: Tensor, y: Tensor) -> float:
    x = x.detach().cpu()
    y = y.detach().cpu()
    combined = torch.cat((x, y), dim=0)
    pairwise = torch.cdist(combined, combined, p=2) ** 2
    sigma2 = float(torch.median(pairwise[pairwise > 0]).item()) if torch.any(pairwise > 0) else 1.0
    sigma2 = max(sigma2, 1e-6)

    def kernel(a: Tensor, b: Tensor) -> Tensor:
        return torch.exp(-torch.cdist(a, b, p=2) ** 2 / (2.0 * sigma2))

    k_xx = kernel(x, x).mean()
    k_yy = kernel(y, y).mean()
    k_xy = kernel(x, y).mean()
    return float((k_xx + k_yy - 2.0 * k_xy).item())


def summarize_run_features(
    params: dict[str, Tensor],
    *,
    importance: Tensor | None = None,
    sensitivity: dict[str, Tensor] | None = None,
) -> tuple[Tensor, list[str]]:
    if importance is None:
        importance = compute_gaussian_importance(params["scales"], params["opacities"])
    canonical = canonicalize_gaussian_geometry(params["scales"], params["thetas"])
    theta_major = canonical["theta_major"][:, 0]
    alpha = params["opacities"][:, 0]
    off_canvas_fraction = ((params["offsets"].abs() > 1.0).any(dim=-1)).float().mean()
    fields = {
        "mu_x": params["offsets"][:, 0],
        "mu_y": params["offsets"][:, 1],
        "log_area": canonical["log_area"],
        "log_anisotropy": canonical["log_anisotropy"],
        "alpha": alpha,
        "importance": importance,
    }
    feature_values: list[Tensor] = []
    feature_names: list[str] = []
    for name, values in fields.items():
        feature_values.extend(
            [
                values.mean().detach().cpu(),
                values.std(unbiased=False).detach().cpu(),
                *_feature_percentiles(values),
            ]
        )
        feature_names.extend(
            [
                f"{name}_mean",
                f"{name}_std",
                f"{name}_q25",
                f"{name}_q50",
                f"{name}_q75",
            ]
        )
    theta_resultant = torch.sqrt(torch.cos(2.0 * theta_major).mean() ** 2 + torch.sin(2.0 * theta_major).mean() ** 2)
    feature_values.extend(
        [
            theta_resultant.detach().cpu(),
            _safe_corr(alpha, canonical["log_area"]).detach().cpu(),
            off_canvas_fraction.detach().cpu(),
        ]
    )
    feature_names.extend(["theta_resultant_length", "alpha_log_area_corr", "off_canvas_fraction"])
    if sensitivity is not None:
        sensitivity_fields = {
            "sens_log_anisotropy": sensitivity["log_anisotropy"],
            "sens_alpha": sensitivity["alpha"],
            "sens_mu_x": sensitivity["mu_x"],
        }
        for name, values in sensitivity_fields.items():
            feature_values.extend(
                [
                    values.mean().detach().cpu(),
                    values.std(unbiased=False).detach().cpu(),
                    *_feature_percentiles(values),
                ]
            )
            feature_names.extend(
                [
                    f"{name}_mean",
                    f"{name}_std",
                    f"{name}_q25",
                    f"{name}_q50",
                    f"{name}_q75",
                ]
            )
        feature_values.append(_safe_corr(importance, sensitivity["log_anisotropy"]).detach().cpu())
        feature_names.append("importance_sens_log_anisotropy_corr")
    return torch.stack([value.to(torch.float32) for value in feature_values]), feature_names


def bootstrap_feature_dataset(
    params: dict[str, Tensor],
    *,
    importance: Tensor | None,
    sensitivity: dict[str, Tensor] | None,
    seed: int,
    samples: int = 48,
    sample_ratio: float = 0.80,
) -> tuple[Tensor, list[str]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    num_gaussians = params["offsets"].shape[0]
    subset_size = max(8, min(num_gaussians, int(round(sample_ratio * num_gaussians))))
    features: list[Tensor] = []
    feature_names: list[str] | None = None
    for _ in range(samples):
        indices = torch.randperm(num_gaussians, generator=generator)[:subset_size]
        subset_params = {key: value[indices] for key, value in params.items()}
        subset_importance = None if importance is None else importance[indices]
        subset_sensitivity = None if sensitivity is None else {key: value[indices] for key, value in sensitivity.items()}
        feature_vector, feature_names = summarize_run_features(
            subset_params,
            importance=subset_importance,
            sensitivity=subset_sensitivity,
        )
        features.append(feature_vector)
    return torch.stack(features, dim=0), feature_names or []


def binary_auc(scores: Tensor, labels: Tensor) -> float:
    scores = scores.detach().cpu()
    labels = labels.detach().cpu().to(torch.int64)
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.5
    pairwise = (pos_scores[:, None] > neg_scores[None, :]).float()
    ties = (pos_scores[:, None] == neg_scores[None, :]).float()
    return float((pairwise.mean() + 0.5 * ties.mean()).item())


def detectability_auc_from_feature_sets(
    clean_features: Tensor,
    watermarked_features: Tensor,
    *,
    seed: int = 0,
) -> float | None:
    clean_x = clean_features.detach().cpu().to(torch.float32)
    watermarked_x = watermarked_features.detach().cpu().to(torch.float32)
    if clean_x.ndim == 1:
        clean_x = clean_x.unsqueeze(0)
    if watermarked_x.ndim == 1:
        watermarked_x = watermarked_x.unsqueeze(0)
    if clean_x.shape[0] < 2 or watermarked_x.shape[0] < 2:
        return None

    x = torch.cat((clean_x, watermarked_x), dim=0)
    y = torch.cat((torch.zeros(clean_x.shape[0]), torch.ones(watermarked_x.shape[0])), dim=0)
    scores = torch.zeros(x.shape[0], dtype=torch.float32)
    holdout_order = torch.randperm(x.shape[0], generator=torch.Generator(device="cpu").manual_seed(seed + 101)).tolist()
    for holdout_index in holdout_order:
        train_mask = torch.ones(x.shape[0], dtype=torch.bool)
        train_mask[holdout_index] = False
        x_train = x[train_mask]
        y_train = y[train_mask]
        if int((y_train == 0).sum().item()) == 0 or int((y_train == 1).sum().item()) == 0:
            return None
        x_test = x[holdout_index : holdout_index + 1]
        train_mean = x_train.mean(dim=0, keepdim=True)
        train_std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        x_train_norm = (x_train - train_mean) / train_std
        x_test_norm = (x_test - train_mean) / train_std
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
        scores[holdout_index] = (x_test_norm @ weight + bias)[0, 0]
    return binary_auc(scores, y)


def detectability_auc_from_bootstrap(
    clean_params: dict[str, Tensor],
    watermarked_params: dict[str, Tensor],
    *,
    clean_importance: Tensor | None,
    watermarked_importance: Tensor | None,
    clean_sensitivity: dict[str, Tensor] | None,
    watermarked_sensitivity: dict[str, Tensor] | None,
    seed: int = 0,
    include_sensitivity_features: bool = False,
) -> tuple[float | None, list[str]]:
    clean_x, feature_names = summarize_run_features(
        clean_params,
        importance=clean_importance,
        sensitivity=clean_sensitivity if include_sensitivity_features else None,
    )
    watermarked_x, _ = summarize_run_features(
        watermarked_params,
        importance=watermarked_importance,
        sensitivity=watermarked_sensitivity if include_sensitivity_features else None,
    )
    auc = detectability_auc_from_feature_sets(clean_x, watermarked_x, seed=seed)
    return auc, feature_names


def distribution_distance_summary(
    clean_params: dict[str, Tensor],
    watermarked_params: dict[str, Tensor],
    *,
    clean_metrics: dict[str, object] | None = None,
    watermarked_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    clean_canonical = canonicalize_gaussian_geometry(clean_params["scales"], clean_params["thetas"])
    watermarked_canonical = canonicalize_gaussian_geometry(watermarked_params["scales"], watermarked_params["thetas"])
    clean_alpha = clean_params["opacities"][:, 0]
    watermarked_alpha = watermarked_params["opacities"][:, 0]
    moment_shift = {
        "log_anisotropy_mean_abs_shift": float(
            abs(clean_canonical["log_anisotropy"].mean().item() - watermarked_canonical["log_anisotropy"].mean().item())
        ),
        "log_area_mean_abs_shift": float(abs(clean_canonical["log_area"].mean().item() - watermarked_canonical["log_area"].mean().item())),
        "alpha_mean_abs_shift": float(abs(clean_alpha.mean().item() - watermarked_alpha.mean().item())),
        "alpha_log_area_corr_abs_shift": float(
            abs(
                _safe_corr(clean_alpha, clean_canonical["log_area"]).item()
                - _safe_corr(watermarked_alpha, watermarked_canonical["log_area"]).item()
            )
        ),
    }
    distances = {
        "moment_shift": moment_shift,
        "MMD_rbf": compute_mmd_rbf(
            canonical_feature_matrix(clean_params),
            canonical_feature_matrix(watermarked_params),
        ),
        "off_canvas_delta": None if clean_metrics is None or watermarked_metrics is None else int(watermarked_metrics["off_canvas_count"]) - int(clean_metrics["off_canvas_count"]),
        "saturation_delta": None if clean_metrics is None or watermarked_metrics is None else float(watermarked_metrics["saturated_value_fraction"]) - float(clean_metrics["saturated_value_fraction"]),
    }
    return distances


@torch.no_grad()
def quantize_physical_parameters(
    params: dict[str, Tensor],
    *,
    step: float,
) -> dict[str, Tensor]:
    if step <= 0.0:
        raise ValueError("step must be positive")
    quantized = {
        "offsets": torch.round(params["offsets"] / step) * step,
        "scales": clamp_scales_strictly_above_eps(torch.round(params["scales"] / step) * step),
        "thetas": torch.round(params["thetas"] / step) * step,
        "colors": torch.clamp(torch.round(params["colors"] / step) * step, 0.0, 1.0),
        "opacities": torch.clamp(torch.round(params["opacities"] / step) * step, 0.0, 1.0),
    }
    quantized["importance"] = compute_gaussian_importance(quantized["scales"], quantized["opacities"])
    return quantized


@torch.no_grad()
def zero_lowest_importance(
    params: dict[str, Tensor],
    *,
    drop_count: int,
    use_opacity: bool = False,
) -> dict[str, Tensor]:
    if drop_count < 0:
        raise ValueError("drop_count must be non-negative")
    dropped = {key: value.detach().clone() for key, value in params.items() if key != "importance"}
    if drop_count == 0:
        dropped["importance"] = compute_gaussian_importance(dropped["scales"], dropped["opacities"])
        return dropped

    scores = dropped["opacities"][:, 0] if use_opacity else compute_gaussian_importance(dropped["scales"], dropped["opacities"])
    drop_indices = torch.argsort(scores, descending=False)[:drop_count]
    dropped["opacities"][drop_indices, 0] = 0.0
    dropped["importance"] = compute_gaussian_importance(dropped["scales"], dropped["opacities"])
    return dropped


@torch.no_grad()
def add_small_parameter_noise(
    params: dict[str, Tensor],
    *,
    carrier_indices: Tensor,
    mode: str,
    step: float,
    seed: int,
    distribution: str = "uniform",
) -> dict[str, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perturbed = {key: value.detach().clone() for key, value in params.items() if key != "importance"}
    def sample(shape: tuple[int, ...], *, dtype: torch.dtype) -> Tensor:
        if distribution == "uniform":
            return torch.empty(shape, dtype=dtype).uniform_(-step, step, generator=generator)
        if distribution == "gaussian":
            return torch.empty(shape, dtype=dtype).normal_(mean=0.0, std=step, generator=generator)
        raise ValueError(f"Unsupported parameter-noise distribution: {distribution!r}")

    if mode_uses_scale_carrier(mode):
        noise = sample((carrier_indices.numel(), 2), dtype=perturbed["scales"].dtype)
        perturbed["scales"][carrier_indices] = clamp_scales_strictly_above_eps(
            perturbed["scales"][carrier_indices] + noise
        )
    else:
        noise = sample((carrier_indices.numel(), 1), dtype=perturbed["thetas"].dtype)
        perturbed["thetas"][carrier_indices] = perturbed["thetas"][carrier_indices] + noise
    perturbed["importance"] = compute_gaussian_importance(perturbed["scales"], perturbed["opacities"])
    return perturbed


def renderer_from_params(params: dict[str, Tensor], device: torch.device) -> Gaussian2DRenderer:
    sanitized = sanitize_renderer_params(params)
    renderer = Gaussian2DRenderer(num_gaussians=params["offsets"].shape[0]).to(device)
    renderer.set_physical_parameters(
        offsets=sanitized["offsets"].to(device),
        scales=sanitized["scales"].to(device),
        thetas=sanitized["thetas"].to(device),
        colors=sanitized["colors"].to(device),
        opacities=sanitized["opacities"].to(device),
    )
    return renderer


def reconstruction_only_finetune(
    params: dict[str, Tensor],
    *,
    target: Tensor,
    grid: Tensor,
    device: torch.device,
    steps: int = 200,
    lr: float = 0.005,
) -> dict[str, Tensor]:
    renderer = renderer_from_params(params, device=device)
    optimizer = torch.optim.Adam(renderer.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = renderer(grid[..., 0], grid[..., 1], clamp_output=False)
        loss = F.mse_loss(reconstruction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(renderer.parameters(), max_norm=1.0)
        optimizer.step()
    final_params = renderer.decode_parameters()
    output = {key: value.detach().cpu() for key, value in final_params.items()}
    output["importance"] = compute_gaussian_importance(output["scales"], output["opacities"])
    return output


def make_grid_from_run(run: ExportedRun, device: torch.device) -> Tensor:
    return create_grid(int(run.config["height"]), int(run.config["width"]), device)


def experiment_config_from_export(run: ExportedRun, device: torch.device, *, steps: int | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        device=device,
        height=int(run.config["height"]),
        width=int(run.config["width"]),
        num_gaussians=int(run.config["num_gaussians"]),
        steps=int(steps if steps is not None else run.config["steps"]),
        lr=float(run.config["lr"]),
        seed=int(run.config["seed"]),
        target_path=run.config.get("target_path"),
        target_variant=str(run.config.get("target_variant", "default_synthetic")),
        init_mode=str(run.config["init_mode"]),
        scheduler=str(run.config["scheduler"]),
        use_param_groups=True,
        make_overlay=False,
        make_animation=False,
        export_params=False,
        export_trajectory_every=None,
        analysis_tag=run.config.get("analysis_tag"),
        output_dir=None,
    )


# ============================================================================
# v3.0 per-channel BER helpers
# Added for v3.0 to report theta and color_lum channel BER independently.
# ============================================================================

def theta_channel_ber_from_report(
    watermark_report: dict,
    clean_params: dict[str, Tensor],
    wm_params: dict[str, Tensor],
) -> "float | None":
    """Compute BER for the theta carrier channel from a v3.0 watermark report.

    Returns None if the theta channel was not active in this run.
    """
    active_channels = watermark_report.get("active_channels", [])
    if "theta" not in active_channels:
        return None
    try:
        from stego_protocol_v3 import theta_channel_ber as _ber_fn
        from utils import canonicalize_gaussian_geometry
        rc = canonicalize_gaussian_geometry(clean_params["scales"], clean_params["thetas"])
        rw = canonicalize_gaussian_geometry(wm_params["scales"],   wm_params["thetas"])
        clean_t = rc["theta_major"] if isinstance(rc, dict) else rc[2]
        wm_t    = rw["theta_major"] if isinstance(rw, dict) else rw[2]
        if clean_t.dim() > 1:
            clean_t = clean_t[:, 0]
            wm_t    = wm_t[:, 0]
        asgns = [a for a in watermark_report.get("assignments", []) if a.get("theta_indices")]
        return _ber_fn(clean_t, wm_t, asgns)
    except Exception:
        return None


def color_lum_channel_ber_from_report(
    watermark_report: dict,
    clean_params: dict[str, Tensor],
    wm_params: dict[str, Tensor],
) -> "float | None":
    """Compute BER for the color luminance channel from a v3.0 watermark report.

    Returns None if color_lum was not active or color data is unavailable.
    """
    active_channels = watermark_report.get("active_channels", [])
    if "color_lum" not in active_channels:
        return None
    if "colors" not in clean_params or "colors" not in wm_params:
        return None
    try:
        from stego_protocol_v3 import color_lum_channel_ber as _ber_fn
        lum_delta = float(watermark_report.get("lum_delta", 0.008))
        asgns = [a for a in watermark_report.get("assignments", []) if a.get("lum_indices")]
        return _ber_fn(clean_params["colors"], wm_params["colors"], asgns, lum_delta)
    except Exception:
        return None


def v3_per_channel_ber_summary(
    watermark_report: dict,
    clean_params: dict[str, Tensor],
    wm_params: dict[str, Tensor],
) -> "dict[str, float | None]":
    """Return per-channel BER summary for a v3.0 watermark report.

    Keys: 'theta_ber', 'color_lum_ber'
    Values: float in [0, 1] or None if the channel was not active.
    """
    return {
        "theta_ber":     theta_channel_ber_from_report(watermark_report, clean_params, wm_params),
        "color_lum_ber": color_lum_channel_ber_from_report(watermark_report, clean_params, wm_params),
    }
