from __future__ import annotations

import csv
import io
import json
import math
import random
from datetime import datetime
from pathlib import Path
import textwrap
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch
from torch import Tensor


VALID_SYNTHETIC_TARGET_VARIANTS = (
    "default_synthetic",
    "stripe_circle",
    "checker_blobs",
)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def get_best_device() -> torch.device:
    """Select the best available device in the order CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_run_output_dir(root: Path, prefix: str) -> Path:
    """Create a unique output directory for one script invocation."""
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = root / f"{prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def archive_legacy_output_files(root: Path, legacy_names: Sequence[str]) -> Path | None:
    """Move old fixed-name root artifacts aside so new runs cannot be mistaken for them."""
    existing_files = [root / name for name in legacy_names if (root / name).is_file()]
    if not existing_files:
        return None
    archive_root = root / "legacy" / f"root_fixed_name_archive_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    archive_root.mkdir(parents=True, exist_ok=False)
    for path in existing_files:
        path.rename(archive_root / path.name)
    return archive_root


def _image_to_numpy(image: Tensor) -> object:
    """Convert a tensor image to a clamped CPU numpy array for plotting."""
    return image.detach().clamp(0.0, 1.0).cpu().numpy()


def _positive_visualization_vmax(values: Tensor, floor: float = 1e-6) -> float:
    """Choose a positive heatmap upper bound that follows the observed data."""
    return max(floor, float(values.detach().amax().item()))


def _normalized_to_pixel_x(x: float, width: int) -> float:
    """Map normalized x in [-1, 1] to pixel coordinates."""
    return 0.5 * (x + 1.0) * (width - 1)


def _normalized_to_pixel_y(y: float, height: int) -> float:
    """Map normalized y in [-1, 1] to pixel coordinates."""
    return 0.5 * (y + 1.0) * (height - 1)


def create_grid(height: int, width: int, device: torch.device) -> Tensor:
    """Create a normalized pixel grid in [-1, 1] using ij indexing.

    Returns:
        Tensor of shape [H, W, 2], where the last dimension is (x, y).
    """
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


@torch.no_grad()
def synthetic_target(grid: Tensor, variant: str = "default_synthetic") -> Tensor:
    """Create a nontrivial self-contained RGB target image."""
    if variant not in VALID_SYNTHETIC_TARGET_VARIANTS:
        raise ValueError(f"unsupported synthetic target variant: {variant}")

    x = grid[..., 0]
    y = grid[..., 1]
    image = torch.zeros((*grid.shape[:2], 3), device=grid.device, dtype=grid.dtype)
    if variant == "default_synthetic":
        circle_center_x = -0.38
        circle_center_y = -0.12
        circle_radius = 0.28
        circle_mask = ((x - circle_center_x) ** 2 + (y - circle_center_y) ** 2) <= circle_radius ** 2
        image[..., 0] = torch.where(circle_mask, torch.full_like(x, 0.92), image[..., 0])

        rect_left, rect_right = 0.05, 0.62
        rect_bottom, rect_top = -0.72, -0.20
        edge_sharpness = 40.0
        rect_mask = (
            torch.sigmoid(edge_sharpness * (x - rect_left))
            * torch.sigmoid(edge_sharpness * (rect_right - x))
            * torch.sigmoid(edge_sharpness * (y - rect_bottom))
            * torch.sigmoid(edge_sharpness * (rect_top - y))
        )
        image[..., 1] = torch.maximum(image[..., 1], 0.85 * rect_mask)

        blob = torch.exp(-(((x - 0.30) / 0.25) ** 2 + ((y - 0.42) / 0.17) ** 2) * 1.6)
        image[..., 2] = torch.maximum(image[..., 2], 0.95 * blob)

        overlap = torch.exp(-(((x + 0.05) / 0.35) ** 2 + ((y - 0.20) / 0.25) ** 2) * 2.2)
        image[..., 0] = torch.clamp(image[..., 0] + 0.18 * overlap, 0.0, 1.0)
        image[..., 1] = torch.clamp(image[..., 1] + 0.12 * overlap, 0.0, 1.0)
        return image.clamp(0.0, 1.0)

    if variant == "stripe_circle":
        stripe_a = 0.5 + 0.5 * torch.sin(6.0 * math.pi * (0.55 * x + 0.20 * y))
        stripe_b = 0.5 + 0.5 * torch.sin(5.0 * math.pi * (-0.25 * x + 0.70 * y) + 0.7)
        circle = torch.exp(-(((x + 0.38) / 0.24) ** 2 + ((y - 0.05) / 0.24) ** 2) * 3.0)
        ring = torch.exp(-(((torch.sqrt((x - 0.34) ** 2 + (y + 0.34) ** 2) - 0.24) / 0.06) ** 2))
        image[..., 0] = 0.15 + 0.70 * stripe_a
        image[..., 1] = torch.maximum(0.10 + 0.75 * circle, 0.20 * stripe_b)
        image[..., 2] = torch.clamp(0.20 + 0.55 * stripe_b + 0.35 * ring, 0.0, 1.0)
        image[..., 1] = torch.clamp(image[..., 1] + 0.18 * ring, 0.0, 1.0)
        return image.clamp(0.0, 1.0)

    checker = 0.5 * (
        1.0
        + torch.sign(torch.sin(5.0 * math.pi * x + 0.3))
        * torch.sign(torch.sin(5.0 * math.pi * y - 0.2))
    )
    blob_a = torch.exp(-(((x + 0.46) / 0.16) ** 2 + ((y + 0.34) / 0.20) ** 2) * 2.1)
    blob_b = torch.exp(-(((x - 0.20) / 0.26) ** 2 + ((y - 0.10) / 0.18) ** 2) * 1.5)
    blob_c = torch.exp(-(((x - 0.42) / 0.18) ** 2 + ((y + 0.38) / 0.14) ** 2) * 2.5)
    image[..., 0] = torch.clamp(0.12 + 0.60 * checker + 0.35 * blob_a, 0.0, 1.0)
    image[..., 1] = torch.clamp(0.10 + 0.45 * (1.0 - checker) + 0.55 * blob_b, 0.0, 1.0)
    image[..., 2] = torch.clamp(0.08 + 0.28 * checker + 0.62 * blob_c, 0.0, 1.0)

    return image.clamp(0.0, 1.0)


@torch.no_grad()
def load_target_image(
    target_path: Path,
    height: int,
    width: int,
    device: torch.device,
) -> Tensor:
    """Load an exact-size RGB target image if Pillow is available."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for --target-path usage. Install it with `pip install Pillow`."
        ) from exc

    image = Image.open(target_path).convert("RGB")
    if image.size != (width, height):
        raise ValueError(
            f"target image must already have size {width}x{height}; "
            f"got {image.size[0]}x{image.size[1]} for {target_path}"
        )
    image_tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
    image_tensor = image_tensor.view(height, width, 3) / 255.0
    return image_tensor.to(device)


@torch.no_grad()
def compute_parameter_diagnostics(
    offsets: Tensor,
    scales: Tensor,
    *,
    frame_limit: float = 1.0,
    large_scale_threshold: float = 1.0,
) -> dict[str, float]:
    """Summarize parameter drift that matters for presentation honesty."""
    offsets_cpu = offsets.detach().cpu()
    scales_cpu = scales.detach().cpu()
    off_canvas_count = int(((offsets_cpu.abs() > frame_limit).any(dim=-1)).sum().item())
    large_scale_count = int(((scales_cpu > large_scale_threshold).any(dim=-1)).sum().item())
    return {
        "off_canvas_count": off_canvas_count,
        "large_scale_count": large_scale_count,
        "max_abs_offset": float(offsets_cpu.abs().max().item()),
        "max_scale": float(scales_cpu.max().item()),
    }


@torch.no_grad()
def compute_gaussian_importance(scales: Tensor, opacities: Tensor) -> Tensor:
    """Compute the repository-standard Gaussian importance heuristic.

    The heuristic matches the overlay ranking and combines opacity with the
    Gaussian footprint proxy `s_x * s_y`.
    """
    if scales.ndim != 2 or scales.shape[-1] != 2:
        raise ValueError("scales must have shape [N, 2]")
    if opacities.ndim != 2 or opacities.shape[-1] != 1 or opacities.shape[0] != scales.shape[0]:
        raise ValueError("opacities must have shape [N, 1] and match scales")
    return opacities[:, 0] * scales[:, 0] * scales[:, 1]


@torch.no_grad()
def canonicalize_gaussian_geometry(scales: Tensor, thetas: Tensor) -> dict[str, Tensor]:
    """Canonicalize anisotropic geometry into major-axis coordinates."""
    if scales.ndim != 2 or scales.shape[-1] != 2:
        raise ValueError("scales must have shape [N, 2]")
    if thetas.ndim != 2 or thetas.shape[-1] != 1 or thetas.shape[0] != scales.shape[0]:
        raise ValueError("thetas must have shape [N, 1] and match scales")

    sx = scales[:, 0]
    sy = scales[:, 1]
    theta = thetas[:, 0]
    keep_order = sx >= sy
    s_major = torch.where(keep_order, sx, sy)
    s_minor = torch.where(keep_order, sy, sx)
    theta_major = torch.where(keep_order, theta, theta + (math.pi / 2.0))
    theta_major = torch.remainder(theta_major, math.pi)
    log_area = torch.log(s_major * s_minor)
    log_anisotropy = torch.log(s_major / s_minor)
    signed_logratio = torch.log(sx / sy)
    return {
        "s_major": s_major,
        "s_minor": s_minor,
        "theta_major": theta_major.unsqueeze(-1),
        "log_area": log_area,
        "log_anisotropy": log_anisotropy,
        "signed_logratio": signed_logratio,
        "axis_swapped": (~keep_order).to(scales.dtype),
    }


def render_gaussians_from_parameters(
    grid_x: Tensor,
    grid_y: Tensor,
    offsets: Tensor,
    scales: Tensor,
    thetas: Tensor,
    colors: Tensor,
    opacities: Tensor,
) -> Tensor:
    """Render the additive Gaussian image from explicit physical parameters."""
    cos_theta = torch.cos(thetas[:, 0])
    sin_theta = torch.sin(thetas[:, 0])
    inv_sx2 = 1.0 / (scales[:, 0] * scales[:, 0])
    inv_sy2 = 1.0 / (scales[:, 1] * scales[:, 1])
    dx = grid_x.unsqueeze(0) - offsets[:, 0].view(-1, 1, 1)
    dy = grid_y.unsqueeze(0) - offsets[:, 1].view(-1, 1, 1)
    local_x = cos_theta.view(-1, 1, 1) * dx + sin_theta.view(-1, 1, 1) * dy
    local_y = -sin_theta.view(-1, 1, 1) * dx + cos_theta.view(-1, 1, 1) * dy
    quadratic = inv_sx2.view(-1, 1, 1) * local_x * local_x + inv_sy2.view(-1, 1, 1) * local_y * local_y
    gaussian_response = torch.exp(-0.5 * quadratic)
    weights = gaussian_response * opacities[:, 0].view(-1, 1, 1)
    return torch.sum(weights.unsqueeze(-1) * colors.view(-1, 1, 1, 3), dim=0)


def compute_parameter_sensitivity_summary(
    *,
    target: Tensor,
    grid_x: Tensor,
    grid_y: Tensor,
    offsets: Tensor,
    scales: Tensor,
    thetas: Tensor,
    colors: Tensor,
    opacities: Tensor,
) -> dict[str, Tensor]:
    """Compute a reconstruction-gradient sensitivity proxy in physical parameter space."""
    offsets_probe = offsets.detach().clone().requires_grad_(True)
    scales_probe = scales.detach().clone().requires_grad_(True)
    thetas_probe = thetas.detach().clone().requires_grad_(True)
    opacities_probe = opacities.detach().clone().requires_grad_(True)
    colors_probe = colors.detach().clone()
    reconstruction = render_gaussians_from_parameters(
        grid_x,
        grid_y,
        offsets_probe,
        scales_probe,
        thetas_probe,
        colors_probe,
        opacities_probe,
    )
    loss = torch.mean((reconstruction - target) ** 2)
    grad_offsets, grad_scales, grad_thetas, grad_opacities = torch.autograd.grad(
        loss,
        (offsets_probe, scales_probe, thetas_probe, opacities_probe),
    )
    return {
        "mu_x": grad_offsets[:, 0].abs().detach(),
        "mu_y": grad_offsets[:, 1].abs().detach(),
        "theta_major": grad_thetas[:, 0].abs().detach(),
        "alpha": grad_opacities[:, 0].abs().detach(),
        "log_anisotropy": (grad_scales[:, 0] * scales[:, 0] - grad_scales[:, 1] * scales[:, 1]).abs().detach(),
        "log_area": (grad_scales[:, 0] * scales[:, 0] + grad_scales[:, 1] * scales[:, 1]).abs().detach(),
    }


def save_json(data: object, path: Path) -> None:
    """Write JSON with stable formatting for experiment metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


@torch.no_grad()
def export_parameter_table(
    *,
    offsets: Tensor,
    scales: Tensor,
    thetas: Tensor,
    colors: Tensor,
    opacities: Tensor,
    importance: Tensor,
    height: int,
    width: int,
    path: Path,
    sensitivity: dict[str, Tensor] | None = None,
) -> None:
    """Export decoded Gaussian parameters to a flat CSV table."""
    if not (
        offsets.shape[0]
        == scales.shape[0]
        == thetas.shape[0]
        == colors.shape[0]
        == opacities.shape[0]
        == importance.shape[0]
    ):
        raise ValueError("all parameter tensors must share the same leading dimension")

    offsets_cpu = offsets.detach().cpu()
    scales_cpu = scales.detach().cpu()
    thetas_cpu = thetas.detach().cpu()
    colors_cpu = colors.detach().cpu()
    opacities_cpu = opacities.detach().cpu()
    importance_cpu = importance.detach().cpu()
    canonical = canonicalize_gaussian_geometry(scales_cpu, thetas_cpu)
    sensitivity_cpu = None if sensitivity is None else {key: value.detach().cpu() for key, value in sensitivity.items()}
    center_rows = torch.clamp(((offsets_cpu[:, 1] + 1.0) * 0.5 * (height - 1)).round().long(), 0, height - 1)
    center_cols = torch.clamp(((offsets_cpu[:, 0] + 1.0) * 0.5 * (width - 1)).round().long(), 0, width - 1)
    area = scales_cpu[:, 0] * scales_cpu[:, 1]
    log_sx = torch.log(scales_cpu[:, 0])
    log_sy = torch.log(scales_cpu[:, 1])
    on_canvas_mask = (offsets_cpu.abs() <= 1.0).all(dim=-1)
    ranks = torch.argsort(torch.argsort(importance_cpu, descending=True)) + 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gaussian_id",
                "importance_rank",
                "importance",
                "mu_x",
                "mu_y",
                "center_row",
                "center_col",
                "on_canvas",
                "s_x",
                "s_y",
                "area",
                "s_major",
                "s_minor",
                "log_area",
                "log_s_x",
                "log_s_y",
                "log_anisotropy",
                "signed_logratio",
                "theta",
                "theta_major",
                "alpha",
                "color_r",
                "color_g",
                "color_b",
                "sens_mu_x",
                "sens_mu_y",
                "sens_log_anisotropy",
                "sens_log_area",
                "sens_theta_major",
                "sens_alpha",
            ],
        )
        writer.writeheader()
        for index in range(offsets_cpu.shape[0]):
            writer.writerow(
                {
                    "gaussian_id": index,
                    "importance_rank": int(ranks[index].item()),
                    "importance": f"{float(importance_cpu[index].item()):.10f}",
                    "mu_x": f"{float(offsets_cpu[index, 0].item()):.10f}",
                    "mu_y": f"{float(offsets_cpu[index, 1].item()):.10f}",
                    "center_row": int(center_rows[index].item()),
                    "center_col": int(center_cols[index].item()),
                    "on_canvas": int(on_canvas_mask[index].item()),
                    "s_x": f"{float(scales_cpu[index, 0].item()):.10f}",
                    "s_y": f"{float(scales_cpu[index, 1].item()):.10f}",
                    "area": f"{float(area[index].item()):.10f}",
                    "s_major": f"{float(canonical['s_major'][index].item()):.10f}",
                    "s_minor": f"{float(canonical['s_minor'][index].item()):.10f}",
                    "log_area": f"{float(canonical['log_area'][index].item()):.10f}",
                    "log_s_x": f"{float(log_sx[index].item()):.10f}",
                    "log_s_y": f"{float(log_sy[index].item()):.10f}",
                    "log_anisotropy": f"{float(canonical['log_anisotropy'][index].item()):.10f}",
                    "signed_logratio": f"{float(canonical['signed_logratio'][index].item()):.10f}",
                    "theta": f"{float(thetas_cpu[index, 0].item()):.10f}",
                    "theta_major": f"{float(canonical['theta_major'][index, 0].item()):.10f}",
                    "alpha": f"{float(opacities_cpu[index, 0].item()):.10f}",
                    "color_r": f"{float(colors_cpu[index, 0].item()):.10f}",
                    "color_g": f"{float(colors_cpu[index, 1].item()):.10f}",
                    "color_b": f"{float(colors_cpu[index, 2].item()):.10f}",
                    "sens_mu_x": (
                        f"{float(sensitivity_cpu['mu_x'][index].item()):.10f}" if sensitivity_cpu is not None else ""
                    ),
                    "sens_mu_y": (
                        f"{float(sensitivity_cpu['mu_y'][index].item()):.10f}" if sensitivity_cpu is not None else ""
                    ),
                    "sens_log_anisotropy": (
                        f"{float(sensitivity_cpu['log_anisotropy'][index].item()):.10f}"
                        if sensitivity_cpu is not None
                        else ""
                    ),
                    "sens_log_area": (
                        f"{float(sensitivity_cpu['log_area'][index].item()):.10f}" if sensitivity_cpu is not None else ""
                    ),
                    "sens_theta_major": (
                        f"{float(sensitivity_cpu['theta_major'][index].item()):.10f}"
                        if sensitivity_cpu is not None
                        else ""
                    ),
                    "sens_alpha": (
                        f"{float(sensitivity_cpu['alpha'][index].item()):.10f}" if sensitivity_cpu is not None else ""
                    ),
                }
            )


@torch.no_grad()
def save_raw_tensor_image(image: Tensor, path: Path) -> None:
    """Save an RGB tensor as an exact-resolution PNG without matplotlib resampling."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to save exact-resolution PNGs. Install it with `pip install Pillow`."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = torch.round(image.detach().clamp(0.0, 1.0).cpu() * 255.0).to(torch.uint8).numpy()
    Image.fromarray(image_uint8, mode="RGB").save(path)


@torch.no_grad()
def save_tensor_image(image: Tensor, path: Path, title: str) -> None:
    """Save a single RGB tensor image with a clean matplotlib layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image_np = _image_to_numpy(image)

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=160)
    ax.imshow(image_np, interpolation="nearest", origin="upper")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_single_gaussian_diagnostic(image: Tensor, path: Path, title: str, zoom_radius: int = 4) -> None:
    """Save a single-Gaussian figure with a peak-centered zoom and line profiles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image_cpu = image.detach().clamp(0.0, 1.0).cpu()
    image_np = image_cpu.numpy()
    height, width = image_cpu.shape[:2]

    intensity = image_cpu.amax(dim=-1)
    peak_flat = int(intensity.argmax().item())
    peak_row = peak_flat // width
    peak_col = peak_flat % width

    row_start = max(0, peak_row - zoom_radius)
    row_end = min(height, peak_row + zoom_radius + 1)
    col_start = max(0, peak_col - zoom_radius)
    col_end = min(width, peak_col + zoom_radius + 1)
    zoom_np = image_np[row_start:row_end, col_start:col_end]

    horizontal_profile = intensity[peak_row].numpy()
    vertical_profile = intensity[:, peak_col].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4), dpi=170)
    fig.suptitle(title, fontsize=12)

    axes[0].imshow(image_np, interpolation="nearest", origin="upper")
    axes[0].set_title("Full 64x64 View")
    axes[0].axis("off")

    axes[1].imshow(zoom_np, interpolation="nearest", origin="upper")
    axes[1].set_title(f"Peak Zoom ({zoom_np.shape[1]}x{zoom_np.shape[0]})")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].plot(horizontal_profile, label=f"row {peak_row}", linewidth=1.8, color="#d62728")
    axes[2].plot(vertical_profile, label=f"col {peak_col}", linewidth=1.8, color="#1f77b4")
    axes[2].set_title("Peak-Centered Profiles")
    axes[2].set_xlabel("Pixel Index")
    axes[2].set_ylabel("Max-Channel Intensity")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_comparison_figure(target: Tensor, reconstruction: Tensor, path: Path) -> None:
    """Save target, clamped reconstruction, raw error, and saturation side by side."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target_cpu = target.detach().cpu()
    reconstruction_cpu = reconstruction.detach().cpu()
    target_np = target_cpu.clamp(0.0, 1.0).numpy()
    reconstruction_clamped = reconstruction_cpu.clamp(0.0, 1.0)
    recon_np = reconstruction_clamped.numpy()
    error_map = (target_cpu - reconstruction_cpu).abs().amax(dim=-1)
    error_np = error_map.numpy()
    over = torch.clamp(reconstruction_cpu - 1.0, min=0.0)
    under = torch.clamp(-reconstruction_cpu, min=0.0)
    saturation_map = torch.maximum(over, under).amax(dim=-1)
    saturation_np = saturation_map.numpy()
    error_vmax = _positive_visualization_vmax(error_map)
    saturation_vmax = _positive_visualization_vmax(saturation_map)

    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.8), dpi=170)
    axes[0].imshow(target_np, interpolation="nearest", origin="upper")
    axes[0].set_title("Target")
    axes[0].axis("off")

    axes[1].imshow(recon_np, interpolation="nearest", origin="upper")
    axes[1].set_title("Clamped Reconstruction")
    axes[1].axis("off")

    error_im = axes[2].imshow(
        error_np,
        cmap="magma",
        vmin=0.0,
        vmax=error_vmax,
        interpolation="nearest",
        origin="upper",
    )
    axes[2].set_title("Raw Max-Channel Error")
    axes[2].axis("off")
    fig.colorbar(error_im, ax=axes[2], fraction=0.046, pad=0.04)

    saturation_im = axes[3].imshow(
        saturation_np,
        cmap="viridis",
        vmin=0.0,
        vmax=saturation_vmax,
        interpolation="nearest",
        origin="upper",
    )
    axes[3].set_title("Out-of-Range Excess")
    axes[3].axis("off")
    fig.colorbar(saturation_im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_reconstruction_evolution_gif(
    target: Tensor,
    frames: Sequence[tuple[str, float, Tensor]],
    path: Path,
    frame_duration_ms: int = 180,
    panel_scale: int = 4,
) -> None:
    """Save a short GIF showing reconstruction progress."""
    if not frames:
        raise ValueError("frames must not be empty")
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to save animation GIFs. Install it with `pip install Pillow`."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    target_cpu = target.detach().clamp(0.0, 1.0).cpu()
    height, width = target_cpu.shape[:2]
    panel_width = width * panel_scale
    panel_height = height * panel_scale
    margin = 18
    header_height = 52
    label_height = 22
    footer_height = 14
    canvas_width = margin * 4 + panel_width * 3
    canvas_height = margin + header_height + label_height + panel_height + footer_height
    resampling = getattr(Image, "Resampling", Image)
    nearest = resampling.NEAREST
    magma = matplotlib.colormaps["magma"]

    target_rgb = torch.round(target_cpu * 255.0).to(torch.uint8).numpy()
    target_image = Image.fromarray(target_rgb, mode="RGB").resize((panel_width, panel_height), resample=nearest)
    rendered_frames: list[Image.Image] = []
    for label, loss_value, reconstruction in frames:
        reconstruction_cpu = reconstruction.detach().cpu()
        reconstruction_rgb = torch.round(reconstruction_cpu.clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
        reconstruction_image = Image.fromarray(reconstruction_rgb, mode="RGB").resize(
            (panel_width, panel_height),
            resample=nearest,
        )

        error_map = (target_cpu - reconstruction_cpu).abs().amax(dim=-1)
        error_scale = _positive_visualization_vmax(error_map)
        error_rgb = torch.from_numpy((magma((error_map / error_scale).clamp(0.0, 1.0).numpy())[..., :3] * 255.0).astype("uint8"))
        error_image = Image.fromarray(error_rgb.numpy(), mode="RGB").resize((panel_width, panel_height), resample=nearest)

        canvas = Image.new("RGB", (canvas_width, canvas_height), color="#f7f4ee")
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 10), "Reconstruction Evolution", fill="#111111")
        draw.text((margin, 30), f"{label} | raw MSE {loss_value:.6f}", fill="#444444")

        label_y = margin + header_height
        panel_y = label_y + label_height
        x_positions = [
            margin,
            margin * 2 + panel_width,
            margin * 3 + panel_width * 2,
        ]
        labels = ["Target", "Clamped Reconstruction", "Raw Error"]
        images = [target_image, reconstruction_image, error_image]
        for x_position, panel_label, panel_image in zip(x_positions, labels, images):
            draw.text((x_position, label_y), panel_label, fill="#333333")
            canvas.paste(panel_image, (x_position, panel_y))

        rendered_frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=255))

    rendered_frames[0].save(
        path,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
    )


@torch.no_grad()
def save_loss_curve(loss_history: Sequence[float], path: Path) -> None:
    """Save a simple publication-style loss curve."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=170)
    ax.plot(loss_history, color="#1f77b4", linewidth=2.0)
    ax.set_title("Optimization Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_metrics_card(
    title: str,
    metric_rows: Sequence[tuple[str, str]],
    notes: Sequence[str],
    path: Path,
) -> None:
    """Save a compact metrics summary as a PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)

    wrapped_rows = [
        (
            label,
            textwrap.wrap(value, width=42, break_long_words=True, break_on_hyphens=False) or [value],
        )
        for label, value in metric_rows
    ]
    wrapped_notes = [
        textwrap.wrap(note, width=78, break_long_words=True, break_on_hyphens=False) or [note]
        for note in notes
    ]

    metric_line_count = sum(len(lines) for _, lines in wrapped_rows)
    note_line_count = sum(len(lines) for lines in wrapped_notes)
    fig_height = max(4.1, 1.6 + 0.34 * metric_line_count + 0.24 * note_line_count + (0.42 if notes else 0.0))
    fig, ax = plt.subplots(figsize=(8.2, fig_height), dpi=170)
    fig.patch.set_facecolor("#f7f4ee")
    ax.set_facecolor("#f7f4ee")
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.text(
        0.05,
        0.95,
        title,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
        family="DejaVu Sans",
        color="#161616",
        transform=ax.transAxes,
    )
    ax.plot([0.05, 0.95], [0.90, 0.90], color="#cfc7b8", linewidth=1.2, transform=ax.transAxes)

    y = 0.84
    value_x = 0.38
    metric_step = 0.052
    row_gap = 0.022
    for label, value_lines in wrapped_rows:
        ax.text(
            0.06,
            y,
            label,
            fontsize=11.5,
            fontweight="bold",
            va="top",
            ha="left",
            family="DejaVu Sans",
            color="#303030",
            transform=ax.transAxes,
        )
        value_y = y
        for line in value_lines:
            ax.text(
                value_x,
                value_y,
                line,
                fontsize=11.5,
                va="top",
                ha="left",
                family="DejaVu Sans Mono",
                color="#121212",
                transform=ax.transAxes,
            )
            value_y -= metric_step
        y = value_y - row_gap

    if notes:
        y -= 0.01
        ax.text(
            0.05,
            y,
            "Notes",
            fontsize=12.5,
            fontweight="bold",
            va="top",
            ha="left",
            family="DejaVu Sans",
            color="#202020",
            transform=ax.transAxes,
        )
        y -= 0.07
        note_step = 0.048
        for note_lines in wrapped_notes:
            for line_index, line in enumerate(note_lines):
                prefix = "- " if line_index == 0 else "  "
                ax.text(
                    0.06,
                    y,
                    f"{prefix}{line}",
                    fontsize=10.8,
                    va="top",
                    ha="left",
                    family="DejaVu Sans",
                    color="#444444",
                    transform=ax.transAxes,
                )
                y -= note_step
            y -= 0.014

    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


@torch.no_grad()
def save_gaussian_overlay(
    background: Tensor,
    offsets: Tensor,
    scales: Tensor,
    thetas: Tensor,
    colors: Tensor,
    opacities: Tensor,
    path: Path,
    top_k: int = 12,
    contour_sigma: float = 2.0,
) -> None:
    """Overlay the most important learned Gaussian ellipses on a background image.

    The ranking heuristic uses opacity * s_x * s_y, which roughly captures both
    amplitude and spatial footprint. Ellipse outlines are drawn at a fixed
    contour level of `contour_sigma` standard deviations.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    background_np = _image_to_numpy(background)
    height, width = background.shape[:2]

    offsets_cpu = offsets.detach().cpu()
    scales_cpu = scales.detach().cpu()
    thetas_cpu = thetas.detach().cpu()
    colors_cpu = colors.detach().cpu()
    opacities_cpu = opacities.detach().cpu()

    importance = compute_gaussian_importance(scales_cpu, opacities_cpu)
    num_to_draw = min(top_k, importance.numel())
    top_indices = torch.topk(importance, k=num_to_draw).indices

    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=170)
    ax.imshow(background_np, interpolation="nearest", origin="upper")
    ax.set_title(f"Learned Gaussian Overlay (top {num_to_draw})")
    ax.axis("off")

    x_scale = 0.5 * (width - 1)
    y_scale = 0.5 * (height - 1)

    for rank, index in enumerate(top_indices.tolist()):
        mu_x = float(offsets_cpu[index, 0])
        mu_y = float(offsets_cpu[index, 1])
        sx = float(scales_cpu[index, 0])
        sy = float(scales_cpu[index, 1])
        theta = float(thetas_cpu[index, 0])

        center_x = _normalized_to_pixel_x(mu_x, width)
        center_y = _normalized_to_pixel_y(mu_y, height)

        # The renderer works in an image-style coordinate system where y grows
        # downward with row index, so the displayed angle uses the negative sign
        # to stay visually consistent with the rendered image.
        width_pixels = 2.0 * contour_sigma * sx * x_scale
        height_pixels = 2.0 * contour_sigma * sy * y_scale
        angle_degrees = -torch.rad2deg(torch.tensor(theta)).item()

        color = colors_cpu[index]
        edge_color = (
            float(torch.clamp(0.25 + 0.75 * color[0], 0.0, 1.0)),
            float(torch.clamp(0.25 + 0.75 * color[1], 0.0, 1.0)),
            float(torch.clamp(0.25 + 0.75 * color[2], 0.0, 1.0)),
        )
        linewidth = max(0.8, 1.8 - 0.05 * rank)

        ellipse = Ellipse(
            xy=(center_x, center_y),
            width=width_pixels,
            height=height_pixels,
            angle=angle_degrees,
            fill=False,
            edgecolor=edge_color,
            linewidth=linewidth,
            alpha=0.95,
        )
        ax.add_patch(ellipse)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_training_progress_animation(
    target: Tensor,
    snapshots: Sequence[tuple[int, float, Tensor]],
    loss_history: Sequence[float],
    path: Path,
    frame_duration_ms: int = 300,
) -> None:
    """Save a compact 2x2 training progress animation as a GIF."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to write training_progress_combo.gif. Install it with `pip install Pillow`."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)

    target_cpu = target.detach().cpu()
    target_np = _image_to_numpy(target)
    y_max = max(loss_history) * 1.05 if loss_history else 1.0
    frames: list[Image.Image] = []

    for iteration, loss_value, reconstruction in snapshots:
        recon_np = _image_to_numpy(reconstruction)
        error_np = (target_cpu - reconstruction.detach().cpu()).abs().mean(dim=-1).numpy()

        fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.8), dpi=140)
        fig.suptitle(f"Iter {iteration} | Loss {loss_value:.4f}", fontsize=12)

        axes[0, 0].imshow(target_np, interpolation="nearest", origin="upper")
        axes[0, 0].set_title("Target")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(recon_np, interpolation="nearest", origin="upper")
        axes[0, 1].set_title("Reconstruction")
        axes[0, 1].axis("off")

        error_im = axes[1, 0].imshow(error_np, cmap="magma", interpolation="nearest", origin="upper")
        axes[1, 0].set_title("Absolute Error")
        axes[1, 0].axis("off")
        fig.colorbar(error_im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        axes[1, 1].plot(loss_history[: iteration + 1], color="#1f77b4", linewidth=2.0)
        axes[1, 1].scatter([iteration], [loss_value], color="#d62728", s=32, zorder=3)
        axes[1, 1].set_title("Loss Curve")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("MSE")
        axes[1, 1].set_xlim(0, max(1, len(loss_history) - 1))
        axes[1, 1].set_ylim(0.0, y_max)
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        frame = Image.open(buffer).convert("RGB")
        frame.load()
        frames.append(frame)
        buffer.close()

    if not frames:
        raise ValueError("No progress snapshots were provided for animation generation.")

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
