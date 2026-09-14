from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from gaussian2d import Gaussian2DRenderer
from utils import (
    VALID_SYNTHETIC_TARGET_VARIANTS,
    archive_legacy_output_files,
    canonicalize_gaussian_geometry,
    compute_gaussian_importance,
    compute_parameter_sensitivity_summary,
    create_grid,
    create_run_output_dir,
    compute_parameter_diagnostics,
    export_parameter_table,
    get_best_device,
    load_target_image,
    save_gaussian_overlay,
    save_comparison_figure,
    save_loss_curve,
    save_json,
    save_metrics_card,
    save_raw_tensor_image,
    save_reconstruction_evolution_gif,
    save_tensor_image,
    set_seed,
    synthetic_target,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ASSET_DIR = PROJECT_ROOT / "assets"
DEFAULT_GRAD_CLIP = 1.0
OVERLAY_TOP_K = 12
ANIMATION_MAX_FRAMES = 18
ANIMATION_FRAME_DURATION_MS = 180
VALID_INIT_MODES = ("random", "jittered_grid", "target_aware")
VALID_SCHEDULERS = ("none", "cosine")
LEGACY_ROOT_ARTIFACTS = (
    "sanity_check.png",
    "target_image.png",
    "reconstruction_image.png",
    "final_comparison.png",
    "loss_curve.png",
    "metrics_card.png",
    "fit_summary.md",
    "gaussian_overlay.png",
    "reconstruction_evolution.gif",
)


@dataclass
class ExperimentConfig:
    device: torch.device
    height: int = 64
    width: int = 64
    num_gaussians: int = 96
    steps: int = 3000
    lr: float = 0.03
    seed: int = 7
    target_path: str | None = None
    target_variant: str = "default_synthetic"
    init_mode: str = "target_aware"
    resolution_aware_init: bool = False
    offset_clamp: float | None = None
    offset_lr_scale: float = 1.0
    scheduler: str = "cosine"
    use_param_groups: bool = True
    make_overlay: bool = False
    make_animation: bool = False
    export_params: bool = True
    export_trajectory_every: int | None = None
    analysis_tag: str | None = None
    output_dir: Path | None = None


@dataclass
class AnimationSnapshot:
    label: str
    loss_value: float
    reconstruction: Tensor


@dataclass
class ParameterSnapshot:
    step: int
    loss_value: float
    offsets: Tensor
    scales: Tensor
    thetas: Tensor
    colors: Tensor
    opacities: Tensor
    importance: Tensor
    parameter_diagnostics: dict[str, float]


@dataclass
class ExperimentResult:
    renderer: Gaussian2DRenderer
    loss_history: list[float]
    runtime_seconds: float
    best_iteration: int
    best_loss: float
    final_loss: float
    best_reconstruction: Tensor
    raw_psnr: float
    clamped_psnr: float
    clamped_best_loss: float
    saturated_value_fraction: float
    saturation_max_excess: float
    parameter_diagnostics: dict[str, float]
    animation_snapshots: list[AnimationSnapshot]
    parameter_snapshots: list[ParameterSnapshot] = field(default_factory=list)


@dataclass
class OutputBundle:
    output_dir: Path
    sanity_path: Path
    target_path: Path
    reconstruction_path: Path
    comparison_path: Path
    loss_curve_path: Path
    metrics_card_path: Path
    summary_path: Path
    overlay_path: Path | None
    animation_path: Path | None
    data_dir: Path | None = None
    learned_params_path: Path | None = None
    learned_params_csv_path: Path | None = None
    importance_ranking_path: Path | None = None
    run_metadata_path: Path | None = None
    sensitivity_summary_path: Path | None = None
    sensitivity_summary_csv_path: Path | None = None
    trajectory_dir: Path | None = None


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


def existing_file_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {value}")
    return str(path.resolve())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a target image with differentiable additive 2D Gaussian primitives.")
    parser.add_argument("--height", type=positive_int, default=64, help="Target image height.")
    parser.add_argument("--width", type=positive_int, default=64, help="Target image width.")
    parser.add_argument("--num-gaussians", type=positive_int, default=96, help="Number of learnable Gaussians.")
    parser.add_argument("--steps", type=positive_int, default=3000, help="Number of optimization iterations.")
    parser.add_argument("--lr", type=positive_float, default=0.03, help="Base Adam learning rate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument(
        "--target-path",
        type=existing_file_path,
        default=None,
        help="Optional path to an RGB target image.",
    )
    parser.add_argument(
        "--target-variant",
        choices=VALID_SYNTHETIC_TARGET_VARIANTS,
        default="default_synthetic",
        help="Built-in synthetic target variant used when --target-path is not provided.",
    )
    parser.add_argument(
        "--init-mode",
        choices=VALID_INIT_MODES,
        default="target_aware",
        help="Initialization mode for Gaussian centers and colors.",
    )
    parser.add_argument(
        "--scheduler",
        choices=VALID_SCHEDULERS,
        default="cosine",
        help="Learning-rate schedule used during fitting.",
    )
    parser.add_argument(
        "--resolution-aware-init",
        action="store_true",
        help="Scale initial Gaussian footprints with image resolution and primitive density.",
    )
    parser.add_argument(
        "--offset-clamp",
        type=positive_float,
        default=None,
        help="Optionally project Gaussian centers into [-VALUE, VALUE] after each fitting step.",
    )
    parser.add_argument(
        "--offset-lr-scale",
        type=positive_float,
        default=1.0,
        help="Relative learning-rate multiplier for Gaussian centers (default: 1.0).",
    )
    parser.add_argument("--make-overlay", action="store_true", help="Save a learned Gaussian overlay figure.")
    parser.add_argument(
        "--make-animation",
        action="store_true",
        help="Save a short reconstruction-evolution GIF.",
    )
    parser.add_argument(
        "--no-export-params",
        action="store_false",
        dest="export_params",
        help="Disable structured final-parameter export under the run data directory.",
    )
    parser.add_argument(
        "--export-trajectory-every",
        type=positive_int,
        default=None,
        help="Optionally save decoded parameter snapshots every K optimization steps.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output directory. Defaults to a new timestamped folder under outputs/.",
    )
    parser.add_argument(
        "--analysis-tag",
        type=str,
        default=None,
        help="Optional short tag recorded in exported run metadata.",
    )
    return parser.parse_args(argv)


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        device=get_best_device(),
        height=args.height,
        width=args.width,
        num_gaussians=args.num_gaussians,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        target_path=args.target_path,
        target_variant=getattr(args, "target_variant", "default_synthetic"),
        init_mode=args.init_mode,
        resolution_aware_init=args.resolution_aware_init,
        offset_clamp=args.offset_clamp,
        offset_lr_scale=args.offset_lr_scale,
        scheduler=args.scheduler,
        use_param_groups=True,
        make_overlay=args.make_overlay,
        make_animation=args.make_animation,
        export_params=getattr(args, "export_params", True),
        export_trajectory_every=getattr(args, "export_trajectory_every", None),
        analysis_tag=getattr(args, "analysis_tag", None),
        output_dir=None if args.output_dir is None else Path(args.output_dir).expanduser().resolve(),
    )


def validate_experiment_config(config: ExperimentConfig) -> None:
    if config.height <= 0:
        raise ValueError("height must be a positive integer")
    if config.width <= 0:
        raise ValueError("width must be a positive integer")
    if config.num_gaussians <= 0:
        raise ValueError("num_gaussians must be a positive integer")
    if config.steps <= 0:
        raise ValueError("steps must be a positive integer")
    if config.lr <= 0.0:
        raise ValueError("lr must be a positive number")
    if config.offset_lr_scale <= 0.0:
        raise ValueError("offset_lr_scale must be a positive number")
    if config.target_variant not in VALID_SYNTHETIC_TARGET_VARIANTS:
        raise ValueError(f"unsupported target_variant: {config.target_variant}")
    if config.init_mode not in VALID_INIT_MODES:
        raise ValueError(f"unsupported init_mode: {config.init_mode}")
    if config.scheduler not in VALID_SCHEDULERS:
        raise ValueError(f"unsupported scheduler: {config.scheduler}")
    if config.export_trajectory_every is not None and config.export_trajectory_every <= 0:
        raise ValueError("export_trajectory_every must be a positive integer when provided")
    if config.target_path is not None and not Path(config.target_path).expanduser().is_file():
        raise ValueError(f"target_path does not exist: {config.target_path}")


@torch.no_grad()
def run_sanity_check(device: torch.device, height: int, width: int, output_dir: Path) -> Path:
    """Render one known Gaussian to verify anisotropic rotation and visibility."""
    grid = create_grid(height, width, device)
    renderer = Gaussian2DRenderer(num_gaussians=1).to(device)
    renderer.set_physical_parameters(
        offsets=torch.tensor([[0.02, -0.03]], dtype=torch.float32, device=device),
        scales=torch.tensor([[0.22, 0.07]], dtype=torch.float32, device=device),
        thetas=torch.tensor([[torch.pi / 4.0]], dtype=torch.float32, device=device),
        colors=torch.tensor([[0.96, 0.42, 0.18]], dtype=torch.float32, device=device),
        opacities=torch.tensor([[0.92]], dtype=torch.float32, device=device),
    )
    sanity_image = renderer(grid[..., 0], grid[..., 1], clamp_output=True)
    sanity_path = output_dir / "sanity_check.png"
    save_tensor_image(sanity_image, sanity_path, title="Sanity Check: One Rotated Gaussian")
    return sanity_path


def load_or_create_target(config: ExperimentConfig, grid: Tensor) -> Tensor:
    """Load an external target or fall back to a synthetic test image."""
    if config.target_path is None:
        return synthetic_target(grid, variant=config.target_variant)
    return load_target_image(Path(config.target_path), height=config.height, width=config.width, device=config.device)


def describe_target_source(config: ExperimentConfig, *, compact: bool = False) -> str:
    if config.target_path is None:
        if compact:
            return f"{config.target_variant} ({config.width}x{config.height})"
        return f"synthetic target variant {config.target_variant}"
    target_path = Path(config.target_path)
    if compact:
        return f"{target_path.name} ({config.width}x{config.height}, exact size)"
    return f"external image {target_path} ({config.width}x{config.height}, exact size)"


@torch.no_grad()
def _offsets_to_pixel_indices(offsets: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
    rows = torch.clamp(((offsets[:, 1] + 1.0) * 0.5 * (height - 1)).round().long(), 0, height - 1)
    cols = torch.clamp(((offsets[:, 0] + 1.0) * 0.5 * (width - 1)).round().long(), 0, width - 1)
    return rows, cols


@torch.no_grad()
def initialize_renderer(
    renderer: Gaussian2DRenderer,
    target: Tensor,
    grid_x: Tensor,
    grid_y: Tensor,
    config: ExperimentConfig,
) -> None:
    """Apply optional smarter initialization without changing the renderer core."""
    if config.init_mode == "random":
        return

    num_gaussians = renderer.num_gaussians
    device = target.device
    height, width = target.shape[:2]

    if config.init_mode == "jittered_grid":
        num_cols = math.ceil(math.sqrt(num_gaussians))
        num_rows = math.ceil(num_gaussians / num_cols)
        xs = torch.linspace(-0.85, 0.85, steps=num_cols, device=device)
        ys = torch.linspace(-0.85, 0.85, steps=num_rows, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        offsets = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)[:num_gaussians]

        cell_x = 1.7 / max(num_cols - 1, 1)
        cell_y = 1.7 / max(num_rows - 1, 1)
        jitter_x = torch.empty(num_gaussians, device=device).uniform_(-0.18 * cell_x, 0.18 * cell_x)
        jitter_y = torch.empty(num_gaussians, device=device).uniform_(-0.18 * cell_y, 0.18 * cell_y)
        offsets[:, 0] = torch.clamp(offsets[:, 0] + jitter_x, -0.95, 0.95)
        offsets[:, 1] = torch.clamp(offsets[:, 1] + jitter_y, -0.95, 0.95)

        rows, cols = _offsets_to_pixel_indices(offsets, height, width)
        colors = target[rows, cols]
        opacities = torch.clamp(0.25 + 0.65 * colors.mean(dim=-1, keepdim=True), 0.15, 0.85)
        scale_floor = 2.0 / max(height - 1, width - 1) if config.resolution_aware_init else 0.05
        scales = torch.stack(
            (
                torch.full((num_gaussians,), max(scale_floor, 0.55 * cell_x), device=device),
                torch.full((num_gaussians,), max(scale_floor, 0.55 * cell_y), device=device),
            ),
            dim=-1,
        )
        thetas = torch.empty(num_gaussians, 1, device=device).uniform_(-torch.pi, torch.pi)
    elif config.init_mode == "target_aware":
        luminance = 0.2126 * target[..., 0] + 0.7152 * target[..., 1] + 0.0722 * target[..., 2]
        sampling_weights = luminance.reshape(-1) + 1e-3
        flat_indices = torch.multinomial(sampling_weights, num_samples=num_gaussians, replacement=True)
        rows = flat_indices // width
        cols = flat_indices % width

        offsets = torch.stack((grid_x[rows, cols], grid_y[rows, cols]), dim=-1)
        pixel_x = 2.0 / max(width - 1, 1)
        pixel_y = 2.0 / max(height - 1, 1)
        offsets[:, 0] = torch.clamp(
            offsets[:, 0] + torch.empty(num_gaussians, device=device).uniform_(-0.35 * pixel_x, 0.35 * pixel_x),
            -0.98,
            0.98,
        )
        offsets[:, 1] = torch.clamp(
            offsets[:, 1] + torch.empty(num_gaussians, device=device).uniform_(-0.35 * pixel_y, 0.35 * pixel_y),
            -0.98,
            0.98,
        )

        colors = torch.clamp(
            target[rows, cols] + torch.empty(num_gaussians, 3, device=device).uniform_(-0.03, 0.03),
            0.0,
            1.0,
        )
        local_luminance = luminance[rows, cols].unsqueeze(-1)
        opacities = torch.clamp(0.18 + 0.72 * local_luminance, 0.12, 0.92)

        scale_floor = 2.0 / max(height - 1, width - 1) if config.resolution_aware_init else 0.045
        coverage_scale = max(scale_floor, 0.58 * 2.0 / math.sqrt(num_gaussians))
        scales = coverage_scale * torch.empty(num_gaussians, 2, device=device).uniform_(0.70, 1.10)
        thetas = torch.empty(num_gaussians, 1, device=device).uniform_(-torch.pi, torch.pi)
    else:
        raise ValueError(f"unsupported init_mode: {config.init_mode}")

    renderer.set_physical_parameters(
        offsets=offsets,
        scales=scales,
        thetas=thetas,
        colors=colors,
        opacities=opacities,
    )


def build_optimizer(renderer: Gaussian2DRenderer, config: ExperimentConfig) -> torch.optim.Optimizer:
    if not config.use_param_groups:
        return torch.optim.Adam(renderer.parameters(), lr=config.lr)

    return torch.optim.Adam(
        [
            {"params": [renderer.raw_offsets], "lr": config.lr * config.offset_lr_scale},
            {"params": [renderer.raw_scales], "lr": config.lr * 0.60},
            {"params": [renderer.raw_thetas], "lr": config.lr * 0.35},
            {"params": [renderer.raw_colors], "lr": config.lr * 1.10},
            {"params": [renderer.raw_opacities], "lr": config.lr * 0.60},
        ]
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps, eta_min=config.lr * 0.10)
    if config.scheduler == "none":
        return None
    raise ValueError(f"unsupported scheduler: {config.scheduler}")


def select_animation_capture_steps(total_steps: int, max_frames: int) -> set[int]:
    if total_steps <= 0 or max_frames <= 0:
        return set()
    if total_steps <= max_frames:
        return set(range(total_steps))
    if max_frames == 1:
        return {0}
    return {
        int(round(frame_index * (total_steps - 1) / (max_frames - 1)))
        for frame_index in range(max_frames)
    }


@torch.no_grad()
def gradients_are_finite(renderer: Gaussian2DRenderer) -> bool:
    for parameter in renderer.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return False
    return True


@torch.no_grad()
def copy_state_dict_to_cpu(renderer: Gaussian2DRenderer) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in renderer.state_dict().items()}


@torch.no_grad()
def compute_psnr(target: Tensor, reconstruction: Tensor) -> float:
    mse = torch.mean((target - reconstruction) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def compute_saturation_statistics(reconstruction: Tensor) -> tuple[float, float]:
    over = torch.clamp(reconstruction - 1.0, min=0.0)
    under = torch.clamp(-reconstruction, min=0.0)
    excess = torch.maximum(over, under)
    saturated_fraction = float((excess > 0.0).float().mean().item())
    max_excess = float(excess.max().item())
    return saturated_fraction, max_excess


@torch.no_grad()
def collect_parameter_snapshot(
    renderer: Gaussian2DRenderer,
    *,
    step: int,
    loss_value: float,
) -> ParameterSnapshot:
    """Capture a decoded parameter snapshot on CPU for later export."""
    params = renderer.decode_parameters()
    offsets_cpu = params["offsets"].detach().cpu().clone()
    scales_cpu = params["scales"].detach().cpu().clone()
    thetas_cpu = params["thetas"].detach().cpu().clone()
    colors_cpu = params["colors"].detach().cpu().clone()
    opacities_cpu = params["opacities"].detach().cpu().clone()
    importance_cpu = compute_gaussian_importance(scales_cpu, opacities_cpu)
    return ParameterSnapshot(
        step=step,
        loss_value=loss_value,
        offsets=offsets_cpu,
        scales=scales_cpu,
        thetas=thetas_cpu,
        colors=colors_cpu,
        opacities=opacities_cpu,
        importance=importance_cpu,
        parameter_diagnostics=compute_parameter_diagnostics(offsets_cpu, scales_cpu),
    )


@torch.no_grad()
def _parameter_export_dict(
    *,
    config: ExperimentConfig,
    result: ExperimentResult,
    target_descriptor: str,
    target: Tensor,
) -> dict[str, object]:
    params = result.renderer.decode_parameters()
    offsets_cpu = params["offsets"].detach().cpu()
    scales_cpu = params["scales"].detach().cpu()
    thetas_cpu = params["thetas"].detach().cpu()
    colors_cpu = params["colors"].detach().cpu()
    opacities_cpu = params["opacities"].detach().cpu()
    importance_cpu = compute_gaussian_importance(scales_cpu, opacities_cpu)
    canonical_cpu = canonicalize_gaussian_geometry(scales_cpu, thetas_cpu)
    grid = create_grid(config.height, config.width, config.device)
    with torch.enable_grad():
        sensitivity = compute_parameter_sensitivity_summary(
            target=target.to(config.device),
            grid_x=grid[..., 0],
            grid_y=grid[..., 1],
            offsets=params["offsets"],
            scales=params["scales"],
            thetas=params["thetas"],
            colors=params["colors"],
            opacities=params["opacities"],
        )
    sensitivity_cpu = {key: value.detach().cpu() for key, value in sensitivity.items()}
    return {
        "schema_version": 3,
        "artifact_type": "clean_fit_export",
        "metrics_kind": "fit_metrics",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "device": str(config.device),
            "height": config.height,
            "width": config.width,
            "num_gaussians": config.num_gaussians,
            "steps": config.steps,
            "lr": config.lr,
            "seed": config.seed,
            "target_path": config.target_path,
            "target_descriptor": target_descriptor,
            "target_type": "synthetic" if config.target_path is None else "external",
            "target_variant": config.target_variant if config.target_path is None else "external",
            "init_mode": config.init_mode,
            "resolution_aware_init": config.resolution_aware_init,
            "offset_clamp": config.offset_clamp,
            "offset_lr_scale": config.offset_lr_scale,
            "scheduler": config.scheduler,
            "export_params": config.export_params,
            "export_trajectory_every": config.export_trajectory_every,
            "analysis_tag": config.analysis_tag,
            "canonicalization_version": 1,
            "sensitivity_probe": "recon_loss_grad_l2_proxy_in_physical_coordinates",
        },
        "metrics": {
            "runtime_seconds": result.runtime_seconds,
            "best_iteration": result.best_iteration,
            "best_loss": result.best_loss,
            "final_loss": result.final_loss,
            "raw_psnr": result.raw_psnr,
            "clamped_psnr": result.clamped_psnr,
            "clamped_best_loss": result.clamped_best_loss,
            "saturated_value_fraction": result.saturated_value_fraction,
            "saturation_max_excess": result.saturation_max_excess,
            "parameter_diagnostics": result.parameter_diagnostics,
        },
        "parameters": {
            "offsets": offsets_cpu,
            "scales": scales_cpu,
            "thetas": thetas_cpu,
            "colors": colors_cpu,
            "opacities": opacities_cpu,
            "importance": importance_cpu,
        },
        "canonical_parameters": canonical_cpu,
        "sensitivity": sensitivity_cpu,
    }


@torch.no_grad()
def export_run_data(
    *,
    config: ExperimentConfig,
    result: ExperimentResult,
    target_descriptor: str,
    output_dir: Path,
    target: Tensor,
) -> tuple[
    Path | None,
    Path | None,
    Path | None,
    Path | None,
    Path | None,
    Path | None,
    Path | None,
    Path | None,
]:
    """Export structured run data for downstream analysis and watermarking."""
    if not config.export_params and not result.parameter_snapshots:
        return None, None, None, None, None, None, None, None

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    export_dict = _parameter_export_dict(
        config=config,
        result=result,
        target_descriptor=target_descriptor,
        target=target,
    )

    learned_params_path = None
    learned_params_csv_path = None
    importance_ranking_path = None
    run_metadata_path = data_dir / "run_metadata.json"
    sensitivity_summary_path = data_dir / "sensitivity_summary.pt"
    sensitivity_summary_csv_path = data_dir / "sensitivity_summary.csv"

    params = export_dict["parameters"]
    canonical = export_dict["canonical_parameters"]
    sensitivity = export_dict["sensitivity"]
    offsets_cpu = params["offsets"]
    scales_cpu = params["scales"]
    thetas_cpu = params["thetas"]
    colors_cpu = params["colors"]
    opacities_cpu = params["opacities"]
    importance_cpu = params["importance"]

    if config.export_params:
        learned_params_path = data_dir / "learned_params.pt"
        torch.save(export_dict, learned_params_path)
        learned_params_csv_path = data_dir / "learned_params.csv"
        export_parameter_table(
            offsets=offsets_cpu,
            scales=scales_cpu,
            thetas=thetas_cpu,
            colors=colors_cpu,
            opacities=opacities_cpu,
            importance=importance_cpu,
            height=config.height,
            width=config.width,
            path=learned_params_csv_path,
            sensitivity=sensitivity,
        )
        importance_ranking_path = data_dir / "importance_ranking.csv"
        ranking = torch.argsort(importance_cpu, descending=True)
        importance_ranking_lines = ["gaussian_id,importance_rank,importance"]
        for rank, gaussian_index in enumerate(ranking.tolist(), start=1):
            importance_ranking_lines.append(
                f"{gaussian_index},{rank},{float(importance_cpu[gaussian_index].item()):.10f}"
            )
        importance_ranking_path.write_text("\n".join(importance_ranking_lines) + "\n", encoding="utf-8")
        torch.save(
            {
                "schema_version": 2,
                "gaussian_id": torch.arange(offsets_cpu.shape[0], dtype=torch.long),
                **{key: value.detach().cpu() for key, value in sensitivity.items()},
                "signed_logratio": canonical["signed_logratio"].detach().cpu(),
                "log_anisotropy": canonical["log_anisotropy"].detach().cpu(),
            },
            sensitivity_summary_path,
        )
        sensitivity_summary_csv_lines = [
            "gaussian_id,sens_mu_x,sens_mu_y,sens_log_anisotropy,sens_log_area,sens_theta_major,sens_alpha,signed_logratio,log_anisotropy"
        ]
        for index in range(offsets_cpu.shape[0]):
            sensitivity_summary_csv_lines.append(
                ",".join(
                    [
                        str(index),
                        f"{float(sensitivity['mu_x'][index].item()):.10f}",
                        f"{float(sensitivity['mu_y'][index].item()):.10f}",
                        f"{float(sensitivity['log_anisotropy'][index].item()):.10f}",
                        f"{float(sensitivity['log_area'][index].item()):.10f}",
                        f"{float(sensitivity['theta_major'][index].item()):.10f}",
                        f"{float(sensitivity['alpha'][index].item()):.10f}",
                        f"{float(canonical['signed_logratio'][index].item()):.10f}",
                        f"{float(canonical['log_anisotropy'][index].item()):.10f}",
                    ]
                )
            )
        sensitivity_summary_csv_path.write_text("\n".join(sensitivity_summary_csv_lines) + "\n", encoding="utf-8")

    save_json(
        {
            "schema_version": export_dict["schema_version"],
            "artifact_type": export_dict["artifact_type"],
            "metrics_kind": export_dict["metrics_kind"],
            "exported_at_utc": export_dict["exported_at_utc"],
            "config": export_dict["config"],
            "metrics": export_dict["metrics"],
        },
        run_metadata_path,
    )

    trajectory_dir = None
    if result.parameter_snapshots:
        trajectory_dir = output_dir / "trajectory"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        manifest_entries: list[dict[str, object]] = []
        for snapshot in result.parameter_snapshots:
            snapshot_path = trajectory_dir / f"step_{snapshot.step:06d}.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "step": snapshot.step,
                    "loss_value": snapshot.loss_value,
                    "offsets": snapshot.offsets,
                    "scales": snapshot.scales,
                    "thetas": snapshot.thetas,
                    "colors": snapshot.colors,
                    "opacities": snapshot.opacities,
                    "importance": snapshot.importance,
                    "parameter_diagnostics": snapshot.parameter_diagnostics,
                },
                snapshot_path,
            )
            manifest_entries.append(
                {
                    "step": snapshot.step,
                    "loss_value": snapshot.loss_value,
                    "path": snapshot_path.name,
                }
            )
        save_json(
            {
                "schema_version": 1,
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "steps": manifest_entries,
            },
            trajectory_dir / "manifest.json",
        )

    return (
        data_dir,
        learned_params_path,
        learned_params_csv_path,
        importance_ranking_path,
        run_metadata_path,
        sensitivity_summary_path if config.export_params else None,
        sensitivity_summary_csv_path if config.export_params else None,
        trajectory_dir,
    )


def fit_gaussians(
    target: Tensor,
    grid_x: Tensor,
    grid_y: Tensor,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Optimize a set of Gaussians to fit a target image."""
    validate_experiment_config(config)
    renderer = Gaussian2DRenderer(num_gaussians=config.num_gaussians).to(config.device)
    initialize_renderer(renderer, target, grid_x, grid_y, config)
    if config.offset_clamp is not None:
        with torch.no_grad():
            renderer.raw_offsets.clamp_(-config.offset_clamp, config.offset_clamp)

    optimizer = build_optimizer(renderer, config)
    scheduler = build_scheduler(optimizer, config)

    loss_history: list[float] = []
    best_loss = float("inf")
    best_iteration = -1
    best_state_dict = copy_state_dict_to_cpu(renderer)
    target_cpu = target.detach().cpu()
    animation_snapshots: list[AnimationSnapshot] = []
    parameter_snapshots: list[ParameterSnapshot] = []
    capture_steps = select_animation_capture_steps(config.steps, ANIMATION_MAX_FRAMES) if config.make_animation else set()

    start_time = time.perf_counter()
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = renderer(grid_x, grid_y, clamp_output=False, chunk_size=None)
        loss = F.mse_loss(reconstruction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(renderer.parameters(), max_norm=DEFAULT_GRAD_CLIP)
        if not gradients_are_finite(renderer):
            raise FloatingPointError("Encountered NaN or Inf gradients during training.")

        loss_value = float(loss.detach().cpu())
        loss_history.append(loss_value)

        if loss_value < best_loss:
            best_loss = loss_value
            best_iteration = step
            best_state_dict = copy_state_dict_to_cpu(renderer)
        if step in capture_steps:
            animation_snapshots.append(
                AnimationSnapshot(
                    label=f"step {step}",
                    loss_value=loss_value,
                    reconstruction=reconstruction.detach().cpu(),
                )
            )
        if config.export_trajectory_every is not None and (
            step == 0 or step == config.steps - 1 or step % config.export_trajectory_every == 0
        ):
            parameter_snapshots.append(
                collect_parameter_snapshot(
                    renderer,
                    step=step,
                    loss_value=loss_value,
                )
            )

        optimizer.step()
        if config.offset_clamp is not None:
            with torch.no_grad():
                renderer.raw_offsets.clamp_(-config.offset_clamp, config.offset_clamp)
        if scheduler is not None:
            scheduler.step()

        if step % 100 == 0 or step == config.steps - 1:
            print(f"[Iter {step:04d}/{config.steps}] loss = {loss_value:.8f}")

    runtime_seconds = time.perf_counter() - start_time
    renderer.load_state_dict(best_state_dict)
    best_reconstruction = renderer(grid_x, grid_y, clamp_output=False, chunk_size=None).detach().cpu()
    clamped_best_reconstruction = best_reconstruction.clamp(0.0, 1.0)
    raw_psnr = compute_psnr(target_cpu, best_reconstruction)
    clamped_psnr = compute_psnr(target_cpu, clamped_best_reconstruction)
    clamped_best_loss = float(torch.mean((target_cpu - clamped_best_reconstruction) ** 2).item())
    saturated_value_fraction, saturation_max_excess = compute_saturation_statistics(best_reconstruction)
    params = renderer.decode_parameters()
    parameter_diagnostics = compute_parameter_diagnostics(params["offsets"], params["scales"])
    if config.make_animation:
        animation_snapshots.append(
            AnimationSnapshot(
                label=f"best @ {best_iteration}",
                loss_value=best_loss,
                reconstruction=best_reconstruction,
            )
        )
    return ExperimentResult(
        renderer=renderer,
        loss_history=loss_history,
        runtime_seconds=runtime_seconds,
        best_iteration=best_iteration,
        best_loss=best_loss,
        final_loss=loss_history[-1],
        best_reconstruction=best_reconstruction,
        raw_psnr=raw_psnr,
        clamped_psnr=clamped_psnr,
        clamped_best_loss=clamped_best_loss,
        saturated_value_fraction=saturated_value_fraction,
        saturation_max_excess=saturation_max_excess,
        parameter_diagnostics=parameter_diagnostics,
        animation_snapshots=animation_snapshots,
        parameter_snapshots=parameter_snapshots,
    )


def save_fit_summary(
    config: ExperimentConfig,
    result: ExperimentResult,
    path: Path,
    target_descriptor: str,
    overlay_generated: bool,
    animation_generated: bool,
    data_exported: bool = False,
    trajectory_exported: bool = False,
) -> None:
    """Save a compact experiment summary with raw/clamped diagnostics."""
    summary = "\n".join(
        [
            "# Fit Summary",
            "",
            "## Configuration",
            f"- Device: {config.device}",
            f"- Resolution: {config.height}x{config.width}",
            f"- Number of Gaussians: {config.num_gaussians}",
            f"- Optimization steps: {config.steps}",
            f"- Adam learning rate: {config.lr}",
            f"- Seed: {config.seed}",
            f"- Target source: {target_descriptor}",
            f"- Analysis tag: {config.analysis_tag or 'none'}",
            f"- Initialization mode: {config.init_mode}",
            f"- Resolution-aware initialization: {'enabled' if config.resolution_aware_init else 'disabled'}",
            f"- Offset projection: {config.offset_clamp if config.offset_clamp is not None else 'disabled'}",
            f"- Offset learning-rate scale: {config.offset_lr_scale:.3f}",
            f"- Scheduler: {config.scheduler}",
            f"- Structured parameter export: {'enabled' if config.export_params else 'disabled'}",
            (
                f"- Parameter trajectory export: every {config.export_trajectory_every} steps"
                if config.export_trajectory_every is not None
                else "- Parameter trajectory export: disabled"
            ),
            "- Compositing model: additive Gaussian accumulation (not visibility-aware splatting)",
            "",
            "## Optimization Outcome",
            f"- Initial loss: {result.loss_history[0]:.8f}",
            f"- Final raw MSE: {result.final_loss:.8f}",
            f"- Best raw MSE: {result.best_loss:.8f}",
            f"- Best clamped MSE: {result.clamped_best_loss:.8f}",
            f"- Best iteration: {result.best_iteration}",
            f"- Raw PSNR: {result.raw_psnr:.2f} dB",
            f"- Clamped PSNR: {result.clamped_psnr:.2f} dB",
            f"- Saturated value fraction: {100.0 * result.saturated_value_fraction:.3f}%",
            f"- Max saturation excess: {result.saturation_max_excess:.4f}",
            "",
            "## Timing",
            f"- Total runtime: {result.runtime_seconds:.2f} seconds",
            f"- Mean time per step: {1000.0 * result.runtime_seconds / max(1, config.steps):.3f} ms",
            "",
            "## Interpretation Notes",
            "- Default runs use a built-in synthetic target unless --target-path is provided.",
            "- External target images must already match the requested resolution; no implicit resize is applied.",
            "- The raw model output can exceed [0, 1] because the renderer is additive; clamped metrics are reported separately for display honesty.",
            "- Canonical geometry uses the major axis with theta wrapped to [0, pi).",
            "- Sensitivity export uses a reconstruction-gradient proxy in physical parameter space.",
            "- This summary is not evidence of cross-device reproducibility; PyTorch only offers limited same-environment determinism guarantees.",
            "- This experiment is evidence for this specific image-space fit only, not a general image-representation guarantee.",
            "",
            "## Parameter Diagnostics",
            f"- Off-canvas Gaussian centers: {int(result.parameter_diagnostics['off_canvas_count'])}",
            f"- Gaussians with scale > 1.0: {int(result.parameter_diagnostics['large_scale_count'])}",
            f"- Max |offset|: {result.parameter_diagnostics['max_abs_offset']:.4f}",
            f"- Max scale entry: {result.parameter_diagnostics['max_scale']:.4f}",
            "",
            "## Generated Files",
            f"- Directory: {path.parent}",
            "- sanity_check.png",
            "- target_image.png",
            "- reconstruction_image.png",
            "- final_comparison.png",
            "- loss_curve.png",
            "- metrics_card.png",
            f"- gaussian_overlay.png: {'yes' if overlay_generated else 'no'}",
            f"- reconstruction_evolution.gif: {'yes' if animation_generated else 'no'}",
            f"- data/learned_params.pt: {'yes' if data_exported else 'no'}",
            f"- data/learned_params.csv: {'yes' if data_exported else 'no'}",
            f"- data/importance_ranking.csv: {'yes' if data_exported else 'no'}",
            "- data/run_metadata.json",
            f"- data/sensitivity_summary.pt: {'yes' if data_exported else 'no'}",
            f"- data/sensitivity_summary.csv: {'yes' if data_exported else 'no'}",
            f"- trajectory/: {'yes' if trajectory_exported else 'no'}",
            "",
        ]
    )
    path.write_text(summary, encoding="utf-8")


@torch.no_grad()
def save_outputs(
    config: ExperimentConfig,
    target: Tensor,
    result: ExperimentResult,
    sanity_path: Path,
    output_dir: Path,
) -> OutputBundle:
    target_path = output_dir / "target_image.png"
    reconstruction_path = output_dir / "reconstruction_image.png"
    comparison_path = output_dir / "final_comparison.png"
    loss_curve_path = output_dir / "loss_curve.png"
    metrics_card_path = output_dir / "metrics_card.png"
    summary_path = output_dir / "fit_summary.md"
    overlay_path = output_dir / "gaussian_overlay.png" if config.make_overlay else None
    animation_path = output_dir / "reconstruction_evolution.gif" if config.make_animation else None

    save_raw_tensor_image(target, target_path)
    save_raw_tensor_image(result.best_reconstruction.clamp(0.0, 1.0), reconstruction_path)
    save_comparison_figure(target, result.best_reconstruction, comparison_path)
    save_loss_curve(result.loss_history, loss_curve_path)
    target_descriptor = describe_target_source(config)
    data_dir = None
    learned_params_path = None
    learned_params_csv_path = None
    importance_ranking_path = None
    run_metadata_path = None
    sensitivity_summary_path = None
    sensitivity_summary_csv_path = None
    trajectory_dir = None
    save_metrics_card(
        title="Fit Metrics",
        metric_rows=[
            ("Target", describe_target_source(config, compact=True)),
            ("Best iteration", str(result.best_iteration)),
            ("Best raw MSE", f"{result.best_loss:.8f}"),
            ("Best clamped MSE", f"{result.clamped_best_loss:.8f}"),
            ("Raw PSNR", f"{result.raw_psnr:.2f} dB"),
            ("Clamped PSNR", f"{result.clamped_psnr:.2f} dB"),
            ("Saturated values", f"{100.0 * result.saturated_value_fraction:.3f}%"),
            ("Max saturation excess", f"{result.saturation_max_excess:.4f}"),
            ("Off-canvas centers", str(int(result.parameter_diagnostics["off_canvas_count"]))),
            ("Scale > 1.0", str(int(result.parameter_diagnostics["large_scale_count"]))),
            ("Runtime", f"{result.runtime_seconds:.2f} s"),
        ],
        notes=(
            "Additive Gaussian accumulation, not visibility-aware splatting",
            "External targets are exact-size only; no implicit resize",
            "Image-space fit evidence only, not general image representation",
        ),
        path=metrics_card_path,
    )

    if overlay_path is not None:
        params = result.renderer.decode_parameters()
        save_gaussian_overlay(
            background=result.best_reconstruction,
            offsets=params["offsets"],
            scales=params["scales"],
            thetas=params["thetas"],
            colors=params["colors"],
            opacities=params["opacities"],
            path=overlay_path,
            top_k=OVERLAY_TOP_K,
        )
    if animation_path is not None:
        save_reconstruction_evolution_gif(
            target=target,
            frames=[(snapshot.label, snapshot.loss_value, snapshot.reconstruction) for snapshot in result.animation_snapshots],
            path=animation_path,
            frame_duration_ms=ANIMATION_FRAME_DURATION_MS,
        )
    (
        data_dir,
        learned_params_path,
        learned_params_csv_path,
        importance_ranking_path,
        run_metadata_path,
        sensitivity_summary_path,
        sensitivity_summary_csv_path,
        trajectory_dir,
    ) = export_run_data(
        config=config,
        result=result,
        target_descriptor=target_descriptor,
        output_dir=output_dir,
        target=target,
    )

    save_fit_summary(
        config=config,
        result=result,
        path=summary_path,
        target_descriptor=target_descriptor,
        overlay_generated=overlay_path is not None,
        animation_generated=animation_path is not None,
        data_exported=learned_params_path is not None,
        trajectory_exported=trajectory_dir is not None,
    )

    return OutputBundle(
        output_dir=output_dir,
        sanity_path=sanity_path,
        target_path=target_path,
        reconstruction_path=reconstruction_path,
        comparison_path=comparison_path,
        loss_curve_path=loss_curve_path,
        metrics_card_path=metrics_card_path,
        summary_path=summary_path,
        overlay_path=overlay_path,
        animation_path=animation_path,
        data_dir=data_dir,
        learned_params_path=learned_params_path,
        learned_params_csv_path=learned_params_csv_path,
        importance_ranking_path=importance_ranking_path,
        run_metadata_path=run_metadata_path,
        sensitivity_summary_path=sensitivity_summary_path,
        sensitivity_summary_csv_path=sensitivity_summary_csv_path,
        trajectory_dir=trajectory_dir,
    )


def run_experiment(config: ExperimentConfig) -> tuple[Tensor, ExperimentResult, OutputBundle]:
    validate_experiment_config(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    archive_legacy_output_files(OUTPUT_DIR, LEGACY_ROOT_ARTIFACTS)
    output_dir = config.output_dir if config.output_dir is not None else create_run_output_dir(OUTPUT_DIR, "run_fit")
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = create_grid(config.height, config.width, config.device)
    grid_x = grid[..., 0]
    grid_y = grid[..., 1]

    sanity_path = run_sanity_check(config.device, config.height, config.width, output_dir)
    target = load_or_create_target(config, grid)
    result = fit_gaussians(target=target, grid_x=grid_x, grid_y=grid_y, config=config)
    output_bundle = save_outputs(
        config=config,
        target=target,
        result=result,
        sanity_path=sanity_path,
        output_dir=output_dir,
    )
    return target, result, output_bundle


def main() -> None:
    args = parse_args()
    config = build_experiment_config(args)
    set_seed(config.seed, deterministic=True)
    try:
        _, result, outputs = run_experiment(config)
    except (ImportError, OSError, ValueError, FloatingPointError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print()
    print("Project finished successfully.")
    print(f"Device: {config.device}")
    print(f"Output directory: {outputs.output_dir}")
    print(f"Target source: {describe_target_source(config)}")
    print(f"Final raw MSE: {result.final_loss:.8f}")
    print(f"Best raw MSE: {result.best_loss:.8f}")
    print(f"Best clamped MSE: {result.clamped_best_loss:.8f}")
    print(f"Best iteration: {result.best_iteration}")
    print(f"Raw PSNR: {result.raw_psnr:.2f} dB")
    print(f"Clamped PSNR: {result.clamped_psnr:.2f} dB")
    print(f"Saturated value fraction: {100.0 * result.saturated_value_fraction:.3f}%")
    print(f"Max saturation excess: {result.saturation_max_excess:.4f}")
    print(f"Off-canvas Gaussian centers: {int(result.parameter_diagnostics['off_canvas_count'])}")
    print(f"Gaussians with scale > 1.0: {int(result.parameter_diagnostics['large_scale_count'])}")
    print(f"Max |offset|: {result.parameter_diagnostics['max_abs_offset']:.4f}")
    print(f"Max scale entry: {result.parameter_diagnostics['max_scale']:.4f}")
    print(f"Total runtime: {result.runtime_seconds:.2f} seconds")
    print("Compositing model: additive Gaussian accumulation (not visibility-aware splatting)")
    print(f"Sanity check image: {outputs.sanity_path}")
    print(f"Target image: {outputs.target_path}")
    print(f"Reconstruction image: {outputs.reconstruction_path}")
    print(f"Comparison figure: {outputs.comparison_path}")
    print(f"Loss curve: {outputs.loss_curve_path}")
    print(f"Metrics card: {outputs.metrics_card_path}")
    if outputs.overlay_path is not None:
        print(f"Gaussian overlay: {outputs.overlay_path}")
    if outputs.animation_path is not None:
        print(f"Reconstruction animation: {outputs.animation_path}")
    if outputs.learned_params_path is not None:
        print(f"Learned parameters (.pt): {outputs.learned_params_path}")
    if outputs.learned_params_csv_path is not None:
        print(f"Learned parameters (.csv): {outputs.learned_params_csv_path}")
    if outputs.importance_ranking_path is not None:
        print(f"Importance ranking: {outputs.importance_ranking_path}")
    if outputs.run_metadata_path is not None:
        print(f"Run metadata: {outputs.run_metadata_path}")
    if outputs.trajectory_dir is not None:
        print(f"Parameter trajectory: {outputs.trajectory_dir}")
    print(f"Fit summary: {outputs.summary_path}")


if __name__ == "__main__":
    main()
