from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_SCALE_EPS = 1e-6
DEFAULT_DETERMINANT_EPS = 0.0
INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _inverse_softplus(value: Tensor, eps: float = 1e-6) -> Tensor:
    """Map a positive target value back to the raw domain of softplus."""
    adjusted = torch.clamp(value - eps, min=eps)
    return torch.log(torch.expm1(adjusted))


def _safe_logit(value: Tensor, eps: float = 1e-6) -> Tensor:
    """Map a bounded probability in [0, 1] back to the raw logit domain."""
    clamped = torch.clamp(value, min=eps, max=1.0 - eps)
    return torch.log(clamped / (1.0 - clamped))


class Gaussian2DRenderer(nn.Module):
    """Differentiable additive 2D Gaussian renderer implemented in pure PyTorch.

    Each Gaussian is parameterized by:
    - offset:   (mu_x, mu_y)
    - scale:    (s_x, s_y)
    - rotation: theta
    - color:    RGB
    - opacity:  alpha

    The renderer uses additive compositing:
        image(x) = sum_i alpha_i * G_i(x) * color_i

    where
        G_i(x) = exp(-0.5 * (x - mu_i)^T Sigma_i^{-1} (x - mu_i))

    The Gaussian normalization constant is intentionally omitted because we only
    need a smooth spatial weighting kernel for image reconstruction.
    """

    def __init__(
        self,
        num_gaussians: int,
        scale_eps: float = DEFAULT_SCALE_EPS,
        determinant_eps: float = DEFAULT_DETERMINANT_EPS,
    ) -> None:
        super().__init__()
        if num_gaussians <= 0:
            raise ValueError("num_gaussians must be a positive integer")
        self.num_gaussians = num_gaussians
        self.scale_eps = scale_eps
        self.determinant_eps = determinant_eps

        offset_init = torch.empty(num_gaussians, 2).uniform_(-0.8, 0.8)
        target_scales = torch.empty(num_gaussians, 2).uniform_(0.06, 0.18)
        raw_scale_init = _inverse_softplus(target_scales, eps=scale_eps)
        theta_init = torch.empty(num_gaussians, 1).uniform_(-math.pi, math.pi)
        color_logit_init = torch.empty(num_gaussians, 3).uniform_(-0.5, 0.5)
        opacity_logit_init = torch.empty(num_gaussians, 1).uniform_(-1.0, 0.0)

        self.raw_offsets = nn.Parameter(offset_init)
        self.raw_scales = nn.Parameter(raw_scale_init)
        self.raw_thetas = nn.Parameter(theta_init)
        self.raw_colors = nn.Parameter(color_logit_init)
        self.raw_opacities = nn.Parameter(opacity_logit_init)
        self.register_buffer("exact_color_values", torch.full((num_gaussians, 3), float("nan")))
        self.register_buffer("exact_color_raw_snapshots", torch.full((num_gaussians, 3), float("nan")))
        self.register_buffer("exact_opacity_values", torch.full((num_gaussians, 1), float("nan")))
        self.register_buffer("exact_opacity_raw_snapshots", torch.full((num_gaussians, 1), float("nan")))

    def decode_parameters(self) -> Dict[str, Tensor]:
        """Decode raw learnable parameters into physically valid values."""
        offsets, scales, thetas, colors, opacities = self._decode_parameter_tensors()
        return {
            "offsets": offsets,
            "scales": scales,
            "thetas": thetas,
            "colors": colors,
            "opacities": opacities,
        }

    def _decode_parameter_tensors(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Decode raw learnable parameters without building a dictionary."""
        offsets = self.raw_offsets
        scales = F.softplus(self.raw_scales) + self.scale_eps
        thetas = self.raw_thetas
        colors = torch.sigmoid(self.raw_colors)
        opacities = torch.sigmoid(self.raw_opacities)
        color_exact_mask = torch.isfinite(self.exact_color_values) & torch.eq(self.raw_colors, self.exact_color_raw_snapshots)
        opacity_exact_mask = torch.isfinite(self.exact_opacity_values) & torch.eq(
            self.raw_opacities,
            self.exact_opacity_raw_snapshots,
        )
        colors = torch.where(color_exact_mask, self.exact_color_values, colors)
        opacities = torch.where(opacity_exact_mask, self.exact_opacity_values, opacities)
        return offsets, scales, thetas, colors, opacities

    @torch.no_grad()
    def _selected_row_indices(self, indices: Tensor | list[int] | tuple[int, ...] | slice | int | None) -> Tensor:
        if indices is None:
            return torch.arange(self.num_gaussians, device=self.raw_offsets.device, dtype=torch.long)
        if isinstance(indices, slice):
            return torch.arange(self.num_gaussians, device=self.raw_offsets.device, dtype=torch.long)[indices]
        if isinstance(indices, int):
            normalized = indices + self.num_gaussians if indices < 0 else indices
            if not 0 <= normalized < self.num_gaussians:
                raise IndexError(f"index {indices} is out of bounds for {self.num_gaussians} Gaussians")
            return torch.tensor([normalized], device=self.raw_offsets.device, dtype=torch.long)

        index_tensor = torch.as_tensor(indices, device=self.raw_offsets.device)
        if index_tensor.dtype == torch.bool:
            if index_tensor.ndim != 1 or index_tensor.shape[0] != self.num_gaussians:
                raise ValueError(f"boolean mask indices must have shape [{self.num_gaussians}]")
            return torch.arange(self.num_gaussians, device=self.raw_offsets.device, dtype=torch.long)[index_tensor]
        if index_tensor.dtype not in INTEGER_DTYPES:
            raise TypeError("indices must be an int, slice, integer tensor/list, or boolean mask")
        if index_tensor.ndim == 0:
            index_tensor = index_tensor.view(1)
        elif index_tensor.ndim != 1:
            raise ValueError("integer indices must be a scalar or a 1D tensor/list")
        index_tensor = index_tensor.to(dtype=torch.long)
        index_tensor = torch.where(index_tensor < 0, index_tensor + self.num_gaussians, index_tensor)
        if torch.any((index_tensor < 0) | (index_tensor >= self.num_gaussians)):
            raise IndexError(f"indices are out of bounds for {self.num_gaussians} Gaussians")
        if torch.unique(index_tensor).numel() != index_tensor.numel():
            raise ValueError("indices must not contain duplicates")
        return index_tensor

    @torch.no_grad()
    def _selection_size(self, indices: Tensor | list[int] | tuple[int, ...] | slice | int | None) -> int:
        return int(self._selected_row_indices(indices).numel())

    def _coerce_matrix_parameter(
        self,
        value: Tensor,
        *,
        name: str,
        rows: int,
        cols: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
        if tensor.ndim == 1:
            if rows == 0 and tensor.numel() == 0:
                tensor = tensor.view(0, cols)
            elif rows == 1 and tensor.numel() == cols:
                tensor = tensor.view(1, cols)
            else:
                raise ValueError(f"{name} must have shape [{rows}, {cols}]")
        if tensor.ndim != 2 or tensor.shape != (rows, cols):
            raise ValueError(f"{name} must have shape [{rows}, {cols}]")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        return tensor

    def _coerce_column_parameter(
        self,
        value: Tensor,
        *,
        name: str,
        rows: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
        if tensor.ndim == 0:
            if rows != 1:
                raise ValueError(f"{name} must have shape [{rows}, 1]")
            tensor = tensor.view(1, 1)
        elif tensor.ndim == 1:
            if rows == 0 and tensor.numel() == 0:
                tensor = tensor.view(0, 1)
            elif tensor.shape[0] != rows:
                raise ValueError(f"{name} must have shape [{rows}, 1]")
            else:
                tensor = tensor.unsqueeze(-1)
        if tensor.ndim != 2 or tensor.shape != (rows, 1):
            raise ValueError(f"{name} must have shape [{rows}, 1]")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        return tensor

    @torch.no_grad()
    def _assign_parameter_subset(
        self,
        parameter: Tensor,
        value: Tensor,
        row_indices: Tensor,
    ) -> None:
        if row_indices.numel() == 0:
            return
        selected = parameter.index_select(0, row_indices)
        assigned_value = value
        if selected.shape != assigned_value.shape:
            raise ValueError(
                f"internal shape mismatch while writing parameters: expected {tuple(selected.shape)}, "
                f"received {tuple(assigned_value.shape)}"
            )
        parameter.index_copy_(0, row_indices, assigned_value)

    @torch.no_grad()
    def _assign_exact_boundary_subset(
        self,
        exact_values: Tensor,
        raw_snapshots: Tensor,
        physical_value: Tensor,
        raw_value: Tensor,
        row_indices: Tensor,
    ) -> None:
        if row_indices.numel() == 0:
            return
        boundary_mask = (physical_value == 0.0) | (physical_value == 1.0)
        nan_values = torch.full_like(physical_value, float("nan"))
        exact_assignment = torch.where(boundary_mask, physical_value, nan_values)
        raw_assignment = torch.where(boundary_mask, raw_value, nan_values)
        exact_values.index_copy_(0, row_indices, exact_assignment)
        raw_snapshots.index_copy_(0, row_indices, raw_assignment)

    @torch.no_grad()
    def set_physical_parameters(
        self,
        offsets: Tensor,
        scales: Tensor,
        thetas: Tensor,
        colors: Tensor,
        opacities: Tensor,
        indices: Tensor | list[int] | tuple[int, ...] | slice | int | None = None,
    ) -> None:
        """Write strictly validated physical parameters into the raw learnable state."""
        row_indices = self._selected_row_indices(indices)
        num_selected = int(row_indices.numel())

        offset_tensor = self._coerce_matrix_parameter(
            offsets,
            name="offsets",
            rows=num_selected,
            cols=2,
            dtype=self.raw_offsets.dtype,
            device=self.raw_offsets.device,
        )
        scale_tensor = self._coerce_matrix_parameter(
            scales,
            name="scales",
            rows=num_selected,
            cols=2,
            dtype=self.raw_scales.dtype,
            device=self.raw_scales.device,
        )
        theta_tensor = self._coerce_column_parameter(
            thetas,
            name="thetas",
            rows=num_selected,
            dtype=self.raw_thetas.dtype,
            device=self.raw_thetas.device,
        )
        color_tensor = self._coerce_matrix_parameter(
            colors,
            name="colors",
            rows=num_selected,
            cols=3,
            dtype=self.raw_colors.dtype,
            device=self.raw_colors.device,
        )
        opacity_tensor = self._coerce_column_parameter(
            opacities,
            name="opacities",
            rows=num_selected,
            dtype=self.raw_opacities.dtype,
            device=self.raw_opacities.device,
        )

        if not torch.all(scale_tensor > self.scale_eps):
            raise ValueError(f"scales must be strictly greater than scale_eps ({self.scale_eps})")
        if not torch.all((0.0 <= color_tensor) & (color_tensor <= 1.0)):
            raise ValueError("colors must lie in [0, 1]")
        if not torch.all((0.0 <= opacity_tensor) & (opacity_tensor <= 1.0)):
            raise ValueError("opacities must lie in [0, 1]")

        color_raw_tensor = _safe_logit(color_tensor)
        opacity_raw_tensor = _safe_logit(opacity_tensor)

        self._assign_parameter_subset(self.raw_offsets, offset_tensor, row_indices)
        self._assign_parameter_subset(
            self.raw_scales,
            _inverse_softplus(scale_tensor, eps=self.scale_eps),
            row_indices,
        )
        self._assign_parameter_subset(self.raw_thetas, theta_tensor, row_indices)
        self._assign_parameter_subset(self.raw_colors, color_raw_tensor, row_indices)
        self._assign_parameter_subset(self.raw_opacities, opacity_raw_tensor, row_indices)
        self._assign_exact_boundary_subset(
            self.exact_color_values,
            self.exact_color_raw_snapshots,
            color_tensor,
            color_raw_tensor,
            row_indices,
        )
        self._assign_exact_boundary_subset(
            self.exact_opacity_values,
            self.exact_opacity_raw_snapshots,
            opacity_tensor,
            opacity_raw_tensor,
            row_indices,
        )

    def covariance_entries(self, scales: Tensor, thetas: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Construct the symmetric covariance entries analytically."""
        sx = scales[:, 0]
        sy = scales[:, 1]
        theta = thetas[:, 0]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        sx2 = sx * sx
        sy2 = sy * sy

        a = cos_theta * cos_theta * sx2 + sin_theta * sin_theta * sy2
        b = sin_theta * cos_theta * (sx2 - sy2)
        d = sin_theta * sin_theta * sx2 + cos_theta * cos_theta * sy2
        return a, b, d

    def inverse_covariance_entries(self, a: Tensor, b: Tensor, d: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute the inverse of a symmetric 2x2 covariance analytically."""
        det = a * d - b * b
        if torch.any(det <= self.determinant_eps):
            raise FloatingPointError("Covariance determinant must stay strictly positive.")

        inv_xx = d / det
        inv_xy = -b / det
        inv_yy = a / det
        return inv_xx, inv_xy, inv_yy, det

    def forward(
        self,
        grid_x: Tensor,
        grid_y: Tensor,
        clamp_output: bool = False,
        chunk_size: int | None = None,
    ) -> Tensor:
        """Render a color image from the current Gaussian parameters."""
        if clamp_output and torch.is_grad_enabled():
            raise RuntimeError(
                "clamp_output=True is display-only because it can zero gradients; "
                "use torch.no_grad() for visualization or clamp_output=False for optimization"
            )
        offsets, scales, thetas, colors, opacities = self._decode_parameter_tensors()
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer when provided")

        cos_theta = torch.cos(thetas[:, 0])
        sin_theta = torch.sin(thetas[:, 0])
        inv_sx2 = 1.0 / (scales[:, 0] * scales[:, 0])
        inv_sy2 = 1.0 / (scales[:, 1] * scales[:, 1])

        if chunk_size is None:
            dx = grid_x.unsqueeze(0) - offsets[:, 0].view(-1, 1, 1)
            dy = grid_y.unsqueeze(0) - offsets[:, 1].view(-1, 1, 1)
            local_x = cos_theta.view(-1, 1, 1) * dx + sin_theta.view(-1, 1, 1) * dy
            local_y = -sin_theta.view(-1, 1, 1) * dx + cos_theta.view(-1, 1, 1) * dy
            quadratic = inv_sx2.view(-1, 1, 1) * local_x * local_x + inv_sy2.view(-1, 1, 1) * local_y * local_y
            gaussian_response = torch.exp(-0.5 * quadratic)
            weights = gaussian_response * opacities[:, 0].view(-1, 1, 1)
            rendered = torch.sum(weights.unsqueeze(-1) * colors.view(-1, 1, 1, 3), dim=0)
        else:
            rendered = torch.zeros((grid_x.shape[0], grid_x.shape[1], 3), dtype=grid_x.dtype, device=grid_x.device)
            for start in range(0, self.num_gaussians, chunk_size):
                end = min(start + chunk_size, self.num_gaussians)
                dx = grid_x.unsqueeze(0) - offsets[start:end, 0].view(-1, 1, 1)
                dy = grid_y.unsqueeze(0) - offsets[start:end, 1].view(-1, 1, 1)
                local_x = cos_theta[start:end].view(-1, 1, 1) * dx + sin_theta[start:end].view(-1, 1, 1) * dy
                local_y = -sin_theta[start:end].view(-1, 1, 1) * dx + cos_theta[start:end].view(-1, 1, 1) * dy
                quadratic = (
                    inv_sx2[start:end].view(-1, 1, 1) * local_x * local_x
                    + inv_sy2[start:end].view(-1, 1, 1) * local_y * local_y
                )
                gaussian_response = torch.exp(-0.5 * quadratic)
                weights = gaussian_response * opacities[start:end, 0].view(-1, 1, 1)
                rendered = rendered + torch.sum(
                    weights.unsqueeze(-1) * colors[start:end].view(-1, 1, 1, 3),
                    dim=0,
                )

        if clamp_output:
            rendered = rendered.clamp(0.0, 1.0)
        return rendered
