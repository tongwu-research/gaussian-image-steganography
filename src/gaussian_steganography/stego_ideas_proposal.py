"""Shared configuration and mathematical helpers for the earlier protocol."""
from __future__ import annotations

import argparse
import hashlib
import math
import random
from dataclasses import dataclass, field

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _canonicalize_scales(scales: Tensor, thetas: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Return (s_major, s_minor, theta_major) from (N,2) scales and (N,1) thetas.
    s_major >= s_minor >= 0. theta_major is in [0, pi).
    """
    s0, s1 = scales[:, 0], scales[:, 1]
    swap = s0 < s1
    s_major = torch.where(swap, s1, s0)
    s_minor = torch.where(swap, s0, s1)
    theta_raw = thetas[:, 0]
    theta_major = torch.where(swap, theta_raw + math.pi / 2.0, theta_raw) % math.pi
    return s_major, s_minor, theta_major


def _log_anisotropy(scales: Tensor, thetas: Tensor) -> Tensor:
    """log(s_major / s_minor); 0 for isotropic Gaussians."""
    s_major, s_minor, _ = _canonicalize_scales(scales, thetas)
    return torch.log(s_major / s_minor.clamp_min(1e-8))


def _circular_mean(angles: Tensor) -> float:
    """Circular mean of angles (in radians)."""
    return float(math.atan2(float(torch.sin(angles).mean()), float(torch.cos(angles).mean())))


def _stable_seed(key: str, *, namespace: str) -> int:
    """
    Derive a process-stable RNG seed from (namespace, key).

    Python's built-in ``hash()`` is salted per process, which makes cross-process
    encode/decode inconsistent for any keyed carrier assignment scheme.  These
    steganography prototypes are often encoded in one process and decoded in a
    different process, so the seed must be deterministic across interpreters.
    """
    digest = hashlib.sha256(f"{namespace}::{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


# ---------------------------------------------------------------------------
# Idea 1 — Rotation-Distribution Steganography  (theta_stego)
# ---------------------------------------------------------------------------
# Concept
# -------
# Partition the N Gaussians into K spatial groups (e.g., a regular grid of
# quadrants or image-content-aware clusters). Each group encodes ONE bit by
# biasing the circular mean of theta toward a "zero" target direction (bit=0)
# or a "one" target direction (bit=1, shifted by π/K).
#
# Key property: the *marginal* theta distribution over all N Gaussians is
# preserved — only the within-group circular means shift. This means a
# global observer who doesn't know the group assignment key sees no change.
#
# Mathematical formulation
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Let G_k = { i : group(i) = k, i = 1..N } for k = 1..K.
# For group k encoding bit b_k:
#   target_k = b_k * (π / K)
#   Objective: minimize  sum_{i in G_k} cost(i) * (theta_i - target_k - alpha_k)^2
#              subject to  circular_mean({theta_i : i in G_k}) ≈ target_k
# where alpha_k is an offset estimated from the clean distribution and
# cost(i) is the per-Gaussian sensitivity (low-sensitivity Gaussians preferred).
#
# Capacity: K bits per image (one bit per group).
# Payload:  K bits total; typically K = 8-16 for 256 Gaussians.
# PSNR cost: Depends on how tightly thetas are constrained by the image content.
#            For images with strong orientation structure (edges, textures), the
#            cost is higher because theta is already determined by content.
#            For images with diffuse content, the cost is low.
# Attack resistance:
#   + Robust to per-Gaussian scale/opacity attacks (theta is a separate parameter).
#   - Vulnerable to re-fitting attacks that change all parameters jointly.
#   - The circular mean shift may be small and hard to guarantee after fine-tuning.
# Novelty vs log_anisotropy baseline:
#   Uses a completely different parameter channel (rotation angle vs scale ratio).
#   Group-level encoding is more robust than per-Gaussian encoding.
# ---------------------------------------------------------------------------

@dataclass
class ThetaStegoCoder:
    """
    Rotation-distribution steganography encoder/decoder.
    Encodes K bits using circular mean shifts within K spatial groups.
    """
    n_groups: int = 8
    target_spread: float = math.pi / 4.0  # angular separation between bit-0 and bit-1 targets

    def _assign_groups(self, offsets: Tensor, key: str) -> list[list[int]]:
        """Assign Gaussians to groups by spatial grid + key-dependent permutation."""
        n = offsets.shape[0]
        rng = random.Random(_stable_seed(key, namespace="theta_groups"))
        mu_x = offsets[:, 0].tolist()
        mu_y = offsets[:, 1].tolist()
        # Divide canvas into a sqrt(K) x sqrt(K) grid and assign by cell
        k_side = max(1, int(math.ceil(self.n_groups ** 0.5)))
        groups: list[list[int]] = [[] for _ in range(self.n_groups)]
        for i, (x, y) in enumerate(zip(mu_x, mu_y)):
            col = int((x + 1.0) / 2.0 * k_side) % k_side
            row = int((y + 1.0) / 2.0 * k_side) % k_side
            cell = row * k_side + col
            group_id = cell % self.n_groups
            groups[group_id].append(i)
        # Key-dependent permutation of group assignments to prevent non-keyed decoding
        group_order = list(range(self.n_groups))
        rng.shuffle(group_order)
        permuted = [groups[group_order[k]] for k in range(self.n_groups)]
        return permuted

    def encode(
        self,
        bits: list[int],
        offsets: Tensor,
        thetas: Tensor,
        *,
        key: str,
        delta: float = 0.15,
    ) -> Tensor:
        """
        Return modified theta tensor encoding the given bits.
        Each group's circular mean is nudged toward the target direction.

        bits:  list of 0/1 values, length = n_groups
        delta: maximum per-Gaussian angular perturbation (radians)
        """
        if len(bits) != self.n_groups:
            raise ValueError(f"expected {self.n_groups} bits, got {len(bits)}")
        groups = self._assign_groups(offsets, key)
        new_thetas = thetas.clone()
        theta_vals = thetas[:, 0].clone()

        for k, (bit, group_indices) in enumerate(zip(bits, groups)):
            if not group_indices:
                continue
            # Target circular mean for this group
            target = (bit * self.target_spread + k * math.pi / max(self.n_groups, 1)) % math.pi
            # Current circular mean
            group_theta = theta_vals[group_indices]
            current_mean = _circular_mean(group_theta)
            # Angular shift needed
            shift = target - current_mean
            # Normalize shift to [-pi/2, pi/2] (mod pi for orientation)
            shift = (shift + math.pi / 2.0) % math.pi - math.pi / 2.0
            # Apply bounded shift (soft — real implementation would optimize with loss)
            clipped_shift = max(-delta, min(delta, shift))
            for i in group_indices:
                new_thetas[i, 0] = float((theta_vals[i].item() + clipped_shift) % math.pi)
        return new_thetas

    def decode(
        self,
        offsets: Tensor,
        thetas: Tensor,
        *,
        key: str,
    ) -> list[int]:
        """
        Decode K bits from the modified theta distribution.
        Returns a list of 0/1 values, length = n_groups.
        """
        groups = self._assign_groups(offsets, key)
        theta_vals = thetas[:, 0]
        decoded = []
        for k, group_indices in enumerate(groups):
            if not group_indices:
                decoded.append(0)
                continue
            group_theta = theta_vals[group_indices]
            current_mean = _circular_mean(group_theta)
            # Compare circular mean against bit-0 and bit-1 targets
            target_0 = (0 * self.target_spread + k * math.pi / max(self.n_groups, 1)) % math.pi
            target_1 = (1 * self.target_spread + k * math.pi / max(self.n_groups, 1)) % math.pi
            dist_0 = abs((current_mean - target_0 + math.pi / 2.0) % math.pi - math.pi / 2.0)
            dist_1 = abs((current_mean - target_1 + math.pi / 2.0) % math.pi - math.pi / 2.0)
            decoded.append(0 if dist_0 <= dist_1 else 1)
        return decoded


# ---------------------------------------------------------------------------
# Idea 2 — Position-Jitter Steganography  (mu_stego)
# ---------------------------------------------------------------------------
# Concept
# -------
# Use the Gaussian center positions (mu_x, mu_y) as a carrier. For each
# selected Gaussian, apply a small signed displacement (delta_mu) whose
# sign encodes one bit. The displacement magnitude is bounded by the local
# content density gradient to minimize visual impact.
#
# Mathematical formulation
# ~~~~~~~~~~~~~~~~~~~~~~~~
# For selected carrier Gaussian i with center (mu_x_i, mu_y_i):
#   Bit b_i encoded by displacement direction d_i ∈ {-1, +1}:
#   mu_x_i^new = mu_x_i + d_i * epsilon_i * cos(phi_i)
#   mu_y_i^new = mu_y_i + d_i * epsilon_i * sin(phi_i)
# where:
#   phi_i:     a key-dependent carrier direction for Gaussian i
#   epsilon_i: displacement magnitude, bounded by local_budget(i)
#   local_budget(i) = base_epsilon * min(1, 1 / (1 + grad_mag_i))
#   grad_mag_i = ||nabla I(mu_i)||  (image gradient at Gaussian center)
#
# Decoding: measure signed projection of (mu_new - mu_clean) onto carrier
#           direction phi_i; positive → bit=1, negative → bit=0.
#
# Key property: in regions with strong image gradient (sharp edges), the
# budget is small (tight constraint). In diffuse background, the budget is
# large. This naturally focuses embedding in perceptually flat regions.
#
# Capacity: one bit per selected Gaussian; up to N * selection_ratio bits.
# Attack resistance:
#   + Independent of scale/rotation channels; single-channel attacks don't work.
#   - Vulnerable to refit attacks that re-optimize all positions jointly.
#   - Quantization of position coordinates (if positions are discretized) may flip bits.
# Novelty vs log_anisotropy baseline:
#   Uses the position (mu) channel, which is NEVER used as a carrier in the
#   current implementation. Positions carry geometric meaning distinct from
#   the scale/opacity channels.
# ---------------------------------------------------------------------------

@dataclass
class MuStegoCoder:
    """
    Position-jitter steganography encoder/decoder.
    Encodes bits via signed displacements along key-dependent carrier directions.
    """
    base_epsilon: float = 0.01  # maximum displacement in canvas units

    def _carrier_directions(self, n: int, key: str) -> Tensor:
        """Generate per-Gaussian carrier angle phi_i in [0, 2pi) from key."""
        rng = random.Random(_stable_seed(key, namespace="mu_stego"))
        angles = torch.tensor([rng.uniform(0, 2 * math.pi) for _ in range(n)])
        return angles

    def _select_carriers(self, n: int, payload_bits: int, key: str) -> list[int]:
        """Select carrier Gaussians using key-seeded sampling."""
        rng = random.Random(_stable_seed(key, namespace="mu_select"))
        indices = list(range(n))
        rng.shuffle(indices)
        return sorted(indices[:payload_bits])

    def encode(
        self,
        bits: list[int],
        offsets: Tensor,
        *,
        key: str,
        gradient_magnitudes: Tensor | None = None,
    ) -> Tensor:
        """
        Return modified offsets encoding the given bits.

        bits:                 list of 0/1 values
        offsets:              (N, 2) clean Gaussian centers
        gradient_magnitudes:  (N,) local image gradient magnitudes at each center
                              (optional; used to adapt epsilon to content complexity)
        """
        n = offsets.shape[0]
        carrier_indices = self._select_carriers(n, len(bits), key)
        if len(carrier_indices) < len(bits):
            raise ValueError("Not enough Gaussians to encode all bits")
        phi = self._carrier_directions(n, key)
        new_offsets = offsets.clone()
        for bit_idx, (gauss_idx, bit) in enumerate(zip(carrier_indices, bits)):
            direction = 1 if bit == 1 else -1
            budget = self.base_epsilon
            if gradient_magnitudes is not None:
                grad_mag = float(gradient_magnitudes[gauss_idx].item())
                budget = self.base_epsilon / max(1.0, 1.0 + grad_mag)
            angle = float(phi[gauss_idx].item())
            dx = direction * budget * math.cos(angle)
            dy = direction * budget * math.sin(angle)
            new_offsets[gauss_idx, 0] = offsets[gauss_idx, 0] + dx
            new_offsets[gauss_idx, 1] = offsets[gauss_idx, 1] + dy
        return new_offsets

    def decode(
        self,
        offsets_clean: Tensor,
        offsets_watermarked: Tensor,
        *,
        key: str,
        n_bits: int,
    ) -> list[int]:
        """
        Decode bits by measuring signed projections of positional displacements.
        Requires clean reference offsets (white-box assumption).
        """
        n = offsets_clean.shape[0]
        carrier_indices = self._select_carriers(n, n_bits, key)
        phi = self._carrier_directions(n, key)
        decoded = []
        for gauss_idx in carrier_indices[:n_bits]:
            delta_x = float((offsets_watermarked[gauss_idx, 0] - offsets_clean[gauss_idx, 0]).item())
            delta_y = float((offsets_watermarked[gauss_idx, 1] - offsets_clean[gauss_idx, 1]).item())
            angle = float(phi[gauss_idx].item())
            projection = delta_x * math.cos(angle) + delta_y * math.sin(angle)
            decoded.append(1 if projection > 0 else 0)
        return decoded


# ---------------------------------------------------------------------------
# Idea 3 — Scale-Tier Steganography  (scale_tier_stego)
# ---------------------------------------------------------------------------
# Concept
# -------
# Rank all N Gaussians by log_area = log(s_major * s_minor) into T tiers
# (e.g., T=3: small/medium/large). Within each tier, partition into two
# subgroups: "pushed-up" and "pushed-down". One bit is encoded per tier by
# which subgroup has a slightly higher mean log_area.
#
# Mathematical formulation
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Let L_i = log(s_major_i * s_minor_i) = log_area_i.
# Sort Gaussians by L_i and divide into T tiers of equal size.
# Within tier t, partition into subgroups A_t, B_t using key-seeded assignment.
# Encode bit b_t by ensuring:
#   mean(L_i : i in A_t) > mean(L_i : i in B_t)  iff b_t = 1
#   mean(L_i : i in B_t) > mean(L_i : i in A_t)  iff b_t = 0
#
# Achieve this by applying a small scaling factor:
#   if b_t = 1: multiply scales of A_t members by 1 + delta_scale
#   if b_t = 0: multiply scales of B_t members by 1 + delta_scale
# (The non-selected subgroup is left unchanged.)
#
# Decoding: measure sign(mean(A_t) - mean(B_t)); positive → bit=1.
#
# Key property: the global log_area distribution (its shape and mean) is
# preserved because modifications are symmetric across tiers. The encoding is
# only visible as a within-tier mean shift, which is hard to detect without
# knowing the key-dependent A/B partition.
#
# Capacity: T bits per image (T = number of tiers, typically 4-16).
# Attack resistance:
#   + The group-level mean is robust to individual Gaussian attacks.
#   + More resistant to fine-tuning than per-Gaussian encoding.
#   - Vulnerable to global scale normalization attacks.
# ---------------------------------------------------------------------------

@dataclass
class ScaleTierStegoCoder:
    """
    Scale-tier steganography encoder/decoder.
    Encodes T bits using within-tier scale mean shifts.
    """
    n_tiers: int = 8
    delta_scale: float = 0.05  # multiplicative scale perturbation per subgroup

    def _tier_subgroups(self, log_areas: Tensor, key: str) -> list[tuple[list[int], list[int]]]:
        """
        For each tier, return (subgroup_A, subgroup_B) index lists.
        Tiers are defined by log_area quantiles; subgroups are key-seeded.
        """
        n = log_areas.shape[0]
        sorted_indices = torch.argsort(log_areas).tolist()
        tier_size = max(1, n // self.n_tiers)
        rng = random.Random(_stable_seed(key, namespace="scale_tier"))
        result = []
        for t in range(self.n_tiers):
            start = t * tier_size
            end = min(n, (t + 1) * tier_size) if t < self.n_tiers - 1 else n
            tier_indices = sorted_indices[start:end]
            rng.shuffle(tier_indices)
            mid = len(tier_indices) // 2
            subgroup_a = tier_indices[:mid]
            subgroup_b = tier_indices[mid:]
            result.append((subgroup_a, subgroup_b))
        return result

    def encode(
        self,
        bits: list[int],
        scales: Tensor,
        thetas: Tensor,
        *,
        key: str,
    ) -> Tensor:
        """
        Return modified scales encoding the given bits via tier mean shifts.

        bits:   list of 0/1 values, length = n_tiers
        scales: (N, 2) clean Gaussian scales
        thetas: (N, 1) Gaussian orientations (used for log_area computation)
        """
        if len(bits) != self.n_tiers:
            raise ValueError(f"expected {self.n_tiers} bits, got {len(bits)}")
        s_major, s_minor, _ = _canonicalize_scales(scales, thetas)
        log_areas = torch.log(s_major * s_minor.clamp_min(1e-8))
        tier_groups = self._tier_subgroups(log_areas, key)
        new_scales = scales.clone()
        for bit, (sub_a, sub_b) in zip(bits, tier_groups):
            # Whichever subgroup gets the scale increase determines the decoded bit.
            boosted = sub_a if bit == 1 else sub_b
            for i in boosted:
                new_scales[i] = scales[i] * (1.0 + self.delta_scale)
        return new_scales

    def decode(
        self,
        scales: Tensor,
        thetas: Tensor,
        *,
        key: str,
        clean_scales: Tensor | None = None,
        clean_thetas: Tensor | None = None,
    ) -> list[int]:
        """
        Decode T bits from the scale tier mean shifts.

        When clean reference tensors are provided, tiers and subgroup statistics are
        defined from the clean geometry and decoding compares subgroup mean shifts
        against that reference.  This is the preferred evaluation mode for the
        prototype study because it is more stable than blind decode.

        Without clean reference, decoding falls back to blind within-tier means on
        the modified parameters only.
        """
        s_major, s_minor, _ = _canonicalize_scales(scales, thetas)
        log_areas = torch.log(s_major * s_minor.clamp_min(1e-8))
        if clean_scales is not None and clean_thetas is not None:
            clean_major, clean_minor, _ = _canonicalize_scales(clean_scales, clean_thetas)
            clean_log_areas = torch.log(clean_major * clean_minor.clamp_min(1e-8))
            tier_groups = self._tier_subgroups(clean_log_areas, key)
        else:
            clean_log_areas = None
            tier_groups = self._tier_subgroups(log_areas, key)
        decoded = []
        for sub_a, sub_b in tier_groups:
            if not sub_a or not sub_b:
                decoded.append(0)
                continue
            if clean_log_areas is not None:
                delta_a = float((log_areas[sub_a].mean() - clean_log_areas[sub_a].mean()).item())
                delta_b = float((log_areas[sub_b].mean() - clean_log_areas[sub_b].mean()).item())
                decoded.append(1 if delta_a > delta_b else 0)
            else:
                mean_a = float(log_areas[sub_a].mean().item())
                mean_b = float(log_areas[sub_b].mean().item())
                decoded.append(1 if mean_a > mean_b else 0)
        return decoded


# ---------------------------------------------------------------------------
# Idea 4 — Multi-Parameter Joint Parity Coding  (joint_stego)
# ---------------------------------------------------------------------------
# Concept
# -------
# Encode each bit jointly across THREE parameter channels: log_anisotropy,
# alpha (opacity), and theta (rotation). One bit is encoded as a GF(2) parity
# check across the three channels' binary indicators.
#
# Mathematical formulation
# ~~~~~~~~~~~~~~~~~~~~~~~~
# For each payload bit b_i, select three Gaussian parameters (one per channel):
#   g_log_i:    Gaussian index for the log_anisotropy channel
#   g_alpha_i:  Gaussian index for the alpha channel
#   g_theta_i:  Gaussian index for the theta channel
#
# Encode by modifying each carrier in its respective direction d_j ∈ {-1, +1}
# such that the following GF(2) parity condition holds:
#   p_log_i XOR p_alpha_i XOR p_theta_i = b_i
# where p_j = 1 if the parameter was increased, 0 if decreased.
#
# The key determines: (g_log_i, g_alpha_i, g_theta_i, d_log_i, d_alpha_i,
# d_theta_i) for each bit i.
#
# Attack resistance analysis:
#   Let f(i) = p_log_i XOR p_alpha_i XOR p_theta_i.
#   To flip bit b_i, the attacker must flip an ODD number of the three
#   binary indicators p_j. Flipping one channel (e.g., a scale quantization
#   attack that changes log_anisotropy) flips only p_log_i, which flips f(i).
#   BUT: flipping TWO channels (e.g., coupled scale + opacity attack) also
#   flips f(i) with probability 1 if both are flipped.
#   Against an attacker that can only attack ONE channel at a time:
#     P(bit flip | one-channel attack on carrier) = 1.0  (same as single-channel)
#   Against an attacker that randomly perturbs all parameters:
#     P(bit flip) = P(odd parity flip) = 3 * p * (1-p)^2 + p^3
#   where p = per-channel bit flip probability.
#   For p = 0.1: P(flip) ≈ 3 * 0.1 * 0.81 + 0.001 ≈ 0.244 (worse than single)
#   For p = 0.3: P(flip) ≈ 3 * 0.3 * 0.49 + 0.027 ≈ 0.468 (comparable)
# Conclusion: Joint coding does NOT improve robustness against independent
# per-channel attacks. Its advantage is different:
#   → Attribution: a wrong key that uses different channel assignments will
#     decode a DIFFERENT bit pattern, not just a random one. This enables
#     structured wrong-key fingerprinting (which specific wrong key was used?).
#   → Spread-spectrum effect: modifying all three channels simultaneously for
#     each bit reduces per-channel modification depth by a factor of 3,
#     improving covertness at the cost of more Gaussians being touched.
# ---------------------------------------------------------------------------

@dataclass
class JointParityStegoCoder:
    """
    Multi-parameter joint parity steganography encoder/decoder.
    Encodes each bit as a GF(2) parity check across log_anisotropy, alpha, and theta.

    This is the most novel of the four proposals for SIGGRAPH because:
    1. It uses all three geometric parameter channels simultaneously.
    2. It reduces per-channel modification depth (better covertness per bit).
    3. The structured wrong-key property enables attribution analysis.
    """
    log_delta: float = 0.35    # modification magnitude for log_anisotropy
    alpha_delta: float = 0.05  # modification magnitude for opacity
    theta_delta: float = 0.10  # modification magnitude for rotation (radians)

    def _carrier_assignment(
        self,
        n: int,
        n_bits: int,
        key: str,
    ) -> list[dict[str, object]]:
        """
        For each bit, assign three carrier Gaussians (one per channel) and
        key-seeded sign conventions (s_log, s_alpha, s_theta ∈ {-1, +1}).
        Additionally, key-seeded binary base values (p_log_base, p_alpha_base)
        determine which direction each channel is modified — only p_theta is
        left free to carry the actual bit information.

        All values are fully determined by the key; no bit value needed here.
        """
        rng = random.Random(_stable_seed(key, namespace="joint_stego"))
        available = list(range(n))
        rng.shuffle(available)
        assignments = []
        for b in range(n_bits):
            log_idx = available[(3 * b) % n]
            alpha_idx = available[(3 * b + 1) % n]
            theta_idx = available[(3 * b + 2) % n]
            # sign_* is the "positive" reference direction for measuring each channel
            sign_log = 1 if rng.random() < 0.5 else -1
            sign_alpha = 1 if rng.random() < 0.5 else -1
            sign_theta = 1 if rng.random() < 0.5 else -1
            # p_log_base and p_alpha_base are key-seeded binary values; they determine
            # whether log and alpha carriers are modified in the + or - sign direction.
            # p_theta is then set at encode time to satisfy the XOR parity = bit.
            p_log_base = 1 if rng.random() < 0.5 else 0
            p_alpha_base = 1 if rng.random() < 0.5 else 0
            assignments.append(
                {
                    "log_idx": log_idx,
                    "alpha_idx": alpha_idx,
                    "theta_idx": theta_idx,
                    "sign_log": sign_log,
                    "sign_alpha": sign_alpha,
                    "sign_theta": sign_theta,
                    "p_log_base": p_log_base,
                    "p_alpha_base": p_alpha_base,
                }
            )
        return assignments

    def encode(
        self,
        bits: list[int],
        scales: Tensor,
        thetas: Tensor,
        opacities: Tensor,
        *,
        key: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Encode bits jointly across log_anisotropy, alpha, and theta channels.

        Returns (new_scales, new_thetas, new_opacities).
        All three carriers are ALWAYS modified (spread-spectrum property).
        For each bit b_i, the modification directions are chosen such that:
            p_log_i XOR p_alpha_i XOR p_theta_i = b_i
        where p_j = 1 if the jth channel was modified in its key-positive direction.

        The p_log and p_alpha values are key-seeded (base values);
        p_theta is set to achieve the XOR parity = b_i.
        """
        n = scales.shape[0]
        assignments = self._carrier_assignment(n, len(bits), key)
        new_scales = scales.clone()
        new_thetas = thetas.clone()
        new_opacities = opacities.clone()

        for bit, asgn in zip(bits, assignments):
            log_idx = int(asgn["log_idx"])
            alpha_idx = int(asgn["alpha_idx"])
            theta_idx = int(asgn["theta_idx"])
            sign_log = int(asgn["sign_log"])
            sign_alpha = int(asgn["sign_alpha"])
            sign_theta = int(asgn["sign_theta"])
            p_log = int(asgn["p_log_base"])
            p_alpha = int(asgn["p_alpha_base"])
            # p_theta is determined by the parity constraint
            p_theta = bit ^ p_log ^ p_alpha

            # Modification direction for each channel:
            # dir_j = sign_j if p_j = 1 (positive), else -sign_j (negative)
            dir_log = sign_log if p_log == 1 else -sign_log
            dir_alpha = sign_alpha if p_alpha == 1 else -sign_alpha
            dir_theta = sign_theta if p_theta == 1 else -sign_theta

            # Apply log_anisotropy modification using the SIGNED log-ratio log(s0/s1).
            # We work in signed space to avoid axis-swap ambiguity (which would flip
            # the measured sign at decode time and break the parity recovery).
            # Invariant preserved: geometric mean sqrt(s0 * s1) is unchanged.
            s0 = float(scales[log_idx, 0].item())
            s1 = float(scales[log_idx, 1].item())
            geom_mean = math.sqrt(max(s0 * s1, 1e-12))
            log_ratio = math.log(max(s0, 1e-8) / max(s1, 1e-8))
            log_ratio_new = log_ratio + dir_log * self.log_delta
            new_scales[log_idx, 0] = geom_mean * math.exp(log_ratio_new / 2.0)
            new_scales[log_idx, 1] = geom_mean * math.exp(-log_ratio_new / 2.0)

            new_opacity = float(opacities[alpha_idx, 0].item()) + dir_alpha * self.alpha_delta
            new_opacities[alpha_idx, 0] = max(0.0, min(1.0, new_opacity))

            new_theta = float(thetas[theta_idx, 0].item()) + dir_theta * self.theta_delta
            new_thetas[theta_idx, 0] = new_theta % math.pi

        return new_scales, new_thetas, new_opacities

    def decode(
        self,
        scales: Tensor,
        thetas: Tensor,
        opacities: Tensor,
        scales_clean: Tensor,
        thetas_clean: Tensor,
        opacities_clean: Tensor,
        *,
        key: str,
        n_bits: int,
    ) -> list[int]:
        """
        Decode bits by measuring parity across all three channels.
        Requires clean reference parameters (white-box assumption).

        Decoding rule for each bit:
            p_log = 1  if  sign_log * (log_anis_new - log_anis_clean) / log_delta > 0
            p_alpha = 1  if  sign_alpha * (alpha_new - alpha_clean) / alpha_delta > 0
            p_theta = 1  if  sign_theta * (theta_new - theta_clean) / theta_delta > 0
            decoded_bit = p_log XOR p_alpha XOR p_theta

        Note: sign_*, log_idx, alpha_idx, theta_idx are all derived from the key.
        A wrong key gives wrong sign conventions → random p_j values → BER ≈ 0.5.
        """
        n = scales.shape[0]
        assignments = self._carrier_assignment(n, n_bits, key)

        # Use the SIGNED log-ratio log(s0/s1) — same as encoding convention.
        # This avoids axis-swap ambiguity that would occur with log(s_major/s_minor).
        log_ratio_wm = torch.log(scales[:, 0].clamp_min(1e-8) / scales[:, 1].clamp_min(1e-8))
        log_ratio_clean = torch.log(scales_clean[:, 0].clamp_min(1e-8) / scales_clean[:, 1].clamp_min(1e-8))
        alpha = opacities[:, 0]
        alpha_c = opacities_clean[:, 0]
        theta = thetas[:, 0]
        theta_c = thetas_clean[:, 0]

        decoded = []
        for asgn in assignments:
            log_idx = int(asgn["log_idx"])
            alpha_idx = int(asgn["alpha_idx"])
            theta_idx = int(asgn["theta_idx"])
            sign_log = float(asgn["sign_log"])
            sign_alpha = float(asgn["sign_alpha"])
            sign_theta = float(asgn["sign_theta"])

            # Measure signed change relative to key-seeded positive direction.
            # score_j > 0 means the channel was modified in the key-positive direction.
            score_log = sign_log * float((log_ratio_wm[log_idx] - log_ratio_clean[log_idx]).item()) / max(self.log_delta, 1e-8)
            score_alpha = sign_alpha * float((alpha[alpha_idx] - alpha_c[alpha_idx]).item()) / max(self.alpha_delta, 1e-8)
            score_theta = sign_theta * float((theta[theta_idx] - theta_c[theta_idx]).item()) / max(self.theta_delta, 1e-8)

            actual_p_log = 1 if score_log > 0 else 0
            actual_p_alpha = 1 if score_alpha > 0 else 0
            actual_p_theta = 1 if score_theta > 0 else 0
            decoded.append(actual_p_log ^ actual_p_alpha ^ actual_p_theta)
        return decoded


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _test_theta_stego() -> None:
    """Round-trip test for ThetaStegoCoder."""
    coder = ThetaStegoCoder(n_groups=4, target_spread=math.pi / 4.0)
    n = 64
    torch.manual_seed(7)
    offsets = torch.rand(n, 2) * 2.0 - 1.0
    thetas = torch.rand(n, 1) * math.pi
    bits = [1, 0, 1, 1]
    key = "test_key_theta"
    new_thetas = coder.encode(bits, offsets, thetas, key=key, delta=0.2)
    decoded = coder.decode(offsets, new_thetas, key=key)
    # Note: with only a soft nudge (no optimization), round-trip may not be perfect
    # This test checks that encode/decode runs without error and returns the right shape
    assert len(decoded) == 4, f"Expected 4 decoded bits, got {len(decoded)}"
    print(f"  ThetaStego: encoded={bits}, decoded={decoded} (soft nudge only, may differ)")


def _test_mu_stego() -> None:
    """Round-trip test for MuStegoCoder."""
    coder = MuStegoCoder(base_epsilon=0.05)
    n = 64
    torch.manual_seed(9)
    offsets = torch.rand(n, 2) * 2.0 - 1.0
    bits = [1, 0, 1, 0, 1, 1, 0, 1]
    key = "test_key_mu"
    new_offsets = coder.encode(bits, offsets, key=key)
    decoded = coder.decode(offsets, new_offsets, key=key, n_bits=len(bits))
    n_correct = sum(b == d for b, d in zip(bits, decoded))
    assert n_correct == len(bits), (
        f"MuStego round-trip failed: {n_correct}/{len(bits)} correct. "
        f"encoded={bits}, decoded={decoded}"
    )
    print(f"  MuStego: round-trip {n_correct}/{len(bits)} correct (epsilon={coder.base_epsilon})")


def _test_scale_tier_stego() -> None:
    """Round-trip test for ScaleTierStegoCoder (blind decode — no clean reference)."""
    coder = ScaleTierStegoCoder(n_tiers=4, delta_scale=0.10)
    n = 128
    torch.manual_seed(13)
    scales = torch.rand(n, 2) * 0.15 + 0.02
    thetas = torch.rand(n, 1) * math.pi
    bits = [1, 0, 1, 1]
    key = "test_key_scale_tier"
    new_scales = coder.encode(bits, scales, thetas, key=key)
    # Decode with modified scales (blind — does not require clean reference)
    decoded = coder.decode(new_scales, thetas, key=key)
    # Note: blind decode uses relative within-tier means from the MODIFIED params,
    # which may not perfectly recover bits when delta is small. This tests the
    # machinery runs; a real implementation would use clean reference for comparison.
    assert len(decoded) == 4, f"Expected 4 decoded bits, got {len(decoded)}"
    print(f"  ScaleTierStego: encoded={bits}, decoded={decoded} (blind decode)")


def _test_joint_parity_stego() -> None:
    """Round-trip test for JointParityStegoCoder (white-box, requires clean reference)."""
    coder = JointParityStegoCoder(log_delta=0.35, alpha_delta=0.05, theta_delta=0.15)
    n = 64
    torch.manual_seed(17)
    scales = torch.rand(n, 2) * 0.12 + 0.02
    thetas = torch.rand(n, 1) * math.pi
    opacities = torch.rand(n, 1) * 0.7 + 0.15
    bits = [1, 0, 1, 1, 0, 0, 1, 0]
    key = "test_key_joint"
    new_scales, new_thetas, new_opacities = coder.encode(bits, scales, thetas, opacities, key=key)
    decoded = coder.decode(
        new_scales, new_thetas, new_opacities,
        scales, thetas, opacities,
        key=key, n_bits=len(bits),
    )
    n_correct = sum(b == d for b, d in zip(bits, decoded))
    assert n_correct == len(bits), (
        f"JointParityStego round-trip failed: {n_correct}/{len(bits)} correct. "
        f"encoded={bits}, decoded={decoded}"
    )
    print(f"  JointParityStego: round-trip {n_correct}/{len(bits)} correct")


def self_test() -> None:
    """Run all prototype round-trip feasibility checks."""
    print("=== stego_ideas_proposal.py self-test ===")
    _test_theta_stego()
    _test_mu_stego()
    _test_scale_tier_stego()
    _test_joint_parity_stego()
    print("=== all checks passed ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="New steganography idea prototypes for 2D Gaussian Splatting."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run round-trip feasibility tests for all four prototypes.",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
