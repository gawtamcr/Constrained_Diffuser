"""
STL (Signal Temporal Logic) compiler for the Constrained Diffuser.

Builds a small DSL from Python classes representing STL formulas and compiles
them into differentiable PyTorch barrier functions.

Barrier convention (matches the existing constraint interface in diffusion.py):
    b(τ) ≥ 0  ⟺  trajectory τ satisfies the formula
    b(τ) < 0  ⟺  violation; magnitude is |b(τ)|

Callable signature produced by compile():
    barrier(trajectory: Tensor, s: float) -> Tensor
        trajectory : [batch, horizon, transition_dim]  (normalized)
        s          : denoising progress in [0, 1]  (1 - t / T)
                     Only used by Eventually nodes (funnel slack).
        returns    : [batch, horizon]  per-timestep barrier values
                     (the constraint algorithms use .sum() for the dual update,
                      and autograd over the whole tensor for gradients — the
                      same shape the existing g_x_funcs return)

Design note on shape:
    The existing g_x_funcs(state) return [batch, horizon] (or [batch, horizon, 1]
    which gets squeezed). We match that exactly so calc_grad / dual_update work
    unchanged.  For temporal operators (Always / Eventually) the returned tensor
    already has the min/max folded in over the time window; we broadcast it back
    to [batch, horizon] by repeating it so that gradient signals reach every
    relevant timestep.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Smooth min / max
# ---------------------------------------------------------------------------

def smooth_min(x: Tensor, kappa: float, dim: int) -> Tensor:
    """Log-sum-exp approximation of min along `dim`.

    smooth_min(x, κ) = -(1/κ) * log( Σ exp(-κ * xᵢ) )
    """
    return -(1.0 / kappa) * torch.logsumexp(-kappa * x, dim=dim)


def smooth_max(x: Tensor, kappa: float, dim: int) -> Tensor:
    """Log-sum-exp approximation of max along `dim`.

    smooth_max(x, κ) = (1/κ) * log( Σ exp(κ * xᵢ) )
    """
    return (1.0 / kappa) * torch.logsumexp(kappa * x, dim=dim)


# ---------------------------------------------------------------------------
# STL formula nodes
# ---------------------------------------------------------------------------

class STLFormula:
    """Abstract base for all STL formula nodes."""

    def barrier(self, traj: Tensor, s: float, kappa: float) -> Tensor:
        """Evaluate the barrier value.

        Args:
            traj  : [batch, horizon, transition_dim] normalized trajectory
            s     : denoising progress in [0, 1]
            kappa : smoothing temperature

        Returns:
            Tensor of shape [batch, horizon] — per-timestep barrier values.
            For point-wise predicates this is simply h(x_t) at every t.
            For temporal operators the aggregate value is broadcast back to
            [batch, horizon] so gradients flow to all participating timesteps.
        """
        raise NotImplementedError

    # Convenience operators so users can write φ1 & φ2, φ1 | φ2
    def __and__(self, other: STLFormula) -> STLFormula:
        return And(self, other)

    def __or__(self, other: STLFormula) -> STLFormula:
        return Or(self, other)


# -- Predicate ---------------------------------------------------------------

class Predicate(STLFormula):
    """Atomic predicate: b(τ, t) = h(x_t).

    Args:
        h : Callable[[Tensor], Tensor]
            Takes a single state slice x_t of shape [batch, state_dim] and
            returns a scalar-per-batch tensor of shape [batch].
            'state_dim' here means the full transition_dim slice at time t
            (actions + observations); the predicate is responsible for
            indexing into the dimensions it cares about.
        name : optional label for debugging
    """

    def __init__(self, h: Callable[[Tensor], Tensor], name: str = "μ"):
        self.h = h
        self.name = name

    def barrier(self, traj: Tensor, s: float, kappa: float) -> Tensor:
        # traj: [batch, horizon, transition_dim]
        batch, horizon, _ = traj.shape
        values = self.h(traj)           # h can operate on full traj or slice
        # If h returns [batch, horizon] already, use as-is.
        # If it returns [batch] (single timestep), expand.
        if values.dim() == 1:
            values = values.unsqueeze(1).expand(batch, horizon)
        return values                   # [batch, horizon]


# -- Temporal operators ------------------------------------------------------

class Always(STLFormula):
    """□[a,b] φ  —  φ must hold at every t ∈ {a, …, b}.

    Barrier = smooth_min_{t∈[a,b]} b_φ(τ, t), broadcast to [batch, horizon].
    """

    def __init__(self, phi: STLFormula, a: int, b: int):
        self.phi = phi
        self.a = a
        self.b = b

    def barrier(self, traj: Tensor, s: float, kappa: float) -> Tensor:
        batch, horizon, _ = traj.shape
        b_end = min(self.b + 1, horizon)
        a_start = min(self.a, horizon)

        phi_vals = self.phi.barrier(traj, s, kappa)  # [batch, horizon]
        window = phi_vals[:, a_start:b_end]           # [batch, window_len]

        if window.shape[1] == 0:
            # Window outside trajectory — treat as unconstrained (always satisfied)
            return torch.zeros(batch, horizon, device=traj.device, dtype=traj.dtype)

        agg = smooth_min(window, kappa, dim=1)        # [batch]
        # Broadcast back: set all timesteps in window to the aggregate so
        # gradient flows everywhere; timesteps outside window get 0 (no penalty).
        out = torch.zeros(batch, horizon, device=traj.device, dtype=traj.dtype)
        out[:, a_start:b_end] = agg.unsqueeze(1).expand(batch, b_end - a_start)
        return out


class Eventually(STLFormula):
    """◇[a,b] φ  —  φ must hold for some t ∈ {a, …, b}.

    Barrier = smooth_max_{t∈[a,b]} b_φ(τ, t) + β(s), broadcast to [batch, horizon].

    The funnel slack β(s) = β₀ · (1 - s) starts at β₀ when s=0 (early denoising,
    t=T) and reaches 0 when s=1 (final step, t=0), encouraging progressive
    commitment to satisfying the eventually constraint.
    """

    def __init__(self, phi: STLFormula, a: int, b: int, beta0: float = 0.3):
        self.phi = phi
        self.a = a
        self.b = b
        self.beta0 = beta0

    def barrier(self, traj: Tensor, s: float, kappa: float) -> Tensor:
        batch, horizon, _ = traj.shape
        b_end = min(self.b + 1, horizon)
        a_start = min(self.a, horizon)

        phi_vals = self.phi.barrier(traj, s, kappa)   # [batch, horizon]
        window = phi_vals[:, a_start:b_end]            # [batch, window_len]

        if window.shape[1] == 0:
            return torch.zeros(batch, horizon, device=traj.device, dtype=traj.dtype)

        agg = smooth_max(window, kappa, dim=1)         # [batch]

        # Funnel slack: relaxes the constraint early in denoising
        beta = self.beta0 * (1.0 - s)
        agg = agg + beta

        out = torch.zeros(batch, horizon, device=traj.device, dtype=traj.dtype)
        out[:, a_start:b_end] = agg.unsqueeze(1).expand(batch, b_end - a_start)
        return out


# -- Boolean connectives -----------------------------------------------------

class And(STLFormula):
    """φ₁ ∧ φ₂  — barrier = smooth_min(b₁, b₂) element-wise."""

    def __init__(self, phi1: STLFormula, phi2: STLFormula):
        self.phi1 = phi1
        self.phi2 = phi2

    def barrier(self, traj: Tensor, s: float, kappa: float) -> Tensor:
        b1 = self.phi1.barrier(traj, s, kappa)
        b2 = self.phi2.barrier(traj, s, kappa)
        # Element-wise smooth min over the two barriers
        stacked = torch.stack([b1, b2], dim=-1)        # [batch, horizon, 2]
        return smooth_min(stacked, kappa, dim=-1)       # [batch, horizon]


class Or(STLFormula):
    """φ₁ ∨ φ₂  — barrier = smooth_max(b₁, b₂) element-wise."""

    def __init__(self, phi1: STLFormula, phi2: STLFormula):
        self.phi1 = phi1
        self.phi2 = phi2

    def barrier(self, traj: Tensor, s: float, kappa: float) -> Tensor:
        b1 = self.phi1.barrier(traj, s, kappa)
        b2 = self.phi2.barrier(traj, s, kappa)
        stacked = torch.stack([b1, b2], dim=-1)        # [batch, horizon, 2]
        return smooth_max(stacked, kappa, dim=-1)       # [batch, horizon]


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

def compile(
    phi: STLFormula,
    kappa: float = 10.0,
) -> Callable[[Tensor, float], Tensor]:
    """Compile an STL formula into a differentiable barrier callable.

    Args:
        phi   : root STL formula node
        kappa : smoothing temperature for smooth min/max (default 10.0)

    Returns:
        barrier : Callable[[Tensor, float], Tensor]
            barrier(trajectory, s) -> [batch, horizon]

            trajectory : [batch, horizon, transition_dim] normalized tensor
            s          : denoising progress ∈ [0, 1]  (pass 0.0 if not needed)

    Usage in diffusion.py g_x_funcs:
        The existing interface calls g_x_func(state) and expects a tensor with
        shape compatible with [batch, horizon] (or [batch, horizon, 1]).
        The compiled barrier returns exactly [batch, horizon].

        For the denoising-progress-aware path, the diffusion loop should call
            g_x_func(state, s=s)
        where s is the current progress. The default s=0.0 means no funnel
        relaxation (safe, conservative fallback).
    """

    def barrier(trajectory: Tensor, s: float = 0.0) -> Tensor:
        return phi.barrier(trajectory, s, kappa)

    barrier.__stl_formula__ = phi       # attach for introspection
    barrier.__kappa__ = kappa
    return barrier


# ---------------------------------------------------------------------------
# Helper: split a conjunction into a list of sub-barriers
# ---------------------------------------------------------------------------

def split_conjunction(
    phi: STLFormula,
    kappa: float = 10.0,
) -> List[Callable[[Tensor, float], Tensor]]:
    """Walk the top-level And tree and compile each leaf as a separate barrier.

    This is the recommended entry-point for registering STL specs into
    diffusion.py: conjunctions of sub-formulae get independent dual variables,
    which handles scale differences between sub-formulae much better than a
    single combined barrier.

    Example:
        spec = Always(obs_pred, 0, H) & Eventually(goal_pred, 20, 40)
        barriers = split_conjunction(spec, kappa=10.0)
        diffusion.set_stl_barriers(barriers)
    """

    leaves: List[STLFormula] = []

    def _collect(node: STLFormula):
        # Unwrap smooth-And nodes recursively
        if isinstance(node, And):
            _collect(node.phi1)
            _collect(node.phi2)
        else:
            leaves.append(node)

    _collect(phi)
    return [compile(leaf, kappa=kappa) for leaf in leaves]
