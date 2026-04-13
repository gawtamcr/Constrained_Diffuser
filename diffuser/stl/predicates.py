"""
Predicate library for maze2d STL specifications.

All predicates operate on **normalized** trajectory tensors:
    traj : [batch, horizon, transition_dim]

For maze2d the transition_dim layout is:
    [ action_0, action_1, obs_0, obs_1, obs_2, obs_3 ]
      ↑ action_dim = 2       ↑ obs indices 0-3

The obs layout for maze2d (after LimitsNormalizer, mapped to [-1, 1]):
    obs[0] = x_pos   (normalized)
    obs[1] = y_pos   (normalized)
    obs[2] = x_vel   (normalized)
    obs[3] = y_vel   (normalized)

Each factory function returns a callable h : Tensor -> Tensor where:
    input  : traj  [batch, horizon, transition_dim]
    output : [batch, horizon]  (positive = feasible)

These are designed to be passed directly to the Predicate() node:
    from diffuser.stl.compiler import Predicate
    from diffuser.stl.predicates import obstacle_avoidance
    pred = Predicate(obstacle_avoidance(center=(...), radius=..., action_dim=2))
"""

from __future__ import annotations
from typing import Sequence, Tuple
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _obs_slice(traj: Tensor, action_dim: int) -> Tensor:
    """Extract observation dimensions from a trajectory tensor."""
    return traj[:, :, action_dim:]          # [batch, horizon, obs_dim]


def _pos_slice(traj: Tensor, action_dim: int) -> Tensor:
    """Extract (x_pos, y_pos) — first two observation dims."""
    obs = _obs_slice(traj, action_dim)
    return obs[:, :, :2]                    # [batch, horizon, 2]


def _vel_slice(traj: Tensor, action_dim: int) -> Tensor:
    """Extract (x_vel, y_vel) — obs dims 2 and 3."""
    obs = _obs_slice(traj, action_dim)
    return obs[:, :, 2:4]                   # [batch, horizon, 2]


# ---------------------------------------------------------------------------
# Predicate factories
# ---------------------------------------------------------------------------

def obstacle_avoidance(
    center: Tuple[float, float],
    radius: float,
    action_dim: int = 2,
) -> callable:
    """Keep-OUT of a circular region (obstacle avoidance).

    h(x_t) = ‖pos_t - center‖² - radius²   ≥ 0 outside the circle.

    All values are in *normalized* observation space.

    Args:
        center    : (cx, cy) center of the obstacle in normalized space
        radius    : obstacle radius in normalized space
        action_dim: number of action dimensions (prefix of transition_dim)
    """
    cx, cy = center
    r2 = radius ** 2

    def h(traj: Tensor) -> Tensor:
        pos = _pos_slice(traj, action_dim)          # [batch, horizon, 2]
        cx_ = traj.new_tensor(cx)
        cy_ = traj.new_tensor(cy)
        center_t = torch.stack([cx_, cy_], dim=0)   # [2]
        dist2 = ((pos - center_t) ** 2).sum(dim=-1) # [batch, horizon]
        return dist2 - r2

    h.__name__ = f"obstacle_avoidance(center={center}, r={radius})"
    return h


def keepout_circle(
    center: Tuple[float, float],
    radius: float,
    action_dim: int = 2,
) -> callable:
    """Alias for obstacle_avoidance — must stay OUTSIDE the circle.

    Identical math; provided as a named alias for readability.
    """
    return obstacle_avoidance(center, radius, action_dim)


def keep_inside_circle(
    center: Tuple[float, float],
    radius: float,
    action_dim: int = 2,
) -> callable:
    """Must stay INSIDE a circular region.

    h(x_t) = radius² - ‖pos_t - center‖²   ≥ 0 inside the circle.
    """
    cx, cy = center
    r2 = radius ** 2

    def h(traj: Tensor) -> Tensor:
        pos = _pos_slice(traj, action_dim)
        cx_ = traj.new_tensor(cx)
        cy_ = traj.new_tensor(cy)
        center_t = torch.stack([cx_, cy_], dim=0)
        dist2 = ((pos - center_t) ** 2).sum(dim=-1)
        return r2 - dist2

    h.__name__ = f"keep_inside_circle(center={center}, r={radius})"
    return h


def goal_reaching(
    goal: Tuple[float, float],
    tolerance: float,
    action_dim: int = 2,
) -> callable:
    """Position is within `tolerance` of `goal` (normalized).

    h(x_t) = tolerance² - ‖pos_t - goal‖²   ≥ 0 when close enough.

    Use with Eventually(Predicate(goal_reaching(...)), a, b) to express
    "reach goal between timesteps a and b".
    """
    gx, gy = goal
    tol2 = tolerance ** 2

    def h(traj: Tensor) -> Tensor:
        pos = _pos_slice(traj, action_dim)
        gx_ = traj.new_tensor(gx)
        gy_ = traj.new_tensor(gy)
        goal_t = torch.stack([gx_, gy_], dim=0)
        dist2 = ((pos - goal_t) ** 2).sum(dim=-1)
        return tol2 - dist2

    h.__name__ = f"goal_reaching(goal={goal}, tol={tolerance})"
    return h


def velocity_bound(
    max_speed: float,
    action_dim: int = 2,
) -> callable:
    """Keep total speed below `max_speed` (in normalized space).

    h(x_t) = max_speed² - ‖vel_t‖²   ≥ 0 when speed is within bound.
    """
    vmax2 = max_speed ** 2

    def h(traj: Tensor) -> Tensor:
        vel = _vel_slice(traj, action_dim)            # [batch, horizon, 2]
        speed2 = (vel ** 2).sum(dim=-1)               # [batch, horizon]
        return vmax2 - speed2

    h.__name__ = f"velocity_bound(max_speed={max_speed})"
    return h


def velocity_component_bound(
    max_val: float,
    dim_idx: int,
    action_dim: int = 2,
) -> callable:
    """Keep one velocity component below `max_val` (signed bound).

    h(x_t) = max_val - |vel_t[dim_idx]|   ≥ 0.

    Args:
        max_val   : maximum absolute value of the velocity component
        dim_idx   : 0 = x_vel, 1 = y_vel (within obs[2:4])
        action_dim: prefix action dimensions
    """
    def h(traj: Tensor) -> Tensor:
        vel = _vel_slice(traj, action_dim)             # [batch, horizon, 2]
        component = vel[:, :, dim_idx]                 # [batch, horizon]
        return max_val - component.abs()

    h.__name__ = f"vel_component_bound(dim={dim_idx}, max={max_val})"
    return h


def position_bound(
    dim_idx: int,
    lo: float,
    hi: float,
    action_dim: int = 2,
) -> callable:
    """Keep position dimension `dim_idx` within [lo, hi] (normalized).

    Returns h₁ ∧ h₂ style: both must be ≥ 0.
    h(x_t) = min(pos_t[dim_idx] - lo, hi - pos_t[dim_idx]).

    This is a *single* predicate that encodes the box bound.
    Use it with Predicate() directly.
    """
    def h(traj: Tensor) -> Tensor:
        obs = _obs_slice(traj, action_dim)
        coord = obs[:, :, dim_idx]                     # [batch, horizon]
        return torch.minimum(coord - lo, traj.new_tensor(hi) - coord)

    h.__name__ = f"position_bound(dim={dim_idx}, [{lo}, {hi}])"
    return h


def halfspace_constraint(
    normal: Tuple[float, ...],
    offset: float,
    action_dim: int = 2,
) -> callable:
    """Linear halfspace over observation dimensions: normal · obs ≤ offset.

    h(x_t) = offset - normal · obs_t   ≥ 0.

    This reproduces the original g_x1 constraint:
        g_x1  ⟺  halfspace_constraint((1, 1), 1.3, action_dim=2)
        (obs[0] + obs[1] ≤ 1.3, i.e. normal = [1, 1], offset = 1.3)

    Args:
        normal    : coefficients over obs dimensions (length = obs_dim)
        offset    : right-hand side
        action_dim: prefix action dimensions
    """
    def h(traj: Tensor) -> Tensor:
        obs = _obs_slice(traj, action_dim)             # [batch, horizon, obs_dim]
        n = traj.new_tensor(normal)                    # [obs_dim_used]
        # Dot product over the last axis, only over len(normal) dims
        dot = (obs[:, :, :len(normal)] * n).sum(dim=-1)  # [batch, horizon]
        return offset - dot

    h.__name__ = f"halfspace(normal={normal}, offset={offset})"
    return h
