"""
plan_maze2d_stl.py — Planning with an STL specification on maze2d-umaze.

Specification (in plain English):
    "Always stay outside the triangle obstacle AND eventually reach the goal."

Formally (H = horizon):
    φ = □[0,H](h_triangle(x))  ∧  ◇[20,40](h_goal(x))

The triangle obstacle is the same halfspace constraint used by the original
Constrained Diffuser:
    h(x_t) = 1.3 - obs[0] - obs[1] ≥ 0   (x_pos + y_pos ≤ 1.3, normalized)

The spec is split into two barriers (one per conjunct) so that the
primal-dual algorithm gives each sub-formula its own dual variable.

Usage:
    python scripts/plan_maze2d_stl.py --dataset maze2d-umaze-v1
"""

import json
import numpy as np
import os
import time
from os.path import join

import torch

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

# STL compiler
from diffuser.stl.compiler import Predicate, Always, Eventually, split_conjunction
from diffuser.stl.predicates import (
    halfspace_constraint,
    goal_reaching,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args = Parser().parse_args('plan')

env = datasets.load_environment(args.dataset)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

# ---------------------------------------------------------------------------
# STL specification
# ---------------------------------------------------------------------------
# Triangle constraint — same halfspace as the original Constrained Diffuser:
#     h(x_t) = 1.3 - obs[0] - obs[1] ≥ 0   (x_pos + y_pos ≤ 1.3, normalized)
# This is encoded directly in normalized observation space; no conversion needed.

obs_normalizer = dataset.normalizer.normalizers['observations']
norm_mins = obs_normalizer.mins
norm_maxs = obs_normalizer.maxs

def to_norm(raw_xy, dim):
    """Convert a single raw coordinate to normalized space."""
    return 2.0 * (raw_xy - norm_mins[dim]) / (norm_maxs[dim] - norm_mins[dim]) - 1.0

print(f"[STL] obs norm_mins: {norm_mins}")
print(f"[STL] obs norm_maxs: {norm_maxs}")

# -- Goal tolerance in normalized space --
GOAL_TOLERANCE_RAW = 0.4
goal_tolerance_norm = GOAL_TOLERANCE_RAW * 2.0 / ((norm_maxs[0] - norm_mins[0] + norm_maxs[1] - norm_mins[1]) / 2.0)

# -- Time windows --
HORIZON = diffusion.horizon
EVENTUALLY_A = 0
EVENTUALLY_B = min(5, HORIZON - 1)

ACTION_DIM = diffusion.action_dim

# ---------------------------------------------------------------------------
# Build and register STL barriers
# ---------------------------------------------------------------------------

def build_stl_barriers(goal_norm_xy):
    """Build two barriers:
       - Triangle avoidance: bare Predicate (per-timestep, identical to g_x1)
       - Goal reaching: ◇[EVENTUALLY_A, EVENTUALLY_B] (temporal, STL addition)
    """

    # φ₁: triangle avoidance — bare Predicate with no temporal aggregation,
    # identical in behaviour to the original g_x1: 1.3 - obs[0] - obs[1] ≥ 0
    phi_triangle = Predicate(
        halfspace_constraint(normal=(1.0, 1.0), offset=1.3, action_dim=ACTION_DIM),
        name="triangle_avoidance",
    )

    # φ₂: ◇[EVENTUALLY_A, EVENTUALLY_B] goal reaching (with funnel slack β₀=0.3)
    phi_goal = Eventually(
        Predicate(
            goal_reaching(
                goal=goal_norm_xy,
                tolerance=goal_tolerance_norm,
                action_dim=ACTION_DIM,
            ),
            name="goal_reaching",
        ),
        a=EVENTUALLY_A, b=EVENTUALLY_B,
        beta0=0.1, # 0.3,
    )

    # One dual variable per sub-formula
    full_spec = phi_triangle & phi_goal
    barriers = split_conjunction(full_spec, kappa=1.0)
    return barriers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# ---------------------------------------------------------------------------
# Main rollout loop
# ---------------------------------------------------------------------------

safe_batch = []
score_batch = []
comp_time = []

for episode_idx in range(1):
    print(f"\n=== Episode {episode_idx} ===")

    # Sample a random start position from valid reset locations
    start_xy = env.reset_locations[np.random.randint(len(env.reset_locations))]
    observation = env.reset_to_location(start_xy)
    print("start observation:", observation)

    # Sample a random goal from valid goal locations (different from start)
    goal_candidates = [loc for loc in env.goal_locations
                       if np.linalg.norm(np.array(loc) - np.array(start_xy)) > 0.5]
    if not goal_candidates:
        goal_candidates = env.goal_locations
    target = tuple(goal_candidates[np.random.randint(len(goal_candidates))])
    print("target:", target)

    # Normalize goal for STL barrier
    goal_norm_xy = (
        to_norm(target[0], dim=0),
        to_norm(target[1], dim=1),
    )
    print(f"[STL] goal normalized: {goal_norm_xy}")

    # Build and register barriers for this episode's goal
    barriers = build_stl_barriers(goal_norm_xy)
    print(f"[STL] registered {len(barriers)} barrier(s) in diffusion model")
    diffusion.set_stl_barriers(barriers, num_batch=args.batch_size)

    # Re-create Policy after updating barriers (normalizer stays the same)
    policy = Policy(diffusion, dataset.normalizer)

    # Condition: start (t=0) and goal (t=H-1)
    cond = {
        0: observation,
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    rollout = [observation.copy()]
    total_reward = 0.0
    euclidean_reward = 0.0

    for t in range(env.max_episode_steps):
        state = env.state_vector().copy()

        if t == 0:
            start_time = time.time()
            action, samples, diffusion_paths, safe = policy(cond, batch_size=args.batch_size)
            elapsed = time.time() - start_time

            comp_time.append(elapsed)
            safe_batch.append(safe.sum().cpu() if hasattr(safe, 'cpu') else float(safe))

            actions = samples.actions[0]
            sequence = samples.observations[0]
            diffusion_paths_ep = diffusion_paths[0]

            print(f"[STL] planning took {elapsed:.2f}s | "
                  f"constraint violation (safe metric): {safe_batch[-1]:.4f}")

            # Render diffusion denoising animation
            makedirs(join(args.savepath, 'png'))
            renderer.render_diffusion(join(args.savepath, 'diffusion_stl.mp4'), diffusion_paths_ep)

            diff_step = diffusion_paths_ep.shape[0]
            for kk in range(diff_step):
                imgpath = join(args.savepath, f'png/{kk}.png')
                renderer.composite(imgpath, diffusion_paths_ep[kk:kk+1], ncol=1)

        # Execute waypoint-following controller (same as plan_maze2d.py)
        if t < len(sequence) - 1:
            next_waypoint = sequence[t + 1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0

        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward

        euclidean_distance = np.linalg.norm(next_waypoint[:2] - np.array(target))
        euclidean_reward += np.exp(-euclidean_distance)

        score = env.get_normalized_score(total_reward)
        rollout.append(next_observation.copy())

        if terminal or t == HORIZON - 2:
            break

        observation = next_observation

    # Save final rollout composite image
    imgpath = join(args.savepath, 'png/action_stl.png')
    renderer.composite(imgpath, np.expand_dims(np.array(rollout), axis=0), ncol=1)

    score_batch.append(score)
    euclidean_batch_val = euclidean_reward
    print(f"Episode {episode_idx}: score={score:.3f} | "
          f"euclidean_reward={euclidean_batch_val:.3f} | "
          f"safe_violation={safe_batch[-1]:.4f}")

print("\n=== Summary ===")
print(f"Scores:            {score_batch}")
print(f"Safe violations:   {safe_batch}")
print(f"Planning times(s): {[f'{t:.2f}' for t in comp_time]}")
