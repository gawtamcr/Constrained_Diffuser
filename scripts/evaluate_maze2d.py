"""
evaluate_maze2d.py — Head-to-head evaluation of constraint-handling methods on maze2d-umaze.

Methods compared (6 total):
    baseline_primal_dual : original g_x1 triangle constraint, primal-dual
    baseline_projected   : original g_x1, projected gradient (≡ SafeDiffuser [10])
    baseline_alm         : original g_x1, augmented Lagrangian
    stl_primal_dual      : STL spec φ = □(triangle) ∧ ◇(goal), primal-dual
    stl_projected        : STL spec, projected gradient
    stl_alm              : STL spec, augmented Lagrangian

STL spec (hardcoded):
    φ = □[0,H](h_triangle(x))  ∧  ◇[20,40](h_goal(x))

Notes on SafeDiffuser:
    Per Table 4 of the Constrained Diffusers paper, Projected Diffusion is their
    reimplementation of SafeDiffuser [10]. baseline_projected == SafeDiffuser.

Metrics per episode:
    safe_violation          : Σ relu(-g(x_τ)) summed over all g_x_funcs, from
                              the final denoised trajectory (planning-time metric)
    normalized_score        : D4RL normalized score over executed rollout
    goal_reached            : 1 if agent came within 0.5 units of target during rollout
    planning_time_total (s) : wall-clock time for policy() call
    planning_time_per_step  : planning_time_total / horizon
    steps                   : env steps taken in rollout

Usage:
    python scripts/evaluate_maze2d.py [--n_episodes 10] [--out results/eval_maze2d.csv]
"""

import csv
import os
import time

import numpy as np
import torch

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

from diffuser.stl.compiler import Predicate, Eventually, split_conjunction
from diffuser.stl.predicates import halfspace_constraint, goal_reaching


# ---------------------------------------------------------------------------
# CLI — reuse utils.Parser so diffusion_loadpath format strings are resolved
# ---------------------------------------------------------------------------

class Parser(utils.Parser):
    dataset:    str = 'maze2d-umaze-v1'
    config:     str = 'config.maze2d'
    n_episodes: int = 10
    out:        str = 'results/eval_maze2d.csv'


# ---------------------------------------------------------------------------
# STL helpers
# ---------------------------------------------------------------------------

def make_stl_barriers(goal_norm_xy, obs_normalizer, action_dim, horizon):
    norm_mins = obs_normalizer.mins
    norm_maxs = obs_normalizer.maxs

    GOAL_TOLERANCE_RAW = 0.4
    goal_tolerance_norm = GOAL_TOLERANCE_RAW * 2.0 / (
        (norm_maxs[0] - norm_mins[0] + norm_maxs[1] - norm_mins[1]) / 2.0
    )
    EVENTUALLY_A = 20
    EVENTUALLY_B = min(40, horizon - 1)

    phi_triangle = Predicate(
        halfspace_constraint(normal=(1.0, 1.0), offset=1.3, action_dim=action_dim),
        name="triangle_avoidance",
    )
    phi_goal = Eventually(
        Predicate(
            goal_reaching(
                goal=goal_norm_xy,
                tolerance=goal_tolerance_norm,
                action_dim=action_dim,
            ),
            name="goal_reaching",
        ),
        a=EVENTUALLY_A, b=EVENTUALLY_B,
        beta0=0.1,
    )
    return split_conjunction(phi_triangle & phi_goal, kappa=1.0)


def to_norm(raw_xy, dim, norm_mins, norm_maxs):
    return 2.0 * (raw_xy - norm_mins[dim]) / (norm_maxs[dim] - norm_mins[dim]) - 1.0


# ---------------------------------------------------------------------------
# Diffusion state reset between episodes
# ---------------------------------------------------------------------------

def reset_diffusion(diffusion, algorithm, num_constraints, num_batch, device):
    """Reset dual variables, penalty, and slack variables for a fresh episode."""
    diffusion.algorithm = algorithm
    diffusion.penalty = 5e-2
    diffusion.dual_vars = torch.zeros(
        (num_constraints, num_batch, diffusion.horizon),
        dtype=torch.float32, device=device,
    )
    if algorithm == 'augmented_lagrangian':
        diffusion.slack_variables = torch.zeros(
            (num_constraints, num_batch, diffusion.horizon),
            dtype=torch.float32, device=device,
        )


# ---------------------------------------------------------------------------
# Constraint violation metrics from the planned trajectory
# ---------------------------------------------------------------------------

def compute_violations(diffusion, use_stl):
    """
    Compute per-constraint violation metrics from the final denoised trajectory.

    Uses diffusion.last_sample — the normalized full transition tensor
    [batch, horizon, transition_dim] stored at the end of p_sample_loop.
    This is the correct input format for all g_x_funcs.

    triangle_violation    : Σ_t relu(-h_triangle(x_t))  [paper's Σ[g(x)]+]
    goal_window_satisfied : (STL only) 1 if ◇[a,b] goal was satisfied in plan
                            For baseline this is reported as N/A (-1).
    """
    # last_sample: [batch, horizon, transition_dim], normalized, on GPU
    traj = diffusion.last_sample[:1]   # take first batch element: [1, H, trans_dim]

    result = {'triangle_violation': 0.0, 'goal_window_satisfied': -1}

    with torch.no_grad():
        # --- triangle constraint (g_x_funcs[0] for both baseline and STL) ---
        try:
            g_tri = diffusion.g_x_funcs[0](traj, 0.0)
        except (TypeError, IndexError):
            g_tri = diffusion.g_x_funcs[0](traj)
        result['triangle_violation'] = torch.relu(-g_tri).sum().cpu().item()

        # --- goal window (STL only: g_x_funcs[1] = Eventually barrier) ---
        if use_stl and len(diffusion.g_x_funcs) > 1:
            # Evaluate at s=1.0 (no funnel slack) — final denoising quality
            try:
                g_goal = diffusion.g_x_funcs[1](traj, 1.0)
            except (TypeError, IndexError):
                g_goal = diffusion.g_x_funcs[1](traj)
            # Positive anywhere in the output means Eventually was satisfied
            result['goal_window_satisfied'] = int((g_goal >= 0).any().cpu().item())

    return result


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(env, diffusion, dataset, observation, target,
                algorithm, use_stl, batch_size):
    obs_normalizer = dataset.normalizer.normalizers['observations']
    norm_mins = obs_normalizer.mins
    norm_maxs = obs_normalizer.maxs
    action_dim = diffusion.action_dim
    horizon    = diffusion.horizon
    device     = diffusion.betas.device

    # --- set up constraints ---
    if use_stl:
        goal_norm_xy = (
            to_norm(target[0], 0, norm_mins, norm_maxs),
            to_norm(target[1], 1, norm_mins, norm_maxs),
        )
        barriers = make_stl_barriers(goal_norm_xy, obs_normalizer, action_dim, horizon)
        diffusion.set_stl_barriers(barriers, num_batch=batch_size)
        # set_stl_barriers already resets dual_vars; also reset penalty/algorithm
        diffusion.algorithm = algorithm
        diffusion.penalty   = 5e-2
        if algorithm == 'augmented_lagrangian':
            diffusion.slack_variables = torch.zeros(
                (len(barriers), batch_size, horizon),
                dtype=torch.float32, device=device,
            )
    else:
        # Restore original g_x1 (baked into __init__) and reset state
        obs_n = dataset.normalizer.normalizers['observations']

        def g_x1(state, t=None):
            if t is not None:
                return 1.3 - state[:, t, 2] - state[:, t, 3]
            return 1.3 - state[:, :, 2] - state[:, :, 3]

        diffusion.g_x_funcs = [g_x1]
        diffusion.is_cons   = True
        reset_diffusion(diffusion, algorithm, num_constraints=1,
                        num_batch=batch_size, device=device)

    policy = Policy(diffusion, dataset.normalizer)

    cond = {
        0:          observation,
        horizon - 1: np.array([*target, 0, 0]),
    }

    rollout      = [observation.copy()]
    total_reward = 0.0
    goal_reached = False
    steps        = 0

    for t in range(env.max_episode_steps):
        state = env.state_vector().copy()

        if t == 0:
            t0 = time.time()
            action, samples, diffusion_paths, _safe = policy(cond, batch_size=batch_size)
            planning_time_total = time.time() - t0

            sequence   = samples.observations[0]  # [horizon, obs_dim] unnormalized, for rollout
            violations = compute_violations(diffusion, use_stl)

        if t < len(sequence) - 1:
            next_waypoint = sequence[t + 1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0

        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        rollout.append(next_observation.copy())
        steps += 1

        # Goal reached: proximity check (terminal is never True in maze2d live rollout)
        if np.linalg.norm(next_observation[:2] - np.array(target)) < 0.5:
            goal_reached = True
            break

        if terminal:
            break

        observation = next_observation

    score = env.get_normalized_score(total_reward)

    return {
        'triangle_violation':     violations['triangle_violation'],
        'goal_window_satisfied':  violations['goal_window_satisfied'],
        'normalized_score':       score,
        'goal_reached':           int(goal_reached),
        'planning_time_total':    planning_time_total,
        'planning_time_per_step': planning_time_total / horizon,
        'steps':                  steps,
        'rollout':                np.array(rollout),   # [T, obs_dim]
        'sequence':               sequence,            # [H, transition_dim] planned traj
    }


# ---------------------------------------------------------------------------
# Composite image: overlay all episode trajectories for one method
# ---------------------------------------------------------------------------

def save_composite_all_episodes(renderer, rollouts, sequences, savepath, method_name):
    """
    rollouts  : list of [T, obs_dim] arrays (executed)
    sequences : list of [H, obs_dim] arrays (planned)
    Saves two images: executed rollouts composite and planned sequences composite.
    """
    os.makedirs(savepath, exist_ok=True)

    # Stack as [N, T, obs_dim] — pad shorter rollouts to the same length
    max_T = max(r.shape[0] for r in rollouts)
    padded = []
    for r in rollouts:
        pad = np.repeat(r[-1:], max_T - r.shape[0], axis=0)
        padded.append(np.concatenate([r, pad], axis=0))
    rollouts_arr = np.array(padded)   # [N, T, obs_dim]

    # renderer.composite expects [N, T, obs_dim]
    renderer.composite(
        os.path.join(savepath, f'{method_name}_rollouts.png'),
        rollouts_arr, ncol=1,
    )

    sequences_arr = np.array(sequences)  # [N, H, obs_dim]
    renderer.composite(
        os.path.join(savepath, f'{method_name}_planned.png'),
        sequences_arr, ncol=1,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    args = Parser().parse_args('plan')

    env = datasets.load_environment(args.dataset)

    diffusion_experiment = utils.load_diffusion(
        args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch
    )
    diffusion = diffusion_experiment.ema
    dataset   = diffusion_experiment.dataset
    renderer  = diffusion_experiment.renderer

    # Output dirs
    out_dir   = os.path.dirname(args.out) or '.'
    img_dir   = os.path.join(out_dir, 'images')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # Methods: (display_name, algorithm_str, use_stl, note)
    methods = [
        ('baseline_primal_dual', 'primal_dual',          False, 'original g_x1'),
        ('baseline_projected',   'projected_gradient',   False, 'original g_x1 / SafeDiffuser[10]'),
        ('baseline_alm',         'augmented_lagrangian', False, 'original g_x1'),
        ('stl_primal_dual',      'primal_dual',          True,  'STL spec'),
        ('stl_projected',        'projected_gradient',   True,  'STL spec'),
        ('stl_alm',              'augmented_lagrangian', True,  'STL spec'),
    ]

    # Shared (start, goal) pairs — same seed for all methods
    rng = np.random.default_rng(seed=42)
    episodes = []
    for _ in range(args.n_episodes):
        start_xy = env.reset_locations[rng.integers(len(env.reset_locations))]
        goal_candidates = [
            loc for loc in env.goal_locations
            if np.linalg.norm(np.array(loc) - np.array(start_xy)) > 0.5
        ]
        if not goal_candidates:
            goal_candidates = env.goal_locations
        target = tuple(goal_candidates[rng.integers(len(goal_candidates))])
        episodes.append((start_xy, target))

    fieldnames = [
        'episode', 'method', 'note',
        'triangle_violation', 'goal_window_satisfied',
        'normalized_score', 'goal_reached',
        'planning_time_total', 'planning_time_per_step', 'steps',
        'start_x', 'start_y', 'goal_x', 'goal_y',
    ]
    rows = []

    for method_name, algorithm, use_stl, note in methods:
        print(f"\n{'='*65}")
        print(f"Method: {method_name}  [{note}]")
        print(f"{'='*65}")

        ep_rollouts  = []
        ep_sequences = []

        for ep_idx, (start_xy, target) in enumerate(episodes):
            observation = env.reset_to_location(start_xy)

            print(f"  Ep {ep_idx+1:>3}/{args.n_episodes} | "
                  f"start=({start_xy[0]:.2f},{start_xy[1]:.2f}) "
                  f"goal=({target[0]:.2f},{target[1]:.2f})",
                  end=' ... ', flush=True)

            try:
                result = run_episode(
                    env=env,
                    diffusion=diffusion,
                    dataset=dataset,
                    observation=observation,
                    target=target,
                    algorithm=algorithm,
                    use_stl=use_stl,
                    batch_size=args.batch_size,
                )
                ep_rollouts.append(result['rollout'])
                ep_sequences.append(result['sequence'])
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"ERROR: {e}")
                result = {
                    'triangle_violation':     float('nan'),
                    'goal_window_satisfied':  0,
                    'normalized_score':       float('nan'),
                    'goal_reached':           0,
                    'planning_time_total':    float('nan'),
                    'planning_time_per_step': float('nan'),
                    'steps':                  0,
                    'rollout':                np.zeros((1, 4)),
                    'sequence':               np.zeros((diffusion.horizon, 4)),
                }
                ep_rollouts.append(result['rollout'])
                ep_sequences.append(result['sequence'])

            print(f"score={result['normalized_score']:.3f} | "
                  f"tri_viol={result['triangle_violation']:.4f} | "
                  f"goal_win={result['goal_window_satisfied']} | "
                  f"reached={result['goal_reached']} | "
                  f"t={result['planning_time_total']:.2f}s")

            rows.append({
                'episode':                ep_idx,
                'method':                 method_name,
                'note':                   note,
                'triangle_violation':     result['triangle_violation'],
                'goal_window_satisfied':  result['goal_window_satisfied'],
                'normalized_score':       result['normalized_score'],
                'goal_reached':           result['goal_reached'],
                'planning_time_total':    result['planning_time_total'],
                'planning_time_per_step': result['planning_time_per_step'],
                'steps':                  result['steps'],
                'start_x':                start_xy[0],
                'start_y':                start_xy[1],
                'goal_x':                 target[0],
                'goal_y':                 target[1],
            })

        # Composite image for this method (all episodes overlaid)
        try:
            save_composite_all_episodes(
                renderer, ep_rollouts, ep_sequences, img_dir, method_name
            )
            print(f"  -> images saved to {img_dir}/{method_name}_*.png")
        except Exception as e:
            print(f"  -> image save failed: {e}")

    # Write CSV
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary table
    print(f"\n{'='*75}")
    print(f"{'SUMMARY':^75}")
    print(f"{'='*75}")
    hdr = (f"{'Method':<24} {'Score':>7} {'Goal%':>7} {'GoalWin%':>9} "
           f"{'TriViol(mean±std)':>20} {'Time/step':>10} {'TotalTime':>10}")
    print(hdr)
    print('-' * len(hdr))

    for method_name, *_ in methods:
        mr = [r for r in rows if r['method'] == method_name]

        def _mean(key):
            vals = [r[key] for r in mr if not (isinstance(r[key], float) and np.isnan(r[key]))]
            return np.mean(vals) if vals else float('nan')

        def _std(key):
            vals = [r[key] for r in mr if not (isinstance(r[key], float) and np.isnan(r[key]))]
            return np.std(vals) if vals else float('nan')

        tri_str  = f"{_mean('triangle_violation'):.4f}±{_std('triangle_violation'):.4f}"
        gw_vals  = [r['goal_window_satisfied'] for r in mr if r['goal_window_satisfied'] != -1]
        gw_str   = f"{100*np.mean(gw_vals):6.1f}%" if gw_vals else "    N/A"
        print(f"{method_name:<24} "
              f"{_mean('normalized_score'):>7.3f} "
              f"{100*_mean('goal_reached'):>6.1f}% "
              f"{gw_str:>8} "
              f"{tri_str:>20} "
              f"{_mean('planning_time_per_step'):>10.4f}s "
              f"{_mean('planning_time_total'):>10.2f}s")

    print(f"\nCSV  : {args.out}")
    print(f"Images: {img_dir}/")


if __name__ == '__main__':
    main()
