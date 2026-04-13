import torch
from diffuser.models.temporal import TemporalUnet
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.stl.compiler import Predicate, Always, Eventually, split_conjunction
from diffuser.stl.predicates import obstacle_avoidance, velocity_bound, goal_reaching

HORIZON, OBS_DIM, ACT_DIM = 64, 4, 2
BATCH = 1
DEVICE = 'cuda:0'

# Minimal model
model = TemporalUnet(HORIZON, OBS_DIM + ACT_DIM, dim=32, dim_mults=(1, 2, 4)).to(DEVICE)
diffusion = GaussianDiffusion(
    model, HORIZON, OBS_DIM, ACT_DIM,
    n_timesteps=64, clip_denoised=True, predict_epsilon=False,
).to(DEVICE)

# Build STL spec
spec = (
    Always(Predicate(obstacle_avoidance(center=(0.0, 0.0), radius=0.2, action_dim=ACT_DIM)), 0, HORIZON-1)
    & Eventually(Predicate(goal_reaching(goal=(0.8, 0.8), tolerance=0.1, action_dim=ACT_DIM)), 20, 40)
    & Always(Predicate(velocity_bound(max_speed=0.5, action_dim=ACT_DIM)), 0, HORIZON-1)
)
barriers = split_conjunction(spec, kappa=10.0)

# Test all three algorithms
for algo in ['primal_dual', 'projected_gradient', 'augmented_lagrangian']:
    diffusion.algorithm = algo
    diffusion.set_stl_barriers(barriers, num_batch=BATCH)

    # calc_grad with s
    x = torch.randn(BATCH, HORIZON, OBS_DIM + ACT_DIM, device=DEVICE)
    grads, vios = diffusion.calc_grad(x, s=0.5)
    assert grads.shape == (len(barriers), BATCH, HORIZON, OBS_DIM + ACT_DIM)
    assert vios.shape == (len(barriers), BATCH, HORIZON)
    print(f"[{algo}] calc_grad: OK, vios mean = {vios.mean():.4f}")

    # projection
    if algo == 'projected_gradient':
        x_proj = diffusion._project_to_feasible_region(x.clone(), s=0.8)
        assert x_proj.shape == x.shape
        print(f"[{algo}] projection: OK")

print("\nIntegration tests passed.")
