import torch
from diffuser.models.temporal import TemporalUnet
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.stl.compiler import Predicate, Always, split_conjunction
from diffuser.stl.predicates import obstacle_avoidance, velocity_bound

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH, HORIZON, OBS_DIM, ACT_DIM = 4, 32, 4, 2
TRANSITION_DIM = OBS_DIM + ACT_DIM

# --- 1. Build model ---
model = TemporalUnet(
    horizon=HORIZON,
    transition_dim=TRANSITION_DIM,
    cond_dim=OBS_DIM,          # required positional arg
    dim=32,
    dim_mults=(1, 2, 4),
).to(DEVICE)

diffusion = GaussianDiffusion(
    model=model,
    horizon=HORIZON,
    observation_dim=OBS_DIM,
    action_dim=ACT_DIM,
    n_timesteps=64,
).to(DEVICE)

# --- 2. Build STL barriers ---
phi = (
    Always(Predicate(obstacle_avoidance(center=(0.0, 0.0), radius=0.1, action_dim=ACT_DIM)), a=0, b=HORIZON-1) &
    Always(Predicate(velocity_bound(max_speed=0.5, action_dim=ACT_DIM)), a=0, b=HORIZON-1)
)
barriers = split_conjunction(phi, kappa=10.0)

# --- 3. Register barriers ---
diffusion.set_stl_barriers(barriers, num_batch=BATCH)
assert len(diffusion.g_x_funcs) == 2

# --- 4. calc_grad ---
x = torch.randn(BATCH, HORIZON, TRANSITION_DIM, device=DEVICE)
for algo in ['safe_diffuser', 'primal_dual', 'augmented_lagrangian']:
    diffusion.algorithm = algo
    if algo == 'augmented_lagrangian':
        diffusion.set_stl_barriers(barriers, num_batch=BATCH)  # re-alloc slack vars
    grads, vios = diffusion.calc_grad(x, s=0.5)
    assert grads.shape == (2, BATCH, HORIZON, TRANSITION_DIM), f"{algo}: {grads.shape}"
    print(f"{algo}: grads OK, vios min={vios.min():.3f}")

print("Level 2 passed.")
