import numpy as np
import torch
from torch import nn
import pdb
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers
import multiprocessing as mp
import time

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# STL compiler integration — imported lazily so the package is usable even
# if the stl sub-package is not on the path yet.
try:
    from diffuser.stl.compiler import compile as stl_compile, split_conjunction
    _STL_AVAILABLE = True
except ImportError:
    _STL_AVAILABLE = False

def run_env(env_name, init_state, u_r):
    env_name.reset()
    env_name.set_state(init_state[:2], init_state[2:])
    states = np.expand_dims(init_state, axis=0)  # Initial state
    for k in range(u_r.shape[0]):
        next_state,_, _, _ = env_name.step(u_r[k, :])
        # print('next', next_state)
        next_state = np.expand_dims(next_state, axis=0)
        # print('next', next_state.shape)
        # print('state', states.shape)
        if k < u_r.shape[0] - 1:
            states = np.concatenate([states, next_state], axis=0)
        # print(states.shape)

    return states




class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.norm_mins = [0.39643136, 0.44179875]
        self.norm_maxs = [7.2163844, 10.219488]
        

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.penalty = 5e-2
        self.use_equality = False
        
        algorithm = ['primal_dual', 'projected_gradient', 'augmented_lagrangian']
        
        self.algorithm = algorithm[1]
        if self.algorithm == 'augmented_lagrangian':
            self.use_equality = False


        def g_x1(state, t=None):
            proj = False
            if proj:
                if t is not None:
                    g_x = torch.clamp((1.3 - state[:, t, 2] - state[:, t, 3]), max=0)
                else:
                    g_x = torch.clamp((1.3 - state[:, :, 2] - state[:, :, 3]), max=0)
                return g_x
            else:
                if t is not None:
                    g_x = 1.3 - state[:, t, 2] - state[:, t, 3]
                else:

                    g_x = 1.3 - state[:, :, 2] - state[:, :, 3]

                return g_x



        g_x_funcs = [g_x1]
        self.g_x_funcs = g_x_funcs
        # dual variables for constraints
        self.is_cons = True
        self.alpha= 0.8

        num_batch = 1
        num_constraints = len(self.g_x_funcs)
        self.safe = torch.zeros(num_constraints)
        self.dual_vars = torch.zeros((num_constraints, num_batch, self.horizon), dtype=torch.float32, device='cuda:0')*(5 / self.n_timesteps)
        if self.algorithm == 'augmented_lagrangian':
            self.slack_variables = torch.zeros((num_constraints, num_batch, self.horizon), dtype=torch.float32, device='cuda:0')

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        # self.register_buffer('coeff_fx', 1 + betas/2)
        # self.register_buffer('coeff_gx2', betas)
        # self.register_buffer('coeff_gradient', torch.sqrt(1/((1 - alphas_cumprod) * alphas)))


        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)


        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    # ------------------------------------------------------------------
    # STL barrier registration
    # ------------------------------------------------------------------

    def set_stl_barriers(self, barriers, num_batch=1):
        """Replace the constraint functions with compiled STL barriers.

        Call this after constructing GaussianDiffusion and before sampling.

        Args:
            barriers : list of callables produced by stl.compiler.compile()
                       or stl.compiler.split_conjunction().
                       Each barrier has signature:
                           barrier(traj: Tensor, s: float = 0.0) -> Tensor
                           returns [batch, horizon]
            num_batch: batch size for dual variable allocation (default 1)
        """
        self.g_x_funcs = barriers
        num_constraints = len(barriers)
        device = self.betas.device

        self.safe = torch.zeros(num_constraints)
        self.dual_vars = torch.zeros(
            (num_constraints, num_batch, self.horizon),
            dtype=torch.float32, device=device,
        )
        if self.algorithm == 'augmented_lagrangian':
            self.slack_variables = torch.zeros(
                (num_constraints, num_batch, self.horizon),
                dtype=torch.float32, device=device,
            )

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    
    def q_posterior(self, x_start, x_t, t, s: float = 0.0):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if self.is_cons:
            # ['primal_dual', 'projected_gradient', 'augmented_lagrangian', 'admm']
            cbf = False
            if self.algorithm == 'primal_dual':
                grad, vio = self.calc_grad(x_t, s)

                if cbf:
                    shift_dual = self.dual_vars[:,:,:-1]
                    padding = torch.zeros((self.dual_vars.shape[0],self.dual_vars.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_dual = torch.cat([padding, shift_dual], dim=2)

                    posterior_mean = posterior_mean + torch.sum(padding_dual.unsqueeze(-1) * grad, dim=0) - (1-self.alpha) * torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0)

                else:
                    posterior_mean = posterior_mean + torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0) * (-1 if self.use_equality else 1)
                self.dual_update(x_t, cbf, s=s)
            elif self.algorithm == 'augmented_lagrangian':
                grad, vio = self.calc_grad(x_t, s)
                if cbf:
                    shift_dual = self.dual_vars[:,:,:-1]
                    padding = torch.zeros((self.dual_vars.shape[0],self.dual_vars.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_dual = torch.cat([padding, shift_dual], dim=2)
                    shift_vio = vio[:,:,:-1]
                    padding_vio = torch.zeros((vio.shape[0],vio.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_vio_back = torch.cat([padding_vio, shift_vio], dim=2)
                    shift_vio = vio[:,:,1:]
                    padding_vio = torch.zeros((vio.shape[0],vio.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_vio_forw = torch.cat([shift_vio, padding_vio], dim=2)
                    shift_slack = self.slack_variables[:,:,:-1]
                    padding_slack = torch.zeros((self.slack_variables.shape[0],self.slack_variables.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_slack = torch.cat([padding_slack, shift_slack], dim=2)

                    posterior_mean = posterior_mean + (- torch.sum(padding_dual.unsqueeze(-1) * grad, dim=0) + (1-self.alpha) * torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0) - self.penalty * torch.sum((vio - (1 - self.alpha)*padding_vio_back - padding_slack).unsqueeze(-1) * grad , dim=0) + self.penalty * (1 - self.alpha) * torch.sum((padding_vio_forw - (1 - self.alpha)*vio - self.slack_variables).unsqueeze(-1) * grad , dim=0))
                else:
                    posterior_mean = posterior_mean - torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0) - self.penalty * torch.sum((vio - self.slack_variables).unsqueeze(-1) * grad , dim=0)
                self.dual_update_aug(x_t, cbf, s=s)

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, s: float = 0.0):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t, s=s)
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance_langevin(self, x, cond, t, s: float = 0.0):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x.shape)
        alphas_cumprod = extract(self.alphas_cumprod, t, x.shape)
        score = (sqrt_alpha * x_recon - x) / (1 - alphas_cumprod)

        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x.shape)
        beta = extract(self.betas, t, x.shape)
        model_mean = x + posterior_log_variance_clipped.exp() / 2 * score
        x_t = x
        posterior_mean = model_mean
        if self.is_cons:
            # ['primal_dual', 'projected_gradient', 'augmented_lagrangian']
            cbf = False
            if self.algorithm == 'primal_dual':
                grad, vio = self.calc_grad(x_t, s)

                if cbf:
                    shift_dual = self.dual_vars[:,:,:-1]
                    padding = torch.zeros((self.dual_vars.shape[0],self.dual_vars.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_dual = torch.cat([padding, shift_dual], dim=2)

                    posterior_mean = posterior_mean + torch.sum(padding_dual.unsqueeze(-1) * grad, dim=0) - (1-self.alpha) * torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0)

                else:
                    posterior_mean = posterior_mean + torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0) * (-1 if self.use_equality else 1)
                self.dual_update(x_t, cbf, s=s)
            elif self.algorithm == 'augmented_lagrangian':
                grad, vio = self.calc_grad(x_t, s)
                if cbf:
                    shift_dual = self.dual_vars[:,:,:-1]
                    padding = torch.zeros((self.dual_vars.shape[0],self.dual_vars.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_dual = torch.cat([padding, shift_dual], dim=2)
                    shift_vio = vio[:,:,:-1]
                    padding_vio = torch.zeros((vio.shape[0],vio.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_vio_back = torch.cat([padding_vio, shift_vio], dim=2)
                    shift_vio = vio[:,:,1:]
                    padding_vio = torch.zeros((vio.shape[0],vio.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_vio_forw = torch.cat([shift_vio, padding_vio], dim=2)
                    shift_slack = self.slack_variables[:,:,:-1]
                    padding_slack = torch.zeros((self.slack_variables.shape[0],self.slack_variables.shape[1], 1), dtype=torch.float32, device=posterior_mean.device)
                    padding_slack = torch.cat([padding_slack, shift_slack], dim=2)

                    posterior_mean = posterior_mean + (- torch.sum(padding_dual.unsqueeze(-1) * grad, dim=0) + (1-self.alpha) * torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0) - self.penalty * torch.sum((vio - (1 - self.alpha)*padding_vio_back - padding_slack).unsqueeze(-1) * grad , dim=0) + self.penalty * (1 - self.alpha) * torch.sum((padding_vio_forw - (1 - self.alpha)*vio - self.slack_variables).unsqueeze(-1) * grad , dim=0))
                else:
                    posterior_mean = posterior_mean - torch.sum(self.dual_vars.unsqueeze(-1) * grad, dim=0) - self.penalty * torch.sum((vio - self.slack_variables).unsqueeze(-1) * grad , dim=0)
                self.dual_update_aug(x_t, cbf, s=s)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def calc_grad(self, x, s: float = 0.0):
            """Calculate gradients for all barrier / g(x) functions.

            Args:
                x : [batch, horizon, transition_dim] noisy trajectory tensor
                s : denoising progress in [0, 1]  (1 - t / n_timesteps).
                    Forwarded to STL barriers so Eventually funnels work.

            Returns:
                grads : [n_constraints, batch, horizon, transition_dim]
                vios  : [n_constraints, batch, horizon]
            """
            with torch.enable_grad():
                state = x.clone().detach().requires_grad_(True)

                grads = []
                vios = []
                for g_x_func in self.g_x_funcs:
                    # Support both the legacy signature g(state) and the new
                    # STL signature g(state, s).
                    try:
                        g_x = g_x_func(state, s)
                    except (TypeError, IndexError):
                        g_x = g_x_func(state)

                    grad = torch.autograd.grad(g_x.sum(), state)[0]
                    vio = g_x

                    grads.append(grad)
                    vios.append(vio)
                self.safe = torch.vstack(vios)
                return torch.stack(grads, dim=0), torch.stack(vios, dim=0)
        
    def dual_update(self, x, cbf, learning_rate=2.5e-2, s: float = 0.0):

        if cbf == False:
            for i, g_x_func in enumerate(self.g_x_funcs):
                try:
                    g_x = g_x_func(x, s)
                except (TypeError, IndexError):
                    g_x = g_x_func(x)

                if self.use_equality:
                    self.dual_vars[i] = self.dual_vars[i] + self.penalty * g_x.squeeze(-1)
                else:
                    self.dual_vars[i] = torch.clamp(
                        self.dual_vars[i] - learning_rate * g_x.squeeze(-1),
                        min=0
                    )
        else:
            for i, g_x_func in enumerate(self.g_x_funcs):
                for t in range(self.horizon-1):
                    self.dual_vars[i,:, t] = torch.clamp(
                            self.dual_vars[i,:, t] - learning_rate * (g_x_func(x,t+1) - (1 - self.alpha)*g_x_func(x, t)).squeeze(-1),
                            min=0
                        )

    def dual_update_aug(self, x, cbf, s: float = 0.0):

        if cbf == False:
            for i, g_x_func in enumerate(self.g_x_funcs):
                try:
                    g_x = g_x_func(x, s)
                except (TypeError, IndexError):
                    g_x = g_x_func(x)

                if self.use_equality:
                    self.dual_vars[i] = self.dual_vars[i] + self.penalty * (g_x.squeeze(-1) - self.slack_variables[i])
                else:
                    self.slack_variables[i] = torch.clamp(self.dual_vars[i] / self.penalty + g_x.squeeze(-1), min=0)
                    self.dual_vars[i] = self.dual_vars[i] + self.penalty * (g_x.squeeze(-1) - self.slack_variables[i])
                    self.penalty *= 1.0002

        else:
            for i, g_x_func in enumerate(self.g_x_funcs):
                for t in range(self.horizon-1):
                    g_x = g_x_func(x)

                    if self.use_equality:
                        self.dual_vars[i,:,t] = self.dual_vars[i,:,t] + self.penalty * (g_x.squeeze(-1) - self.slack_variables[i])
                    else:
                        self.slack_variables[i,:,t] = torch.clamp(self.dual_vars[i,:,t] / self.penalty + (g_x_func(x,t+1) - (1 - self.alpha)*g_x_func(x, t)).squeeze(-1), min=0)
                        self.dual_vars[i,:,t] = self.dual_vars[i,:,t] + self.penalty * ((g_x_func(x,t+1) - (1 - self.alpha)*g_x_func(x, t)).squeeze(-1) - self.slack_variables[i,:,t])
                        self.penalty *= 1.0001

    def _project_to_feasible_region(self, x, s: float = 0.0):
        """Project trajectory onto the feasible region using a single Newton step.

        For each barrier bᵢ(τ), if bᵢ(τ) < 0 (constraint violated), apply:

            τ ← τ + max(0, -bᵢ(τ)) · ∇bᵢ(τ) / ‖∇bᵢ(τ)‖²

        This is a single Newton step onto the zero-level set of bᵢ, which is the
        closest feasible point for convex barriers and a good local approximation
        for non-convex ones. Applied sequentially over all constraints.

        Args:
            x : [batch, horizon, transition_dim]  current sample (may be in-place modified)
            s : denoising progress passed to STL barriers

        Returns:
            Projected trajectory (same tensor, modified in-place).
        """
        for g_x_func in self.g_x_funcs:
            with torch.enable_grad():
                x_var = x.detach().clone().requires_grad_(True)
                try:
                    g = g_x_func(x_var, s)
                except (TypeError, IndexError):
                    g = g_x_func(x_var)

                # g : [batch, horizon]
                violation = torch.relu(-g)            # positive where violated

                if violation.sum() < 1e-8:
                    continue                           # already feasible for this barrier

                # Gradient of the barrier w.r.t. trajectory
                grad = torch.autograd.grad(g.sum(), x_var)[0]  # [batch, horizon, td]

                # ‖∇b‖² per (batch, horizon) position, summed over td
                grad_norm2 = (grad ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)

                # Newton step: move toward feasibility where violated
                # violation is [batch, horizon]; unsqueeze for broadcast over td
                step = violation.unsqueeze(-1) * grad / grad_norm2   # [batch, H, td]

                x = x + step.detach()

        return x


    @torch.no_grad()
    def p_sample(self, x, cond, t, s: float = 0.0):
        b, *_, device = *x.shape, x.device
        use_ddpm = True
        if use_ddpm:
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, s=s)
        else:
            model_mean, _, model_log_variance = self.p_mean_variance_langevin(x=x, cond=cond, t=t, s=s)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 1).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        xp1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        if self.is_cons and self.algorithm == 'projected_gradient':
            xp1 = self._project_to_feasible_region(xp1, s=s)

        for g_x in self.g_x_funcs:
            try:
                self.safe = torch.relu(-g_x(xp1, s)).sum()
            except (TypeError, IndexError):
                self.safe = torch.relu(-g_x(xp1)).sum()

        return xp1
    

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, env=None, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(-100, self.n_timesteps)):  #-100
            if i <= 0:
                i_ = 1
            elif i > self.n_timesteps:
                i_ = self.n_timesteps - 1
            else:
                i_ = i

            # Denoising progress s ∈ [0, 1]: 0 at the start (most noise),
            # 1 at the end (clean sample). Used by Eventually funnels.
            s = 1.0 - i_ / self.n_timesteps

            timesteps = torch.full((batch_size,), i_, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, s=s)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, env=None, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        print('shape',shape)
        if env is not None:
            return self.p_sample_cob_loop(shape, cond, return_diffusion= True, env=env, *args, **kwargs)
        return self.p_sample_loop(shape, cond, return_diffusion= True, env=None, *args, **kwargs)   ## debug

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, env=None):
        device = x_start.device
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon.to(torch.double), noise.to(torch.double))
        else:
            loss, info = self.loss_fn(x_recon.to(torch.double), x_start.to(torch.double))

        return loss, info
    

    def loss(self, x, cond, env=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, env)

    def forward(self, cond, env=None, *args, **kwargs):
        return self.conditional_sample(cond=cond, env=env, *args, **kwargs)

