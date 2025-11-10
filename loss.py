# import torch
# import numpy as np

# class SILoss:
#     def __init__(
#             self,
#             path_type="linear",
#             weighting="uniform",
#             # New parameters
#             time_sampler="logit_normal",  # Time sampling strategy: "uniform" or "logit_normal"
#             time_mu=-0.4,                 # Mean parameter for logit_normal distribution
#             time_sigma=1.0,               # Std parameter for logit_normal distribution
#             ratio_r_not_equal_t=0.75,     # Ratio of samples where r≠t
#             adaptive_p=1.0,               # Power param for adaptive weighting
#             label_dropout_prob=0.1,       # Drop out label
#             # CFG related params
#             cfg_omega=1.0,                # CFG omega param, default 1.0 means no CFG
#             cfg_kappa=0.0,                # CFG kappa param for mixing class-cond and uncond u
#             cfg_min_t=0.0,                # Minium CFG trigger time 
#             cfg_max_t=0.8,                # Maximum CFG trigger time
#             ):
#         self.weighting = weighting
#         self.path_type = path_type
        
#         # Time sampling config
#         self.time_sampler = time_sampler
#         self.time_mu = time_mu
#         self.time_sigma = time_sigma
#         self.ratio_r_not_equal_t = ratio_r_not_equal_t
#         self.label_dropout_prob = label_dropout_prob
#         # Adaptive weight config
#         self.adaptive_p = adaptive_p
        
#         # CFG config
#         self.cfg_omega = cfg_omega
#         self.cfg_kappa = cfg_kappa
#         self.cfg_min_t = cfg_min_t
#         self.cfg_max_t = cfg_max_t
        

#     def interpolant(self, t):
#         """Define interpolation function"""
#         if self.path_type == "linear":
#             alpha_t = 1 - t
#             sigma_t = t
#             d_alpha_t = -1
#             d_sigma_t =  1
#         elif self.path_type == "cosine":
#             alpha_t = torch.cos(t * np.pi / 2)
#             sigma_t = torch.sin(t * np.pi / 2)
#             d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
#             d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
#         else:
#             raise NotImplementedError()

#         return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
#     def sample_time_steps(self, batch_size, device):
#         """Sample time steps (r, t) according to the configured sampler"""
#         # Step1: Sample two time points
#         if self.time_sampler == "uniform":
#             time_samples = torch.rand(batch_size, 2, device=device)
#         elif self.time_sampler == "logit_normal":
#             normal_samples = torch.randn(batch_size, 2, device=device)
#             normal_samples = normal_samples * self.time_sigma + self.time_mu
#             time_samples = torch.sigmoid(normal_samples)
#         else:
#             raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
#         # Step2: Ensure t > r by sorting
#         sorted_samples, _ = torch.sort(time_samples, dim=1)
#         r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
#         # Step3: Control the proportion of r=t samples
#         fraction_equal = 1.0 - self.ratio_r_not_equal_t  # e.g., 0.75 means 75% of samples have r=t
#         # Create a mask for samples where r should equal t
#         equal_mask = torch.rand(batch_size, device=device) < fraction_equal
#         # Apply the mask: where equal_mask is True, set r=t (replace)
#         r = torch.where(equal_mask, t, r)
        
#         return r, t 
    
#     def __call__(self, model, images, model_kwargs=None):
#         """
#         Compute MeanFlow loss function with bootstrap mechanism
#         """
#         if model_kwargs == None:
#             model_kwargs = {}
#         else:
#             model_kwargs = model_kwargs.copy()

#         batch_size = images.shape[0]
#         device = images.device

#         unconditional_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
#         if model_kwargs.get('y') is not None and self.label_dropout_prob > 0:
#             y = model_kwargs['y'].clone()  
#             batch_size = y.shape[0]
#             num_classes = model.module.num_classes
#             dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob
            
#             y[dropout_mask] = num_classes
#             model_kwargs['y'] = y
#             unconditional_mask = dropout_mask  # Used for unconditional velocity computation

#         # Sample time steps
#         r, t = self.sample_time_steps(batch_size, device)

#         noises = torch.randn_like(images)
        
#         # Calculate interpolation and z_t
#         alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
#         z_t = alpha_t * images + sigma_t * noises #(1-t) * images + t * noise
        
#         # Calculate instantaneous velocity v_t 
#         v_t = d_alpha_t * images + d_sigma_t * noises
#         time_diff = (t - r).view(-1, 1, 1, 1)
                
#         u_target = torch.zeros_like(v_t)
        
#         u = model(z_t, r, t, **model_kwargs)
        
        
#         # Check if CFG should be applied (exclude unconditional samples)
#         cfg_time_mask = (t >= self.cfg_min_t) & (t <= self.cfg_max_t) & (~unconditional_mask)
        
#         if model_kwargs.get('y') is not None and cfg_time_mask.any():
#             # Split samples into CFG and non-CFG
#             cfg_indices = torch.where(cfg_time_mask)[0]
#             no_cfg_indices = torch.where(~cfg_time_mask)[0]
            
#             u_target = torch.zeros_like(v_t)
            
#             # Process CFG samples
#             if len(cfg_indices) > 0:
#                 cfg_z_t = z_t[cfg_indices]
#                 cfg_v_t = v_t[cfg_indices]
#                 cfg_r = r[cfg_indices]
#                 cfg_t = t[cfg_indices]
#                 cfg_time_diff = time_diff[cfg_indices]
                
#                 cfg_kwargs = {}
#                 for k, v in model_kwargs.items():
#                     if torch.is_tensor(v) and v.shape[0] == batch_size:
#                         cfg_kwargs[k] = v[cfg_indices]
#                     else:
#                         cfg_kwargs[k] = v
                
#                 # Compute v_tilde for CFG samples
#                 cfg_y = cfg_kwargs.get('y')
#                 num_classes = model.module.num_classes
                
#                 cfg_z_t_batch = torch.cat([cfg_z_t, cfg_z_t], dim=0)
#                 cfg_t_batch = torch.cat([cfg_t, cfg_t], dim=0)
#                 cfg_t_end_batch = torch.cat([cfg_t, cfg_t], dim=0)
#                 cfg_y_batch = torch.cat([cfg_y, torch.full_like(cfg_y, num_classes)], dim=0)
                
#                 cfg_combined_kwargs = cfg_kwargs.copy()
#                 cfg_combined_kwargs['y'] = cfg_y_batch
                
#                 with torch.no_grad():
#                     cfg_combined_u_at_t = model(cfg_z_t_batch, cfg_t_batch, cfg_t_end_batch, **cfg_combined_kwargs)
#                     cfg_u_cond_at_t, cfg_u_uncond_at_t = torch.chunk(cfg_combined_u_at_t, 2, dim=0)
#                     cfg_v_tilde = (self.cfg_omega * cfg_v_t + 
#                             self.cfg_kappa * cfg_u_cond_at_t + 
#                             (1 - self.cfg_omega - self.cfg_kappa) * cfg_u_uncond_at_t)
                
#                 # Compute JVP with CFG velocity
#                 def fn_current_cfg(z, cur_r, cur_t):
#                     return model(z, cur_r, cur_t, **cfg_kwargs)
                
#                 primals = (cfg_z_t, cfg_r, cfg_t)
#                 tangents = (cfg_v_tilde, torch.zeros_like(cfg_r), torch.ones_like(cfg_t))
#                 _, cfg_dudt = jvp(fn_current_cfg, primals,tangents)
                
#                 cfg_u_target = cfg_v_tilde - cfg_time_diff * cfg_dudt
#                 u_target[cfg_indices] = cfg_u_target
            
#             # Process non-CFG samples (including unconditional ones)
#             if len(no_cfg_indices) > 0:
#                 no_cfg_z_t = z_t[no_cfg_indices]
#                 no_cfg_v_t = v_t[no_cfg_indices]
#                 no_cfg_r = r[no_cfg_indices]
#                 no_cfg_t = t[no_cfg_indices]
#                 no_cfg_time_diff = time_diff[no_cfg_indices]
                
#                 no_cfg_kwargs = {}
#                 for k, v in model_kwargs.items():
#                     if torch.is_tensor(v) and v.shape[0] == batch_size:
#                         no_cfg_kwargs[k] = v[no_cfg_indices]
#                     else:
#                         no_cfg_kwargs[k] = v
                
#                 def fn_current_no_cfg(z, cur_r, cur_t):
#                     return model(z, cur_r, cur_t, **no_cfg_kwargs)
                
#                 primals = (no_cfg_z_t, no_cfg_r, no_cfg_t)
#                 tangents = (no_cfg_v_t, torch.zeros_like(no_cfg_r), torch.ones_like(no_cfg_t))
#                 _, no_cfg_dudt = jvp(fn_current_no_cfg,primals,tangents)
                
#                 no_cfg_u_target = no_cfg_v_t - no_cfg_time_diff * no_cfg_dudt
#                 u_target[no_cfg_indices] = no_cfg_u_target
#         else:
#             # No labels or no CFG applicable samples, use standard JVP
#             primals = (z_t, r, t)
#             tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
            
#             def fn_current(z, cur_r, cur_t):
#                 return model(z, cur_r, cur_t, **model_kwargs)

#             _, dudt = jvp(fn_current,primals,tangents)
            
#             u_target = v_t - time_diff * dudt
                
#         # Detach the target to prevent gradient flow        
#         error = u - u_target.detach()
#         loss_mid = torch.sum((error**2).reshape(error.shape[0],-1), dim=-1)
#         # Apply adaptive weighting based on configuration
#         if self.weighting == "adaptive":
#             weights = 1.0 / (loss_mid.detach() + 1e-3).pow(self.adaptive_p)
#             loss = weights * loss_mid          
#         else:
#             loss = loss_mid
#         loss_mean_ref = torch.mean((error**2))
#         return loss, loss_mean_ref

import torch
from torch.autograd.functional import jvp as torch_jvp

class SILoss:
    """
    Implements the Mean-Flow Bootstrap loss from the first script.

    This class encapsulates the time sampling, target velocity calculation
    (including CFG), and the efficient single-call JVP loss.
    """
    def __init__(
        self,
        rate_same=0.25,
        p_mean=-0.4,
        p_std=1.0,
        cfg_omega=1.0,
        adaptive_p=1.0
    ):
        """
        Initializes the loss hyperparameters.

        Args:
            rate_same (float): Probability of sampling r == t.
            p_mean (float): Mean for the logit-normal time sampler.
            p_std (float): Standard deviation for the logit-normal time sampler.
            cfg_omega (float): Classifier-Free Guidance weight. 1.0 disables CFG.
            adaptive_p (float): Power for adaptive loss weighting.
        """
        self.rate_same = rate_same
        self.p_mean = p_mean
        self.p_std = p_std
        self.cfg_omega = cfg_omega
        self.adaptive_p = adaptive_p

    def __call__(self, model, x, model_kwargs=None):
        """
        Calculates the Mean-Flow Bootstrap loss.

        Args:
            model (torch.nn.Module): The DiT model (should be the DDP wrapper).
            x (torch.Tensor): The input latents (x_0).
            model_kwargs (dict, optional): Dictionary of conditional inputs, e.g., {"y": ...}.

        Returns:
            tuple:
                - torch.Tensor: The final weighted loss (for backward pass).
                - torch.Tensor: The unweighted loss (for logging).
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        y = model_kwargs.get("y")
        # Access num_classes from the underlying module if model is DDP
        try:
            num_classes = model.module.num_classes
        except AttributeError:
            num_classes = model.num_classes

        device = x.device
        b = x.shape[0]

        # 1. Sample time steps r and t
        t_rand = torch.randn((2, b), device=device)
        # Apply rate_same logic
        t_rand = torch.where(
            torch.randn((1, b), device=device) < self.rate_same, 
            t_rand, 
            t_rand[:1].repeat(2, 1)
        )
        # Logit-normal sampling and sorting
        r, t = t_rand.mul(self.p_std).add(self.p_mean).sigmoid().msort().unbind()

        # 2. Define the forward function for JVP
        # This wrapper captures the current model and kwargs
        def forward_fn(z, r_in, t_in):
            return model(z, r_in, t_in, **model_kwargs)

        # 3. Calculate noise, z_t, and target velocity v_t
        noise = torch.randn_like(x)
        z = (1 - t).view(-1, 1, 1, 1) * x + t.view(-1, 1, 1, 1) * noise
        v = noise - x  # Target velocity

        # 4. Apply Classifier-Free Guidance to the *target velocity*
        if self.cfg_omega != 1.0 and y is not None:
            with torch.no_grad():
                # Get unconditional velocity prediction
                uncond_kwargs = model_kwargs.copy()
                uncond_kwargs['y'] = num_classes * torch.ones_like(y)
                v_uncond = model(z, t, t, **uncond_kwargs)
                
                # Modify the target velocity v
                v = self.cfg_omega * v + (1 - self.cfg_omega) * v_uncond

        # 5. Set up JVP tangents
        # We want the derivative d(model output) / d(time)
        # Tangents are (dz/dt, dr/dt, dt/dt)
        # dz/dt = v (velocity), dr/dt = 0, dt/dt = 1
        tangents = (v, torch.zeros((b), device=device), torch.ones((b), device=device))  # v is now (B, 4, 32, 32)

        # 6. Perform efficient JVP call
        # This one call runs the forward pass *once* and gets both results
        u, dudt = torch_jvp(
            forward_fn, 
            (z, r, t),  # Corrected primals order
            tangents,
            create_graph=True
        )

        # 7. Calculate the loss target
        # u_tgt = v - (t - r) * (du/dt)
        u_tgt = v - (t - r).view(-1, 1, 1, 1) * dudt.detach()

        # 8. Calculate loss
        # Per-sample unweighted loss: (u - u_tgt)^2
        loss_unweighted = (u - u_tgt).pow(2).sum((1, 2, 3))

        # 9. Apply adaptive weighting
        w = 1 / (loss_unweighted.detach() + 1e-3).pow(self.adaptive_p)
        
        # 10. Return final weighted loss and unweighted loss for logging
        loss_weighted = (loss_unweighted * w).mean()
        
        return loss_weighted, loss_unweighted.detach().mean()
