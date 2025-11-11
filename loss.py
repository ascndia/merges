import torch
import numpy as np
# from torch.func import jvp  # REMOVED

class SILoss:
    def __init__(
            self,
            path_type="linear",
            weighting="uniform",
            # New parameters
            time_sampler="logit_normal",  # Time sampling strategy: "uniform" or "logit_normal"
            time_mu=-0.4,                 # Mean parameter for logit_normal distribution
            time_sigma=1.0,               # Std parameter for logit_normal distribution
            ratio_r_not_equal_t=0.75,     # Ratio of samples where r≠t
            adaptive_p=1.0,               # Power param for adaptive weighting
            label_dropout_prob=0.1,       # Drop out label
            # CFG related params
            cfg_omega=1.0,                # CFG omega param, default 1.0 means no CFG
            cfg_kappa=0.0,                # CFG kappa param for mixing class-cond and uncond u
            cfg_min_t=0.0,                # Minium CFG trigger time 
            cfg_max_t=0.8,                # Maximum CFG trigger time
            
            # CHANGED: Added DDE epsilon
            differential_epsilon=0.005,
            ):
        self.weighting = weighting
        self.path_type = path_type
        
        # Time sampling config
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.label_dropout_prob = label_dropout_prob
        # Adaptive weight config
        self.adaptive_p = adaptive_p
        
        # CFG config
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.cfg_min_t = cfg_min_t
        self.cfg_max_t = cfg_max_t

        # CHANGED: Added DDE epsilon
        self.differential_epsilon = differential_epsilon
        

    def interpolant(self, t):
        """Define interpolation function"""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_time_steps(self, batch_size, device):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step1: Sample two time points
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        # Step2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        # Step3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t  # e.g., 0.75 means 75% of samples have r=t
        # Create a mask for samples where r should equal t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        # Apply the mask: where equal_mask is True, set r=t (replace)
        r = torch.where(equal_mask, t, r)
        
        return r, t 
    
    # # CHANGED: Added DDE helper method
    # @torch.no_grad()
    # def _dde_derivative(self, model, z_t, r, t, v_t, model_kwargs):
    #     """
    #     Compute the DDE approximation of the JVP: (du/dz_t * v_t + du/dt * 1)
    #     using central difference.
    #     """
    #     epsilon = self.differential_epsilon
        
    #     # f(x + eps * v)
    #     # x = (z_t, r, t), v = (v_t, 0, 1)
    #     # x + eps * v = (z_t + eps * v_t, r, t + eps)
    #     f_plus = model(
    #         z_t + epsilon * v_t, 
    #         r, 
    #         t + epsilon, 
    #         **model_kwargs
    #     )
        
    #     # f(x - eps * v)
    #     # x - eps * v = (z_t - eps * v_t, r, t - eps)
    #     f_minus = model(
    #         z_t - epsilon * v_t, 
    #         r, 
    #         t - epsilon, 
    #         **model_kwargs
    #     )
        
    #     # Central difference: (f_plus - f_minus) / (2 * epsilon)
    #     dudt = (f_plus - f_minus) / (2 * epsilon)
        
    #     return dudt
    @torch.no_grad()
    def _dde_derivative(self, model, x, z, t, r, model_kwargs, rng_state=None):
        """
        Differential Derivation Equation (DDE) approximation for df/dt
        Equivalent to TransitionSchedule.dde_derivative from TiM.
        Args:
            model: neural network f_θ(x_t, t, r, ...)
            x: clean input (B, C, H, W)
            z: noise sample (B, C, H, W)
            t: timestep tensor (B,)
            r: reference timestep tensor (B,)
            model_kwargs: dict for model inputs (copied before in-place)
            rng_state: RNG state to ensure determinism
        Returns:
            dF_dv_dt: tensor of shape (B, C, H, W)
        """
        eps = self.differential_epsilon
        B = x.size(0)

        # ensure device and dtype consistency
        device, dtype = x.device, x.dtype
        t = t.view(-1).to(device=device, dtype=dtype)
        r = r.view(-1).to(device=device, dtype=dtype)

        # compute t±ε and corresponding interpolants
        t_plus = (t + eps).clamp(0.0, 1.0)
        t_minus = (t - eps).clamp(0.0, 1.0)

        alpha_plus, sigma_plus, _, _ = self.interpolant(t_plus.view(-1, 1, 1, 1))
        alpha_minus, sigma_minus, _, _ = self.interpolant(t_minus.view(-1, 1, 1, 1))

        # x_{t±ε} = α * x + σ * z
        x_plus = alpha_plus * x + sigma_plus * z
        x_minus = alpha_minus * x + sigma_minus * z

        # make deep copies of kwargs
        kwargs_plus, kwargs_minus = {}, {}
        for k, v in model_kwargs.items():
            if torch.is_tensor(v) and v.shape[0] == B:
                kwargs_plus[k] = v.clone()
                kwargs_minus[k] = v.clone()
            else:
                kwargs_plus[k] = v
                kwargs_minus[k] = v

        # preserve RNG determinism
        if rng_state is None:
            rng_state = torch.cuda.get_rng_state()

        torch.cuda.set_rng_state(rng_state)
        f_plus = model(x_plus, t_plus, r, **kwargs_plus)

        torch.cuda.set_rng_state(rng_state)
        f_minus = model(x_minus, t_minus, r, **kwargs_minus)

        dF_dv_dt = (f_plus - f_minus) / (2 * eps)
        return dF_dv_dt

    
    def __call__(self, model, images, model_kwargs=None):
        """
        Compute MeanFlow loss function with bootstrap mechanism
        """
        if model_kwargs == None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()

        batch_size = images.shape[0]
        device = images.device
        rng_state = torch.cuda.get_rng_state()
        unconditional_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        if model_kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = model_kwargs['y'].clone()  
            batch_size = y.shape[0]
            if hasattr(model, 'module') and hasattr(model.module, 'num_classes'):
                num_classes = model.module.num_classes
            else:
                num_classes = model.num_classes
            dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob
            
            y[dropout_mask] = num_classes
            model_kwargs['y'] = y
            unconditional_mask = dropout_mask  # Used for unconditional velocity computation

        # Sample time steps
        r, t = self.sample_time_steps(batch_size, device)

        noises = torch.randn_like(images)
        
        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises #(1-t) * images + t * noise
        
        # Calculate instantaneous velocity v_t 
        v_t = d_alpha_t * images + d_sigma_t * noises
        time_diff = (t - r).view(-1, 1, 1, 1)
                
        u_target = torch.zeros_like(v_t)
        
        u = model(z_t, r, t, **model_kwargs)
        
        
        # Check if CFG should be applied (exclude unconditional samples)
        cfg_time_mask = (t >= self.cfg_min_t) & (t <= self.cfg_max_t) & (~unconditional_mask)
        
        if model_kwargs.get('y') is not None and cfg_time_mask.any():
            # Split samples into CFG and non-CFG
            cfg_indices = torch.where(cfg_time_mask)[0]
            no_cfg_indices = torch.where(~cfg_time_mask)[0]
            
            u_target = torch.zeros_like(v_t)
            
            # Process CFG samples
            if len(cfg_indices) > 0:
                cfg_z_t = z_t[cfg_indices]
                cfg_v_t = v_t[cfg_indices]
                cfg_r = r[cfg_indices]
                cfg_t = t[cfg_indices]
                cfg_time_diff = time_diff[cfg_indices]
                
                cfg_kwargs = {}
                for k, v in model_kwargs.items():
                    if torch.is_tensor(v) and v.shape[0] == batch_size:
                        cfg_kwargs[k] = v[cfg_indices]
                    else:
                        cfg_kwargs[k] = v
                
                # Compute v_tilde for CFG samples
                cfg_y = cfg_kwargs.get('y')
                if hasattr(model, 'module') and hasattr(model.module, 'num_classes'):
                    num_classes = model.module.num_classes
                else:
                    num_classes = model.num_classes

                cfg_z_t_batch = torch.cat([cfg_z_t, cfg_z_t], dim=0)
                cfg_t_batch = torch.cat([cfg_t, cfg_t], dim=0)
                cfg_t_end_batch = torch.cat([cfg_t, cfg_t], dim=0)
                cfg_y_batch = torch.cat([cfg_y, torch.full_like(cfg_y, num_classes)], dim=0)
                
                cfg_combined_kwargs = cfg_kwargs.copy()
                cfg_combined_kwargs['y'] = cfg_y_batch
                
                with torch.no_grad():
                    cfg_combined_u_at_t = model(cfg_z_t_batch, cfg_t_batch, cfg_t_end_batch, **cfg_combined_kwargs)
                    cfg_u_cond_at_t, cfg_u_uncond_at_t = torch.chunk(cfg_combined_u_at_t, 2, dim=0)
                    cfg_v_tilde = (self.cfg_omega * cfg_v_t + 
                                   self.cfg_kappa * cfg_u_cond_at_t + 
                                   (1 - self.cfg_omega - self.cfg_kappa) * cfg_u_uncond_at_t)
                
                # CHANGED: Compute JVP with CFG velocity using DDE
                cfg_dudt = self._dde_derivative(
                    model,
                    images[cfg_indices],   # x (clean)
                    noises[cfg_indices],   # z (noise)
                    cfg_t,                 # t
                    cfg_r,                 # r
                    cfg_kwargs,
                    rng_state=rng_state
                )
                
                cfg_u_target = cfg_v_tilde - cfg_time_diff * cfg_dudt
                u_target[cfg_indices] = cfg_u_target
            
            # Process non-CFG samples (including unconditional ones)
            if len(no_cfg_indices) > 0:
                no_cfg_z_t = z_t[no_cfg_indices]
                no_cfg_v_t = v_t[no_cfg_indices]
                no_cfg_r = r[no_cfg_indices]
                no_cfg_t = t[no_cfg_indices]
                no_cfg_time_diff = time_diff[no_cfg_indices]
                
                no_cfg_kwargs = {}
                for k, v in model_kwargs.items():
                    if torch.is_tensor(v) and v.shape[0] == batch_size:
                        no_cfg_kwargs[k] = v[no_cfg_indices]
                    else:
                        no_cfg_kwargs[k] = v
                
                # CHANGED: Compute JVP for non-CFG samples using DDE
                no_cfg_dudt = self._dde_derivative(
                    model,
                    images[no_cfg_indices],
                    noises[no_cfg_indices],
                    no_cfg_t,
                    no_cfg_r,
                    no_cfg_kwargs,
                    rng_state=rng_state
                )

                
                no_cfg_u_target = no_cfg_v_t - no_cfg_time_diff * no_cfg_dudt
                u_target[no_cfg_indices] = no_cfg_u_target
        else:
            # No labels or no CFG applicable samples, use standard DDE
            
            # CHANGED: Compute JVP for all samples using DDE
            dudt = self._dde_derivative(
                model,
                images,    # x (clean)
                noises,    # z (noise)
                t,         # t
                r,         # r
                model_kwargs,
                rng_state=rng_state
            )

            
            u_target = v_t - time_diff * dudt
                
        # Detach the target to prevent gradient flow       
        error = u - u_target.detach()
        loss_mid = torch.sum((error**2).reshape(error.shape[0],-1), dim=-1)
        # Apply adaptive weighting based on configuration
        if self.weighting == "adaptive":
            weights = 1.0 / (loss_mid.detach() + 1e-3).pow(self.adaptive_p)
            loss = weights * loss_mid         
        else:
            loss = loss_mid
        loss_mean_ref = torch.mean((error**2))
        return loss, loss_mean_ref