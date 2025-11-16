import torch
import numpy as np
import torch.nn.functional as F
# from torch.func import jvp  # REMOVED

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

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
            # REPA & REG
            encoders=[]
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

        self.encoders = encoders
        

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
    
    # @torch.no_grad()
    # def _dde_derivative(self, model, x, z, t, r, model_kwargs, rng_state=None):
    #     """
    #     Differential Derivation Equation (DDE) approximation for df/dt
    #     Equivalent to TransitionSchedule.dde_derivative from TiM.
    #     Args:
    #         model: neural network f_θ(x_t, t, r, ...)
    #         x: clean input (B, C, H, W)
    #         z: noise sample (B, C, H, W)
    #         t: timestep tensor (B,)
    #         r: reference timestep tensor (B,)
    #         model_kwargs: dict for model inputs (copied before in-place)
    #         rng_state: RNG state to ensure determinism
    #     Returns:
    #         dF_dv_dt: tensor of shape (B, C, H, W)
    #     """
    #     eps = self.differential_epsilon
    #     B = x.size(0)

    #     # ensure device and dtype consistency
    #     device, dtype = x.device, x.dtype
    #     t = t.view(-1).to(device=device, dtype=dtype)
    #     r = r.view(-1).to(device=device, dtype=dtype)

    #     # compute t±ε and corresponding interpolants
    #     t_plus = (t + eps).clamp(0.0, 1.0)
    #     t_minus = (t - eps).clamp(0.0, 1.0)

    #     alpha_plus, sigma_plus, _, _ = self.interpolant(t_plus.view(-1, 1, 1, 1))
    #     alpha_minus, sigma_minus, _, _ = self.interpolant(t_minus.view(-1, 1, 1, 1))

    #     # x_{t±ε} = α * x + σ * z
    #     x_plus = alpha_plus * x + sigma_plus * z
    #     x_minus = alpha_minus * x + sigma_minus * z

    #     # make deep copies of kwargs
    #     kwargs_plus, kwargs_minus = {}, {}
    #     for k, v in model_kwargs.items():
    #         if torch.is_tensor(v) and v.shape[0] == B:
    #             kwargs_plus[k] = v.clone()
    #             kwargs_minus[k] = v.clone()
    #         else:
    #             kwargs_plus[k] = v
    #             kwargs_minus[k] = v

    #     # preserve RNG determinism
    #     if rng_state is None:
    #         rng_state = torch.cuda.get_rng_state()

    #     torch.cuda.set_rng_state(rng_state)
    #     f_plus = model(x_plus, r, t_plus, **kwargs_plus)

    #     torch.cuda.set_rng_state(rng_state)
    #     f_minus = model(x_minus, r, t_minus, **kwargs_minus)

    #     dF_dv_dt = (f_plus - f_minus) / (2 * eps)
    #     return dF_dv_dt


    @torch.no_grad() # by gemini
    def _dde_derivative(self, model, x, z, t, r, model_kwargs, 
                        cls_token, noises_cls, # <-- MODIFIED: Added token inputs
                        rng_state=None):
        """
        Differential Derivation Equation (DDE) approximation for df/dt
        """
        eps = self.differential_epsilon
        B = x.size(0)

        device, dtype = x.device, x.dtype
        t = t.view(-1).to(device=device, dtype=dtype)
        r = r.view(-1).to(device=device, dtype=dtype)

        t_plus = (t + eps).clamp(0.0, 1.0)
        t_minus = (t - eps).clamp(0.0, 1.0)

        alpha_plus, sigma_plus, _, _ = self.interpolant(t_plus.view(-1, 1, 1, 1))
        alpha_minus, sigma_minus, _, _ = self.interpolant(t_minus.view(-1, 1, 1, 1))

        x_plus = alpha_plus * x + sigma_plus * z
        x_minus = alpha_minus * x + sigma_minus * z
        
        # <-- MODIFIED: Calculate noisy cls_token for t+eps and t-eps
        cls_input_plus = None
        cls_input_minus = None
        if cls_token is not None:
            alpha_plus_cls = alpha_plus.squeeze().unsqueeze(-1)
            sigma_plus_cls = sigma_plus.squeeze().unsqueeze(-1)
            alpha_minus_cls = alpha_minus.squeeze().unsqueeze(-1)
            sigma_minus_cls = sigma_minus.squeeze().unsqueeze(-1)
            
            cls_input_plus = alpha_plus_cls * cls_token + sigma_plus_cls * noises_cls
            cls_input_minus = alpha_minus_cls * cls_token + sigma_minus_cls * noises_cls
        # --- End Modification ---

        kwargs_plus, kwargs_minus = {}, {}
        for k, v in model_kwargs.items():
            if torch.is_tensor(v) and v.shape[0] == B:
                kwargs_plus[k] = v.clone()
                kwargs_minus[k] = v.clone()
            else:
                kwargs_plus[k] = v
                kwargs_minus[k] = v

        if rng_state is None:
            rng_state = torch.cuda.get_rng_state()

        torch.cuda.set_rng_state(rng_state)
        # <-- MODIFIED: Pass cls_token and get 3 outputs
        f_plus, _, _ = model(x_plus, r, t_plus, **kwargs_plus, cls_token=cls_input_plus)

        torch.cuda.set_rng_state(rng_state)
        # <-- MODIFIED: Pass cls_token and get 3 outputs
        f_minus, _, _ = model(x_minus, r, t_minus, **kwargs_minus, cls_token=cls_input_minus)

        dF_dv_dt = (f_plus - f_minus) / (2 * eps)
        return dF_dv_dt
    
    def __call__(self, model, images, model_kwargs=None, zs=None, cls_token=None):
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
        noises_cls = torch.randn_like(cls_token) if cls_token is not None else None

        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises #(1-t) * images + t * noise
        
        # <-- MODIFIED: Calculate noisy cls_input for REG
        cls_input = None
        if cls_token is not None:
            alpha_t_cls = alpha_t.squeeze().unsqueeze(-1)
            sigma_t_cls = sigma_t.squeeze().unsqueeze(-1)
            cls_input = alpha_t_cls * cls_token + sigma_t_cls * noises_cls

        # Calculate instantaneous velocity v_t 
        v_t = d_alpha_t * images + d_sigma_t * noises

        # <-- MODIFIED: Calculate cls_target for REG
        cls_target = None
        if cls_token is not None:
            # d_alpha_t_cls = d_alpha_t.squeeze().unsqueeze(-1)
            # d_sigma_t_cls = d_sigma_t.squeeze().unsqueeze(-1)
            # cls_target = d_alpha_t_cls * cls_token + d_sigma_t_cls * noises_cls
            cls_target = d_alpha_t * cls_token + d_sigma_t * noises_cls

        time_diff = (t - r).view(-1, 1, 1, 1)
                
        # u_target = torch.zeros_like(v_t)
        
        # u = model(z_t, r, t, **model_kwargs)
        u, zs_tilde, cls_output = model(z_t, r, t, **model_kwargs, cls_token=cls_input)  # <-- MODIFIED: Get 3 outputs
        
        
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
                cfg_r_batch = torch.cat([cfg_r, cfg_r], dim=0) 
                cfg_t_batch = torch.cat([cfg_t, cfg_t], dim=0) 
                cfg_y_batch = torch.cat([cfg_y, torch.full_like(cfg_y, num_classes)], dim=0)

                cfg_combined_kwargs = cfg_kwargs.copy()
                cfg_combined_kwargs['y'] = cfg_y_batch

                cfg_cls_input_at_t = cls_input[cfg_indices] if cls_input is not None else None
                cfg_cls_input_batch = None
                if cfg_cls_input_at_t is not None:
                    cfg_cls_input_batch = torch.cat([cfg_cls_input_at_t, cfg_cls_input_at_t], dim=0)

                with torch.no_grad():
                    cfg_combined_u_at_t, _, _ = model(
                        cfg_z_t_batch, 
                        cfg_r_batch, 
                        cfg_t_batch, 
                        **cfg_combined_kwargs,
                        cls_token=cfg_cls_input_batch # <-- Pass batched token
                    )
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
                    cls_token[cfg_indices] if cls_token is not None else None,
                    noises_cls[cfg_indices] if noises_cls is not None else None,
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
                    cls_token[no_cfg_indices] if cls_token is not None else None,
                    noises_cls[no_cfg_indices] if noises_cls is not None else None,
                    rng_state=rng_state
                )

                
                no_cfg_u_target = no_cfg_v_t - no_cfg_time_diff * no_cfg_dudt
                u_target[no_cfg_indices] = no_cfg_u_target
        else:
            # No labels or no CFG applicable samples, use standard DDE
            
            # <-- MODIFIED: Pass REG tokens to DDE
            dudt = self._dde_derivative(
                model,
                images,      # x (clean)
                noises,      # z (noise)
                t,           # t
                r,           # r
                model_kwargs,
                cls_token,
                noises_cls,
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

        proj_loss = torch.tensor(0.0, device=device)
        if zs is not None and zs_tilde is not None:
            proj_loss_val = 0.
            bsz = zs[0].shape[0]
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)): # Loop over encoders
                z_norm = F.normalize(z.detach(), dim=-1) # (B, T, D)
                z_tilde_norm = F.normalize(z_tilde, dim=-1) # (B, T, D)
                
                # Negative cosine similarity, averaged over tokens and summed over batch
                neg_cos_sim = -torch.sum(z_norm * z_tilde_norm, dim=-1)
                proj_loss_per_item = torch.mean(neg_cos_sim, dim=1) # (B,)
                proj_loss_val += torch.sum(proj_loss_per_item) # Sum over batch
                
            if len(zs) > 0:
                proj_loss = proj_loss_val / (len(zs) * bsz) # Mean over encoders and batch

        denoising_loss_cls = torch.tensor(0.0, device=device)
        if cls_output is not None and cls_target is not None:
            denoising_loss_cls = mean_flat((cls_output - cls_target.detach()) ** 2)
            
        return loss, loss_mean_ref, proj_loss, denoising_loss_cls