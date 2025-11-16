import torch

# @torch.no_grad()
# def meanflow_sampler(
#     model, 
#     latents,
#     cls_latents, 
#     y=None, 
#     cfg_scale=1.0,
#     cls_cfg_scale=1.0,
#     num_steps=1, 
#     **kwargs
# ):
#     """
#     MeanFlow sampler supporting both single-step and multi-step generation
    
#     Based on Eq.(12): z_r = z_t - (t-r)u(z_t, r, t)
#     For single-step: z_0 = z_1 - u(z_1, 0, 1)
#     For multi-step: iteratively apply the Eq.(12) with intermediate steps
#     """
#     batch_size = latents.shape[0]
#     device = latents.device
    
#     # Prepare for CFG if needed
#     do_cfg = y is not None and cfg_scale > 1.0
#     if do_cfg:
#         if hasattr(model, 'module'):  # DDP
#             num_classes = model.module.num_classes
#         else:
#             num_classes = model.num_classes
#         null_y = torch.full_like(y, num_classes)
    
#     if num_steps == 1:
#         r = torch.zeros(batch_size, device=device)
#         t = torch.ones(batch_size, device=device)
        
#         if do_cfg:
#             z_combined = torch.cat([latents, latents], dim=0)
#             r_combined = torch.cat([r, r], dim=0)
#             t_combined = torch.cat([t, t], dim=0)
#             y_combined = torch.cat([y, null_y], dim=0)
            
#             cls_combined = torch.cat([cls_latents, cls_latents], dim=0)

#             u_combined = model(z_combined, r_combined, t_combined, y=y_combined)
#             u_cond, u_uncond = u_combined.chunk(2, dim=0)
            
#             u = u_uncond + cfg_scale * (u_cond - u_uncond)
#         else:
#             u = model(latents, r, t, y=y)
        
#         # x_0 = x_1 - u(x_1, 0, 1)
#         x0 = latents - u
        
#     else:
#         z = latents
        
#         time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
        
#         for i in range(num_steps):
#             t_cur = time_steps[i]
#             t_next = time_steps[i + 1]
            
#             t = torch.full((batch_size,), t_cur, device=device)
#             r = torch.full((batch_size,), t_next, device=device)
            
#             if do_cfg:
#                 z_combined = torch.cat([z, z], dim=0)
#                 r_combined = torch.cat([r, r], dim=0)
#                 t_combined = torch.cat([t, t], dim=0)
#                 y_combined = torch.cat([y, null_y], dim=0)
                
#                 u_combined = model(z_combined, r_combined, t_combined, y=y_combined)
#                 u_cond, u_uncond = u_combined.chunk(2, dim=0)
                
#                 # Apply CFG
#                 u = u_uncond + cfg_scale * (u_cond - u_uncond)
#             else:
#                 u = model(z, r, t, y=y)
            
#             # Update z: z_r = z_t - (t-r)*u(z_t, r, t)
#             z = z - (t_cur - t_next) * u
        
#         x0 = z
    
#     return x0

import torch

@torch.no_grad()
def meanflow_sampler(
    model, 
    latents, 
    cls_latents,               # <-- BARU: Membutuhkan noise untuk class token
    y=None, 
    cfg_scale=1.0,
    cls_cfg_scale=1.0,         # <-- BARU: Skala CFG untuk class token
    num_steps=1, 
    **kwargs
):
    """
    MeanFlow sampler yang dimodifikasi untuk mendukung REG (image + token denoising).
    
    Berdasarkan Eq.(12): z_r = z_t - (t-r)u(z_t, r, t)
    """
    batch_size = latents.shape[0]
    device = latents.device
    
    # Siapkan untuk CFG jika diperlukan
    do_cfg = y is not None and cfg_scale > 1.0
    if do_cfg:
        if hasattr(model, 'module'):  # DDP
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes
        null_y = torch.full_like(y, num_classes)
    
    if num_steps == 1:
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)
        
        if do_cfg:
            # Gandakan input untuk CFG
            z_combined = torch.cat([latents, latents], dim=0)
            r_combined = torch.cat([r, r], dim=0)
            t_combined = torch.cat([t, t], dim=0)
            y_combined = torch.cat([y, null_y], dim=0)
            
            # <-- MODIFIKASI: Gandakan juga cls_token
            cls_combined = torch.cat([cls_latents, cls_latents], dim=0)
            
            # <-- MODIFIKASI: Panggil model dan unpack 3 output
            u_img_combined, _, u_cls_combined = model(
                z_combined, r_combined, t_combined, y=y_combined, cls_token=cls_combined
            )
            
            # Pisahkan conditional dan unconditional
            u_img_cond, u_img_uncond = u_img_combined.chunk(2, dim=0)
            u_cls_cond, u_cls_uncond = u_cls_combined.chunk(2, dim=0)
            
            # Terapkan CFG ke output gambar
            u_img = u_img_uncond + cfg_scale * (u_img_cond - u_img_uncond)
            # Terapkan CFG ke output token (meskipun kita tidak menggunakannya di sini)
            # u_cls = u_cls_uncond + cls_cfg_scale * (u_cls_cond - u_cls_uncond)

        else: # Tanpa CFG
            # <-- MODIFIKASI: Panggil model dan unpack 3 output
            u_img, _, u_cls = model(latents, r, t, y=y, cls_token=cls_latents)
        
        # x_0 = x_1 - u(x_1, 0, 1)
        x0 = latents - u_img # <-- MODIFIKASI: Gunakan u_img
        
    else: # Multi-step
        z = latents
        z_cls = cls_latents  # <-- BARU: Lacak laten token
        
        time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]
            
            t = torch.full((batch_size,), t_cur, device=device)
            r = torch.full((batch_size,), t_next, device=device)
            
            if do_cfg:
                # Gandakan input untuk CFG
                z_combined = torch.cat([z, z], dim=0)
                r_combined = torch.cat([r, r], dim=0)
                t_combined = torch.cat([t, t], dim=0)
                y_combined = torch.cat([y, null_y], dim=0)
                
                # <-- MODIFIKASI: Gandakan juga cls_token
                cls_combined = torch.cat([z_cls, z_cls], dim=0)
                
                # <-- MODIFIKASI: Panggil model dan unpack 3 output
                u_img_combined, _, u_cls_combined = model(
                    z_combined, r_combined, t_combined, y=y_combined, cls_token=cls_combined
                )
                
                # Pisahkan conditional dan unconditional
                u_img_cond, u_img_uncond = u_img_combined.chunk(2, dim=0)
                u_cls_cond, u_cls_uncond = u_cls_combined.chunk(2, dim=0)
                
                # Terapkan CFG ke output gambar
                u_img = u_img_uncond + cfg_scale * (u_img_cond - u_img_uncond)
                # Terapkan CFG ke output token
                u_cls = u_cls_uncond + cls_cfg_scale * (u_cls_cond - u_cls_uncond)
                
            else: # Tanpa CFG
                # <-- MODIFIKASI: Panggil model dan unpack 3 output
                u_img, _, u_cls = model(z, r, t, y=y, cls_token=z_cls)
            
            # Update z (image): z_r = z_t - (t-r)*u(z_t, r, t)
            z = z - (t_cur - t_next) * u_img
            # <-- BARU: Update z_cls (token)
            z_cls = z_cls - (t_cur - t_next) * u_cls
            
        x0 = z
    
    return x0
