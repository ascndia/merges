import torch
import os
from meanflow_sampler import meanflow_sampler
from torchvision.utils import make_grid
from PIL import Image

class AbstractHandler:
    def __init__(self):
        pass
    def handle(self, *args, **kwargs):
        raise NotImplementedError

class CheckpointHandler(AbstractHandler):
    def __init__(self, checkpoint_dir, model, optimizer, ema, args, interval):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.ema = ema
        self.args = args
        self.interval = interval
    
    def handle(self, step):
        if step % self.interval == 0 and step > 0 or step >= self.args.max_train_steps or step == 1:
            checkpoint = {
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "opt": self.optimizer.state_dict(),
                "args": self.args,
                "steps": step,
            }
            checkpoint_path = f"{self.checkpoint_dir}/{step:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

            # remove first check point 
            if step == 1:
                try:
                    os.remove(checkpoint_path)
                except Exception as e:
                    pass

    def save_final(self, step):
        checkpoint = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.optimizer.state_dict(),
            "args": self.args,
            "steps": step,
        }
        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_final.pt"
        torch.save(checkpoint, checkpoint_path)

class MeanFlowSampler:
    def __init__(self, batch_size, num_classes, latent_size, cfg_scale=1.0):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.cfg_scale = cfg_scale
    
    @torch.no_grad()
    def sample(self, model, num_steps):
        device = next(model.parameters()).device
        z = torch.randn(self.batch_size, model.in_channels, self.latent_size, self.latent_size, device=device)
        y = torch.randint(0, self.num_classes, (self.batch_size,), device=device)
        samples = meanflow_sampler(
            model=model,
            latents=z,
            y=y,
            cfg_scale=self.cfg_scale,
            num_steps=num_steps
        )
        return samples

class SampleHandler(AbstractHandler):
    def __init__(self, sample_dir, model, sampler,  vae=None, interval = 1000, nfe_list=None):
        super().__init__()
        self.sample_dir = sample_dir
        self.model = model
        self.sampler = sampler
        self.vae = vae
        self.interval = interval
        self.nfe_list = nfe_list or [1]  # default to single step if not provided
        self.latents_scale = torch.tensor([0.18125, 0.18125, 0.18125, 0.18125]).view(1, 4, 1, 1).to(next(model.parameters()).device)
        self.latents_bias = torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1).to(next(model.parameters()).device)
    
    @torch.no_grad()
    def decode(self, latents):
        decoded = self.vae.decode((latents - self.latents_bias) / self.latents_scale).sample
        decoded = (decoded + 1) / 2
        decoded = torch.clamp(decoded, 0, 1)
        return decoded
    
    def handle(self, step):
        if step % self.interval == 0 or step == 1:
            for nfe in self.nfe_list:
                samples = self.sampler.sample(self.model, nfe)
                if self.vae is not None:
                    decoded = self.decode(samples)
                else:
                    decoded = samples
                grid = make_grid(decoded, nrow=4, normalize=False)  # assuming batch_size=32, 4x8 grid
                grid = grid.permute(1, 2, 0).cpu().numpy()
                grid = (grid * 255).astype('uint8')
                img = Image.fromarray(grid)
                sample_path = f"{self.sample_dir}/samples_{step}_nfe_{nfe}.png"
                img.save(sample_path)