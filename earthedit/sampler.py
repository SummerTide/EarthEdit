"""EarthEdit sampling: RePaint-style SD3 flow sampling inside the change region, then a pixel composite."""
import numpy as np
import torch

from .captions import NEUTRAL
from .data import CLASSES, eval_item, segmap_to_rgb
from .masks import alpha_mask, cell_mask, latent_mask
from .model import condition, encode, load_controlnet


class Sampler:
    def __init__(self, dataset, ckpt, pretrained_model, device='cuda:0'):
        from diffusers import StableDiffusion3Pipeline
        self.dataset, self.device, self.dtype = dataset, torch.device(device), torch.float16
        with torch.inference_mode():
            self.pipe = StableDiffusion3Pipeline.from_pretrained(pretrained_model, torch_dtype=self.dtype).to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
            self.cn = load_controlnet(ckpt, self.dtype, self.device)
        self.prompt_cache = {}
        self.white = None

    def text(self, prompt):
        if prompt not in self.prompt_cache:
            e, _, p, _ = self.pipe.encode_prompt(prompt=prompt, prompt_2=None, prompt_3=None, device=self.device, do_classifier_free_guidance=False)
            self.prompt_cache[prompt] = (e, p)
        return self.prompt_cache[prompt]

    @torch.inference_mode()
    def __call__(self, row, cfg, dilate, feather, steps, seed, cfg_mode='both'):
        """Manifest row -> uint8 image. cfg_mode 'both' drops layout and text in the unconditional branch, 'text' only the text."""
        pipe, dev, dt = self.pipe, self.device, self.dtype
        pre, t, c, prompt = eval_item(self.dataset, row)
        if not c.any():
            return pre
        alpha = alpha_mask(c, dilate, feather)
        gen_pix = torch.from_numpy((alpha > 0.01).astype(np.float32))[None, None].to(dev)
        m_lat = latent_mask(gen_pix)
        m_cell = cell_mask(m_lat)
        pre_t = torch.from_numpy(pre.copy()).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1.0
        x_pre = encode(pipe.vae, pre_t, dev, dt)
        x_prem = encode(pipe.vae, pre_t * (1 - m_cell), dev, dt)
        x_seg = encode(pipe.vae, segmap_to_rgb(t, CLASSES[self.dataset]), dev, dt)
        if self.white is None:
            self.white = encode(pipe.vae, np.full((512, 512, 3), 255, np.uint8), dev, dt)
        m = m_lat.to(dt)
        seg_u = self.white if cfg_mode == 'both' else x_seg
        cond = torch.cat([condition(seg_u, x_prem, m), condition(x_seg, x_prem, m)])
        e_c, p_c = self.text(prompt)
        e_u, p_u = self.text(NEUTRAL)
        emb, pooled = torch.cat([e_u, e_c]), torch.cat([p_u, p_c])
        g = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn(x_pre.shape, generator=g).to(dev, dt)
        scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
        scheduler.set_timesteps(steps, device=dev)
        x = noise.clone()
        for i, ts in enumerate(scheduler.timesteps):
            sigma = scheduler.sigmas[i].to(dt)
            x = m * x + (1 - m) * ((1 - sigma) * x_pre + sigma * noise)  # outside the mask: the source noised to this time
            xin = torch.cat([x, x])
            tt = ts.expand(2)
            blocks = self.cn(hidden_states=xin, timestep=tt, encoder_hidden_states=emb, pooled_projections=pooled,
                             controlnet_cond=cond, conditioning_scale=1.0, return_dict=False)[0]
            v = pipe.transformer(hidden_states=xin, timestep=tt, encoder_hidden_states=emb, pooled_projections=pooled,
                                 block_controlnet_hidden_states=blocks, return_dict=False)[0]
            vu, vc = v.chunk(2)
            x = scheduler.step(vu + cfg * (vc - vu), ts, x, return_dict=False)[0]
        x = m * x + (1 - m) * x_pre
        if not torch.isfinite(x).all():
            raise FloatingPointError(f'nonfinite latent {row["stem"]}')
        dec = pipe.vae.decode(x / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor, return_dict=False)[0]
        img = ((dec[0].float().clamp(-1, 1) + 1) * 127.5).round().permute(1, 2, 0).cpu().numpy()
        a3 = alpha[..., None]
        return np.clip(np.round(a3 * img + (1 - a3) * pre.astype(np.float32)), 0, 255).astype(np.uint8)
