"""The 33-channel ControlNet (build, save, load) and the condition encoding."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

EXTRA = 17  # masked pre latent (16) + latent mask (1)
WEIGHTS = 'inpaint_controlnet.safetensors'


def widen(controlnet):
    """Replace pos_embed_input.proj (16 in) by a 33-in conv: old weights for the first 16, zeros for the new 17."""
    old = controlnet.pos_embed_input.proj
    new = nn.Conv2d(old.in_channels + EXTRA, old.out_channels, old.kernel_size, old.stride, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_channels].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    controlnet.pos_embed_input.proj = new.to(old.weight.device, old.weight.dtype)
    return controlnet


def build_controlnet(init, dtype=torch.float32):
    """Layout ControlNet checkpoint -> widened ControlNet."""
    from diffusers import SD3ControlNetModel
    return widen(SD3ControlNetModel.from_pretrained(str(init), torch_dtype=dtype))


def save_controlnet(controlnet, path, meta=None):
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    controlnet.save_config(str(path))
    sd = {k: v.detach().contiguous().cpu() for k, v in controlnet.state_dict().items()}
    save_file(sd, str(path / WEIGHTS))
    (path / 'inpaint_meta.json').write_text(json.dumps({'extra_in_channels': EXTRA, **(meta or {})}, indent=1))


def load_controlnet(path, dtype, device):
    from diffusers import SD3ControlNetModel
    from safetensors.torch import load_file
    cn = widen(SD3ControlNetModel.from_config(SD3ControlNetModel.load_config(str(path))))
    cn.load_state_dict(load_file(str(Path(path) / WEIGHTS)))
    return cn.to(device=device, dtype=dtype).eval()


def encode(vae, x, device, dtype, sample=False):
    """Image in [-1, 1] (tensor) or uint8 (HxWx3 array) -> SD3 latent (z - shift) * scale. Training samples, sampling takes the mode."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1.0
    d = vae.encode(x.to(device=device, dtype=dtype)).latent_dist
    z = d.sample() if sample else d.mode()
    return (z - vae.config.shift_factor) * vae.config.scaling_factor


def condition(x_seg, x_prem, m_lat):
    """The 33-channel ControlNet input: layout latent, masked pre latent, latent mask."""
    return torch.cat([x_seg, x_prem, m_lat.to(x_seg.dtype)], 1)


@torch.no_grad()
def embed_captions(pipe, captions, device):
    """Caption -> (prompt, pooled) on CPU, computed once; the text encoders then leave the GPU."""
    cache = {}
    for prompt in sorted(captions):
        e, _, p, _ = pipe.encode_prompt(prompt=prompt, prompt_2=None, prompt_3=None, device=device, do_classifier_free_guidance=False)
        cache[prompt] = (e.cpu(), p.cpu())
    for comp in [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]:
        if comp is not None:
            comp.cpu()
    return cache
