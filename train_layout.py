"""Train the layout ControlNet the generator starts from, on one GPU: frozen SD3, single-date images and layouts, flow matching.

--stage base    starts from the SD3 transformer
--stage refine  continues from --init_controlnet
Each stage takes its settings from the layout_base or layout_refine section of the config.

python train_layout.py --stage base --config configs/second.yaml --csv second_dates.csv --output_dir runs/second_base
python train_layout.py --stage refine --config configs/second.yaml --csv second_dates_refine.csv \
    --init_controlnet runs/second_base/checkpoint-5000 --output_dir runs/second_refine
"""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from diffusers import SD3ControlNetModel, StableDiffusion3Pipeline
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler

from earthedit.captions import NEUTRAL
from earthedit.config import load_config
from earthedit.dataset import LayoutDataset
from earthedit.model import embed_captions, encode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['base', 'refine'], required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--csv', required=True, help='columns image, layout, caption and an optional weight, see README')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--init_controlnet', default=None, help='required for --stage refine')
    p.add_argument('--pretrained_model', default=None, help='SD3 model id or local path; default from the config')
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--grad_accum', type=int, default=None)
    p.add_argument('--max_train_steps', type=int, default=None)
    p.add_argument('--checkpointing_steps', type=int, default=None)
    p.add_argument('--mixed_precision', choices=['bf16', 'no'], default='bf16')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.stage == 'refine' and not a.init_controlnet:
        p.error('--stage refine needs --init_controlnet')
    cfg = load_config(a.config)
    dataset = cfg['dataset']
    if f'layout_{a.stage}' not in cfg:
        p.error(f'{a.config} has no layout_{a.stage} section')
    c = cfg[f'layout_{a.stage}']
    for k in ('batch_size', 'grad_accum', 'max_train_steps', 'checkpointing_steps'):
        if getattr(a, k) is not None:
            c[k] = getattr(a, k)
    torch.manual_seed(c['seed']); random.seed(c['seed']); np.random.seed(c['seed'])
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    wd = torch.bfloat16 if a.mixed_precision == 'bf16' else torch.float32

    data = LayoutDataset(dataset, a.csv)
    weights = data.weights()
    loader = DataLoader(data, batch_size=c['batch_size'], num_workers=c['num_workers'], drop_last=True,
                        sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True))

    pipe = StableDiffusion3Pipeline.from_pretrained(a.pretrained_model or cfg['pretrained_model'], torch_dtype=torch.float16)
    transformer, vae = pipe.transformer, pipe.vae
    if a.init_controlnet:
        controlnet = SD3ControlNetModel.from_pretrained(a.init_controlnet, torch_dtype=torch.float32)
    else:
        controlnet = SD3ControlNetModel.from_transformer(transformer)
    controlnet = controlnet.to(dev).train()
    transformer.requires_grad_(False); vae.requires_grad_(False)
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=c['learning_rate'], weight_decay=1e-2)
    lr_scheduler = get_scheduler('constant_with_warmup', optimizer=optimizer, num_warmup_steps=c['lr_warmup_steps'],
                                 num_training_steps=c['max_train_steps'])
    transformer.to(dev, dtype=wd); vae.to(dev, dtype=wd); pipe.to(dev)
    cache = embed_captions(pipe, {NEUTRAL} | {r['caption'] for r in data.rows}, dev)
    del pipe
    torch.cuda.empty_cache()

    log = (out / 'train_log.jsonl').open('a')
    step, micro, t0 = 0, 0, time.time()
    while step < c['max_train_steps']:
        for batch in loader:
            B = batch['image'].shape[0]
            with torch.no_grad():
                seg = batch['seg'].clone()
                seg[torch.rand(B) < c['layout_dropout']] = 1.0  # all white = empty layout
                x0 = encode(vae, batch['image'], dev, wd, sample=True)
                c_in = encode(vae, seg, dev, wd, sample=True)
                caps = [NEUTRAL if random.random() < c['text_dropout'] else x for x in batch['caption']]
                emb = torch.cat([cache[x][0] for x in caps]).to(dev, dtype=wd)
                pooled = torch.cat([cache[x][1] for x in caps]).to(dev, dtype=wd)
                t = torch.rand(B, device=dev, dtype=wd) if c['timesteps'] == 'uniform' else torch.sigmoid(torch.randn(B, device=dev))
                noise = torch.randn_like(x0)
                s = t.view(-1, 1, 1, 1)
                hidden = ((1 - s) * x0 + s * noise).to(wd)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.mixed_precision == 'bf16'):
                blocks = controlnet(hidden_states=hidden, timestep=t * 1000, encoder_hidden_states=emb, pooled_projections=pooled,
                                    controlnet_cond=c_in, return_dict=False)[0]
            blocks = [b.to(wd) for b in blocks]
            v = transformer(hidden_states=hidden, timestep=t * 1000, encoder_hidden_states=emb, pooled_projections=pooled,
                            block_controlnet_hidden_states=blocks, return_dict=False)[0].float()
            loss = (v - (noise.float() - x0.float())).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss at step {step}')
            (loss / c['grad_accum']).backward()
            micro += 1
            if micro % c['grad_accum']:
                continue
            optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
            step += 1
            if step % 25 == 0 or step == 1:
                rec = {'step': step, 'loss': float(loss), 'lr': lr_scheduler.get_last_lr()[0], 'elapsed': time.time() - t0}
                log.write(json.dumps(rec) + '\n'); log.flush()
                print(rec, flush=True)
            if step % c['checkpointing_steps'] == 0 or step == c['max_train_steps']:
                controlnet.save_pretrained(str(out / f'checkpoint-{step}'))
            if step >= c['max_train_steps']:
                break


if __name__ == '__main__':
    main()
