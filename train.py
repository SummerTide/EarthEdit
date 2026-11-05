"""Train the context-aware generator on one GPU: frozen SD3, ControlNet widened to 33 channels, weighted rectified-flow loss.

python train.py --config configs/second.yaml --index second_pairs.csv \
    --init_controlnet layout_controlnet/second --output_dir runs/second
"""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler

from earthedit.captions import NEUTRAL
from earthedit.config import load_config
from earthedit.dataset import PairDataset, load_index, pair_weights
from earthedit.loss import weighted_flow_loss
from earthedit.masks import cell_mask, latent_mask
from earthedit.model import build_controlnet, condition, embed_captions, encode, save_controlnet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--index', required=True, help='training pairs CSV, see README')
    p.add_argument('--init_controlnet', required=True, help='layout ControlNet checkpoint the generator starts from')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--pretrained_model', default=None, help='SD3 model id or local path; default from the config')
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--grad_accum', type=int, default=None)
    p.add_argument('--max_train_steps', type=int, default=None)
    p.add_argument('--checkpointing_steps', type=int, default=None)
    p.add_argument('--mixed_precision', choices=['bf16', 'no'], default='bf16')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    cfg = load_config(a.config)
    c, dataset = cfg['generator'], cfg['dataset']
    for k in ('batch_size', 'grad_accum', 'max_train_steps', 'checkpointing_steps'):
        if getattr(a, k) is not None:
            c[k] = getattr(a, k)
    torch.manual_seed(c['seed']); random.seed(c['seed']); np.random.seed(c['seed'])
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    wd = torch.bfloat16 if a.mixed_precision == 'bf16' else torch.float32

    index = load_index(dataset, a.index, c['num_workers'])
    data = PairDataset(dataset, index, c['p_blob'], c['dilation'])
    weights = pair_weights(index)
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(c['seed']))
    loader = DataLoader(data, batch_size=c['batch_size'], sampler=sampler, num_workers=c['num_workers'], drop_last=True,
                        persistent_workers=c['num_workers'] > 0)

    pipe = StableDiffusion3Pipeline.from_pretrained(a.pretrained_model or cfg['pretrained_model'], torch_dtype=torch.float16)
    transformer, vae = pipe.transformer, pipe.vae
    controlnet = build_controlnet(a.init_controlnet, torch.float32).to(dev).train()
    transformer.requires_grad_(False); vae.requires_grad_(False)
    proj = list(controlnet.pos_embed_input.proj.parameters())
    proj_ids = {id(x) for x in proj}
    body = [x for x in controlnet.parameters() if id(x) not in proj_ids]
    optimizer = torch.optim.AdamW([{'params': body, 'lr': c['learning_rate']}, {'params': proj, 'lr': c['input_lr']}], weight_decay=1e-2)
    lr_scheduler = get_scheduler('constant_with_warmup', optimizer=optimizer, num_warmup_steps=c['lr_warmup_steps'],
                                 num_training_steps=c['max_train_steps'])
    transformer.to(dev, dtype=wd); vae.to(dev, dtype=wd); pipe.to(dev)
    cache = embed_captions(pipe, {NEUTRAL} | {r['caption'] for r in index}, dev)
    del pipe
    torch.cuda.empty_cache()

    log = (out / 'train_log.jsonl').open('a')
    step, micro, t0 = 0, 0, time.time()
    while step < c['max_train_steps']:
        for batch in loader:
            B = batch['pre'].shape[0]
            with torch.no_grad():
                m_lat = latent_mask(batch['mask'].to(dev))
                m_cell = cell_mask(m_lat)
                x0 = encode(vae, batch['post'], dev, wd, sample=True)
                x_prem = encode(vae, batch['pre'].to(dev) * (1 - m_cell), dev, wd, sample=True)
                seg = batch['seg_post'].clone()
                seg[torch.rand(B) < c['layout_dropout']] = 1.0  # all white = empty layout
                x_seg = encode(vae, seg, dev, wd, sample=True)
                cond = condition(x_seg, x_prem, m_lat)
                caps = [NEUTRAL if random.random() < c['text_dropout'] else x for x in batch['caption']]
                emb = torch.cat([cache[x][0] for x in caps]).to(dev, dtype=wd)
                pooled = torch.cat([cache[x][1] for x in caps]).to(dev, dtype=wd)
                t = torch.sigmoid(torch.randn(B, device=dev))  # logit-normal
                noise = torch.randn_like(x0)
                s = t.view(-1, 1, 1, 1)
                hidden = ((1 - s) * x0 + s * noise).to(wd)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.mixed_precision == 'bf16'):
                blocks = controlnet(hidden_states=hidden, timestep=t * 1000, encoder_hidden_states=emb, pooled_projections=pooled,
                                    controlnet_cond=cond.to(wd), return_dict=False)[0]
            blocks = [b.to(wd) for b in blocks]
            v = transformer(hidden_states=hidden, timestep=t * 1000, encoder_hidden_states=emb, pooled_projections=pooled,
                            block_controlnet_hidden_states=blocks, return_dict=False)[0].float()
            loss = weighted_flow_loss(v, noise, x0, m_lat, c['outside_weight'])
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
                save_controlnet(controlnet, out / f'checkpoint-{step}', {'step': step, 'dataset': dataset})
            if step >= c['max_train_steps']:
                break


if __name__ == '__main__':
    main()
