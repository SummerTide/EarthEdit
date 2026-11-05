"""EarthEdit sampling. Writes <output_dir>/<stem>.png; finished images are skipped on a rerun.

python sample.py --config configs/second.yaml --checkpoint weights/second --csv pairs.csv --output_dir outputs/second
python sample.py --config configs/second.yaml --checkpoint weights/second --pre pre.png --layout layout.png --mask mask.png --output_dir outputs/one
"""
import argparse, csv, os
from pathlib import Path
from PIL import Image

from earthedit.config import load_config
from earthedit.data import fixed_seed
from earthedit.sampler import Sampler


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--csv', default=None, help='columns stem, pre, layout, mask and an optional caption, see README')
    p.add_argument('--pre', default=None, help='single image mode: pre-change image')
    p.add_argument('--layout', default=None, help='single image mode: target layout')
    p.add_argument('--mask', default=None, help='single image mode: change mask')
    p.add_argument('--caption', default=None, help='single image mode: caption, derived from the layout if omitted')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--pretrained_model', default=None, help='SD3 model id or local path; default from the config')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if bool(a.csv) == bool(a.pre and a.layout and a.mask):
        p.error('give either --csv or all of --pre, --layout, --mask')
    cfg = load_config(a.config)
    ds, s = cfg['dataset'], cfg['sampling']
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.csv))) if a.csv else [{'stem': Path(a.pre).stem, 'pre': a.pre, 'layout': a.layout, 'mask': a.mask, 'caption': a.caption}]
    rows = rows[:a.limit] if a.limit else rows
    rows = [r for r in rows if not (out / f'{r["stem"]}.png').exists()]
    if not rows:
        return
    gen = Sampler(ds, a.checkpoint, a.pretrained_model or cfg['pretrained_model'], a.device)
    for r in rows:
        img = gen(r, s['cfg'], s['dilate'], s['feather'], s['steps'], fixed_seed(ds, r['stem'], s['seed']), s['cfg_mode'])
        dst = out / f'{r["stem"]}.png'
        tmp = dst.with_suffix('.tmp.png')
        Image.fromarray(img).save(tmp)
        os.replace(tmp, dst)
    print(f'{ds}: {len(rows)} images in {out}')


if __name__ == '__main__':
    main()
