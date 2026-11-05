"""Training pairs: index rows, rare-class weights, masks with random dilation and blobs, flips and rotations."""
import csv, random
from multiprocessing import Pool
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import CLASSES, caption_for, load_layout, load_mask, rgb, segmap_to_rgb
from .masks import dilate, random_blobs


def _index_row(args):
    dataset, r = args
    t, change = load_layout(dataset, r['layout']), load_mask(dataset, r['mask'])
    k = len(CLASSES[dataset])
    hist = np.bincount(t.ravel(), minlength=k)
    if r.get('layout_pre'):
        hist = hist + np.bincount(load_layout(dataset, r['layout_pre']).ravel(), minlength=k)
    return {**r, 'changed_frac': float(change.mean()), 'caption': caption_for(dataset, r, t), 'hist': hist.tolist()}


def load_index(dataset, path, workers=4):
    """Training CSV -> rows with the derived fields changed_frac, caption and hist."""
    rows = list(csv.DictReader(open(path)))
    with Pool(workers) as pool:
        return pool.map(_index_row, [(dataset, r) for r in rows], chunksize=16)


def rarity(index):
    """Images rich in rare classes score higher."""
    hist = np.array([r['hist'] for r in index], dtype=np.float64)
    freq = hist.sum(0) / hist.sum()
    inv = np.minimum((freq.max() / np.maximum(freq, 1e-9)) ** 0.5, 10.0)
    return (hist * inv).sum(1) / np.maximum(hist.sum(1), 1)


def pair_weights(index):
    """Sampling weights: rarity on changed pairs, 0 on unchanged pairs."""
    w = np.where(np.array([r['changed_frac'] for r in index]) > 0, rarity(index), 0.0)
    return w / w.sum()


def augment(arrays, rng=random):
    """The same random flips and 90 degree rotation for all arrays."""
    if rng.random() < 0.5:
        arrays = [a[:, ::-1] for a in arrays]
    if rng.random() < 0.5:
        arrays = [a[::-1] for a in arrays]
    k = rng.randint(0, 3)
    return [np.rot90(a, k) for a in arrays]


def to_tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 127.5 - 1.0


class PairDataset(Dataset):
    """Real change pairs. With probability p_blob, a uniformly drawn pair gets random blobs added to its support."""
    def __init__(self, dataset, index, p_blob=0.3, dilation=(2, 12)):
        self.dataset, self.index, self.p_blob, self.dilation, self.classes = dataset, index, p_blob, tuple(dilation), CLASSES[dataset]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        blob = random.random() < self.p_blob
        if blob:
            i = random.randrange(len(self.index))
        r = self.index[i]
        pre, post = rgb(r['pre']), rgb(r['post'])
        t, change = load_layout(self.dataset, r['layout']), load_mask(self.dataset, r['mask'])
        mask = dilate(change, random.randint(*self.dilation)) if change.any() else np.zeros(change.shape, bool)
        if blob or not mask.any():
            mask = mask | random_blobs(*mask.shape)
        arrays = augment([pre, post, segmap_to_rgb(t, self.classes), mask[..., None].astype(np.uint8) * 255])
        return {'pre': to_tensor(arrays[0]), 'post': to_tensor(arrays[1]), 'seg_post': to_tensor(arrays[2]),
                'mask': torch.from_numpy(np.ascontiguousarray(arrays[3])).permute(2, 0, 1).float() / 255.0,
                'caption': r['caption']}


class LayoutDataset(Dataset):
    """Single-date samples for layout training, from a CSV with image, layout, caption and an optional sampling weight."""
    def __init__(self, dataset, path):
        self.dataset, self.classes, self.rows = dataset, CLASSES[dataset], list(csv.DictReader(open(path)))

    def __len__(self):
        return len(self.rows)

    def weights(self):
        return [float(r.get('weight') or 1.0) for r in self.rows]

    def __getitem__(self, i):
        r = self.rows[i]
        image, seg = augment([rgb(r['image']), segmap_to_rgb(load_layout(self.dataset, r['layout']), self.classes)])
        return {'image': to_tensor(image), 'seg': to_tensor(seg), 'caption': r['caption']}
