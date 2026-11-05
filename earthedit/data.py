"""Dataset adapters: class tables and label loading. Formats are documented in the README."""
import hashlib
import numpy as np
from PIL import Image

from .captions import layout_caption, xbd_caption

SECOND_CLASSES = {
    0: {'name': 'no-change', 'color': (255, 255, 255)},
    1: {'name': 'non-vegetated ground surface', 'color': (128, 128, 128)},
    2: {'name': 'low vegetation', 'color': (0, 128, 0)},
    3: {'name': 'tree', 'color': (0, 255, 0)},
    4: {'name': 'water', 'color': (0, 0, 255)},
    5: {'name': 'buildings', 'color': (128, 0, 0)},
    6: {'name': 'playground', 'color': (255, 0, 0)},
}
HIUCD_CLASSES = {
    0: {'name': 'unlabeled', 'color': (255, 255, 255)},
    1: {'name': 'water', 'color': (0, 153, 255)},
    2: {'name': 'grass', 'color': (202, 255, 122)},
    3: {'name': 'buildings', 'color': (230, 0, 0)},
    4: {'name': 'green house', 'color': (230, 0, 255)},
    5: {'name': 'roads', 'color': (255, 230, 0)},
    6: {'name': 'bridge', 'color': (255, 181, 197)},
    7: {'name': 'others', 'color': (0, 255, 230)},
    8: {'name': 'bare land', 'color': (175, 122, 255)},
    9: {'name': 'woodland', 'color': (26, 255, 0)},
}
XBD_CLASSES = {
    0: {'name': 'background', 'color': (0, 0, 0)},
    1: {'name': 'intact buildings', 'color': (128, 0, 0)},
    2: {'name': 'slightly damaged buildings', 'color': (255, 255, 0)},
    3: {'name': 'severely damaged buildings', 'color': (255, 128, 0)},
    4: {'name': 'destroyed buildings', 'color': (128, 0, 128)},
}
CLASSES = {'second': SECOND_CLASSES, 'hiucd': HIUCD_CLASSES, 'xbd': XBD_CLASSES}


def rgb(p):
    return np.asarray(Image.open(p).convert('RGB'))


def segmap_to_rgb(labels, classes):
    """(H, W) class ids -> (H, W, 3) uint8 colour rendering."""
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, info in classes.items():
        out[labels == i] = info['color']
    return out


def second_ids(path):
    """SECOND RGB label -> class ids; an unknown colour is an error."""
    a = rgb(path)
    labels, known = np.zeros(a.shape[:2], np.int64), np.zeros(a.shape[:2], bool)
    for i, info in SECOND_CLASSES.items():
        m = (a == info['color']).all(-1)
        labels[m] = i
        known |= m
    if not known.all():
        raise ValueError(f'Unexpected palette in {path}')
    return labels


def fixed_seed(dataset, stem, seed=42):
    """Per-image sampling seed."""
    return int.from_bytes(hashlib.sha256(f'{seed}:{dataset}:{stem}'.encode()).digest()[:8], 'big') % (2 ** 31)


def load_layout(dataset, path):
    """Layout PNG -> (H, W) class ids. SECOND: palette RGB. Hi-UCD: 3-channel mask (post class in G) or class ids. xBD: class ids."""
    if dataset == 'second':
        return second_ids(path)
    a = np.asarray(Image.open(path))
    return (a[:, :, 1] if a.ndim == 3 else a).astype(np.int64)


def load_mask(dataset, path):
    """Change mask PNG -> (H, W) bool. White is changed; a 3-channel Hi-UCD mask is changed where B == 2."""
    im = Image.open(path)
    if dataset == 'hiucd' and len(im.getbands()) == 3:
        return np.asarray(im)[:, :, 2] == 2
    return np.asarray(im.convert('L')) > 127


def caption_for(dataset, row, labels):
    """The row's caption if given, else derived from the layout. xBD takes the disaster type from the stem."""
    if row.get('caption'):
        return row['caption']
    if dataset == 'xbd':
        return xbd_caption(labels, XBD_CLASSES, row['stem'])
    return layout_caption(labels, CLASSES[dataset], dataset)


def eval_item(dataset, row):
    """Row with pre, layout, mask -> pre (uint8), layout, change mask C (bool), caption."""
    t = load_layout(dataset, row['layout'])
    return rgb(row['pre']), t, load_mask(dataset, row['mask']), caption_for(dataset, row, t)
