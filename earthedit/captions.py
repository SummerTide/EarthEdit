"""Text conditions derived from a label map."""
import numpy as np
from scipy import ndimage

NEUTRAL = 'an aerial remote sensing image'
# Hi-UCD: 'roads' components with inscribed half-width >= GEO_WIDTH px are paved areas, not lane roads.
GEO_NAMES = {'hiucd': {5: 'parking lot', 7: 'construction site'}, 'second': {}}
GEO_WIDTH = 150
DISASTER = {'hurricane': 'hurricane', 'flooding': 'flood', 'fire': 'wildfire', 'wildfire': 'wildfire', 'bushfire': 'wildfire',
            'earthquake': 'earthquake', 'tsunami': 'tsunami', 'volcano': 'volcanic eruption', 'tornado': 'tornado'}


def _with(text, names):
    if names:
        text += ' with ' + (names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1])
    return text


def class_caption(labels, classes, top_k=3):
    """Caption of the base layout stage: the top-k classes by area, class 0 excluded."""
    counts = {i: int((labels == i).sum()) for i in classes if i != 0}
    top = sorted([i for i, c in counts.items() if c > 0], key=counts.get, reverse=True)[:top_k]
    return _with(NEUTRAL, [classes[i]['name'] for i in top])


def layout_caption(labels, classes, dataset, width=GEO_WIDTH, top_k=4):
    """Top-k caption of one label map, with geometry-derived sub-classes (label map only, no image information)."""
    names = {i: classes[i]['name'] for i in classes if i != 0}
    counts = {names[i]: int((labels == i).sum()) for i in names}
    for cid, sub in GEO_NAMES[dataset].items():
        if cid == 5:
            comps, n = ndimage.label(labels == 5)
            if n:
                half = ndimage.distance_transform_edt(labels == 5)
                wide = [k for k in range(1, n + 1) if half[comps == k].max() >= width]
                moved = int(np.isin(comps, wide).sum())
                counts[names[5]] -= moved
                counts[sub] = counts.get(sub, 0) + moved
        else:
            counts[sub] = counts.get(sub, 0) + counts.pop(names[cid])
    top = sorted([k for k, c in counts.items() if c > 0], key=counts.get, reverse=True)[:top_k]
    return _with(NEUTRAL, top)


def disaster_type(stem):
    event = stem.rsplit('_', 1)[0]
    for key, name in DISASTER.items():
        if key in event:
            return name
    raise ValueError(f'unknown disaster in {stem}')


def xbd_caption(labels, classes, stem, when='post'):
    """xBD caption: date, disaster type, building classes present by area."""
    d = disaster_type(stem)
    counts = {classes[i]['name']: int((labels == i).sum()) for i in range(1, len(classes))}
    names = [n for n, c in sorted(counts.items(), key=lambda kv: -kv[1]) if c > 0][:3]
    return _with(f'a {"pre" if when == "pre" else "post"}-disaster satellite image of {"an" if d[0] in "aeiou" else "a"} {d} area', names)
