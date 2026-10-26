"""
Perception for the pick-and-trace arm, built on two open-weight models from Hugging Face.

  Object detection : google/owlv2-base-patch16-ensemble   (OWLv2, Apache-2.0)
                     open-vocabulary detector, asked for "a small wooden cube".
                     Each detected cube is then named by the hue of its pixels.
  Path detection   : CIDAS/clipseg-rd64-refined   (CLIPSeg, open weights)
                     text-prompted segmentation, asked for "a hand drawn black curved line";
                     its heatmap chooses which dark stroke on the paper is the drawn path.
                     The stroke is thinned to a 1-pixel centre line and resampled into
                     evenly spaced points from left to right.

Everything here works in camera-image pixels.
"""
import cv2
import numpy as np
import torch
from PIL import Image
from skimage.morphology import skeletonize
from transformers import (CLIPSegForImageSegmentation, CLIPSegProcessor,
                          Owlv2ForObjectDetection, Owlv2Processor)

OWL_ID = "google/owlv2-base-patch16-ensemble"
CLIPSEG_ID = "CIDAS/clipseg-rd64-refined"

_models = {}


def _load(name):
    if name not in _models:
        if name == "owl":
            _models[name] = (Owlv2Processor.from_pretrained(OWL_ID),
                             Owlv2ForObjectDetection.from_pretrained(OWL_ID).eval())
        else:
            _models[name] = (CLIPSegProcessor.from_pretrained(CLIPSEG_ID),
                             CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_ID).eval())
    return _models[name]


# ------------------------------------------------------------------ cubes ----
HUE_RANGES = {  # OpenCV hue 0-179
    "red": [(0, 8), (165, 179)],
    "orange": [(9, 16)],
    "yellow": [(17, 38)],
    "green": [(39, 90)],
    "blue": [(95, 130)],
}


def colour_of(crop_bgr, min_fill=0.0):
    """Name the colour of a cube crop by the dominant hue of its saturated pixels, or None.
    min_fill: the coloured pixels must cover at least this share of the crop - a cube is a
    solid block of colour, a drawn line through the box is not."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1] > 90
    val = hsv[..., 2] > 70
    sel = sat & val
    if sel.sum() < 20 or sel.mean() < min_fill:
        return None
    hue = np.median(hsv[..., 0][sel])
    for name, ranges in HUE_RANGES.items():
        if any(lo <= hue <= hi for lo, hi in ranges):
            return name
    return None


def _owl_boxes(frame_bgr, region, queries, threshold):
    """OWLv2 on one region (x0, y0, x1, y1) of the frame -> [(box in frame pixels, score)]."""
    processor, model = _load("owl")
    ox, oy, ex, ey = region
    crop = frame_bgr[oy:ey, ox:ex]
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    inputs = processor(text=[queries], images=image, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    side = max(crop.shape[:2])  # OWLv2 pads the picture to a square
    res = processor.post_process_grounded_object_detection(
        out, threshold=threshold, target_sizes=torch.tensor([[side, side]]))[0]
    return [((x0 + ox, y0 + oy, x1 + ox, y1 + oy), score)
            for (x0, y0, x1, y1), score in zip(res["boxes"].tolist(), res["scores"].tolist())]


def detect_cubes(frame_bgr, threshold=0.15, paper_mask=None, colour=None, zoom_region=None):
    """Return [{'colour', 'centre', 'box', 'score'}] for every cube found.

    Only the paper is shown to the detector (small objects are found far more reliably when
    they fill more of the model's input). If a target colour is given it is named in the
    prompts ("a yellow cube"), which lifts that cube's score well clear of the background;
    if it is still not found, the zoom_region (e.g. the arm's reach) is searched close up.
    Every box must also be a solid block of one colour.
    """
    h, w = frame_bgr.shape[:2]
    if paper_mask is not None:
        ys, xs = np.nonzero(paper_mask)
        region = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    else:
        region = (0, 0, w, h)
    queries = ["a small wooden cube", "a colored cube"]
    if colour:
        queries += ["a %s cube" % colour, "a %s block" % colour]

    def candidates(boxes):
        found = []
        for (x0, y0, x1, y1), score in boxes:
            x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            bw, bh = x1 - x0, y1 - y0
            if not (20 <= bw <= 90 and 20 <= bh <= 90) or max(bw, bh) > 1.7 * min(bw, bh):
                continue  # not cube-sized
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            if paper_mask is not None and not paper_mask[cy, cx]:
                continue
            name = colour_of(frame_bgr[y0:y1, x0:x1], min_fill=0.3)
            if name is None:
                continue  # grey/black things such as the arm, or a drawn line
            found.append({"colour": name, "centre": (int(cx), int(cy)), "box": (x0, y0, x1, y1), "score": score})
        return found

    cands = candidates(_owl_boxes(frame_bgr, region, queries, threshold))
    if colour and zoom_region is not None and not any(c["colour"] == colour for c in cands):
        cands += candidates(_owl_boxes(frame_bgr, zoom_region, queries, threshold))

    cands.sort(key=lambda c: -c["score"])
    kept = []
    for c in cands:
        if all(np.hypot(c["centre"][0] - k["centre"][0], c["centre"][1] - k["centre"][1]) > 25 for k in kept):
            kept.append(c)
    return kept


# ------------------------------------------------------------------- path ----
def detect_path(frame_bgr, n_points=12, paper_mask=None, exclude_boxes=()):
    """Return (points, mask): n_points (x, y) pixels along the drawn curve, left to right.

    Darkness finds candidate strokes on the paper; CLIPSeg's heatmap picks which stroke
    is the drawn line. The areas in exclude_boxes (the cubes and their shadows) are cut out
    of the strokes; strokes that touch the paper's border (cables, shadows, the arm) and
    small blobs such as a pen dot are ignored.
    """
    processor, model = _load("clipseg")
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(text=["a hand drawn black curved line"], images=image,
                       padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    prob = torch.sigmoid(logits).squeeze().numpy()
    prob = cv2.resize(prob, (frame_bgr.shape[1], frame_bgr.shape[0]))

    gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    local_bg = cv2.medianBlur(gray, 41)
    mask = ((local_bg.astype(int) - gray.astype(int)) > 18).astype(np.uint8)
    for x0, y0, x1, y1 in exclude_boxes:
        # cut the cube (and its shadow) out; a path drawn next to a cube keeps the rest
        cv2.rectangle(mask, (x0 - 10, y0 - 10), (x1 + 10, y1 + 10), 0, -1)
    inner = np.ones(mask.shape, np.uint8) if paper_mask is None else paper_mask.astype(np.uint8)
    inner = cv2.erode(inner, np.ones((25, 25), np.uint8))       # stay off the paper's edge
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    strokes = []
    for i in range(1, n):
        comp = labels == i
        if stats[i, cv2.CC_STAT_AREA] < 40 or (comp & (inner == 0)).any():
            continue    # a speck, or something running off the paper
        if np.hypot(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) < 40:
            continue    # a dot or a smudge (under ~2.5 cm across), not a line
        strokes.append((prob[comp].mean() * np.log(stats[i, cv2.CC_STAT_AREA]), i))
    if not strokes:
        raise RuntimeError("No path found")
    strokes.sort(reverse=True)
    best = strokes[0][1]
    # a faint line can break into pieces: also keep sizeable pieces close to the best one
    near = cv2.dilate((labels == best).astype(np.uint8), np.ones((41, 41), np.uint8))
    keep = [best] + [i for _, i in strokes[1:]
                     if stats[i, cv2.CC_STAT_AREA] >= 0.15 * stats[best, cv2.CC_STAT_AREA]
                     and near[labels == i].any()]
    mask = np.isin(labels, keep).astype(np.uint8)

    skel = skeletonize(mask > 0)
    ys, xs = np.nonzero(skel)
    pix = np.stack([xs, ys], axis=1).astype(float)

    # Walk along the centre line from its left end, so any shape works (curves that turn
    # back, run up the page, or wiggle), not only one point per column.
    pad = np.pad(skel, 1).astype(np.uint8)
    neighbours = sum(np.roll(np.roll(pad, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                     if dy or dx)[1:-1, 1:-1][ys, xs]
    ends = np.nonzero(neighbours == 1)[0]
    candidates = ends if len(ends) else np.arange(len(pix))
    current = candidates[np.argmin(pix[candidates, 0])]          # leftmost end
    visited = np.zeros(len(pix), bool)
    order = []
    while True:
        visited[current] = True
        order.append(current)
        d = np.hypot(*(pix - pix[current]).T)
        d[visited] = np.inf
        nxt = int(np.argmin(d))
        if d[nxt] > 25:          # no more line within 25 px (bridges small gaps in the pen line)
            break
        current = nxt
    line = pix[order]

    # resample to n_points evenly spaced along the line's length
    seg = np.hypot(*np.diff(line, axis=0).T)
    dist = np.concatenate([[0], np.cumsum(seg)])
    targets = np.linspace(0, dist[-1], n_points)
    pts = [(float(np.interp(t, dist, line[:, 0])), float(np.interp(t, dist, line[:, 1]))) for t in targets]
    return pts, mask * 255
