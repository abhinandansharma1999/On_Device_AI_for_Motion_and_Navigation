"""
Find the sheet of paper in a camera picture, every run, so a nudged camera or paper does
not silently shift every coordinate.

Paper is nearly colourless where it is lit, and slightly tinted but dim where a box wall
shades it; cardboard is strongly coloured, the robot is dark, and a light wooden table is
tinted but bright. The outline is the four-cornered shape that best fits the paper region;
it is accepted only if no edge runs off the picture and its proportions match the sheet's.
"""
import cv2
import numpy as np


class PaperNotFound(Exception):
    """The message says what is wrong and what to do."""


def paper_mask(frame):
    """255 where the picture looks like paper: lit (colourless) or shaded (slightly tinted, dim)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    lit = (s < 30) & (v > 100)
    shaded = (s < 65) & (v > 90) & (v < 165)
    return (lit | shaded).astype(np.uint8) * 255


def outline_matches(frame, corners, inset=6, need=0.6):
    """Does the paper still sit where this outline says? Near each corner (away from the robot
    base in the middle of the bottom edge), points just inside each edge should look like paper
    and points just outside mostly not. One edge partly off can be a cube or a cable lying
    there; two edges off, or one far off, means the camera or the paper has moved.
    Returns (ok, names of the edges that are off)."""
    mask = paper_mask(frame) > 0
    h, w = mask.shape
    c = np.array(corners, np.float32)
    centre = c.mean(axis=0)
    off, far_off = [], False
    near_corners = np.concatenate([np.linspace(0.04, 0.2, 9), np.linspace(0.8, 0.96, 9)])
    for name, (a, b) in zip(("top", "right", "bottom", "left"), ((0, 1), (1, 2), (2, 3), (3, 0))):
        inside = outside = total = 0
        for t in near_corners:
            p = c[a] + (c[b] - c[a]) * t
            n = (centre - p) / (np.linalg.norm(centre - p) + 1e-6)
            pi, po = (p + n * inset).astype(int), (p - n * inset).astype(int)
            if not (0 <= po[0] < w and 0 <= po[1] < h and 0 <= pi[0] < w and 0 <= pi[1] < h):
                continue
            total += 1
            inside += mask[pi[1], pi[0]]
            outside += mask[po[1], po[0]]
        if total == 0:
            off.append(name)
            far_off = True                 # the edge is outside the picture altogether
        elif inside / total < need or outside / total > need:
            off.append(name)
            far_off |= inside / total < 0.3 or outside / total > 0.85
    return not (len(off) >= 2 or far_off), off


def find_paper(frame, size_cm=(27.3, 21.6), tolerance=0.15):
    """Return the sheet's corners [TL, TR, BR, BL] in camera pixels (as seen in the picture)."""
    h, w = frame.shape[:2]
    paper = paper_mask(frame)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(paper)
    if n < 2:
        raise PaperNotFound("I cannot see the white paper in the camera picture.")
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < 0.2 * h * w:
        raise PaperNotFound("The paper covers too little of the camera picture - is the camera pointing at it?")
    ys, xs = np.nonzero(labels == big)
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))

    # an edge that runs along the picture's border is cut off
    pts = hull.reshape(-1, 2)
    cut = []
    for name, on_border in (("top", pts[:, 1] <= 1), ("bottom", pts[:, 1] >= h - 2),
                            ("left", pts[:, 0] <= 1), ("right", pts[:, 0] >= w - 2)):
        border = pts[on_border]
        if len(border) >= 2 and np.ptp(border[:, 0 if name in ("top", "bottom") else 1]) > 20:
            cut.append(name)
    if cut:
        raise PaperNotFound("The %s edge of the paper is outside the camera picture. Point the camera so "
                            "the whole sheet is visible, then run again." % " and ".join(cut))

    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.08, 36):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        raise PaperNotFound("The paper's outline does not look like a rectangle in the picture "
                            "(something lying across an edge?).")
    s, d = quad.sum(axis=1), np.diff(quad, axis=1).ravel()
    tl, tr, br, bl = quad[np.argmin(s)], quad[np.argmin(d)], quad[np.argmax(s)], quad[np.argmax(d)]
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    ratio, expected = width / height, size_cm[0] / size_cm[1]
    if abs(ratio / expected - 1) > tolerance:
        raise PaperNotFound("The paper outline found has proportions %.2f, but a %.1f x %.1f cm sheet should "
                            "be about %.2f - part of it may be hidden or out of view."
                            % (ratio, size_cm[0], size_cm[1], expected))
    return [tuple(float(v) for v in p) for p in (tl, tr, br, bl)]


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        try:
            print(path, "->", [tuple(int(round(v)) for v in p) for p in find_paper(img)])
        except PaperNotFound as e:
            print(path, "->", e)
