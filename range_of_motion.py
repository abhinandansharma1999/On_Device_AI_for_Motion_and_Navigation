"""
Measure the range of motion of the ESP32 robot arm on a sheet of white paper.

How it works
  1. A top-down webcam looks at the paper. The paper is found and flattened
     (perspective-corrected) so that 1 cm on the paper is PX_PER_CM pixels.
  2. A baseline picture is taken with the arm parked.
  3. The arm is sent to a grid of (x, y) points with PtpCmd(x, y, z). At each
     point the picture is compared with the baseline and the arm tip is taken
     as the changed pixel farthest from the arm base.
  4. The reachable area (convex hull of the tips) is drawn on the paper and
     saved together with a CSV of every measurement.

Usage
  python range_of_motion.py                        # full sweep, arm on COM10
  python range_of_motion.py --camera 2 --z 10      # other camera / hover height
  python range_of_motion.py --image frame.png      # only test paper detection
  python range_of_motion.py --list-cameras

Needs: pip install opencv-python numpy mpremote
"""
import argparse
import csv
import math
import os
import sys
import time

import cv2
import numpy as np

PAPER_SIZES_CM = {"letter": (27.94, 21.59), "a4": (29.7, 21.0)}  # (long, short)
PX_PER_CM = 20

# PtpCmd limits copied from libs/ACB_ARM.py, so unreachable points are skipped
# on the PC instead of being silently ignored by the arm.
X_RANGE = (-19, 19)
Y_RANGE = (0, 19)
Z_RANGE = (0, 27)
SPHERE_CENTRE = (0, 0, 8)
R_MIN, R_MAX = 4, 19

SERVO_PINS = (5, 16, 17, 18)  # chassis, shoulder, elbow, claws

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ---------------------------------------------------------------- camera ----
def list_cameras():
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok, frame = cap.read()
        cap.release()
        if ok:
            path = os.path.join(OUT_DIR, "camera_%d.png" % i)
            cv2.imwrite(path, frame)
            print("camera %d works, %dx%d, saved %s" % (i, frame.shape[1], frame.shape[0], path))


class Camera:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            sys.exit("Cannot open camera %d (try --list-cameras)" % index)
        for _ in range(15):  # let exposure settle
            self.cap.read()

    def grab(self):
        for _ in range(5):  # drop buffered frames so we get the current one
            self.cap.read()
        ok, frame = self.cap.read()
        if not ok:
            sys.exit("Camera read failed")
        return frame

    def close(self):
        self.cap.release()


# ----------------------------------------------------------------- paper ----
def order_corners(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)  # tl tr br bl


def find_paper(frame):
    """Return the 4 paper corners (tl, tr, br, bl) in image pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 150), (180, 70, 255))  # bright and unsaturated
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No white paper found")
    # The robot base can cut a notch in the paper, so use the convex hull.
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 30):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return order_corners(approx)
    return order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))


def paper_homography(corners, paper):
    long_cm, short_cm = PAPER_SIZES_CM[paper]
    top = np.linalg.norm(corners[1] - corners[0])
    left = np.linalg.norm(corners[3] - corners[0])
    w_cm, h_cm = (long_cm, short_cm) if top >= left else (short_cm, long_cm)
    w, h = int(w_cm * PX_PER_CM), int(h_cm * PX_PER_CM)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    return cv2.getPerspectiveTransform(corners, dst), (w, h)


# ------------------------------------------------------------------- arm ----
def reachable(x, y, z):
    if not (X_RANGE[0] <= x <= X_RANGE[1] and Y_RANGE[0] <= y <= Y_RANGE[1]
            and Z_RANGE[0] <= z <= Z_RANGE[1]):
        return False
    if x == 0 and y == 0 and z < 20:
        return False
    d = math.dist((x, y, z), SPHERE_CENTRE)
    return R_MIN <= d <= R_MAX


class Arm:
    def __init__(self, port):
        from mpremote.transport_serial import SerialTransport
        self.t = SerialTransport(port, baudrate=115200)
        self.t.enter_raw_repl()
        self.t.exec("from libs.ACB_ARM import *\nARM_init(%d, %d, %d, %d)\nZero()" % SERVO_PINS)

    def move(self, x, y, z):
        out = self.t.exec("PtpCmd(%s, %s, %s)" % (x, y, z)).decode(errors="replace")
        return not ("Exceeds" in out or "Out of range" in out)

    def park(self):
        self.t.exec("Zero()")

    def close(self):
        try:
            self.park()
        finally:
            self.t.exit_raw_repl()
            self.t.close()


# ------------------------------------------------------------- detection ----
def find_tip(baseline, current, base_px):
    """Most forward arm pixel, or None if the arm is not over the paper.

    The arm is near-black; the paper is bright. A pixel counts as arm if it is
    dark now but was paper-bright in the baseline (so the pen line and the
    parked arm are ignored).
    """
    now = cv2.GaussianBlur(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    before = cv2.GaussianBlur(cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    mask = ((now < 85) & (before > 120)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < 150:
        return None
    ys, xs = np.nonzero(labels == big)
    # Most forward point of the arm (smallest image y = farthest from the base);
    # average x over a thin band so the open jaws give the claw centre.
    top = ys.min()
    if top < 10:  # touching the paper's far edge is a false hit, not the arm
        return None
    band = ys <= top + 15
    return int(xs[band].mean()), int(top)


def px_to_arm_cm(px, base_px):
    """Warped-paper pixel -> arm frame in cm (x right, y forward, origin = base axis)."""
    return (px[0] - base_px[0]) / PX_PER_CM, (base_px[1] - px[1]) / PX_PER_CM


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM10")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", help="test paper detection on this image; the arm is not moved")
    ap.add_argument("--corners", help="paper corners in camera pixels as 'x,y x,y x,y x,y' "
                    "(top-left top-right bottom-right bottom-left); use when the arm hides a corner")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--paper", choices=PAPER_SIZES_CM, default="letter")
    ap.add_argument("--paper-cm", help="measured size as LONGxSHORT in cm, e.g. 27.3x21.6 (overrides --paper)")
    ap.add_argument("--z", type=float, default=8, help="height used for every point (cm)")
    ap.add_argument("--step", type=float, default=3, help="grid spacing (cm)")
    ap.add_argument("--base-y", type=float, default=-1.6,
                    help="arm base axis, in cm below the paper's bottom edge (negative = below)")
    args = ap.parse_args()

    if args.paper_cm:
        long_cm, short_cm = (float(v) for v in args.paper_cm.lower().split("x"))
        PAPER_SIZES_CM["custom"] = (long_cm, short_cm)
        args.paper = "custom"

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.list_cameras:
        return list_cameras()

    cam = None
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit("Cannot read " + args.image)
    else:
        cam = Camera(args.camera)
        frame = cam.grab()

    if args.corners:
        corners = order_corners([[float(v) for v in p.split(",")] for p in args.corners.split()])
    else:
        corners = find_paper(frame)
    H, (w, h) = paper_homography(corners, args.paper)
    flat = lambda img: cv2.warpPerspective(img, H, (w, h))
    base_px = (w / 2, h - args.base_y * PX_PER_CM)  # assumed centred on the paper

    preview = flat(frame)
    cv2.circle(preview, (int(base_px[0]), int(base_px[1])), 6, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(OUT_DIR, "paper_flat.png"), preview)
    marked = frame.copy()
    cv2.polylines(marked, [corners.astype(np.int32)], True, (0, 255, 0), 3)
    cv2.imwrite(os.path.join(OUT_DIR, "paper_detected.png"), marked)
    print("Paper found: %.1f x %.1f cm  (see output/paper_detected.png, paper_flat.png)" % (w / PX_PER_CM, h / PX_PER_CM))
    if args.image:
        return

    arm = Arm(args.port)
    results = []
    try:
        time.sleep(1.5)
        baseline = flat(cam.grab())
        cv2.imwrite(os.path.join(OUT_DIR, "baseline.png"), baseline)

        xs = np.arange(X_RANGE[0], X_RANGE[1] + 0.01, args.step)
        ys = np.arange(Y_RANGE[0], Y_RANGE[1] + 0.01, args.step)
        grid = [(float(x), float(y)) for y in ys for x in xs if reachable(x, y, args.z)]
        print("Sweeping %d reachable grid points at z=%g (this is slow, ~2 s each)" % (len(grid), args.z))

        for i, (x, y) in enumerate(grid, 1):
            try:
                ok = arm.move(x, y, args.z)
            except Exception as e:  # PtpCmd's IK can raise a math error near its limits
                ok = False
                print("[%d/%d] (%g, %g) IK error: %s" % (i, len(grid), x, y, str(e).strip().splitlines()[-1]))
            if not ok:
                print("[%d/%d] (%g, %g) skipped" % (i, len(grid), x, y))
                continue
            time.sleep(0.8)
            tip = find_tip(baseline, flat(cam.grab()), base_px)
            if tip is None:
                print("[%d/%d] (%g, %g) no tip detected" % (i, len(grid), x, y))
                continue
            tx, ty = px_to_arm_cm(tip, base_px)
            results.append((x, y, args.z, tip[0], tip[1], tx, ty))
            print("[%d/%d] cmd (%g, %g) -> tip at (%.1f, %.1f) cm" % (i, len(grid), x, y, tx, ty))
    finally:
        arm.close()
        if cam:
            cam.close()

    if len(results) < 3:
        sys.exit("Too few detections to build a range")

    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cmd_x", "cmd_y", "cmd_z", "tip_px_x", "tip_px_y", "tip_x_cm", "tip_y_cm"])
        wr.writerows(results)

    pts = np.array([[r[3], r[4]] for r in results], dtype=np.int32)
    hull = cv2.convexHull(pts)
    area = cv2.contourArea(hull) / PX_PER_CM ** 2
    overlay = baseline.copy()
    cv2.polylines(overlay, [hull], True, (0, 200, 0), 3)
    for p in pts:
        cv2.circle(overlay, tuple(int(v) for v in p), 5, (255, 0, 0), -1)
    cv2.circle(overlay, (int(base_px[0]), int(base_px[1])), 8, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(OUT_DIR, "range_of_motion.png"), overlay)

    tx = [r[5] for r in results]
    ty = [r[6] for r in results]
    print("\nRange of motion on the paper (arm frame, cm):")
    print("  x: %.1f to %.1f    y: %.1f to %.1f" % (min(tx), max(tx), min(ty), max(ty)))
    print("  max reach: %.1f cm    reachable area: %.0f cm^2" % (max(math.hypot(a, b) for a, b in zip(tx, ty)), area))
    print("  saved to", OUT_DIR)


if __name__ == "__main__":
    main()
