"""
Final demo: pick up a cube by colour, carry it along the drawn path from left to right,
put it down at the end of the path, and return to the default pose. All moves are slow
and eased (see arm_ik.py). A live camera window shows what the demo sees and does.

    python final_demo.py                    # red cube, full run
    python final_demo.py --dry-run          # look and plan only; the arm only turns aside and back
    python final_demo.py --colour green
    python final_demo.py --no-view          # without the live window

Steps
  1. Turn the arm aside so the camera sees the whole paper.
  2. OWLv2 finds the cubes, CLIPSeg finds the drawn path (vision.py).
  3. Camera pixels -> cm on the paper -> ground coordinates around the arm centre.
  4. Pick the cube, go to the path start, follow the path, place the cube at the path end.
  5. Back to the default pose (all servos at 90).
In the live window: Esc or closing it stops the demo after the current move (the arm parks).
"""
import argparse
import json
import os
import textwrap
import time

import cv2
import numpy as np

import arm_ik
import paper_finder
import quick_calibrate as qc
import range_of_motion as rom
import vision
from live_view import LiveView

PORT = "COM13"
CAMERA = 1
PAPER_CORNERS = [(104, 16), (557, 10), (559, 366), (106, 362)]   # camera pixels: TL TR BR BL
PAPER_CM = (27.3, 21.6)                                          # long side, short side
PAPER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_corners.json")
if os.path.exists(PAPER_FILE):                                   # saved by set_paper.py
    with open(PAPER_FILE) as _f:
        PAPER_CORNERS = [tuple(p) for p in json.load(_f)["corners"]]
# where the arm centre (turntable axis) is, in cm: right of the paper's centre line,
# and above the paper's bottom edge (negative = below it). quick_calibrate.py measures it.
ARM_CENTRE_ON_PAPER = (0.0, -3.5)
if "turntable_on_paper" in arm_ik.CALIBRATION:
    # measured from the paper edge on the robot's right, which is the camera picture's right edge
    _t = arm_ik.CALIBRATION["turntable_on_paper"]
    ARM_CENTRE_ON_PAPER = (PAPER_CM[0] / 2 - _t["from_robot_right_edge"], -_t["behind_bottom_edge"])

LOOK_POSE = arm_ik.LOOK      # arm turned to the robot's right, off the paper
Z_CARRY = 5.0                 # height of the claw tip while following the path

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


class Stop(Exception):
    """The demo cannot or should not go on; the message says why."""


# ---------------------------------------------------------- coordinates ----
def _paper():
    rom.PAPER_SIZES_CM["custom"] = PAPER_CM
    H, (w, h) = rom.paper_homography(rom.order_corners(PAPER_CORNERS), "custom")
    return H, np.linalg.inv(H), w, h


def camera_to_ground(points_px):
    """Camera pixels -> ground cm (origin = arm centre, +x right, +y ahead)."""
    H, _, w, h = _paper()
    flat = cv2.perspectiveTransform(np.asarray(points_px, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
    px = rom.PX_PER_CM
    cx, cy = ARM_CENTRE_ON_PAPER
    return [(float(x / px - w / px / 2 - cx), float((h - y) / px - cy)) for x, y in flat]


def ground_to_camera(points_cm):
    """Ground cm -> camera pixels (for drawing)."""
    _, Hinv, w, h = _paper()
    px = rom.PX_PER_CM
    cx, cy = ARM_CENTRE_ON_PAPER
    flat = np.array([((gx + cx) * px + w / 2, h - (gy + cy) * px) for gx, gy in points_cm], np.float32)
    return [tuple(p) for p in cv2.perspectiveTransform(flat.reshape(-1, 1, 2), Hinv).reshape(-1, 2)]


def reachable(x, y, z):
    try:
        arm_ik.solve(x, y, z)
        return True
    except ValueError:
        return False


def line_reachable(p0, p1):
    """Every point of the straight move p0 -> p1 (x, y, z in cm) can be reached."""
    n = max(1, int(np.ceil(np.linalg.norm(np.subtract(p1, p0)) / arm_ik.LINE_STEP_CM)))
    return all(reachable(*(np.add(p0, np.subtract(p1, p0) * i / n))) for i in range(n + 1))


def reach_zones():
    """Outlines (camera pixels) of where the arm can grab, and where it can follow a path."""
    H, Hinv, w, h = _paper()
    step = 4
    gys, gxs = np.mgrid[0:h:step, 0:w:step]
    ground = camera_to_ground(cv2.perspectiveTransform(
        np.stack([gxs.ravel(), gys.ravel()], 1).astype(np.float32).reshape(-1, 1, 2), Hinv).reshape(-1, 2))
    zones = []
    for z, colour, label in ((arm_ik.Z_GRAB, (80, 220, 80), "grab zone"), (Z_CARRY, (0, 200, 255), "path zone")):
        mask = np.array([reachable(x, y, z) for x, y in ground], np.uint8).reshape(gxs.shape) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 4:
                continue
            flat = (c.reshape(-1, 2) * step).astype(np.float32)
            cam = cv2.perspectiveTransform(flat.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
            zones.append(([tuple(p) for p in cam], colour, label))
    return zones


# ----------------------------------------------------------------- demo ----
def run(arm, view, args):
    def step(*lines):
        print(lines[0])
        for line in lines[1:]:
            print("   " + line)
        view.show(status=list(lines) + (["DRY RUN - the arm will not pick or trace"] if args.dry_run else []))

    def check_stop():
        if view.stop_requested:
            raise Stop("Stopped by you (Esc / window closed).")

    step("0. Checking that every joint moves...", "keep your hands out of the camera view")
    failed = qc.check_joints(arm, view.snapshot)
    if failed:
        raise Stop("The %s did not move when told to, so the arm would miss. Stopped before it could "
                   "strain. " % failed + qc.joint_help(failed).replace("\n", " "))
    check_stop()

    step("1. Looking at the paper...", "arm turns aside so the camera sees everything")
    arm.move_servos(*LOOK_POSE)
    time.sleep(0.3)
    frame = view.snapshot()
    check_stop()
    arm.park()

    lined_up, off = paper_finder.outline_matches(frame, PAPER_CORNERS)
    if not lined_up:
        cv2.imwrite(os.path.join(OUT, "final_plan.png"), view.render(frame))
        raise Stop("The camera or the paper has moved since the paper outline was saved (the %s edge%s no "
                   "longer line up - see the white outline). Run  python set_paper.py  then try again."
                   % (" and ".join(off), "s" if len(off) > 1 else ""))
    paper = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(paper, np.array(PAPER_CORNERS, np.int32), 1)
    paper = paper.astype(bool)
    bright = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[paper] > 140
    if bright.mean() < 0.8:
        cv2.imwrite(os.path.join(OUT, "final_plan.png"), frame)
        raise Stop("The camera's view of the paper is blocked (only %d%% looks like white paper). "
                   "Clear the view and run again." % (100 * bright.mean()))

    step("2. Finding cubes (OWLv2) and the path (CLIPSeg)...", "about 10 seconds")
    reach_pts = np.array([p for poly, _, _ in reach_zones() for p in poly], np.float32)
    x0, y0 = np.maximum(reach_pts.min(axis=0) - 15, 0).astype(int)
    x1, y1 = np.minimum(reach_pts.max(axis=0) + 15, (frame.shape[1], frame.shape[0])).astype(int)
    cubes = vision.detect_cubes(frame, paper_mask=paper, colour=args.colour, zoom_region=(x0, y0, x1, y1))
    for c in cubes:
        gx, gy = camera_to_ground([c["centre"]])[0]
        c["ground"] = (gx, gy)
        c["label"] = "%s (%.1f, %.1f) cm" % (c["colour"], gx, gy)
    view.show(cubes=cubes)
    check_stop()
    match = [c for c in cubes if c["colour"] == args.colour]
    if not match:
        cv2.imwrite(os.path.join(OUT, "final_plan.png"), view.render(frame))
        seen = sorted({c["colour"] for c in cubes})
        raise Stop("No %s cube on the paper. I can see: %s. %s"
                   % (args.colour, ", ".join(seen) or "no cubes",
                      "Run with --colour %s, or put the %s cube on the paper." % (seen[0], args.colour)
                      if seen else "Put a cube on the paper, inside the grab zone."))
    cube = max(match, key=lambda c: c["score"])
    cube["role"] = "target"
    path_px, _ = vision.detect_path(frame, args.points, paper, [c["box"] for c in cubes])
    path_xy = camera_to_ground(path_px)
    start, end = path_xy[0], path_xy[-1]
    cube_xy = cube["ground"]

    # check every position before anything moves
    path_ok = [reachable(x, y, Z_CARRY) for x, y in path_xy]
    problems = []
    if not (reachable(*cube_xy, arm_ik.Z_GRAB) and reachable(*cube_xy, arm_ik.Z_TRAVEL)):
        problems.append("the %s cube is outside the grab zone - move it closer to the arm" % args.colour)
        cube["label"] += "  OUT OF REACH"
    else:
        cube["label"] = "TARGET " + cube["label"]
    if not (reachable(*end, arm_ik.Z_GRAB) and reachable(*end, arm_ik.Z_TRAVEL)):
        problems.append("the path end is outside the grab zone (the cube is put down there)")
    bad = [i for i, ok in enumerate(path_ok) if not ok]
    if bad:
        problems.append("path points %s are outside the path zone" % ", ".join(map(str, bad)))
    if not problems:
        # the straight moves between the steps: down to the cube and up, across to the path
        # start at travel height, down to carry height, and down to put the cube down
        zt, zg = arm_ik.Z_TRAVEL, arm_ik.Z_GRAB
        moves = [((*cube_xy, zt), (*cube_xy, zg)), ((*cube_xy, zt), (*start, zt)),
                 ((*start, zt), (*start, Z_CARRY)), ((*end, Z_CARRY), (*end, zg)), ((*end, zg), (*end, zt))]
        if not all(line_reachable(a, b) for a, b in moves):
            problems.append("a move between the cube and the path passes out of reach - "
                            "put the cube closer to the path")
    view.show(cubes=cubes, path=path_px, path_ok=path_ok)
    step("Plan: pick the %s cube at (%.1f, %.1f) cm, trace %d path points, put it down at the end"
         % (args.colour, cube_xy[0], cube_xy[1], len(path_xy)),
         "problems: " + "; ".join(problems) if problems else "everything is within reach")
    cv2.imwrite(os.path.join(OUT, "final_plan.png"), view.render(frame))
    print("   %s cube at (%.1f, %.1f) cm; path: %d points from (%.1f, %.1f) to (%.1f, %.1f) cm"
          % (args.colour, cube_xy[0], cube_xy[1], len(path_xy), start[0], start[1], end[0], end[1]))
    print("   plan picture: output/final_plan.png")
    if problems:
        raise Stop("Cannot do it: " + "; ".join(problems) + ".")
    if args.dry_run:
        return "Dry run OK: everything is reachable. Run without --dry-run to move the arm."
    check_stop()

    step("3. Picking up the %s cube..." % args.colour, "open, lower, close, lift")
    view.show(target=(ground_to_camera([cube_xy])[0], "grab here"))
    arm.pick(*cube_xy)
    cube["role"] = "carried"
    view.show(cubes=cubes)
    check_stop()

    step("4. Going to the path start...", "up, across, down - in straight lines")
    view.show(target=(path_px[0], "path start"))
    arm.travel(start[0], start[1], Z_CARRY)
    check_stop()

    step("5. Tracing the path left to right...", "the traced part turns cyan")

    def started(dense, ms_each):
        view.show(progress=(time.time(), ms_each, ground_to_camera(dense)))

    try:
        arm.follow(path_xy, Z_CARRY, on_start=started, should_stop=lambda: view.stop_requested)
    except arm_ik.Stopped:
        raise Stop("Stopped by you while tracing.")
    view.show(progress=None)

    step("6. Placing the cube at the path end...")
    view.show(target=(path_px[-1], "put down here"))
    arm.place(*end)
    view.show(target=None, placed=(path_px[-1], args.colour))
    return "Done: the %s cube was carried along the path and put down at its end." % args.colour


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colour", default="red")
    ap.add_argument("--dry-run", action="store_true", help="plan only, do not pick or trace")
    ap.add_argument("--points", type=int, default=12, help="points sampled along the path")
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--no-view", action="store_true", help="do not open the live camera window")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.dry_run and "chassis" not in arm_ik.CALIBRATION:
        raise SystemExit("The arm is not fully calibrated yet (no turntable calibration in arm_calibration.json),\n"
                         "so it would miss the cube. Run  python quick_calibrate.py  first, or use --dry-run.")

    view = LiveView(CAMERA, window=not args.no_view)
    view.show(paper=PAPER_CORNERS, zones=reach_zones(), status=["Starting..."])
    arm = None
    finished = False
    message = "Stopped."
    try:
        arm = arm_ik.ArmIK(args.port)
        message = run(arm, view, args)
        finished = True
    except Stop as s:
        message = str(s)
        finished = True
    finally:
        if arm is not None:
            view.show(target=None, progress=None, status=["Parking the arm (slowly)..."])
            arm.close()                    # parks slowly (rises first if it is low)
        print(message)
        if finished and view.gui:
            view.show(status=textwrap.wrap(message, 70) + ["Arm parked. Press Esc or close this window to exit."])
            view.wait_until_closed()
        view.close()


if __name__ == "__main__":
    main()
