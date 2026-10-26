"""
Quick arm calibration: 4 poses, 3 numbers each, about 4 minutes.

    python quick_calibrate.py            # full calibration
    python quick_calibrate.py --check    # only the final check of a saved calibration

Sideways distances (X) are measured from the paper edge on the ROBOT'S RIGHT-hand side,
which is your LEFT when you stand facing the robot.

1. Joint check (keep your hands out of the camera view): the turntable, shoulder and elbow
   each move 15 degrees and back, and the overhead camera confirms the arm really moved.
   If a joint does not move, the script stops before asking you anything.
2. Where the turntable centre is on the paper (two ruler numbers, asked once).
3. Four poses, claw closed (the 4th is turned to the robot's left). At each pose measure
   the CLOSED claw tip with a ruler and type three numbers:
       X  how far the tip is from the paper edge on the robot's right (cm)
       Y  how far the tip is in front of the paper's BOTTOM edge (cm)
       H  how high above the paper the tip is (cm)
   e.g.  15 8 17.5     (s = skip this pose, q = stop; the arm parks slowly)
   The arm rests at 90 between poses, so no pose is held longer than needed.
4. The arm model (reach, height, turning) is fitted. It is saved to arm_calibration.json
   (used by arm_ik.py, manual_grab.py and final_demo.py) only if it matches your
   measurements closely. If one straight-ahead pose's X disagrees with the others, it is
   left out of the direction fit (and you are told).
5. Check: the tip goes to 3 cm above the paper, 8 cm in front of the turntable and 4 cm to
   the robot's left. Press Enter if that is right, or type the measured X Y H to refine.
"""
import csv
import datetime
import json
import math
import os
import sys
import time

import cv2
import numpy as np

import arm_ik
import range_of_motion as rom

CAMERA = 1
REST = (90, 90, 90)                  # chassis, shoulder, elbow
# (chassis, shoulder, elbow). Chosen in simulation: whatever the horn offsets and elbow
# linkage, the tip stays at least 3 cm clear of paper and turntable, the shoulder never
# goes past 110, and the poses tell the possible elbow linkages apart. The last one is
# turned to the robot's left, to measure how the turntable angle maps to direction.
POSES = [(90, 90, 90), (90, 110, 90), (90, 100, 70), (125, 110, 90)]
CHECK = (-4.0, 8.0, 3.0)             # ground x (+ = robot's right), y, tip height for the check
GOOD_FIT_CM = 0.7
GOOD_TURN_DEG = 3.0
MOVE_SEEN = 5000                     # changed pixels that count as "the arm moved" (10 deg = 9k-26k)
POINTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_points.csv")
X_EDGE = "the paper edge on the robot's RIGHT (your left when facing the robot)"


# ---------------------------------------------------------------- helpers ----
def changed_pixels(a, b):
    d = cv2.absdiff(cv2.GaussianBlur(a, (5, 5), 0), cv2.GaussianBlur(b, (5, 5), 0)).max(axis=2)
    return int((d > 40).sum())


def joint_moves(arm, snap, name, pose, rest_pose=REST):
    """Nudge one joint and back. True if the camera saw the arm go there AND come back (a hand
    moving through the view does not do that). snap() returns a frame, or None if there is
    no camera, in which case you are asked."""
    if snap() is None:
        arm.move_servos(*pose)
        ok = input("   Did the %s move? (y/n): " % name).strip().lower().startswith("y")
        arm.move_servos(*rest_pose)
        return ok
    for attempt in range(3):
        before = snap()
        arm.move_servos(*pose)
        time.sleep(0.4)
        there = changed_pixels(before, snap())
        arm.move_servos(*rest_pose)
        time.sleep(0.4)
        back = changed_pixels(before, snap())
        if there >= MOVE_SEEN and back < there / 3:
            return True
        if there < MOVE_SEEN and back < MOVE_SEEN:
            return False                          # view was still, and the arm did not move
        print("   something else is moving in the camera view (hands?) - trying again")
    return input("   Camera unsure. Did the %s move? (y/n): " % name).strip().lower().startswith("y")


JOINT_NUDGES = (("turntable", (105, 90, 90)), ("shoulder", (90, 105, 90)), ("elbow", (90, 90, 105)))


def joint_help(name):
    pin = arm_ik.SERVO_PINS[[n for n, _ in JOINT_NUDGES].index(name)]
    return ("Check: is the %s servo plugged into header %s (GPIO%d), fully pushed in, the right way\n"
            "round (orange = signal, red = 5V, brown = GND)? If it is on another header, change\n"
            "SERVO_PINS in arm_ik.py. Does it hum or feel warm (jammed - unplug it)? Is the servo\n"
            "power on / battery charged? The camera must see the arm, too. Then run this again."
            % (name, arm_ik.HEADER.get(pin, "?"), pin))


def check_joints(arm, snap):
    """Every joint must visibly move before the arm is trusted. Returns the failed joint or None."""
    arm.move_servos(*REST)
    time.sleep(0.4)
    for name, pose in JOINT_NUDGES:
        if not joint_moves(arm, snap, name, pose):
            return name
        print("   %s moves - ok" % name)
    return None


class Eyes:
    """The overhead camera, or the user's eyes if the camera cannot be opened."""

    def __init__(self):
        cap = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        self.cam = rom.Camera(CAMERA) if ok else None
        if not ok:
            print("(camera %d not available - I will ask you instead)" % CAMERA)

    def snap(self):
        return None if self.cam is None else self.cam.grab()

    def close(self):
        if self.cam is not None:
            self.cam.close()


def ask_numbers(prompt, count, default=None):
    """Return a tuple of `count` numbers, `default` on Enter, or 'skip' / 'quit'."""
    while True:
        text = input(prompt).strip().lower()
        if text == "" and default is not None:
            return default
        if text in ("s", "skip"):
            return "skip"
        if text in ("q", "quit"):
            return "quit"
        try:
            values = tuple(float(v) for v in text.replace(",", " ").split())
            if len(values) == count:
                return values
        except ValueError:
            pass
        print("   type %d numbers separated by spaces (s = skip, q = quit)" % count)


def record(chassis, shoulder, elbow, x, y, h):
    new = not os.path.exists(POINTS_CSV)
    with open(POINTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "chassis", "shoulder", "elbow", "tip_x_from_robot_right_edge_cm",
                        "tip_y_from_bottom_edge_cm", "tip_height_cm"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), chassis, shoulder, elbow, x, y, h])


# ----------------------------------------------------------------- fitting ----
def fit_turning(rows):
    """rows: (chassis servo angle, measured heading deg, + = robot's right).
    heading = (straight - chassis) * scale  ->  (straight, scale, rms deg)."""
    A = np.array([[1.0, -c] for c, _ in rows])
    b = np.array([h for _, h in rows])
    (p0, scale), *_ = np.linalg.lstsq(A, b, rcond=None)
    rms = float(np.sqrt(np.mean((A @ [p0, scale] - b) ** 2)))
    return float(p0 / scale), float(scale), rms


def fit_turning_robust(rows):
    """Like fit_turning, but if one reading at a turntable angle that other readings share
    disagrees, leave it out. Returns (straight, scale, rms, index left out or None)."""
    straight, scale, rms = fit_turning(rows)
    if rms <= GOOD_TURN_DEG:
        return straight, scale, rms, None
    best = (straight, scale, rms, None)
    angles = [c for c, _ in rows]
    for i, (c, _) in enumerate(rows):
        if angles.count(c) < 3:          # keep at least two readings at that angle
            continue
        s, k, r = fit_turning(rows[:i] + rows[i + 1:])
        if r < best[2]:
            best = (s, k, r, i)
    return best


def fit_and_maybe_save(measured, turntable):
    """measured: [(chassis, shoulder, elbow, X, Y, H)]: X from the paper edge on the robot's
    right, Y from the bottom edge, H above the paper. turntable: (X, distance behind the
    bottom edge) of the turntable centre."""
    tx, ty = turntable
    planar, turning = [], []
    for ch, sh, el, x, y, h in measured:
        dx, dy = tx - x, y + ty               # tip relative to the turntable: + dx = robot's right
        planar.append((sh, el, math.hypot(dx, dy), h))
        turning.append((ch, math.degrees(math.atan2(dx, dy))))
    # different poses measured at (nearly) the same place: a joint did not move or a typo
    for i in range(len(measured)):
        for j in range(i + 1, len(measured)):
            if measured[i][:3] != measured[j][:3] and math.dist(measured[i][3:], measured[j][3:]) < 1.0:
                print("Poses %d and %d were measured at the same place although the joints were at\n"
                      "different angles - a joint probably did not move, or a number was mistyped.\n"
                      "Nothing saved." % (i + 1, j + 1))
                return None
    if len({c for c, _ in turning}) < 2:
        print("Need the turned pose (pose 4) to measure the turntable. Nothing saved.")
        return None

    rms, best = arm_ik.fit_model(planar)[0]
    best = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in best.items()}
    straight, scale, turn_rms, left_out = fit_turning_robust(turning)
    print("\nModel vs. your measurements:")
    print("   turntable shoulder elbow   reach, height measured -> model     direction measured -> model")
    for i, ((ch, sh, el, *_), (sh2, el2, r, h), (_, head)) in enumerate(zip(measured, planar, turning)):
        mr, mh = arm_ik.tip_position(sh, el, best)
        note = "   <- X left out of the direction fit" if i == left_out else ""
        print("   %5d %8d %6d       (%5.1f, %5.1f) -> (%5.1f, %5.1f)        %+6.1f -> %+6.1f deg%s"
              % (ch, sh, el, r, h, mr, mh, head, (straight - ch) * scale, note))
    print("   reach/height error %.2f cm, direction error %.1f deg" % (rms, turn_rms))
    print("   elbow type %s, shoulder horn %+.1f deg, elbow horn %+.1f deg, elbow-to-tip %.1f cm"
          % (best["kind"], best["shoulder_offset"], best["elbow_offset"], best["reach"]))
    print("   turntable points straight ahead at %.1f deg, turns %.2f deg per servo degree" % (straight, scale))
    if left_out is not None:
        print("   Note: pose %d's X (%.1f cm) disagrees with the other poses at the same turntable angle,\n"
              "   so it was left out of the direction fit (it still counts for reach and height)."
              % (left_out + 1, measured[left_out][3]))
    problems = []
    if rms > GOOD_FIT_CM:
        problems.append("reach/height is %.1f cm off" % rms)
    if turn_rms > GOOD_TURN_DEG:
        problems.append("direction is %.1f deg off" % turn_rms)
    if not (0.7 <= scale <= 1.3 and 60 <= straight <= 120):
        problems.append("the turntable numbers look wrong (did it turn for pose 4? X from %s?)" % X_EDGE)
    if problems:
        print("NOT saved (%s); any earlier calibration is kept.\n"
              "Re-measure the closed claw tip (X from %s, Y from the bottom edge, H above the paper)\n"
              "and run again." % ("; ".join(problems), X_EDGE))
        return None
    with open(arm_ik.CALIBRATION_FILE, "w") as f:
        json.dump({"model": best,
                   "chassis": {"straight": straight, "scale": scale},
                   "turntable_on_paper": {"from_robot_right_edge": tx, "behind_bottom_edge": ty},
                   "rms_cm": float(rms), "turn_rms_deg": turn_rms,
                   "x_left_out_of_direction_fit": left_out,
                   "date": datetime.datetime.now().isoformat(timespec="seconds"),
                   "measured_chassis_shoulder_elbow_x_y_h": measured}, f, indent=2)
    arm_ik.MODEL.update(best)
    arm_ik.CHASSIS_STRAIGHT, arm_ik.CHASSIS_SCALE = straight, scale
    print("Saved to arm_calibration.json")
    return rms


def check_pose(arm, turntable, measured):
    """Hold the tip over a known spot; on request refine the fit with what you measure there."""
    x, y, z = CHECK
    print("\nCheck: tip to 3 cm above the paper, 8 cm in front of the turntable, 4 cm to the robot's left")
    print("   expected: X = %.1f cm from %s,\n             Y = %.1f cm from the bottom edge, H = %.1f cm"
          % (turntable[0] - x, X_EDGE, y - turntable[1], z))
    try:
        arm.goto(x, y, z + 4)
        arm.line_to(x, y, z)
    except ValueError as e:
        print("   cannot reach the check point:", e)
        return
    ans = ask_numbers("   Enter if that is right, or type the measured  X Y H : ", 3, ())
    if ans not in ((), "skip", "quit"):
        ch, sh, el = arm.angles()[:3]
        measured.append((round(ch, 1), round(sh, 1), round(el, 1)) + ans)
        record(ch, sh, el, *ans)
        fit_and_maybe_save(measured, turntable)


# ------------------------------------------------------------------- main ----
def main():
    if "--check" in sys.argv:
        if "turntable_on_paper" not in arm_ik.CALIBRATION:
            raise SystemExit("No saved calibration to check yet - run  python quick_calibrate.py")
        t = arm_ik.CALIBRATION["turntable_on_paper"]
        turntable = (t["from_robot_right_edge"], t["behind_bottom_edge"])
        measured = [tuple(m) for m in arm_ik.CALIBRATION["measured_chassis_shoulder_elbow_x_y_h"]]
        arm = arm_ik.ArmIK()
        try:
            arm.grab()
            check_pose(arm, turntable, measured)
        finally:
            arm.close()
        return

    arm = arm_ik.ArmIK()
    eyes = Eyes()
    measured = []
    try:
        input("Joint check: take your hands out of the camera view, then press Enter...")
        failed = check_joints(arm, eyes.snap)
        if failed:
            print("\nThe %s did NOT move when told to turn 15 degrees. Stopping before it can strain.\n"
                  % failed + joint_help(failed))
            return

        print("\nSideways distances (X) are measured from %s." % X_EDGE)
        print("Where is the turntable centre (the axle it turns on)?")
        tx = ask_numbers("   X: distance from that edge (cm): ", 1)
        if tx in ("skip", "quit"):
            return
        ty = ask_numbers("   distance behind the paper's BOTTOM edge (cm) [Enter = 3.5]: ", 1, (3.5,))
        if ty in ("skip", "quit"):
            return
        turntable = (tx[0], ty[0])

        print("\nClosing the claw - measure the CLOSED claw tip.")
        arm.grab()
        for i, (ch, sh, el) in enumerate(POSES, 1):
            before = eyes.snap()
            print("\nPose %d of %d: turntable %d, shoulder %d, elbow %d - moving slowly..." % (i, len(POSES), ch, sh, el))
            arm.move_servos(ch, sh, el)
            time.sleep(0.4)
            if before is not None and (ch, sh, el) != REST and changed_pixels(before, eyes.snap()) < MOVE_SEEN:
                print("The arm did not move to pose %d. Stopping - check the servo plugs and power." % i)
                return
            ans = ask_numbers("   tip  X (from robot's right edge)  Y (from bottom edge)  H (above paper), cm: ", 3)
            if ans == "quit":
                return
            if ans != "skip":
                measured.append((ch, sh, el) + ans)
                record(ch, sh, el, *ans)
            arm.move_servos(*REST)
            time.sleep(0.4)
        if len(measured) < 4:
            print("Need all 4 poses; nothing saved.")
            return
        if fit_and_maybe_save(measured, turntable) is None:
            return
        check_pose(arm, turntable, measured)
        print("\nCalibration finished. Next: python final_demo.py --dry-run --colour yellow")
    finally:
        eyes.close()
        arm.close()           # parks slowly (rises first if it is low)


if __name__ == "__main__":
    main()
