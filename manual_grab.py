"""
Drive the arm by hand: go to a ground coordinate, then try claw angles until the cube is held.

    python manual_grab.py                 # interactive
    python manual_grab.py -8 10.4         # go to (-8, 10.4) first, then interactive

Ground coordinates are in cm: origin = centre of the arm's turntable, +y ahead, +x right.

Commands at the prompt:
    go X Y [Z]     move the closed claw tip to (X, Y) at height Z above the paper (default 1 = grab height)
    z Z            change only the height at the same X, Y (e.g. z 0, z -1 to go lower)
    up [Z]         raise to height Z at the same X, Y (default 8)
    servo C S E    set chassis, shoulder, elbow servo angles directly (slowly), for calibration
    log X Y H      save a measurement: where the closed claw tip is now, in cm from the
                   turntable centre (X right, Y forward) and H above the paper.
                   Saved with the current servo angles to calib_points.csv
    where          print the current target and servo angles
    claw A         set the claw servo to A degrees (0-180, NOT limited to 90-180)
    open           claw to the open angle
    grab           claw to the closed angle (change it with: close A)
    close A        remember A as the closed angle and apply it
    park           all servos to 90
    q              quit (parks the arm)
"""
import csv
import os
import sys

import arm_ik

CALIB_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_points.csv")


def main():
    arm = arm_ik.ArmIK(sys.argv[3] if len(sys.argv) > 3 else "COM13")
    closed = arm_ik.CLAW_CLOSED
    pos = None

    def claw(angle):
        arm.claw(angle)             # slow, eased; any angle 0-180
        print("  claw ->", angle)

    def go(x, y, z):
        nonlocal pos
        try:
            angles = arm.goto(x, y, z)
        except ValueError as e:
            print("  ", e)
            return
        pos = (x, y, z)
        print("  (%.1f, %.1f) z=%.1f -> chassis %.1f  shoulder %.1f  elbow %.1f" % ((x, y, z) + angles))

    try:
        claw(arm_ik.CLAW_OPEN)
        if len(sys.argv) >= 3:
            x, y = float(sys.argv[1]), float(sys.argv[2])
            go(x, y, arm_ik.Z_TRAVEL)
            go(x, y, arm_ik.Z_GRAB)
        print(__doc__.split("Commands at the prompt:")[1])
        while True:
            try:
                parts = input("arm> ").split()
            except EOFError:
                break
            if not parts:
                continue
            try:
                cmd, args = parts[0].lower(), [float(v) for v in parts[1:]]
            except ValueError:
                print("  numbers only after the command")
                continue
            if cmd == "go" and len(args) >= 2:
                go(args[0], args[1], args[2] if len(args) > 2 else arm_ik.Z_GRAB)
            elif cmd == "z" and args:
                if pos is None:
                    print("  use go first")
                else:
                    go(pos[0], pos[1], args[0])
            elif cmd == "servo" and len(args) == 3:
                # raw servo angles, for calibration: chassis shoulder elbow
                try:
                    arm.move_servos(args[0], args[1], args[2])
                except ValueError as e:
                    print("  ", e)
                    continue
                pos = None
                print("  servos -> chassis %.0f  shoulder %.0f  elbow %.0f" % tuple(args))
            elif cmd == "log" and len(args) == 3:
                ch, sh, el, cl = arm.angles()
                new = not os.path.exists(CALIB_CSV)
                with open(CALIB_CSV, "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["chassis", "shoulder", "elbow", "claw", "tip_x_cm", "tip_y_cm", "tip_height_cm"])
                    w.writerow([ch, sh, el, cl] + args)
                print("  saved: servos %.0f/%.0f/%.0f  tip (%.1f, %.1f) height %.1f  -> calib_points.csv"
                      % (ch, sh, el, *args))
            elif cmd == "where":
                print("  target", pos, " servo angles (chassis, shoulder, elbow, claw):", arm.angles())
            elif cmd == "up":
                if pos is None:
                    print("  use go first")
                else:
                    go(pos[0], pos[1], args[0] if args else arm_ik.Z_TRAVEL)
            elif cmd == "claw" and args:
                claw(int(args[0]))
            elif cmd == "open":
                claw(arm_ik.CLAW_OPEN)
            elif cmd == "close" and args:
                closed = int(args[0])
                claw(closed)
            elif cmd == "grab":
                claw(closed)
            elif cmd == "park":
                arm.park()
                pos = None
            elif cmd in ("q", "quit", "exit"):
                break
            else:
                print("  unknown command; see the list above")
        print("Closed angle used:", closed, "- put it in CLAW_CLOSED in arm_ik.py once it holds the cube")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
