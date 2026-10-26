"""
Move the robot arm to a point on the ground, given in centimetres.

    from arm_ik import ArmIK
    arm = ArmIK("COM13")
    arm.goto(5, 12)          # 5 cm right of the arm centre, 12 cm in front of it
    arm.grab()               # close the claw
    arm.lift()               # raise it
    arm.release()            # open the claw
    arm.park()

Ground coordinates: origin = centre of the arm's turntable, +y straight ahead, +x to the right, cm.
The inverse kinematics is done here on the PC; only finished servo angles go to the ESP32
(a small eased-motion helper is uploaded to the board; every move is slow and smooth).

Run `python arm_ik.py X Y [Z]` to go to a point, `python arm_ik.py park` to return slowly
to the default pose (all servos at 90), or `python arm_ik.py test` to watch each joint turn
15 degrees and back. Calibrate the arm model with `python quick_calibrate.py`.
"""
import json
import math
import os
import sys
import time

# ---- arm geometry (cm), measured on this arm ---------------------------------------------
TURNTABLE_RADIUS = 3.5    # the turntable is 7 cm across
SHOULDER_HEIGHT = 7.5     # shoulder axle above the paper (turntable top 4 + 3.5), over the turntable centre
L1 = 7.5                  # shoulder axle -> elbow axle
TIP_REACH = 11.0          # elbow axle -> closed claw tip

# How servo angles become arm angles. quick_calibrate.py measures this and saves it to
# arm_calibration.json, which is loaded below; until then the course library's convention is used.
#   kind "interior"  : the elbow servo sets the angle between upper arm and forearm
#   kind "absolute+" : the elbow servo sets the forearm's angle to the table (via a linkage);
#   kind "absolute-"   a larger servo angle tilts the forearm up (+) or down (-)
#   shoulder_offset, elbow_offset : how far each horn sits from its nominal position (degrees)
#   reach : elbow axle -> closed claw tip;  claw_dx, claw_dz : extra offset of a level claw
MODEL = {"kind": "interior", "shoulder_offset": 0.0, "elbow_offset": 0.0, "reach": TIP_REACH,
         "claw_dx": 0.0, "claw_dz": 0.0}
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_calibration.json")

# ---- calibration you may need to tune for your arm --------------------------------------
CHASSIS_STRAIGHT = 90.0   # turntable servo angle that points straight ahead (calibrated)
CHASSIS_SCALE = 1.0       # degrees the turntable turns per servo degree (calibrated)
CLAW_OPEN = 150           # as in the course's Robot_Arm_Stacking.py
CLAW_CLOSED = 5           # found by hand with manual_grab.py; holds a 2.5 cm cube
PARK = (90, 90, 90, 90)   # chassis, shoulder, elbow, claw: the default pose
LOOK = (0, 60, 80, 90)    # arm turned to the robot's right, off the paper, for the camera's picture

# Conservative servo limits, enforced on every move (a shoulder servo burnt when driven
# close to 180). Widen only after checking the arm moves freely there.
SAFE_RANGE = {"chassis": (0, 180), "shoulder": (40, 150), "elbow": (30, 150), "claw": (0, 180)}

# ---- speed: everything is slow and eased to protect the servos ---------------------------
MS_PER_DEGREE = 25        # a 90 degree turn takes about 2.3 s
MIN_MOVE_MS = 600
LINE_SPEED_CM_S = 2.0     # speed of the claw tip while following a path
LINE_STEP_CM = 0.5

# heights of the closed claw tip above the paper (cm)
Z_TRAVEL = 8.0            # moving between points, above any cube
Z_GRAB = 1.0              # picking up / putting down a 2.5 cm cube

# GPIO pins of chassis (turntable), shoulder, elbow, claws. The course wires the turntable to
# header H1 = GPIO5; on this arm it is plugged into H5 = GPIO19. Change it here if you move it.
SERVO_PINS = (19, 16, 17, 18)
HEADER = {5: "H1", 16: "H2", 17: "H3", 18: "H4", 19: "H5", 23: "H6",
          12: "H7", 13: "H8", 14: "H9", 25: "H10", 26: "H11", 27: "H12"}

# measured values from quick_calibrate.py, if it has been run
CALIBRATION = {}
if os.path.exists(CALIBRATION_FILE):
    with open(CALIBRATION_FILE) as _f:
        CALIBRATION = json.load(_f)
    MODEL.update(CALIBRATION["model"])
    if "chassis" in CALIBRATION:
        CHASSIS_STRAIGHT = CALIBRATION["chassis"]["straight"]
        CHASSIS_SCALE = CALIBRATION["chassis"]["scale"]


class Stopped(Exception):
    """Raised when a long move is stopped on request (e.g. Esc in the live window)."""


def _forearm_angle(a1, elbow, m):
    """Forearm angle above horizontal (deg) for upper-arm angle a1 and an elbow servo angle."""
    if m["kind"] == "interior":
        return a1 + elbow + m["elbow_offset"] - 180.0
    if m["kind"] == "absolute+":
        return elbow + m["elbow_offset"] - 90.0
    return 90.0 - elbow - m["elbow_offset"]


def _elbow_servo(a1, a2, m):
    """Inverse of _forearm_angle."""
    if m["kind"] == "interior":
        return a2 - a1 + 180.0 - m["elbow_offset"]
    if m["kind"] == "absolute+":
        return a2 + 90.0 - m["elbow_offset"]
    return 90.0 - a2 - m["elbow_offset"]


def tip_position(shoulder, elbow, model=None):
    """Servo angles -> closed claw tip: (distance from the turntable axis, height above paper), cm."""
    m = model or MODEL
    a1 = 180.0 - shoulder - m["shoulder_offset"]          # upper arm angle above horizontal
    a2 = _forearm_angle(a1, elbow, m)
    r1, r2 = math.radians(a1), math.radians(a2)
    return (L1 * math.cos(r1) + m["reach"] * math.cos(r2) + m["claw_dx"],
            SHOULDER_HEIGHT + L1 * math.sin(r1) + m["reach"] * math.sin(r2) - m["claw_dz"])


def solve(x, y, z):
    """Ground point (x, y) and tip height z, in cm -> (chassis, shoulder, elbow) servo angles.

    chassis turns towards the point; shoulder and elbow come from two-link inverse
    kinematics in the arm's vertical plane (elbow up), using the calibrated MODEL.
    Raises ValueError, before anything moves, if the point cannot be reached safely.
    """
    m = MODEL
    r = math.hypot(x, y)
    heading = math.degrees(math.atan2(x, y))          # 0 = straight ahead, + = right
    chassis = CHASSIS_STRAIGHT - heading / CHASSIS_SCALE   # right = smaller servo angle
    if not 0 <= chassis <= 180:
        raise ValueError("(%.1f, %.1f) is too far round to the side" % (x, y))
    if z < 0:
        raise ValueError("height %.1f cm is below the paper" % z)
    if r < TURNTABLE_RADIUS + 1.5 and z < 6:
        raise ValueError("(%.1f, %.1f) is too close to the turntable" % (x, y))

    dr = r - m["claw_dx"]                             # where the forearm must end, relative
    dz = z + m["claw_dz"] - SHOULDER_HEIGHT           # to the shoulder axle
    d = math.hypot(dr, dz)
    R = m["reach"]
    if not abs(L1 - R) < d < L1 + R:
        raise ValueError("(%.1f, %.1f, %.1f) is out of reach (%.1f cm from the shoulder, arm spans %.1f to %.1f)"
                         % (x, y, z, d, abs(L1 - R), L1 + R))
    a1 = math.atan2(dz, dr) + math.acos((L1 * L1 + d * d - R * R) / (2 * L1 * d))
    a2 = math.atan2(dz - L1 * math.sin(a1), dr - L1 * math.cos(a1))
    shoulder = 180.0 - math.degrees(a1) - m["shoulder_offset"]
    elbow = _elbow_servo(math.degrees(a1), math.degrees(a2), m)
    for name, a in (("shoulder", shoulder), ("elbow", elbow)):
        lo, hi = SAFE_RANGE[name]
        if not lo <= a <= hi:
            raise ValueError("(%.1f, %.1f, %.1f) needs %s = %.0f degrees, outside the safe range %d-%d"
                             % (x, y, z, name, a, lo, hi))
    return chassis, shoulder, elbow


def fit_model(samples):
    """Fit MODEL to measured poses [(shoulder, elbow, tip_r, tip_h), ...].

    Tries each kind of elbow; with 4+ poses also a level-claw offset. Returns a list of
    (rms_cm, model) with the best first.
    """
    from scipy.optimize import least_squares

    def fit(kind, free):
        base = {"kind": kind, "shoulder_offset": 0.0, "elbow_offset": 0.0, "reach": TIP_REACH,
                "claw_dx": 0.0, "claw_dz": 0.0}

        def residuals(p):
            m = dict(base, **dict(zip(free, p)))
            res = []
            for sh, el, r, h in samples:
                tr, th = tip_position(sh, el, m)
                res += [tr - r, th - h]
            # weak pull towards the measured geometry, so few poses still give a sane fit
            res += [m["shoulder_offset"] / 30.0, m["elbow_offset"] / 30.0, (m["reach"] - TIP_REACH) / 3.0]
            return res

        sol = least_squares(residuals, [base[k] for k in free])
        data = sol.fun[:2 * len(samples)]
        return math.sqrt(sum(v * v for v in data) / len(data)), dict(base, **dict(zip(free, sol.x)))

    rigid = ["shoulder_offset", "elbow_offset", "reach"]
    found = [fit(kind, rigid) for kind in ("interior", "absolute+", "absolute-")]
    if len(samples) >= 4:
        found += [fit(kind, rigid + ["claw_dx", "claw_dz"]) for kind in ("interior", "absolute+", "absolute-")]
    # a level-claw fit has two more parameters, so it must be clearly better to win
    found.sort(key=lambda f: f[0] * (1.25 if f[1]["claw_dx"] or f[1]["claw_dz"] else 1.0))
    return found


# Runs on the ESP32: moves all four servos together from where they are to the targets,
# with an ease-in/ease-out profile, one small step every 20 ms. Servo.set_angle accepts
# 0-180 (the library's claws_cmd would clamp the claw to 90-180).
BOARD_HELPERS = """
import libs.ACB_ARM as m
import time
def _servos():
    return (m.servo1, m.servo2, m.servo3, m.servo4)
def use_pins(pins):
    # drive each servo on its configured pin; one that is (re)created starts at its last angle
    for name, pin in zip(('servo1', 'servo2', 'servo3', 'servo4'), pins):
        s = getattr(m, name)
        if s is not None and ('Pin(%d)' % pin) in str(s.pwm):
            continue
        angle = 90 if s is None or s.current_angle is None else s.current_angle
        if s is not None:
            s.pwm.deinit()
        s = m.Servo(pin)
        s.set_angle(angle)
        setattr(m, name, s)
def smooth(targets, ms):
    S = _servos()
    start = [s.current_angle if s.current_angle is not None else 90 for s in S]
    n = max(1, ms // 20)
    for i in range(1, n + 1):
        t = i / n
        e = t * t * (3 - 2 * t)
        for s, a0, a in zip(S, start, targets):
            if a is not None:
                s.set_angle(a0 + (a - a0) * e)
        time.sleep_ms(20)
def follow(points, ms_each):
    S = _servos()[:3]
    for p in points:
        start = [s.current_angle for s in S]
        n = max(1, ms_each // 20)
        for i in range(1, n + 1):
            t = i / n
            for s, a0, a in zip(S, start, p):
                s.set_angle(a0 + (a - a0) * t)
            time.sleep_ms(20)
def angles():
    print([s.current_angle for s in _servos()])
"""


class ArmIK:
    def __init__(self, port="COM13"):
        from mpremote.transport_serial import SerialTransport
        self.t = SerialTransport(port, baudrate=115200)
        # no soft reset: if the servos are already driven, keep them where they are
        self.t.enter_raw_repl(soft_reset=False)
        self.t.exec(BOARD_HELPERS)
        # After power-up/reset every servo starts at 90 (so start with the arm resting in its
        # default pose); a servo found on a different pin is moved to the configured one.
        self.t.exec("use_pins(%r)" % (SERVO_PINS,))
        self.pos = None            # last (x, y, z) visited

    # -- low level ----------------------------------------------------------------------
    def _run(self, code, timeout=60):
        """Run code on the board and wait for it (the library's exec gives up after 10 s)."""
        from mpremote.transport import TransportExecError
        ret, err = self.t.exec_raw(code, timeout=timeout)
        if err:
            raise TransportExecError(ret, err.decode())
        return ret

    def angles(self):
        out = self.t.exec("angles()").decode()
        return [float(v) for v in out.strip().strip("[]").split(",")]

    def move_servos(self, chassis=None, shoulder=None, elbow=None, claw=None):
        """Eased move of the given servos (None = leave alone); duration grows with distance."""
        targets = [chassis, shoulder, elbow, claw]
        for name, a in zip(("chassis", "shoulder", "elbow", "claw"), targets):
            lo, hi = SAFE_RANGE[name]
            if a is not None and not lo <= a <= hi:
                raise ValueError("%s %.0f degrees is outside the safe range %d-%d; not moving" % (name, a, lo, hi))
        now = self.angles()
        biggest = max([abs(a - b) for a, b in zip(targets, now) if a is not None] or [0])
        ms = int(max(MIN_MOVE_MS, biggest * MS_PER_DEGREE))
        # plain floats: a numpy float would be sent as "np.float64(...)", which the board cannot read
        self._run("smooth(%r, %d)" % ([None if a is None else round(float(a), 2) for a in targets], int(ms)))

    # -- public -------------------------------------------------------------------------
    def goto(self, x, y, z=Z_TRAVEL):
        """Move the closed claw tip to ground coordinates (x, y), height z above the paper (cm)."""
        angles = solve(x, y, z)       # raises before anything moves if unreachable
        self.move_servos(*angles)
        self.pos = (x, y, z)
        return angles

    def _stream(self, points, should_stop=None):
        """Move the tip through closely spaced (x, y, z) points at a steady, slow speed.
        All points are solved before anything moves."""
        angles = [[round(float(a), 2) for a in solve(*p)] for p in points]
        ms_each = int(LINE_STEP_CM / LINE_SPEED_CM_S * 1000)
        for i in range(0, len(angles), 20):        # send in chunks to keep each command short
            if should_stop and should_stop():
                if i:
                    self.pos = points[i - 1]
                raise Stopped("stopped on request")
            self._run("follow(%r, %d)" % (angles[i:i + 20], ms_each))
        self.pos = points[-1]

    def line_to(self, x, y, z, should_stop=None):
        """Move the tip in a straight line from where it is to (x, y, z), slowly and steadily."""
        if self.pos is None:
            return self.goto(x, y, z)
        x0, y0, z0 = self.pos
        n = max(1, int(round(math.dist((x0, y0, z0), (x, y, z)) / LINE_STEP_CM)))
        self._stream([(x0 + (x - x0) * i / n, y0 + (y - y0) * i / n, z0 + (z - z0) * i / n)
                      for i in range(1, n + 1)], should_stop)

    def travel(self, x, y, z, should_stop=None):
        """Go to (x, y, z) without sweeping across the paper low down: straight up to travel
        height, straight across, straight down. If already above (x, y), just straight down/up.
        From rest, the first move is an eased swing to above the point."""
        solve(x, y, z)
        solve(x, y, Z_TRAVEL)                     # both ends checked before anything moves
        if self.pos is not None and math.hypot(self.pos[0] - x, self.pos[1] - y) < 0.3:
            return self.line_to(x, y, z, should_stop)
        if self.pos is None:
            self.goto(x, y, Z_TRAVEL)
        else:
            if self.pos[2] < Z_TRAVEL:
                self.line_to(self.pos[0], self.pos[1], Z_TRAVEL, should_stop)
            self.line_to(x, y, Z_TRAVEL, should_stop)
        if z != Z_TRAVEL:
            self.line_to(x, y, z, should_stop)

    def follow(self, points, z, on_start=None, should_stop=None):
        """Follow a list of ground points (x, y) at height z at a steady, slow speed.

        on_start(dense_points, ms_per_point) is called when the steady motion begins;
        should_stop() is checked between chunks (about every 5 s) and raises Stopped.
        """
        dense = [points[0]]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / LINE_STEP_CM))
            dense += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(1, n + 1)]
        for x, y in dense:
            solve(x, y, z)                         # check the whole path before moving
        self.travel(dense[0][0], dense[0][1], z, should_stop)
        if on_start:
            on_start(dense, int(LINE_STEP_CM / LINE_SPEED_CM_S * 1000))
        if len(dense) > 1:
            self._stream([(x, y, z) for x, y in dense[1:]], should_stop)

    def claw(self, angle):
        self.move_servos(claw=angle)

    def grab(self):
        """Close the claw (slowly)."""
        self.claw(CLAW_CLOSED)
        time.sleep(0.3)

    def release(self):
        """Open the claw (slowly)."""
        self.claw(CLAW_OPEN)
        time.sleep(0.3)

    def lift(self, z=Z_TRAVEL):
        """Raise straight up from the current position."""
        self.line_to(self.pos[0], self.pos[1], z)

    def pick(self, x, y):
        """Open, go over the point, straight down, close, straight up."""
        solve(x, y, Z_GRAB)           # check it is reachable before moving
        self.release()
        self.travel(x, y, Z_GRAB)
        self.grab()
        self.lift()

    def place(self, x, y):
        """Carry to (x, y), straight down, open, straight up."""
        solve(x, y, Z_GRAB)
        self.travel(x, y, Z_GRAB)
        self.release()
        self.lift()

    def park(self):
        """Back to the default pose, all servos at 90, slowly. Rises first if it is low."""
        if self.pos is not None and self.pos[2] < Z_TRAVEL:
            self.lift()
        self.move_servos(*PARK)
        self.pos = None

    def close(self):
        try:
            self.park()
        finally:
            self.t.exit_raw_repl()
            self.t.close()


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        x, y = float(sys.argv[1]), float(sys.argv[2])
        z = float(sys.argv[3]) if len(sys.argv) > 3 else Z_TRAVEL
        print("servo angles (chassis, shoulder, elbow):", [round(v, 1) for v in solve(x, y, z)])
        a = ArmIK()
        try:
            a.release()
            a.goto(x, y, z)
            time.sleep(2)
        finally:
            a.close()
    elif len(sys.argv) == 2 and sys.argv[1] == "test":
        # watch the arm: each joint turns 15 degrees and back, slowly
        a = ArmIK()
        try:
            a.move_servos(90, 90, 90)
            for i, (name, pose) in enumerate((("turntable", (105, 90, 90)), ("shoulder", (90, 105, 90)),
                                              ("elbow", (90, 90, 105)))):
                pin = SERVO_PINS[i]
                print("%s (header %s = GPIO%d): 15 degrees and back..." % (name, HEADER.get(pin, "?"), pin))
                a.move_servos(*pose)
                time.sleep(0.5)
                a.move_servos(90, 90, 90)
                time.sleep(0.5)
        finally:
            a.close()
    elif len(sys.argv) == 2 and sys.argv[1] == "park":
        a = ArmIK()
        print("angles before:", a.angles())
        a.close()                  # close() parks slowly
        print("parked at", PARK)
    else:
        print(__doc__)
