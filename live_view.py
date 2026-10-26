"""
Live camera window for the arm demo.

Shows the overhead camera in real time with what the demo knows drawn on top: the paper,
the arm's grab and path zones, every cube found (the target highlighted), the path points,
the part of the path already traced, and a crosshair where the arm is heading.
It runs in its own thread, so the picture keeps moving while the arm does.

Esc or closing the window asks the demo to stop; it finishes the current move and parks.
"""
import threading
import time

import cv2
import numpy as np

WINDOW = "Arm demo - live camera   (Esc = stop)"
CUBE_BGR = {"red": (40, 40, 230), "green": (60, 190, 60), "yellow": (0, 215, 235),
            "blue": (230, 130, 40), "orange": (0, 140, 255)}
PATH_BGR = (255, 0, 255)
TRACED_BGR = (255, 255, 0)


def _text(img, s, org, colour=(255, 255, 255), size=0.5, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, colour, thick, cv2.LINE_AA)


def _crosshair(img, p, colour, label):
    x, y = p
    cv2.circle(img, p, 14, colour, 2, cv2.LINE_AA)
    cv2.line(img, (x - 22, y), (x + 22, y), colour, 2, cv2.LINE_AA)
    cv2.line(img, (x, y - 22), (x, y + 22), colour, 2, cv2.LINE_AA)
    _text(img, label, (x + 18, y - 18), colour, 0.55, 2)


class LiveView:
    def __init__(self, camera_index, window=True, scale=1.5):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open camera %d" % camera_index)
        self.scale = scale
        self.gui = window
        self.lock = threading.Lock()
        self.frame = None
        self.count = 0                 # frames read so far
        self.items = {}                # what to draw, see show()
        self.stop_requested = False
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        t0 = time.time()
        while self.frame is None and time.time() - t0 < 5:
            time.sleep(0.05)
        if self.frame is None:
            self.close()
            raise RuntimeError("camera %d gives no picture" % camera_index)

    # ---- called by the demo -------------------------------------------------------------
    def show(self, **items):
        """Change what is drawn. Keys (all in camera pixels):
        paper=[4 corners], zones=[(polygon, bgr, label)], cubes=[{box, colour, label, role}],
        path=[points], path_ok=[bool per point], progress=(t0, ms_each, dense points),
        target=(point, label) or None, placed=(point, colour) or None, status=[lines]."""
        with self.lock:
            self.items.update(items)

    def snapshot(self, fresh=4):
        """A camera frame taken after this call (waits for `fresh` new frames)."""
        with self.lock:
            start = self.count
        t0 = time.time()
        while time.time() - t0 < 3:
            with self.lock:
                if self.count >= start + fresh:
                    return self.frame.copy()
            time.sleep(0.02)
        with self.lock:
            return self.frame.copy()

    def render(self, frame=None):
        """The current picture with everything drawn on it (for saving)."""
        with self.lock:
            frame = self.frame.copy() if frame is None else frame
            items = dict(self.items)
        return self._draw(frame, items)

    def wait_until_closed(self):
        while self.running and self.gui and not self.stop_requested:
            time.sleep(0.1)

    def close(self):
        self.running = False
        self.thread.join(timeout=3)
        self.cap.release()

    # ---- window thread ------------------------------------------------------------------
    def _loop(self):
        if self.gui:
            try:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW, int(640 * self.scale), int(480 * self.scale))
            except cv2.error:
                print("(this OpenCV cannot open windows: pip install opencv-python)")
                self.gui = False
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            with self.lock:
                self.frame = frame
                self.count += 1
                items = dict(self.items)
            if self.gui:
                cv2.imshow(WINDOW, self._draw(frame, items))
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    self.stop_requested = True
                    self.gui = False
                    cv2.destroyWindow(WINDOW)
        if self.gui:
            cv2.destroyWindow(WINDOW)
            cv2.waitKey(1)

    def _draw(self, frame, it):
        s = self.scale
        img = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

        def P(p):
            return int(round(p[0] * s)), int(round(p[1] * s))

        if it.get("paper"):
            cv2.polylines(img, [np.array([P(p) for p in it["paper"]], np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
        for poly, colour, label in it.get("zones", []):
            pts = np.array([P(p) for p in poly], np.int32)
            cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)
            top = pts[pts[:, 1].argmin()]
            _text(img, label, (int(top[0]) + 6, int(top[1]) - 6), colour, 0.5)

        path = it.get("path")
        if path:
            ok = it.get("path_ok") or [True] * len(path)
            pts = [P(p) for p in path]
            cv2.polylines(img, [np.array(pts, np.int32)], False, PATH_BGR, 2, cv2.LINE_AA)
            for i, (p, good) in enumerate(zip(pts, ok)):
                cv2.circle(img, p, 5, PATH_BGR if good else (0, 0, 255), -1, cv2.LINE_AA)
                _text(img, str(i), (p[0] + 6, p[1] - 6), PATH_BGR if good else (0, 0, 255), 0.4)
            cv2.circle(img, pts[0], 9, (0, 200, 0), 2, cv2.LINE_AA)
            _text(img, "start", (pts[0][0] - 20, pts[0][1] + 26), (0, 220, 0), 0.5)
            cv2.circle(img, pts[-1], 9, (0, 0, 255), 2, cv2.LINE_AA)
            _text(img, "end", (pts[-1][0] - 12, pts[-1][1] + 26), (0, 0, 255), 0.5)

        prog = it.get("progress")
        if prog:
            t0, ms_each, dense = prog
            k = int(min(len(dense) - 1, max(0, (time.time() - t0) * 1000.0 / ms_each)))
            done = np.array([P(p) for p in dense[:k + 1]], np.int32)
            if len(done) > 1:
                cv2.polylines(img, [done], False, TRACED_BGR, 4, cv2.LINE_AA)
            _crosshair(img, P(dense[k]), TRACED_BGR, "tracing %d%%" % (100 * k / max(1, len(dense) - 1)))

        for c in it.get("cubes", []):
            x0, y0, x1, y1 = c["box"]
            colour = CUBE_BGR.get(c["colour"], (200, 200, 200))
            role = c.get("role", "other")
            if role == "carried":
                cv2.rectangle(img, P((x0, y0)), P((x1, y1)), (150, 150, 150), 1, cv2.LINE_AA)
                _text(img, "picked up", (P((x0, y0))[0], P((x0, y0))[1] - 8), (180, 180, 180), 0.5)
                continue
            cv2.rectangle(img, P((x0, y0)), P((x1, y1)), colour, 3 if role == "target" else 1, cv2.LINE_AA)
            label = c.get("label", c["colour"])
            _text(img, label, (P((x0, y0))[0], P((x0, y0))[1] - 8),
                  (0, 0, 255) if "OUT OF REACH" in label else colour, 0.55 if role == "target" else 0.45,
                  2 if role == "target" else 1)

        if it.get("placed"):
            p, colour = it["placed"]
            q = P(p)
            cv2.rectangle(img, (q[0] - 16, q[1] - 16), (q[0] + 16, q[1] + 16), CUBE_BGR.get(colour, (200, 200, 200)), 3)
            _text(img, "placed", (q[0] - 24, q[1] - 22), CUBE_BGR.get(colour, (200, 200, 200)), 0.5, 2)
        if it.get("target") and not prog:
            p, label = it["target"]
            _crosshair(img, P(p), (0, 255, 255), label)

        lines = it.get("status") or []
        if lines:
            bar = img.copy()
            cv2.rectangle(bar, (0, 0), (img.shape[1], 14 + 26 * len(lines)), (0, 0, 0), -1)
            img = cv2.addWeighted(bar, 0.55, img, 0.45, 0)
            for i, line in enumerate(lines):
                _text(img, line, (12, 30 + 26 * i), (255, 255, 255) if i == 0 else (200, 230, 255),
                      0.7 if i == 0 else 0.55, 2 if i == 0 else 1)
        _text(img, "Esc = stop", (img.shape[1] - 110, img.shape[0] - 12), (220, 220, 220), 0.5)
        return img
