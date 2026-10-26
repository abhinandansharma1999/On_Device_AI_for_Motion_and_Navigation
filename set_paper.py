"""
Tell the demo where the paper is in the camera picture. Run it once, and again whenever the
camera or the paper has been moved (final_demo.py tells you when).

    python set_paper.py

The arm turns aside, then a live camera window opens:
  - the outline found automatically is drawn in green. Press Enter to accept it.
  - or click the paper's 4 corners yourself (any order); the outline turns yellow. Enter saves.
  - R clears your clicks, Esc cancels without saving.
The whole sheet must be visible in the picture. Saved to paper_corners.json.
"""
import datetime
import json
import os
import textwrap

import cv2
import numpy as np

import arm_ik
import paper_finder

CAMERA = 1
PAPER_CM = (27.3, 21.6)
SCALE = 1.5
WINDOW = "Paper setup - Enter = save, click 4 corners to set them yourself, R = redo, Esc = cancel"
PAPER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_corners.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def order(points):
    p = np.array(points, np.float32)
    s, d = p.sum(axis=1), np.diff(p, axis=1).ravel()
    return [tuple(float(v) for v in q) for q in (p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)])]


def draw(frame, auto, auto_msg, clicks):
    img = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
    P = lambda p: (int(p[0] * SCALE), int(p[1] * SCALE))
    if len(clicks) == 4:
        cv2.polylines(img, [np.array([P(p) for p in order(clicks)], np.int32)], True, (0, 230, 255), 2)
        lines = ["Your outline (yellow). Enter = save, R = redo."]
    elif auto:
        cv2.polylines(img, [np.array([P(p) for p in auto], np.int32)], True, (0, 220, 0), 2)
        lines = ["Found the paper (green). Enter = save it,", "or click its 4 corners yourself."]
    else:
        lines = ["Could not find the paper by itself:"] + textwrap.wrap(auto_msg, 62) \
                + ["Fix the camera/paper, or click the 4 corners yourself."]
    for p in clicks:
        cv2.circle(img, P(p), 6, (0, 0, 255), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (12, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (12, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    arm = arm_ik.ArmIK()
    cap = None
    try:
        print("Turning the arm aside so the camera can see the whole paper...")
        arm.move_servos(*arm_ik.LOOK)
        cap = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
        for _ in range(15):
            cap.read()
        clicks = []
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, int(640 * SCALE), int(480 * SCALE))
        cv2.setMouseCallback(WINDOW, lambda ev, x, y, *_: (ev == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4
                                                            and clicks.append((x / SCALE, y / SCALE))))
        auto, auto_msg, n, saved = None, "", 0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if n % 10 == 0:                      # re-detect a few times a second
                try:
                    auto, auto_msg = paper_finder.find_paper(frame, PAPER_CM), ""
                except paper_finder.PaperNotFound as e:
                    auto, auto_msg = None, str(e)
            n += 1
            cv2.imshow(WINDOW, draw(frame, auto, auto_msg, clicks))
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                print("Cancelled - nothing saved.")
                break
            if key in (ord("r"), ord("R")):
                clicks.clear()
            if key in (13, 10):
                corners = order(clicks) if len(clicks) == 4 else auto
                if corners is None:
                    continue
                with open(PAPER_FILE, "w") as f:
                    json.dump({"corners": corners, "source": "clicked" if len(clicks) == 4 else "found",
                               "date": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=2)
                os.makedirs(OUT, exist_ok=True)
                cv2.imwrite(os.path.join(OUT, "paper_setup.png"), draw(frame, corners, "", []))
                saved = corners
                break
        cv2.destroyAllWindows()
        if saved:
            print("Saved the paper outline to paper_corners.json:", [tuple(int(v) for v in p) for p in saved])
            print("Picture: output/paper_setup.png. Next: python final_demo.py --dry-run --colour yellow")
    finally:
        if cap is not None:
            cap.release()
        arm.close()                              # parks slowly


if __name__ == "__main__":
    main()
