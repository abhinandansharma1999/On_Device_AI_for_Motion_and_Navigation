# On-Device AI for Motion and Navigation

A desktop robot arm that sees with open-weight AI models and moves by itself. Name a colour
and it finds that cube on the paper, picks it up, carries it along a hand-drawn path, and puts
it down at the end of the path. All AI runs locally on the laptop's CPU, with no cloud API.

Built for Hacktoberfest 2026 (Best Open-Source AI Project).

## How it works

```
overhead webcam --> paper outline (homography) --> camera pixels -> cm on the paper -> arm coordinates
               |--> OWLv2:   "a yellow cube"            --> where the cube is
               '--> CLIPSeg: "a hand drawn curved line" --> the path, sampled along the line

calibrated inverse kinematics (on the PC) --> slow, eased servo motion streamed to the ESP32 (MicroPython)
```

1. **Look.** The arm turns aside, the camera takes a picture, and the saved paper outline is
   checked against it, so a nudged camera is caught instead of silently shifting every position.
2. **Find.** OWLv2 finds the cube of the requested colour. CLIPSeg picks the drawn path out of
   the dark strokes on the paper; the path is thinned to a centre line and sampled into points.
3. **Plan.** Positions are converted to centimetres around the arm's turntable. Every position,
   and every straight-line move between them, is checked against the arm's reach before anything moves.
4. **Move.** Straight down to grab, up, across, down to the path, along the path, down to put the
   cube down, back to rest. The PC does the inverse kinematics; the ESP32 only plays smooth servo motion.
5. **Watch.** A live camera window shows the cubes, the path, the arm's reach zones and the
   progress. Esc stops safely.

## Open-weight models

| Job | Model | Licence |
|---|---|---|
| Find the cube (open-vocabulary detection, prompt such as "a yellow cube") | [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) | Apache-2.0 |
| Find the drawn path (text-prompted segmentation) | [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined) | see the model card |

Both run on the CPU through Hugging Face `transformers`, about 10 seconds per look on a laptop.
The weights download automatically the first time.

## Hardware

- ACEBOTT QD022 4-DOF robot arm kit (ESP32-Max board, four servos)
- Servo headers on this build: turntable **H5 (GPIO19)**, shoulder H2 (GPIO16), elbow H3 (GPIO17),
  claw H4 (GPIO18). The kit's manual puts the turntable on H1 (GPIO5); set `SERVO_PINS` in
  `arm_ik.py` to match your wiring.
- A USB webcam looking straight down at the paper, about 37 cm above it
- A sheet of paper, 27.3 x 21.6 cm (change `PAPER_CM` in `final_demo.py` and `set_paper.py` for
  other sizes), with the arm's turntable just behind its bottom edge
- Coloured wooden cubes (about 2.5 cm) and a pen for the path

## Setup

PC (tested with Python 3.13 on Windows):

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

ESP32: MicroPython (tested with v1.23, `ESP32_GENERIC`) plus the kit's servo library
`libs/ACB_ARM.py`. The library comes with the kit's course files
(`Python(Experienced Learner)/4.Library Files/libs/`) and is not included in this repository.

```
mpremote connect COM13 fs mkdir :libs
mpremote connect COM13 fs cp ACB_ARM.py :libs/ACB_ARM.py
```

Settings to check for your setup: the serial port (`COM13`, `--port` or `ArmIK(port)`) and the
camera index (`CAMERA = 1` in `final_demo.py`, `quick_calibrate.py` and `set_paper.py`).

## Run

Start with the arm resting at 90 degrees on every servo, and keep hands clear while it moves.

| Step | Command | What it does |
|---|---|---|
| 1 | `python arm_ik.py test` | Each joint turns 15 degrees and back, to check every servo responds |
| 2 | `python manual_grab.py` | Find the claw angle that holds a cube (`claw A`), then set `CLAW_CLOSED` in `arm_ik.py` |
| 3 | `python quick_calibrate.py` | Camera joint check, then four poses measured with a ruler, which give the arm model |
| 4 | `python quick_calibrate.py --check` | Optional: holds the claw tip over a known point to confirm the calibration |
| 5 | `python set_paper.py` | Accept (Enter) or click the paper's four corners in the camera view |
| 6 | `python final_demo.py --dry-run --colour yellow` | Look and plan only |
| 7 | `python final_demo.py --colour yellow` | The demo |

Run step 5 again whenever the camera or the paper moves (the demo tells you), and step 3 when the
robot moves relative to the paper.

## Safety

- Every move is slow and eased. Near the paper the claw tip follows straight lines at 2 cm/s.
- The shoulder and elbow servos are limited to 40-150 and 30-150 degrees.
- Before each run every joint is moved a little and the camera confirms it actually moved.
- Positions out of reach are refused before the arm moves.
- Esc, or closing the live window, stops after the current move and parks the arm.

## Files

| File | Purpose |
|---|---|
| `final_demo.py` | The demo |
| `vision.py` | OWLv2 cube detection, CLIPSeg path detection |
| `live_view.py` | Live camera window with everything the demo knows drawn on it |
| `arm_ik.py` | Arm model, inverse kinematics, smooth motion on the ESP32 |
| `quick_calibrate.py` | Measures the arm model |
| `set_paper.py`, `paper_finder.py` | Where the paper is in the camera picture |
| `manual_grab.py` | Drive the arm by hand |
| `range_of_motion.py` | Camera and paper geometry helpers; reach sweep |
| `arm_calibration.json`, `paper_corners.json`, `calib_points.csv` | Calibration of the author's setup; redo it for yours |
