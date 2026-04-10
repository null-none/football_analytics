# Football Analytics

YOLO-based player analytics for football: detection, speed tracking, sprint stats.

## Install

```bash
pip install -r requirements.txt
```

## Commands overview

| Command | What it does |
|---|---|
| `train` | Fine-tune a YOLO model on your dataset |
| `category` | Quick sanity-check — bbox + class label, no extras |
| `detect` | Full overlay suite: spider web, convex hull, defense line/zone, pressure halos |
| `speed` | Per-player speed, sprint detection, distance stats with optional perspective calibration |

---

## Train

```bash
python main.py train \
    --model yolo26s.pt \
    --data data.yaml \
    --epochs 50 \
    --batch 8
```

---

## Category — quick class check

Draws bounding boxes with class labels only. Useful for verifying which class IDs your model uses.

```bash
# All classes
python main.py category --weights best.pt --source input.mp4

# Specific classes
python main.py category --weights best.pt --source input.mp4 --classes 0 1
```

> **Tip:** check class IDs without running detection:
> ```bash
> python -c "from ultralytics import YOLO; m = YOLO('best.pt'); print(m.names)"
> ```

Output: `out_category/result_category.mp4`, `out_category/detections_category.csv`

---

## Detect — player overlays

All overlays are **off by default** — enable individually:

```bash
python main.py detect \
    --weights best.pt \
    --source input.mp4 \
    --classes 0 1 \
    --show_spider_web \
    --show_convex_hull \
    --show_defense_line \
    --show_pressure
```

### Overlay flags

| Flag | Description |
|---|---|
| `--show_spider_web` | Lines connecting every pair of detected players |
| `--show_convex_hull` | Filled convex hull around all detected players |
| `--show_defense_line` | Polyline through the N deepest defenders per team (sorted left→right) |
| `--show_defense_zone` | Vertical zone line per team through the N defenders furthest from midfield |
| `--show_pressure` | Colour-coded pressing halo per player based on nearby opponents |

### Tuning flags

| Flag | Default | Description |
|---|---|---|
| `--defense_n` | `4` | Number of defenders used for defense line / zone |
| `--pressure_r` | `120` | Opponent detection radius in pixels for pressure halo |
| `--classes` | `0` | YOLO class IDs to track; use `--classes 0 1` for two teams |
| `--dot_radius` | `5` | Radius of player dot markers |

### Pressure halo colours

| Opponents within `--pressure_r` | Halo |
|---|---|
| 1 | Yellow |
| 2 | Orange |
| 3+ | Red |

Requires two classes (`--classes 0 1`) to distinguish teams.

Press `Q` to stop early.
Output: `out/result.mp4`, `out/detections.csv`

---

## Speed — speed & sprint tracking

### Simple mode (no calibration)

Provide the field width in both metres and pixels. Measure `--field_w_px` by hand from the video.

```bash
python main.py speed \
    --weights best.pt \
    --source input.mp4 \
    --field_w_m 68 \
    --field_w_px 950 \
    --out_dir out_speed
```

### Calibration mode — 6-point perspective correction

Recommended when the camera is at an angle or uses a wide-angle lens.
Before tracking starts, a calibration window opens on the first video frame.

```bash
python main.py speed \
    --weights best.pt \
    --source input.mp4 \
    --field_w_m 68 \
    --field_h_m 105 \
    --calibrate \
    --out_dir out_speed
```

**How to calibrate:** click the 6 field landmarks in the order shown on screen.

```
TL(1) ──── TC(6) ──── TR(5)
  │                     │
BL(2) ──── BC(3) ──── BR(4)
```

| # | Point |
|---|---|
| 1 | Top-Left corner |
| 2 | Bottom-Left corner |
| 3 | Bottom-Center (midpoint of bottom edge) |
| 4 | Bottom-Right corner |
| 5 | Top-Right corner |
| 6 | Top-Center (midpoint of top edge) |

- `ESC` — undo the last point
- `Q` — confirm and start tracking (minimum 4 points)

**Fisheye detection:** after 6 points are placed, the system checks whether the edge midpoints (TC, BC) deviate from the straight line between their respective corners. Deviation > 1.5% of image diagonal triggers a fisheye warning.

### Speed flags

| Flag | Default | Description |
|---|---|---|
| `--field_w_m` | `68.0` | Field width in metres |
| `--field_h_m` | `105.0` | Field length in metres (used for calibration) |
| `--field_w_px` | `950.0` | Field width in pixels (fallback without `--calibrate`) |
| `--calibrate` | off | Open interactive 6-point calibration window |
| `--smooth` | `15` | History window for speed smoothing (frames) |
| `--classes` | `0` | YOLO class IDs to track |
| `--show_frame_id` | off | Overlay current frame number on output video |

Press `Q` to stop early.
Output: `out_speed/result_speed.mp4`, `out_speed/speeds.csv`, `out_speed/summary.txt`

---

## Project structure

```
football_analytics/
├── main.py                # FootballAnalytics (inherits all modules) + CLI
├── trainer.py             # YOLOTrainer
├── category_detector.py   # CategoryDetector  — simple bbox + label per class
├── player_detector.py     # PlayerDetector    — overlays (hull, defense, pressure)
├── pressure_analyzer.py   # PressureAnalyzer  — pressing pressure logic
├── speed_tracker.py       # FieldCalibrator + SpeedTracker — speed, sprints, CSV
└── requirements.txt
```

---

## Library usage

```python
from main import FootballAnalytics

# Speed tracking with 6-point calibration
fa = FootballAnalytics(
    weights="best.pt",
    source="input.mp4",
    field_w_m=68,
    field_h_m=105,
    calibrate=True,
)
fa.track()

# Detection with overlays
fa2 = FootballAnalytics(
    weights="best.pt",
    source="input.mp4",
    show_spider_web=True,
    show_convex_hull=True,
    show_pressure=True,
    pressure_r=120,
    classes=[0, 1],
)
fa2.detect()
```
