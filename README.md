# Football Analytics

YOLO-based player analytics for football: detection, speed tracking, sprint stats.

## Install

```bash
pip install -r requirements.txt
```

## Usage

### Train a model

```bash
python main.py train \
    --model yolo26s.pt \
    --data data.yaml \
    --epochs 50 \
    --batch 8
```

### Simple detection by category

Minimal detection mode — draws bounding boxes with class labels, no extra overlays.
Useful for quickly checking what the model detects and which class IDs it uses.

```bash
# Detect all classes
python main.py category \
    --weights best.pt \
    --source input.mp4

# Detect specific classes only
python main.py category \
    --weights best.pt \
    --source input.mp4 \
    --classes 0 1
```

Output: `out_category/result_category.mp4`, `out_category/detections_category.csv`

---

### Detect players

> **Tip:** first check which class IDs your model uses:
> ```bash
> python -c "from ultralytics import YOLO; m = YOLO('best.pt'); print(m.names)"
> ```
> Then pass the correct ID via `--classes` (default: `0`).

```bash
python main.py detect \
    --weights best.pt \
    --source input.mp4 \
    --classes 0 \
    --show_spider_web \
    --show_convex_hull
```

All visual overlays are **disabled by default**. Enable them individually with flags:

| Flag | Overlay |
|---|---|
| `--show_spider_web` | Lines connecting every pair of detected players |
| `--show_convex_hull` | Filled convex hull around all detected players |
| `--show_defense_line` | Defensive line through each team's backline (by Y position) |
| `--show_defense_zone` | Vertical defensive line per team sorted by X (requires `--classes 0 1`) |
| `--show_pressure` | Pressing pressure halos on each player (requires `--classes 0 1`) |

| Flag | Default | Description |
|---|---|---|
| `--defense_n 4` | `4` | Number of defenders per side |
| `--pressure_r 120` | `120` | Opponent detection radius in pixels for pressure halo |
| `--classes 0` | `0` | Only class 0 |
| `--classes 0 1` | — | Classes 0 and 1 |

#### Defensive line (`--show_defense_line`)

Draws a polyline through the N deepest defenders on each side, sorted left→right. Each defender's real Y position is preserved — you can see if the line is flat or broken.
Extends to both frame edges with a semi-transparent ghost line.

```bash
python main.py detect \
    --weights best.pt \
    --source input.mp4 \
    --show_defense_line \
    --defense_n 4
```

#### Defensive zone (`--show_defense_zone`)

Draws a vertical line for each team through the last N defenders (the ones furthest from midfield), sorted by Y. Shows horizontal positioning and gaps. Requires `--classes 0 1`.

```bash
python main.py detect \
    --weights best.pt \
    --source input.mp4 \
    --show_defense_zone \
    --defense_n 4 \
    --classes 0 1
```

#### Pressing pressure (`--show_pressure`)

Draws a colour-coded halo around each player based on how many opponents are within `--pressure_r` pixels:

| Opponents nearby | Halo colour |
|---|---|
| 1 | Yellow |
| 2 | Orange |
| 3+ | Red |

Requires two classes (`--classes 0 1`) to distinguish teams.

```bash
python main.py detect \
    --weights best.pt \
    --source input.mp4 \
    --show_pressure \
    --pressure_r 120 \
    --classes 0 1
```

Press `Q` to stop playback early.

Output: `out/result.mp4`, `out/detections.csv`

### Track speed & sprints

```bash
python main.py speed \
    --weights best.pt \
    --source input.mp4 \
    --field_w_m 68 \
    --field_w_px 950 \
    --out_dir out_speed
```

`--field_w_px` — measure the field width in pixels manually from the video.
Press `Q` to stop early.

**Class filter** — by default tracks class `1`. Use `--classes` to override:

```bash
python main.py speed \
    --weights best.pt \
    --source input.mp4 \
    --field_w_px 950 \
    --classes 0
```

**Frame counter** — add `--show_frame_id` to display the current frame number on each frame of the output video.

Output: `out_speed/result_speed.mp4`, `out_speed/speeds.csv`, `out_speed/summary.txt`

## Project structure

```
football_analytics/
├── main.py                # FootballAnalytics class (inherits all) + CLI
├── trainer.py             # YOLOTrainer
├── category_detector.py   # CategoryDetector  — simple bbox + label per class
├── player_detector.py     # PlayerDetector    — overlays
├── pressure_analyzer.py   # PressureAnalyzer  — pressing pressure halos
├── speed_tracker.py       # SpeedTracker      — speed, sprints, CSV
└── requirements.txt
```

## Use as a library

```python
from football_analytics.main import FootballAnalytics

fa = FootballAnalytics(
    weights="best.pt",
    source="input.mp4",
    field_w_m=68,
    field_w_px=950,
)

fa.detect()   # detection only (no overlays by default)

# With overlays:
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
fa.track()    # speed + sprint stats
```
