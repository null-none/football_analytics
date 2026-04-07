# Football Analytics

YOLO-based player analytics for football: detection, heatmap, radar, speed tracking, sprint stats.

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
    --show_heatmap \
    --show_radar \
    --show_spider_web \
    --show_convex_hull
```

All visual overlays are **disabled by default**. Enable them individually with flags:

| Flag | Overlay |
|---|---|
| `--show_heatmap` | Accumulated heat-map of player positions (top-left) |
| `--show_radar` | Mini football-field radar with player dots (top-right) |
| `--show_spider_web` | Lines connecting every pair of detected players |
| `--show_convex_hull` | Filled convex hull around all detected players |

| Flag | Description |
|---|---|
| `--classes 0` | Only class 0 (default) |
| `--classes 0 1` | Classes 0 and 1 |

Example — radar and heatmap only:

```bash
python main.py detect \
    --weights best.pt \
    --source input.mp4 \
    --show_radar \
    --show_heatmap
```

Example — all overlays:

```bash
python main.py detect \
    --weights best.pt \
    --source input.mp4 \
    --show_heatmap \
    --show_radar \
    --show_spider_web \
    --show_convex_hull
```

On startup you will be asked to click 6 field corners for homography (radar), or press `Esc` to skip.
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
├── main.py               # FootballAnalytics class (inherits all) + CLI
├── trainer.py            # YOLOTrainer
├── category_detector.py  # CategoryDetector — simple bbox + label per class
├── player_detector.py    # PlayerDetector   — heatmap, radar, overlays
├── speed_tracker.py      # SpeedTracker     — speed, sprints, CSV
└── requirements.txt
```

## Use as a library

```python
from football_analytics.main import FootballAnalytics

fa = FootballAnalytics(
    weights="best.pt",
    source="match.mp4",
    field_w_m=68,
    field_w_px=950,
)

fa.detect()   # detection only (no overlays by default)

# With overlays:
fa2 = FootballAnalytics(
    weights="best.pt",
    source="match.mp4",
    show_heatmap=True,
    show_radar=True,
    show_spider_web=True,
    show_convex_hull=True,
)
fa2.detect()
fa.track()    # speed + sprint stats
```
