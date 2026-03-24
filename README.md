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

### Detect players (heatmap + radar + convex hull)

```bash
python main.py detect \
    --weights best.pt \
    --source input.mp4 \
    --out_dir out \
    --conf 0.25
```

On startup you will be asked to click 4 court corners for the radar (or press `Esc` to skip).
Press `Q` to stop playback early.

Output: `out/result.mp4`, `out/detections.csv`

### Track speed & sprints

```bash
python main.py speed \
    --weights best.pt \
    --source input.mp4 \
    --court_w_m 40 \
    --court_w_px 950 \
    --out_dir out_speed
```

`--court_w_px` — measure the court width in pixels manually from the video.
Press `Q` to stop early.

Output: `out_speed/result_speed.mp4`, `out_speed/speeds.csv`, `out_speed/summary.txt`

## Project structure

```
football_analytics/
├── main.py            # FootballAnalytics class (inherits all three) + CLI
├── trainer.py         # YOLOTrainer
├── player_detector.py # PlayerDetector
├── speed_tracker.py   # SpeedTracker
└── requirements.txt
```

## Use as a library

```python
from football_analytics.main import FootballAnalytics

fa = FootballAnalytics(
    weights="best.pt",
    source="match.mp4",
    court_w_m=40,
    court_w_px=950,
)

fa.detect()   # detection + radar + heatmap
fa.track()    # speed + sprint stats
```
