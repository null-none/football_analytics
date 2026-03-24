#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FootballAnalytics — single entry point for the entire pipeline.

Examples:
    # Model training
    python main.py train --model yolo26s.pt --data data.yaml --epochs 50

    # Player detection (heatmap, radar, convex hull, spider web)
    python main.py detect --weights best.pt --source input.mp4

    # Speed tracking
    python main.py speed --weights best.pt --source input.mp4 --field_w_px 950
"""

import argparse

from trainer import YOLOTrainer
from player_detector import PlayerDetector
from speed_tracker import SpeedTracker


class FootballAnalytics(YOLOTrainer, PlayerDetector, SpeedTracker):
    """
    Combines YOLOTrainer, PlayerDetector and SpeedTracker into one class.
    All parameters are passed through the constructor and stored as attributes.
    """

    def __init__(
        self,
        # common
        weights: str = "best.pt",
        source: str = "input.mp4",
        out_dir: str = "out",
        conf: float = 0.25,
        imgsz: int = 1280,
        # training
        model_path: str = "yolo26s.pt",
        data_yaml: str = "data.yaml",
        epochs: int = 50,
        batch: int = 8,
        workers: int = 4,
        # detection
        dot_radius: int = 5,
        # speed
        field_w_m: float = 68.0,   # Standard football field width
        field_w_px: float = 950.0,
        smooth: int = 15,
    ):
        YOLOTrainer.__init__(
            self,
            model_path=model_path,
            data_yaml=data_yaml,
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            workers=workers,
        )
        PlayerDetector.__init__(
            self,
            weights=weights,
            source=source,
            out_dir=out_dir,
            conf=conf,
            imgsz=imgsz,
            dot_radius=dot_radius,
        )
        SpeedTracker.__init__(
            self,
            weights=weights,
            source=source,
            out_dir=out_dir,
            conf=conf,
            imgsz=imgsz,
            field_w_m=field_w_m,
            field_w_px=field_w_px,
            smooth=smooth,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Football Analytics — YOLO-based player detection and tracking")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- train ---
    t = sub.add_parser("train", help="Train YOLO model")
    t.add_argument("--model",   default="yolo26s.pt")
    t.add_argument("--data",    default="data.yaml")
    t.add_argument("--imgsz",   type=int,   default=1280)
    t.add_argument("--epochs",  type=int,   default=50)
    t.add_argument("--batch",   type=int,   default=8)
    t.add_argument("--workers", type=int,   default=4)

    # --- detect ---
    d = sub.add_parser("detect", help="Player detection (heatmap, radar, spider web)")
    d.add_argument("--weights",    required=True)
    d.add_argument("--source",     required=True)
    d.add_argument("--out_dir",    default="out")
    d.add_argument("--conf",       type=float, default=0.25)
    d.add_argument("--imgsz",      type=int,   default=1280)
    d.add_argument("--dot_radius", type=int,   default=5)

    # --- speed ---
    s = sub.add_parser("speed", help="Speed and sprint tracking")
    s.add_argument("--weights",      required=True)
    s.add_argument("--source",       required=True)
    s.add_argument("--out_dir",      default="out_speed")
    s.add_argument("--conf",         type=float, default=0.25)
    s.add_argument("--imgsz",        type=int,   default=1280)
    s.add_argument("--field_w_m",    type=float, default=68.0,
                   help="Football field width in metres (default: 68)")
    s.add_argument("--field_w_px",   type=float, required=True,
                   help="Football field width in pixels in the source video")
    s.add_argument("--smooth",       type=int,   default=15)

    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "train":
        fa = FootballAnalytics(
            model_path=args.model,
            data_yaml=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
        )
        fa.train()

    elif args.cmd == "detect":
        fa = FootballAnalytics(
            weights=args.weights,
            source=args.source,
            out_dir=args.out_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            dot_radius=args.dot_radius,
        )
        fa.detect()

    elif args.cmd == "speed":
        fa = FootballAnalytics(
            weights=args.weights,
            source=args.source,
            out_dir=args.out_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            field_w_m=args.field_w_m,
            field_w_px=args.field_w_px,
            smooth=args.smooth,
        )
        fa.track()


if __name__ == "__main__":
    main()
