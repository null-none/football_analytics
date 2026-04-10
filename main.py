#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FootballAnalytics — single entry point for the entire pipeline.

Examples:
    # Model training
    python main.py train --model yolo26s.pt --data data.yaml --epochs 50

    # Player detection (all overlays disabled by default; enable selectively)
    python main.py detect --weights best.pt --source input.mp4
    python main.py detect --weights best.pt --source input.mp4 --show_spider_web --show_convex_hull

    # Simple detection by category (bbox + label only)
    python main.py category --weights best.pt --source input.mp4
    python main.py category --weights best.pt --source input.mp4 --classes 0 1

    # Speed tracking
    python main.py speed --weights best.pt --source input.mp4 --field_w_px 950
"""

import argparse

from trainer import YOLOTrainer
from player_detector import PlayerDetector
from speed_tracker import SpeedTracker
from category_detector import CategoryDetector


class FootballAnalytics(YOLOTrainer, PlayerDetector, SpeedTracker, CategoryDetector):
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
        show_spider_web: bool = False,
        show_convex_hull: bool = False,
        show_defense_line: bool = False,
        defense_n: int = 4,
        show_defense_zone: bool = False,
        show_pressure: bool = False,
        pressure_r: int = 120,
        classes: list = None,
        # category detection
        cat_out_dir: str = "out_category",
        # speed
        field_w_m: float = 68.0,  # Standard football field width
        field_w_px: float = 950.0,
        smooth: int = 15,
        show_frame_id: bool = False,
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
            show_spider_web=show_spider_web,
            show_convex_hull=show_convex_hull,
            show_defense_line=show_defense_line,
            defense_n=defense_n,
            show_defense_zone=show_defense_zone,
            show_pressure=show_pressure,
            pressure_r=pressure_r,
            classes=classes,
        )
        CategoryDetector.__init__(
            self,
            weights=weights,
            source=source,
            out_dir=cat_out_dir,
            conf=conf,
            imgsz=imgsz,
            classes=classes,
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
            classes=classes,
            show_frame_id=show_frame_id,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        description="Football Analytics — YOLO-based player detection and tracking"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- train ---
    t = sub.add_parser("train", help="Train YOLO model")
    t.add_argument("--model", default="yolo26s.pt")
    t.add_argument("--data", default="data.yaml")
    t.add_argument("--imgsz", type=int, default=1280)
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--workers", type=int, default=4)

    # --- detect ---
    d = sub.add_parser("detect", help="Player detection (spider web, convex hull)")
    d.add_argument("--weights", required=True)
    d.add_argument("--source", required=True)
    d.add_argument("--out_dir", default="out")
    d.add_argument("--conf", type=float, default=0.25)
    d.add_argument("--imgsz", type=int, default=1280)
    d.add_argument("--dot_radius", type=int, default=5)
    d.add_argument(
        "--show_spider_web",
        action="store_true",
        default=False,
        help="Show spider-web lines between players (default: off)",
    )
    d.add_argument(
        "--show_convex_hull",
        action="store_true",
        default=False,
        help="Show convex hull around players (default: off)",
    )
    d.add_argument(
        "--show_defense_line",
        action="store_true",
        default=False,
        help="Show defensive lines for both sides (default: off)",
    )
    d.add_argument(
        "--defense_n",
        type=int,
        default=4,
        help="Number of defenders per side for the defensive line (default: 4)",
    )
    d.add_argument(
        "--show_defense_zone",
        action="store_true",
        default=False,
        help="Show vertical defensive zone lines for both sides (default: off)",
    )
    d.add_argument(
        "--show_pressure",
        action="store_true",
        default=False,
        help="Show pressing pressure halos on each player (default: off)",
    )
    d.add_argument(
        "--pressure_r",
        type=int,
        default=120,
        help="Radius in pixels to count opponents as pressing (default: 120)",
    )
    d.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="YOLO class IDs to detect, e.g. --classes 0 1 (default: 0)",
    )

    # --- category ---
    c = sub.add_parser("category", help="Simple detection by category (bbox + label)")
    c.add_argument("--weights", required=True)
    c.add_argument("--source", required=True)
    c.add_argument("--out_dir", default="out_category")
    c.add_argument("--conf", type=float, default=0.25)
    c.add_argument("--imgsz", type=int, default=1280)
    c.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="YOLO class IDs to detect (default: all classes)",
    )

    # --- speed ---
    s = sub.add_parser("speed", help="Speed and sprint tracking")
    s.add_argument("--weights", required=True)
    s.add_argument("--source", required=True)
    s.add_argument("--out_dir", default="out_speed")
    s.add_argument("--conf", type=float, default=0.25)
    s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument(
        "--field_w_m",
        type=float,
        default=68.0,
        help="Football field width in metres (default: 68)",
    )
    s.add_argument(
        "--field_w_px",
        type=float,
        required=True,
        help="Football field width in pixels in the source video",
    )
    s.add_argument("--smooth", type=int, default=15)
    s.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="YOLO class IDs to track, e.g. --classes 1 (default: 1)",
    )
    s.add_argument(
        "--show_frame_id",
        action="store_true",
        default=False,
        help="Show current frame number on each output frame (default: off)",
    )

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

    elif args.cmd == "category":
        fa = FootballAnalytics(
            weights=args.weights,
            source=args.source,
            cat_out_dir=args.out_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            classes=args.classes,
        )
        fa.run()

    elif args.cmd == "detect":
        fa = FootballAnalytics(
            weights=args.weights,
            source=args.source,
            out_dir=args.out_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            dot_radius=args.dot_radius,
            show_spider_web=args.show_spider_web,
            show_convex_hull=args.show_convex_hull,
            show_defense_line=args.show_defense_line,
            defense_n=args.defense_n,
            show_defense_zone=args.show_defense_zone,
            show_pressure=args.show_pressure,
            pressure_r=args.pressure_r,
            classes=args.classes,
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
            classes=args.classes,
            show_frame_id=args.show_frame_id,
        )
        fa.track()


if __name__ == "__main__":
    main()
