#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import cv2
from pathlib import Path
from ultralytics import YOLO


class CategoryDetector:
    """
    Simple YOLO-based detector that draws bounding boxes with class labels.
    No overlays, no heatmaps — just raw detections per category.
    """


    _COLORS = [
        (0, 200, 255),
        (0, 255, 100),
        (255, 100, 0),
        (200, 0, 255),
        (0, 160, 255),
        (255, 200, 0),
        (0, 255, 200),
        (180, 0, 180),
    ]


    def __init__(
        self,
        weights: str = "best.pt",
        source: str = "input.mp4",
        out_dir: str = "out_category",
        conf: float = 0.1,
        imgsz: int = 1280,
        classes: list = None,
    ):
        self.weights = weights
        self.source  = source
        self.out_dir = Path(out_dir)
        self.conf    = conf
        self.imgsz   = imgsz
        self.classes = classes  # None → detect all classes

    def _class_color(self, cls_id: int) -> tuple:
        return CategoryDetector._COLORS[cls_id % len(CategoryDetector._COLORS)]

    def run(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(self.weights)

        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = self.out_dir / "result_category.mp4"
        writer   = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        csv_path = self.out_dir / "detections_category.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["frame", "x1", "y1", "x2", "y2", "conf", "cls_id", "label"])

        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                frame, imgsz=self.imgsz, conf=self.conf,
                verbose=False, classes=self.classes,
            )

            rows = []
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    conf_val = float(b.conf[0])
                    cls_id   = int(b.cls[0])
                    label    = r.names.get(cls_id, str(cls_id))
                    color    = self._class_color(cls_id)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                    text = f"{label} {conf_val:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

                    rows.append([frame_id, x1, y1, x2, y2,
                                 round(conf_val, 3), cls_id, label])

            if rows:
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(rows)

            writer.write(frame)
            cv2.imshow("Football Analytics — Category Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_id += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"[OK] Video: {out_path}")
        print(f"[OK] CSV:   {csv_path}")
