#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO

# Sprint threshold for football: >5.5 m/s (~20 km/h high-intensity running)
SPRINT_THRESHOLD_MS = 5.5


class SpeedTracker:
    def __init__(
        self,
        weights: str = "best.pt",
        source: str = "input.mp4",
        out_dir: str = "out_speed",
        conf: float = 0.15,
        imgsz: int = 1280,
        field_w_m: float = 68.0,  # Standard football field width, metres
        field_w_px: float = 950.0,
        smooth: int = 15,
        classes: list = None,
        class_colors: dict = None,
        sprint_color: tuple = (0, 0, 220),
        show_frame_id: bool = False,
    ):
        self.weights = weights
        self.source = source
        self.out_dir = Path(out_dir)
        self.conf = conf
        self.imgsz = imgsz
        self.field_w_m = field_w_m
        self.field_w_px = field_w_px
        self.smooth = smooth
        self.classes = classes if classes is not None else [0]
        self.class_colors = (
            class_colors
            if class_colors is not None
            else {
                0: (255, 0, 128),
                1: (0, 220, 100),
                2: (220, 180, 0),
                3: (180, 0, 220),
            }
        )
        self.sprint_color = sprint_color
        self.show_frame_id = show_frame_id

        # Runtime state (reset on each run)
        self._history = None
        self._max_speeds = None
        self._cur_speeds = None
        self._total_dist = None
        self._sprint_dist = None
        self._sprint_active = None
        self._sprint_durations = None
        self._last_pos = None
        self._class_total_dist = None
        self._class_sprint_dist = None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _get_class_color(self, cls_id: int):
        return self.class_colors.get(cls_id, (180, 180, 180))

    def _draw_top_overlay(self, frame):
        h, w = frame.shape[:2]
        bar_h = 82

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.line(frame, (0, bar_h), (w, bar_h), (60, 60, 60), 1)

        classes_sorted = sorted(self._class_total_dist.keys())
        if not classes_sorted:
            total_dist_m = sum(self._total_dist.values())
            total_km = total_dist_m / 1000.0
            label = f"TOTAL DISTANCE  {total_dist_m:,.0f} m / {total_km:.3f} km"
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
            (lw, lh), _ = cv2.getTextSize(label, font, scale, thick)
            tx = (w - lw) // 2
            ty = 50
            cv2.putText(
                frame, label, (tx + 1, ty + 1), font, scale, (0, 0, 0), thick + 1
            )
            cv2.putText(frame, label, (tx, ty), font, scale, (220, 220, 220), thick)
            return

        margin_x = 14
        gap = 12
        block_h = 50
        block_y = 16
        available_w = w - 2 * margin_x - gap * (len(classes_sorted) - 1)
        block_w = max(180, available_w // max(1, len(classes_sorted)))

        x = margin_x
        for cls_id in classes_sorted:
            color = self._get_class_color(cls_id)

            total_m = self._class_total_dist[cls_id]
            total_km = total_m / 1000.0
            sprint_m = self._class_sprint_dist[cls_id]

            cv2.rectangle(
                frame, (x, block_y), (x + block_w, block_y + block_h), (35, 35, 35), -1
            )
            cv2.rectangle(
                frame, (x, block_y), (x + block_w, block_y + block_h), color, 2
            )

            title = f"TEAM {cls_id}"
            line2 = f"{total_m:,.0f} m / {total_km:.3f} km"
            line3 = f"Sprint: {sprint_m:,.0f} m"

            cv2.putText(
                frame,
                title,
                (x + 10, block_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line2,
                (x + 10, block_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line3,
                (x + 10, block_y + 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            x += block_w + gap

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def track(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        scale = self.field_w_m / self.field_w_px

        model = YOLO(self.weights)

        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {src}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = self.out_dir / "result_speed.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        csv_path = self.out_dir / "speeds.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "frame",
                    "player_id",
                    "class_id",
                    "cx",
                    "cy",
                    "speed_ms",
                    "max_speed_ms",
                ]
            )

        self._history = defaultdict(lambda: deque(maxlen=self.smooth + 1))
        self._max_speeds = defaultdict(float)
        self._cur_speeds = {}
        self._total_dist = defaultdict(float)
        self._sprint_dist = defaultdict(float)
        self._sprint_active = {}
        self._sprint_durations = []
        self._last_pos = {}
        self._class_total_dist = defaultdict(float)
        self._class_sprint_dist = defaultdict(float)

        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.track(
                frame,
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
                classes=self.classes,
                persist=True,
            )

            csv_rows = []
            for r in results:
                if r.boxes.id is None:
                    continue
                if r.boxes.cls is None:
                    continue

                for b, pid, cls_id in zip(
                    r.boxes, r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()
                ):
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    self._history[pid].append((cx, cy))
                    hist = list(self._history[pid])

                    if len(hist) >= 2:
                        segment_speeds = []
                        for i in range(1, len(hist)):
                            dx = (hist[i][0] - hist[i - 1][0]) * scale
                            dy = (hist[i][1] - hist[i - 1][1]) * scale
                            segment_speeds.append(np.sqrt(dx**2 + dy**2) * fps)
                        speed = float(np.mean(segment_speeds))
                    else:
                        speed = 0.0

                    self._cur_speeds[pid] = speed
                    if speed > self._max_speeds[pid]:
                        self._max_speeds[pid] = speed

                    if pid in self._last_pos:
                        px, py = self._last_pos[pid]
                        step = np.sqrt((cx - px) ** 2 + (cy - py) ** 2) * scale

                        self._total_dist[pid] += step
                        self._class_total_dist[cls_id] += step

                        if speed > SPRINT_THRESHOLD_MS:
                            self._sprint_dist[pid] += step
                            self._class_sprint_dist[cls_id] += step

                    self._last_pos[pid] = (cx, cy)

                    if speed > SPRINT_THRESHOLD_MS:
                        if pid not in self._sprint_active:
                            self._sprint_active[pid] = frame_id
                    else:
                        if pid in self._sprint_active:
                            duration = (frame_id - self._sprint_active.pop(pid)) / fps
                            self._sprint_durations.append(duration)

                    is_sprint = speed > SPRINT_THRESHOLD_MS
                    base_color = self._get_class_color(cls_id)
                    color = self.sprint_color if is_sprint else base_color

                    box_w = x2 - x1
                    rx = max(int(box_w * 0.45), 20)
                    ry = max(int(rx * 0.28), 7)
                    shadow_center = (int(cx), y2)

                    overlay = frame.copy()
                    cv2.ellipse(overlay, shadow_center, (rx, ry), 0, 0, 180, color, -1)
                    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                    cv2.ellipse(frame, shadow_center, (rx, ry), 0, 0, 180, color, 2)

                    label = f"{speed:.1f} m/s"
                    (lw, lh), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
                    )
                    y_label_top = max(0, y1 - lh - 10)
                    cv2.rectangle(
                        frame, (x1, y_label_top), (x1 + lw + 4, y1), color, -1
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 0, 0),
                        2,
                    )

                    csv_rows.append(
                        [
                            frame_id,
                            pid,
                            cls_id,
                            int(cx),
                            int(cy),
                            round(speed, 3),
                            round(self._max_speeds[pid], 3),
                        ]
                    )

            self._draw_top_overlay(frame)

            if self.show_frame_id:
                fh, fw = frame.shape[:2]
                label = f"frame {frame_id}"
                cv2.putText(
                    frame,
                    label,
                    (fw - 160, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    label,
                    (fw - 160, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(frame)

            if csv_rows:
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(csv_rows)

            cv2.imshow("Football Analytics — Speed Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_id += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        for pid, start_frame in self._sprint_active.items():
            duration = (frame_id - start_frame) / fps
            self._sprint_durations.append(duration)

        return self._build_summary(out_path, csv_path)

    def _build_summary(self, out_path, csv_path):
        total_km = sum(self._total_dist.values()) / 1000.0
        sprint_km = sum(self._sprint_dist.values()) / 1000.0
        max_sprint = max(self._sprint_durations, default=0.0)
        sprints_5s = sum(1 for d in self._sprint_durations if d >= 5)

        lines = [
            "=== FOOTBALL ANALYTICS — SUMMARY ===",
            f"Total distance run:                        {total_km:.3f} km",
            f"Distance in sprint (>{SPRINT_THRESHOLD_MS} m/s):  {sprint_km:.3f} km",
            f"Max sprint duration:                       {max_sprint:.1f} s",
            f"Sprints >= 5 sec:                          {sprints_5s}",
            "",
            "=== DISTANCE BY CLASS ===",
        ]

        for cls_id in sorted(self._class_total_dist.keys()):
            lines.append(
                f"Class {cls_id}: total={self._class_total_dist[cls_id]/1000.0:.3f} km, "
                f"sprint={self._class_sprint_dist[cls_id]/1000.0:.3f} km"
            )

        summary_path = self.out_dir / "summary.txt"
        summary_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"[OK] Video:   {out_path}")
        print(f"[OK] CSV:     {csv_path}")
        print()
        for line in lines:
            print(line)
        print(f"[OK] Summary: {summary_path}")

        return lines
