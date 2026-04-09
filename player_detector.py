#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class PlayerDetector:
    """
    Detects players in a video using a YOLO model and renders configurable overlays.

    Overlay flags (all disabled by default):
        show_spider_web  — lines connecting every pair of detected players.
        show_convex_hull — filled convex hull around all detected players.

    Class filter:
        classes — list of YOLO class IDs to detect (default: [0]).
                  Use None to detect all classes.
                  Check your model's class IDs with: model.names
    """

    def __init__(
        self,
        weights: str = "best.pt",
        source: str = "input.mp4",
        out_dir: str = "out",
        conf: float = 0.25,
        imgsz: int = 1280,
        dot_radius: int = 5,
        show_spider_web: bool = False,
        show_convex_hull: bool = False,
        show_defense_line: bool = False,
        defense_n: int = 4,
        show_defense_zone: bool = False,
        classes: list = None,
    ):
        self.weights = weights
        self.source = source
        self.out_dir = Path(out_dir)
        self.conf = conf
        self.imgsz = imgsz
        self.dot_radius = dot_radius
        self.show_spider_web = show_spider_web
        self.show_convex_hull = show_convex_hull
        self.show_defense_line = show_defense_line
        self.defense_n = defense_n
        self.show_defense_zone = show_defense_zone
        # smoothing state: {cls_id: {'pts': np.array, 'ttl': int}}
        self._dzone_state = {}
        self.classes = classes if classes is not None else [0]

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_center_dot(frame, cx, cy, r=5):
        h_img, w_img = frame.shape[:2]
        cx_i = max(0, min(w_img - 1, int(cx)))
        cy_i = max(0, min(h_img - 1, int(cy)))
        cv2.circle(frame, (cx_i, cy_i), int(r), (0, 0, 255), -1, cv2.LINE_AA)

    @staticmethod
    def _draw_convex_hull(frame, points):
        if len(points) < 3:
            return
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [hull], (0, 0, 255))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(
            frame,
            [hull],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    @staticmethod
    def _draw_defense_lines(frame, centers, n):
        """
        Draw the defensive line through all n defenders sorted left→right by X.
        Each point keeps its real Y, so the line accurately reflects how deep
        each defender is positioned — not a straight average.
        """
        if len(centers) < n:
            return

        frame_w = frame.shape[1]
        sorted_centers = sorted(centers, key=lambda p: p[1])

        for group, color in (
            (sorted_centers[:n],  (0, 220, 255)),   # top side — yellow
            (sorted_centers[-n:], (255, 160, 0)),   # bottom side — blue
        ):
            # Sort left → right by X
            pts = sorted(group, key=lambda p: p[0])
            arr = np.array([(int(x), int(y)) for x, y in pts], dtype=np.int32)

            # Extended ghost line: extrapolate from leftmost and rightmost to frame edges
            (lx, ly), (rx, ry) = arr[0], arr[-1]
            if rx != lx:
                slope = (ry - ly) / (rx - lx)
                y_edge_l = int(ly - slope * lx)
                y_edge_r = int(ry + slope * (frame_w - rx))
            else:
                y_edge_l = y_edge_r = int((ly + ry) / 2)

            overlay = frame.copy()
            cv2.line(overlay, (0, y_edge_l), (lx, ly), color, 1, cv2.LINE_AA)
            cv2.line(overlay, (rx, ry), (frame_w, y_edge_r), color, 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            # Main polyline through all defender points
            cv2.polylines(frame, [arr], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Dot on each defender
            for x, y in arr:
                cv2.circle(frame, (x, y), 6, color, -1, cv2.LINE_AA)

            mid_x = int(arr[:, 0].mean())
            mid_y = int(arr[:, 1].mean())
            cv2.putText(
                frame, "DEF",
                (mid_x - 18, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
            )

    @staticmethod
    def _draw_defense_zone(frame, centers_by_cls, n, state):
        """
        For each detected team draw a vertical defensive line through the last n defenders.

        The left/right side is determined automatically by average X — no hardcoded class IDs.
        Positions are smoothed with EMA and fade out gradually when detections disappear.
        """
        _ALPHA      = 0.35   # EMA weight for new positions (lower = smoother)
        _FADE_FRAMES = 20    # frames to fully fade out after losing detections
        _COLORS = [(0, 220, 255), (255, 160, 0), (100, 255, 100), (255, 80, 200)]

        # Assign colors and determine which side each class belongs to
        avg_x = {cls_id: float(np.mean([p[0] for p in pts]))
                 for cls_id, pts in centers_by_cls.items() if pts}
        sorted_cls = sorted(avg_x, key=avg_x.get)  # left → right by average X

        for color_idx, cls_id in enumerate(sorted_cls):
            color = _COLORS[color_idx % len(_COLORS)]
            pts = centers_by_cls.get(cls_id, [])

            # Determine side: left-side teams defend on the left (pick lowest X defenders)
            is_left = (color_idx < len(sorted_cls) / 2)

            if len(pts) >= 2:
                # Pick last n defenders by X
                defenders_raw = sorted(pts, key=lambda p: p[0], reverse=not is_left)[:n]
                # Sort by Y for a clean vertical polyline
                defenders_raw = sorted(defenders_raw, key=lambda p: p[1])
                new_pts = np.array([[p[0], p[1]] for p in defenders_raw], dtype=np.float32)

                s = state.get(cls_id)
                if s is None or len(s['pts']) != len(new_pts):
                    # First detection or count changed — reset
                    state[cls_id] = {'pts': new_pts.copy(), 'ttl': _FADE_FRAMES}
                else:
                    # EMA smooth
                    s['pts'] = _ALPHA * new_pts + (1 - _ALPHA) * s['pts']
                    s['ttl'] = _FADE_FRAMES
            else:
                # No detection this frame — decay ttl
                s = state.get(cls_id)
                if s is None:
                    continue
                s['ttl'] = max(0, s['ttl'] - 1)
                if s['ttl'] == 0:
                    continue

            s = state[cls_id]
            opacity = s['ttl'] / _FADE_FRAMES
            arr = s['pts'].astype(np.int32)

            overlay = frame.copy()
            cv2.polylines(overlay, [arr], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
            for x, y in arr:
                cv2.circle(overlay, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    @staticmethod
    def _draw_spider_web(frame, centers):
        """Draw lines between every pair of detected players (spider-web effect)."""
        n = len(centers)
        if n < 2:
            return
        for i in range(n):
            for j in range(i + 1, n):
                p1 = (int(centers[i][0]), int(centers[i][1]))
                p2 = (int(centers[j][0]), int(centers[j][1]))
                # Compute distance for color intensity (closer = brighter)
                dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                alpha = max(0.08, min(0.55, 1.0 - dist / 1200.0))
                overlay = frame.copy()
                cv2.line(overlay, p1, p2, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def detect(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(self.weights)

        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = self.out_dir / "result.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

        csv_path = self.out_dir / "detections.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["frame", "x1", "y1", "x2", "y2", "cx", "cy", "conf", "cls", "label"]
            )

        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
                classes=self.classes,
            )
            rows = []
            centers = []
            centers_by_cls = {}

            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf_val = float(b.conf[0])
                    cls_id = int(b.cls[0])
                    label = r.names.get(cls_id, str(cls_id))
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    if not self.show_defense_zone:
                        self._draw_center_dot(frame, cx, cy, r=self.dot_radius)
                    centers.append((cx, cy))
                    centers_by_cls.setdefault(cls_id, []).append((cx, cy))

                    rows.append(
                        [
                            frame_id,
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                            int(cx),
                            int(cy),
                            round(conf_val, 3),
                            cls_id,
                            label,
                        ]
                    )

            if self.show_defense_line:
                self._draw_defense_lines(frame, centers, self.defense_n)

            if self.show_defense_zone:
                self._draw_defense_zone(frame, centers_by_cls, self.defense_n, self._dzone_state)

            if self.show_spider_web and len(centers) >= 2:
                self._draw_spider_web(frame, centers)

            if self.show_convex_hull and len(centers) >= 3:
                self._draw_convex_hull(frame, centers)

            if rows:
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(rows)

            writer.write(frame)
            cv2.imshow("Football Analytics — Player Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_id += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"[OK] Video: {out_path}")
        print(f"[OK] CSV:   {csv_path}")
