#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import csv
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Radar overlay dimensions
RADAR_W, RADAR_H = 260, 170
HEAT_W, HEAT_H   = 220, 140
PAD = 10
_RADAR_TITLE_H = 18

# Football field proportions (105 m × 68 m → ratio ≈ 1.544)
_FIELD_ASPECT = 105.0 / 68.0


class PlayerDetector:
    def __init__(
        self,
        weights: str = "best.pt",
        source: str = "input.mp4",
        out_dir: str = "out",
        conf: float = 0.25,
        imgsz: int = 1280,
        dot_radius: int = 5,
    ):
        self.weights   = weights
        self.source    = source
        self.out_dir   = Path(out_dir)
        self.conf      = conf
        self.imgsz     = imgsz
        self.dot_radius = dot_radius

    # ------------------------------------------------------------------
    # Field corner selection — 6 points
    # ------------------------------------------------------------------

    @staticmethod
    def select_field_corners(first_frame, timeout_sec=20):
        """
        Click 6 points on the football field in this order:
          1 — top-left corner
          2 — top-center (where halfway line meets top touchline)
          3 — top-right corner
          4 — bottom-right corner
          5 — bottom-center (where halfway line meets bottom touchline)
          6 — bottom-left corner
        """
        LABELS = [
            "1: top-left",
            "2: top-center (mid)",
            "3: top-right",
            "4: bottom-right",
            "5: bottom-center (mid)",
            "6: bottom-left",
        ]
        COLORS = [
            (0, 255, 255),
            (0, 220, 200),
            (0, 180, 255),
            (0, 120, 255),
            (0, 80, 220),
            (0, 40, 200),
        ]
        N = 6

        pts = []
        img = first_frame.copy()
        win = "Select 6 football field points | Esc/Space - skip | R - reset"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        def on_mouse(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < N:
                pts.append((x, y))

        cv2.setMouseCallback(win, on_mouse)
        deadline = time.time() + timeout_sec

        while True:
            remaining = max(0, deadline - time.time())
            disp = img.copy()
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 52), (0, 0, 0), -1)
            cv2.addWeighted(disp, 0.5, img, 0.5, 0, disp)

            if len(pts) < N:
                cv2.putText(disp, f"Click {LABELS[len(pts)]}  ({remaining:.0f}s)",
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                            (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(disp, "Done! Press any key...",
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                            (0, 255, 120), 2, cv2.LINE_AA)

            for i, (px, py) in enumerate(pts):
                cv2.circle(disp, (px, py), 8, COLORS[i], -1, cv2.LINE_AA)
                cv2.circle(disp, (px, py), 8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(disp, str(i + 1), (px + 10, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if len(pts) >= 2:
                # Draw polygon outline while selecting
                cv2.polylines(disp, [np.array(pts, dtype=np.int32)],
                              isClosed=(len(pts) == N),
                              color=(200, 200, 0), thickness=2, lineType=cv2.LINE_AA)

            cv2.imshow(win, disp)
            key = cv2.waitKey(30) & 0xFF

            if key in (27, 32, 13):
                cv2.destroyWindow(win)
                return None
            if key in (ord('r'), ord('R')):
                pts.clear()
            if len(pts) == N and key != 255:
                break
            if remaining == 0:
                print("[!] Timeout — radar running without homography")
                cv2.destroyWindow(win)
                return None

        cv2.destroyWindow(win)
        return np.array(pts, dtype=np.float32)

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
        cv2.polylines(frame, [hull], isClosed=True,
                      color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

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

    @staticmethod
    def _draw_football_radar(frame, centers, frame_w, frame_h, homography):
        """Draw a mini football-field radar with player positions."""
        x0 = frame_w - RADAR_W - PAD
        y0 = PAD
        rx0 = x0 + 2
        ry0 = y0 + _RADAR_TITLE_H
        rw = RADAR_W - 4
        rh = RADAR_H - _RADAR_TITLE_H - 2

        # --- background (green field) ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx0, ry0), (rx0 + rw, ry0 + rh), (34, 100, 34), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        lc = (200, 230, 200)  # line color (light green-white)

        # Outer boundary
        cv2.rectangle(frame, (rx0, ry0), (rx0 + rw, ry0 + rh), lc, 1)

        # Halfway line (vertical center)
        cx_mid = rx0 + rw // 2
        cy_mid = ry0 + rh // 2
        cv2.line(frame, (cx_mid, ry0), (cx_mid, ry0 + rh), lc, 1)

        # Center circle
        r_circle = max(6, int(rh * 0.16))
        cv2.circle(frame, (cx_mid, cy_mid), r_circle, lc, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx_mid, cy_mid), 2, lc, -1)

        # Penalty areas (left and right) — ~16.5 m on a 105 m field ≈ 15.7% of length
        pen_w = max(6, int(rw * 0.157))
        pen_h = max(10, int(rh * 0.588))  # 40 m / 68 m of field height
        pen_y0 = ry0 + (rh - pen_h) // 2
        # Left penalty area
        cv2.rectangle(frame, (rx0, pen_y0), (rx0 + pen_w, pen_y0 + pen_h), lc, 1)
        # Right penalty area
        cv2.rectangle(frame, (rx0 + rw - pen_w, pen_y0), (rx0 + rw, pen_y0 + pen_h), lc, 1)

        # Goal areas (5.5 m / 105 m ≈ 5.2%, height 18.32 m / 68 m ≈ 26.9%)
        goal_w = max(3, int(rw * 0.052))
        goal_h = max(5, int(rh * 0.269))
        goal_y0 = ry0 + (rh - goal_h) // 2
        cv2.rectangle(frame, (rx0, goal_y0), (rx0 + goal_w, goal_y0 + goal_h), lc, 1)
        cv2.rectangle(frame, (rx0 + rw - goal_w, goal_y0), (rx0 + rw, goal_y0 + goal_h), lc, 1)

        # Title bar
        cv2.rectangle(frame, (x0, y0), (x0 + RADAR_W, y0 + RADAR_H), (180, 180, 180), 1)
        cv2.putText(frame, "RADAR", (x0 + 6, y0 + _RADAR_TITLE_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"n={len(centers)}",
                    (x0 + RADAR_W - 46, y0 + _RADAR_TITLE_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

        if not centers:
            return

        # Map player positions onto the radar
        src_pts = np.array(centers, dtype=np.float32).reshape(-1, 1, 2)
        if homography is not None:
            dst = cv2.perspectiveTransform(src_pts, homography)
            radar_pts = []
            for p in dst:
                nx, ny = float(p[0][0]), float(p[0][1])
                px = rx0 + int(np.clip(nx, 0, 1) * rw)
                py = ry0 + int(np.clip(ny, 0, 1) * rh)
                radar_pts.append((px, py))
        else:
            radar_pts = []
            for cx, cy in centers:
                px = rx0 + int(cx / frame_w * rw)
                py = ry0 + int(cy / frame_h * rh)
                px = max(rx0, min(rx0 + rw - 1, px))
                py = max(ry0, min(ry0 + rh - 1, py))
                radar_pts.append((px, py))

        # Spider web on radar
        n = len(radar_pts)
        for i in range(n):
            for j in range(i + 1, n):
                cv2.line(frame, radar_pts[i], radar_pts[j],
                         (0, 200, 255), 1, cv2.LINE_AA)

        # Player dots
        for px, py in radar_pts:
            cv2.circle(frame, (px, py), 4, (0, 60, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_heatmap(frame, heat_acc):
        x0, y0 = PAD, PAD
        h_small = cv2.resize(heat_acc, (HEAT_W, HEAT_H))
        max_val = h_small.max()
        if max_val > 0:
            h_norm = (h_small / max_val * 255).astype(np.uint8)
        else:
            h_norm = np.zeros((HEAT_H, HEAT_W), dtype=np.uint8)
        h_color = cv2.applyColorMap(h_norm, cv2.COLORMAP_JET)
        roi = frame[y0:y0 + HEAT_H, x0:x0 + HEAT_W]
        cv2.addWeighted(h_color, 0.6, roi, 0.4, 0, roi)
        frame[y0:y0 + HEAT_H, x0:x0 + HEAT_W] = roi
        cv2.rectangle(frame, (x0, y0), (x0 + HEAT_W, y0 + HEAT_H), (180, 180, 180), 1)
        cv2.putText(frame, "HEATMAP", (x0 + 6, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

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

        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read first frame")

        # 6-point field selection for homography
        field_corners = self.select_field_corners(first_frame)
        homography = None
        if field_corners is not None:
            # Normalized destination: 6 key points on a [0,1]×[0,1] rectangle
            # Order: top-left, top-center, top-right, bottom-right, bottom-center, bottom-left
            dst_norm = np.array([
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.5, 1.0],
                [0.0, 1.0],
            ], dtype=np.float32)
            homography, _ = cv2.findHomography(field_corners, dst_norm)
            print("[OK] Homography computed from 6 football field points")
        else:
            print("[!] No corners selected — radar using simple normalization")

        out_path = self.out_dir / "result.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        csv_path = self.out_dir / "detections.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["frame", "x1", "y1", "x2", "y2", "cx", "cy", "conf", "cls", "label"])

        heat_acc = np.zeros((h, w), dtype=np.float32)
        blob_r = max(w, h) // 12
        frames_to_process = [first_frame]

        frame_id = 0
        while True:
            if frames_to_process:
                frame = frames_to_process.pop(0)
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            results = model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                    verbose=False, classes=[0])
            rows = []
            centers = []

            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf_val = float(b.conf[0])
                    cls_id = int(b.cls[0])
                    label = r.names.get(cls_id, str(cls_id))
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    self._draw_center_dot(frame, cx, cy, r=self.dot_radius)
                    centers.append((cx, cy))

                    cx_i = max(0, min(w - 1, int(cx)))
                    cy_i = max(0, min(h - 1, int(cy)))
                    cv2.circle(heat_acc, (cx_i, cy_i), blob_r, 1.0, -1)

                    rows.append([frame_id, int(x1), int(y1), int(x2), int(y2),
                                  int(cx), int(cy), round(conf_val, 3), cls_id, label])

            heat_acc *= 0.97
            heat_acc = cv2.GaussianBlur(heat_acc, (0, 0), sigmaX=max(w, h) // 30)

            # Spider web between all detected players
            if len(centers) >= 2:
                self._draw_spider_web(frame, centers)

            if len(centers) >= 3:
                self._draw_convex_hull(frame, centers)

            self._draw_heatmap(frame, heat_acc)
            self._draw_football_radar(frame, centers, w, h, homography)

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
