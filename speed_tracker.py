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


# ---------------------------------------------------------------------------
# Field calibration — 6-point perspective + fisheye detection
# ---------------------------------------------------------------------------


class FieldCalibrator:
    """
    Interactive tool: user clicks 6 field landmarks in a fixed order.

    Click order (matches the numbered labels shown on screen):
        1 = Top-Left      2 = Bottom-Left   3 = Bottom-Center
        4 = Bottom-Right  5 = Top-Right     6 = Top-Center

    Layout:
        TL(1) ─── TC(6) ─── TR(5)
          │                   │
        BL(2) ─── BC(3) ─── BR(4)

    After 6 clicks the calibrator:
      • fits a homography  pixel → metres
      • checks collinearity of top/bottom edge midpoints to detect
        fisheye distortion (a curved edge means its midpoint deviates
        from the straight chord between the two corner points)
    """

    LABELS = [
        "1 Top-Left",
        "2 Bottom-Left",
        "3 Bottom-Center",
        "4 Bottom-Right",
        "5 Top-Right",
        "6 Top-Center",
    ]
    COLORS = [
        (0, 255, 0),  # TL  – green
        (255, 128, 0),  # BL  – orange
        (0, 200, 255),  # BC  – cyan
        (255, 0, 128),  # BR  – pink
        (128, 0, 255),  # TR  – violet
        (255, 255, 0),  # TC  – yellow
    ]
    # Midpoint deviation > 1.5% of image diagonal → fisheye
    FISHEYE_THRESHOLD = 0.015

    def __init__(self, frame: np.ndarray, field_w_m: float, field_h_m: float):
        self.frame = frame.copy()
        self.field_w_m = field_w_m
        self.field_h_m = field_h_m
        self.pts_px: list[tuple[int, int]] = []
        self._win = "Field Calibration — click 6 points  |  ESC = undo  |  Q = confirm"

    # ------------------------------------------------------------------
    # Real-world coordinates for the 6 landmarks (x across, y down field)
    # ------------------------------------------------------------------

    def _real_pts(self) -> np.ndarray:
        W, H = self.field_w_m, self.field_h_m
        return np.array(
            [
                [0, 0],  # TL
                [0, H],  # BL
                [W / 2, H],  # BC
                [W, H],  # BR
                [W, 0],  # TR
                [W / 2, 0],  # TC
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _on_click(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.pts_px) < 6:
            self.pts_px.append((x, y))

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_guide(self, base: np.ndarray) -> np.ndarray:
        """Draws a small field-layout diagram in the top-right corner."""
        h, w = base.shape[:2]
        margin = 12
        gw, gh = 200, 90
        gx, gy = w - gw - margin, margin

        cv2.rectangle(
            base, (gx - 4, gy - 4), (gx + gw + 4, gy + gh + 4), (20, 20, 20), -1
        )
        cv2.rectangle(base, (gx, gy), (gx + gw, gy + gh), (60, 60, 60), 2)

        positions = [
            (gx, gy),  # 0 TL
            (gx, gy + gh),  # 1 BL
            (gx + gw // 2, gy + gh),  # 2 BC
            (gx + gw, gy + gh),  # 3 BR
            (gx + gw, gy),  # 4 TR
            (gx + gw // 2, gy),  # 5 TC
        ]
        for idx, pos in enumerate(positions):
            color = self.COLORS[idx]
            filled = idx < len(self.pts_px)
            cv2.circle(base, pos, 7, color, -1 if filled else 2)
            short = self.LABELS[idx].split()[-1]  # e.g. "Top-Left"
            cv2.putText(
                base,
                short,
                (pos[0] + 5, pos[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )
        return base

    def _draw_state(self) -> np.ndarray:
        img = self.frame.copy()
        self._draw_guide(img)

        # Already placed points
        for idx, pt in enumerate(self.pts_px):
            color = self.COLORS[idx]
            cv2.circle(img, pt, 8, color, -1)
            cv2.circle(img, pt, 10, (255, 255, 255), 1)
            cv2.putText(
                img,
                self.LABELS[idx],
                (pt[0] + 12, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        # Instruction for the next point
        if len(self.pts_px) < 6:
            next_label = self.LABELS[len(self.pts_px)]
            next_color = self.COLORS[len(self.pts_px)]
            msg = f"Click: {next_label}"
            cv2.putText(
                img,
                msg,
                (14, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                msg,
                (14, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                next_color,
                2,
                cv2.LINE_AA,
            )
        else:
            msg = "All 6 points set — press Q to confirm"
            cv2.putText(
                img,
                msg,
                (14, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                msg,
                (14, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 100),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            img,
            "ESC = undo last point",
            (14, img.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        return img

    # ------------------------------------------------------------------
    # Fisheye detection
    # ------------------------------------------------------------------

    def _check_fisheye(self) -> bool:
        """
        In a standard pinhole-camera image, the top edge (TL → TR) and
        bottom edge (BL → BR) are straight lines.  A fisheye lens bends them,
        so their midpoints (TC, BC) deviate from the chord.
        Returns True if the deviation exceeds FISHEYE_THRESHOLD.
        """
        pts = np.array(self.pts_px, dtype=np.float64)
        h, w = self.frame.shape[:2]
        diag = np.sqrt(w**2 + h**2)

        def _chord_deviation(
            p_start: np.ndarray, p_end: np.ndarray, p_mid: np.ndarray
        ) -> float:
            d = p_end - p_start
            length = np.linalg.norm(d)
            if length < 1e-6:
                return 0.0
            normal = np.array([-d[1], d[0]]) / length
            return abs(float(np.dot(p_mid - p_start, normal)))

        # top edge:    TL(0) → TR(4),  midpoint TC(5)
        top_dev = _chord_deviation(pts[0], pts[4], pts[5])
        # bottom edge: BL(1) → BR(3),  midpoint BC(2)
        bot_dev = _chord_deviation(pts[1], pts[3], pts[2])

        max_rel = max(top_dev, bot_dev) / diag
        if max_rel > self.FISHEYE_THRESHOLD:
            print(
                f"[Calibration] Fisheye detected — max midpoint deviation "
                f"{max_rel * 100:.2f}% of diagonal "
                f"(threshold {self.FISHEYE_THRESHOLD * 100:.1f}%)"
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self) -> tuple[np.ndarray, bool]:
        """
        Open a window, collect up to 6 clicks, return (homography, is_fisheye).
        The homography maps (px, py) → (x_m, y_m) in field coordinates.
        Requires at least 4 points; fisheye check requires all 6.
        """
        cv2.namedWindow(self._win)
        cv2.setMouseCallback(self._win, self._on_click)

        while True:
            cv2.imshow(self._win, self._draw_state())
            key = cv2.waitKey(20) & 0xFF
            if key == 27 and self.pts_px:  # ESC → undo
                self.pts_px.pop()
            elif key in (ord("q"), ord("Q")) and len(self.pts_px) >= 4:
                break

        cv2.destroyWindow(self._win)

        n = len(self.pts_px)
        src = np.array(self.pts_px, dtype=np.float64)
        dst = self._real_pts()[:n]

        # findHomography with RANSAC; with 6 points it uses the overdetermined LS fit
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            raise RuntimeError(
                "Could not compute homography — check that calibration points "
                "are not collinear and cover enough of the field."
            )

        is_fisheye = self._check_fisheye() if n == 6 else False
        if is_fisheye:
            print(
                "[Calibration] WARNING: fisheye lens detected. "
                "Speed values may be less accurate without lens correction."
            )
        else:
            print("[Calibration] Standard (non-fisheye) camera confirmed.")

        return H, is_fisheye


# ---------------------------------------------------------------------------
# Speed tracker
# ---------------------------------------------------------------------------


class SpeedTracker:
    def __init__(
        self,
        weights: str = "best.pt",
        source: str = "input.mp4",
        out_dir: str = "out_speed",
        conf: float = 0.15,
        imgsz: int = 1280,
        field_w_m: float = 68.0,  # field width in metres (across)
        field_h_m: float = 105.0,  # field length in metres (along); used for calibration
        field_w_px: float = 950.0,  # fallback pixel width when calibrate=False
        calibrate: bool = False,  # open interactive 6-point calibration window
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
        self.field_h_m = field_h_m
        self.field_w_px = field_w_px
        self.calibrate = calibrate
        self.smooth = smooth
        self.classes = classes if classes is not None else [0]
        self.class_colors = class_colors or {
            0: (255, 0, 128),
            1: (0, 220, 100),
            2: (220, 180, 0),
            3: (180, 0, 220),
        }
        self.sprint_color = sprint_color
        self.show_frame_id = show_frame_id

        # Set during track(); None → use simple scale fallback
        self._homography: np.ndarray | None = None

        # Runtime state — reset at the start of each track() call
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
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _px_to_real(self, px: float, py: float) -> tuple[float, float]:
        """Convert a pixel position to real-world metres.

        Uses the calibrated homography when available; otherwise falls back
        to a uniform scale derived from field_w_m / field_w_px.
        """
        if self._homography is not None:
            pt = np.array([[[px, py]]], dtype=np.float64)
            out = cv2.perspectiveTransform(pt, self._homography)
            return float(out[0, 0, 0]), float(out[0, 0, 1])
        scale = self.field_w_m / self.field_w_px
        return px * scale, py * scale

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _get_class_color(self, cls_id: int) -> tuple:
        return self.class_colors.get(cls_id, (180, 180, 180))

    def _draw_top_overlay(self, frame: np.ndarray) -> None:
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
            text = f"TOTAL DISTANCE  {total_dist_m:,.0f} m / {total_km:.3f} km"
            font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
            (tw, _), _ = cv2.getTextSize(text, font, fscale, thick)
            tx, ty = (w - tw) // 2, 50
            cv2.putText(
                frame, text, (tx + 1, ty + 1), font, fscale, (0, 0, 0), thick + 1
            )
            cv2.putText(frame, text, (tx, ty), font, fscale, (220, 220, 220), thick)
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
            sprint_m = self._class_sprint_dist[cls_id]

            cv2.rectangle(
                frame, (x, block_y), (x + block_w, block_y + block_h), (35, 35, 35), -1
            )
            cv2.rectangle(
                frame, (x, block_y), (x + block_w, block_y + block_h), color, 2
            )

            cv2.putText(
                frame,
                f"TEAM {cls_id}",
                (x + 10, block_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{total_m:,.0f} m / {total_m/1000:.3f} km",
                (x + 10, block_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Sprint: {sprint_m:,.0f} m",
                (x + 10, block_y + 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            x += block_w + gap

    def _draw_player(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        cx: float,
        speed: float,
        color: tuple,
    ) -> None:
        # Shadow ellipse under the player's feet
        box_w = x2 - x1
        e_rx = max(int(box_w * 0.45), 20)
        e_ry = max(int(e_rx * 0.28), 7)
        foot = (int(cx), y2)

        shadow = frame.copy()
        cv2.ellipse(shadow, foot, (e_rx, e_ry), 0, 0, 180, color, -1)
        cv2.addWeighted(shadow, 0.35, frame, 0.65, 0, frame)
        cv2.ellipse(frame, foot, (e_rx, e_ry), 0, 0, 180, color, 2)

        # Speed label above the bounding box
        label = f"{speed:.1f} m/s"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        y_top = max(0, y1 - lh - 10)
        cv2.rectangle(frame, (x1, y_top), (x1 + lw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2
        )

    def _draw_frame_id(self, frame: np.ndarray, frame_id: int) -> None:
        fh, fw = frame.shape[:2]
        text = f"frame {frame_id}"
        pos = (fw - 160, fh - 10)
        cv2.putText(
            frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Main tracking loop
    # ------------------------------------------------------------------

    def track(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(self.weights)

        video_src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {video_src}")

        # --- Optional interactive 6-point calibration ---
        if self.calibrate:
            ok, first_frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read first frame for calibration")
            calibrator = FieldCalibrator(first_frame, self.field_w_m, self.field_h_m)
            self._homography, _ = calibrator.run()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self._homography = None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = self.out_dir / "result_speed.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h)
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
                if r.boxes.id is None or r.boxes.cls is None:
                    continue

                for box, pid, cls_id in zip(
                    r.boxes,
                    r.boxes.id.int().tolist(),
                    r.boxes.cls.int().tolist(),
                ):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    # --- Speed (smoothed over history window) ---
                    self._history[pid].append((cx, cy))
                    hist = list(self._history[pid])

                    if len(hist) >= 2:
                        speeds = []
                        for i in range(1, len(hist)):
                            rx0, ry0 = self._px_to_real(hist[i - 1][0], hist[i - 1][1])
                            rx1, ry1 = self._px_to_real(hist[i][0], hist[i][1])
                            speeds.append(
                                np.sqrt((rx1 - rx0) ** 2 + (ry1 - ry0) ** 2) * fps
                            )
                        speed = float(np.mean(speeds))
                    else:
                        speed = 0.0

                    self._cur_speeds[pid] = speed
                    if speed > self._max_speeds[pid]:
                        self._max_speeds[pid] = speed

                    # --- Cumulative distance ---
                    if pid in self._last_pos:
                        lx, ly = self._last_pos[pid]
                        rx0, ry0 = self._px_to_real(lx, ly)
                        rx1, ry1 = self._px_to_real(cx, cy)
                        step = np.sqrt((rx1 - rx0) ** 2 + (ry1 - ry0) ** 2)

                        self._total_dist[pid] += step
                        self._class_total_dist[cls_id] += step
                        if speed > SPRINT_THRESHOLD_MS:
                            self._sprint_dist[pid] += step
                            self._class_sprint_dist[cls_id] += step

                    self._last_pos[pid] = (cx, cy)

                    # --- Sprint state machine ---
                    is_sprint = speed > SPRINT_THRESHOLD_MS
                    if is_sprint:
                        self._sprint_active.setdefault(pid, frame_id)
                    elif pid in self._sprint_active:
                        duration = (frame_id - self._sprint_active.pop(pid)) / fps
                        self._sprint_durations.append(duration)

                    # --- Draw ---
                    base_color = self._get_class_color(cls_id)
                    draw_color = self.sprint_color if is_sprint else base_color
                    self._draw_player(frame, x1, y1, x2, y2, cx, speed, draw_color)

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
                self._draw_frame_id(frame, frame_id)

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

        # Flush any still-active sprints at end of video
        for pid, start_frame in self._sprint_active.items():
            self._sprint_durations.append((frame_id - start_frame) / fps)

        return self._build_summary(out_path, csv_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(self, out_path: Path, csv_path: Path) -> list[str]:
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
                f"Class {cls_id}: "
                f"total={self._class_total_dist[cls_id]/1000:.3f} km, "
                f"sprint={self._class_sprint_dist[cls_id]/1000:.3f} km"
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
