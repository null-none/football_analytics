#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class PressureAnalyzer:
    """
    Overlay that visualises pressing pressure on each player.

    For every player of team A, counts how many opponents (team B) are within
    `pressure_r` pixels and renders a colour-coded halo:
        0 opponents  → no halo
        1 opponent   → yellow
        2 opponents  → orange
        3+ opponents → red

    Works with `centers_by_cls` produced by PlayerDetector, so it requires
    at least two distinct class IDs to be detected (--classes 0 1).
    """

    # pressure level → BGR colour
    _PRESSURE_COLORS = {
        1: (0, 220, 255),   # yellow
        2: (0, 140, 255),   # orange
        3: (0, 60,  255),   # red
    }

    @staticmethod
    def draw_pressure(frame, centers_by_cls, pressure_r: int = 120):
        """
        Draw pressure halos on all detected players.

        Parameters
        ----------
        frame          : current video frame (modified in-place)
        centers_by_cls : dict {cls_id: [(cx, cy), ...]} — output of PlayerDetector
        pressure_r     : radius in pixels within which opponents count as pressing
        """
        cls_ids = list(centers_by_cls.keys())
        if len(cls_ids) < 2:
            return

        # Determine opponent dict for each class
        # Team with lower average X → left side; opponent = the other class.
        # Works for any two class IDs, not just 0 and 1.
        avg_x = {c: float(np.mean([p[0] for p in pts]))
                 for c, pts in centers_by_cls.items() if pts}
        sorted_cls = sorted(avg_x, key=avg_x.get)   # [left_cls, right_cls]

        opponents = {
            sorted_cls[0]: centers_by_cls.get(sorted_cls[1], []),
            sorted_cls[1]: centers_by_cls.get(sorted_cls[0], []),
        }

        for cls_id, pts in centers_by_cls.items():
            opp_pts = opponents.get(cls_id, [])
            if not opp_pts:
                continue

            opp_arr = np.array(opp_pts, dtype=np.float32)   # (M, 2)

            for cx, cy in pts:
                # Euclidean distances to all opponents
                diffs = opp_arr - np.array([cx, cy], dtype=np.float32)
                dists = np.linalg.norm(diffs, axis=1)
                count = int(np.sum(dists <= pressure_r))

                if count == 0:
                    continue

                level = min(count, 3)
                color = PressureAnalyzer._PRESSURE_COLORS[level]
                radius = int(pressure_r * 0.35 + level * 8)

                overlay = frame.copy()
                cv2.circle(overlay, (int(cx), int(cy)), radius, color, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
                cv2.circle(frame, (int(cx), int(cy)), radius, color, 2, cv2.LINE_AA)
