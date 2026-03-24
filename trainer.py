#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO


class YOLOTrainer:
    def __init__(
        self,
        model_path: str = "yolo26s.pt",
        data_yaml: str = "data.yaml",
        imgsz: int = 1280,
        epochs: int = 50,
        batch: int = 8,
        workers: int = 4,
    ):
        self.model_path = model_path
        self.data_yaml  = data_yaml
        self.imgsz      = imgsz
        self.epochs     = epochs
        self.batch      = batch
        self.workers    = workers
        self._model     = None

    def train(self):
        self._model = YOLO(self.model_path)
        self._model.train(
            data=self.data_yaml,
            imgsz=self.imgsz,
            epochs=self.epochs,
            batch=self.batch,
            workers=self.workers,
        )
