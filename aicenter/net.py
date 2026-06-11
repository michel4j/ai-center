from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy
from ultralytics import YOLO
from ultralytics.utils import checks
import torch

from . import utils


@dataclass
class Result:
    type: str
    x1: float
    x2: float
    y1: float
    y2: float
    score: float
    cx: int = 0
    cy: int = 0

    def __post_init__(self):
        self.cx = round((self.x1 + self.x2)/2)
        self.cy = round((self.y1 + self.y2)/2)

    def box(self):
        return round(self.x1), round(self.y1), round(self.x2), round(self.y2)


class Net:
    def __init__(self, model_path, threshold: float):
        self.model_path = model_path
        self.threshold = threshold
        self.load_model()
        self.setup_model()

    def setup_model(self):
        """Perform Model Specific configuration"""
        pass

    def load_model(self):
        """Model-specific loading"""
        raise NotImplementedError

    def predict(self, image: numpy.ndarray) -> list[dict]:
        return []

    @staticmethod
    def group_objects(items: list[dict], **kwargs):
        results = defaultdict(list)
        for item in items:
            obj = Result(
                type=item['name'],
                score=item['confidence'],
                **item['box']
            )
            logging.debug(f'{obj.type} found at: {obj.cx} {obj.cy}, prob={obj.score}')
            results[obj.type].append(obj)
        for label, llist in results.items():
            results[label] = sorted(llist, key=lambda result: result.score, reverse=True)
        return results


class UltralyticsYOLO(Net):
    """
    UltraLytics YOLO
    """
    model:  YOLO

    def load_model(self):
        self.model_path = Path(self.model_path)

        try:
            self.model = YOLO(self.model_path, task='detect')
        except Exception as err:
            raise RuntimeError('No valid YOLO Model found')

    def setup_model(self):
        checks.check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

    def predict(self, image: numpy.ndarray) -> list:
        results = self.model.predict(source=image, conf=self.threshold)
        return results[0].summary()


def load_model(model_path: str | Path, threshold) -> Net:
    for model_class in [UltralyticsYOLO,]:
        try:
            logging.info(f'Loading {model_class.__name__} model from {model_path}')
            net = model_class(model_path, threshold)
        except Exception as err:
            logging.warning(f'Unable to load model of type {model_class.__name__}: {err}!')
            continue
        else:
            return net
    raise ValueError('No such model')
