from __future__ import annotations

import logging
import os.path
from pathlib import Path
from typing import Iterator

import cv2
import numpy
import yaml

import utils


class Net:
    size = None
    net = None
    names = None

    def __init__(self, model_path, threshold):
        self.model_path = model_path
        self.threshold = threshold

    def parse_output(self, output, width, height) -> Iterator[tuple[list[int], float, int]]:
        raise NotImplementedError


class DarkNet(Net):
    size = 416

    def __init__(self, model_path, threshold):
        super().__init__(model_path, threshold)
        with open(os.path.join(model_path, 'yolov3.names'), 'r', encoding='utf-8') as fobj:
            self.names = [line.strip() for line in fobj.readlines()]
        self.net = cv2.dnn.readNetFromDarknet(
            os.path.join(model_path, 'yolov3.cfg'),
            os.path.join(model_path, 'yolov3.weights'),
        )

    def parse_output(self, output, width, height) -> Iterator[tuple[list[int], float, int]]:
        for detection in output:
            scores = detection[5:]
            class_id = numpy.argmax(scores)
            confidence = scores[class_id]

            if confidence > self.threshold:
                cx, cy, w, h = (detection[0:4] * numpy.array([width, height, width, height])).astype(int)
                coords = [utils.nearest_int(v, 5) for v in [cx - w / 2, cy - h / 2, w, h]]
                yield coords, float(confidence), int(class_id)


class ONNXNet(Net):
    size = 640

    def __init__(self, model_path, threshold):
        super().__init__(model_path, threshold)
        self.model_path = Path(model_path)
        yaml_files = list(self.model_path.glob('*.yaml'))
        onnx_files = list(self.model_path.glob('*.onnx'))

        if len(yaml_files) and len(onnx_files):
            with open(yaml_files[0], 'r') as fobj:
                data = yaml.safe_load(fobj)
            self.names = list(data['names'].values())
            onnx = str(onnx_files[0])
            self.net = cv2.dnn.readNetFromONNX(onnx)
        else:
            raise FileNotFoundError('ONNX Model not found')

    def parse_output(self, output, width, height) -> Iterator[tuple[list[int], float, int]]:
        for i in range(output.shape[-1]):
            detection = output[0, ..., i]
            scores = detection[4:]
            class_id = numpy.argmax(scores)
            confidence = scores[class_id]

            if confidence > self.threshold:
                scale = numpy.array([width, height, width, height]) / self.size
                step = int(numpy.ceil(scale.max()))
                cx, cy, w, h = (detection[0:4] * scale).astype(int)
                coords = [utils.nearest_int(v, step) for v in [cx - w / 2, cy - h / 2, w, h]]
                yield coords, float(confidence), int(class_id)


def load_model(model_path: str | Path, threshold: float) -> Net:
    for net_class in [DarkNet, ONNXNet]:
        try:
            logging.info(f'Loading {net_class.__name__} model from {model_path}')
            net = net_class(model_path, threshold)
        except Exception as err:
            logging.exception(err)
            continue
        else:
            return net
    raise ValueError('No such model')
