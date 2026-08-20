from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy
import cv2

from aicenter import img, utils
from aicenter.log import get_module_logger
from aicenter.net import load_model, Result


logger = get_module_logger(__name__)

CONF_THRESH = 0.1
VIDEO_URI_PATTERN = re.compile(
    r'^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*)://'  # Protocol/Scheme
    r'(?P<host>[^:/ \n]*)'                      # Hostname or IP address
    r'(?::(?P<port>\d+))?'                      # Optional port number
    r'(?P<path>/[^?#\s]*)?'                     # Optional path
)


class AiCenter:
    sam: Any = None
    running: bool = False

    def __init__(self, model, video, threshold=CONF_THRESH, sam_model: str | Path = '', tracking=True):
        """
        AiCenter
        :param model: YOLO model path
        :param video: Video URI
        :param threshold: confidence threshold
        :param sam_model: SAM2 model path if using segmentation
        :param tracking: enable tracking
        """
        self.server = video
        self.video = None
        self.model_path = model
        self.sam_path = sam_model
        self.threshold = threshold if threshold else CONF_THRESH

        # prepare neural network for detection
        self.net = load_model(self.model_path, self.threshold, tracking=tracking)

        self.uri = video
        match = VIDEO_URI_PATTERN.match(self.uri)
        if match:
            self.src = {k: v for k, v in match.groupdict().items() if v is not None}
            frame_generator = utils.VIDEO_SOURCES.get(self.src['scheme'], None)
            if frame_generator is None:
                raise NotImplementedError(f'Unsupported Video Source Scheme: {self.src["scheme"]}')
            self.images = frame_generator(**self.src)
        else:
            raise NotImplementedError(f'Unsupported Video Source URI: {self.uri}')

    def get_frame(self):
        try:
            frame = next(self.images)
        except StopIteration:
            self.running = False
        else:
            return frame

    def process_frame(self, frame):
        if frame is not None:
            # Object detection with YOLO
            outputs = self.net.predict(frame)
            results = self.net.group_objects(outputs)
            # Image processing fallback
            if not results:
                results = img.process_frame(frame)
            return results
        return {}

