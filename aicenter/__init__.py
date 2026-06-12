from __future__ import annotations

from collections import defaultdict

import numpy
import cv2

from aicenter import img, utils
from aicenter.log import get_module_logger
from aicenter.net import load_model, Result

try:
    from aicenter.sam import TrackingSAM
except ModuleNotFoundError as e:
    TrackingSAM = None

logger = get_module_logger(__name__)

CONF_THRESH = 0.1


class AiCenter:
    def __init__(self, yolo_model, sam_model=None, server=None, camera=None, threshold=CONF_THRESH):
        self.key = f'{camera}:JPG'
        self.server = server
        self.video = None
        self.yolo_path = yolo_model
        self.sam_path = sam_model
        threshold = threshold if threshold else CONF_THRESH

        # prepare neural network for detection
        self.net = load_model(self.yolo_path, threshold)

        # setup SAM2 for segmentation
        if TrackingSAM is not None:
            self.sam = TrackingSAM(model_path=self.sam_path)
        else:
            self.sam = None

    def get_frame(self):
        try:
            data = self.video.get(self.key)
            image = numpy.frombuffer(data, numpy.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except TypeError as err:
            logger.error('Unable to grab frame')
            return None
        else:
            return frame

    def process_frame(self, frame):
        tracking = 0
        if frame is not None:
            # Object detection
            outputs = self.net.predict(frame)
            results = self.net.group_objects(outputs)
            # Image processing fallback
            if not results:
                results = img.process_frame(frame)
            return results
        return {}

    def process_tracking(self, frame, result: Result | None = None):
        """
        Process tracking for this frame. Provide a new result object to start tracking
        otherwise simply predict for existing object

        :param frame: image frame
        :param result: new identified object to track
        :return: predicted object from tracking
        """

        if self.sam and frame is not None:
            height, width = frame.shape[:2]

            if result is not None:
                # Prompt segmentation with objects
                self.sam.track_object(frame, result, width, height)

            # Segmentation
            if not self.sam.tracked_object:
                return None

            mask, score, obj = self.sam.predict(frame)
            if mask is not None:
                return self.sam.process_result(mask, score, obj)
        return None

