from collections import defaultdict

import numpy

from aicenter import img, utils
from aicenter.log import get_module_logger
from aicenter.net import load_model, Result

try:
    from aicenter.sam import TrackingSAM
except ModuleNotFoundError as e:
    TrackingSAM = None

logger = get_module_logger(__name__)

CONF_THRESH = 0.125


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
        if frame is not None:
            # Object detection
            height, width = frame.shape[:2]
            outputs = self.net.predict(frame)
            results = self.net.group_objects(outputs)
            # Prompt segmentation with objects
            if self.sam:
                if results:
                    self.sam.track_objects(frame, results, width, height)
                # Segmentation
                if self.sam.tracked_objects:
                    mask_outputs = self.sam.predict(frame)
                    mask_results = self.sam.process_results(*mask_outputs)
                    if not results:
                        results = defaultdict(list)
                    for label in mask_results.keys():
                        results[label].extend(mask_results[label])
                        # Keep list sorted by score
                        results[label] = sorted(results[label], key=lambda result: result.score, reverse=True)
            # Image processing fallback
            if not results:
                results = img.process_frame(frame)
            return results
        return {}
