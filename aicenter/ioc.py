import os
import threading
import time
import warnings

import cv2
import numpy
import redis

warnings.filterwarnings("ignore")

from enum import Enum
from queue import Queue
from collections import namedtuple, defaultdict

from devioc import models, log
import gepics

logger = log.get_module_logger('aicenter')

# Result Type
Result = namedtuple('Result', 'type x y w h score')

CONF_THRESH, NMS_THRESH = 0.5, 0.5


class StatusType(Enum):
    VALID, INVALID = range(2)


# Create your models here. Modify the example below as appropriate
class AiCenter(models.Model):
    x = models.Integer('x', default=0, desc='X')
    y = models.Integer('y', default=0, desc='Y')
    w = models.Integer('w', default=0, desc='Width')
    h = models.Integer('h', default=0, desc='Height')
    score = models.Float('score', default=0.0, desc='Reliability')
    label = models.String('label', default='', desc='Object Type')
    status = models.Enum('status', choices=StatusType, desc="Status")


class AiCenterApp(object):
    def __init__(self, device, model=None, server=None, camera=None):
        logger.info(f'device={device!r}, model={model!r}, server={server!r}, camera={camera!r}')
        self.running = False
        self.ioc = AiCenter(device, callbacks=self)
        self.key = f'{camera}:JPG'
        self.server = server
        self.video = None
        self.model_path = model

        # prepare neural network for detection
        with open(os.path.join(model, 'yolov3.names'), 'r', encoding='utf-8') as fobj:
            names = [line.strip() for line in fobj.readlines()]

        self.darknet = {
            'weights': os.path.join(model, 'yolov3.weights'),
            'config': os.path.join(model, 'yolov3.cfg'),
            'names': names,
        }

        self.net = cv2.dnn.readNetFromDarknet(self.darknet['config'], self.darknet['weights'])
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layers = self.net.getLayerNames()
        self.output_layers = [self.layers[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.start_monitor()

    def start_monitor(self):
        self.running = False
        monitor_thread = threading.Thread(target=self.video_monitor, daemon=True)
        monitor_thread.start()

    def get_frame(self):
        try:
            data = self.video.get(self.key)
            image = numpy.frombuffer(data, numpy.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except TypeError:
            logger.error('Unable to grab frame')
            return
        else:
            return frame

    def process_results(self, width, height, outputs):
        class_ids, confidences, bboxes = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = numpy.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONF_THRESH:
                    cx, cy, w, h = (detection[0:4] * numpy.array([width, height, width, height])).astype(int)

                    x = int(cx - w / 2)
                    y = int(cy - h / 2)

                    bboxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        if bboxes:
            indices = cv2.dnn.NMSBoxes(bboxes, confidences, CONF_THRESH, NMS_THRESH).flatten()
            scores = [confidences[index] for index in indices]
            index = indices[numpy.argmax(scores)]
            x, y, w, h = bboxes[index]
            score = confidences[index]
            label = self.darknet['names'][class_ids[index]]

            self.ioc.status.put(StatusType.VALID.value)
            self.ioc.x.put(x)
            self.ioc.y.put(y)
            self.ioc.w.put(w)
            self.ioc.h.put(h)
            self.ioc.label.put(label)
            self.ioc.score.put(score)

            logger.debug(f'{label} found at: {x} {y} [{w} {h}], prob={score}')
        else:
            self.ioc.status.put(StatusType.INVALID.value)
            self.ioc.score.put(0.0)

    def video_monitor(self):
        gepics.threads_init()
        self.running = True
        self.video = redis.Redis(host=self.server, port=6379, db=0)
        while self.running:
            frame = self.get_frame()
            if frame is not None:
                height, width = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)
                outputs = self.net.forward(self.output_layers)
                self.process_results(width, height, outputs)
            else:
                self.ioc.status.put(StatusType.INVALID.value)
                self.ioc.score.put(0.0)
            time.sleep(0.001)

    def shutdown(self):
        # needed for proper IOC shutdown
        self.running = False
        self.ioc.shutdown()
