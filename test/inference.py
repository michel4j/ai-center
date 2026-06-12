#!/usr/bin/env python3
import argparse
import glob
import logging
import os
import time
import warnings
from pathlib import Path

import cv2
import redis

from aicenter import AiCenter
from aicenter.log import get_module_logger
try:
    from aicenter.sam import MaskResult, show_mask_from_result
except ModuleNotFoundError:
    MaskResult = None
    show_mask_from_result = None

warnings.filterwarnings("ignore")
logger = get_module_logger("inference")

CONF_THRESH, NMS_THRESH = 0.25, 0.25


class AiCenterApp(AiCenter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(f'yolo={self.yolo_path!r}, sam={self.sam_path!r}, server={self.server!r}, camera={self.key!r}')
        self.running = False
        if self.server:
            self.video = redis.Redis(host=self.server, port=6379, db=0, protocol=2)

    def run(self, scale=0.5):
        self.running = True
        while self.running:
            raw_frame = self.get_frame()
            if raw_frame is None:
                continue
            frame = cv2.resize(raw_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            results = self.process_frame(frame)

            to_track = None
            if results:
                for label, objects in results.items():
                    for i, res in enumerate(objects):
                        if res.type == 'crystal' and i == 0:
                            to_track = res
                        x1, y1, x2, y2 = res.box()
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        cv2.putText(
                            frame,
                            f'{res.type}:{res.score:0.2f}',
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
            if self.sam:
                tracked_result = None
                if self.sam.tracked_object:
                    tracked_result = self.process_tracking(frame)
                elif to_track:
                    tracked_result = self.process_tracking(frame, to_track)
                if tracked_result:
                    frame = show_mask_from_result(frame, tracked_result)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


class AiCenterImagesApp(AiCenterApp):
    def __init__(self, **kwargs):
        images_dir = kwargs.pop('images')
        self.images = self.frame_generator(images_dir)
        logger.info(f"Simulating stream from {images_dir!r}")
        super().__init__(**kwargs)

    @staticmethod
    def frame_generator(images):
        for filename in sorted(glob.glob(os.path.join(images, "*[.png,.jpg,.jpeg]"))):
            t = time.perf_counter()
            try:
                image = cv2.imread(filename)
            except TypeError as err:
                logger.error('Unable to grab frame')
                return
            else:
                yield image
            delay = t + 0.1 - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def get_frame(self):
        try:
            frame = next(self.images)
        except StopIteration:
            self.running = False
        else:
            return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate a video stream using a pre-trained object detection model')
    parser.add_argument('--yolo', type=str, help='Path to YOLO model')
    parser.add_argument('--sam', type=str, help='Path to SAM model')
    parser.add_argument('--server', type=str, help='Redis camera server address',  default="IOC1608-304.clsi.ca")
    parser.add_argument('--camera', type=str, help='Redis camera ID', default="0030180F06E5")
    parser.add_argument('--images', type=str, help='Path to directory of images (simulate stream)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--confidence', type=float, help='Object Detection Confidence Threshold')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    if args.images:
        app = AiCenterImagesApp(
            yolo_model=args.yolo,
            sam_model=args.sam,
            images=args.images,
            threshold=args.confidence
        )
    else:
        app = AiCenterApp(
            yolo_model=args.yolo,
            sam_model=args.sam,
            server=args.server,
            camera=args.camera,
            threshold=args.confidence
        )
    app.run()
