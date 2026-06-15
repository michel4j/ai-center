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

from ultralytics import YOLO


warnings.filterwarnings("ignore")
logger = get_module_logger("inference")


class TrackingApp:
    def __init__(self, model, images):
        self.model = YOLO(model, task="detect")
        self.images = self.frame_generator(images)
        self.running = False
        logger.info(f"Simulating stream from {images!r}")

    def run(self, scale=0.5):
        self.running = True
        while self.running:
            raw_frame = self.get_frame()
            if raw_frame is None:
                continue
            frame = cv2.resize(raw_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            results = self.model.track(frame, persist=True, conf=0.01)

            # Visualize results on the frame
            annotated_frame = results[0].plot()

            cv2.imshow("YOLO Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

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
    parser.add_argument('--model', type=str, help='Path to YOLO model')
    parser.add_argument('--images', type=str, help='Path to directory of images (simulate stream)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    app = TrackingApp(
        model=args.model,
        images=args.images,
    )
    app.run()
