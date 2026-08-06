#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
import warnings
from typing import Generator

import cv2
from datetime import datetime
from ultralytics import YOLO

from aicenter import utils
from aicenter.log import get_module_logger

warnings.filterwarnings("ignore")
logger = get_module_logger("inference")

VIDEO_URI_PATTERN = re.compile(
    r'^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*)://'  # Protocol/Scheme
    r'(?P<host>[^:/ \n]*)'  # Hostname or IP address
    r'(?::(?P<port>\d+))?'  # Optional port number
    r'(?P<path>/[^?#\s]*)?'  # Optional path
)


class TrackingApp:
    images: Generator
    running: bool = False
    clicked: bool = False


    def __init__(self, model, uri):
        self.model = YOLO(model, task="detect")
        self.uri = uri
        match = VIDEO_URI_PATTERN.match(self.uri)
        if match:
            self.src = {k: v for k, v in match.groupdict().items() if v is not None}
            frame_generator = utils.VIDEO_SOURCES.get(self.src['scheme'], None)
            if frame_generator is None:
                raise NotImplementedError(f'Unsupported Video Source Scheme: {self.src["scheme"]}')
            self.images = frame_generator(**self.src)
        else:
            raise NotImplementedError(f'Unsupported Video Source URI: {self.uri}')

    def mouse_callback(self, event, x, y, flags, param):
        # Check if the left mouse button was clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = True

    def run(self, scale=1.0):
        self.running = True
        window_name = "AI-Centering Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 1024)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        index = 0
        suffix = datetime.now().strftime('%Y%m%d-%H%M%S')
        while self.running:
            frame = self.get_frame()
            if frame is None:
                continue

            #frame = cv2.resize(raw_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            #results = self.model.track(frame, persist=True, conf=0.01, tracker="bytetrack.yaml")
            results = self.model.predict(frame, conf=0.01)

            # Visualize results on the frame
            annotated_frame = results[0].plot()

            cv2.imshow(window_name, annotated_frame)
            while True:
                key = cv2.waitKey(10)
                if self.clicked or key in [32, 13]:
                    self.clicked = False
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'/home/michel/target-{suffix}-{index:04d}.png', annotated_frame)
                    index += 1
                    break
                elif key in [27, ord('q')]:
                    self.running = False
                    break
        cv2.destroyAllWindows()

    def get_frame(self):
        try:
            frame = next(self.images)
        except StopIteration:
            self.running = False
        else:
            return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate a video stream using a pre-trained object detection model')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model')
    parser.add_argument('--video', type=str, required=True, help='Video URI')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    app = TrackingApp(model=args.model, uri=args.video)
    app.run()
