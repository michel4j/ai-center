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
    def run(self, scale=1.2):
        self.running = True
        cv2.namedWindow('AI-Centering Viewer', cv2.WINDOW_NORMAL)
        self.waiting = False
        while self.running:

            if self.waiting:
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    self.waiting = False
                    time.sleep(0.1)
                continue

            raw_frame = self.get_frame()
            if raw_frame is None:
                continue
            frame = cv2.resize(raw_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            t = time.time_ns()
            results = self.process_frame(frame)
            print(f'Processing time: {(time.time_ns() - t) / 1e6:.4f}ms')

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

            cv2.imshow('AI-Centering Viewer', frame)
            time.sleep(.1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
            elif cv2.waitKey(1) & 0xFF == ord('s'):
                self.waiting = not self.waiting


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate a video stream using a pre-trained object detection model')
    parser.add_argument('--sam', type=str, help='Path to SAM model')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model')
    parser.add_argument('--video', type=str, required=True, help='Video URI')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--confidence', type=float, help='Object Detection Confidence Threshold')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    app = AiCenterApp(
        model=args.model,
        sam_model=args.sam,
        video=args.video,
        threshold=args.confidence
    )
    app.run()
