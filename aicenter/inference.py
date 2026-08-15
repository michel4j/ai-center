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

warnings.filterwarnings("ignore")
logger = get_module_logger("inference")

CONF_THRESH, NMS_THRESH = 0.25, 0.25


class InferenceApp(AiCenter):
    def run(self, scale=1.0):
        self.running = True
        cv2.namedWindow('AI-Centering Viewer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI-Centering Viewer", 800, 600)
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

            print(raw_frame.shape)
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

            cv2.imshow('AI-Centering Viewer', frame)
            time.sleep(.1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
            elif cv2.waitKey(1) & 0xFF == ord('s'):
                self.waiting = not self.waiting

