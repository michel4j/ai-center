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

        cv2.namedWindow('AI-Centering Viewer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI-Centering Viewer", 720, 568)

        while True:
            raw_frame = self.get_frame()
            if raw_frame is None:
                continue

            frame = cv2.resize(raw_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            results = self.net.model.predict(frame, conf=self.threshold)
            annotated_frame = results[0].plot()
            cv2.imshow('AI-Centering Viewer', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

