#!/usr/bin/env python3
import argparse
import glob
import logging
import os
import time
import warnings

import cv2
import redis

from aicenter import AiCenter

warnings.filterwarnings("ignore")
logger = logging.getLogger('aicenter')

CONF_THRESH, NMS_THRESH = 0.25, 0.25


class AiCenterApp(AiCenter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f'model={self.model_path!r}, server={self.server!r}, camera={self.key!r}')
        self.running = False
        if self.server:
            self.video = redis.Redis(host=self.server, port=6379, db=0)

    def run(self, scale=0.5):
        self.running = True
        while self.running:
            raw_frame = self.get_frame()
            if raw_frame is None:
                continue
            frame = cv2.resize(raw_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            results = self.process_frame(frame)

            if results is not None:
                for label, reslist in results.items():
                    for res in reslist:
                        cv2.rectangle(frame, (res.x, res.y), (res.x+res.w, res.y+res.h), (255, 0, 0), 1)
                        cv2.putText(frame, f'{res.type}:{res.score:0.2f}', (res.x, res.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


class AiCenterImagesApp(AiCenterApp):
    def __init__(self, **kwargs):
        images_dir = kwargs.pop('images')
        self.images = self.frame_generator(images_dir)
        print(f"Simulating stream from {images_dir!r}")
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
    parser.add_argument('--model', type=str, help='Path to model directory',
                        default="/cmcf_apps/ai-centering/model")
    parser.add_argument('--server', type=str, help='Redis camera server address',
                        default="IOC1608-304.clsi.ca")
    parser.add_argument('--camera', type=str, help='Redis camera ID',
                        default="0030180F06E5")
    parser.add_argument('--images', type=str, help='Path to directory of images (simulate stream)')
    args = parser.parse_args()
    if args.images:
        app = AiCenterImagesApp(model=args.model, images=args.images)
    else:
        app = AiCenterApp(model=args.model, server=args.server, camera=args.camera)
    app.run()
