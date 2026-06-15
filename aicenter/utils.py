from __future__ import annotations

import glob
import os
import time
from itertools import cycle
from pathlib import Path
from typing import Generator

import cv2
import numpy
import redis

from aicenter.log import get_module_logger

logger = get_module_logger(__name__)


def nearest_int(value, step=1):
    """
    Round to the nearest step
    :param value: value to round
    :param step: unit to round to
    """

    return int(round(value / step) * step)


def inside_bbox(x, y, bbox) -> bool:
    """
    Check if point is inside bounding box
    :param x: x coordinate of point
    :param y: y coordinate of point
    :param bbox: Tuple, list or array (x, y, w, h) of bounding box
    :return: bool
    """

    bx, by, bw, bh = bbox
    return bx <= x <= bx + bw and by <= y <= by + bh


def file_frame_generator(path: str | Path, **kwargs) -> Generator:
    """
    Generate frames from disk files
    :param path: Path to directory of images
    """
    logger.info(f"Simulating stream from {path!r}")
    for filename in cycle(sorted(glob.glob(os.path.join(path, "*[.png,.jpg,.jpeg]")))):
        t = time.perf_counter()
        try:
            image = cv2.imread(filename)
        except TypeError as err:
            logger.error('Unable to grab frame')
            return
        except KeyboardInterrupt:
            logger.info('Exiting ...')
            return
        else:
            yield image

        delay = t + 0.1 - time.perf_counter()
        if delay > 0:
            time.sleep(delay)


def redis_frame_generator(host: str, path: str, port: int = None, **kwargs) -> Generator:
    """
    Generate frames from redis stream

    :param host: Host name or IP address
    :param path: Camera path
    :param port: Redis port, defaults to 6379
    :return: Generator
    """
    port = 6379 if port is None else int(port)

    server = redis.Redis(host=host, port=port, db=0, protocol=2)
    key = f"{path.strip('/')}:JPG"
    logger.info(f"Fetching video stream from {key!r}")
    while True:
        t = time.perf_counter()
        try:
            data = server.get(key)
            image = numpy.frombuffer(data, numpy.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except TypeError as err:
            logger.error(f'Unable to fetch redis frame: {err}')
            logger.exception(err)
        except KeyboardInterrupt:
            logger.info('Exiting ...')
            return
        else:
            yield frame

        delay = t + 0.1 - time.perf_counter()
        if delay > 0:
            time.sleep(delay)


VIDEO_SOURCES = {
    'file': file_frame_generator,
    'redis': redis_frame_generator
}