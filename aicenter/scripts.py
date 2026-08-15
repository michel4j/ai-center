import logging
import argparse
import sys

# Twisted boiler-plate code.
from twisted.internet import gireactor

from .inference import InferenceApp

gireactor.install()
from twisted.internet import reactor

# add the project to the python path and import it
from devioc import log
from . import ioc


def server_main():
    # Setup single argument for verbose logging
    parser = argparse.ArgumentParser(description='Ai Centering')
    parser.add_argument('-v', action='store_true', help='Verbose Logging')
    parser.add_argument('--device', type=str, help='Device Name', required=True)
    parser.add_argument('--model', type=str, help='YOLOModel Path', required=True)
    parser.add_argument('--video', type=str, help='Video URI', required=True)
    parser.add_argument('--confidence', type=float, default=0.1, help='Object Detection Confidence Threshold')

    args = parser.parse_args()
    if args.v:
        log.log_to_console(logging.DEBUG)
    else:
        log.log_to_console(logging.INFO)

    # initialize App
    app = ioc.IOCApp(
        args.device,
        model=args.model,
        video=args.video,
        threshold=args.confidence,
    )
    reactor.addSystemEventTrigger('before', 'shutdown', app.shutdown)   # make sure app is properly shutdown
    sys.exit(
        reactor.run()
    )   # run main-loop


def inference_main():
    parser = argparse.ArgumentParser(
        description='Annotate a video stream using a pre-trained object detection model'
    )
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model')
    parser.add_argument('--video', type=str, required=True, help='Video URI')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--confidence', type=float, help='Object Detection Confidence Threshold')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    app = InferenceApp(
        model=args.model,
        video=args.video,
        threshold=args.confidence
    )
    app.run()