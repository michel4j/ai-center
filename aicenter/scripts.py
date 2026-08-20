import logging
import argparse
import sys
from pathlib import Path

# Twisted boiler-plate code.
from twisted.internet import gireactor

from .inference import InferenceApp

gireactor.install()
from twisted.internet import reactor

# add the project to the python path and import it
from devioc import log
from . import ioc

MODEL_ID = 'michel4j/mxsamples'
MODEL_FILE = 'model.pt'

logger = log.get_module_logger('scripts')


def get_default_model():
    try:
        from huggingface_hub import hf_hub_download
        file_path = hf_hub_download(repo_id=MODEL_ID, filename='model.pt')
    except Exception as e:
        logger.exception(e)
        file_path = None
    return file_path


def server_main():
    # Setup single argument for verbose logging
    parser = argparse.ArgumentParser(description='Ai Centering')
    parser.add_argument('-v', action='store_true', help='Verbose Logging')
    parser.add_argument('--device', type=str, help='Device Name', required=True)
    parser.add_argument('--model', type=str, help='YOLOModel Path', default=get_default_model())
    parser.add_argument('--video', type=str, help='Video URI', required=True)
    parser.add_argument('--confidence', type=float, default=0.1, help='Object Detection Confidence Threshold')

    args = parser.parse_args()
    if args.v:
        log.log_to_console(logging.DEBUG)
    else:
        log.log_to_console(logging.INFO)

    if not args.model:
        raise RuntimeError('Model not found! Either provide a model or install the [model] extra!')

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
    parser.add_argument('--model', type=str, help='Path to YOLO model', default=get_default_model())
    parser.add_argument('--video', type=str, required=True, help='Video URI')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--confidence', type=float, help='Object Detection Confidence Threshold')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if not args.model:
        raise RuntimeError('Model not found! Either provide a model or install the [model] extra!')

    app = InferenceApp(
        model=args.model,
        video=args.video,
        threshold=args.confidence
    )
    app.run()


if __name__ == '__main__':
    inference_main()

