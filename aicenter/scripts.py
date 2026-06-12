import logging
import argparse
import sys

# Twisted boiler-plate code.
from twisted.internet import gireactor
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
    parser.add_argument('--yolo', type=str, help='YOLOModel Path', required=True)
    parser.add_argument('--sam', type=str, help='SAM2 Model Path')
    parser.add_argument('--server', type=str, help='Video Server', required=True)
    parser.add_argument('--camera', type=str, help='Camera ID', required=True)
    parser.add_argument('--confidence', type=float, default=0.1, help='Object Detection Confidence Threshold')

    args = parser.parse_args()
    if args.v:
        log.log_to_console(logging.DEBUG)
    else:
        log.log_to_console(logging.INFO)

    # initialize App
    app = ioc.AiCenterApp(
        args.device,
        yolo=args.yolo,
        sam=args.sam,
        server=args.server,
        camera=args.camera,
        threshold=args.confidence
    )
    reactor.addSystemEventTrigger('before', 'shutdown', app.shutdown)   # make sure app is properly shutdown
    sys.exit(
        reactor.run()
    )   # run main-loop

