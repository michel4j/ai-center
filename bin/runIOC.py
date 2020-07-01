#!/usr/bin/env python3
import os
import logging
import sys
import argparse

# Twisted boiler-plate code.
from twisted.internet import gireactor
gireactor.install()
from twisted.internet import reactor

# add the project to the python path and inport it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from devioc import log
from aicenter import ioc

# Setup single argument for verbose logging
parser = argparse.ArgumentParser(description='Ai Centering')
parser.add_argument('-v', action='store_true', help='Verbose Logging')
parser.add_argument('--device', type=str, help='Device Name', required=True)
parser.add_argument('--server', type=str, help='Video Server', required=True)
parser.add_argument('--camera', type=str, help='Camera ID', required=True)
parser.add_argument('--scale', type=int, help='Scale down factor', default=4)

   


if __name__== '__main__':
    args = parser.parse_args()
    if args.v:
        log.log_to_console(logging.DEBUG)
    else:
        log.log_to_console(logging.INFO)

    app = ioc.AicenterApp(args.device, server=args.server, camera=args.camera, scale=args.scale)  # initialize App
    reactor.addSystemEventTrigger('before', 'shutdown', app.shutdown) # make sure app is properly shutdown
    reactor.run()               # run main-loop

