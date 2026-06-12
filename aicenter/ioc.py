import threading
import time
import warnings

import numpy
import redis
from matplotlib.pyplot import yscale

from aicenter import Result

warnings.filterwarnings("ignore")

from enum import IntEnum

from devioc import models, log
import gepics

from . import utils
from . import AiCenter


logger = log.get_module_logger('aicenter')


class EnableType(IntEnum):
    DISABLED, ENABLED = range(2)


class StatusType(IntEnum):
    VALID, INVALID = range(2)


class ObjectType(IntEnum):
    NONE, LOOP, CRYSTAL, PIN = range(4)


# Create your models here. Modify the example below as appropriate
class AiCenterModel(models.Model):
    # Loop bounding box
    x = models.Integer('x', default=0, desc='X')
    y = models.Integer('y', default=0, desc='Y')
    w = models.Integer('w', default=0, desc='Width')
    h = models.Integer('h', default=0, desc='Height')
    score = models.Float('score', default=0.0, desc='Reliability')
    label = models.String('label', default='', desc='Object Type')
    status = models.Enum('status', choices=StatusType, desc="Status")
    enable = models.Enum('enable', choices=EnableType, default=1, desc="Enable/Disable")

    # Tracking
    track = models.Enum('track:enable', choices=EnableType, default=EnableType.DISABLED, desc="Track Object")
    track_input = models.Array('track:src', type=int, length=4, desc="Track Input Box")
    track_valid = models.Enum('track:valid', choices=StatusType, default=StatusType.INVALID, desc="Track Input Valid")
    track_output = models.Array('track:fbk', type=int, length=4, desc="Track Output Box")
    track_score = models.Float('score', default=0.0, desc='Track Score')

    # Many-object centers
    objects_x = models.Array('objects:x', type=int, desc="Objects X")
    objects_y = models.Array('objects:y', type=int, desc="Objects Y")
    objects_type = models.Array('objects:type', type=int, desc="Objects Type")
    objects_score = models.Array('objects:score', type=float, desc="Objects Score")
    objects_valid = models.Integer('objects:valid', default=0, mdel=0, desc="Valid objects")


class AiCenterApp(AiCenter):
    def __init__(self, device, yolo=None, sam=None,server=None, camera=None, threshold=None):
        """
        AiCenter IOC
        :param device:  device root name for PVs
        :param yolo:  YOLO Model path
        :param sam:  SAM2 Model path
        :param server:  Redis server for video stream
        :param camera:  Camera name for video stream
        """
        super().__init__(yolo_model=yolo, sam_model=sam, server=server, camera=camera, threshold=threshold)
        logger.info(f'device={device!r}, yolo={yolo!r}, sam={sam!r}, server={server!r}, camera={camera!r}')
        self.running = False
        self.enabled = True
        self.tracking = False
        self.ioc = AiCenterModel(device, callbacks=self)

        self.start_monitor()

    def start_monitor(self):
        self.running = False
        monitor_thread = threading.Thread(target=self.video_monitor, daemon=True)
        monitor_thread.start()

    def get_best_object(self, kind: ObjectType = ObjectType.CRYSTAL) -> Result:
        types = self.ioc.objects_type.get()
        scores = self.ioc.objects_score.get()
        xs = self.ioc.objects_x.get()
        ys = self.ioc.objects_y.get()
        ws = self.io
        for i in range(self.ioc.objects_valid.get()):
            if self.ioc.objects_type[i] == kind:
                return Result(
                    type=kind.name.lower(),
                    x1=self.ioc.objects_x.get()[i],

                )

    def video_monitor(self):
        gepics.threads_init()
        self.running = True
        self.video = redis.Redis(host=self.server, port=6379, db=0, protocol=2)
        while self.running:
            if self.ioc.enable.get() != EnableType.ENABLED:
                if self.ioc.score.get() > 0:
                    self.ioc.status.put(StatusType.INVALID)     # Reset object count
                    self.ioc.score.put(0.0)                     # Reset score to invalidate current object
                    self.ioc.objects_valid.put(0)               # Reset object count
                time.sleep(0.1)
                continue

            frame = self.get_frame()
            results = self.process_frame(frame)
            best_crystal = None

            if results:
                if 'loop' in results:
                    # Only return highest-scoring loop
                    result = results['loop'][0]
                    self.ioc.x.put(result.x1)
                    self.ioc.y.put(result.y1)
                    self.ioc.w.put(result.x2 - result.x1)
                    self.ioc.h.put(result.y2 - result.y1)
                    self.ioc.label.put(result.type)
                    self.ioc.score.put(result.score - numpy.random.uniform(0, 0.0001))
                    self.ioc.status.put(StatusType.VALID)

                xs, ys, scores, types = [], [], [], []

                for label, res_list in results.items():
                    object_type = {
                        'loop': ObjectType.LOOP,
                        'crystal': ObjectType.CRYSTAL,
                        'pin': ObjectType.PIN
                    }.get(label, ObjectType.NONE)

                    if object_type == ObjectType.CRYSTAL:
                        # Add all crystals when no loop is present, otherwise must be inside loop
                        valid_xtals = res_list
                        if 'loop' in results:
                            loop_bbox = (self.ioc.x.get(), self.ioc.y.get(), self.ioc.w.get(), self.ioc.h.get())
                            valid_xtals = [
                                result for result in res_list
                                if utils.inside_bbox(result.x1, result.y1, loop_bbox)
                            ]
                        best_crystal = valid_xtals[0] if valid_xtals else None
                        xs += [result.cx for result in valid_xtals]
                        ys += [result.cy for result in valid_xtals]
                        scores += [result.score for result in valid_xtals]
                        types += [object_type for _ in valid_xtals]
                    elif object_type == ObjectType.LOOP:
                        # Loop and crystal are centered in the bounding box
                        xs += [result.cx for result in res_list]
                        ys += [result.cy for result in res_list]
                        scores += [result.score for result in res_list]
                        types += [object_type for _ in res_list]
                    elif object_type == ObjectType.PIN:
                        # Pin centered at 90% horizontally, and 50% vertically
                        xs += [result.x1 + int(result.w * 0.9) for result in res_list]
                        ys += [result.y1 + int(result.h * 0.5) for result in res_list]
                        scores += [result.score for result in res_list]
                        types += [object_type for _ in res_list]

                if xs:
                    self.ioc.objects_x.put(numpy.array(xs).astype(int))
                    self.ioc.objects_y.put(numpy.array(ys).astype(int))
                    self.ioc.objects_score.put(numpy.array(scores).astype(float))
                    self.ioc.objects_type.put(numpy.array(types))
                    self.ioc.objects_valid.put(len(xs))
                else:
                    self.ioc.objects_valid.put(0)

                if best_crystal:
                    X1 = numpy.array(best_crystal.box(), dtype=int)
                    X0 = self.ioc.track_input.get()
                    if len(X0) == 0 or numpy.allclose(X1, X0, rtol=0, atol=10):
                        self.ioc.track_input.put(X1)

                    self.ioc.track_valid.put(StatusType.VALID)
                else:
                    self.ioc.track_valid.put(StatusType.INVALID)
            else:
                self.ioc.status.put(StatusType.INVALID)
                self.ioc.score.put(0.0)

            if self.sam and self.ioc.track.get() == EnableType.ENABLED:
                tracked_result = None
                if self.sam.tracked_object:
                    tracked_result = self.process_tracking(frame)
                elif self.ioc.track_valid.get():
                    x1, y1, x2, y2 = self.ioc.track_input.get()
                    to_track = Result(type='crystal', x1=x1, y1=y1, x2=x2, y2=y2)
                    tracked_result = self.process_tracking(frame, to_track)

                if tracked_result:
                    self.ioc.track_output.put(tracked_result.box())
                    self.ioc.track_score.put(tracked_result.score)

            #time.sleep(0.01)

    def do_enable(self, pv, value, ioc):
        self.enabled = (value == EnableType.ENABLED)

    def do_track(self, pv, value, ioc):
        self.tracking = (value == EnableType.ENABLED)
        if not self.tracking:
            self.ioc.track_output.put([0, 0, 0, 0])
            self.ioc.track_score.put(0.0)

    def shutdown(self):
        # needed for proper IOC shutdown
        self.running = False
        self.ioc.shutdown()
