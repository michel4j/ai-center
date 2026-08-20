import threading
import time
import warnings

import numpy

from aicenter import Result

warnings.filterwarnings("ignore")

from enum import IntEnum

from devioc import models, log
import gepics

from . import AiCenter

logger = log.get_module_logger('aicenter')


class EnableType(IntEnum):
    DISABLED, ENABLED = range(2)


class StatusType(IntEnum):
    INVALID, VALID = range(2)


class ObjectType(IntEnum):
    NONE, LOOP, CRYSTAL, PIN = range(4)


# Create your models here. Modify the example below as appropriate
class AiCenterModel(models.Model):
    enable = models.Enum('enable', choices=EnableType, default=1, mdel=0, desc="Enable/Disable")

    # Loop
    loop_box = models.Array('loop:box', type=int, length=4, desc="Loop Coordinates")
    loop_score = models.Float('loop:score', default=0.0, mdel=0, desc='Loop Score')
    loop_id = models.Integer('loop:id', default=0, mdel=0, desc='Loop ID')
    loop_valid = models.Enum('loop:valid', choices=StatusType, default=StatusType.INVALID, desc="Loop Valid")

    # Crystal
    crystal_box = models.Array('crystal:box', type=int, length=4, desc="Crystal Coordinates")
    crystal_score = models.Float('crystal:score', default=0.0, mdel=0, desc='Crystal Score')
    crystal_id = models.Integer('crystal:id', default=0, mdel=0, desc='Crystal ID')
    crystal_valid = models.Enum('crystal:valid', choices=StatusType, default=StatusType.INVALID, desc="Crystal Valid")

    # Extra Crystals
    crystals = models.Array('extra:box', type=int, desc="Extra Boxes")
    scores = models.Array('extra:score', type=float, desc="Extra Scores")
    extra_ids = models.Array('extra:id', type=int, desc="Extra IDs")
    num_crystals = models.Integer('extra:valid', default=0, min_val=0, max_val=64, desc="Number of Extra Crystals")

    # Pin
    pin_box = models.Array('pin:box', type=int, length=4, desc="Pin Coordinates")
    pin_score = models.Float('pin:score', default=0.0, mdel=0, desc='Pin Score')
    pin_id = models.Integer('pin:id', default=0, mdel=0, desc='Pin ID')
    pin_valid = models.Enum('pin:valid', choices=StatusType, default=StatusType.INVALID, desc="Pin Valid")


class IOCApp(AiCenter):
    def __init__(self, device, model, video, threshold=None):
        """
        AiCenter IOC
        :param device:  device root name for PVs
        :param model:  YOLO Model path
        :param video:  Video URI
        """
        super().__init__(model=model, video=video, threshold=threshold, tracking=True)
        logger.info(f'device={device!r}, model={model!r}, video={video!r}')
        self.running = False
        self.enabled = True
        self.tracking = False
        self.ioc = AiCenterModel(device, callbacks=self)
        self.pvs = {
            'loop': (
                self.ioc.loop_box,
                self.ioc.loop_score,
                self.ioc.loop_valid,
                self.ioc.loop_id
            ),
            'crystal': (
                self.ioc.crystal_box,
                self.ioc.crystal_score,
                self.ioc.crystal_valid,
                self.ioc.crystal_id
            ),
            'pin': (
                self.ioc.pin_box,
                self.ioc.pin_score,
                self.ioc.pin_valid,
                self.ioc.pin_id
            ),
        }
        self.start_monitor()

    def start_monitor(self):
        self.running = False
        monitor_thread = threading.Thread(target=self.video_monitor, daemon=True)
        monitor_thread.start()

    def video_monitor(self):
        gepics.threads_init()
        self.running = True
        while self.running:
            if self.ioc.enable.get() != EnableType.ENABLED:
                self.ioc.loop_valid.put(StatusType.INVALID)
                self.ioc.crystal_valid.put(StatusType.INVALID)
                self.ioc.pin_valid.put(StatusType.INVALID)
                self.ioc.num_crystals.put(0)
                time.sleep(0.1)
                continue

            frame = self.get_frame()
            results = self.process_frame(frame)

            for label, objects in results.items():

                if not objects:
                    validity = StatusType.INVALID
                    best = Result(type=label, x1=0, y1=0, x2=0, y2=0, score=0)
                    extra = []
                else:
                    validity = StatusType.VALID
                    best = objects[0]
                    extra = objects[1:]

                box_pv, score_pv, valid_pv, id_pv = self.pvs.get(label, (None, None, None, None))
                if box_pv and score_pv and valid_pv and id_pv:
                    valid_pv.put(validity)  # Put this first
                    score_pv.put(best.score)
                    id_pv.put(best.id)
                    box_pv.put(numpy.array([best.x1, best.y1, best.x2, best.y2]).astype(int))

                if label == 'crystal' and extra:
                    num_crystals = len(extra)
                    boxes = numpy.array([[obj.x1, obj.y1, obj.x2, obj.y2] for obj in extra]).ravel().astype(int)
                    scores = numpy.array([obj.score for obj in extra]).ravel()
                    ids = numpy.array([obj.id for obj in extra]).ravel()
                    self.ioc.num_crystals.put(num_crystals)  # put this first
                    self.ioc.scores.put(scores)
                    self.ioc.extra_ids.put(ids)
                    self.ioc.crystals.put(boxes)
                else:
                    self.ioc.num_crystals.put(0)

                if label == 'crystal' and best.score:
                    pass

    def do_enable(self, pv, value, ioc):
        self.enabled = (value == EnableType.ENABLED)

    def shutdown(self):
        # needed for proper IOC shutdown
        self.running = False
        self.ioc.shutdown()
