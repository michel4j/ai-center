from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy

from aicenter.log import get_module_logger
from aicenter.net import Result

logger = get_module_logger(__name__)

try:
    import torch
    from .lib.make_sam_v2 import make_samv2_from_original_state_dict
except ModuleNotFoundError as e:
    logger.error(f"Missing SAM2 import: {e}")
    raise e


@dataclass
class MaskResult(Result):
    mask: numpy.ndarray = None
    contours: numpy.ndarray = None


class SAM2:
    predictor: Any
    device: torch.device

    def __init__(self, model_path: Path | str):
        self.model = Path(model_path)
        self.setup_device()
        self.predictor = self.setup_predictor()
        logger.info(f"Loading sam model from {self.model}")

    def setup_device(self):
        # select the device for computation
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.debug(f"Using device: {self.device}")

        if self.device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def setup_predictor(self):
        pass

    def predict_input_boxes(self, image: numpy.ndarray, input_boxes: numpy.ndarray):
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=None, point_labels=None, box=input_boxes, multimask_output=False,
        )
        return masks, scores

    @staticmethod
    def process_result(mask, score, obj: TrackedObject):

        result = None
        score = score.item()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours):
            # Calculate image moments of the detected contour
            moments = cv2.moments(contours[0])
            try:
                x_centroid = int(round(moments['m10'] / moments['m00']))
                y_centroid = int(round(moments['m01'] / moments['m00']))
            except ZeroDivisionError:
                x_centroid = -1
                y_centroid = -1

            else:
                logger.debug(f"Segmentation mask centroid: {x_centroid}, {y_centroid}")

            x, y, w, h = cv2.boundingRect(contours[0])
            logger.info(f'Tracked {obj.source.type} found at: {x} {y} [{w} {h}], prob={score:.2f}')
            result = MaskResult(
                type=obj.source.type,
                x1=x, y1=y, x2=x+w, y2=y+h,
                score=score, cx=x_centroid,
                cy=y_centroid, mask=mask, contours=contours
            )

        return result


@dataclass
class TrackedObject:
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    prompt_memory_encodings: list[torch.Tensor] = field(default_factory=list)
    prompt_object_pointers: list[torch.Tensor] = field(default_factory=list)
    prev_memory_encodings: deque[torch.Tensor] = field(default_factory=deque)
    prev_object_pointers: deque[torch.Tensor] = field(default_factory=deque)
    source: Result = None

    def __post_init__(self):
        # Set max lengths of previous knowledge
        self.prev_memory_encodings = deque([], maxlen=15)
        self.prev_object_pointers = deque([], maxlen=15)

    @property
    def video_masking_inputs(self):
        return self.prompt_memory_encodings, self.prompt_object_pointers, self.prev_memory_encodings, self.prev_object_pointers


class TrackingSAM(SAM2):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.tracked_object = None

    def setup_predictor(self):
        _, sam_model = make_samv2_from_original_state_dict(str(self.model))
        sam_model.to(device=self.device)
        return sam_model

    def track_object(self, image: numpy.ndarray, obj: Result, width, height):
        if not self.predictor:
            return
        norm = numpy.array([width, height, width, height])
        box = numpy.array([obj.x1, obj.y1, obj.x2, obj.y2]) / norm
        box = box.reshape(-1, 2, 2)
        init_encoded_img, _, _ = self.predictor.encode_image(image)
        init_mask, init_mem, init_ptr = self.predictor.initialize_video_masking(
            init_encoded_img, box, [], []
        )
        self.tracked_object = TrackedObject(
            prompt_memory_encodings=[init_mem],
            prompt_object_pointers=[init_ptr],
            source=obj
        )
        logger.debug(f"Added new tracked object")

    def predict(self, image: numpy.ndarray):
        mask = None
        score = None

        if self.tracked_object:
            # Process video frames with model
            t1 = time.perf_counter()
            encoded_imgs_list, _, _ = self.predictor.encode_image(image)
            obj_score, mask_pred, mem_enc, obj_ptr = self.predictor.step_video_masking(
                encoded_imgs_list, *self.tracked_object.video_masking_inputs,
            )
            t2 = time.perf_counter()
            logger.info(f"Inference took {round(1000 * (t2 - t1))} ms, score={obj_score[0][0]:.2f}")

            # Store object results for future frames
            if obj_score < 0:
                # TODO Need to keep object for occlusion support, maybe drop when scores stay low for a long time
                # Suggested handling for occluded objects is to keep tracking them but do not update prev_memory_encodings/object_pointers
                logger.info(f"Bad object score {float(obj_score):0.3f}! Implies broken tracking! Dropping tracked object.")
                self.tracked_object = None
            else:
                self.tracked_object.prev_memory_encodings.appendleft(mem_enc)
                self.tracked_object.prev_object_pointers.appendleft(obj_ptr)

                # Create mask for display
                dispres_mask = torch.nn.functional.interpolate(
                    mask_pred, size=image.shape[0:2], mode="bilinear", align_corners=False,
                )
                mask = ((dispres_mask > 0.0).byte() * 255).cpu().numpy().squeeze()
                score = obj_score
                logger.debug('Tracked object mask calculated')

        return mask, score, self.tracked_object


def show_masks(image, mask, random_color=False, borders=True, centroid=True, bbox=True):
    if random_color:
        rng = numpy.random.default_rng()
        color = rng.integers(0, 255, size=3, dtype=numpy.uint8)
    else:
        color = numpy.array([30 / 255, 144 / 255, 255 / 255], dtype=numpy.uint8)
    h, w = mask.shape[-2:]
    mask = mask.astype(numpy.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders or centroid or bbox:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = max(contours, key=cv2.contourArea) if len(contours) > 0 else None
        if borders:
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (0, 255, 0, 0.5), thickness=2)

        if centroid and max_contour is not None:
            # Calculate image moments of the detected contour
            moments = cv2.moments(max_contour)
            try:
                x_centroid = round(moments['m10'] / moments['m00'])
                y_centroid = round(moments['m01'] / moments['m00'])
            except ZeroDivisionError:
                pass
            else:
                logger.debug(f"Segmentation mask centroid: {x_centroid}, {y_centroid}")
                # Draw a marker centered at centroid coordinates
                image = cv2.drawMarker(image, (x_centroid, y_centroid), (255, 0, 0, 1), thickness=1, markerSize=20)

        if bbox and max_contour is not None:
            rect = cv2.boundingRect(max_contour)
            image = cv2.rectangle(
                image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0, 1), thickness=1
            )

    return cv2.addWeighted(image, 1, mask_image, 1, 0)


def show_mask_from_result(image, result: MaskResult, random_color=False, borders=True, centroid=True, bbox=False):
    if result.mask is None:
        return image

    if random_color:
        rng = numpy.random.default_rng()
        color = rng.integers(0, 255, size=3, dtype=numpy.uint8)
    else:
        color = numpy.array([30 / 255, 144 / 255, 255 / 255], dtype=numpy.uint8)
    mask = result.mask
    contours = result.contours
    h, w = mask.shape[-2:]
    mask = mask.astype(numpy.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders or centroid or bbox:
        if borders:
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (0, 255, 0, 0.5), thickness=2)
        if centroid:
            image = cv2.drawMarker(image, (result.cx, result.cy), (255, 0, 0, 1), thickness=1, markerSize=20)
        if bbox:
            x1, y1, x2, y2 = result.box()
            image = cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 0, 0, 1), thickness=1
            )
    return cv2.addWeighted(image, 1, mask_image, 1, 0)
