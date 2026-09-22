# Copyright (c) Meta Platforms, Inc. and affiliates.
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to

from articulate.shape_estimator.sam3d_customized.sam_3d_body_estimator import (
    SAM3DBodyEstimator,
)


@dataclass
class PreparedSAM3DInput:
    """One detected and transformed image ready for body-model inference."""

    image: np.ndarray
    batch: Dict


@dataclass
class SAM3DImagePreparation:
    """CPU-preprocessed image and optional input for batched human detection."""

    image: np.ndarray
    detector_input: Optional[Dict]
    cam_int: torch.Tensor


class BatchedSAM3DBodyEstimator(SAM3DBodyEstimator):
    """Local SAM 3D Body wrapper for batching one detected person per image."""

    @staticmethod
    def _prepare_detectron2_input(img: np.ndarray) -> Dict:
        import detectron2.data.transforms as transforms

        height, width = img.shape[:2]
        resize = transforms.ResizeShortestEdge(
            short_edge_length=1024,
            max_size=1024,
        )
        resized = resize(transforms.AugInput(img)).apply_image(img)
        image = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
        return {
            "image": image,
            "height": height,
            "width": width,
        }

    def preprocess_one_image(
        self,
        img,
        cam_int: torch.Tensor,
    ) -> SAM3DImagePreparation:
        """Prepare image formats and the CPU-side ViTDet input."""
        if isinstance(img, str):
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            image_format = "rgb"

        if image_format == "rgb":
            image = img
            detector_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            detector_image = img
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detector_input = None
        if self.detector is not None:
            detector_input = self._prepare_detectron2_input(detector_image)

        return SAM3DImagePreparation(
            image=image,
            detector_input=detector_input,
            cam_int=cam_int,
        )

    @staticmethod
    def _boxes_from_detection(
        detection,
        *,
        det_cat_id: int,
        bbox_thr: float,
        default_to_full_image: bool,
        height: int,
        width: int,
    ) -> np.ndarray:
        instances = detection["instances"]
        valid = (instances.pred_classes == det_cat_id) & (instances.scores > bbox_thr)
        if valid.sum() == 0 and default_to_full_image:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
        else:
            boxes = instances.pred_boxes.tensor[valid].cpu().numpy()

        sorted_indices = np.lexsort(
            (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
        )
        return boxes[sorted_indices]

    def detect_preprocessed_batch(
        self,
        prepared: Sequence[SAM3DImagePreparation],
        *,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
    ) -> List[np.ndarray]:
        """Run one ViTDet forward pass for several independently rendered views."""
        del nms_thr
        if not prepared:
            return []
        if self.detector is None:
            return [
                np.array([0, 0, item.image.shape[1], item.image.shape[0]]).reshape(1, 4)
                for item in prepared
            ]

        detector_model = getattr(self.detector, "detector", None)
        if detector_model is None:
            raise TypeError("The configured human detector does not support batching")
        if any(item.detector_input is None for item in prepared):
            raise ValueError("Missing a preprocessed detector input")

        with torch.no_grad():
            detections = detector_model([item.detector_input for item in prepared])
        if len(detections) != len(prepared):
            raise RuntimeError(
                "Human detector batch output count does not match its inputs"
            )

        boxes = []
        for item, detection in zip(prepared, detections):
            height, width = item.image.shape[:2]
            boxes.append(
                self._boxes_from_detection(
                    detection,
                    det_cat_id=det_cat_id,
                    bbox_thr=bbox_thr,
                    default_to_full_image=False,
                    height=height,
                    width=width,
                )
            )
        return boxes

    def finalize_preprocessed_image(
        self,
        prepared: SAM3DImagePreparation,
        boxes: np.ndarray,
    ) -> PreparedSAM3DInput:
        """Crop and transform one detected view for SAM 3D Body inference."""
        if len(boxes) != 1:
            raise ValueError(
                "Batched SAM 3D Body inference requires exactly one detected "
                f"person per image, found {len(boxes)}"
            )
        batch = prepare_batch(
            prepared.image,
            self.transform,
            boxes,
            cam_int=prepared.cam_int,
        )
        return PreparedSAM3DInput(image=prepared.image, batch=batch)

    def prepare_one_image(
        self,
        img,
        cam_int: torch.Tensor,
        *,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
    ) -> PreparedSAM3DInput:
        if isinstance(img, str):
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            image_format = "rgb"

        height, width = img.shape[:2]
        if self.detector is not None:
            detector_image = img
            if image_format == "rgb":
                detector_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            boxes = self.detector.run_human_detection(
                detector_image,
                det_cat_id=det_cat_id,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                default_to_full_image=False,
            )
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)

        if len(boxes) != 1:
            raise ValueError(
                "Batched SAM 3D Body inference requires exactly one detected "
                f"person per image, found {len(boxes)}"
            )

        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        batch = prepare_batch(
            img,
            self.transform,
            boxes,
            cam_int=cam_int,
        )
        return PreparedSAM3DInput(image=img, batch=batch)

    @staticmethod
    def _stack_prepared_batches(prepared: Sequence[PreparedSAM3DInput]) -> Dict:
        if not prepared:
            raise ValueError("Cannot stack an empty SAM 3D Body batch")

        keys = prepared[0].batch.keys()
        if any(item.batch.keys() != keys for item in prepared[1:]):
            raise ValueError("Prepared SAM 3D Body batches have different fields")

        stacked = {}
        for key in keys:
            values = [item.batch[key] for item in prepared]
            if torch.is_tensor(values[0]):
                stacked[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], list):
                stacked[key] = [entry for value in values for entry in value]
            else:
                raise TypeError(
                    f"Unsupported prepared batch field {key!r}: "
                    f"{type(values[0]).__name__}"
                )
        return stacked

    @staticmethod
    def _format_body_outputs(batch: Dict, pose_output: Dict) -> List[Dict]:
        out = recursive_to(pose_output["mhr"], "cpu")
        out = recursive_to(out, "numpy")
        flat_bbox = batch["bbox"].flatten(0, 1).detach().cpu().numpy()
        count = batch["img"].shape[0] * batch["img"].shape[1]

        if len(out["pred_vertices"]) != count:
            raise RuntimeError(
                "SAM 3D Body returned an unexpected number of predictions: "
                f"expected {count}, got {len(out['pred_vertices'])}"
            )

        all_out = []
        for index in range(count):
            all_out.append(
                {
                    "bbox": flat_bbox[index],
                    "focal_length": out["focal_length"][index],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][index],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][index],
                    "pred_vertices": out["pred_vertices"][index],
                    "pred_cam_t": out["pred_cam_t"][index],
                    "pred_pose_raw": out["pred_pose_raw"][index],
                    "global_rot": out["global_rot"][index],
                    "body_pose_params": out["body_pose"][index],
                    "hand_pose_params": out["hand"][index],
                    "scale_params": out["scale"][index],
                    "shape_params": out["shape"][index],
                    "expr_params": out["face"][index],
                    "mask": None,
                    "pred_joint_coords": out["pred_joint_coords"][index],
                    "pred_global_rots": out["joint_global_rots"][index],
                    "mhr_model_params": out["mhr_model_params"][index],
                }
            )
        return all_out

    @torch.no_grad()
    def infer_prepared_batch(
        self,
        prepared: Sequence[PreparedSAM3DInput],
    ) -> List[Dict]:
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []

        batch = self._stack_prepared_batches(prepared)
        batch = recursive_to(batch, self.device)
        self.model._initialize_batch(batch)
        pose_output = self.model.run_inference(
            prepared[0].image,
            batch,
            inference_type="body",
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        return self._format_body_outputs(batch, pose_output)

    def infer_prepared_one(self, prepared: PreparedSAM3DInput) -> Dict:
        outputs = self.infer_prepared_batch([prepared])
        if len(outputs) != 1:
            raise RuntimeError(
                f"Expected one SAM 3D Body output, received {len(outputs)}"
            )
        return outputs[0]
