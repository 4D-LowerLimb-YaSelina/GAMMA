from typing import Dict

import cv2
import numpy as np
import torch
from skimage.filters import gaussian
from yacs.config import CfgNode

from amr.datasets.utils import (
    convert_cvimg_to_tensor,
    expand_to_aspect_ratio,
    generate_image_patch_cv2,
)


class AniMerInferenceDataset(torch.utils.data.Dataset):
    """AniMer inference crops without the upstream per-crop debug print."""

    def __init__(
        self,
        cfg: CfgNode,
        img_cv2: np.ndarray,
        boxes: np.ndarray,
        rescale_factor=1,
        train: bool = False,
        **kwargs,
    ):
        super().__init__()
        if train:
            raise ValueError("AniMerInferenceDataset is only for inference")

        self.cfg = cfg
        self.img_cv2 = img_cv2
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std = 255.0 * np.array(cfg.MODEL.IMAGE_STD)

        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.animalid = np.arange(len(boxes), dtype=np.int32)

    def __len__(self):
        return len(self.animalid)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        center = self.center[idx].copy()
        bbox_shape = self.cfg.MODEL.get("BBOX_SHAPE", None)
        bbox_size = expand_to_aspect_ratio(
            self.scale[idx] * 200,
            target_aspect_ratio=bbox_shape,
        ).max()

        cvimg = self.img_cv2.copy()
        downsampling_factor = (bbox_size / self.img_size) / 2.0
        if downsampling_factor > 1.1:
            cvimg = gaussian(
                cvimg,
                sigma=(downsampling_factor - 1) / 2,
                channel_axis=2,
                preserve_range=True,
            )

        img_patch_cv, _, _ = generate_image_patch_cv2(
            cvimg,
            center[0],
            center[1],
            bbox_size,
            bbox_size,
            self.img_size,
            self.img_size,
            False,
            1.0,
            0.0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        img_patch = convert_cvimg_to_tensor(img_patch_cv[:, :, ::-1])

        for channel in range(min(self.img_cv2.shape[2], 3)):
            img_patch[channel] = (
                img_patch[channel] - self.mean[channel]
            ) / self.std[channel]

        return {
            "img": img_patch,
            "animalid": int(self.animalid[idx]),
            "box_center": center,
            "box_size": bbox_size,
            "img_size": np.array([cvimg.shape[1], cvimg.shape[0]], dtype=float),
            "focal_length": np.array(
                [self.cfg.EXTRA.FOCAL_LENGTH, self.cfg.EXTRA.FOCAL_LENGTH]
            ),
        }
