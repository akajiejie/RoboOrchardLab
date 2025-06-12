# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, TypeVar

import torch
from foxglove_schemas_protobuf.CameraCalibration_pb2 import (
    CameraCalibration as FgCameraCalibration,
)
from foxglove_schemas_protobuf.CompressedImage_pb2 import (
    CompressedImage as FgCompressedImage,
)
from foxglove_schemas_protobuf.FrameTransform_pb2 import (
    FrameTransform as FgFrameTransform,
)
from google.protobuf.timestamp_pb2 import Timestamp
from pydantic import SkipValidation
from robo_orchard_core.datatypes.camera_data import BatchCameraData, Distortion

from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    ClassConfig,
    MessageConverterConfig,
    MessageConverterStateless,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.compressed_img import (  # noqa: E501
    CompressedImage2NumpyConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.frame_transform import (  # noqa: E501
    ToBatchFrameTransformStampedConfig,
)

__all__ = [
    "FgCameraCompressedImages",
    "BatchCameraDataWithTimestamps",
    "ToBatchCameraData",
    "ToBatchCameraDataConfig",
    "CameraDataConfigMixin",
]


@dataclass
class FgCameraCompressedImages:
    images: list[FgCompressedImage]
    """List of compressed images."""

    calib: FgCameraCalibration | None = None
    """Calibration data associated with the images."""

    tf: FgFrameTransform | list[FgFrameTransform] | None = None
    """Frame transform associated with the images."""

    def __post_init__(self):
        if self.tf is not None and not isinstance(self.tf, list):
            for img in self.images:
                if img.frame_id != self.tf.child_frame_id:
                    raise ValueError(
                        "All images must have the same frame_id as the "
                        "child_frame_id of the FrameTransform."
                    )
        if isinstance(self.tf, list):
            for img, tf in zip(self.images, self.tf, strict=True):
                if img.frame_id != tf.child_frame_id:
                    raise ValueError(
                        "All images must have the same frame_id as the "
                        "child_frame_id of the FrameTransform."
                    )


class BatchCameraDataWithTimestamps(BatchCameraData):
    """BatchCameraData with an associated timestamp."""

    timestamps: list[SkipValidation[Timestamp] | None] | None = None
    """Timestamp for the batch of camera data."""


class ToBatchCameraData(
    MessageConverterStateless[
        FgCameraCompressedImages, BatchCameraDataWithTimestamps
    ]
):
    """Convert FgCameraCompressedImages to BatchCameraDataWithTimestamps."""

    def __init__(self, cfg: ToBatchCameraDataConfig | None = None):
        """Initialize the converter."""
        if cfg is None:
            cfg = ToBatchCameraDataConfig()

        if cfg.pix_fmt is None:
            cfg.pix_fmt = "bgr"

        self._cfg = cfg
        self._to_numpy = CompressedImage2NumpyConfig()()
        self._to_tf = ToBatchFrameTransformStampedConfig()()

    def convert(
        self, src: FgCameraCompressedImages
    ) -> BatchCameraDataWithTimestamps:
        np_imgs = [self._to_numpy.convert(img) for img in src.images]
        timestamps = []
        for img in np_imgs:
            if img.frame_id != np_imgs[0].frame_id:
                raise ValueError(
                    "All images must have the same frame_id, "
                    f"but got {img.frame_id} and {np_imgs[0].frame_id}."
                )
            timestamps.append(img.timestamp)

        if all(timestamp is None for timestamp in timestamps):
            timestamps = None

        batch_img_tensor = torch.stack(
            [torch.from_numpy(img.data) for img in np_imgs]
        )
        ret = BatchCameraDataWithTimestamps(
            sensor_data=batch_img_tensor,
            frame_id=np_imgs[0].frame_id,
            pix_fmt=self._cfg.pix_fmt,
            image_shape=(batch_img_tensor.shape[1], batch_img_tensor.shape[2]),
            timestamps=timestamps,
        )
        if self._cfg.pix_fmt == "rgb":
            # Convert BGR to RGB if needed
            ret.sensor_data = ret.sensor_data[..., ::-1]

        self._set_camera_calib(calib=src.calib, target=ret)
        self._set_tf(tf=src.tf, target=ret, timestamps=timestamps)
        return ret

    def _set_tf(
        self,
        tf: FgFrameTransform | list[FgFrameTransform] | None,
        timestamps: list[Timestamp | None] | None,
        target: BatchCameraData,
    ):
        """Set the frame transform."""
        if tf is None:
            target.pose = None
            return
        target.pose = self._to_tf.convert(tf)

        if target.pose.batch_size == 1 and target.batch_size > 1:
            target.pose = target.pose.repeat(
                target.batch_size, timestamps=timestamps
            )

    def _set_camera_calib(
        self, calib: FgCameraCalibration | None, target: BatchCameraData
    ) -> None:
        """Set the camera calibration."""
        if calib is None:
            return

        batch_size = target.batch_size

        if target.image_shape is None:
            target.image_shape = (
                calib.height,
                calib.width,
            )
        else:
            if target.image_shape != (calib.height, calib.width):
                raise ValueError(
                    f"Image shape {target.image_shape} does not match "
                    f"calibration shape {(calib.height, calib.width)}."
                )
        target.intrinsic_matrices = (
            torch.tensor(calib.K, dtype=torch.float32)
            .reshape(-1, 3, 3)
            .repeat(batch_size, 1, 1)
        )

        target.distortion = Distortion(
            model=calib.distortion_model,  # type: ignore
            coefficients=torch.tensor(calib.D, dtype=torch.float32),
        )


T = TypeVar("T")


class CameraDataConfigMixin(ClassConfig[T]):
    pix_fmt: Literal["rgb", "bgr", "depth"] | None = None
    """Pixel format for the input images.

    For openCV implementation, color images are expected to be in BGR format.
    For PIL implementation, color images are expected to be in RGB format.

    pix_fmt should be "depth" for depth images, which are expected to be in
    single channel format (e.g., float32 or int32 for depth values).

    Note:
        CameraMsgs2BatchCameraData does not check the pixel format of the
        input images. It only sets the pixel format in the output.

    """


class ToBatchCameraDataConfig(
    MessageConverterConfig[ToBatchCameraData],
    CameraDataConfigMixin[ToBatchCameraData],
):
    """Configuration class for CameraMsgs2BatchCameraData."""

    class_type: type[ToBatchCameraData] = ToBatchCameraData
