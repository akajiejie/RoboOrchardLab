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

import torch
from foxglove_schemas_protobuf.FrameTransform_pb2 import (
    FrameTransform as FgFrameTransform,
)
from google.protobuf.timestamp_pb2 import Timestamp
from pydantic import SkipValidation
from robo_orchard_core.datatypes.camera_data import BatchFrameTransform
from robo_orchard_core.utils.torch_utils import dtype_str2torch

from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverterConfig,
    MessageConverterStateless,
    TensorTargetConfigMixin,
)

__all__ = [
    "BatchFrameTransformStamped",
    "ToBatchFrameTransformStamped",
    "ToBatchFrameTransformStampedConfig",
]


class BatchFrameTransformStamped(BatchFrameTransform):
    timestamps: list[SkipValidation[Timestamp] | None] | None = None
    """Timestamp for the frame transform"""

    def repeat(
        self, batch_size: int, timestamps: list[Timestamp | None] | None
    ) -> BatchFrameTransform:
        if timestamps is not None:
            assert len(timestamps) == batch_size, (
                f"Batch size {batch_size} does not match number of "
                f"timestamps {len(timestamps)}."
            )
        ret = super().repeat(batch_size)
        return BatchFrameTransformStamped(
            child_frame_id=ret.child_frame_id,
            parent_frame_id=ret.parent_frame_id,
            xyz=ret.xyz,
            quat=ret.quat,
            timestamps=timestamps,
        )


class ToBatchFrameTransformStamped(
    MessageConverterStateless[
        FgFrameTransform | list[FgFrameTransform],
        BatchFrameTransformStamped,
    ]
):
    """Convert a Foxglove FrameTransform message to a FrameTransform Type."""

    def __init__(
        self,
        cfg: ToBatchFrameTransformStampedConfig,
    ):
        self._cfg = cfg
        self._dtype = dtype_str2torch(cfg.dtype)

    def convert(
        self, src: FgFrameTransform | list[FgFrameTransform]
    ) -> BatchFrameTransformStamped:
        if not isinstance(src, list):
            tf_trans = torch.tensor(
                [src.translation.x, src.translation.y, src.translation.z],
                dtype=self._dtype,
                device=self._cfg.device,
            )
            tf_rot = torch.tensor(
                [
                    src.rotation.w,
                    src.rotation.x,
                    src.rotation.y,
                    src.rotation.z,
                ],
                dtype=self._dtype,
                device=self._cfg.device,
            )

            return BatchFrameTransformStamped(
                child_frame_id=src.child_frame_id,
                parent_frame_id=src.parent_frame_id,
                xyz=tf_trans,
                quat=tf_rot,
                timestamps=[src.timestamp],
            )
        else:
            assert len(src) > 0, "List of FrameTransform cannot be empty."
            tf_trans = torch.zeros(
                (len(src), 3),
                dtype=self._dtype,
            )
            tf_rot = torch.zeros(
                (len(src), 4),
                dtype=self._dtype,
            )
            for i, tf in enumerate(src):
                tf_trans[i, :] = torch.tensor(
                    [tf.translation.x, tf.translation.y, tf.translation.z],
                    dtype=self._dtype,
                )
                tf_rot[i, :] = torch.tensor(
                    [
                        tf.rotation.w,
                        tf.rotation.x,
                        tf.rotation.y,
                        tf.rotation.z,
                    ],
                    dtype=self._dtype,
                )
            return BatchFrameTransformStamped(
                child_frame_id=src[0].child_frame_id,
                parent_frame_id=src[0].parent_frame_id,
                xyz=tf_trans.to(device=self._cfg.device),
                quat=tf_rot.to(device=self._cfg.device),
                timestamps=[tf.timestamp for tf in src],
            )


class ToBatchFrameTransformStampedConfig(
    MessageConverterConfig[ToBatchFrameTransformStamped],
    TensorTargetConfigMixin[ToBatchFrameTransformStamped],
):
    class_type: type[ToBatchFrameTransformStamped] = (
        ToBatchFrameTransformStamped
    )
