# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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
from foxglove_schemas_protobuf.Pose_pb2 import Pose as FgPose
from foxglove_schemas_protobuf.PoseInFrame_pb2 import (
    PoseInFrame as FgPoseInFrame,
)
from foxglove_schemas_protobuf.PosesInFrame_pb2 import (
    PosesInFrame as FgPosesInFrame,
)
from google.protobuf.timestamp_pb2 import Timestamp
from pydantic import SkipValidation
from robo_orchard_core.datatypes.geometry import BatchPose
from robo_orchard_core.utils.torch_utils import dtype_str2torch

from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverterConfig,
    MessageConverterStateless,
    TensorTargetConfigMixin,
)
from robo_orchard_lab.utils.protobuf import (
    is_list_of_protobuf_msg_type,
    is_protobuf_msg_type,
)

__all__ = [
    "BatchPoseStamped",
    "ToBatchPoseStamped",
    "ToBatchPoseStampedConfig",
]


class BatchPoseStamped(BatchPose):
    timestamps: list[SkipValidation[Timestamp] | None] | None = None
    """Timestamp for the joint states"""

    def __post_init__(self):
        super().__post_init__()
        if self.timestamps is not None:
            assert len(self.timestamps) == self.batch_size, (
                f"Batch size {self.batch_size} does not match number of "
                f"timestamps {len(self.timestamps)}."
            )


ToBatchPose_SRC_TYPE = FgPoseInFrame | FgPosesInFrame | list[FgPoseInFrame]


class ToBatchPoseStamped(
    MessageConverterStateless[ToBatchPose_SRC_TYPE, BatchPoseStamped]
):
    def __init__(self, cfg: ToBatchPoseStampedConfig):
        super().__init__()
        self._cfg = cfg
        self._dtype = dtype_str2torch(cfg.dtype)

    def convert(self, src: ToBatchPose_SRC_TYPE) -> BatchPoseStamped:
        ts_list: list[Timestamp] = []
        pose_list: list[FgPose] = []
        frame_id = ""
        if isinstance(src, list):
            if not is_list_of_protobuf_msg_type(src, FgPoseInFrame):
                raise TypeError(
                    f"Expected list of {FgPoseInFrame.__name__}, "
                    f"got {type(src)}."
                )
            for i, pose_in_frame in enumerate(src):
                ts_list.append(pose_in_frame.timestamp)
                pose_list.append(pose_in_frame.pose)
                assert pose_in_frame.frame_id == src[0].frame_id, (
                    f"All poses in the list must have the same frame_id, "
                    f"but pose {i} has frame_id {pose_in_frame.frame_id} "
                    f"while the first pose has frame_id {src[0].frame_id}."
                )
            frame_id = src[0].frame_id
        elif is_protobuf_msg_type(src, FgPoseInFrame):
            frame_id = src.frame_id
            ts_list.append(src.timestamp)
            pose_list.append(src.pose)
        elif is_protobuf_msg_type(src, FgPosesInFrame):
            frame_id = src.frame_id
            ts_list.append(src.timestamp)
            pose_list.extend(src.poses)
        else:
            raise TypeError(
                f"Expected {FgPoseInFrame.__name__} or "
                f"{FgPosesInFrame.__name__}, got {type(src)}."
            )

        if len(ts_list) == 1 and len(pose_list) > 1:
            # If only one timestamp but multiple poses, repeat the timestamp
            for _ in range(len(pose_list) - 1):
                ts = Timestamp()
                ts.CopyFrom(ts_list[0])
                ts_list.append(ts)

        ret = BatchPoseStamped(
            xyz=torch.zeros(
                size=(len(pose_list), 3),
                dtype=self._dtype,
            ),
            quat=torch.zeros(
                size=(len(pose_list), 4),
                dtype=self._dtype,
            ),
            frame_id=frame_id,
            timestamps=ts_list,  # type: ignore
        )

        for i, pose in enumerate(pose_list):
            ret.xyz[i, :] = torch.tensor(
                [pose.position.x, pose.position.y, pose.position.z],
                dtype=self._dtype,
            )
            ret.quat[i, :] = torch.tensor(
                [
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                ],
                dtype=self._dtype,
            )
        return ret.to(device=self._cfg.device)


class ToBatchPoseStampedConfig(
    MessageConverterConfig[ToBatchPoseStamped],
    TensorTargetConfigMixin[ToBatchPoseStamped],
):
    class_type: type[ToBatchPoseStamped] = ToBatchPoseStamped
