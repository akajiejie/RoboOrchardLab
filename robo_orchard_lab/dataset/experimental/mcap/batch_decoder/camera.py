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

from robo_orchard_lab.dataset.experimental.mcap.batch_decoder.base import (
    McapBatchDecoder,
    McapBatchDecoderConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.batch_camera import (  # noqa: E501
    BatchCameraDataWithTimestamps,
    CameraDataConfigMixin,
    FgCameraCompressedImages,
    ToBatchCameraDataConfig,
)

__all__ = [
    "McapBatch2BatchCameraData",
    "McapBatch2BatchCameraDataConfig",
]


class McapBatch2BatchCameraData(
    McapBatchDecoder[BatchCameraDataWithTimestamps]
):
    def __init__(self, config: McapBatch2BatchCameraDataConfig):
        super().__init__()
        self._cfg = config
        self._msg_cvt = ToBatchCameraDataConfig(
            pix_fmt=config.pix_fmt,
        )()
        self._required_topics = set([config.image_topic])
        if config.calib_topic is not None:
            self._required_topics.add(config.calib_topic)
        if config.tf_topic is not None:
            self._required_topics.add(config.tf_topic)

    def require_topics(self) -> set[str]:
        return self._required_topics

    def format_batch(
        self, decoded_msgs: dict[str, list]
    ) -> BatchCameraDataWithTimestamps:
        # prepare input message
        msg_batch = FgCameraCompressedImages(
            images=decoded_msgs[self._cfg.image_topic]
        )
        if self._cfg.calib_topic is not None:
            calibration = decoded_msgs.get(self._cfg.calib_topic, [])
            if len(calibration) > 0:
                msg_batch.calib = calibration[0]

            if len(calibration) > 1:
                raise ValueError(
                    f"Multiple camera calibration messages found for topic "
                    f"{self._cfg.calib_topic}. Expected only one."
                )
        if self._cfg.tf_topic is not None:
            tf_msgs = decoded_msgs.get(self._cfg.tf_topic, [])
            if len(tf_msgs) > 0:
                msg_batch.tf = tf_msgs
            else:
                msg_batch.tf = None

        return self._msg_cvt.convert(msg_batch)


class McapBatch2BatchCameraDataConfig(
    McapBatchDecoderConfig[McapBatch2BatchCameraData],
    CameraDataConfigMixin[McapBatch2BatchCameraData],
):
    class_type: type[McapBatch2BatchCameraData] = McapBatch2BatchCameraData

    image_topic: str
    """The source topic of camera image."""

    calib_topic: str | None = None
    """The source topic of camera calibration.

    If None, no calibration will be used."""

    tf_topic: str | None = None
    """Frame transform topic of camera.

    The frame transform is usually referred as camera extrinsics, which
    is the transformation from the camera frame to the world/parent frame.

    If None, no frame transform will be used.
    """
