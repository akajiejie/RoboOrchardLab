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
import copy
from dataclasses import dataclass
from typing import Any, Literal

import pyarrow as pa
from robo_orchard_core.datatypes.camera_data import (
    BatchCameraData as _BatchCameraData,
    BatchCameraDataEncoded as _BatchCameraDataEncoded,
    BatchCameraInfo as _BatchCameraInfo,
    Distortion as _Distortion,
)

from robo_orchard_lab.dataset.datatypes.geometry import (
    BatchFrameTransformFeature,
)
from robo_orchard_lab.dataset.datatypes.hg_features import (
    FeatureDecodeMixin,
    RODataFeature,
    ToDataFeatureMixin,
    check_fields_consistency,
    hg_dataset_feature,
)
from robo_orchard_lab.dataset.datatypes.hg_features.tensor import (
    AnyTensorFeature,
    TypedTensorFeature,
)

__all__ = [
    "DistortionFeature",
    "Distortion",
    "BatchCameraInfoFeature",
    "BatchCameraInfo",
    "BatchCameraDataEncodedFeature",
    "BatchCameraDataEncoded",
    "BatchCameraDataFeature",
    "BatchCameraData",
]


@hg_dataset_feature
@dataclass
class DistortionFeature(RODataFeature, FeatureDecodeMixin):
    """A feature for storing distortion parameters in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the distortion parameters.
    """

    dtype: Literal["float32", "float64"] = "float32"
    decode: bool = True

    def __post_init__(self):
        self._typed_tensor_feature = TypedTensorFeature(
            dtype=self.dtype, as_torch_tensor=True
        )

    @property
    def pa_type(self) -> pa.StructType:
        # initialize with None
        ret_dict: dict[str, pa.DataType] = {
            "model": pa.string(),
            "coefficients": self._typed_tensor_feature.pa_type,
        }
        return pa.struct([pa.field(k, v) for k, v in ret_dict.items()])

    def encode_example(self, value: _Distortion | None) -> dict | None:
        if value is None:
            return None
        return {
            "model": value.model,
            "coefficients": self._typed_tensor_feature.encode_example(
                value.coefficients
            ),
        }

    def decode_example(
        self, value: dict[str, Any] | None
    ) -> Distortion | None:
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use "
                "DistortionFeature(decode=True) instead."
            )
        if value is None:
            return None
        value = copy.copy(value)
        value["coefficients"] = self._typed_tensor_feature.decode_example(
            value["coefficients"]
        )
        return Distortion(**value)


class Distortion(_Distortion, ToDataFeatureMixin):
    """A class for distortion parameters with dataset feature support."""

    @classmethod
    def dataset_feature(
        cls, dtype: Literal["float32", "float64"] = "float32"
    ) -> DistortionFeature:
        ret = DistortionFeature(dtype=dtype)
        check_fields_consistency(cls, ret.pa_type)
        return ret


@hg_dataset_feature
@dataclass
class BatchCameraInfoFeature(RODataFeature, FeatureDecodeMixin):
    """A feature for storing batch camera info in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the camera info.
    """

    dtype: Literal["float32", "float64"] = "float32"
    decode: bool = True

    def __post_init__(self):
        self._typed_tensor_feature = TypedTensorFeature(
            dtype=self.dtype, as_torch_tensor=True
        )
        self._distortion_feature = DistortionFeature(dtype=self.dtype)
        self._tf_feature = BatchFrameTransformFeature(dtype=self.dtype)

    @property
    def pa_type(self) -> pa.StructType:
        # initialize with None
        ret_dict: dict[str, pa.DataType] = {
            "topic": pa.string(),
            "frame_id": pa.string(),
            "image_shape": pa.list_(pa.int32()),
            "intrinsic_matrices": self._typed_tensor_feature.pa_type,
            "distortion": self._distortion_feature.pa_type,
            "pose": self._tf_feature.pa_type,
        }
        return pa.struct([pa.field(k, v) for k, v in ret_dict.items()])

    def encode_example(
        self, value: _BatchCameraInfo | None
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        tensor_feature = self._typed_tensor_feature
        distortion_feature = self._distortion_feature
        tf_feature = self._tf_feature

        encoded_data = {
            "topic": value.topic,
            "frame_id": value.frame_id,
            "image_shape": value.image_shape,
            "intrinsic_matrices": tensor_feature.encode_example(
                value.intrinsic_matrices
            ),
            "distortion": distortion_feature.encode_example(value.distortion),
            "pose": tf_feature.encode_example(value.pose),
        }
        return encoded_data

    def decode_example(
        self, value: dict[str, Any] | None, **kwargs
    ) -> BatchCameraInfo | None:
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use "
                "BatchCameraInfoFeature(decode=True) instead."
            )
        if value is None:
            return None
        value = copy.copy(value)
        value["intrinsic_matrices"] = (
            self._typed_tensor_feature.decode_example(
                value["intrinsic_matrices"]
            )
        )
        value["distortion"] = self._distortion_feature.decode_example(
            value["distortion"]
        )
        value["pose"] = self._tf_feature.decode_example(value["pose"])
        return BatchCameraInfo(**value)


class BatchCameraInfo(_BatchCameraInfo, ToDataFeatureMixin):
    """A class for batch camera info with dataset feature support."""

    @classmethod
    def dataset_feature(
        cls, dtype: Literal["float32", "float64"] = "float32"
    ) -> BatchCameraInfoFeature:
        ret = BatchCameraInfoFeature(dtype=dtype)
        check_fields_consistency(cls, ret.pa_type)
        return ret


@hg_dataset_feature
@dataclass
class BatchCameraDataEncodedFeature(RODataFeature, FeatureDecodeMixin):
    """A feature for storing batch camera data in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the camera data.
    """

    dtype: Literal["float32", "float64"] = "float32"
    decode: bool = True

    def __post_init__(self):
        self._camera_info_feature = BatchCameraInfoFeature(
            dtype=self.dtype, decode=self.decode
        )

    @property
    def pa_type(self) -> pa.StructType:
        field_list = [
            pa.field("sensor_data", pa.list_(pa.binary()), nullable=True),
            pa.field("format", pa.string(), nullable=True),
            pa.field("timestamps", pa.list_(pa.int64()), nullable=True),
        ]
        field_list.extend(self._camera_info_feature.pa_type.fields)
        return pa.struct(field_list)

    def encode_example(
        self, value: _BatchCameraDataEncoded | None
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        camera_info_feature = self._camera_info_feature
        encoded_data = {
            "sensor_data": value.sensor_data,
            "format": value.format,
            "timestamps": value.timestamps,
        }
        parent_dict = camera_info_feature.encode_example(value)
        assert parent_dict is not None
        encoded_data.update(parent_dict)
        return encoded_data

    def decode_example(
        self, value: dict[str, Any] | None, **kwargs
    ) -> BatchCameraDataEncoded | None:
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use "
                "BatchCameraDataEncodedFeature(decode=True) instead."
            )
        if value is None:
            return None
        value = copy.copy(value)
        parent_ret = self._camera_info_feature.decode_example(value)
        if parent_ret is None:
            return None
        return BatchCameraDataEncoded(
            sensor_data=value["sensor_data"],
            format=value["format"],
            timestamps=value["timestamps"],
            **parent_ret.__dict__,
        )


class BatchCameraDataEncoded(BatchCameraInfo, _BatchCameraDataEncoded):
    """A class for batch camera data with dataset feature support."""

    @classmethod
    def dataset_feature(
        cls, dtype: Literal["float32", "float64"] = "float32"
    ) -> BatchCameraDataEncodedFeature:
        ret = BatchCameraDataEncodedFeature(dtype=dtype)
        check_fields_consistency(cls, ret.pa_type)
        return ret


@hg_dataset_feature
@dataclass
class BatchCameraDataFeature(RODataFeature, FeatureDecodeMixin):
    """A feature for storing batch camera data in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the camera data.
    """

    dtype: Literal["float32", "float64"] = "float32"
    decode: bool = True

    def __post_init__(self):
        self._camera_info_feature = BatchCameraInfoFeature(
            dtype=self.dtype, decode=self.decode
        )
        # Camera data can have any tensor type, so we use AnyTensorFeature
        # to store the sensor data. This allows for flexibility in the type of
        # sensor data stored (e.g., images, point clouds, etc.).
        self._sensor_data_feature = AnyTensorFeature()

    @property
    def pa_type(self) -> pa.StructType:
        field_list = [
            pa.field(
                "sensor_data", self._sensor_data_feature.pa_type, nullable=True
            ),
            pa.field("pix_fmt", pa.string(), nullable=True),
            pa.field("timestamps", pa.list_(pa.int64()), nullable=True),
        ]
        field_list.extend(self._camera_info_feature.pa_type.fields)
        return pa.struct(field_list)

    def encode_example(
        self, value: _BatchCameraData | None
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        camera_info_feature = self._camera_info_feature
        encoded_data = camera_info_feature.encode_example(value)
        assert encoded_data is not None
        encoded_data["sensor_data"] = self._sensor_data_feature.encode_example(
            value.sensor_data
        )
        encoded_data["pix_fmt"] = value.pix_fmt
        encoded_data["timestamps"] = value.timestamps
        return encoded_data

    def decode_example(
        self, value: dict[str, Any] | None, **kwargs
    ) -> BatchCameraData | None:
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use "
                "BatchCameraDataFeature(decode=True) instead."
            )
        if value is None:
            return None
        parent_ret = self._camera_info_feature.decode_example(value)
        if parent_ret is None:
            return None
        return BatchCameraData(
            pix_fmt=value["pix_fmt"],
            timestamps=value["timestamps"],
            sensor_data=self._sensor_data_feature.decode_example(
                value["sensor_data"]
            ),  # type: ignore
            **parent_ret.__dict__,
        )


class BatchCameraData(BatchCameraInfo, _BatchCameraData):
    """A class for batch camera data with dataset feature support."""

    @classmethod
    def dataset_feature(
        cls, dtype: Literal["float32", "float64"] = "float32"
    ) -> BatchCameraDataFeature:
        ret = BatchCameraDataFeature(dtype=dtype)
        check_fields_consistency(cls, ret.pa_type)
        return ret
