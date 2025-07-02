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
from typing import Literal

import pyarrow as pa
from robo_orchard_core.datatypes.geometry import (
    BatchFrameTransform as _BatchFrameTransform,
    BatchPose as _BatchPose,
    BatchTransform3D as _BatchTransform3D,
)

from robo_orchard_lab.dataset.datatypes.hg_features import (
    FeatureDecodeMixin,
    RODataFeature,
    ToDataFeatureMixin,
    check_fields_consistency,
    hg_dataset_feature,
)
from robo_orchard_lab.dataset.datatypes.hg_features.tensor import (
    TypedTensorFeature,
)

__all__ = [
    "BatchTransform3DFeature",
    "BatchTransform3D",
    "BatchPoseFeature",
    "BatchPose",
    "BatchFrameTransformFeature",
    "BatchFrameTransform",
]


@hg_dataset_feature
@dataclass
class BatchTransform3DFeature(RODataFeature, FeatureDecodeMixin):
    """A feature for storing batch frame transforms in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the frame transforms.
    """

    dtype: Literal["float32", "float64"] = "float32"
    decode: bool = True

    def __post_init__(self):
        self._typed_tensor_feature = TypedTensorFeature(
            dtype=self.dtype, as_torch_tensor=True
        )

    @property
    def pa_type(self) -> pa.StructType:
        fields = _BatchTransform3D.model_fields

        ret_dict: dict[str, pa.DataType] = {}
        ret_dict["xyz"] = self._typed_tensor_feature.pa_type
        ret_dict["quat"] = self._typed_tensor_feature.pa_type
        ret_dict["timestamps"] = pa.list_(pa.int64())

        for field_name in fields:
            if field_name not in ret_dict:
                raise ValueError(
                    f"Field {field_name} has no type defined in "
                    "BatchTransform3DFeature. This means that the "
                    "feature is not fully implemented."
                )
        return pa.struct([pa.field(k, v) for k, v in ret_dict.items()])

    def encode_example(self, value: _BatchTransform3D | None) -> dict | None:
        if value is None:
            return None
        tensor_feature = self._typed_tensor_feature
        return {
            "xyz": tensor_feature.encode_example(value.xyz),
            "quat": tensor_feature.encode_example(value.quat),
            "timestamps": value.timestamps,
        }

    def decode_example(
        self, value: dict | None, **kwargs
    ) -> BatchTransform3D | None:
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use "
                "BatchTransform3DFeature(decode=True) instead."
            )
        if value is None:
            return None
        tensor_feature = self._typed_tensor_feature
        value = copy.copy(value)
        value["xyz"] = tensor_feature.decode_example(value["xyz"])
        value["quat"] = tensor_feature.decode_example(value["quat"])
        return BatchTransform3D(**value)


class BatchTransform3D(_BatchTransform3D, ToDataFeatureMixin):
    @classmethod
    def dataset_feature(
        cls, dtype: Literal["float32", "float64"] = "float32"
    ) -> BatchTransform3DFeature:
        ret = BatchTransform3DFeature(dtype=dtype)

        check_fields_consistency(cls, ret.pa_type)

        return ret


@hg_dataset_feature
@dataclass
class BatchPoseFeature(BatchTransform3DFeature):
    """A feature for storing batch poses in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the poses.
    """

    @property
    def pa_type(self) -> pa.StructType:
        parent_fields = list(super().pa_type.fields)
        parent_fields.append(pa.field("frame_id", pa.string()))
        return pa.struct(parent_fields)

    def encode_example(self, value: _BatchPose | None) -> dict | None:
        if value is None:
            return None
        parent_ret = super().encode_example(value)
        assert parent_ret is not None
        parent_ret["frame_id"] = value.frame_id
        return parent_ret

    def decode_example(self, value: dict | None, **kwargs) -> BatchPose | None:
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use "
                "BatchPoseFeature(decode=True) instead."
            )
        if value is None:
            return None
        parent_ret = super().decode_example(value)
        return BatchPose(frame_id=value["frame_id"], **parent_ret.__dict__)


class BatchPose(_BatchPose, ToDataFeatureMixin):
    @classmethod
    def dataset_feature(
        cls, dtype: Literal["float32", "float64"] = "float32"
    ) -> BatchPoseFeature:
        ret = BatchPoseFeature(dtype=dtype)
        check_fields_consistency(cls, ret.pa_type)
        return ret


@hg_dataset_feature
@dataclass
class BatchFrameTransformFeature(BatchTransform3DFeature):
    """A feature for storing batch frame transforms in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the frame transforms.
    """

    @property
    def pa_type(self) -> pa.StructType:
        parent_fields = list(super().pa_type.fields)
        parent_fields.append(pa.field("parent_frame_id", pa.string()))
        parent_fields.append(pa.field("child_frame_id", pa.string()))
        return pa.struct(parent_fields)

    def encode_example(
        self, value: _BatchFrameTransform | None
    ) -> dict | None:
        if value is None:
            return None
        parent_ret = super().encode_example(value)
        assert parent_ret is not None
        parent_ret["parent_frame_id"] = value.parent_frame_id
        parent_ret["child_frame_id"] = value.child_frame_id
        return parent_ret

    def decode_example(
        self, value: dict | None, **kwargs
    ) -> BatchFrameTransform | None:
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use "
                "BatchFrameTransformFeature(decode=True) instead."
            )
        if value is None:
            return None
        parent_ret = super().decode_example(value)
        return BatchFrameTransform(
            parent_frame_id=value["parent_frame_id"],
            child_frame_id=value["child_frame_id"],
            **parent_ret.__dict__,
        )


class BatchFrameTransform(_BatchFrameTransform, ToDataFeatureMixin):
    @classmethod
    def dataset_feature(
        cls, dtype: Literal["float32", "float64"] = "float32"
    ) -> BatchFrameTransformFeature:
        ret = BatchFrameTransformFeature(dtype=dtype)
        check_fields_consistency(cls, ret.pa_type)
        return ret
