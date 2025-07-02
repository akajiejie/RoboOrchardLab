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
from dataclasses import dataclass
from typing import Any, Literal

import pyarrow as pa
from robo_orchard_core.datatypes.joint_state import (
    BatchJointsState as _BatchJointsState,
)
from robo_orchard_core.utils.torch_utils import dtype_str2torch

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
    "BatchJointsStateFeature",
    "BatchJointsState",
]


@hg_dataset_feature
@dataclass
class BatchJointsStateFeature(RODataFeature, FeatureDecodeMixin):
    dtype: Literal["float32", "float64"] = "float32"
    decode: bool = True

    def __post_init__(self):
        self._typed_tensor_feature = TypedTensorFeature(
            dtype=self.dtype, as_torch_tensor=True
        )

    @property
    def pa_type(self) -> pa.DataType:
        fields = _BatchJointsState.model_fields
        # initialize with None
        ret_dict: dict[str, pa.DataType] = {k: None for k in fields.keys()}
        tensor_names = ["position", "velocity", "effort"]
        for name in tensor_names:
            assert name in fields, f"{name} not in fields"
            ret_dict[name] = self._typed_tensor_feature.pa_type
        ret_dict["names"] = pa.list_(pa.string())
        ret_dict["timestamps"] = pa.list_(pa.int64())
        for field_name in ret_dict:
            if ret_dict[field_name] is None:
                raise ValueError(
                    f"Field {field_name} has no type defined in "
                    "BatchJointsStateStampedFeature. This means that "
                    "`pa_type` is not properly implemented!"
                )

        return pa.struct(ret_dict)

    def encode_example(
        self, value: BatchJointsState | None
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        tensor_feature = self._typed_tensor_feature
        value = value.to(dtype=dtype_str2torch(self.dtype), device="cpu")
        encoded_data = {
            "position": tensor_feature.encode_example(value.position),
            "velocity": tensor_feature.encode_example(value.velocity),
            "effort": tensor_feature.encode_example(value.effort),
            "names": value.names,
            "timestamps": value.timestamps,
        }
        return encoded_data

    def decode_example(
        self, value: dict[str, Any] | None, **kwargs
    ) -> BatchJointsState | None:
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. Please use "
                "TensorFeature(decode=True) instead."
            )
        if value is None:
            return None
        tensor_feature = TypedTensorFeature(dtype=self.dtype)
        position = tensor_feature.decode_example(value["position"])
        velocity = tensor_feature.decode_example(value["velocity"])
        effort = tensor_feature.decode_example(value["effort"])
        names = value["names"]
        timestamps = value["timestamps"]
        param_dict = dict(
            position=position,
            velocity=velocity,
            effort=effort,
            names=names,
            timestamps=timestamps,
        )

        return BatchJointsState(**param_dict)


class BatchJointsState(_BatchJointsState, ToDataFeatureMixin):
    """Batch joint states of robot.

    This class extends the base BatchJointsState and provides methods
    to encode and decode joint states for dataset storage.
    """

    @classmethod
    def dataset_feature(
        cls, dtype: Literal["float32", "float64"] = "float32"
    ) -> BatchJointsStateFeature:
        ret = BatchJointsStateFeature(dtype=dtype)
        check_fields_consistency(cls, ret.pa_type)
        return ret
