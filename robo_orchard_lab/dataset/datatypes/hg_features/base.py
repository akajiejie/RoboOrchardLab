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
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from datasets.features.features import register_feature
from pydantic import BaseModel
from typing_extensions import TypeVar

__all__ = [
    "RODataFeature",
    "FeatureDecodeMixin",
    "ToDataFeatureMixin",
    "hg_dataset_feature",
    "check_fields_consistency",
]


@dataclass
class FeatureDecodeMixin(metaclass=ABCMeta):
    """Mixin class for features that support decoding."""

    @abstractmethod
    def decode_example(self, value: Any) -> Any:
        """Decode the example value from its stored format."""
        raise NotImplementedError(
            "Subclasses must implement decode_example method."
        )


@dataclass
class RODataFeature(metaclass=ABCMeta):
    """Base class for RoboOrchard dataset features.

    User should implement the `pa_type` property and `encode_example` method
    to define the specific feature type and how to encode example values.

    """

    _type: str
    """The class name of the feature type. Needed for serialization
    and deserialization. Should be set in subclasses."""

    def __call__(self) -> pa.DataType:
        """Return the pyarrow data type for this feature."""
        return self.pa_type

    @property
    @abstractmethod
    def pa_type(self) -> pa.DataType:
        """Return the pyarrow data type for this feature."""
        raise NotImplementedError(
            "Subclasses must implement pa_type property."
        )

    @abstractmethod
    def encode_example(self, value: Any) -> Any:
        """Encode the example value into a format suitable for storage."""
        raise NotImplementedError(
            "Subclasses must implement encode_example method."
        )


class ToDataFeatureMixin(metaclass=ABCMeta):
    """Mixin class for features that can be converted to a pyarrow DataType."""

    @abstractmethod
    def dataset_feature(self) -> RODataFeature:
        raise NotImplementedError(
            "Subclasses must implement dataset_feature method."
        )


RODataFeatureType = TypeVar("RODataFeatureType", bound=RODataFeature)


def hg_dataset_feature(
    cls: type[RODataFeatureType],
) -> type[RODataFeatureType]:
    """Decorator to register a feature class with its type."""
    if not issubclass(cls, RODataFeature):
        raise TypeError("Feature class must inherit from RODataFeature.")
    register_feature(cls, cls._type)
    return cls


def check_fields_consistency(
    cls: type[BaseModel],
    pa_struct: pa.StructType,
):
    pydantic_fields = set(cls.model_fields.keys())
    pa_fields = set([field.name for field in pa_struct.fields])
    if pydantic_fields != pa_fields:
        raise TypeError(
            f"Pydantic fields {pydantic_fields} do not match "
            f"pyarrow fields {pa_fields} for {cls.__name__}."
            " This means that the feature is not fully implemented."
        )
