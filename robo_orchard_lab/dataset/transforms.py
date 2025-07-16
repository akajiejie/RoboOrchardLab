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
import inspect
from abc import ABCMeta, abstractmethod
from typing import Sequence, Type, TypeVar

from pydantic import Field
from robo_orchard_core.datatypes.adaptor import (
    TypeAdaptorImpl,
)
from robo_orchard_core.utils.config import (
    ClassConfig,
)


class RowDictTransform(TypeAdaptorImpl[dict, dict], metaclass=ABCMeta):
    """A class that defines the interface for transforming a row dict."""

    cfg: RowDictTransformConfig

    @property
    def input_columns(self) -> list[str]:
        """The input columns that this transform requires.

        This should be a list of column names that are required
        for the transformation. The transform method will be called
        with these columns as keyword arguments.
        """
        # use inspect to get the parameters of the transform method
        sig = inspect.signature(self.transform)
        return [
            param.name
            for param in sig.parameters.values()
            if param.name != "self"
        ]

    @abstractmethod
    def transform(self, **kwargs) -> dict:
        """Transform the input columns into a new row dict to be updated.

        All input columns will be passed as keyword arguments.
        """
        raise NotImplementedError

    def __call__(self, row: dict) -> dict:
        """Call the transform on a row dict.

        This method will extract the input columns from the row dict,
        call the transform method, and return a new row dict with the
        transformed values added to the original row dict.

        """
        mapped_input = row.copy()
        for src_name, dst_name in self.cfg.input_column_mapping.items():
            if src_name not in mapped_input:
                raise ValueError(
                    f"Input column {src_name} not found in row dict."
                )
            mapped_input[dst_name] = mapped_input.pop(src_name)

        columns_after = self.transform(
            **{k: mapped_input[k] for k in self.input_columns}
        )

        for src_name, dst_name in self.cfg.output_column_mapping.items():
            if dst_name in columns_after:
                raise ValueError(
                    f"Output column {dst_name} already exists in transformed "
                    "columns."
                )
            if src_name in columns_after:
                columns_after[dst_name] = columns_after.pop(src_name)

        ret = row.copy()
        ret.update(columns_after)
        return ret


RowDictTransformType = TypeVar("RowDictTransformType", bound=RowDictTransform)


class RowDictTransformConfig(ClassConfig[RowDictTransformType]):
    class_type: Type[RowDictTransformType]

    input_column_mapping: dict[str, str] = Field(default_factory=dict)
    """The input columns that need to be mapped to fit
    the transform's input_columns."""

    output_column_mapping: dict[str, str] = Field(default_factory=dict)
    """The output columns that the transform will produce.
    This should be a mapping from the output column names to the
    names that the transform will use to return the transformed values.
    If the transform does not produce any output, this can be an empty dict.
    """


class ConcatDictTransform(RowDictTransform):
    cfg: ConcatDictTransformConfig

    def __init__(self, cfg: ConcatDictTransformConfig) -> None:
        self.cfg = cfg

        self._transforms = [
            t() for t in self.cfg.transforms
        ]  # Instantiate all transforms

    @property
    def input_columns(self) -> list[str]:
        """Get the input columns for this transform."""
        return self._transforms[0].input_columns

    def transform(self, **kwargs) -> dict:
        raise RuntimeError(
            "ConcatDictTransform does not implement transform method. "
            "Use the __call__ method instead."
        )

    def __call__(self, row: dict) -> dict:
        """Concatenate the input columns into a new row dict.

        This method will apply all transforms in the order they are defined
        in the configuration, and return a new row dict with the transformed
        values added to the original row dict.
        """
        for transform in self._transforms:
            row = transform(row)
        return row


class ConcatDictTransformConfig(RowDictTransformConfig[ConcatDictTransform]):
    class_type: Type[ConcatDictTransform] = ConcatDictTransform

    transforms: Sequence[RowDictTransformConfig[RowDictTransform]] = Field(
        min_length=1,
    )
    """A sequence of transforms to apply to the input columns."""
