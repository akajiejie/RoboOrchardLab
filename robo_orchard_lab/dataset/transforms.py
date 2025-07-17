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
from dataclasses import is_dataclass
from typing import Generic, Sequence, Type

from pydantic import BaseModel, Field
from robo_orchard_core.datatypes.adaptor import (
    ClassInitFromConfigMixin,
)
from robo_orchard_core.utils.config import (
    ClassConfig,
)
from typing_extensions import TypeVar


class DictTransform(ClassInitFromConfigMixin, metaclass=ABCMeta):
    """A class that defines the interface for transforming a dict.

    The dict is usually a row in a dataset, and the transform
    will take the input columns, apply some transformation, and return
    a new dict with the transformed values added to the original dict.

    User should implement the `transform` method to define the specific
    transformation logic.

    If you use a dataclass or BaseModel as the return type of
    the transform method, `output_columns` will be automatically
    inferred from the fields of the dataclass or BaseModel. Otherwise,
    you should implement the `output_columns` property to return the
    expected output columns for the transform.

    """

    cfg: DictTransformConfig

    @property
    def input_columns(self) -> list[str]:
        """The input columns that this transform requires.

        This should be a list of column names that are required
        for the transformation. The transform method will be called
        with these columns as keyword arguments.
        """
        # use inspect to get the parameters of the transform method

        cached_ret: list | None = getattr(self, "_input_columns", None)

        if cached_ret is not None:
            return cached_ret.copy()

        sig = inspect.signature(self.transform)
        self._input_columns = [
            param.name
            for param in sig.parameters.values()
            if param.name != "self"
        ]
        return self._input_columns

    @property
    def mapped_input_columns(self) -> list[str]:
        """The input columns that this transform requires for column mapping."""  # noqa: E501

        cached_ret: list | None = getattr(self, "_mapped_input_columns", None)

        if cached_ret is not None:
            return cached_ret.copy()

        old_input_columns = self.input_columns
        inverted_mapping = {
            v: k for k, v in self.cfg.input_column_mapping.items()
        }
        mapped_input_columns = [
            inverted_mapping.get(col, col) for col in old_input_columns
        ]
        self._mapped_input_columns = mapped_input_columns
        return self._mapped_input_columns.copy()

    @property
    def output_columns(self) -> list[str]:
        """The output columns that this transform produces.

        This should be a list of column names that the transform will
        produce as output. The transform method will return a dict
        with these keys.

        Note that this property contains all possible output columns,
        not just the ones that are actually produced by the transform.
        The transform method may return a subset of these columns,
        but the output_columns property should list all columns that
        the transform can produce.

        """

        cached_ret: list | None = getattr(self, "_output_columns", None)
        if cached_ret is not None:
            return cached_ret.copy()

        def generate_output_columns() -> list[str]:
            # using signature to get the return type of the transform method
            sig = inspect.signature(self.transform, eval_str=True)
            return_annotation = sig.return_annotation
            # get the classtype from the return annotation
            # if the return type is dataclass, we need to extract the fields
            if is_dataclass(return_annotation):
                return list(return_annotation.__dataclass_fields__.keys())
            # if the return type is a subclass of BaseModel,
            # we can use its fields
            elif isinstance(return_annotation, type) and issubclass(
                return_annotation, BaseModel
            ):
                return list(return_annotation.model_fields.keys())
            else:
                raise NotImplementedError(
                    "Cannot determine output columns for "
                    f"{self.__class__.__name__}. Return type "
                    f"{return_annotation} is not a dataclass or BaseModel. "
                    "You should implement the output_columns property to "
                    "return the expected output columns for this transform."
                )

        self._output_columns = generate_output_columns()
        return self._output_columns.copy()

    @property
    def mapped_output_columns(self) -> list[str]:
        """The output columns that this transform produces after mapping."""
        cached_ret: list | None = getattr(self, "_mapped_output_columns", None)
        if cached_ret is not None:
            return cached_ret.copy()

        old_output_columns = self.output_columns
        mapped_output_columns = [
            self.cfg.output_column_mapping.get(col, col)
            for col in old_output_columns
        ]
        self._mapped_output_columns = mapped_output_columns
        return self._mapped_output_columns.copy()

    @abstractmethod
    def transform(self, **kwargs) -> dict | BaseModel:
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

        ts_input = {}
        for col in self.input_columns:
            if col not in mapped_input:
                raise KeyError(f"Input column `{col}` not found in row dict.")
            ts_input[col] = mapped_input[col]
        columns_after = self.transform(**ts_input)
        if isinstance(columns_after, BaseModel):
            columns_after = columns_after.model_dump(mode="python")
        elif is_dataclass(columns_after):
            columns_after = {
                field.name: getattr(columns_after, field.name)
                for field in columns_after.__dataclass_fields__.values()
            }

        if not isinstance(columns_after, dict):
            raise TypeError(
                f"Transform {self.__class__.__name__} must return a dict, "
                f"got {type(columns_after)}."
            )

        # check that the output columns match the expected output columns
        for col in columns_after.keys():
            if col not in self.output_columns:
                raise ValueError(
                    f"Output column {col} not in expected output columns: "
                    f"{self.output_columns}."
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


DictTransformType = TypeVar(
    "DictTransformType", bound=DictTransform, covariant=True
)


class DictTransformConfig(ClassConfig[DictTransformType]):
    class_type: Type[DictTransformType]

    input_column_mapping: dict[str, str] = Field(default_factory=dict)
    """The input columns that need to be mapped to fit
    the transform's input_columns."""

    output_column_mapping: dict[str, str] = Field(default_factory=dict)
    """The output columns that the transform will produce.
    This should be a mapping from the output column names to the
    names that the transform will use to return the transformed values.
    If the transform does not produce any output, this can be an empty dict.
    """


class ConcatDictTransform(DictTransform):
    cfg: ConcatDictTransformConfig[DictTransform]

    def __init__(self, cfg: ConcatDictTransformConfig[DictTransform]) -> None:
        self.cfg = cfg

        self._transforms = [
            t() for t in self.cfg.transforms
        ]  # Instantiate all transforms

        input_columns: set[str] = set()
        output_columns: set[str] = set()
        for transform in self._transforms:
            cur_input_set = set(transform.mapped_input_columns)
            # consume the output columns if required as input
            to_consume = cur_input_set.intersection(output_columns)
            output_columns.difference_update(to_consume)
            # remove the input that is taken from the previous output
            # and update the input columns
            cur_input_set.difference_update(to_consume)
            input_columns.update(cur_input_set)
            # update the output columns
            output_columns.update(transform.mapped_output_columns)

        self._mapped_input_columns = list(input_columns)
        self._mapped_output_columns = list(output_columns)

    @property
    def input_columns(self) -> list[str]:
        raise RuntimeError(
            "ConcatDictTransform does not implement input_columns. "
            "Use the mapped_input_columns property instead."
        )

    @property
    def mapped_input_columns(self) -> list[str]:
        """Get the input columns for this transform."""
        return self._mapped_input_columns.copy()

    @property
    def output_columns(self) -> list[str]:
        """Get the output columns for this transform."""
        raise RuntimeError(
            "ConcatDictTransform does not implement output_columns. "
            "Use the mapped_output_columns property instead."
        )

    @property
    def mapped_output_columns(self) -> list[str]:
        """Get the output columns for this transform."""
        return self._mapped_output_columns.copy()

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


class ConcatDictTransformConfig(
    DictTransformConfig[ConcatDictTransform], Generic[DictTransformType]
):
    class_type: Type[ConcatDictTransform] = ConcatDictTransform

    transforms: Sequence[DictTransformConfig[DictTransformType]] = Field(
        min_length=1,
    )
    """A sequence of transforms to apply to the input columns."""
