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
import json
import os
from typing import Any, Iterable, TypeAlias, TypeVar, overload

import fsspec
import torch
from datasets import (
    Dataset as HFDataset,
    Features,
)
from datasets.arrow_dataset import Column
from sqlalchemy import URL, Engine, select
from sqlalchemy.orm import Session, make_transient

from robo_orchard_lab.dataset.datatypes import *
from robo_orchard_lab.dataset.robot.columns import (
    PreservedIndexColumnsKeys,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    Task,
)
from robo_orchard_lab.dataset.robot.engine import create_engine
from robo_orchard_lab.dataset.robot.row_sampler import (
    MultiRowSampler,
    MultiRowSamplerConfig,
)

__all__ = ["RODataset", "ROMultiRowDataset"]

MetaType = TypeVar("MetaType", Episode, Instruction, Robot, Task)
"""A type variable for metadata types in the RoboOrchard dataset."""

TorchDataset: TypeAlias = torch.utils.data.Dataset


class RODataset(TorchDataset):
    """The RoboOrchard dataset for robot data.

    We use a tabular dataset to store the frame-level information, and a
    separate database to store the episode-level information. The huggingface
    datasets (pyarrow_dataset) is used as table format, and SQLAlchemy with
    DuckDB are used to manage the database.

    Args:
        dataset_path (str): The path to the dataset directory.
            It should contain a `dataset.arrow` file and a `meta_db.*` file.
        storage_options (dict | None, optional): Additional Key/value pairs to
            be passed on to the file-system backend, if any. This is passed
            to the `datasets.Dataset.load_from_disk` method. Defaults to None.
        meta_index2meta (bool, optional): Whether to convert the index-based
            metadata to actual metadata objects when accessing the dataset.
            If True, the `episode`, `task`, `robot`, and `instruction` fields
            will be added and the corresponding index fields will be removed.
            Defaults to True.

    """

    frame_dataset: HFDataset
    """The Hugging Face Dataset object containing the frame data."""
    db_engine: Engine
    """The SQLAlchemy engine for the meta database"""
    index_dataset: HFDataset
    """The same as `frame_dataset`, but only contains the preserved index columns."""  # noqa: E501
    meta_index2meta: bool
    """Whether to convert the index-based metadata to actual metadata
    objects when accessing the dataset."""

    def __init__(
        self,
        dataset_path: str,
        storage_options: dict | None = None,
        meta_index2meta: bool = False,
    ):
        dataset_path = os.path.expanduser(dataset_path)
        self.frame_dataset = HFDataset.load_from_disk(
            dataset_path, storage_options=storage_options
        )
        self.index_dataset = self.frame_dataset.select_columns(
            column_names=list(PreservedIndexColumnsKeys)
        )
        self.meta_index2meta = meta_index2meta
        # recover state dict
        from datasets import config as hg_datasets_config

        state_file = os.path.join(
            dataset_path, hg_datasets_config.DATASET_STATE_JSON_FILENAME
        )
        state: dict = json.load(open(state_file, "r"))
        self._dataset_format_version = state.get("robo_orchard_state", {}).get(
            "dataset_format_version", None
        )
        # load db
        self.db_engine = self._load_db(dataset_path)

    def _get_state_(self) -> dict:
        """Get all internal state of the dataset.

        This method is used to share the internal state of the dataset
        within single process, not for pickling!
        """
        return self.__dict__.copy()

    def __getstate__(self) -> dict:
        """Get the state of the dataset for pickling."""
        state = self._get_state_()
        # remove db_engine from state to avoid pickling issues
        engine: Engine = state.pop("db_engine")
        state["db_engine_url"] = engine.url
        return state

    def __setstate__(self, state: dict):
        """Set the state of the dataset from a pickled state."""
        # restore db_engine from url
        state = state.copy()
        db_engine_url = state.pop("db_engine_url")
        state["db_engine"] = create_engine(db_engine_url, readonly=True)
        # restore other state
        self.__dict__.update(state)

    def _load_db(self, dataset_path: str) -> Engine:
        fs: fsspec.AbstractFileSystem = fsspec.core.url_to_fs(dataset_path)[0]
        file_list = fs.ls(dataset_path, detail=False)
        db_candidate = [
            f for f in file_list if os.path.basename(f).startswith("meta_db.")
        ]
        if len(db_candidate) == 0:
            raise ValueError(
                f"No meta db file found in {dataset_path}. "
                "Please ensure the dataset has been properly packaged."
            )
        if len(db_candidate) > 1:
            raise ValueError(
                f"Multiple meta db files found in {dataset_path}: {db_candidate}"  # noqa: E501
            )
        db_path = db_candidate[0]
        # get drivername from file extension
        _, ext = os.path.splitext(db_path)
        drivername = ext[1:]
        return create_engine(
            url=URL.create(drivername=drivername, database=db_path),
            readonly=True,
        )

    @property
    def features(self) -> Features:
        return self.frame_dataset.features

    def select_columns(
        self, column_names: str | list[str], new_fingerprint: str | None = None
    ) -> RODataset:
        """Select one or more columns from the dataset.

        Args:
            column_names (str | list[str]): The name(s) of the column(s) to
                select. If a single column name is provided, it can be a
                string.
            new_fingerprint (str | None, optional): The new fingerprint of
                the frame dataset after transform. If `None`, the new
                fingerprint is computed using a hash of the previous
                fingerprint, and the transform arguments. This argument is
                used in the `select_columns` method of the Hugging Face
                Dataset to ensure that the dataset is properly cached and
                can be loaded efficiently
                in the future. Defaults to None.

        Returns:
            RODataset: A new instance of `RODataset` with the selected columns.

        """
        state_dict = self._get_state_()
        state_dict["frame_dataset"] = self.frame_dataset.select_columns(
            column_names=column_names, new_fingerprint=new_fingerprint
        )
        ret = RODataset.__new__(RODataset)
        ret.__dict__.update(state_dict)
        return ret

    def save_to_disk(
        self,
        dataset_path: str,
        max_shard_size: str | int = "2000MB",
        num_shards: int | None = None,
        num_proc: int | None = None,
        storage_options: dict | None = None,
    ):
        """Saves a dataset to filesystem.

        Args:
            dataset_path (str): The path to the dataset directory where
                the dataset will be saved.
            max_shard_size (str | int , optional): The maximum size of
                each shard. Defaults to "2000MB". This can be a string
                (e.g., "2000MB") or an integer (e.g., 2000 * 1024 * 1024
                for 2000MB).
            num_shards (int | None, optional): The number of shards to create.
                Number of shards to write. By default the number of shards
                depends on `max_shard_size` and `num_proc`.
            num_proc (int | None, optional): The number of processes to use
                for saving the dataset. Defaults to None.
            storage_options (dict | None, optional): Additional Key/value pairs
                to be passed on to the file-system backend, if any. Defaults
                to None.
        """
        self.frame_dataset.save_to_disk(
            dataset_path=dataset_path,
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            num_proc=num_proc,
            storage_options=storage_options,
        )
        src_meta_db_path = self.db_engine.url.database
        assert src_meta_db_path is not None
        fs: fsspec.AbstractFileSystem = fsspec.core.url_to_fs(dataset_path)[0]
        # copy the meta db file to the new dataset path
        dst_meta_db_path = os.path.join(
            dataset_path, os.path.basename(src_meta_db_path)
        )
        if fs.exists(dst_meta_db_path):
            fs.rm(dst_meta_db_path)
        fs.copy(src_meta_db_path, dst_meta_db_path)

    def select(
        self,
        indices: Iterable,
        keep_in_memory: bool = False,
        indices_cache_file_name: str | None = None,
        writer_batch_size: int = 1000,
        new_fingerprint: str | None = None,
    ) -> RODataset:
        """Select a subset of the dataset based on indices.

        This method is similar to the `select` method in Hugging Face
        Datasets.

        Args:
            indices (Iterable): The indices of the frames to select.
                This can be a list of integers or a slice object.
            keep_in_memory (bool, optional): Whether to keep the indices
                mapping in memory instead of writing to a cache file. If
                indices is too large, it is recommended to set this
                to False to avoid memory issues. Defaults to False.
            indices_cache_file_name (str | None, optional): The name of the
                cache file to store the indices mapping. If `None`, the
                indices mapping will not be cached. This argument should
                be set to a valid file path if `keep_in_memory` is False.
                Defaults to None.
            writer_batch_size (int , optional): The batch size to use
                when writing the indices mapping to the cache file. Higher
                batch size can improve performance, but may also increase
                memory usage. Defaults to 1000.
            new_fingerprint (str | None, optional): The new fingerprint of
                the frame dataset after transform. If `None`, the new
                fingerprint is computed using a hash of the previous
                fingerprint, and the transform arguments.
        """
        state_dict = self._get_state_()
        state_dict["frame_dataset"] = self.frame_dataset.select(
            indices=indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
        )
        ret = RODataset.__new__(RODataset)
        ret.__dict__.update(state_dict)
        return ret

    def _meta_index2meta(self, src: dict[str, Any]) -> dict:
        """Convert the index-based metadata in `src` to actual metadata objects."""  # noqa: E501
        dst = src.copy()
        dict_to_update = {}
        if "episode_index" in dst:
            dict_to_update["episode"] = self.get_meta(
                Episode, dst.pop("episode_index", None)
            )
        if "task_index" in dst:
            dict_to_update["task"] = self.get_meta(
                Task, dst.pop("task_index", None)
            )
        if "robot_index" in dst:
            dict_to_update["robot"] = self.get_meta(
                Robot, dst.pop("robot_index", None)
            )
        if "instruction_index" in dst:
            dict_to_update["instruction"] = self.get_meta(
                Instruction, dst.pop("instruction_index", None)
            )
        dst.update(dict_to_update)
        return dst

    @overload
    def __getitem__(self, index: int | slice | list[int]) -> dict: ...

    @overload
    def __getitem__(self, index: str) -> list[Any]: ...

    def __getitem__(self, index: int | slice | list[int] | str) -> dict | list:
        """Get the frame data at the specified index.

        Args:
            index (int | slice | list[int] | str): The index of the frame
                data to retrieve. If `index` is a slice, it returns
                a dict with values of list type.
                If `index` is a string, it is treated as a column name
                and returns the data for that column. Note that string
                index will load the entire column data, which may
                consume a lot of memory if the column is large.

        Returns:
            dict | list: The frame data at the specified index.
                if `index` is a string, returns the data for that column.
                Otherwise, returns a dict with the frame data.
        """

        ret: dict | list = self.frame_dataset[index]
        if self.meta_index2meta:
            ret = self.convert_meta_index2meta(data=ret, column_name=index)  # type: ignore
        return ret

    @overload
    def convert_meta_index2meta(
        self, data: dict[str, Any]
    ) -> dict[str, Any]: ...

    @overload
    def convert_meta_index2meta(
        self, data: list[Any], column_name: str
    ) -> list[Any]: ...

    def convert_meta_index2meta(
        self, data: dict[str, Any] | list, column_name: str | None = None
    ) -> dict | list:
        """Convert the metadata index in `data` to actual metadata objects.

        Args:
            data (dict | list): The data to convert. If `data` is a dict
                with index-based metadata, it will be converted to actual
                metadata objects. If `data` is a list, it will be converted
                to a list of metadata objects.
            column_name (str | None, optional): The name of the column
                to convert. If `data` is a list, this argument must be
                provided to convert the index-based metadata to actual
                metadata objects. Defaults to None.

        """

        ret = data
        if isinstance(data, list) and column_name is None:
            raise KeyError(
                "If data is a list, column_name must be provided to convert "
                "the index-based metadata to actual metadata objects."
            )

        if isinstance(ret, dict):
            ret = self._meta_index2meta(ret)
        else:
            ret_dict = {f"{column_name}": ret}
            ret_dict = self._meta_index2meta(ret_dict)
            assert len(ret_dict) == 1, "Expected only one key in the dict"
            ret = ret_dict[next(iter(ret_dict))]
        return ret

    def __len__(self) -> int:
        """Get the number of frames in the dataset."""
        return len(self.frame_dataset)

    @overload
    def get_meta(
        self, meta_type: type[MetaType], index: int | None
    ) -> MetaType | None: ...

    @overload
    def get_meta(
        self, meta_type: type[MetaType], index: list[int | None] | Column
    ) -> list[MetaType | None]: ...

    def get_meta(
        self,
        meta_type: type[MetaType],
        index: int | None | list[int | None] | Column,
    ) -> MetaType | None | list[MetaType | None]:
        """Get metadata of a specific type.

        This method retrieves metadata from the database using index.
        Possible metadata types include `Episode`, `Instruction`, `Robot`,
        and `Task`.

        Args:
            meta_type (type[MetaType]): The type of metadata to retrieve.
            index (int | None | list[int | None]): The index of the metadata
                to retrieve. If None, returns None.

        Returns:
            MetaType | None | list[MetaType | None]: The metadata object or
                None if not found. If `index` is a list, returns a list of
                metadata objects or None for each index.

        """
        if index is None:
            return None

        if isinstance(index, (list, Column)):
            # get all not None value
            non_none_index = [i for i in index if i is not None]
            if len(non_none_index) == 0:
                return [None for _ in index]
            # If index is a list, retrieve multiple metadata objects
            stmt = select(meta_type).where(meta_type.index.in_(index))
            with Session(self.db_engine) as session:
                ret = session.scalars(stmt).all()
                # make transient to avoid session issues
                for item in ret:
                    make_transient(item)
                # fill None for missing indices
                ret_dict: dict[int | None, Any] = {
                    item.index: item for item in ret
                }
                ret_dict[None] = None
                return [ret_dict.get(i, None) for i in index]
        else:
            with Session(self.db_engine) as session:
                ret = session.get(meta_type, index)
                if ret is not None:
                    make_transient(ret)
                return ret

    @property
    def dataset_format_version(self) -> str | None:
        """Get the dataset format version of loaded dataset."""
        return self._dataset_format_version


class ROMultiRowDataset(RODataset):
    """A dataset that returns multiple rows for each index.

    This class extends `RODataset` to support multi-row sampling.
    It provides a method to sample multiple rows based on the index dataset.


    If column is in the row_sampler, it will sample multiple rows
    for that column based on the index dataset, and the column in
    the returned row will be a list of rows. If the column is not
    in the row_sampler, it will return a single row for that column.

    Args:
        dataset_path (str): The path to the dataset directory.
        row_sampler (MultiRowSamplerConfig): The configuration for the
            multi-row sampler. It defines how to sample multiple
            rows based on the index dataset.
        storage_options (dict | None, optional): Additional Key/value pairs
            to be passed on to the file-system backend, if any.
            Defaults to None.
        meta_index2meta (bool, optional): Whether to convert the index-based
            metadata to actual metadata objects when accessing the dataset.
            Defaults to True.
    """

    def __init__(
        self,
        dataset_path: str,
        row_sampler: MultiRowSamplerConfig,
        storage_options: dict | None = None,
        meta_index2meta: bool = True,
    ):
        super().__init__(dataset_path, storage_options, meta_index2meta)
        self._set_row_sampler(row_sampler)

    def _set_row_sampler(self, row_sampler: MultiRowSamplerConfig) -> None:
        """Set the row sampler for the dataset."""
        self._row_sampler: MultiRowSampler = row_sampler()
        self._column_datasets = {
            col_name: self.frame_dataset.select_columns(column_names=col_name)
            for col_name in self._row_sampler.column_rows_keys
        }

    @staticmethod
    def from_dataset(
        dataset: RODataset,
        row_sampler: MultiRowSamplerConfig,
    ) -> ROMultiRowDataset:
        """Create a ROMultiRowDataset from an existing RODataset.

        Args:
            dataset (RODataset): The base dataset to extend.
            row_sampler (MultiRowSamplerConfig): The configuration for the
                multi-row sampler.

        """
        parent_state_dict = dataset._get_state_()
        ret = ROMultiRowDataset.__new__(ROMultiRowDataset)
        ret.__dict__.update(parent_state_dict)
        ret._set_row_sampler(row_sampler)
        return ret

    def __getitem__(self, index: int | slice | list[int]) -> dict:
        if isinstance(index, int):
            cur_row = super().__getitem__(index)
            for col_name, idx_rows in self._row_sampler.sample_row_idx(
                self.index_dataset, index
            ).items():
                col_dataset = self._column_datasets[col_name]
                cur_row[col_name] = [
                    col_dataset[idx][col_name] if idx is not None else None
                    for idx in idx_rows
                ]

            return cur_row
        else:
            if isinstance(index, slice):
                index = [
                    i for i in range(index.start, index.stop, index.step or 1)
                ]
            assert isinstance(index, list), (
                "Index must be an int, slice, or list of ints."
            )
            cur_rows = super().__getitem__(index)
            new_rows = {k: [] for k in self._row_sampler.column_rows_keys}
            for cur_idx in index:
                for col_name, idx_rows in self._row_sampler.sample_row_idx(
                    self.index_dataset, cur_idx
                ).items():
                    col_dataset = self._column_datasets[col_name]
                    new_rows[col_name].append(
                        [
                            col_dataset[idx][col_name]
                            if idx is not None
                            else None
                            for idx in idx_rows
                        ]
                    )
            for k in cur_rows:
                cur_rows[k] = new_rows.get(k, cur_rows[k])
            return cur_rows
