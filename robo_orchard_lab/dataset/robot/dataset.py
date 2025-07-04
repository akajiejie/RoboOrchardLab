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

import json
import os
from typing import TypeVar

import fsspec
from datasets import Dataset
from sqlalchemy import URL, Engine
from sqlalchemy.orm import Session, make_transient

from robo_orchard_lab.dataset.datatypes import *
from robo_orchard_lab.dataset.robot.columns import (
    PreservedIndexColumnsKeys,
    # PreservedIndexColumns,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    Task,
)
from robo_orchard_lab.dataset.robot.engine import create_engine

MetaType = TypeVar("MetaType", Episode, Instruction, Robot, Task)
"""A type variable for metadata types in the RoboOrchard dataset."""


class RODataset:
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

    """

    frame_dataset: Dataset
    """The Hugging Face Dataset object containing the frame data."""
    db_engine: Engine
    """The SQLAlchemy engine for the meta database"""

    index_dataset: Dataset
    """The same as `frame_dataset`, but only contains the preserved index columns."""  # noqa: E501

    def __init__(self, dataset_path: str, storage_options: dict | None = None):
        self.frame_dataset = Dataset.load_from_disk(
            dataset_path, storage_options=storage_options
        )
        self.index_dataset = self.frame_dataset.select_columns(
            column_names=list(PreservedIndexColumnsKeys)
        )
        # recover state dict
        from datasets import config as hg_datasets_config

        state_file = os.path.join(
            dataset_path, hg_datasets_config.DATASET_STATE_JSON_FILENAME
        )
        state: dict = json.load(open(state_file, "r"))
        self._dataset_format_version = state.get("robo_orchard_state", {}).get(
            "dataset_format_version", None
        )
        #
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
        self.db_engine = create_engine(
            url=URL.create(drivername=drivername, database=db_path),
            readonly=True,
        )

    def get_meta(
        self, meta_type: type[MetaType], index: int | None
    ) -> MetaType | None:
        """Get metadata of a specific type.

        This method retrieves metadata from the database using index.
        Possible metadata types include `Episode`, `Instruction`, `Robot`,
        and `Task`.

        Args:
            meta_type (type[MetaType]): The type of metadata to retrieve.
            index (int | None): The index of the metadata to retrieve.
                If None, returns None.
        """
        if index is None:
            return None

        with Session(self.db_engine) as session:
            ret = session.get(meta_type, index)
            if ret is not None:
                make_transient(ret)
            return ret

    @property
    def dataset_format_version(self) -> str | None:
        """Get the dataset format version of loaded dataset."""
        return self._dataset_format_version
