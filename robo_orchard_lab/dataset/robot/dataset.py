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

import os
from typing import TypeVar

import fsspec
from datasets import Dataset
from sqlalchemy import URL, Engine
from sqlalchemy.orm import Session, make_transient

from robo_orchard_lab.dataset.datatypes import *
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    Task,
)
from robo_orchard_lab.dataset.robot.engine import create_engine

MetaType = TypeVar("MetaType", Episode, Instruction, Robot, Task)


class RODataset:
    frame_dataset: Dataset
    """The Hugging Face Dataset object containing the frame data."""
    db_engine: Engine
    """The SQLAlchemy engine for the meta database"""

    def __init__(self, dataset_path: str, storage_options: dict | None = None):
        self.frame_dataset = Dataset.load_from_disk(
            dataset_path, storage_options=storage_options
        )
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
        """Get all metadata of a specific type."""
        if index is None:
            return None

        with Session(self.db_engine) as session:
            ret = session.get(meta_type, index)
            if ret is not None:
                make_transient(ret)
            return ret
