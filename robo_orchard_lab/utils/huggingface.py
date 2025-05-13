# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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
import re

__all__ = ["get_accelerate_project_last_checkpoint_id"]


def get_accelerate_project_last_checkpoint_id(project_dir: str) -> int:
    """Helper function to get last checkpoint id.

    Retrieves the ID of the last checkpoint in the specified project directory.

    This function specifically handles checkpoints saved using the
    `Accelerator.save_state` method from the Hugging Face `accelerate`
    library, which follows an automatic checkpoint naming convention.
    It searches the specified `project_dir/checkpoints` directory,
    extracts numerical IDs from folder names, and returns the highest ID,
    representing the most recent checkpoint.

    Args:
        project_dir (str): Path to the project directory containing the
            `checkpoints` folder. This directory should contain only
            checkpoints saved by `Accelerator.save_state`.

    Returns:
        int: The ID of the last (most recent) checkpoint found in the
            project directory. Returns `-1` if the `checkpoints` directory
            does not exist or is empty.

    Raises:
        ValueError: If no valid checkpoint IDs are found in the `checkpoints`
            directory.

    Example:
        >>> get_accelerate_project_last_checkpoint_id("/path/to/project")
        42

    Note:
        This function assumes that all entries in the `checkpoints` directory
        follow the automatic checkpoint naming pattern used by
        `Accelerator.save_state`. Checkpoints not saved with
        `Accelerator.save_state` may cause this function to fail.
    """
    input_dir = os.path.join(project_dir, "checkpoints")

    if not os.path.exists(input_dir):
        return -1

    iter_ids = []
    for folder_i in os.listdir(input_dir):
        iter_ids.append(
            int(re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder_i)[0])
        )

    iter_ids.sort()

    return iter_ids[-1]
