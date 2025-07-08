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
import subprocess
import tempfile

import pytest


def test_robotwin_lmdb_data_packer(
    PROJECT_ROOT: str, ROBO_ORCHARD_TEST_WORKSPACE: str
):
    """Test robotwin lmdb data packer."""
    test_data_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/",
        "robotwin/cvpr_round2_branch",
    )
    test_data_path_v2 = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/",
        "robotwin/v2.0/origin_data",
    )
    os.chdir(PROJECT_ROOT)
    with tempfile.TemporaryDirectory() as workspace_root:
        cmd = (
            " ".join(
                [
                    "python3",
                    "robo_orchard_lab/dataset/robotwin/robotwin_packer.py",
                    f"--input_path {test_data_path}",
                    f"--output_path {workspace_root}",
                    "--task_names blocks_stack_three",
                    "--embodiment aloha-agilex-1",
                    "--robotwin_aug m1_b1_l1_h0.03_c0",
                    "--camera_name D435",
                ]
            ),
        )
        ret_code = subprocess.check_call(cmd, shell=True)

        # Check if the script ran successfully
        assert ret_code == 0, f"Script failed with return code: {ret_code}"

        cmd = (
            " ".join(
                [
                    "python3",
                    "robo_orchard_lab/dataset/robotwin/robotwin_packer.py",
                    f"--input_path {test_data_path_v2}",
                    f"--output_path {workspace_root}",
                    "--task_names place_empty_cup",
                    "--config_name base_setting",
                ]
            ),
        )
        ret_code = subprocess.check_call(cmd, shell=True)

        # Check if the script ran successfully
        assert ret_code == 0, f"Script failed with return code: {ret_code}"


def test_robotwin_lmdb_data(
    PROJECT_ROOT: str, ROBO_ORCHARD_TEST_WORKSPACE: str
):
    """Test robotwin lmdb data packer."""
    test_data_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/",
        "robotwin/main_branch",
    )
    os.chdir(PROJECT_ROOT)
    from robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset import (
        RoboTwinLmdbDataset,
    )

    dataset = RoboTwinLmdbDataset(
        paths=os.path.join(test_data_path, "lmdb"),
    )
    assert dataset.num_episode == 1
    assert len(dataset) == 268

    data = dataset[0]
    for key in [
        "uuid",
        "step_index",
        "intrinsic",
        "T_world2cam",
        "T_base2world",
        "joint_state",
        "ee_state",
        "imgs",
        "depths",
        "text",
    ]:
        assert key in data


if __name__ == "__main__":
    pytest.main(["-s", __file__])
