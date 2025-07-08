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
import functools
import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import yaml
from robo_orchard_core.envs.env_base import EnvBase, EnvBaseCfg
from robo_orchard_core.utils.logging import LoggerManager

if TYPE_CHECKING:
    from envs._base_task import Base_Task


logger = LoggerManager().get_child(__name__)


@dataclass
class RoboTwinEnvStepReturn:
    obvervation: dict[str, Any]
    task_success: bool
    take_action_cnt: int
    step_lim: int


class RoboTwinEnv(EnvBase[Any]):
    """RoboTwin environment.

    This class provides RoboTwin environment with robo_orchard_core interface.
    To make it work, you need to install RoboTwin and set the `RoboTwin_PATH`
    environment variable to the path of the RoboTwin package.


    """

    def __init__(self, cfg: RoboTwinEnvCfg):
        self.cfg = cfg
        episode_info = None

        if self.cfg.check_expert:
            episode_info, success = self._check_expert_traj()
            while not success:
                logger.warning(
                    f"Task {self.cfg.task_name} with seed {self.cfg.seed} "
                    "failed to complete using expert trajectory. "
                    "Using a new seed."
                )
                self.cfg.seed += 1
                episode_info, success = self._check_expert_traj()
        self._task = self._create_task()

        from description.utils.generate_episode_instructions import (
            generate_episode_descriptions,
        )

        if episode_info is not None:
            self._instructions = generate_episode_descriptions(
                self.cfg.task_name, [episode_info["info"]]
            )[0]
        else:
            self._instructions = None

    @property
    def instructions(self) -> dict | None:
        """The instructions for the environment.

        Only valid when check_expert is True.
        """
        return self._instructions

    def _create_task(self) -> Base_Task:
        with in_robotwin_workspace():
            task = create_task_from_name(self.cfg.task_name)
            task_config = self.cfg.get_task_config()
            task.setup_demo(**task_config)  # type: ignore
            return task

    def _check_expert_traj(self) -> tuple[dict | None, bool]:
        """Check whether current config can success if using expert trajectory.

        Returns:
            tuple[dict | None, bool]: A tuple containing the episode
                information and a boolean indicating whether the task
                was successful.

        """
        with in_robotwin_workspace():
            task = create_task_from_name(self.cfg.task_name)
            config = self.cfg.get_task_config()
            config["render_freq"] = 0
            try:
                task.setup_demo(**config)  # type: ignore
                episode_info = task.play_once()  # type: ignore
            except Exception as e:
                logger.error(
                    f"Failed to play the task config {self.cfg} "
                    f"with error: {e}"
                )
                return None, False
            finally:
                task.close_env()

        success: bool = task.plan_success and task.check_success()  # type: ignore
        return episode_info, success

    def step(
        self, action: list[float] | np.ndarray
    ) -> RoboTwinEnvStepReturn | None:
        """Take a step in the environment.

        Args:
            action (list[float] | np.ndarray): The action to take in the
                environment. Actually it is the joint positions of the
                robot. The action should be 1-D array with length matching
                the task.

        Returns:
            RoboTwinEnvStepReturn | None: The observation and environment
                state after taking the action, or None if the episode
                has ended.


        """
        if isinstance(action, np.ndarray):
            if action.ndim != 1:
                raise ValueError(
                    "Action should be a 1-D array, "
                    f"but got {action.ndim} dimensions."
                )

        self._task.take_action(action)
        if self._task.take_action_cnt >= self._task.step_lim:  # type: ignore
            logger.warning(
                f"Task {self.cfg.task_name} reached step limit "
                f"{self._task.step_lim}. Ending episode."
            )
            return None

        return RoboTwinEnvStepReturn(
            obvervation=self._task.get_obs(),
            task_success=self._task.eval_success,
            take_action_cnt=self._task.take_action_cnt,
            step_lim=self._task.step_lim,  # type: ignore
        )

    def reset(
        self, env_ids: Sequence[int] | None = None, clear_cache: bool = False
    ):
        """Reset the environment."""

        self.close(clear_cache=clear_cache)
        task_config = self.cfg.get_task_config()
        self._task.setup_demo(**task_config)  # type: ignore

    def close(self, clear_cache: bool = True):
        """Close the environment."""
        self._task.close_env(clear_cache=True)
        if self._task.render_freq > 0:
            self._task.viewer.close()

    @property
    def num_envs(self) -> int:
        # always 1 because RoboTwin does not support multi-envs
        return 1


class RoboTwinEnvCfg(EnvBaseCfg[RoboTwinEnv]):
    """Configuration for the RoboTwin environment."""

    class_type: type[RoboTwinEnv] = RoboTwinEnv

    task_name: str
    """The name of the task to run, e.g., 'place_object_scale'."""

    seed: int = 0
    """The random seed for the environment."""

    episode_id: int = 0
    """The episode ID for the environment, used for logging and tracking."""

    check_expert: bool = False
    """Whether to check the expert trajectory for the task.

    If true, the environment will attempt to run the task with given seed
    to check if the task can be completed successfully using the expert
    trajectory. If fails, new seed will be generated and used.
    """

    eval_mode: bool = False
    """Whether for evaluation.

    If true, the environment will use unseen texture_type.
    """

    max_instruction_num: int = 10
    """The maximum number of instructions to generate for the env."""

    def __post_init__(self):
        task_config_path = self.task_config_path
        if not os.path.exists(task_config_path):
            raise FileNotFoundError(
                f"Task configuration file {task_config_path} does not exist."
            )

    @property
    def task_config_path(self) -> str:
        """Path to the task configuration file."""
        robo_twin_root = config_robotwin_path()
        ret = os.path.join(
            robo_twin_root, "task_config", f"{self.task_name}.yml"
        )
        if os.path.exists(ret):
            return ret
        else:
            # for RoboTwin 2.0.
            return os.path.join(
                robo_twin_root, "task_config", "_config_template.yml"
            )

    @property
    def embodiment_config_path(self) -> str:
        """Path to the embodiment configuration file."""
        robo_twin_root = config_robotwin_path()
        return os.path.join(
            robo_twin_root, "task_config", "_embodiment_config.yml"
        )

    @property
    def camera_config_path(self) -> str:
        """Path to the camera configuration file."""
        robo_twin_root = config_robotwin_path()
        return os.path.join(
            robo_twin_root, "task_config", "_camera_config.yml"
        )

    def get_task_config(self) -> dict[str, Any]:
        """Get the configuration for the task."""

        with (
            open(self.task_config_path, "r", encoding="utf-8") as f,
        ):
            task_config = yaml.load(f.read(), Loader=yaml.FullLoader)
            return self._update_task_config(task_config)

    def _update_task_config(self, task_args: dict[str, Any]) -> dict[str, Any]:
        """Update the task configuration.

        The function reads additional configuration files for task arguments
        such as embodiment and camera settings, and updates the task arguments
        accordingly. The returned dictionary is used for `setup_demo()`.

        """
        embodiment_type: list[str] = task_args.get("embodiment")  # type: ignore
        with open(self.embodiment_config_path, "r", encoding="utf-8") as f:
            embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def get_embodiment_file(embodiment_type: str) -> str:
            robot_file = embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise ValueError("No embodiment files")
            return robot_file

        def get_embodiment_config(robot_file):
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, "r", encoding="utf-8") as f:
                embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
            return embodiment_args

        with open(self.camera_config_path, "r", encoding="utf-8") as f:
            camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_camera_type = task_args["camera"]["head_camera_type"]
        task_args["head_camera_h"] = camera_config[head_camera_type]["h"]
        task_args["head_camera_w"] = camera_config[head_camera_type]["w"]

        if len(embodiment_type) == 1:
            task_args["left_robot_file"] = get_embodiment_file(
                embodiment_type[0]
            )
            task_args["right_robot_file"] = get_embodiment_file(
                embodiment_type[0]
            )
            task_args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            task_args["left_robot_file"] = get_embodiment_file(
                embodiment_type[0]
            )
            task_args["right_robot_file"] = get_embodiment_file(
                embodiment_type[1]
            )
            task_args["embodiment_dis"] = embodiment_type[2]
            task_args["dual_arm_embodied"] = False
        else:
            raise RuntimeError("embodiment items should be 1 or 3")

        task_args["left_embodiment_config"] = get_embodiment_config(
            task_args["left_robot_file"]
        )
        task_args["right_embodiment_config"] = get_embodiment_config(
            task_args["right_robot_file"]
        )
        if len(embodiment_type) == 1:
            embodiment_name = str(embodiment_type[0])
        else:
            embodiment_name = (
                str(embodiment_type[0]) + "+" + str(embodiment_type[1])
            )
        task_args["embodiment_name"] = embodiment_name

        # update attributes in self
        task_args["seed"] = self.seed
        task_args["now_ep_num"] = self.episode_id
        task_args["eval_mode"] = self.eval_mode

        return task_args


@functools.lru_cache(maxsize=1)
def config_robotwin_path() -> str:
    robo_twin_path = os.environ.get("RoboTwin_PATH", default=None)
    if robo_twin_path is None:
        raise ValueError(
            "RoboTwin_PATH environment variable is not set. "
            "Please set it to the path of the RoboTwin package."
        )
    if robo_twin_path not in sys.path:
        sys.path.append(robo_twin_path)
    return robo_twin_path


@contextmanager
def in_robotwin_workspace():
    """Context manager to temporarily change the `cwd` to the RoboTwin root."""
    robotwin_root = config_robotwin_path()
    original_cwd = os.getcwd()
    os.chdir(robotwin_root)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def create_task_from_name(task_name: str) -> Base_Task:
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception as _:
        raise ImportError(
            f"Failed to import environment class {task_name} from "
            f"module {envs_module.__name__}. "
            "Please ensure the class name matches the task name and "
            "is defined in the module."
        )
    return env_instance
