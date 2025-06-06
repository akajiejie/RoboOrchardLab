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
import logging
from typing import Optional

from accelerate import Accelerator

from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)

__all__ = ["SaveCheckpoint", "SaveCheckpointConfig"]


logger = logging.getLogger(__name__)


class SaveCheckpoint(PipelineHooks):
    """Checkpoint hook.

    A checkpointing hook for saving model state at specified epoch or
    step intervals.


    Args:
        cfg (SaveCheckpointConfig): Configuration object containing
            parameters for checkpointing.

    """

    def __init__(
        self,
        cfg: SaveCheckpointConfig,
    ):
        super().__init__()
        self.save_root = cfg.save_root
        self.save_epoch_freq = cfg.save_epoch_freq
        self.save_step_freq = cfg.save_step_freq
        self._is_checked = False

        self.register_hook(
            channel="on_step",
            hook=HookContext.from_callable(after=self._on_step_end),
        )
        self.register_hook(
            channel="on_epoch",
            hook=HookContext.from_callable(after=self._on_epoch_end),
        )

    def _check(self, accelerator: Accelerator) -> None:
        if self._is_checked:
            return
        if (
            not accelerator.project_configuration.automatic_checkpoint_naming
            and self.save_root is None
        ):
            raise ValueError(
                "save_root is required when `automatic_checkpoint_naming of Accelerator is False"  # noqa: E501
            )
        if (
            accelerator.project_configuration.automatic_checkpoint_naming
            and self.save_root is not None
        ):
            logger.warning(
                "Ignore customize save_root = {} since `automatic_checkpoint_naming` is True for Accelerator".format(  # noqa: E501
                    self.save_root
                )
            )
        self._is_checked = True

    def _on_step_end(self, args: PipelineHookArgs) -> None:
        """Callback when step ends.

        Saves a checkpoint at the end of a step if `save_step_freq` conditions
        are met.


        Args:
            args (HookArgs): Arguments containing the `global_step_id` and
                `accelerator` to facilitate saving state.

        Logs:
            A message indicating the checkpoint location.
        """
        if (
            self.save_step_freq is not None
            and (args.global_step_id + 1) % self.save_step_freq == 0
        ):
            self._check(args.accelerator)
            args.accelerator.wait_for_everyone()

            save_location = args.accelerator.save_state(self.save_root)  # type: ignore
            logger.info(
                "Save checkpoint at the end of step {} to {}".format(
                    args.global_step_id, save_location
                )
            )

    def _on_epoch_end(self, args: PipelineHookArgs) -> None:
        """Callback when epoch ends.

        Saves a checkpoint at the end of an epoch if `save_epoch_freq`
        conditions are met.

        Args:
            args (HookArgs): Arguments containing the `epoch_id` and
                `accelerator` to facilitate saving state.

        Logs:
            A message indicating the checkpoint location.
        """
        if (
            self.save_epoch_freq is not None
            and (args.epoch_id + 1) % self.save_epoch_freq == 0
        ):
            self._check(args.accelerator)
            args.accelerator.wait_for_everyone()

            save_location = args.accelerator.save_state(self.save_root)  # type: ignore
            logger.info(
                "Save checkpoint at the end of epoch {} to {}".format(
                    args.epoch_id, save_location
                )
            )


class SaveCheckpointConfig(PipelineHooksConfig[SaveCheckpoint]):
    """Configuration class for SaveCheckpoint."""

    class_type: type[SaveCheckpoint] = SaveCheckpoint

    save_root: Optional[str] = None
    """The root directory where checkpoints are saved."""

    save_epoch_freq: Optional[int] = 1
    """Frequency of saving checkpoints based on epochs."""

    save_step_freq: Optional[int] = None
    """Frequency of saving checkpoints based on steps."""

    def __post_init__(self):
        if self.save_epoch_freq is None and self.save_step_freq is None:
            raise ValueError(
                "Either `save_epoch_freq` or `save_step_freq` must be specified. Both cannot be None."  # noqa: E501
            )
        if self.save_epoch_freq is not None and self.save_epoch_freq < 1:
            raise ValueError(
                "save_epoch_freq = {} < 1 is not allowed".format(
                    self.save_epoch_freq
                )
            )
        if self.save_step_freq is not None and self.save_step_freq < 1:
            raise ValueError(
                "save_step_freq = {} < 1 is not allowed".format(
                    self.save_step_freq
                )
            )
