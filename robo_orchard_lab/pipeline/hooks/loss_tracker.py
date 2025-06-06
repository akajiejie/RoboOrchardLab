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

from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)

__all__ = ["LossMovingAverageTracker", "LossMovingAverageTrackerConfig"]


logger = logging.getLogger(__name__)


class LossMovingAverageTracker(PipelineHooks):
    """Loss moving average tracker.

    A loss tracker hook for log the losses at specified step intervals.

    Args:
        cfg (LossMovingAverageTrackerConfig): Configuration objects containing
            parameters for loss tracker.

    """

    def __init__(self, cfg: LossMovingAverageTrackerConfig):
        super().__init__()
        self.step_log_freq = cfg.step_log_freq
        self.reset()
        self.register_hook(
            "on_step", HookContext.from_callable(after=self._on_step_end)
        )

    def reset(self):
        self.losses = {}

    def _on_step_end(self, args: PipelineHookArgs):
        if not args.accelerator.is_main_process:
            return

        for k, v in args.model_outputs.items():
            if "loss" not in k:
                continue
            if k not in self.losses:
                self.losses[k] = [0, 0]
            self.losses[k][0] += v
            self.losses[k][1] += 1

        if (args.step_id + 1) % self.step_log_freq == 0:
            msg = "Epoch[{}/{}] Step[{}] GlobalStep[{}/{}]: ".format(
                args.epoch_id,
                args.max_epoch - 1 if args.max_epoch is not None else "NA",
                args.step_id,
                args.global_step_id,
                args.max_step - 1 if args.max_step is not None else "NA",
            )
            total_loss = 0
            for k, v in self.losses.items():
                v = v[0].item() / v[1]
                msg += f"{k}[{v:.4f}] "
                total_loss += v

            msg += f"total_loss[{total_loss:.4f}]"

            logger.info(msg)
            self.reset()


class LossMovingAverageTrackerConfig(
    PipelineHooksConfig[LossMovingAverageTracker]
):
    """Configuration class for LossMovingAverageTracker."""

    class_type: type[LossMovingAverageTracker] = LossMovingAverageTracker

    step_log_freq: Optional[int] = 25
