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

from typing import Literal

import torch

from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContextFromCallable,
    PipelineHookArgs,
    PipelineHooks,
)

__all__ = ["CUDAMemoryManager"]


class CUDAMemoryManager(PipelineHooks):
    """A CUDA memory manager to periodically release cached GPU memory.

    This class is designed to work with training pipelines that use hooks
    for step and epoch management. It integrates with PyTorch's
    `torch.cuda.empty_cache()` to clear unused memory, helping to avoid
    out-of-memory (OOM) errors during long-running training loops.

    Attributes:
        empty_cache_at (str): When to empty the CUDA cache. Either "step" or
            "epoch".
        empty_cache_freq (int): Frequency of cache clearing based on the
            specified granularity.

    Args:
        empty_cache_at (Literal["step", "epoch"], optional):
            Specifies whether to clear the cache at the end of each step or
            epoch. Default is "epoch".
        empty_cache_freq (int, optional):
            The frequency of cache clearing. For example:
            - If `empty_cache_at="step"`, this clears the cache every
            `empty_cache_freq` steps.
            - If `empty_cache_at="epoch"`, this clears the cache every
            `empty_cache_freq` epochs. Default is 1 (clear after every step
            or epoch).

    Methods:
        on_step_end: Clears the CUDA cache at the end of a step
            if conditions are met.
        on_epoch_end: Clears the CUDA cache at the end of an epoch if
            conditions are met.

    Examples:
        Basic Usage:
            >>> from robo_orchard_lab.pipeline.hooks.mixin import (
            ...     PipelineHookArgs,
            ... )
            >>> from robo_orchard_lab.pipeline.memory import CUDAMemoryManager
            >>>
            >>> memory_manager = CUDAMemoryManager(
            >>>     empty_cache_at="step", empty_cache_freq=10
            >>> )
            >>> # Simulate a training step
            >>> hook_args = PipelineHookArgs(global_step_id=9, epoch_id=0)
            >>> with memory_manager.begin("on_step", hook_args) as hook_args:
            >>>     ... # Simulate the end of a step
            # Clears the cache after 10 steps

        Epoch-Based Clearing:
            >>> memory_manager = CUDAMemoryManager(
            >>>     empty_cache_at="epoch", empty_cache_freq=2
            >>> )
            >>> # Simulate the end of an epoch
            >>> hook_args = PipelineHookArgs(global_step_id=99, epoch_id=1)
            >>> with memory_manager.begin("on_epoch", hook_args) as hook_args:
            >>>     ... # Simulate the end of an epoch
            # Clears the cache after every 2 epochs
    """

    def __init__(
        self,
        empty_cache_at: Literal["step", "epoch"] = "epoch",
        empty_cache_freq: int = 1,
    ):
        """Initializes the CUDA memory manager.

        Args:
            empty_cache_at (Literal["step", "epoch"], optional):
                Specifies whether to clear the cache at the end of each
                step or epoch. Default is "epoch".
            empty_cache_freq (int, optional):
                The frequency of cache clearing based on the specified
                granularity. Default is 1 (clear every step or epoch).
        """
        super().__init__()
        self.empty_cache_at = empty_cache_at
        self.empty_cache_freq = empty_cache_freq

        self.register_hook(
            "on_step", hook=HookContextFromCallable(after=self._on_step_end)
        )
        self.register_hook(
            "on_epoch", hook=HookContextFromCallable(after=self._on_epoch_end)
        )

    def _on_step_end(self, arg: PipelineHookArgs) -> None:
        """Hook invoked at the end of a training step.

        Clears the CUDA cache if `empty_cache_at` is set to "step" and the
        current step satisfies the clearing frequency (`empty_cache_freq`).

        Args:
            arg (PipelineHookArgs): Arguments passed by the pipeline, including
            `global_step_id`.

        """
        if (
            self.empty_cache_at == "step"
            and (arg.global_step_id + 1) % self.empty_cache_freq == 0
        ):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _on_epoch_end(self, arg: PipelineHookArgs) -> None:
        """Hook invoked at the end of a training epoch.

        Clears the CUDA cache if `empty_cache_at` is set to "epoch" and the
        current epoch satisfies the clearing frequency (`empty_cache_freq`).

        Args:
            arg (PipelineHookArgs): Arguments passed by the pipeline,
            including `epoch_id`.

        """
        if (
            self.empty_cache_at == "epoch"
            and (arg.epoch_id + 1) % self.empty_cache_freq == 0
        ):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
