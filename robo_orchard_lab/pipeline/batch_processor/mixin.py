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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

from accelerate import Accelerator

if TYPE_CHECKING:
    from robo_orchard_lab.pipeline.mixin import PipelineMixin


__all__ = ["BatchProcessorMixin"]


class BatchProcessorMixin(ABC):
    """A processor for handling batches in a training or inference pipeline."""

    @abstractmethod
    def __call__(
        self,
        pipeline: "PipelineMixin",
        accelerator: Accelerator,
        batch: Any,
        model: Callable,
        **hook_kwargs,
    ) -> None:
        """Executes the batch processing pipeline.

        Args:
            pipeline (PipelineMixin): The pipeline object managing hooks.
            accelerator (Accelerator): An instance of `Accelerator`.
            batch (Any): Input batch data.
            model (Callable): The model function or callable.
            **hook_kwargs: Additional arguments for hooks.
        """
        pass
