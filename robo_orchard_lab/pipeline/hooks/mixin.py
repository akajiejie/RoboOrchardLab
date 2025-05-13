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

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
from accelerate import Accelerator

__all__ = ["HookMixin", "HookArgs"]


@dataclass
class HookArgs:
    """A data class for passing arguments to hook functions.

    This class serves as a container for various parameters and state
    information required by hooks at different stages of the training or
    evaluation pipeline. It is designed to be flexible and extensible for
    different training configurations.

    Attributes:
        accelerator (Accelerator): The `Accelerator` instance managing
            distributed training.
        epoch_id (int): The current epoch ID, starting from 0.
        step_id (int): The current step ID within the current epoch,
            starting from 0.
        global_step_id (int): The global step ID across all epochs,
            starting from 0.
        start_epoch (int): The epoch where training or evaluation started.
        max_epoch (Optional[int]): The maximum number of epochs for training.
        start_step (int): The step where training or evaluation started.
        max_step (Optional[int]): The maximum number of steps for training.
        dataloader (Optional[Iterable]): The data loader for feeding
            batches to the model.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer used
            during training.
        lr_scheduler (Optional[torch.optim.lr_scheduler.LRScheduler]):
            The learning rate scheduler used during training.
        batch (Optional[Any]): The current batch of data being processed.
        model_outputs (Optional[Any]): The outputs produced by the model
            during the forward pass.
        reduce_loss (Optional[torch.Tensor]): The loss value after reduction
            for distributed training.
    """

    accelerator: Accelerator
    epoch_id: int = 0
    step_id: int = 0
    global_step_id: int = 0
    start_epoch: int = 0
    max_epoch: Optional[int] = None
    start_step: int = 0
    max_step: Optional[int] = None
    dataloader: Optional[Iterable] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    batch: Optional[Any] = None
    model_outputs: Optional[Any] = None
    reduce_loss: Optional[torch.Tensor] = None


class HookMixin:
    """Abstract base class defining various hook methods.

    Each hook method is called at a specific stage in the training loop,
    allowing for custom logic to be implemented at each stage. Each method
    accepts a `HookArgs` object containing details about the current
    training state.

    Example:
        >>> class CustomHook(HookMixin):
        ...     def on_epoch_begin(self, arg: HookArgs):
        ...         print(f"Epoch {arg.epoch_id} started.")
        >>> custom_hook = CustomHook()
        >>> custom_hook.on_epoch_begin(
        ...     HookArgs(accelerator, epoch_id=1, step_id=0, global_step_id=0)
        ... )

    Notes:
        This class should be subclassed to implement custom hook logic.
        Each method may be overridden as needed to add specific behavior at
        each stage of the training loop.
    """

    def on_loop_begin(self, arg: HookArgs) -> None:
        """Called at the start of the training loop."""
        pass

    def on_loop_end(self, arg: HookArgs) -> None:
        """Called at the end of the training loop."""
        pass

    def on_epoch_begin(self, arg: HookArgs) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, arg: HookArgs) -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_begin(self, arg: HookArgs) -> None:
        """Called at the beginning of each step."""
        pass

    def on_step_end(self, arg: HookArgs) -> None:
        """Called at the end of each step."""
        pass

    def on_batch_begin(self, arg: HookArgs) -> None:
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, arg: HookArgs) -> None:
        """Called at the end of each batch."""
        pass

    def on_backward_begin(self, arg: HookArgs) -> None:
        """Called before the backward pass."""
        pass

    def on_backward_end(self, arg: HookArgs) -> None:
        """Called after the backward pass."""
        pass

    def on_forward_begin(self, arg: HookArgs) -> None:
        """Called before the forward pass."""
        pass

    def on_forward_end(self, arg: HookArgs) -> None:
        """Called after the forward pass."""
        pass

    def on_optimizer_step_begin(self, arg: HookArgs) -> None:
        """Called before the optimizer step."""
        pass

    def on_optimizer_step_end(self, arg: HookArgs) -> None:
        """Called after the optimizer step."""
        pass
