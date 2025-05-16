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
from __future__ import annotations
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeAlias,
    TypeVar,
)

import torch
from accelerate import Accelerator
from accelerate.scheduler import AcceleratedScheduler
from ordered_set import OrderedSet
from robo_orchard_core.utils.hook import RemoveableHandle
from typing_extensions import Self

from robo_orchard_lab.utils import as_sequence

__all__ = [
    "PipelineHooks",
    "PipelineHookArgs",
    "PipelineHookChanelType",
    "HookContext",
    "HookContextFromCallable",
    "HookContextChannel",
]


T = TypeVar("T")


@dataclass
class PipelineHookArgs:
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
    lr_scheduler: Optional[AcceleratedScheduler] = None
    batch: Optional[Any] = None
    model_outputs: Optional[Any] = None
    reduce_loss: Optional[torch.Tensor] = None


PipelineHookChanelType: TypeAlias = Literal[
    "on_loop",  # the whole training loop pipeline
    "on_epoch",  # in one epoch pipeline
    "on_step",  # in one step pipeline.
    "on_batch",  # in one batch pipeline
    # "on_optimizer_step", # not used now
    "on_forward",
    "on_backward",
]


class HookContext(Generic[T], metaclass=ABCMeta):
    """Context manager for executing hooks with a specific set of arguments.

    This class is used to manage the lifecycle of hooks, ensuring that
    the appropriate hook methods are called at the right times. It also
    provides a way to pass arguments to the hooks in a structured manner.
    """

    @abstractmethod
    def before(self, arg: T):
        """Prepare the context for executing hooks.

        This method should set up the context for executing hooks and
        return the arguments referenced by the hooks.

        The returned arguments will be used in the `after` method to clean up
        the context.

        Args:
            arg (T): The arguments to be passed to the hooks.
        """
        raise NotImplementedError("Subclasses must implement before method.")

    @abstractmethod
    def after(self, arg: T):
        """Clean up the context after executing hooks.

        This method should clean up any resources or state used during
        the execution of hooks.

        Args:
            arg (T): The arguments to be passed to the hooks.

        """
        raise NotImplementedError("Subclasses must implement after method.")

    @contextmanager
    def begin(self, arg: T) -> Generator[T, Any, None]:
        """Context manager for executing hooks with given arguments.

        This method sets up the context for executing hooks and yields
        the arguments to be passed to the hooks. After the hooks are executed,
        the context is cleaned up.
        """
        try:
            self.before(arg)
            yield arg
        finally:
            self.after(arg)


class HookContextFromCallable(HookContext[T]):
    """A hook context that is created from a callable.

    This class is used to create a hook context from a callable that
    takes a single argument and returns a value. The callable is called
    in the `before` method and the returned value is passed to the
    `after` method.

    Args:
        func (Callable[[T], T]): The callable to be used as the hook context.
    """

    def __init__(
        self,
        before: Callable[[T], None] | None = None,
        after: Callable[[T], None] | None = None,
    ):
        """Initialize the hook context with a callable."""
        self._before = before
        self._after = after

    def before(self, arg: T):
        if self._before is not None:
            self._before(arg)

    def after(self, arg: T):
        if self._after is not None:
            self._after(arg)


class HookContextChannel(Generic[T]):
    """A channel for managing hook context.

    The hook contexts are registered and executed in a specific order
    when the `begin` method is called. This class provides a way to
    register and unregister hook context handlers.

    """

    def __init__(self, name: str):
        self.name = name
        self._context_handlers: OrderedSet[HookContext[T]] = OrderedSet([])

    def __len__(self) -> int:
        """Get the number of registered hook context handlers."""
        return len(self._context_handlers)

    def register(self, hook: HookContext[T]):
        """Register a hook context handler.

        Args:
            hook (HookContext[T]): The hook context handler to register.

        Returns:
            RemoveableHandle: A handle to remove the registered hook.
        """
        self._context_handlers.add(hook)
        return RemoveableHandle(lambda: self._context_handlers.discard(hook))

    def register_hook_channel(
        self,
        channel: HookContextChannel[T],
    ) -> RemoveableHandle[Callable[[], None]]:
        """Register a hook context channel.

        Args:
            channel (HookContextChannel[T]): The hook context channel
                to register.

        Returns:
            RemoveableHandle: A handle to remove the registered hook.
        """
        if channel.name != self.name:
            raise ValueError(
                f"Cannot register hook channel {channel.name} to {self.name}"
            )
        to_remove = []
        for hook in channel._context_handlers:
            self._context_handlers.add(hook)
            to_remove.append(hook)

        def remove():
            for hook in to_remove:
                self._context_handlers.discard(hook)

        return RemoveableHandle(remove)

    def unregister_all(self):
        """Unregister all hook context handlers."""
        self._context_handlers.clear()

    @contextmanager
    def begin(self, arg: T):
        """Context manager for executing hooks with given arguments.

        This method sets up the context for executing hooks and yields
        the arguments to be passed to the hooks. After the hooks are executed,
        the context is cleaned up.

        """
        try:
            for hook in self._context_handlers:
                hook.before(arg)
            yield arg
        finally:
            for hook in self._context_handlers:
                hook.after(arg)


class PipelineHooks:
    def __init__(self):
        self.hooks: dict[
            PipelineHookChanelType, HookContextChannel[PipelineHookArgs]
        ] = {}
        for c in PipelineHookChanelType.__args__:
            self.hooks[c] = HookContextChannel(c)

    @contextmanager
    def begin(self, channel: PipelineHookChanelType, arg: PipelineHookArgs):
        with self.hooks[channel].begin(arg) as ctx:
            yield ctx

    def register_hook(
        self,
        channel: PipelineHookChanelType,
        hook: HookContext[PipelineHookArgs],
    ) -> RemoveableHandle[Callable[[], None]]:
        """Register a hook context handler.

        Args:
            channel (PipelineHookChanelType): The channel to register the hook.
            hook (HookContext[T]): The hook context handler to register.

        Returns:
            RemoveableHandle: A handle to remove the registered hook.
        """
        return self.hooks[channel].register(hook)

    def register_pipeline_hooks(
        self,
        hooks: PipelineHooks,
    ) -> RemoveableHandle[Callable[[], None]]:
        """Register a set of pipeline hooks.

        Args:
            hooks (PipelineHooks[T]): The pipeline hooks to register.

        Returns:
            RemoveableHandle: A handle to remove the registered hooks.
        """
        handles: list[RemoveableHandle] = []
        for channel, hook in hooks.hooks.items():
            handles.append(self.hooks[channel].register_hook_channel(hook))

        def remove():
            for handle in handles:
                handle()

        return RemoveableHandle(remove)

    def __iadd__(self, other: PipelineHooks) -> Self:
        """Add another set of pipeline hooks to the current instance.

        Args:
            other (PipelineHooks): The other set of pipeline hooks to add.

        Returns:
            PipelineHooks: The updated instance with the added hooks.
        """
        self.register_pipeline_hooks(other)
        return self

    def unregister_all(self):
        """Unregister all hook context handlers."""
        for channel in self.hooks.values():
            channel.unregister_all()

    @classmethod
    def from_hooks(
        cls: Type[Self],
        hooks: Self | Iterable[Self] | None,
    ) -> Self:
        """Create a new instance of the class from a list of hooks.

        Args:
            hooks (Self | Iterable[Self] | None): A list of hooks to register.

        Returns:
            Self: A new instance of the class with the registered hooks.
        """

        input_hooks: Sequence[PipelineHooks] = as_sequence(
            hooks, check_type=True, required_types=PipelineHooks
        )
        ret = cls()
        for hook in input_hooks:
            ret += hook
        return ret
