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

from abc import abstractmethod
from typing import Any, Optional, Sequence, Union

from robo_orchard_lab.pipeline.hooks.mixin import HookMixin
from robo_orchard_lab.utils import as_sequence

__all__ = ["PipelineMixin"]


class PipelineMixin:
    def __init__(
        self, hooks: Optional[Union[HookMixin, Sequence[HookMixin]]] = None
    ) -> None:
        self.hooks = as_sequence(
            hooks, check_type=True, required_types=HookMixin
        )

    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        pass

    def _impl(self, name: str, *args: Any, **kwargs: Any) -> None:
        for hook_i in self.hooks:
            getattr(hook_i, name)(*args, **kwargs)

    def on_loop_begin(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_loop_begin", *args, **kwargs)

    def on_loop_end(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_loop_end", *args, **kwargs)

    def on_epoch_begin(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_epoch_begin", *args, **kwargs)

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_epoch_end", *args, **kwargs)

    def on_step_begin(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_step_begin", *args, **kwargs)

    def on_step_end(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_step_end", *args, **kwargs)

    def on_batch_begin(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_batch_begin", *args, **kwargs)

    def on_batch_end(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_batch_end", *args, **kwargs)

    def on_optimizer_step_begin(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_optimizer_step_begin", *args, **kwargs)

    def on_optimizer_step_end(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_optimizer_step_end", *args, **kwargs)

    def on_forward_begin(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_forward_begin", *args, **kwargs)

    def on_forward_end(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_forward_end", *args, **kwargs)

    def on_backward_begin(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_backward_begin", *args, **kwargs)

    def on_backward_end(self, *args: Any, **kwargs: Any) -> None:
        self._impl("on_backward_end", *args, **kwargs)
