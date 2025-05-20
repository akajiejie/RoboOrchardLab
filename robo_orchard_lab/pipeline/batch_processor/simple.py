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

import logging
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
    Tuple,
)

import torch
from accelerate import Accelerator
from torchvision.transforms import Compose

from robo_orchard_lab.pipeline.batch_processor.mixin import BatchProcessorMixin
from robo_orchard_lab.pipeline.hooks.mixin import (
    PipelineHookArgs,
    PipelineHooks,
)
from robo_orchard_lab.utils import as_sequence

__all__ = ["SimpleBatchProcessor"]


logger = logging.getLogger(__file__)


class LossNotProvidedError(Exception):
    pass


class SimpleBatchProcessor(BatchProcessorMixin):
    """A processor for handling batches in a training or inference pipeline.

    Attributes:
        need_backward (bool): Whether backward computation is needed.
        transform (Compose): Transformation pipeline for batch data.
        is_init (bool): Whether the processor has been initialized.

    Methods:
        do_forward: Abstract method to define the forward pass.
    """

    def __init__(
        self,
        need_backward: bool = True,
        transforms: Optional[Callable | Sequence[Callable]] = None,
    ) -> None:
        """Initializes the batch processor.

        Args:
            need_backward (bool): Whether backward computation is needed.
                If True, the loss should be provided during the forward pass.

            transforms (Optional[Callable | Sequence[Callable]]): A callable
                or a sequence of callables for transforming the batch.
        """
        self.need_backward = need_backward
        self.transform = Compose(as_sequence(transforms))
        self.is_init = False

    def _initialize(self, accelerator: Accelerator) -> None:
        if self.is_init:
            return

        for key, obj in vars(self).items():
            if isinstance(obj, torch.nn.Module):
                new_obj = accelerator.prepare(obj)
                setattr(self, key, new_obj)

        self.is_init = True

    def _get_hook_args(
        self,
        accelerator: Accelerator,
        batch: Optional[Any] = None,
        model_outputs: Optional[Any] = None,
        reduce_loss: Optional[torch.Tensor] = None,
        **hook_kwargs,
    ) -> PipelineHookArgs:
        return PipelineHookArgs(
            accelerator=accelerator,
            batch=batch,
            model_outputs=model_outputs,
            reduce_loss=reduce_loss,
            **hook_kwargs,
        )

    @abstractmethod
    def do_forward(
        self,
        model: Callable,
        batch: Any,
        device: torch.device,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """Defines the forward pass logic for the model.

        This method handles the execution of the forward pass, processing
        the input batch and computing the outputs of the model. It also
        optionally computes a loss tensor if required for training.

        Args:
            model (Callable): The model to be used for inference or training.
                It should be a callable object (e.g., a PyTorch `nn.Module`
                or a function).
            batch (Any): The input batch data. This can be a tuple, dictionary,
                or other structure, depending on the data pipeline's format.
            device (torch.device): The device on which computation will be
                executed (e.g., `torch.device('cuda')` for GPU or
                `torch.device('cpu')`).

        Returns:
            Tuple[Any, Optional[torch.Tensor]]:
                - The first element is the model's outputs. It can be any type
                that the model produces, such as a tensor, a list of tensors,
                or a dictionary.
                - The second element is an optional reduced loss tensor. This
                is used during training when backward computation is required.
                If loss is not applicable (e.g., during inference), this value
                can be `None`.

        Notes:
            - In most cases, the `accelerator` will already ensure that both
            the model and the batch data are moved to the appropriate device
            before the forward pass.
            - However, if additional operations or modifications are performed
            on the batch data or model within this method, it is the
            responsibility of the implementation to confirm they remain on
            the correct device.
            - The returned loss tensor should already be reduced (e.g., mean or
            sum over batch elements) to facilitate the backward pass.
            - This method does not handle backpropagation; it focuses solely
            on the forward pass.
            - The transformation of the input batch, if needed, should already
            be handled prior to this method via the `self.transform` pipeline.
        """
        pass

    def __call__(
        self,
        pipeline_hooks: PipelineHooks,
        accelerator: Accelerator,
        batch: Any,
        model: Callable,
        **hook_kwargs,
    ) -> None:
        """Executes the batch processing pipeline.

        Args:
            pipeline_hooks (PipelineHooks): The pipeline object managing hooks.
            accelerator (Accelerator): An instance of `Accelerator`.
            batch (Any): Input batch data.
            model (Callable): The model function or callable.
            **hook_kwargs: Additional arguments for hooks.
        """

        self._initialize(accelerator=accelerator)

        with pipeline_hooks.begin(
            "on_batch",
            self._get_hook_args(
                accelerator=accelerator, batch=batch, **hook_kwargs
            ),
        ) as on_batch_hook_args:
            ts_batch = self.transform(batch)

            with pipeline_hooks.begin(
                "on_forward",
                arg=self._get_hook_args(
                    accelerator=accelerator, batch=ts_batch, **hook_kwargs
                ),
            ) as on_forward_hook_args:
                outputs, reduce_loss = self.do_forward(
                    model=model,
                    batch=ts_batch,
                    device=accelerator.device,
                )
                on_forward_hook_args.model_outputs = outputs
                on_forward_hook_args.reduce_loss = reduce_loss

            if self.need_backward:
                if reduce_loss is None:
                    raise LossNotProvidedError()

                with pipeline_hooks.begin(
                    "on_backward",
                    arg=self._get_hook_args(
                        accelerator=accelerator,
                        batch=ts_batch,
                        model_outputs=outputs,
                        reduce_loss=reduce_loss,
                        **hook_kwargs,
                    ),
                ):
                    accelerator.backward(reduce_loss)

            on_batch_hook_args.model_outputs = outputs
            on_batch_hook_args.reduce_loss = reduce_loss

        return outputs
