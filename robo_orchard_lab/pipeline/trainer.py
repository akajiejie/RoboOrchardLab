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
import os
from dataclasses import dataclass
from inspect import signature
from typing import Any, Dict, Iterable, List, Literal, Optional

import requests
import torch
from accelerate import Accelerator
from accelerate.scheduler import AcceleratedScheduler
from torch.utils.data import DataLoader

from robo_orchard_lab.pipeline.batch_processor.mixin import BatchProcessorMixin
from robo_orchard_lab.pipeline.hooks.mixin import (
    PipelineHookArgs,
    PipelineHooks,
)

__all__ = ["SimpleTrainer"]


logger = logging.getLogger(__file__)


@dataclass
class TrainerState:
    """A class to manage the state of the training process.

    Attributes:
        epoch (int): The current epoch in the training process.
        step (int): The current step within the current epoch.
        global_step (int): The total number of steps taken across all epochs.
    """

    epoch: int = 0
    step: int = 0
    global_step: int = 0

    def reset(self) -> None:
        """Resets the training state to initial values."""
        self.epoch = 0
        self.step = 0
        self.global_step = 0

    def state_dict(self) -> Dict[str, int]:
        """Returns the current state as a dictionary.

        Returns:
            Dict[str, int]: A dictionary containing the current epoch, step,
            and global_step.
        """
        return dict(
            epoch=self.epoch,
            step=self.step,
            global_step=self.global_step,
        )

    def load_state_dict(self, input: Dict[str, int]) -> None:
        """Loads the state from a dictionary.

        Args:
            input (Dict[str, int]): A dictionary containing the epoch, step,
            and global_step to load.
        """
        self.epoch = input["epoch"]
        self.step = input["step"]
        self.global_step = input["global_step"]

    def update_step(self) -> None:
        """Increments the step and global_step by 1."""
        self.step += 1
        self.global_step += 1

    def update_epoch(self) -> None:
        """Increments the epoch by 1 and resets the step to 0."""
        self.epoch += 1
        self.step = 0

    def sync_pipeline_hook_arg(self, hook_args: PipelineHookArgs) -> None:
        """Synchronizes the training state with the provided hook arguments.

        The hook arguments are updated with the current epoch, step, and
        global_step.

        Args:
            hook_args (PipelineHookArgs): The hook arguments to synchronize
            with.
        """
        hook_args.epoch_id = self.epoch
        hook_args.step_id = self.step
        hook_args.global_step_id = self.global_step


class SimpleTrainer:
    """A base trainer class that extends SimpleTrainer for training models.

    This trainer integrates with the `Accelerate` library for distributed
    training, supports custom batch processors, and provides hooks for
    monitoring and extending the training process.

    Args:
        model (torch.nn.Module): The model to be trained.
        accelerator (Accelerator): The `Accelerator` instance managing
            distributed training.
        batch_processor (Optional[BatchProcessorMixin]): A processor that
            defines how to handle each batch during training.
        dataloader (Optional[DataLoader]): The data loader for feeding batches
            to the model.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer used
            for training.
        lr_scheduler (Optional[torch.optim.lr_scheduler.LRScheduler]):
            The learning rate scheduler.
        lr_scheduler_step_at (str): Whether the learning rate scheduler
            steps at "epoch" or "step".
        max_step (Optional[int]): The maximum number of steps to train.
        max_epoch (Optional[int]): The maximum number of epochs to train.
        val_dataloader (Optional[DataLoader]): The data loader for validation.
        metric (Optional[Any]): The metric used for evaluation, with update,
            compute and reset functions.
        step_eval_freq (Optional[int]): The frequency of evaluation in
            terms of steps.
        epoch_eval_freq (Optional[int]): The frequency of evaluation in
            terms of epochs.
        resume_from (Optional[str]): The path or URL to resume training from.
        resume_share_dir (Optional[str]): The directory to save resume files.
        grad_clip_mode (Optional[str]): The mode for gradient clipping
            ("value" or "norm").
        grad_clip_value (Optional[float]): The value for gradient clipping.
        grad_max_norm (Optional[float]): The maximum norm for gradient
            clipping.
        grad_norm_type (int): The type of norm used for gradient clipping.

    Methods:
        __call__: Executes the training loop, iterating over epochs and steps.
        eval: Executes evaluation on the self.val_dataloader.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        accelerator: Accelerator,
        batch_processor: Optional[BatchProcessorMixin] = None,
        dataloader: Optional[DataLoader | Iterable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        lr_scheduler_step_at: Literal["step", "epoch"] = "step",
        max_step: Optional[int] = None,
        max_epoch: Optional[int] = None,
        val_dataloader: Optional[DataLoader | Iterable] = None,
        metric: Any = None,
        step_eval_freq: Optional[int] = None,
        epoch_eval_freq: Optional[int] = None,
        resume_from: Optional[str] = None,
        resume_share_dir: Optional[str] = None,
        grad_clip_mode: Optional[str] = None,
        grad_clip_value: Optional[float] = None,
        grad_max_norm: Optional[float] = None,
        grad_norm_type: int = 2,
        hooks: PipelineHooks | Iterable[PipelineHooks] | None = None,
    ):
        self.hooks = PipelineHooks.from_hooks(hooks)
        self.batch_processor = batch_processor
        self.lr_scheduler_step_at = lr_scheduler_step_at
        self.max_step = max_step
        self.max_epoch = max_epoch
        self.metric = metric
        self.step_eval_freq = step_eval_freq
        self.epoch_eval_freq = epoch_eval_freq
        self.resume_from = resume_from
        self.resume_share_dir = resume_share_dir
        assert grad_clip_mode in [None, "value", "norm"]
        self.grad_clip_mode = grad_clip_mode
        self.grad_clip_value = grad_clip_value
        self.grad_max_norm = grad_max_norm
        self.grad_norm_type = grad_norm_type

        self.accelerator = accelerator
        self.model: torch.nn.Module = accelerator.prepare(model)

        self.trainer_state = TrainerState()
        accelerator.register_for_checkpointing(self.trainer_state)

        if dataloader is not None:
            self.dataloader: Optional[DataLoader] = accelerator.prepare(
                dataloader
            )
        else:
            self.dataloader = None

        if optimizer is not None:
            self.optimizer: Optional[torch.optim.Optimizer] = (
                accelerator.prepare(optimizer)
            )
        else:
            self.optimizer = None

        if lr_scheduler is not None:
            self.lr_scheduler: Optional[AcceleratedScheduler] = (
                accelerator.prepare(lr_scheduler)
            )
        else:
            self.lr_scheduler = None

        if val_dataloader is not None:
            self.val_dataloader = accelerator.prepare(val_dataloader)
        else:
            self.val_dataloader = None

        self.resume(resume_from)

    def resume(self, resume_from: Optional[str]) -> None:
        """Resumes training from a checkpoint or URL.

        If `resume_from` is a local path, it loads the state directly.
        If `resume_from` is a URL, it downloads the checkpoint files and
        loads the state.
        """
        if resume_from is None:
            return
        if self.accelerator.is_main_process:
            logger.info(f"resume from: {resume_from}")
        if os.path.exists(resume_from):
            self.accelerator.load_state(resume_from)
        elif resume_from.startswith("http"):
            if self.resume_share_dir is not None:
                dir = self.resume_share_dir
            else:
                dir = "/tmp/resume_from"
            _acn = self.accelerator.project_configuration.automatic_checkpoint_naming  # noqa: E501
            self.accelerator.project_configuration.automatic_checkpoint_naming = (  # noqa: E501
                False
            )
            self.accelerator.save_state(dir)
            if self.accelerator.is_main_process:
                file_names = list(os.listdir(dir))
                for file in file_names:
                    with requests.get(
                        f"{resume_from}/{file}", stream=True
                    ) as r:
                        if r.status_code != 200:
                            logger.warning(
                                f"request {file} failed, status {r.status_code}"  # noqa: E501
                            )
                            continue
                        with open(os.path.join(dir, file), "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
            self.accelerator.wait_for_everyone()
            self.accelerator.load_state(dir)
            self.accelerator.project_configuration.automatic_checkpoint_naming = (  # noqa: E501
                _acn
            )
        else:
            raise ValueError(f"resume data is not available: {resume_from}")

    def _get_hook_args(self, **kwargs) -> PipelineHookArgs:
        """Get Hook args.

        Creates and returns a HookArgs object with current training state
        and additional arguments.

        Args:
            **kwargs: Additional arguments to include in the HookArgs object.

        Returns:
            PipelineHookArgs: An object containing the current training state
                and additional arguments.
        """
        hookargs = PipelineHookArgs(
            accelerator=self.accelerator,
            max_step=self.max_step,
            max_epoch=self.max_epoch,
            epoch_id=self.trainer_state.epoch,
            step_id=self.trainer_state.step,
            global_step_id=self.trainer_state.global_step,
            dataloader=self.dataloader,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )

        for k, v in kwargs.items():
            setattr(hookargs, k, v)
        return hookargs

    def __call__(self) -> None:
        """Executes the training loop, iterating over epochs and steps.

        This method handles the main training loop, including evaluation
        at specified intervals, and calls the appropriate hooks at the
        beginning and end of each epoch and step.
        """
        assert self.dataloader is not None, "dataloader should not be None"
        assert self.optimizer is not None, "optimizer should not be None"
        assert self.batch_processor is not None, (
            "batch_processor should not be None"
        )
        if self.accelerator.is_main_process:
            logger.info("\n" + "=" * 50 + "BEGIN TRAINING" + "=" * 50)

        end_loop_flag = False
        self.model.train()

        def step(
            batch: Any,
            batch_processor: BatchProcessorMixin,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: Optional[AcceleratedScheduler],
        ):
            with self.hooks.begin(
                "on_step", self._get_hook_args()
            ) as on_step_hook_args:
                optimizer.zero_grad()
                with self.hooks.begin(
                    "on_batch", self._get_hook_args(batch=batch)
                ) as on_batch_hook_args:
                    batch_processor(
                        pipeline_hooks=self.hooks,
                        on_batch_hook_args=on_batch_hook_args,
                        model=self.model,
                    )
                    # update module_output to on_step_hook_args
                    on_step_hook_args.model_outputs = (
                        on_batch_hook_args.model_outputs
                    )

                self._clip_grad()
                optimizer.step()
                if (
                    self.lr_scheduler_step_at == "step"
                    and lr_scheduler is not None
                ):
                    lr_scheduler.scheduler.step()
                # evaluate when step matches step_eval_freq
                if (
                    self.step_eval_freq is not None
                    and (self.trainer_state.global_step + 1)
                    % self.step_eval_freq
                    == 0
                ):
                    self.eval()

                self.trainer_state.update_step()
                self.trainer_state.sync_pipeline_hook_arg(on_step_hook_args)

        with self.hooks.begin("on_loop", self._get_hook_args()):
            while not end_loop_flag:
                # TODO: Synchronize end_loop_flag when different
                # processes have different batch numbers. !
                #
                # In some cases, the dataloader may not have the same
                # number of batches when dataset is split into
                # different processes.
                #
                # If the dataloader has a different number of batches,
                # the training loop may hang or produce unexpected results.

                with self.hooks.begin(
                    "on_epoch", self._get_hook_args()
                ) as on_epoch_hook_args:
                    for _i, batch in enumerate(self.dataloader):
                        step(
                            batch=batch,
                            batch_processor=self.batch_processor,
                            optimizer=self.optimizer,
                            lr_scheduler=self.lr_scheduler,
                        )

                        if (
                            self.max_step is not None
                            and self.trainer_state.global_step >= self.max_step
                        ):
                            end_loop_flag = True
                            break

                    if (
                        self.lr_scheduler_step_at == "epoch"
                        and self.lr_scheduler is not None
                    ):
                        self.lr_scheduler.scheduler.step()

                    if (
                        self.epoch_eval_freq is not None
                        and (self.trainer_state.epoch + 1)
                        % self.epoch_eval_freq
                        == 0
                    ):
                        self.eval()

                    self.trainer_state.update_epoch()
                    self.trainer_state.sync_pipeline_hook_arg(
                        on_epoch_hook_args
                    )

                if (
                    self.max_epoch is not None
                    and self.trainer_state.epoch >= self.max_epoch
                ):
                    end_loop_flag = True

    def _clip_grad(self) -> None:
        """Clips gradients based on the specified mode and values.

        This method clips gradients either by value or by norm, depending
        on the configuration.
        """
        if self.grad_clip_mode is None:
            return
        assert self.optimizer is not None, "optimizer should not be None"
        params: List[torch.Tensor] = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group["params"])

        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params)
        )
        if len(params) > 0:
            if self.grad_clip_mode == "value":
                self.accelerator.clip_grad_value_(params, self.grad_clip_value)
            elif self.grad_clip_mode == "norm":
                self.accelerator.clip_grad_norm_(
                    params, self.grad_max_norm, self.grad_norm_type
                )

    @torch.no_grad()
    def eval(self) -> Optional[Any]:
        """Evaluates the model on the validation dataset.

        Returns:
            Optional[Any]: The evaluation metric, or None if evaluation
            is not performed.
        """
        assert self.val_dataloader is not None and self.metric is not None, (
            "val_dataloader and metric should not be None"
        )
        training = self.model.training
        self.model.eval()
        torch.cuda.empty_cache()
        if self.accelerator.is_main_process:
            logger.info("\n" + "=" * 50 + "BEGIN EVAL" + "=" * 50)
        for val_step_id, batch in enumerate(self.val_dataloader):
            model_outputs = self.model(batch)
            self.metric.update(batch, model_outputs)
            if (
                val_step_id + 1
            ) % 10 == 0 and self.accelerator.is_main_process:
                logger.info(f"eval: {val_step_id + 1}")
        self.accelerator.wait_for_everyone()
        if "accelerator" in signature(self.metric.compute).parameters:
            metric = self.metric.compute(accelerator=self.accelerator)
        else:
            metric = self.metric.compute()
        self.accelerator.wait_for_everyone()
        self.metric.reset()
        torch.cuda.empty_cache()
        self.model.train(training)
        return metric
