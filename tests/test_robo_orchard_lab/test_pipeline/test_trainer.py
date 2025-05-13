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

from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Metric

from robo_orchard_lab.pipeline import SimpleTrainer
from robo_orchard_lab.pipeline.batch_processor import SimpleBatchProcessor


class DummyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.value = 0

    def update(self, preds, targets):
        self.value += 1

    def compute(self):
        return self.value


class SimpleModel(torch.nn.Module):
    """A simple neural network model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class DummyBatchProcessor(SimpleBatchProcessor):
    """A simple batch processor for testing."""

    def do_forward(self, model, batch, device):
        outputs = model(batch.to(device))
        loss = torch.mean((outputs - 1) ** 2)  # Mean squared error loss
        return outputs, loss


@pytest.fixture(scope="session")
def dummy_trainer():
    """Fixture to create a dummy trainer with a real model and optimizer."""
    model = SimpleModel()
    dataloader = [
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        torch.tensor([[0.1, 0.2]], dtype=torch.float32),
    ]
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator()
    batch_processor = DummyBatchProcessor()

    trainer = SimpleTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=batch_processor,
        max_epoch=1,
    )
    return trainer


def test_trainer_initialization(dummy_trainer):
    """Test trainer initialization."""
    assert dummy_trainer.max_epoch == 1
    assert dummy_trainer.lr_scheduler_step_at == "step"


def test_training_loop(dummy_trainer):
    """Test training loop execution."""
    # Spy on hook methods
    dummy_trainer.on_loop_begin = MagicMock()
    dummy_trainer.on_epoch_begin = MagicMock()
    dummy_trainer.on_step_begin = MagicMock()
    dummy_trainer.on_step_end = MagicMock()
    dummy_trainer.on_epoch_end = MagicMock()
    dummy_trainer.on_loop_end = MagicMock()

    # Run the training loop
    dummy_trainer()

    # Verify that hooks were called the expected number of times
    assert dummy_trainer.on_loop_begin.call_count == 1
    assert dummy_trainer.on_epoch_begin.call_count == 1
    assert dummy_trainer.on_step_begin.call_count == len(
        dummy_trainer.dataloader
    )
    assert dummy_trainer.on_step_end.call_count == len(
        dummy_trainer.dataloader
    )
    assert dummy_trainer.on_epoch_end.call_count == 1
    assert dummy_trainer.on_loop_end.call_count == 1


def test_optimizer_and_scheduler(dummy_trainer):
    """Test optimizer and scheduler behavior."""
    initial_lr = dummy_trainer.optimizer.param_groups[0]["lr"]
    dummy_trainer()
    final_lr = dummy_trainer.optimizer.param_groups[0]["lr"]

    # Check that the learning rate scheduler updated the learning rate
    assert final_lr < initial_lr


def test_metric_computation(dummy_trainer):
    """Test metrics update and computation."""
    dummy_trainer()
    if dummy_trainer.metric is not None:
        dummy_trainer.metric.compute()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
