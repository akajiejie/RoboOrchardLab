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

import argparse
import logging
import os
from multiprocessing import set_start_method
from typing import Any, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy as AccuracyMetric
from torchvision import datasets, models, transforms

from robo_orchard_lab.pipeline import SimpleTrainer
from robo_orchard_lab.pipeline.batch_processor import SimpleBatchProcessor
from robo_orchard_lab.pipeline.hooks import (
    DoCheckpoint,
    MetricEntry,
    MetricTracker,
    StatsMonitor,
)
from robo_orchard_lab.utils import log_basic_config
from robo_orchard_lab.utils.huggingface import (
    get_accelerate_project_last_checkpoint_id,
)

logger = logging.getLogger(__file__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-test",
        action="store_true",
        default=False,
        help="Whether or not use dummy data for fast pipeline test",
    )
    parser.add_argument(
        "--data-root", type=str, required=True, help="Image dataset directory"
    )
    args = parser.parse_args()
    return args


def get_dataset(
    is_pipeline_test: bool, data_root: str
) -> Tuple[Dataset, Dataset]:
    if is_pipeline_test:
        train_dataset = datasets.FakeData(
            1281167, (3, 224, 224), 1000, transforms.ToTensor()
        )
        val_dataset = datasets.FakeData(
            50000, (3, 224, 224), 1000, transforms.ToTensor()
        )
    else:
        train_dataset = datasets.ImageFolder(
            os.path.join(data_root, "train"),
            transform=transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )

        val_dataset = datasets.ImageFolder(
            os.path.join(data_root, "val"),
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )

    return train_dataset, val_dataset


class MyBatchProcessor(SimpleBatchProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()

    def do_forward(
        self,
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        images, target = batch

        output = model(images)

        if self.need_backward:
            loss = self.criterion(output, target)
        else:
            loss = None

        return output, loss


class MyMetricTracker(MetricTracker):
    def update_metric(self, batch: Any, model_outputs: Any):
        _, targets = batch
        for metric_i in self.metrics:
            metric_i(model_outputs, targets)


def main(args):
    train_dataset, val_dataset = get_dataset(
        is_pipeline_test=args.pipeline_test, data_root=args.data_root
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    _ = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    model = models.resnet50()

    optimizer = SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    workspace_root = "./workspace/"
    last_ckpt_iteration = get_accelerate_project_last_checkpoint_id(
        "./workspace"
    )

    accelerator = Accelerator(
        project_config=ProjectConfiguration(
            project_dir=workspace_root,
            logging_dir=os.path.join(workspace_root, "logs"),
            automatic_checkpoint_naming=True,
            total_limit=32,
            iteration=last_ckpt_iteration + 1,
        )
    )

    trainer = SimpleTrainer(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=MyBatchProcessor(
            need_backward=True,
        ),
        max_epoch=90,
        hooks=[
            MyMetricTracker(
                metric_entrys=[
                    MetricEntry(
                        names=["top1_acc"],
                        metric=AccuracyMetric(
                            task="multiclass",
                            num_classes=1000,
                            top_k=1,
                        ),
                    ),
                    MetricEntry(
                        names=["top5_acc"],
                        metric=AccuracyMetric(
                            task="multiclass",
                            num_classes=1000,
                            top_k=5,
                        ),
                    ),
                ],
                step_log_freq=64,
                log_main_process_only=False,
            ),
            StatsMonitor(
                step_log_freq=64,
            ),
            DoCheckpoint(
                save_step_freq=1024,
            ),
        ],
    )

    if accelerator.is_main_process:
        logger.info("\n" + "=" * 50 + "BEGIN TRAINING" + "=" * 50)

    trainer()


if __name__ == "__main__":
    log_basic_config(level=logging.INFO)
    set_start_method("spawn", force=True)

    args = parse_args()
    main(args)
