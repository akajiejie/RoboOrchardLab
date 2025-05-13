# RoboOrchard Lab

TBC

## Features

TBC

## Getting started

### Requirements

* Python 3.10

### Installation

#### Install torch

* torch >= 2.4.0
* torchvision >= 0.19.0

This project requires PyTorch version 2.4.0 or higher. Since the specific PyTorch build (CPU, CUDA 11.x, CUDA 12.x, etc.) depends on your system, please install it manually first by following the official instructions on the [PyTorch website](https://pytorch.org/get-started/locally/)

#### From pip

```bash
pip install robo_orchard_lab
```

#### From source

```bash
make install
```

## Contribution Guide

### Install by editable mode

```bash
make install-editable
```

### Install development requirements

```bash
make dev-env
```

### Lint

```bash
make check-lint
```

### Auto format

```bash
make auto-format
```

### Run test

```bash
make test
```

## Acknowledgement

Our work is inspired by many existing deep learning algorithm frameworks, such as [OpenMM Lab](https://github.com/openmm), [Hugging face transformers](https://github.com/huggingface/transformers) [Hugging face diffusers](https://github.com/huggingface/diffusers) etc. We would like to thank all the contributors of all the open-source projects that we use in RoboOrchard.
