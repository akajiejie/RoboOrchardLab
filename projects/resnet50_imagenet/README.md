# Introduction

Training and evaluation scripts for resnet50 on the ImageNet dataset.

# Usage

## Single GPU Training

```bash
python3 train.py --data-root /path/to/imagenet
```

## Multi GPU Training

```bash
accelerate launch \
    --multi-gpu \
    --num-processes 4 \
    --gpu-ids 0,1,2,3 \
    train.py --data-root /path/to/imagenet
```
