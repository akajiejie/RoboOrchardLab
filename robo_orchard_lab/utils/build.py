import copy
import importlib

import torch


def build(cfg, *args):
    if isinstance(cfg, dict):
        cfg = copy.deepcopy(cfg)
        cls = cfg.pop("type")
        if isinstance(cls, str):
            if ":" in cls:
                module_name, cls_name = cls.split(":")
                module = importlib.import_module(module_name)
                cls = getattr(module, cls_name)
            else:
                cls = importlib.import_module(cls)
        if cls == torch.nn.GroupNorm and len(args) == 1:
            return cls(num_channels=args[0], **cfg)  # type: ignore
        return cls(*args, **cfg)  # type: ignore
    return cfg
