#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 使用我们的导入辅助模块替代直接导入
from .torch_import import torch, torch_import_failed

def get_device_object(device_str):
    """获取torch.device对象"""
    if torch_import_failed:
        return None
    if device_str == "cuda":
        return torch.device("cuda:0")
    return torch.device("cpu")

test_dtypes = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64
]

em_test_dtypes = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64
]

em_input_dtypes = [
    torch.int32,
    torch.int64
]
