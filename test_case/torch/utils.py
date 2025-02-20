#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def get_device_object(device_str):
    """获取torch.device对象"""
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
