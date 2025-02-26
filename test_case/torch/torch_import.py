#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch导入辅助模块
在缺少PyTorch时提供优雅的错误处理
"""

import sys
import pytest

# 检查torch导入状态
torch_import_failed = False
try:
    import torch
except ImportError:
    torch_import_failed = True
    torch = None
    # 在模块导入阶段注册跳过标记
    pytest.skip("PyTorch导入失败，跳过所有测试", allow_module_level=True)

# 导出变量
__all__ = ['torch', 'torch_import_failed']
