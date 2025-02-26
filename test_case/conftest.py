#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os
import platform
from datetime import datetime
import allure

# 检查torch导入状态
torch_import_failed = False
try:
    import torch
except ImportError:
    torch_import_failed = True
    torch = None

# 尝试导入torch_abc
torch_abc = None
try:
    import torch_abc
except ImportError:
    pass

@pytest.fixture(scope="session", autouse=True)
def env_info(request):
    """
    测试环境信息fixture，自动运行
    """
    # 如果torch导入失败，则直接跳过
    if torch_import_failed:
        pytest.skip("PyTorch导入失败，跳过所有测试")
        
    device = request.config.getoption("--device")
    device_info = f"CPU"
    if device == "cuda" and torch.cuda.is_available():
        device_info = f"CUDA - {torch.cuda.get_device_name()}"
    elif device == "abc" and torch_abc is not None:
        device_info = f"ABC - {torch_abc.__version__}"
    
    allure.attach(
        f"""
        测试环境信息:
        操作系统: {platform.system()} {platform.release()}
        Python版本: {platform.python_version()}
        PyTorch版本: {torch.__version__ if torch else '导入失败'}
        测试设备: {device_info}
        测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """,
        "测试环境信息",
        allure.attachment_type.TEXT
    )

@pytest.fixture(autouse=True)
def test_timer():
    """
    测试用例执行时间统计
    """
    start_time = datetime.now()
    yield
    end_time = datetime.now()
    duration = end_time - start_time
    allure.attach(
        f"测试用例执行时间: {duration.total_seconds():.2f} 秒",
        "执行时间",
        allure.attachment_type.TEXT
    )

def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=["cpu", "cuda", "abc"],
        help="选择测试设备：cpu、cuda或abc"
    )

@pytest.fixture(scope="session")
def device(request):
    """
    提供测试设备参数（cpu/cuda/abc）
    """
    # 如果torch导入失败，则直接跳过
    if torch_import_failed:
        pytest.skip("PyTorch导入失败，跳过所有测试")
        
    dev = request.config.getoption("--device")
    
    if dev == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用")
        print(f"\nCUDA版本: {torch.version.cuda}")
        print(f"CUDA设备信息: {torch.cuda.get_device_name()}")
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        return "cuda"
    elif dev == "abc":
        if torch_abc is None:
            pytest.skip("ABC设备不可用（torch_abc导入失败）")
        print(f"\nABC版本: {torch_abc.__version__}")
        return "abc"
    
    return "cpu"

def pytest_runtest_setup(item):
    """
    测试用例开始前的hook
    """
    # 如果torch导入失败，则直接跳过
    if torch_import_failed:
        pytest.skip("PyTorch导入失败，跳过所有测试")
        
    allure.attach(
        f"开始执行测试用例: {item.name}",
        "测试开始",
        allure.attachment_type.TEXT
    )

def pytest_runtest_teardown(item):
    """
    测试用例结束后的hook
    """
    allure.attach(
        f"测试用例执行完成: {item.name}",
        "测试结束",
        allure.attachment_type.TEXT
    )

def pytest_collection_modifyitems(items):
    """
    测试用例收集完成时，将收集到的item的name和nodeid的中文显示在控制台上
    如果torch导入失败，则标记所有测试用例为skip
    """
    # 如果torch导入失败，则跳过所有测试
    if torch_import_failed:
        skip_marker = pytest.mark.skip(reason="PyTorch导入失败，跳过所有测试")
        for item in items:
            item.add_marker(skip_marker)
    
    for item in items:
        item.name = item.name.encode("utf-8").decode("unicode_escape")
        item._nodeid = item.nodeid.encode("utf-8").decode("unicode_escape")

@pytest.mark.slow
@pytest.mark.integration
def test_database_integration():
    pass
