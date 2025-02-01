#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os
import platform
from datetime import datetime
import allure
from utils import yaml_data

@pytest.fixture(scope="session", autouse=True)
def env_info(request):
    """
    测试环境信息fixture，自动运行
    """
    allure.attach(
        f"""
        测试环境信息:
        操作系统: {platform.system()} {platform.release()}
        Python版本: {platform.python_version()}
        测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """,
        "测试环境信息",
        allure.attachment_type.TEXT
    )

@pytest.fixture
def yaml_base_path():
    """
    返回yaml文件的基础路径
    """
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

@pytest.fixture
def load_test_data():
    """
    读取测试数据的fixture
    """
    def _load_test_data(yaml_file_name):
        """
        读取yaml文件数据
        :param yaml_file_name: yaml文件名
        :return: 数据
        """
        data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", yaml_file_name)
        return yaml_data.read_yaml(data_file)
    
    return _load_test_data

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

def pytest_runtest_setup(item):
    """
    测试用例开始前的hook
    """
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
    """
    for item in items:
        item.name = item.name.encode("utf-8").decode("unicode_escape")
        item._nodeid = item.nodeid.encode("utf-8").decode("unicode_escape")
