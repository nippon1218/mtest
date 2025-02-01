#!/usr/bin/env python

import distro
import pytest
import platform
import os

def test_is_ubuntu():
    """测试当前系统是否为Ubuntu"""
    distro_name = distro.linux_distribution()[0]
    assert 'Ubuntu' in distro_name, f"Expected 'Ubuntu', but got {distro_name}"

def test_python_version():
    """测试Python版本是否符合要求"""
    version = platform.python_version_tuple()
    assert int(version[0]) >= 3, "Python major version should be 3 or higher"
    assert int(version[1]) >= 6, "Python minor version should be 6 or higher"

def test_system_environment():
    """测试系统环境变量是否正确设置"""
    # 检查PATH环境变量
    assert 'PATH' in os.environ, "PATH environment variable is not set"
    
    # 检查HOME环境变量
    assert 'HOME' in os.environ, "HOME environment variable is not set"
    assert os.path.exists(os.environ['HOME']), "HOME directory does not exist"

def test_system_architecture():
    """测试系统架构信息"""
    arch = platform.machine()
    assert arch in ['x86_64', 'amd64', 'arm64'], f"Unsupported architecture: {arch}"
