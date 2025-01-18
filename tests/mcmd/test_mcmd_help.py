#!/usr/bin/env python

import subprocess
import pytest
from typing import List, Tuple

def run_mcmd_command(args: List[str]) -> Tuple[str, str, int]:
    """
    运行mcmd命令并返回结果
    
    Args:
        args: mcmd命令的参数列表
        
    Returns:
        Tuple[str, str, int]: 包含标准输出、标准错误和返回码
    """
    try:
        result = subprocess.run(['mcmd'] + args, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        pytest.fail("The 'mcmd' command is not found on this system.")
        return "", "", 1

def test_mcmd_help():
    """测试mcmd帮助命令是否正常工作"""
    stdout, stderr, returncode = run_mcmd_command(['-h'])
    
    assert returncode == 0, f"Help command failed with return code {returncode}"
    assert "Usage" in stdout, f"Expected 'Usage' in output, but got: {stdout}"
    assert stderr == "", f"Expected empty stderr, but got: {stderr}"

def test_mcmd_version():
    """测试mcmd版本命令是否正常工作"""
    stdout, stderr, returncode = run_mcmd_command(['--version'])
    
    assert returncode == 0, f"Version command failed with return code {returncode}"
    assert stdout.strip(), "Version output should not be empty"
    assert stderr == "", f"Expected empty stderr, but got: {stderr}"

def test_mcmd_invalid_option():
    """测试mcmd无效选项的错误处理"""
    stdout, stderr, returncode = run_mcmd_command(['--invalid-option'])
    
    assert returncode != 0, "Command should fail with invalid option"
    assert stderr != "", "Error message should be present for invalid option"
