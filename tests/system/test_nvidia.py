#!/usr/bin/env python

import subprocess
import pytest
import yaml
import re
import os

def load_config():
    """加载系统要求配置"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              'config', 'system_requirements.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_nvidia_info():
    """获取NVIDIA系统信息"""
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    
    info = {}
    
    # 获取驱动版本
    driver_match = re.search(r'Driver Version: (\d+\.\d+)', result.stdout)
    info['driver_version'] = float(driver_match.group(1)) if driver_match else None
    
    # 获取CUDA版本
    cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
    info['cuda_version'] = float(cuda_match.group(1)) if cuda_match else None
    
    # 获取GPU数量
    gpu_count = len(re.findall(r'\d+\s+\w+\s+\w+\s+\w+', result.stdout))
    info['gpu_count'] = gpu_count if gpu_count > 0 else None
    
    # 获取GPU显存信息
    memory_match = re.search(r'(\d+)MiB\s*/\s*(\d+)MiB', result.stdout)
    if memory_match:
        total_memory_mb = int(memory_match.group(2))
        info['gpu_memory_gb'] = total_memory_mb / 1024
    else:
        info['gpu_memory_gb'] = None
    
    # 获取GPU功率限制
    # 使用nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits
    power_result = subprocess.run(
        ['nvidia-smi', '--query-gpu=power.limit', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    if power_result.returncode == 0:
        try:
            info['gpu_power_limit_w'] = float(power_result.stdout.strip())
        except (ValueError, TypeError):
            info['gpu_power_limit_w'] = None
    else:
        info['gpu_power_limit_w'] = None
    
    return info

def test_nvidia_driver_installation():
    """测试NVIDIA环境配置是否满足要求"""
    try:
        # 获取配置要求
        config = load_config()
        nvidia_config = config['nvidia']
        
        # 获取当前系统信息
        nvidia_info = get_nvidia_info()
        assert nvidia_info is not None, "Failed to get NVIDIA information"
        
        # 检查驱动版本
        assert nvidia_info['driver_version'] is not None, "Failed to get driver version"
        assert nvidia_info['driver_version'] >= nvidia_config['min_driver_version'], (
            f"NVIDIA driver version {nvidia_info['driver_version']} is lower than "
            f"required minimum version {nvidia_config['min_driver_version']}"
        )
        
        # 检查CUDA版本
        assert nvidia_info['cuda_version'] is not None, "Failed to get CUDA version"
        assert nvidia_info['cuda_version'] >= nvidia_config['min_cuda_version'], (
            f"CUDA version {nvidia_info['cuda_version']} is lower than "
            f"required minimum version {nvidia_config['min_cuda_version']}"
        )
        
        # 检查GPU数量
        assert nvidia_info['gpu_count'] is not None, "Failed to get GPU count"
        assert nvidia_info['gpu_count'] >= nvidia_config['min_gpu_count'], (
            f"GPU count {nvidia_info['gpu_count']} is lower than "
            f"required minimum count {nvidia_config['min_gpu_count']}"
        )
        
        # 检查GPU显存
        assert nvidia_info['gpu_memory_gb'] is not None, "Failed to get GPU memory information"
        assert nvidia_info['gpu_memory_gb'] >= nvidia_config['min_gpu_memory_gb'], (
            f"GPU memory {nvidia_info['gpu_memory_gb']:.1f}GB is lower than "
            f"required minimum {nvidia_config['min_gpu_memory_gb']}GB"
        )
        
        # 检查GPU功率限制
        if nvidia_info['gpu_power_limit_w'] is not None:
            assert nvidia_config['min_gpu_power_limit_w'] <= nvidia_info['gpu_power_limit_w'] <= nvidia_config['max_gpu_power_limit_w'], (
                f"GPU power limit {nvidia_info['gpu_power_limit_w']}W is outside the required range "
                f"[{nvidia_config['min_gpu_power_limit_w']}W, {nvidia_config['max_gpu_power_limit_w']}W]"
            )
        
    except FileNotFoundError:
        pytest.skip("nvidia-smi command not found - NVIDIA driver might not be installed")
