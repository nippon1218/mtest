import torch
import pytest
import allure
import numpy as np
from utils.device_utils import get_device_object, get_device_info

@allure.epic("PyTorch算子测试")
@allure.feature("交叉熵损失函数")
@allure.description("""
该测试模块验证PyTorch中交叉熵损失函数的功能正确性，包括：
1. 基本功能：验证损失计算和梯度反向传播
2. 数值稳定性：验证在不同输入范围下的稳定性
3. 边界情况：验证特殊样本和极端预测的处理

所有测试都在CPU和CUDA设备上执行，并验证结果的正确性。
""")
class TestCrossEntropy:
    @allure.story("基本功能测试")
    @allure.title("测试交叉熵基本功能 - {dtype}")
    @allure.description("""
    验证交叉熵损失函数的基本功能，测试要点：
    1. 支持不同的数据类型（float32、float64）
    2. 验证损失值的非负性
    3. 验证梯度计算的正确性
    4. 梯度不应包含NaN
    """)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_cross_entropy_basic(self, dtype, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用")
            
        device_obj = get_device_object(device)
        if device == "cuda":
            get_device_info()
            
        # 创建输入数据
        batch_size, num_classes = 32, 10
        logits = torch.randn(batch_size, num_classes, dtype=dtype, device=device_obj, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size,), device=device_obj)
        
        # 计算交叉熵损失
        loss = torch.nn.functional.cross_entropy(logits, targets)
        
        # 验证损失值是标量且为非负数
        assert loss.dim() == 0, "损失应该是标量"
        assert loss.item() >= 0, "交叉熵损失应该是非负数"
        
        # 验证梯度计算
        loss.backward()
        assert logits.grad is not None, "应该能够计算梯度"
        assert not torch.isnan(logits.grad).any(), "梯度不应包含NaN"
        
    @allure.story("数值稳定性测试")
    @allure.title("测试交叉熵数值稳定性 - {device}")
    @allure.description("""
    验证交叉熵在不同输入范围下的数值稳定性，测试要点：
    1. 小值输入（-1e3）的处理
    2. 正常值输入（0.0）的处理
    3. 大值输入（1e3）的处理
    4. 验证损失值和梯度不包含NaN或Inf
    """)
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_cross_entropy_numerical_stability(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用")
            
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        test_cases = [
            ("小值输入", -1e3),
            ("正常值输入", 0.0),
            ("大值输入", 1e3)
        ]
        
        for case_name, scale in test_cases:
            with allure.step(f"测试{case_name}"):
                logits = torch.full((2, 3), scale, dtype=dtype, device=device_obj, requires_grad=True)
                targets = torch.tensor([0, 1], device=device_obj)
                
                loss = torch.nn.functional.cross_entropy(logits, targets)
                loss.backward()
                
                assert not torch.isnan(loss), f"{case_name}的损失值不应为NaN"
                assert not torch.isinf(loss), f"{case_name}的损失值不应为Inf"
                assert not torch.isnan(logits.grad).any(), f"{case_name}的梯度不应包含NaN"
                assert not torch.isinf(logits.grad).any(), f"{case_name}的梯度不应包含Inf"
                
    @allure.story("边界情况测试")
    @allure.title("测试交叉熵边界情况 - {device}")
    @allure.description("""
    验证交叉熵在边界情况下的处理，测试要点：
    1. 单个样本的处理
    2. 完全正确的预测（损失应接近0）
    3. 完全错误的预测（损失应较大）
    4. 验证各种情况下的数值稳定性
    """)
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_cross_entropy_edge_cases(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用")
            
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        # 测试单个样本
        with allure.step("测试单个样本"):
            logits = torch.randn(1, 5, dtype=dtype, device=device_obj, requires_grad=True)
            targets = torch.tensor([2], device=device_obj)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            assert not torch.isnan(loss), "单个样本的损失值不应为NaN"
            
        # 测试完全正确的预测
        with allure.step("测试完全正确的预测"):
            logits = torch.zeros(3, 4, dtype=dtype, device=device_obj, requires_grad=True)
            targets = torch.tensor([0, 1, 2], device=device_obj)
            # 将正确类别的logit设置为很大的值
            logits_data = logits.data
            for i, t in enumerate(targets):
                logits_data[i, t] = 10.0
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            assert loss.item() < 1e-3, "完全正确预测的损失应该接近0"
            
        # 测试完全错误的预测
        with allure.step("测试完全错误的预测"):
            logits = torch.zeros(3, 4, dtype=dtype, device=device_obj, requires_grad=True)
            targets = torch.tensor([0, 1, 2], device=device_obj)
            # 将错误类别的logit设置为很大的值
            logits_data = logits.data
            for i, t in enumerate(targets):
                logits_data[i, (t + 1) % 4] = 10.0
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            assert loss.item() > 5.0, "完全错误预测的损失应该较大"
