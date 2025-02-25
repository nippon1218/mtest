#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import numpy as np
from .utils import get_device_object, test_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("AdamW优化器")
@allure.description("""
该测试模块验证PyTorch中torch.optim.AdamW优化器的功能正确性，包括：
1. 基本功能：验证不同数据类型的优化过程
2. 权重衰减：验证权重衰减的效果
3. 超参数测试：验证不同学习率和beta值的影响
4. 数值稳定性：验证大小梯度的处理
5. 性能测试：验证大规模参数的优化

所有测试都在CPU和CUDA设备上执行，并验证结果的一致性。
""")
class TestAdamW:
    @allure.story("基本功能测试")
    @allure.title("测试AdamW基本优化功能 - {dtype}")
    @allure.description("""
    验证AdamW优化器的基本功能，测试要点：
    1. 支持多种数据类型（float32、float64）
    2. 验证优化收敛性
    3. 验证参数更新的准确性
    4. 比较CPU和CUDA的优化结果
    """)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_adamw_basic(self, dtype, device):
        # 在CPU上计算参考结果
        X_cpu = torch.randn(100, 10, dtype=dtype)
        y_cpu = torch.sum(X_cpu * 2.0, dim=1, keepdim=True)  # 真实权重为2.0
        w_cpu = torch.zeros(10, 1, dtype=dtype, requires_grad=True)
        optimizer_cpu = torch.optim.AdamW([w_cpu], lr=5e-2, weight_decay=0.01)

        # 在指定设备上计算
        device_obj = get_device_object(device)
        X = X_cpu.to(device_obj)
        y = y_cpu.to(device_obj)
        w = torch.zeros(10, 1, dtype=dtype, device=device_obj, requires_grad=True)
        optimizer = torch.optim.AdamW([w], lr=5e-2, weight_decay=0.01)

        # 训练100步
        for step in range(1, 101):
            # CPU上的计算
            pred_cpu = X_cpu @ w_cpu
            loss_cpu = ((pred_cpu - y_cpu) ** 2).mean()
            optimizer_cpu.zero_grad()
            loss_cpu.backward()
            optimizer_cpu.step()

            # 指定设备上的计算
            pred = X @ w
            loss = ((pred - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证参数接近真实值
        torch.testing.assert_close(
            w, torch.full_like(w, 2.0),
            rtol=0.1, atol=0.1
        )

        # 比较CPU和当前设备的结果
        if device == "cuda":
            torch.testing.assert_close(w.cpu(), w_cpu, rtol=1e-4, atol=1e-4)

    @allure.story("权重衰减测试")
    @allure.title("测试AdamW权重衰减效果")
    @allure.description("""
    验证AdamW的权重衰减效果，测试要点：
    1. 验证权重衰减的正确性
    2. 验证权重范数的减小
    3. 验证不同权重衰减率的影响
    4. 比较CPU和CUDA的计算结果
    """)
    def test_adamw_weight_decay(self, device):
        # 在CPU上计算参考结果
        w_cpu = torch.ones(10, 1, dtype=torch.float32) * 10.0
        w_cpu.requires_grad_(True)
        optimizer_cpu = torch.optim.AdamW([w_cpu], lr=0.1, weight_decay=0.1)
        initial_norm_cpu = w_cpu.norm().item()

        # 在指定设备上计算
        device_obj = get_device_object(device)
        w = torch.ones(10, 1, dtype=torch.float32, device=device_obj) * 10.0
        w.requires_grad_(True)
        optimizer = torch.optim.AdamW([w], lr=0.1, weight_decay=0.1)
        initial_norm = w.norm().item()

        # 运行10步优化，只应用权重衰减
        for step in range(10):
            # CPU上的计算
            optimizer_cpu.zero_grad()
            w_cpu.grad = torch.zeros_like(w_cpu)  # 使用零梯度
            optimizer_cpu.step()

            # 指定设备上的计算
            optimizer.zero_grad()
            w.grad = torch.zeros_like(w)  # 使用零梯度
            optimizer.step()

        # 验证权重范数减小
        final_norm_cpu = w_cpu.norm().item()
        final_norm = w.norm().item()
        
        assert final_norm < initial_norm, "权重衰减应该减小权重范数"
        assert final_norm_cpu < initial_norm_cpu, "CPU上的权重衰减应该减小权重范数"

        # 比较CPU和当前设备的结果
        if device == "cuda":
            torch.testing.assert_close(w.cpu(), w_cpu, rtol=1e-4, atol=1e-4)

    @allure.story("超参数测试")
    @allure.title("测试不同学习率和beta值")
    @allure.description("""
    验证AdamW在不同超参数下的表现，测试要点：
    1. 验证不同学习率的影响
    2. 验证不同beta值的影响
    3. 验证数值稳定性
    4. 比较CPU和CUDA的计算结果
    """)
    def test_adamw_hyperparams(self, device):
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        # 测试不同的超参数组合
        hyperparams = [
            {"lr": 1e-2, "betas": (0.9, 0.999)},
            {"lr": 1e-4, "betas": (0.8, 0.99)},
            {"lr": 1e-3, "betas": (0.95, 0.999)}
        ]
        
        for params in hyperparams:
            with allure.step(f"测试学习率={params['lr']}, betas={params['betas']}"):
                # 在CPU上计算参考结果
                w_cpu = torch.ones(5, 1, dtype=dtype, requires_grad=True)
                optimizer_cpu = torch.optim.AdamW([w_cpu], **params)

                # 在指定设备上计算
                w = torch.ones(5, 1, dtype=dtype, device=device_obj, requires_grad=True)
                optimizer = torch.optim.AdamW([w], **params)

                # 运行几步优化
                for step in range(5):
                    # CPU上的计算
                    optimizer_cpu.zero_grad()
                    w_cpu.grad = torch.ones_like(w_cpu)
                    optimizer_cpu.step()

                    # 指定设备上的计算
                    optimizer.zero_grad()
                    w.grad = torch.ones_like(w)
                    optimizer.step()

                # 验证更新是有限的
                assert not torch.isnan(w).any(), "参数包含NaN"
                assert not torch.isinf(w).any(), "参数包含Inf"

                # 比较CPU和当前设备的结果
                if device == "cuda":
                    torch.testing.assert_close(w.cpu(), w_cpu, rtol=1e-4, atol=1e-4)

    @allure.story("数值稳定性测试")
    @allure.title("测试AdamW数值稳定性")
    @allure.description("""
    验证AdamW在不同梯度大小下的稳定性，测试要点：
    1. 处理小梯度（1e-3）
    2. 处理正常梯度（1.0）
    3. 处理大梯度（1e3）
    4. 验证没有NaN和Inf
    5. 比较CPU和CUDA的计算结果
    """)
    def test_adamw_numerical_stability(self, device):
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        test_cases = [
            ("小梯度", 1e-3),
            ("正常梯度", 1.0),
            ("大梯度", 1e3)
        ]
        
        for case_name, scale in test_cases:
            with allure.step(f"测试{case_name}"):
                # 在CPU上生成所有数据
                w_cpu = torch.ones(10, 1, dtype=dtype, requires_grad=True)
                optimizer_cpu = torch.optim.AdamW([w_cpu], lr=0.01)
                grad_cpu = torch.randn(10, 1, dtype=dtype) * scale

                # 在指定设备上计算
                w = w_cpu.detach().clone().to(device_obj).requires_grad_(True)
                optimizer = torch.optim.AdamW([w], lr=0.01)
                grad = grad_cpu.to(device_obj)

                # CPU上的计算
                optimizer_cpu.zero_grad()
                w_cpu.grad = grad_cpu
                optimizer_cpu.step()

                # 指定设备上的计算
                optimizer.zero_grad()
                w.grad = grad
                optimizer.step()

                # 验证输出
                assert not torch.isnan(w).any(), f"{case_name}的输出中包含NaN"
                assert not torch.isinf(w).any(), f"{case_name}的输出中包含Inf"

                # 比较CPU和当前设备的结果
                if device == "cuda":
                    torch.testing.assert_close(w.cpu(), w_cpu, rtol=1e-4, atol=1e-4)

    @allure.story("性能测试")
    @allure.title("测试大规模AdamW")
    @allure.description("""
    验证AdamW在大规模参数下的表现，测试要点：
    1. 处理大规模参数（1024x1024）
    2. 验证计算的准确性
    3. 验证数值稳定性
    4. 比较CPU和CUDA的计算结果
    """)
    def test_adamw_performance(self, device):
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        # 使用较大的参数规模
        param_shape = (1024, 1024)

        # 在CPU上生成所有数据
        w_cpu = torch.ones(param_shape, dtype=dtype, requires_grad=True)
        optimizer_cpu = torch.optim.AdamW([w_cpu], lr=0.01)
        grad_cpu = torch.randn(param_shape, dtype=dtype)

        # 在指定设备上计算
        w = w_cpu.detach().clone().to(device_obj).requires_grad_(True)
        optimizer = torch.optim.AdamW([w], lr=0.01)
        grad = grad_cpu.to(device_obj)

        # CPU上的计算
        optimizer_cpu.zero_grad()
        w_cpu.grad = grad_cpu
        optimizer_cpu.step()

        # 指定设备上的计算
        optimizer.zero_grad()
        w.grad = grad
        optimizer.step()
        # 验证输出
        assert not torch.isnan(w).any(), "参数包含NaN"
        assert not torch.isinf(w).any(), "参数包含Inf"

        # 比较CPU和当前设备的结果
        if device == "cuda":
            torch.testing.assert_close(w.cpu(), w_cpu, rtol=1e-4, atol=1e-4)

            # 测量CUDA性能
            # 预热
            for _ in range(10):
                optimizer.zero_grad()
                w.grad = grad
                optimizer.step()
            
            torch.cuda.synchronize()
            
            # 测试100次迭代的平均时间
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(100):
                optimizer.zero_grad()
                w.grad = grad
                optimizer.step()
            end_event.record()
            
            torch.cuda.synchronize()
            avg_time = start_event.elapsed_time(end_event) / 100
            
            # 记录性能指标
            allure.attach(
                f"平均执行时间: {avg_time:.3f} ms",
                name="性能指标",
                attachment_type=allure.attachment_type.TEXT
            )


