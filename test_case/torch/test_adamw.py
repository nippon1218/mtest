#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch
import allure
import numpy as np
import math
from .utils import get_device_object, test_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("AdamW优化器")
class TestAdamW:
    def adamw_step(self, param, grad, exp_avg, exp_avg_sq, step,
                  lr=5e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        """
        实现AdamW优化器的一步更新
        """
        # 衰减学习率
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        # 应用权重衰减
        param.data = param.data * (1 - lr * weight_decay)
        
        # 对梯度进行裁剪，防止数值过大
        grad_norm = grad.norm()
        if grad_norm > 1.0:
            grad = grad / grad_norm
        
        # 更新动量
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
        # 使用更安全的方式计算分母
        denom = (exp_avg_sq + eps * bias_correction2).sqrt()
        denom = denom / math.sqrt(bias_correction2)
        denom = torch.clamp(denom, min=1e-8)
        
        step_size = lr / bias_correction1
        
        # 更新参数，并裁剪更新幅度
        update = exp_avg / denom
        update = torch.clamp(update, min=-1.0, max=1.0)
        param.data.add_(update, alpha=-step_size)
        
        return param, exp_avg, exp_avg_sq

    @allure.story("基本功能测试")
    @allure.title("测试AdamW基本优化功能 - {dtype}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_adamw_basic(self, dtype, device):
        device_obj = get_device_object(device)
        
        # 创建简单的线性回归问题
        X = torch.randn(100, 10, dtype=dtype, device=device_obj)
        y = torch.sum(X * 2.0, dim=1, keepdim=True)  # 真实权重为2.0
        
        # 初始化参数
        w = torch.zeros(10, 1, dtype=dtype, device=device_obj, requires_grad=True)
        exp_avg = torch.zeros_like(w)
        exp_avg_sq = torch.zeros_like(w)
        
        # 训练100步
        for step in range(1, 101):
            # 前向传播
            pred = X @ w
            loss = ((pred - y) ** 2).mean()
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            with torch.no_grad():
                w, exp_avg, exp_avg_sq = self.adamw_step(
                    w, w.grad, exp_avg, exp_avg_sq, step
                )
                w.grad.zero_()
        
        # 验证参数接近真实值
        torch.testing.assert_close(
            w, torch.full_like(w, 2.0),
            rtol=0.1, atol=0.1
        )

    @allure.story("权重衰减测试")
    @allure.title("测试AdamW权重衰减效果")
    def test_adamw_weight_decay(self, device):
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        # 初始化大权重
        w = torch.ones(10, 1, dtype=dtype, device=device_obj, requires_grad=True) * 10.0
        exp_avg = torch.zeros_like(w)
        exp_avg_sq = torch.zeros_like(w)
        
        # 使用较大的权重衰减
        weight_decay = 0.1
        initial_norm = w.norm().item()
        
        # 只应用权重衰减，不使用梯度
        for step in range(1, 11):
            with torch.no_grad():
                w, exp_avg, exp_avg_sq = self.adamw_step(
                    w, torch.zeros_like(w), exp_avg, exp_avg_sq, step,
                    weight_decay=weight_decay
                )
        
        final_norm = w.norm().item()
        # 验证权重范数减小
        assert final_norm < initial_norm, "权重衰减应该减小权重范数"

    @allure.story("超参数测试")
    @allure.title("测试不同学习率和beta值")
    def test_adamw_hyperparams(self, device):
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        # 测试不同的超参数组合
        hyperparams = [
            {"lr": 1e-2, "beta1": 0.9, "beta2": 0.999},
            {"lr": 1e-4, "beta1": 0.8, "beta2": 0.99},
            {"lr": 1e-3, "beta1": 0.95, "beta2": 0.999}
        ]
        
        for params in hyperparams:
            with allure.step(f"测试学习率={params['lr']}, beta1={params['beta1']}, beta2={params['beta2']}"):
                # 创建简单的优化问题
                w = torch.ones(5, 1, dtype=dtype, device=device_obj, requires_grad=True)
                exp_avg = torch.zeros_like(w)
                exp_avg_sq = torch.zeros_like(w)
                
                # 运行几步优化
                for step in range(1, 6):
                    with torch.no_grad():
                        w, exp_avg, exp_avg_sq = self.adamw_step(
                            w, torch.ones_like(w), exp_avg, exp_avg_sq, step,
                            lr=params["lr"], beta1=params["beta1"], 
                            beta2=params["beta2"]
                        )
                
                # 验证更新是有限的
                assert not torch.isnan(w).any(), "参数包含NaN"
                assert not torch.isinf(w).any(), "参数包含Inf"

    @allure.story("数值稳定性测试")
    @allure.title("测试AdamW数值稳定性")
    def test_adamw_numerical_stability(self, device):
        if device != "cuda" or not torch.cuda.is_available():
            pytest.skip("数值稳定性测试需要CUDA设备")
            
        device_obj = get_device_object(device)
        dtype = torch.float16
        
        test_cases = [
            ("小梯度", 1e-3),  # 进一步增大最小梯度值
            ("正常梯度", 1.0),
            ("大梯度", 1e3)   # 进一步减小最大梯度值
        ]
        
        for case_name, scale in test_cases:
            with allure.step(f"测试{case_name}"):
                w = torch.ones(10, 1, dtype=dtype, device=device_obj, requires_grad=True)
                exp_avg = torch.zeros_like(w)
                exp_avg_sq = torch.zeros_like(w)
                
                grad = torch.randn_like(w) * scale
                
                # 运行优化器
                with torch.no_grad():
                    w, exp_avg, exp_avg_sq = self.adamw_step(
                        w, grad, exp_avg, exp_avg_sq, 1
                    )
                
                # 验证输出
                assert not torch.isnan(w).any(), f"{case_name}的输出中包含NaN"
                assert not torch.isinf(w).any(), f"{case_name}的输出中包含Inf"

    @allure.story("性能测试")
    @allure.title("测试AdamW性能")
    def test_adamw_performance(self, device):
        if device != "cuda" or not torch.cuda.is_available():
            pytest.skip("性能测试需要CUDA设备")
            
        device_obj = get_device_object(device)
        dtype = torch.float16
        
        # 使用较大的参数规模
        param_shape = (1024, 1024)
        w = torch.ones(param_shape, dtype=dtype, device=device_obj, requires_grad=True)
        exp_avg = torch.zeros_like(w)
        exp_avg_sq = torch.zeros_like(w)
        grad = torch.randn_like(w)
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                w, exp_avg, exp_avg_sq = self.adamw_step(
                    w, grad, exp_avg, exp_avg_sq, 1
                )
            
        torch.cuda.synchronize()
        
        # 测试100次迭代的平均时间
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for step in range(1, 101):
            with torch.no_grad():
                w, exp_avg, exp_avg_sq = self.adamw_step(
                    w, grad, exp_avg, exp_avg_sq, step
                )
        end_event.record()
        
        torch.cuda.synchronize()
        avg_time = start_event.elapsed_time(end_event) / 100
        
        # 记录性能指标
        allure.attach(
            f"平均执行时间: {avg_time:.3f} ms",
            name="性能指标",
            attachment_type=allure.attachment_type.TEXT
        )
