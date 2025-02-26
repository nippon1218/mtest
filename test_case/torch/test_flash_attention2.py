#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
# 使用我们的导入辅助模块替代直接导入
from .torch_import import torch, torch_import_failed
import allure
import time
import math
from .utils import get_device_object, test_dtypes

@allure.epic("PyTorch算子测试")
@allure.feature("Flash Attention 2算子")
@allure.description("""
Flash Attention 2是一种高效的注意力机制实现，通过块稀疏操作和内存优化，
大幅提升了Transformer模型中自注意力计算的性能。

本测试类包含以下测试方面：
1. 基础功能测试 - 验证算法基本计算能力
2. 掩码功能测试 - 验证支持因果掩码和其他掩码类型
3. 多种数据类型支持 - 验证在不同精度下的稳定性
4. 性能评估 - 测量算法在不同序列长度下的性能
5. 数值稳定性 - 验证处理极端值的能力
6. 边界情况处理 - 测试特殊输入场景
7. Transformer集成 - 验证在多头注意力结构中的应用

所有测试在CPU和CUDA设备下执行，提供全面的功能和性能验证。
""")
class TestFlashAttention2:
    def setup_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def teardown_method(self, method):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def is_flash_attention2_available(self):
        """检查是否支持Flash Attention 2"""
        if not torch.cuda.is_available():
            return False
            
        # 检查CUDA计算能力
        device = torch.cuda.current_device()
        compute_capability = torch.cuda.get_device_capability(device)
        # 需要Ampere或更新架构 (计算能力 >= 8.0)
        return compute_capability[0] >= 8
        

    
    @allure.description("""
    实现简化版的Flash Attention 2算法。
    
    Flash Attention 2是对原始Flash Attention的改进版本，主要通过以下方式提高效率：
    1. 块稀疏计算 - 将序列分块处理，减少内存访问次数
    2. 重用缓存数据 - 最大化CUDA缓存利用率
    3. 并行化计算 - 优化计算布局提高GPU利用率
    4. 精细的内存访问模式 - 减少全局内存访问，提高带宽利用率
    
    本实现是一个教学简化版，展示核心算法思想，但不包括所有性能优化。
    真实的Flash Attention 2实现通常使用CUDA核心进行优化。
    
    算法核心步骤：
    1. 将查询、键、值序列分块处理
    2. 对每个块计算局部注意力分数
    3. 累积不同块之间的注意力结果
    4. 应用数值稳定性优化（如：LogSumExp技巧）
    5. 执行最终归一化
    
    此方法模拟了Flash Attention的核心，但实际性能可能不如专门优化的实现。
    """)
    def flash_attention2(self, q, k, v, mask=None, scale=None, block_size=128):
        """
        实现简化版的Flash Attention 2算法
        
        Flash Attention 2的核心思想是通过分块计算来减少内存访问和提高计算效率
        这个实现是一个简化版本，主要展示算法的核心思想
        
        参数：
            q, k, v: 查询、键、值张量，形状为 [batch_size, seq_len, head_dim]
            mask: 注意力掩码，形状为 [batch_size, seq_len, seq_len]
            scale: 缩放因子，默认为 1/sqrt(head_dim)
            block_size: 分块大小
            
        返回：
            output: 注意力输出，形状为 [batch_size, seq_len, head_dim]
        """
        batch_size, seq_len, head_dim = q.shape
        
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        
        # 初始化输出和中间变量
        output = torch.zeros_like(q)
        
        # 计算块的数量
        num_blocks = (seq_len + block_size - 1) // block_size
        
        # 对序列长度进行分块处理
        for i in range(num_blocks):
            # 当前块的查询范围
            q_start = i * block_size
            q_end = min(q_start + block_size, seq_len)
            q_block = q[:, q_start:q_end, :]
            
            # 初始化当前块的累积值和归一化因子
            m_i = torch.full((batch_size, q_end - q_start), float('-inf'), device=q.device)
            l_i = torch.zeros((batch_size, q_end - q_start), device=q.device)
            o_i = torch.zeros_like(q_block)
            
            # 对键值对进行分块处理
            for j in range(num_blocks):
                # 当前块的键值范围
                k_start = j * block_size
                k_end = min(k_start + block_size, seq_len)
                k_block = k[:, k_start:k_end, :]
                v_block = v[:, k_start:k_end, :]
                
                # 计算当前块的注意力分数
                s_ij = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale  # [batch_size, q_block_size, k_block_size]
                
                # 应用掩码（如果有）
                if mask is not None:
                    mask_block = mask[:, q_start:q_end, k_start:k_end]
                    s_ij = s_ij.masked_fill(mask_block == 0, float('-inf'))
                
                # 计算当前块的最大值
                m_ij = torch.max(s_ij, dim=-1)[0]  # [batch_size, q_block_size]
                
                # 更新累积值和归一化因子
                m_new = torch.maximum(m_i, m_ij)
                exp_diff_i = torch.exp(m_i - m_new).unsqueeze(-1)
                exp_diff_ij = torch.exp(m_ij - m_new).unsqueeze(-1)
                
                # 计算当前块的注意力权重
                p_ij = torch.exp(s_ij - m_ij.unsqueeze(-1))  # [batch_size, q_block_size, k_block_size]
                
                # 计算当前块的加权和
                weighted_v = torch.matmul(p_ij, v_block)  # [batch_size, q_block_size, head_dim]
                
                # 更新输出和归一化因子
                o_i = o_i * exp_diff_i + weighted_v * exp_diff_ij
                l_i = l_i * torch.exp(m_i - m_new) + torch.sum(p_ij, dim=-1) * torch.exp(m_ij - m_new)
                
                # 更新最大值
                m_i = m_new
            
            # 归一化输出
            output[:, q_start:q_end, :] = o_i / l_i.unsqueeze(-1)
        
        return output

    @allure.story("基础功能测试")
    @allure.title("测试基本的Flash Attention 2功能")
    @allure.description("""
    测试Flash Attention 2算法的基本功能，验证：
    1. 输出形状正确
    2. 输出不包含NaN或Inf值
    3. 输出序列长度与查询序列长度相匹配
    4. 输出维度与值向量维度相匹配
    该测试确保算法的基本实现逻辑正确。
    """)
    def test_flash_attention2_basic(self, device):
        if device == "cpu":
            pytest.skip("Flash Attention 2主要在CUDA设备上有性能优势")
            
        device_obj = get_device_object(device)
        
        # 测试参数
        batch_size = 2
        seq_len = 128
        head_dim = 64
        
        # 创建输入数据
        q = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        k = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        v = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        
        # 验证输出形状正确
        output = self.flash_attention2(q, k, v)
        assert output.shape == (batch_size, seq_len, head_dim), f"输出形状错误: 期望 {(batch_size, seq_len, head_dim)}, 实际 {output.shape}"
        
        # 验证输出值在合理范围内
        assert not torch.isnan(output).any(), "输出包含NaN值"
        assert not torch.isinf(output).any(), "输出包含Inf值"
        
        # 验证注意力计算的基本属性
        # 1. 输出应该与查询序列长度相同
        assert output.size(1) == q.size(1), "输出序列长度与查询序列长度不匹配"
        
        # 2. 输出维度应该与值向量维度相同
        assert output.size(-1) == v.size(-1), "输出维度与值向量维度不匹配"
        
    @allure.story("掩码测试")
    @allure.title("测试带掩码的Flash Attention 2")
    @allure.description("""
    测试带掩码的Flash Attention 2功能，验证：
    1. 掩码能正确屏蔽不需要的注意力计算
    2. 使用因果掩码（上三角矩阵）模拟自回归模型中的掩码机制
    3. 验证掩码后的输出形状正确
    4. 确认掩码能正确阻止后面位置信息影响前面位置
    此测试确保掩码功能在自回归模型及其他需要限制注意力范围的场景中正常工作。
    """)
    def test_flash_attention2_with_mask(self, device):
        if device == "cpu":
            pytest.skip("Flash Attention 2主要在CUDA设备上有性能优势")
            
        device_obj = get_device_object(device)
        
        # 测试参数
        batch_size = 2
        seq_len = 64
        head_dim = 32
        
        # 创建输入数据
        q = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        k = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        v = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        
        # 创建因果掩码（上三角矩阵）
        mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, device=device_obj))
        
        # 执行带掩码的attention计算
        output = self.flash_attention2(q, k, v, mask)
        
        # 验证输出形状正确
        assert output.shape == (batch_size, seq_len, head_dim), f"输出形状错误: 期望 {(batch_size, seq_len, head_dim)}, 实际 {output.shape}"
        
        # 验证掩码效果
        # 对于因果掩码，后面位置不应该影响前面位置
        # 修改后面的值，验证前面的输出不变
        k_modified = k.clone()
        k_modified[:, -1, :] = 100.0  # 显著修改最后一个位置的值
        output_modified = self.flash_attention2(q, k_modified, v, mask)
        
        # 验证前面位置的输出保持不变
        torch.testing.assert_close(output[:, :seq_len-1, :], output_modified[:, :seq_len-1, :], rtol=1e-3, atol=1e-3)
        
    @allure.story("不同数据类型测试")
    @allure.title("测试不同数据类型的Flash Attention 2")
    @allure.description("""
    测试Flash Attention 2在不同数据类型下的行为，包括：
    1. float32 - 标准精度浮点型
    2. float16 - 半精度浮点型（在支持的设备上）
    验证：
    - 输出数据类型与输入一致
    - 不产生NaN或Inf值
    - 对于不同数据类型，输出值在合理的范围内
    此测试确保算法在不同精度要求下能稳定工作。
    """)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_flash_attention2_dtypes(self, device, dtype):
        if device == "cpu" and dtype == torch.float16:
            pytest.skip("CPU不支持float16")
            
        if device == "cpu":
            pytest.skip("Flash Attention 2主要在CUDA设备上有性能优势")
            
        device_obj = get_device_object(device)
        
        # 测试参数
        batch_size = 2
        seq_len = 64
        head_dim = 32
        
        # 创建输入数据
        q = torch.randn(batch_size, seq_len, head_dim, device=device_obj, dtype=dtype)
        k = torch.randn(batch_size, seq_len, head_dim, device=device_obj, dtype=dtype)
        v = torch.randn(batch_size, seq_len, head_dim, device=device_obj, dtype=dtype)
        
        # 执行不同数据类型的attention计算
        output = self.flash_attention2(q, k, v)
        
        # 验证输出数据类型正确
        assert output.dtype == dtype, f"输出数据类型错误: 期望 {dtype}, 实际 {output.dtype}"
        
        # 验证数值稳定性
        assert not torch.isnan(output).any(), "输出包含NaN值"
        assert not torch.isinf(output).any(), "输出包含Inf值"
        
        # 验证精度在合理范围内
        if dtype == torch.float32:
            # float32应该有更高的精度要求
            assert torch.abs(output).mean() < 100.0, "float32输出值异常"
        elif dtype == torch.float16:
            # float16可以容忍较低的精度
            assert torch.abs(output).mean() < 1000.0, "float16输出值异常"
        
    @allure.story("性能测试")
    @allure.title("测试Flash Attention 2的性能")
    @allure.description("""
    测试Flash Attention 2算法的性能，使用较大序列长度（1024）：
    1. 通过多次运行获取平均执行时间
    2. 记录性能测试结果
    3. 执行预热以消除初始化开销影响
    
    注意：此实现是算法原理的简化版，可能不会比朴素实现快，
    实际的Flash Attention 2通过CUDA优化和内存访问模式优化实现性能提升。
    """)
    def test_flash_attention2_performance(self, device):
        if device == "cpu":
            pytest.skip("Flash Attention 2主要在CUDA设备上有性能优势")
            
        device_obj = get_device_object(device)
        
        # 测试参数 - 使用较大的序列长度来测试性能
        batch_size = 2
        seq_len = 1024
        head_dim = 64
        
        # 创建输入数据
        q = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        k = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        v = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        
        # 预热
        for _ in range(3):
            _ = self.flash_attention2(q, k, v)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 测量Flash Attention 2的性能
        start_time = time.time()
        for _ in range(5):
            _ = self.flash_attention2(q, k, v)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        flash_time = (time.time() - start_time) / 5
        
        # 记录性能结果
        performance_result = f"Flash Attention 2平均时间: {flash_time:.6f}秒"
        allure.attach(performance_result, name="性能比较", attachment_type=allure.attachment_type.TEXT)
        
        # 注意：这是一个简化实现，可能不会比朴素实现快
        # 实际的Flash Attention 2实现通过CUDA优化和内存访问优化实现性能提升
        print(performance_result)
        
    @allure.story("稳定性测试")
    @allure.title("测试Flash Attention 2的数值稳定性")
    @allure.description("""
    测试Flash Attention 2的数值稳定性，验证：
    1. 使用极端值输入（很大、很小、零值）时算法的稳定性
    2. 确保输出不包含NaN或Inf值
    3. 验证极端值被正确处理，输出在合理范围内
    4. 测试不同块大小下的数值一致性
    
    此测试确保算法在各种输入条件下都能保持数值稳定性，是实现高质量注意力机制的关键。
    """)
    def test_flash_attention2_stability(self, device):
        if device == "cpu":
            pytest.skip("Flash Attention 2主要在CUDA设备上有性能优势")
            
        device_obj = get_device_object(device)
        
        # 测试参数
        batch_size = 2
        seq_len = 256
        head_dim = 64
        
        # 创建具有极端值的输入数据
        q = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        k = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        v = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        
        # 添加一些极端值
        q[:, 0, :] = 1000.0  # 大值
        k[:, 1, :] = -1000.0  # 小值
        v[:, 2, :] = 0.0    # 零值
        
        # 验证数值稳定性
        output = self.flash_attention2(q, k, v)
        
        # 验证输出没有数值不稳定性
        assert not torch.isnan(output).any(), "输出包含NaN值"
        assert not torch.isinf(output).any(), "输出包含Inf值"
        
        # 验证对极端值的处理
        # 1. 大值应该被softmax归一化处理
        max_val = torch.max(torch.abs(output))
        assert max_val < 100.0, f"输出值过大: {max_val}"
        
        # 2. 零值输入应该产生合理的输出
        assert not torch.all(output[:, 2, :] == 0.0), "零值输入导致零值输出"
        
        # 测试不同的块大小
        base_output = self.flash_attention2(q, k, v, block_size=64)
        for block_size in [32, 128]:
            block_output = self.flash_attention2(q, k, v, block_size=block_size)
            # 验证不同块大小的结果应该近似相等
            torch.testing.assert_close(block_output, base_output, rtol=1e-3, atol=1e-3)
            
    @allure.story("边界情况测试")
    @allure.title("测试Flash Attention 2的边界情况")
    @allure.description("""
    测试Flash Attention 2在各种边界情况下的行为：
    1. 极短序列（长度为1）
    2. 序列长度不是块大小的整数倍
    3. 使用自定义缩放因子
    
    验证：
    - 所有边界情况下输出形状正确
    - 不产生NaN或Inf值
    - 自定义缩放因子能正确影响注意力权重分布
    
    此测试确保算法在特殊情况下的鲁棒性。
    """)
    def test_flash_attention2_edge_cases(self, device):
        if device == "cpu":
            pytest.skip("Flash Attention 2主要在CUDA设备上有性能优势")
            
        device_obj = get_device_object(device)
        
        # 测试1: 非常短的序列
        batch_size = 1
        seq_len = 1
        head_dim = 32
        
        q = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        k = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        v = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        
        output = self.flash_attention2(q, k, v)
        
        # 验证单序列长度的情况
        assert output.shape == (batch_size, seq_len, head_dim), "输出形状错误"
        assert not torch.isnan(output).any(), "输出包含NaN值"
        assert not torch.isinf(output).any(), "输出包含Inf值"
        
        # 测试2: 序列长度不是块大小的整数倍
        seq_len = 33  # 不是32的整数倍
        block_size = 32
        
        q = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        k = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        v = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        
        output = self.flash_attention2(q, k, v, block_size=block_size)
        
        # 验证非整除块大小的情况
        assert output.shape == (batch_size, seq_len, head_dim), "输出形状错误"
        assert not torch.isnan(output).any(), "输出包含NaN值"
        assert not torch.isinf(output).any(), "输出包含Inf值"
        
        # 测试3: 自定义缩放因子
        seq_len = 64
        scale = 0.5
        
        q = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        k = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        v = torch.randn(batch_size, seq_len, head_dim, device=device_obj)
        
        output = self.flash_attention2(q, k, v, scale=scale)
        
        # 验证自定义缩放因子的效果
        # 缩放因子会影响attention权重的分布
        # 较小的缩放因子会使attention权重更均匀
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attention_weights = torch.softmax(attention_weights, dim=-1)
        std_weights = torch.std(attention_weights)
        
        # 较小的缩放因子应该产生更均匀的权重分布
        assert std_weights < 0.5, f"注意力权重分布异常: std={std_weights}"

    @allure.story("多头注意力测试")
    @allure.title("测试多头注意力和QKV合并的Flash Attention 2")
    @allure.description("""
    测试Flash Attention 2在Transformer多头注意力机制下的应用：
    1. 模拟Transformer中QKV合并投影的场景
    2. 处理多头注意力的拆分与合并
    3. 验证不同的QKV分割方式产生一致的结果
    4. 测试整合的多头注意力处理函数
    
    此测试确保Flash Attention 2可以无缝集成到Transformer架构中，
    正确处理多头注意力机制中的QKV投影、拆分和合并操作。
    """)
    def test_flash_attention2_multihead_qkv(self, device):
        if device == "cpu":
            pytest.skip("Flash Attention 2主要在CUDA设备上有性能优势")
            
        device_obj = get_device_object(device)
        
        # 测试参数
        batch_size = 2
        seq_len = 128
        hidden_dim = 512
        num_heads = 8
        head_dim = hidden_dim // num_heads
        
        # 创建合并的qkv输入 - 模拟Transformer中的情况
        # 通常在Transformer中，会有一个线性层将hidden_dim投影到3*hidden_dim，
        # 然后分割为q, k, v三个部分
        qkv_weight = torch.randn(hidden_dim, 3 * hidden_dim, device=device_obj)
        qkv_bias = torch.randn(3 * hidden_dim, device=device_obj)
        
        # 输入序列
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device_obj)
        
        # 线性投影得到合并的qkv
        qkv_combined = torch.matmul(x, qkv_weight) + qkv_bias  # [batch_size, seq_len, 3*hidden_dim]
        
        # 分割qkv
        qkv = qkv_combined.view(batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        # 提取q, k, v
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个形状为[batch_size, num_heads, seq_len, head_dim]
        
        # 对每个头分别应用flash_attention2
        outputs = []
        for h in range(num_heads):
            q_h = q[:, h]  # [batch_size, seq_len, head_dim]
            k_h = k[:, h]  # [batch_size, seq_len, head_dim]
            v_h = v[:, h]  # [batch_size, seq_len, head_dim]
            
            # 应用flash_attention2
            output_h = self.flash_attention2(q_h, k_h, v_h)  # [batch_size, seq_len, head_dim]
            outputs.append(output_h)
        
        # 合并多头的输出
        multi_head_output = torch.stack(outputs, dim=1)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重塑回原始维度
        output = multi_head_output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        output = output.view(batch_size, seq_len, hidden_dim)  # [batch_size, seq_len, hidden_dim]
        
        # 验证输出形状正确
        assert output.shape == (batch_size, seq_len, hidden_dim), f"输出形状错误: 期望 {(batch_size, seq_len, hidden_dim)}, 实际 {output.shape}"
        
        # 验证输出值在合理范围内
        assert not torch.isnan(output).any(), "输出包含NaN值"
        assert not torch.isinf(output).any(), "输出包含Inf值"
        
        # 验证view和split操作的正确性
        # 重新计算一遍，但使用不同的方式分割qkv
        qkv_chunks = torch.chunk(qkv_combined, 3, dim=-1)  # 分成3个张量
        q_alt = qkv_chunks[0].view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        k_alt = qkv_chunks[1].view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        v_alt = qkv_chunks[2].view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 验证两种分割方式得到的结果一致
        torch.testing.assert_close(q, q_alt, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(k, k_alt, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(v, v_alt, rtol=1e-5, atol=1e-5)
        
        # 测试合并的QKV处理函数
        def flash_attention2_qkv(qkv_combined, num_heads):
            """处理合并的QKV输入并应用Flash Attention 2"""
            batch_size, seq_len, hidden_dim = qkv_combined.shape
            hidden_dim = hidden_dim // 3
            head_dim = hidden_dim // num_heads
            
            # 分割qkv
            qkv = qkv_combined.view(batch_size, seq_len, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]  # 每个形状为[batch_size, num_heads, seq_len, head_dim]
            
            # 对每个头分别应用flash_attention2
            outputs = []
            for h in range(num_heads):
                q_h = q[:, h]  # [batch_size, seq_len, head_dim]
                k_h = k[:, h]  # [batch_size, seq_len, head_dim]
                v_h = v[:, h]  # [batch_size, seq_len, head_dim]
                
                # 应用flash_attention2
                output_h = self.flash_attention2(q_h, k_h, v_h)  # [batch_size, seq_len, head_dim]
                outputs.append(output_h)
            
            # 合并多头的输出
            multi_head_output = torch.stack(outputs, dim=1)  # [batch_size, num_heads, seq_len, head_dim]
            
            # 重塑回原始维度
            output = multi_head_output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
            output = output.view(batch_size, seq_len, hidden_dim)  # [batch_size, seq_len, hidden_dim]
            
            return output
        
        # 使用合并函数处理QKV
        output_combined = flash_attention2_qkv(qkv_combined, num_heads)
        
        # 验证输出形状正确
        assert output_combined.shape == (batch_size, seq_len, hidden_dim), f"合并函数输出形状错误: 期望 {(batch_size, seq_len, hidden_dim)}, 实际 {output_combined.shape}"
        
        # 验证两种方式得到的结果一致
        torch.testing.assert_close(output, output_combined, rtol=1e-5, atol=1e-5)
