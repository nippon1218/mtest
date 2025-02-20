import torch
import pytest
import allure
import numpy as np
from utils.device_utils import get_device_object, get_device_info

def compare_dropout_statistics(output, x, p, rtol=0.1):
    """对比Dropout输出的统计特性
    
    Args:
        output: Dropout输出
        x: 原始输入
        p: 丢弃率
        rtol: 相对误差容容许范围
        
    Returns:
        bool: 是否符合统计特性
    """
    # 移到CPU进行计算
    if output.is_cuda:
        output = output.cpu()
    if x.is_cuda:
        x = x.cpu()
        
    # 计算实际丢弃率
    zero_ratio = (output == 0).float().mean().item()
    expected_ratio = p
    
    # 计算非零元素的缩放比例
    non_zero_scale = output[output != 0] / x[output != 0]
    expected_scale = 1.0 / (1.0 - p)
    actual_scale = non_zero_scale.mean().item()
    
    # 验证丢弃率和缩放比例
    return (abs(zero_ratio - expected_ratio) < rtol and 
            abs(actual_scale - expected_scale) < rtol * expected_scale)

@allure.epic("PyTorch算子测试")
@allure.feature("Dropout层")
@allure.description("""
Dropout层的完整测试套件，包含以下测试内容：
1. 基本功能测试：验证训练和推理模式下的行为
2. 不同丢弃率测试：验证各种丢弃率的准确性
3. 数值范围测试：验证对不同量级输入的处理
4. 随机性测试：验证随机行为和可重现性
5. CPU和CUDA精度对比：验证跨设备一致性

每个测试用例都包含详细的参数验证和边界条件检查。
""")
class TestDropout:
    @allure.story("基本功能测试")
    @allure.title("测试Dropout基本功能 - {dtype}")
    @allure.description("""
    验证Dropout层的基本功能，测试要点：
    1. 训练模式：
       - 部分元素被置为0
       - 未被丢弃的元素被正确放大
       - 输出形状保持不变
    2. 推理模式：
       - 输出与输入完全相同
       - 不进行任何丢弃操作
    3. 支持的数据类型：
       - float32和float64
    4. 设备兼容性：
       - 支持CPU和CUDA设备
    """)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dropout_basic(self, dtype, device):
        device_obj = get_device_object(device)
            
        # 创建输入数据
        batch_size, features = 32, 64
        x = torch.ones(batch_size, features, dtype=dtype)
        x_dev = x.to(device=device_obj)
        dropout = torch.nn.Dropout(p=0.5)
        
        # 训练模式测试
        dropout.train()
        output = dropout(x_dev)
        
        # 验证输出在训练模式下的特性
        assert output.shape == x_dev.shape, "输出形状应与输入相同"
        assert torch.any(output == 0), "应该有一些元素被置为0"
        assert torch.any(output == 2.0), "未被丢弃的元素应该被放大"
        
        # 推理模式测试
        dropout.eval()
        output = dropout(x_dev)
        assert torch.allclose(output, x_dev), "推理模式下输出应该与输入相同"
        
        # 切换回训练模式进行CPU和CUDA的对比
        dropout.train()
        output = dropout(x_dev)
        
        if device == "cuda":
            # 在CPU上运行参考结果
            dropout_cpu = torch.nn.Dropout(p=0.5)
            dropout_cpu.train()
            output_cpu = dropout_cpu(x)
            
            # 验证统计特性
            # 1. 检查丢弃率
            zero_ratio_cpu = (output_cpu == 0).float().mean().item()
            zero_ratio_cuda = (output.cpu() == 0).float().mean().item()
            assert abs(zero_ratio_cpu - 0.5) < 0.1, f"CPU的丢弃率应接近 0.5，实际为 {zero_ratio_cpu:.3f}"
            assert abs(zero_ratio_cuda - 0.5) < 0.1, f"CUDA的丢弃率应接近 0.5，实际为 {zero_ratio_cuda:.3f}"
            
            # 2. 检查非零值的缩放
            non_zero_scale_cpu = output_cpu[output_cpu != 0].mean().item()
            non_zero_scale_cuda = output.cpu()[output.cpu() != 0].mean().item()
            assert abs(non_zero_scale_cpu - 2.0) < 0.1, f"CPU的非零值平均应接近 2.0，实际为 {non_zero_scale_cpu:.3f}"
            assert abs(non_zero_scale_cuda - 2.0) < 0.1, f"CUDA的非零值平均应接近 2.0，实际为 {non_zero_scale_cuda:.3f}"
            
            # 推理模式对比
            dropout_cpu.eval()
            output_cpu = dropout_cpu(x)
            
            dropout.eval()
            output_cuda = dropout(x_dev)
            torch.testing.assert_close(output_cuda.cpu(), output_cpu, rtol=1e-5, atol=1e-5)
        
    @allure.story("不同丢弃率测试")
    @allure.title("测试不同丢弃率 - {device}")
    @allure.description("""
    验证Dropout在不同丢弃率下的行为，测试要点：
    1. 测试多个丢弃率：
       - 低丢弃率 (0.1)
       - 中等丢弃率 (0.5)
       - 高丢弃率 (0.8)
    2. 统计验证：
       - 实际丢弃率与目标丢弃率的误差小于5%
       - 通过多次运行获取稳定的统计结果
    3. 设备兼容性：
       - 在CPU和CUDA上保持一致的行为
    """)
    def test_dropout_rates(self, device):
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        input_size = 10000  # 使用大量样本以获得稳定的统计结果
        x = torch.ones(input_size, dtype=dtype)
        x_dev = x.to(device=device_obj)
        
        test_rates = [0.1, 0.5, 0.8]
        for p in test_rates:
            with allure.step(f"测试丢弃率 {p}"):
                dropout = torch.nn.Dropout(p=p)
                dropout.train()
                
                # 运行多次以获得稳定的统计结果
                zero_ratios = []
                for _ in range(10):
                    output = dropout(x_dev)
                    zero_ratio = (output == 0).float().mean().item()
                    zero_ratios.append(zero_ratio)
                
                avg_zero_ratio = sum(zero_ratios) / len(zero_ratios)
                assert abs(avg_zero_ratio - p) < 0.05, f"实际丢弃率 {avg_zero_ratio:.3f} 应接近目标丢弃率 {p}"
                
                if device == "cuda":
                    # 在CPU上运行参考结果
                    dropout_cpu = torch.nn.Dropout(p=p)
                    dropout_cpu.train()
                    output_cpu = dropout_cpu(x)
                    
                    # 验证统计特性
                    # 1. 检查丢弃率
                    zero_ratio_cpu = (output_cpu == 0).float().mean().item()
                    zero_ratio_cuda = (output.cpu() == 0).float().mean().item()
                    assert abs(zero_ratio_cpu - p) < 0.1, f"CPU的丢弃率应接近 {p}，实际为 {zero_ratio_cpu:.3f}"
                    assert abs(zero_ratio_cuda - p) < 0.1, f"CUDA的丢弃率应接近 {p}，实际为 {zero_ratio_cuda:.3f}"
                    
                    # 2. 检查非零值的缩放
                    expected_scale = 1.0 / (1.0 - p)
                    non_zero_scale_cpu = output_cpu[output_cpu != 0].mean().item()
                    non_zero_scale_cuda = output.cpu()[output.cpu() != 0].mean().item()
                    assert abs(non_zero_scale_cpu - expected_scale) < 0.1, f"CPU的非零值平均应接近 {expected_scale}，实际为 {non_zero_scale_cpu:.3f}"
                    assert abs(non_zero_scale_cuda - expected_scale) < 0.1, f"CUDA的非零值平均应接近 {expected_scale}，实际为 {non_zero_scale_cuda:.3f}"
                
    @allure.story("数值范围测试")
    @allure.title("测试Dropout数值范围 - {device}")
    @allure.description("""
    验证Dropout对不同数值范围输入的处理，测试要点：
    1. 输入范围测试：
       - 小值输入 (1e-3)
       - 正常值输入 (1.0)
       - 大值输入 (1e3)
    2. 数值稳定性：
       - 检查NaN和Inf
       - 验证缩放系数的准确性
    3. 输出验证：
       - 非零值是输入的正确倍数
       - 保持数值精度
    """)
    def test_dropout_value_range(self, device):
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        test_cases = [
            ("小值输入", 1e-3),
            ("正常值输入", 1.0),
            ("大值输入", 1e3)
        ]
        
        for case_name, value in test_cases:
            with allure.step(f"测试{case_name}"):
                x = torch.full((100,), value, dtype=dtype)
                x_dev = x.to(device=device_obj)
                dropout = torch.nn.Dropout(p=0.5)
                
                # 训练模式
                dropout.train()
                output = dropout(x_dev)
                assert not torch.isnan(output).any(), f"{case_name}的输出不应包含NaN"
                assert not torch.isinf(output).any(), f"{case_name}的输出不应包含Inf"
                
                # 验证非零值是原始值的倍数
                non_zero_values = output[output != 0]
                if len(non_zero_values) > 0:
                    assert torch.allclose(non_zero_values / value, 
                                        torch.tensor(2.0, device=device_obj),
                                        rtol=1e-5), "非零值应该是输入值的2倍"
                
                if device == "cuda":
                    # 在CPU上运行参考结果
                    dropout_cpu = torch.nn.Dropout(p=0.5)
                    dropout_cpu.train()
                    output_cpu = dropout_cpu(x)
                    
                    # 验证统计特性
                    # 1. 检查丢弃率
                    zero_ratio_cpu = (output_cpu == 0).float().mean().item()
                    zero_ratio_cuda = (output.cpu() == 0).float().mean().item()
                    assert abs(zero_ratio_cpu - 0.5) < 0.2, f"CPU的丢弃率应接近 0.5，实际为 {zero_ratio_cpu:.3f}"
                    assert abs(zero_ratio_cuda - 0.5) < 0.2, f"CUDA的丢弃率应接近 0.5，实际为 {zero_ratio_cuda:.3f}"
                    
                    # 2. 检查非零值的缩放
                    non_zero_scale_cpu = output_cpu[output_cpu != 0].mean().item() / value
                    non_zero_scale_cuda = output.cpu()[output.cpu() != 0].mean().item() / value
                    assert abs(non_zero_scale_cpu - 2.0) < 0.1, f"CPU的非零值缩放应接近 2.0，实际为 {non_zero_scale_cpu:.3f}"
                    assert abs(non_zero_scale_cuda - 2.0) < 0.1, f"CUDA的非零值缩放应接近 2.0，实际为 {non_zero_scale_cuda:.3f}"
                
    @allure.story("随机性测试")
    @allure.title("测试Dropout随机性 - {device}")
    @allure.description("""
    验证Dropout的随机行为和可重现性，测试要点：
    1. 随机性验证：
       - 多次运行产生不同的输出
       - 不同运行之间的输出模式不同
    2. 可重现性：
       - 相同随机种子产生相同结果
       - 在相同设备上保持一致
    3. 随机性质量：
       - 输出分布的均匀性
       - 避免明显的模式或偏差
    """)
    def test_dropout_randomness(self, device):
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        # 创建输入数据
        x = torch.ones(1000, dtype=dtype)
        x_dev = x.to(device=device_obj)
        dropout = torch.nn.Dropout(p=0.5)
        dropout.train()
        
        # 生成多个输出
        outputs = [dropout(x_dev) for _ in range(5)]
        
        # 验证不同运行之间的输出是否不同
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j]), "不同运行之间的输出应该不同"
                
        # 验证种子固定时输出相同
        torch.manual_seed(42)
        output1 = dropout(x_dev)
        torch.manual_seed(42)
        output2 = dropout(x_dev)
        assert torch.allclose(output1, output2), "相同种子下的输出应该相同"
        
        if device == "cuda":
            # 在CPU上运行参考结果
            dropout_cpu = torch.nn.Dropout(p=0.5)
            dropout_cpu.train()
            output_cpu = dropout_cpu(x)
            
            with allure.step("验证CPU和CUDA的统计特性"):
                # 1. 检查丢弃率
                with allure.step("检查丢弃率"):
                    zero_ratio_cpu = (output_cpu == 0).float().mean().item()
                    zero_ratio_cuda = (output1.cpu() == 0).float().mean().item()
                    assert abs(zero_ratio_cpu - 0.5) < 0.2, f"CPU的丢弃率应接近 0.5，实际为 {zero_ratio_cpu:.3f}"
                    assert abs(zero_ratio_cuda - 0.5) < 0.2, f"CUDA的丢弃率应接近 0.5，实际为 {zero_ratio_cuda:.3f}"
                
                # 2. 检查非零值的缩放
                with allure.step("检查非零值的缩放"):
                    non_zero_scale_cpu = output_cpu[output_cpu != 0].mean().item()
                    non_zero_scale_cuda = output1.cpu()[output1.cpu() != 0].mean().item()
                    assert abs(non_zero_scale_cpu - 2.0) < 0.2, f"CPU的非零值平均应接近 2.0，实际为 {non_zero_scale_cpu:.3f}"
                    assert abs(non_zero_scale_cuda - 2.0) < 0.2, f"CUDA的非零值平均应接近 2.0，实际为 {non_zero_scale_cuda:.3f}"
                
    @allure.story("CPU和CUDA精度对比测试")
    @allure.title("对比CPU和CUDA输出精度 - {dtype}")
    @allure.description("""
    验证Dropout在CPU和CUDA设备上的计算一致性，测试要点：
    1. 精度对比：
       - 在不同数据类型下保持一致性
       - 验证数值误差在可接受范围内
    2. 随机性控制：
       - 使用相同的随机种子
       - 确保CUDA操作的确定性
    3. 大规模测试：
       - 使用较大的输入规模
       - 测试不同的丢弃率
    """)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dropout_cpu_cuda_precision(self, dtype, device):
        if device != "cuda":
            pytest.skip("此测试仅在CUDA设备上运行")
            
        device_obj = get_device_object(device)
        
        # 创建输入数据
        batch_size, features = 1024, 1024
        x = torch.randn(batch_size, features, dtype=dtype)
        
        test_rates = [0.1, 0.5, 0.8]
        for p in test_rates:
            with allure.step(f"测试丢弃率 {p}"):
                # 创建CPU和CUDA的dropout层
                dropout_cpu = torch.nn.Dropout(p=p)
                dropout_cuda = torch.nn.Dropout(p=p)
                
                # 训练模式测试
                with allure.step("训练模式测试"):
                    dropout_cpu.train()
                    dropout_cuda.train()
                    
                    # 固定CPU和CUDA的随机种子
                    torch.manual_seed(42)
                    torch.cuda.manual_seed_all(42)
                    
                    # 确保CUDA操作是确定性的
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    
                    # 运行CPU版本
                    output_cpu = dropout_cpu(x)
                    
                    # 重置随机种子
                    torch.manual_seed(42)
                    torch.cuda.manual_seed_all(42)
                    
                    # 运行CUDA版本
                    x_cuda = x.cuda()
                    output_cuda = dropout_cuda(x_cuda)
                    
                    # 对比CPU和CUDA输出的统计特性
                    assert compare_dropout_statistics(output_cpu, x, p), \
                        f"CPU输出的统计特性不符合预期"
                    assert compare_dropout_statistics(output_cuda, x_cuda, p), \
                        f"CUDA输出的统计特性不符合预期"
                    
                    # 验证输出的数值范围
                    for output in [output_cpu, output_cuda]:
                        # 检查非零值的范围
                        non_zero_vals = output[output != 0]
                        if len(non_zero_vals) > 0:
                            assert not torch.isnan(non_zero_vals).any(), "输出包含NaN"
                            assert not torch.isinf(non_zero_vals).any(), "输出包含Inf"
                            
                            # 检查缩放比例
                            scale = 1.0 / (1.0 - p)
                            x_vals = x if output is output_cpu else x_cuda
                            scaled_input = x_vals[output != 0] * scale
                            assert torch.allclose(non_zero_vals, scaled_input, rtol=1e-3), \
                                "非零值的缩放比例不正确"
                
                # 推理模式测试
                with allure.step("推理模式测试"):
                    dropout_cpu.eval()
                    dropout_cuda.eval()
                    
                    output_cpu = dropout_cpu(x)
                    output_cuda = dropout_cuda(x.cuda())
                    
                    # 推理模式下，输出应该与输入相同
                    assert torch.allclose(output_cpu, x, rtol=1e-5, atol=1e-5), \
                        f"CPU推理模式下的输出应该与输入相同"
                    assert torch.allclose(output_cuda.cpu(), x.cuda().cpu(), rtol=1e-5, atol=1e-5), \
                        f"CUDA推理模式下的输出应该与输入相同"
