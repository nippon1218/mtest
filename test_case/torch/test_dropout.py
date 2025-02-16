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
class TestDropout:
    @allure.story("基本功能测试")
    @allure.title("测试Dropout基本功能 - {dtype}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_dropout_basic(self, dtype, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用")
            
        device_obj = get_device_object(device)
        if device == "cuda":
            get_device_info()
            
        # 创建输入数据
        batch_size, features = 32, 64
        x = torch.ones(batch_size, features, dtype=dtype, device=device_obj)
        dropout = torch.nn.Dropout(p=0.5)
        
        # 训练模式测试
        dropout.train()
        output = dropout(x)
        
        # 验证输出在训练模式下的特性
        assert output.shape == x.shape, "输出形状应与输入相同"
        assert torch.any(output == 0), "应该有一些元素被置为0"
        assert torch.any(output == 2.0), "未被丢弃的元素应该被放大"
        
        # 推理模式测试
        dropout.eval()
        output = dropout(x)
        assert torch.allclose(output, x), "推理模式下输出应该与输入相同"
        
    @allure.story("不同丢弃率测试")
    @allure.title("测试不同丢弃率 - {device}")
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_dropout_rates(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用")
            
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        input_size = 10000  # 使用大量样本以获得稳定的统计结果
        x = torch.ones(input_size, dtype=dtype, device=device_obj)
        
        test_rates = [0.1, 0.5, 0.8]
        for p in test_rates:
            with allure.step(f"测试丢弃率 {p}"):
                dropout = torch.nn.Dropout(p=p)
                dropout.train()
                
                # 运行多次以获得稳定的统计结果
                zero_ratios = []
                for _ in range(10):
                    output = dropout(x)
                    zero_ratio = (output == 0).float().mean().item()
                    zero_ratios.append(zero_ratio)
                
                avg_zero_ratio = sum(zero_ratios) / len(zero_ratios)
                assert abs(avg_zero_ratio - p) < 0.05, f"实际丢弃率 {avg_zero_ratio:.3f} 应接近目标丢弃率 {p}"
                
    @allure.story("数值范围测试")
    @allure.title("测试Dropout数值范围 - {device}")
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_dropout_value_range(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用")
            
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        test_cases = [
            ("小值输入", 1e-3),
            ("正常值输入", 1.0),
            ("大值输入", 1e3)
        ]
        
        for case_name, value in test_cases:
            with allure.step(f"测试{case_name}"):
                x = torch.full((100,), value, dtype=dtype, device=device_obj)
                dropout = torch.nn.Dropout(p=0.5)
                
                # 训练模式
                dropout.train()
                output = dropout(x)
                assert not torch.isnan(output).any(), f"{case_name}的输出不应包含NaN"
                assert not torch.isinf(output).any(), f"{case_name}的输出不应包含Inf"
                
                # 验证非零值是原始值的倍数
                non_zero_values = output[output != 0]
                if len(non_zero_values) > 0:
                    assert torch.allclose(non_zero_values / value, 
                                        torch.tensor(2.0, device=device_obj),
                                        rtol=1e-5), "非零值应该是输入值的2倍"
                
    @allure.story("随机性测试")
    @allure.title("测试Dropout随机性 - {device}")
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_dropout_randomness(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用")
            
        device_obj = get_device_object(device)
        dtype = torch.float32
        
        # 创建输入数据
        x = torch.ones(1000, dtype=dtype, device=device_obj)
        dropout = torch.nn.Dropout(p=0.5)
        dropout.train()
        
        # 生成多个输出
        outputs = [dropout(x) for _ in range(5)]
        
        # 验证不同运行之间的输出是否不同
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j]), "不同运行之间的输出应该不同"
                
        # 验证种子固定时输出相同
        torch.manual_seed(42)
        output1 = dropout(x)
        torch.manual_seed(42)
        output2 = dropout(x)
        assert torch.allclose(output1, output2), "相同种子下的输出应该相同"
                
    @allure.story("CPU和CUDA精度对比测试")
    @allure.title("对比CPU和CUDA输出精度 - {dtype}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dropout_cpu_cuda_precision(self, dtype):
        if not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用")
            
        get_device_info()
        
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
