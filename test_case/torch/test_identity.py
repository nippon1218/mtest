import torch
import pytest
import allure
import numpy as np
from utils.device_utils import get_device_object, get_device_info

@allure.feature("Identity操作")
class TestIdentity:
    @allure.story("基本功能测试")
    @allure.title("测试Identity基本功能 - {device}-{dtype}")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int32, torch.int64])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_identity_basic(self, dtype, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用")
            
        if device == "cuda":
            get_device_info()
            
        device_obj = get_device_object(device)
        
        # 测试不同形状的输入
        test_shapes = [
            (1,),           # 标量
            (10,),          # 一维向量
            (32, 64),       # 二维矩阵
            (16, 32, 64),   # 三维张量
            (8, 16, 32, 64) # 四维张量
        ]
        
        for shape in test_shapes:
            with allure.step(f"测试形状 {shape}"):
                # 创建测试数据
                if dtype in [torch.float32, torch.float64]:
                    x = torch.randn(shape, dtype=dtype, device=device_obj)
                else:
                    x = torch.randint(-100, 100, shape, dtype=dtype, device=device_obj)
                
                # 运行Identity操作
                y = torch.nn.Identity()(x)
                
                # 验证形状保持不变
                assert x.shape == y.shape, f"输出形状 {y.shape} 与输入形状 {x.shape} 不一致"
                
                # 验证值保持不变
                assert torch.all(x == y), "输出值与输入值不完全相同"
                
                # 验证数据类型保持不变
                assert x.dtype == y.dtype, f"输出类型 {y.dtype} 与输入类型 {x.dtype} 不一致"
                
                # 验证设备保持不变
                assert x.device == y.device, f"输出设备 {y.device} 与输入设备 {x.device} 不一致"
                
    @allure.story("梯度测试")
    @allure.title("测试Identity梯度传递 - {device}")
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_identity_gradient(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用")
            
        if device == "cuda":
            get_device_info()
            
        device_obj = get_device_object(device)
        
        # 创建需要梯度的输入
        x = torch.randn(32, 64, dtype=torch.float32, device=device_obj, requires_grad=True)
        identity = torch.nn.Identity()
        
        # 前向传播
        y = identity(x)
        
        # 创建随机梯度
        grad_output = torch.randn_like(y)
        
        # 反向传播
        y.backward(grad_output)
        
        # 验证梯度是否正确传递
        assert torch.allclose(x.grad, grad_output), "梯度没有正确传递"
        
    @allure.story("内存使用测试")
    @allure.title("测试Identity内存使用 - {device}")
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_identity_memory(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用")
            
        if device == "cuda":
            get_device_info()
            
        device_obj = get_device_object(device)
        
        # 创建大型输入以测试内存使用
        x = torch.randn(1024, 1024, dtype=torch.float32, device=device_obj)
        identity = torch.nn.Identity()
        
        # 获取输入的存储位置
        x_storage_ptr = x.storage().data_ptr()
        
        # 运行Identity操作
        y = identity(x)
        y_storage_ptr = y.storage().data_ptr()
        
        # 验证是否共享相同的存储空间
        assert x_storage_ptr == y_storage_ptr, "Identity操作创建了新的内存空间"
        
        # 验证修改输入会影响输出
        x[0, 0] = 999.0
        assert y[0, 0] == 999.0, "输入和输出没有共享内存"
        
    @allure.story("类型转换测试")
    @allure.title("测试Identity类型转换")
    def test_identity_type_conversion(self):
        identity = torch.nn.Identity()
        
        # 测试不同类型的张量输入
        test_inputs = [
            # 测试列表转张量
            torch.tensor([1, 2, 3], dtype=torch.int64),
            
            # 测试浮点数列表转张量
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            
            # 测试标量转张量
            torch.tensor(42, dtype=torch.int64),
            torch.tensor(3.14, dtype=torch.float32),
            
            # 测试布尔值转张量
            torch.tensor(True, dtype=torch.bool),
            
            # 测试NumPy数组转张量
            torch.from_numpy(np.array([1, 2, 3]))
        ]
        
        for input_tensor in test_inputs:
            with allure.step(f"测试输入: {input_tensor}"):
                output = identity(input_tensor)
                
                # 验证输出是张量
                assert isinstance(output, torch.Tensor), \
                    f"输出应该是张量，实际是{type(output)}"
                
                # 验证数据类型
                assert output.dtype == input_tensor.dtype, \
                    f"输出类型应该是{input_tensor.dtype}，实际是{output.dtype}"
                
                # 验证形状
                assert output.shape == input_tensor.shape, \
                    f"输出形状应该是{input_tensor.shape}，实际是{output.shape}"
                
                # 验证值的正确性
                assert torch.all(output == input_tensor), \
                    "输出值与输入不符"
            
        # 测试空张量（应该正常工作）
        empty_tensor = torch.tensor([])
        output = identity(empty_tensor)
        assert torch.equal(output, empty_tensor), "空张量应该正常通过Identity"
        
        # 测试各种特殊形状的张量
        special_shapes = [
            torch.zeros(0, 10),      # 第一维为0
            torch.zeros(10, 0),      # 第二维为0
            torch.zeros(0, 0),       # 所有维度都为0
            torch.zeros(1, 1, 1),    # 所有维度都为1
            torch.zeros(()),         # 标量张量
        ]
        
        for tensor in special_shapes:
            output = identity(tensor)
            assert output.shape == tensor.shape, f"形状{tensor.shape}的张量应该保持形状不变"
            assert torch.equal(output, tensor), f"形状{tensor.shape}的张量应该保持值不变"
            
    @allure.story("批处理测试")
    @allure.title("测试Identity批处理 - {device}")
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_identity_batching(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA设备不可用")
            
        if device == "cuda":
            get_device_info()
            
        device_obj = get_device_object(device)
        
        # 测试不同的批大小
        batch_sizes = [1, 16, 256, 1024]
        feature_size = 64
        
        for batch_size in batch_sizes:
            with allure.step(f"测试批大小 {batch_size}"):
                x = torch.randn(batch_size, feature_size, dtype=torch.float32, device=device_obj)
                identity = torch.nn.Identity()
                
                # 运行Identity操作
                y = identity(x)
                
                # 验证每个样本都正确处理
                for i in range(batch_size):
                    assert torch.all(x[i] == y[i]), f"批处理中的第{i}个样本处理不正确"
