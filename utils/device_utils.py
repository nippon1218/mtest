import torch

def get_device_object(device):
    """获取设备对象"""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA设备不可用")
    return torch.device(device)

def get_device_info():
    """获取CUDA设备信息"""
    if not torch.cuda.is_available():
        return
        
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"CUDA设备信息: {torch.cuda.get_device_name(0)}")
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
