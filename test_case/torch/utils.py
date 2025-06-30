#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 使用我们的导入辅助模块替代直接导入
from .torch_import import torch, torch_import_failed

def get_device_object(device_str):
    """获取torch.device对象"""
    if torch_import_failed:
        return None
    if device_str == "cuda":
        return torch.device("cuda:0")
    return torch.device("cpu")

test_dtypes = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64
]

em_test_dtypes = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64
]

em_input_dtypes = [
    torch.int32,
    torch.int64
]

float_dtypes = [
    torch.bfloat16,
    torch.float32
]

import numpy as np
import json
import os

def parse_dtype_from_str(dtype_str):
    """从字符串解析数据类型"""
    if dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "fp32":
        return torch.float32
    elif dtype_str == "fp64":
        return torch.float64
    elif dtype_str == "int32":
        return torch.int32
    elif dtype_str == "int64":
        return torch.int64
    else:
        return torch.float32

def load_binary_data(file_path, dtype, shape):
    """直接从二进制文件加载张量，使用已知的dtype和shape"""
    with open(file_path, 'rb') as f:
        # 计算数据大小
        data_size = int(np.prod(shape))
        print(f"vincent data size is {data_size}")
        
        if dtype == torch.bfloat16:
            # 对于bfloat16，我们需要读取为uint16然后转换
            raw_data = np.fromfile(f, dtype=np.uint16, count=data_size)
            tensor = torch.from_numpy(raw_data).view(dtype=torch.bfloat16)
        else:
            # 对于其他类型，直接读取
            if dtype == torch.float32:
                np_dtype = np.float32
            elif dtype == torch.float64:
                np_dtype = np.float64
            elif dtype == torch.int32:
                np_dtype = np.int32
            elif dtype == torch.int64:
                np_dtype = np.int64
            else:
                np_dtype = np.float32  # 默认
                
            raw_data = np.fromfile(f, dtype=np_dtype, count=data_size)
            tensor = torch.from_numpy(raw_data).to(dtype)
        
        print(f"vincent 1 tensor shape is {tensor.shape}")
        print(f"vincent 2 tensor shape is {shape}")
        tensor = tensor.reshape(shape)
    print(f"已从 {file_path} 加载形状为 {tensor.shape} 的张量")
    return tensor

def validate_data(dir_path=None, device="cpu", op_func=None):
    """从二进制文件加载张量并验证加法操作，使用JSON描述文件获取信息
    支持两种JSON格式：
    1. 对象格式 (以 {} 开头)：包含inputs和expected字段
    2. 数组格式 (以 [] 开头)：包含输入和期望输出的数据项
    """
    # 如果没有提供目录路径，则尝试查找最新的数据目录
    if dir_path is None:
        # 查找data目录下所有以add_开头的子目录
        data_dirs = [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d)) and d.startswith("add_")]
        if not data_dirs:
            raise ValueError("未找到数据目录，请先运行generate_data()")
        
        # 按修改时间排序，选择最新的目录
        data_dirs.sort(key=lambda d: os.path.getmtime(os.path.join("data", d)), reverse=True)
        dir_path = os.path.join("data", data_dirs[0])
    
    # 加载JSON描述文件
    json_path = os.path.join(dir_path, "description.json")
    print(f"加载描述文件: {json_path}")
    try:
        with open(json_path, 'r') as f:
            description = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"未找到描述文件: {json_path}，请确保数据已正确生成")
    
    # 从描述文件获取输入和期望输出的信息
    print("从描述文件解析数据信息...")
    
    # 判断JSON格式是对象还是数组
    if isinstance(description, dict):
        # 原有的对象格式处理逻辑
        # 获取输入x的信息
        x_info = description["inputs"]["x"]
        x_path = x_info["data"]
        x_dtype = parse_dtype_from_str(x_info["dtype"])
        x_shape = tuple(x_info["shape"])
        
        # 获取输入y的信息
        y_info = description["inputs"]["y"]
        y_path = y_info["data"]
        y_dtype = parse_dtype_from_str(y_info["dtype"])
        y_shape = tuple(y_info["shape"])
        
        # 获取期望输出z的信息
        z_info = description["expected"]["z"]
        z_path = z_info["data"]
        z_dtype = parse_dtype_from_str(z_info["dtype"])
        z_shape = tuple(z_info["shape"])
    elif isinstance(description, list):
        # 新增的数组格式处理逻辑
        if len(description) < 1:
            raise ValueError(f"数组格式的JSON文件必须至少包含1个元素，但找到 {len(description)} 个")
        
        # 获取第一个元素（可能是包含完整信息的对象）
        item = description[0]
        
        # 检查item是否包含inputs和expected字段
        if "inputs" in item and "expected" in item:
            # 获取输入x的信息
            x_info = item["inputs"]["x"]
            x_path = x_info["data"]
            x_dtype = parse_dtype_from_str(x_info["dtype"])
            x_shape = tuple(x_info["shape"])
            
            # 获取输入y的信息
            y_info = item["inputs"]["y"]
            y_path = y_info["data"]
            y_dtype = parse_dtype_from_str(y_info["dtype"])
            y_shape = tuple(y_info["shape"])
            
            # 获取期望输出z的信息
            z_info = item["expected"]["z"]
            z_path = z_info["data"]
            z_dtype = parse_dtype_from_str(z_info["dtype"])
            z_shape = tuple(z_info["shape"])
        # 检查是否是简单的三个元素数组（x, y, z）
        elif len(description) >= 3:
            # 获取输入x的信息
            x_info = description[0]
            x_path = x_info["data"]
            x_dtype = parse_dtype_from_str(x_info["dtype"])
            x_shape = tuple(x_info["shape"])
            
            # 获取输入y的信息
            y_info = description[1]
            y_path = y_info["data"]
            y_dtype = parse_dtype_from_str(y_info["dtype"])
            y_shape = tuple(y_info["shape"])
            
            # 获取期望输出z的信息
            z_info = description[2]
            z_path = z_info["data"]
            z_dtype = parse_dtype_from_str(z_info["dtype"])
            z_shape = tuple(z_info["shape"])
        else:
            raise ValueError(f"数组格式的JSON文件结构不正确，无法解析输入和输出信息")
    else:
        raise ValueError(f"不支持的JSON格式: {type(description)}，必须是对象或数组")
    
    # 加载张量数据
    print("从二进制文件加载张量...")
    # 直接加载二进制数据，不需要从目录名解析信息
    x = load_binary_data(x_path, x_dtype, x_shape)
    y = load_binary_data(y_path, y_dtype, y_shape)
    z_saved = load_binary_data(z_path, z_dtype, z_shape)
    
    # 将张量移动到指定设备
    print(f"将张量移动到设备: {device}")
    dev_obj = get_device_object(device)
    x_dev = x.to(device=dev_obj)
    y_dev = y.to(device=dev_obj)
    z_saved_dev = z_saved.to(device=dev_obj)
    
    # 计算 z = op_func(x, y)
    print(f"在{device}设备上重新计算结果...")
    # 如果没有提供操作函数，默认使用torch.add
    if op_func is None:
        op_func = torch.add
    z_computed = op_func(x_dev, y_dev)
    
    # 检查 z_saved 和 z_computed 是否相等
    #is_equal = torch.allclose(z_saved_dev, z_computed, rtol=1e-5, atol=1e-5)
    max_diff = torch.max(torch.abs(z_saved_dev - z_computed)).item()
    
    #print(f"验证结果: {'通过' if is_equal else '失败'}")
    print(f"最大差异: {max_diff}")
    
    #return is_equal, max_diff
    return max_diff


import allure

def csv_to_html_report(csv_file_path, title="测试结果", highlight_column=None, highlight_value=None, highlight_color="#e6f7ff"):
    """
    将CSV文件转换为HTML表格并附加到allure报告中
    
    参数:
        csv_file_path: CSV文件路径，第一行为标题行，后续行为数据行，每行是以逗号分隔的字符串
        title: 报告标题
        highlight_column: 用于高亮的列索引（基于0）
        highlight_value: 要高亮的值，如果该列的值等于此值，则该行会被高亮
        highlight_color: 高亮的背景色
    
    返回:
        无
    """
    # 从文件读取CSV数据
    csv_data = []
    try:
        with open(csv_file_path, 'r') as f:
            csv_data = [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 解析CSV数据
    headers = csv_data[0].split(",") if csv_data else []
    
    # 使用列表构建 HTML
    html_parts = []
    
    # HTML 头部
    html_parts.append('<html>')
    html_parts.append('<head>')
    html_parts.append('    <style>')
    html_parts.append('        table {')
    html_parts.append('            border-collapse: collapse;')
    html_parts.append('            width: 100%;')
    html_parts.append('            font-family: Arial, sans-serif;')
    html_parts.append('        }')
    html_parts.append('        th, td {')
    html_parts.append('            border: 1px solid #dddddd;')
    html_parts.append('            text-align: left;')
    html_parts.append('            padding: 8px;')
    html_parts.append('        }')
    html_parts.append('        th {')
    html_parts.append('            background-color: #f2f2f2;')
    html_parts.append('            font-weight: bold;')
    html_parts.append('        }')
    html_parts.append('        tr:nth-child(even) {')
    html_parts.append('            background-color: #f9f9f9;')
    html_parts.append('        }')
    html_parts.append(f'        .highlighted {{')
    html_parts.append(f'            background-color: {highlight_color};')
    html_parts.append('        }')
    html_parts.append('    </style>')
    html_parts.append('</head>')
    html_parts.append('<body>')
    html_parts.append('    <table>')
    html_parts.append('        <thead>')
    html_parts.append('            <tr>')
    
    # 添加表头
    for header in headers:
        html_parts.append(f'                <th>{header}</th>')
    
    html_parts.append('            </tr>')
    html_parts.append('        </thead>')
    html_parts.append('        <tbody>')
    
    # 添加数据行
    for i, line in enumerate(csv_data):
        if i == 0:  # 跳过标题行
            continue
            
        cols = line.split(",")
        
        # 检查是否需要高亮此行
        row_class = ""
        if highlight_column is not None and highlight_value is not None:
            if 0 <= highlight_column < len(cols) and cols[highlight_column] == highlight_value:
                row_class = " class='highlighted'"
        
        html_parts.append(f'            <tr{row_class}>')
        
        for col in cols:
            html_parts.append(f'                <td>{col}</td>')
            
        html_parts.append('            </tr>')
    
    # HTML 尾部
    html_parts.append('        </tbody>')
    html_parts.append('    </table>')
    html_parts.append('</body>')
    html_parts.append('</html>')
    
    # 将列表转换为字符串
    html_table = '\n'.join(html_parts)
    
    # 将HTML表格附加到Allure报告
    allure.attach(
        html_table,
        name=title,
        attachment_type=allure.attachment_type.HTML
    )
    
    # 同时附加原始CSV数据
    csv_content = '\n'.join(csv_data)
    allure.attach(
        csv_content,
        name=f"{title}(CSV)",
        attachment_type=allure.attachment_type.TEXT
    )
    allure.attach.file(
        './loss_png',
        attachment_type=allure.attachment_type.PNG
    )
