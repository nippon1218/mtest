#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量修改torch测试文件导入
替换直接的import torch为从torch_import模块导入
"""

import os
import re
import sys

def update_file(file_path):
    """更新文件中的torch导入"""
    print(f"正在处理文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查文件是否已经使用了torch_import
    if 'from .torch_import import' in content:
        print(f"文件 {file_path} 已经更新，跳过")
        return
    
    # 替换直接import torch语句
    pattern = r'import torch'
    replacement = '# 使用我们的导入辅助模块替代直接导入\nfrom .torch_import import torch, torch_import_failed'
    
    new_content = re.sub(pattern, replacement, content)
    
    # 处理test_dtypes列表，添加条件
    if 'test_dtypes = [' in new_content:
        pattern = r'(test_dtypes = \[.*?\])(\s)'
        replacement = r'\1 if not torch_import_failed else []\2'
        new_content = re.sub(pattern, replacement, new_content, flags=re.DOTALL)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"文件 {file_path} 更新完成")

def main():
    """主函数"""
    # 获取torch目录下所有测试文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    torch_test_dir = os.path.join(base_dir, 'test_case', 'torch')
    
    for file_name in os.listdir(torch_test_dir):
        if file_name.startswith('test_') and file_name.endswith('.py'):
            file_path = os.path.join(torch_test_dir, file_name)
            update_file(file_path)

if __name__ == '__main__':
    main()
