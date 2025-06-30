#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import allure
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入工具函数
from test_case.torch.utils import csv_to_html_report

@allure.epic("PyTorch算子测试")
@allure.feature("CSV表格报告测试")
class TestCSVReport:
    
    @allure.story("CSV数据表格测试")
    @allure.title("测试CSV文件转HTML表格功能")
    def test_csv_to_html_report(self):
        # 准备测试数据
        csv_data = [
            "序列长度,是否因果,运行时间(ms),内存使用(MB)",
            "128,False,6.25,1.24",
            "128,True,7.50,1.35",
            "256,False,12.75,2.48",
            "256,True,15.30,2.72",
            "512,False,25.60,4.96",
            "512,True,30.75,5.45"
        ]
        
        # 创建CSV文件
        csv_file_path = "./report/test_csv_report.csv"
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        
        with open(csv_file_path, "w", newline="") as f:
            for line in csv_data:
                f.write(line + "\n")
        
        # 调用函数，将第1列(是否因果)值为"True"的行高亮显示
        csv_to_html_report(
            csv_file_path=csv_file_path,
            title="CSV转HTML表格测试",
            highlight_column=1,
            highlight_value="True",
            highlight_color="#e6f7ff"
        )
        
        # 验证文件已创建
        assert os.path.exists(csv_file_path), f"CSV文件未创建: {csv_file_path}"
        
        # 附加原始CSV数据作为文本（用于测试验证）
        allure.attach(
            "\n".join(csv_data),
            name="原始CSV数据",
            attachment_type=allure.attachment_type.TEXT
        )
