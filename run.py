#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os
from utils.other_tools.allure_data.allure_report_data import AllureFileClean

def run():
    """
    运行测试用例并生成报告
    """
    import sys
    # 获取命令行参数
    args = sys.argv[1:]
    
    # 基本参数
    pytest_args = [
        '-s',                     # 显示print输出
        '-v',                     # 显示详细信息
        '--alluredir',           # 指定allure结果目录
        './report/tmp',
        "--clean-alluredir"      # 清理已有的结果
    ]
    
    # 添加命令行参数
    pytest_args.extend(args)
    
    # 运行测试用例
    pytest.main(pytest_args)
    
    # 生成allure报告
    os.system("allure generate ./report/tmp -o ./report/html --clean")
    
    # 显示测试统计信息
    allure_data = AllureFileClean().get_case_count()
    print(allure_data)
    
    # 启动allure报告服务
    # os.system("allure serve ./report/tmp -h 127.0.0.1 -p 8000")

if __name__ == '__main__':
    run()
