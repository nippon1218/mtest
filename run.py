#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os

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
    
    ## 生成allure报告
    #os.system("allure generate ./report/tmp -o ./report/html --clean")
    #
    ## 启动allure报告服务
    #os.system("allure serve ./report/tmp -h localhost -p 8280")

if __name__ == '__main__':
    run()
