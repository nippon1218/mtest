#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os

def run():
    """
    运行测试用例并生成报告
    
    命令行参数:
        --debug: 开启调试模式，会启动allure服务器
    """
    import sys
    # 获取命令行参数
    args = sys.argv[1:]
    
    # 检查是否开启调试模式
    debug_mode = False
    if '--debug' in args:
        debug_mode = True
        args.remove('--debug')
    
    # 基本参数
    pytest_args = [
        '-s',                     # 显示print输出
        '-v',                     # 显示详细信息
        '--alluredir',           # 指定allure结果目录
        './report/tmp',
        "--clean-alluredir"      # 清理已有的结果
    ]
    
    # 处理测试路径
    test_path = 'test_case'  # 默认运行所有测试
    for i, arg in enumerate(args):
        if not arg.startswith('-'):
            test_path = arg
            args.pop(i)
            break
    
    # 添加命令行参数
    pytest_args.extend(args)
    
    # 添加测试路径
    pytest_args.append(test_path)
    
    # 运行测试用例
    pytest.main(pytest_args)
    
    # 如果开启调试模式，启动allure服务器
    if debug_mode:
        os.system("allure serve ./report/tmp -h localhost -p 8280")

if __name__ == '__main__':
    run()
