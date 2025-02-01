#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os
import psutil
import allure

@allure.epic("系统测试")
@allure.feature("系统资源测试")
class TestSystemResources:
    
    @allure.story("内存测试")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_memory_check(self, load_test_data):
        """
        测试系统内存是否满足要求
        """
        with allure.step("加载测试数据"):
            test_data = load_test_data('system_test.yaml')['system_info']['memory_check']
        
        with allure.step("检查系统内存"):
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)  # 转换为GB
            
            allure.attach(
                f"""
                内存信息:
                总内存: {total_gb:.2f}GB
                使用率: {memory.percent}%
                可用内存: {memory.available / (1024**3):.2f}GB
                """,
                "内存详情",
                allure.attachment_type.TEXT
            )
            
            assert total_gb >= test_data['min_memory_gb'], f"系统内存不足: {total_gb:.2f}GB < {test_data['min_memory_gb']}GB"

    @allure.story("磁盘测试")
    @allure.severity(allure.severity_level.BLOCKER)  
    def test_disk_space(self, load_test_data):
        """
        测试磁盘空间是否满足要求
        """
        with allure.step("加载测试数据"):
            test_data = load_test_data('system_test.yaml')['system_info']['disk_check']
        
        with allure.step(f"检查磁盘空间 {test_data['mount_point']}"):
            disk = psutil.disk_usage(test_data['mount_point'])
            total_gb = disk.total / (1024**3)  # 转换为GB
            
            allure.attach(
                f"""
                磁盘信息:
                总空间: {total_gb:.2f}GB
                使用率: {disk.percent}%
                可用空间: {disk.free / (1024**3):.2f}GB
                """,
                "磁盘详情",
                allure.attachment_type.TEXT
            )
            
            assert total_gb >= test_data['min_disk_gb'], f"磁盘空间不足: {total_gb:.2f}GB < {test_data['min_disk_gb']}GB"

    @allure.story("进程测试")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.parametrize("process_info", ["sshd", "systemd", "fake_process"])
    def test_process_status(self, load_test_data, process_info):
        """
        测试关键进程是否正在运行
        """
        with allure.step("加载测试数据"):
            test_data = load_test_data('system_test.yaml')['process_check']['critical_processes']
            process_data = next(item for item in test_data if item['name'] == process_info)
        
        with allure.step(f"检查进程 {process_info}"):
            def is_process_running(name):
                for proc in psutil.process_iter(['name']):
                    try:
                        if name.lower() in proc.info['name'].lower():
                            return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                return False
            
            is_running = is_process_running(process_data['name'])
            allure.attach(
                f"进程 {process_data['name']} {'正在运行' if is_running else '未运行'}",
                "进程状态",
                allure.attachment_type.TEXT
            )
            
            assert is_running == process_data['should_run'], \
                f"进程 {process_data['name']} {'应该运行但未运行' if process_data['should_run'] else '不应该运行但正在运行'}"
