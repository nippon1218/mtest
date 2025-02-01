#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os
import platform
import socket
import time
import subprocess
import re
import allure

@allure.epic("系统测试")
@allure.feature("网络连接测试")
class TestNetwork:
    
    @allure.story("基础连通性测试")
    @allure.severity(allure.severity_level.CRITICAL)
    @allure.title("测试百度站点连通性")
    def test_ping_baidu(self, load_test_data):
        """
        测试是否能ping通百度
        """
        with allure.step("加载测试数据"):
            test_data = load_test_data('network_test.yaml')['test_ping_baidu']
        
        with allure.step(f"Ping {test_data['host']}"):
            # 根据操作系统选择ping命令
            if platform.system().lower() == 'windows':
                command = f'ping {test_data["host"]} -n 1'
            else:
                command = f'ping {test_data["host"]} -c 1'
                
            # 执行ping命令
            result = os.system(command)
            
            # 在Linux/Unix中，0表示成功
            assert (result == 0) == test_data['expected_result'], f"Ping {test_data['host']} 失败"

    @allure.story("DNS服务器测试")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.parametrize("dns_server", ["8.8.8.8", "114.114.114.114"])
    def test_dns_connectivity(self, load_test_data, dns_server):
        """
        测试DNS服务器连通性
        """
        with allure.step("加载测试数据"):
            test_data = load_test_data('network_test.yaml')['network_connectivity']['dns_checks']
            server_data = next(item for item in test_data if item['server'] == dns_server)

        with allure.step(f"测试DNS服务器 {dns_server}"):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            try:
                result = sock.connect_ex((server_data['server'], server_data['port'])) == 0
            finally:
                sock.close()
            
            assert result == server_data['expected_result'], f"DNS服务器 {dns_server} 连接失败"

    @allure.story("端口连通性测试")
    @allure.severity(allure.severity_level.NORMAL)
    def test_port_connectivity(self, load_test_data):
        """
        测试重要端口的连通性
        """
        with allure.step("加载测试数据"):
            test_data = load_test_data('network_test.yaml')['network_connectivity']['port_checks']

        for check in test_data:
            with allure.step(f"测试 {check['host']}:{check['port']} 连通性"):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                try:
                    host_ip = socket.gethostbyname(check['host'])
                    result = sock.connect_ex((host_ip, check['port'])) == 0
                finally:
                    sock.close()
                
                assert result == check['expected_result'], f"端口 {check['host']}:{check['port']} 连接失败"

    @allure.story("网络延迟测试")
    @allure.severity(allure.severity_level.NORMAL)
    def test_network_latency(self, load_test_data):
        """
        测试网络延迟
        """
        with allure.step("加载测试数据"):
            test_data = load_test_data('network_test.yaml')['network_connectivity']['latency_checks']

        for check in test_data:
            with allure.step(f"测试 {check['host']} 延迟"):
                if platform.system().lower() == 'windows':
                    command = ['ping', check['host'], '-n', '1']
                else:
                    command = ['ping', check['host'], '-c', '1']
                
                try:
                    # 使用subprocess获取ping输出
                    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        # 从输出中提取时间，支持中英文输出
                        # 尝试多种可能的正则表达式模式
                        patterns = [
                            r'时间[=](\d+\.?\d*)\s*(?:毫秒|ms)',  # 中文格式1
                            r'time[=](\d+\.?\d*)\s*ms',          # 英文格式
                            r'rtt\s+min/avg/max.*\s=\s+[\d.]+/([\d.]+)/[\d.]+/[\d.]+\s+ms'  # RTT统计行
                        ]
                        
                        latency = None
                        for pattern in patterns:
                            match = re.search(pattern, result.stdout)
                            if match:
                                latency = float(match.group(1))
                                break
                        
                        if latency is not None:
                            allure.attach(
                                f"""
                                延迟测试详情:
                                目标主机: {check['host']}
                                实际延迟: {latency:.2f}ms
                                最大允许延迟: {check['max_latency_ms']}ms
                                测试结果: {'通过' if latency <= check['max_latency_ms'] else '失败'}
                                完整输出:
                                {result.stdout}
                                """,
                                "延迟详情",
                                allure.attachment_type.TEXT
                            )
                            
                            assert latency <= check['max_latency_ms'], \
                                f"{check['host']} 延迟过高: {latency:.2f}ms > {check['max_latency_ms']}ms"
                        else:
                            allure.attach(
                                f"""
                                无法解析的ping输出:
                                {result.stdout}
                                """,
                                "Ping输出",
                                allure.attachment_type.TEXT
                            )
                            raise AssertionError(f"无法从ping输出中提取延迟时间，请检查输出格式")
                    else:
                        raise AssertionError(f"Ping {check['host']} 失败: {result.stderr}")
                except subprocess.TimeoutExpired:
                    raise AssertionError(f"Ping {check['host']} 超时")
