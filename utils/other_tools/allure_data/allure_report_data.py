#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from dataclasses import dataclass

@dataclass
class TestMetrics:
    passed: int = 0
    failed: int = 0
    broken: int = 0
    skipped: int = 0
    total: int = 0
    pass_rate: float = 0.0
    time: float = 0.0

class AllureFileClean:
    def __init__(self):
        self.report_path = "./report/tmp"

    def get_case_count(self) -> TestMetrics:
        """
        获取allure报告中的测试统计数据
        """
        summary_json = os.path.join(self.report_path, "summary.json")
        if not os.path.exists(summary_json):
            return TestMetrics()

        with open(summary_json, "r", encoding="utf-8") as f:
            result = json.load(f)
            
        if not result:
            return TestMetrics()

        metrics = TestMetrics(
            passed=result.get("statistic", {}).get("passed", 0),
            failed=result.get("statistic", {}).get("failed", 0),
            broken=result.get("statistic", {}).get("broken", 0),
            skipped=result.get("statistic", {}).get("skipped", 0),
            total=result.get("statistic", {}).get("total", 0),
            time=result.get("time", {}).get("duration", 0) / 1000  # 转换为秒
        )
        
        if metrics.total > 0:
            metrics.pass_rate = round(metrics.passed / metrics.total * 100, 2)
            
        return metrics

if __name__ == '__main__':
    AllureFileClean().get_case_count()
