#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Post-test processing script example

This script runs automatically after all pytest tests have completed.
It can retrieve test result information through environment variables and perform appropriate post-processing operations.

Environment variables:
- PYTEST_EXIT_STATUS: pytest exit status code
- PYTEST_TESTS_TOTAL: total number of tests
- PYTEST_TESTS_FAILED: number of failed tests
- PYTEST_TESTS_PASSED: number of passed tests
"""

import os
import sys
import json
import shutil

# 确保脚本可以处理中文字符
import locale
locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
import logging
import subprocess
from datetime import datetime
# pathlib 已被 os 模块替换
import pandas as pd

# Import custom modules
from utils.pdf_generator import create_consolidated_pdf_report, load_config

def main():
    """Main function"""
    print("=" * 60)
    print("Starting post-test script")
    print("=" * 60)
    
    # Get test result information
    exit_status = int(os.environ.get('PYTEST_EXIT_STATUS', '0'))
    tests_total = int(os.environ.get('PYTEST_TESTS_TOTAL', '0'))
    tests_failed = int(os.environ.get('PYTEST_TESTS_FAILED', '0'))
    tests_passed = int(os.environ.get('PYTEST_TESTS_PASSED', '0'))
    report_dir = os.environ.get('PYTEST_REPORT_DIR', './report/tmp')
    
    print(f"Test Results Summary:")
    print(f"  Exit Status: {exit_status}")
    print(f"  Total Tests: {tests_total}")
    print(f"  Passed Tests: {tests_passed}")
    print(f"  Failed Tests: {tests_failed}")
    success_rate = (tests_passed/tests_total*100) if tests_total > 0 else 0
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Report Directory: {report_dir}")
    
    # 1. Generate test summary report
    generate_summary_report(exit_status, tests_total, tests_passed, tests_failed, report_dir)
    
    # 2. Process test reports
    process_test_reports(report_dir)
    
    # 3. Send notifications (example)
    send_notification(exit_status, tests_total, tests_passed, tests_failed)
    
    # 4. Clean up temporary files
    cleanup_temp_files()
    
    print("\nPost-processing script completed!")
    print("=" * 60)

def generate_summary_report(exit_status, tests_total, tests_passed, tests_failed, report_dir):
    """Generate test summary report"""
    print("\n1. Generating test summary report...")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "exit_status": exit_status,
        "tests": {
            "total": tests_total,
            "passed": tests_passed,
            "failed": tests_failed,
            "success_rate": round(tests_passed/tests_total*100, 2) if tests_total > 0 else 0
        },
        "report_directory": report_dir
    }
    
    # 保存到JSON文件
    summary_file = "./report/test_summary.json"
    summary_dir = os.path.dirname(summary_file)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir, exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"   摘要报告已保存到: {summary_file}")
    
    # 处理result目录中的CSV文件并生成PDF报告
    process_csv_to_pdf()

def process_csv_to_pdf():
    """Process CSV files from subdirectories and generate PDF report"""
    print("   Processing CSV files and generating PDF report...")
    
    # Load configuration
    config = load_config()
    
    # Get pytest environment variables
    pytest_info = {
        'exit_status': int(os.environ.get('PYTEST_EXIT_STATUS', '0')),
        'tests_total': int(os.environ.get('PYTEST_TESTS_TOTAL', '0')),
        'tests_failed': int(os.environ.get('PYTEST_TESTS_FAILED', '0')),
        'tests_passed': int(os.environ.get('PYTEST_TESTS_PASSED', '0'))
    }
    
    # Get paths from config
    result_dir = config['directories']['result']
    if not os.path.exists(result_dir) or not os.path.isdir(result_dir):
        print("   Warning: Result directory not found, skipping CSV processing")
        return

    subdirs = [os.path.join(result_dir, d) for d in os.listdir(result_dir) 
              if os.path.isdir(os.path.join(result_dir, d))]
    if not subdirs:
        print("   Warning: No subdirectories found in result directory, skipping CSV processing")
        return

    all_data = {}
    for subdir in subdirs:
        subdir_name = os.path.basename(subdir)
        csv_files = [os.path.join(subdir, f) for f in os.listdir(subdir) 
                   if f.endswith('.csv') and os.path.isfile(os.path.join(subdir, f))]
        if not csv_files:
            print(f"   Warning: No CSV files found in {subdir_name}, skipping")
            continue

        print(f"   Processing {len(csv_files)} CSV files in {subdir_name} directory")
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['Source'] = os.path.splitext(os.path.basename(csv_file))[0]
                df['Category'] = subdir_name
                dfs.append(df)
            except Exception as e:
                print(f"   Warning: Failed to read {csv_file}: {e}")

        if dfs:
            all_data[subdir_name] = pd.concat(dfs, ignore_index=True)
    
    if not all_data:
        print("   Warning: No CSV data found, skipping PDF generation")
        return
    
    # 过滤数据，只保留需要的列（第0、2、3列）
    filtered_data = {}
    for category, df in all_data.items():
        # 获取列名列表
        columns = df.columns.tolist()
        # 如果列数足够，只保留第0、2、3列
        if len(columns) > 3:
            # 创建要保留的列索引列表
            keep_columns = [columns[0], columns[2], columns[3]]
            # 只保留指定列
            filtered_df = df[keep_columns]
            filtered_data[category] = filtered_df
        else:
            # 如果列数不足，保留原始数据
            filtered_data[category] = df
            print(f"   Warning: {category} data has fewer than 4 columns, keeping all columns")
    
    # Generate PDF report using the utility function with filtered data
    pdf_path = create_consolidated_pdf_report(filtered_data, config, pytest_info)
    print(f"   PDF report generated: {pdf_path}")


# Data visualization functions have been moved to utils/visualization.py

# PDF report generation functions have been moved to utils/pdf_generator.py

def process_test_reports(report_dir):
    """Process test reports"""
    print("\n2. Processing test reports...")
    
    if not os.path.exists(report_dir):
        print("   Warning: Report directory does not exist")
        return
    
    # Count report files
    json_files = [os.path.join(report_dir, f) for f in os.listdir(report_dir) 
                if f.endswith('.json') and os.path.isfile(os.path.join(report_dir, f))]
    print(f"   Found {len(json_files)} JSON report files")
    
    # Additional report processing logic can be added here
    # For example: Parse allure reports, generate HTML summaries, etc.
    
    # Backup reports to timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join("./report/backup", f"report_{timestamp}")
    
    if json_files:
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir, exist_ok=True)
        for json_file in json_files:
            shutil.copy2(json_file, backup_dir)
        print(f"   Reports backed up to: {backup_dir}")

def send_notification(exit_status, tests_total, tests_passed, tests_failed):
    """Send notification (example implementation)"""
    print("\n3. Sending notifications...")
    
    # Various notification methods can be integrated here
    # For example: Email, Slack, DingTalk, WeChat Work, etc.
    
    if exit_status == 0:
        print("   ✅ All tests passed, sending success notification")
    else:
        print("   ❌ Some tests failed, sending warning notification")
    
    # Example: Write to notification log
    notification_log = "./report/notifications.log"
    notification_dir = os.path.dirname(notification_log)
    if not os.path.exists(notification_dir):
        os.makedirs(notification_dir, exist_ok=True)
    
    with open(notification_log, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "SUCCESS" if exit_status == 0 else "FAILED"
        f.write(f"[{timestamp}] {status} - Total:{tests_total}, Passed:{tests_passed}, Failed:{tests_failed}\n")
    
    print(f"   Notification log written to: {notification_log}")

def cleanup_temp_files():
    """Clean up temporary files"""
    print("\n4. Cleaning up temporary files...")
    
    # Clean pytest cache
    cache_dir = ".pytest_cache"
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print("   Pytest cache cleaned")
        except Exception as e:
            print(f"   Failed to clean pytest cache: {e}")
    
    # Clean Python cache
    pycache_dirs = []
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_dirs.append(os.path.join(root, dir_name))
    
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
        except Exception:
            pass
    
    if pycache_dirs:
        print(f"   Cleaned {len(pycache_dirs)} Python cache directories")
    
    print("   Temporary file cleanup completed")

if __name__ == "__main__":
    main()
