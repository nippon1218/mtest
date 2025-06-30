# 测试后处理脚本使用说明

## 功能概述

本项目现在支持在所有pytest测试执行完成后自动执行自定义脚本，用于执行后处理任务，如：
- 生成测试摘要报告
- 发送测试结果通知
- 清理临时文件
- 备份测试报告
- 集成CI/CD流程

## 使用方法

### 1. 使用默认脚本

如果项目根目录存在 `post_test_script.py` 文件，系统会自动执行该脚本：

```bash
# 运行测试，会自动执行默认后处理脚本
python run.py

# 或直接使用pytest
pytest test_case/
```

### 2. 指定自定义脚本

可以通过 `--post-script` 参数指定自定义的后处理脚本：

```bash
# 使用run.py指定脚本
python run.py --post-script /path/to/your/script.py

# 直接使用pytest指定脚本
pytest test_case/ --post-script /path/to/your/script.py
```

### 3. 结合其他参数使用

```bash
# 指定设备、脚本和调试模式
python run.py --device cuda --post-script ./my_script.py --debug

# 运行特定测试文件并执行后处理
python run.py test_case/torch/test_add.py --post-script ./cleanup.py
```

## 环境变量

后处理脚本可以通过以下环境变量获取测试结果信息：

| 环境变量 | 说明 | 示例值 |
|---------|------|--------|
| `PYTEST_EXIT_STATUS` | pytest的退出状态码 | `0` (成功) 或 `1` (失败) |
| `PYTEST_TESTS_TOTAL` | 总测试数量 | `25` |
| `PYTEST_TESTS_FAILED` | 失败测试数量 | `2` |
| `PYTEST_TESTS_PASSED` | 通过测试数量 | `23` |
| `PYTEST_REPORT_DIR` | 测试报告目录 | `./report/tmp` |

## 默认脚本功能

项目提供的默认脚本 `post_test_script.py` 包含以下功能：

### 1. 生成测试摘要报告
- 创建JSON格式的测试摘要文件 `./report/test_summary.json`
- 包含测试时间、结果统计、成功率等信息

### 2. 处理测试报告
- 统计allure报告文件
- 备份报告到带时间戳的目录 `./report/backup/report_YYYYMMDD_HHMMSS/`

### 3. 发送通知
- 记录通知日志到 `./report/notifications.log`
- 可扩展集成邮件、Slack、钉钉等通知方式

### 4. 清理临时文件
- 清理pytest缓存 `.pytest_cache`
- 清理Python缓存 `__pycache__`

## 自定义脚本开发

### 基本模板

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime

def main():
    # 获取测试结果信息
    exit_status = int(os.environ.get('PYTEST_EXIT_STATUS', '0'))
    tests_total = int(os.environ.get('PYTEST_TESTS_TOTAL', '0'))
    tests_failed = int(os.environ.get('PYTEST_TESTS_FAILED', '0'))
    tests_passed = int(os.environ.get('PYTEST_TESTS_PASSED', '0'))
    report_dir = os.environ.get('PYTEST_REPORT_DIR', './report/tmp')
    
    print(f"测试完成！总计: {tests_total}, 通过: {tests_passed}, 失败: {tests_failed}")
    
    # 在这里添加你的后处理逻辑
    if exit_status == 0:
        print("所有测试通过，执行成功后处理...")
        # 成功时的处理逻辑
    else:
        print("存在测试失败，执行失败后处理...")
        # 失败时的处理逻辑

if __name__ == "__main__":
    main()
```

### 注意事项

1. **超时限制**: 脚本执行时间限制为5分钟，超时会被强制终止
2. **错误处理**: 脚本执行失败不会影响pytest的退出状态
3. **权限要求**: 确保脚本有执行权限
4. **路径问题**: 脚本中的相对路径基于pytest执行目录

## 常见用例

### 1. CI/CD集成

```python
# ci_integration.py
import os
import requests

def main():
    exit_status = int(os.environ.get('PYTEST_EXIT_STATUS', '0'))
    
    # 发送结果到CI系统
    if exit_status == 0:
        # 触发部署流程
        trigger_deployment()
    else:
        # 发送失败通知
        send_failure_notification()

def trigger_deployment():
    # 触发部署的逻辑
    pass

def send_failure_notification():
    # 发送失败通知的逻辑
    pass
```

### 2. 报告生成

```python
# report_generator.py
import json
from pathlib import Path

def main():
    # 生成HTML测试报告
    generate_html_report()
    
    # 上传报告到服务器
    upload_report()

def generate_html_report():
    # 生成HTML报告的逻辑
    pass
```

### 3. 数据库记录

```python
# db_logger.py
import sqlite3
from datetime import datetime

def main():
    # 将测试结果记录到数据库
    log_test_results()

def log_test_results():
    # 数据库记录逻辑
    pass
```

## 故障排除

### 脚本不执行
1. 检查脚本路径是否正确
2. 确认脚本有执行权限
3. 查看pytest输出中的错误信息

### 脚本执行失败
1. 检查脚本语法错误
2. 确认依赖包已安装
3. 查看错误输出信息

### 环境变量获取失败
1. 确认使用正确的环境变量名
2. 提供默认值处理异常情况

## 更新日志

- **v1.0**: 初始版本，支持基本的后处理脚本功能
- 添加了pytest_sessionfinish钩子函数
- 提供了默认的后处理脚本示例
- 支持通过命令行参数指定自定义脚本
