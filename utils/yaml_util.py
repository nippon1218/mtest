#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import os

class YamlUtil:
    def __init__(self):
        pass
    
    def read_yaml(self, yaml_path):
        """
        读取yaml文件数据
        :param yaml_path: yaml文件路径
        :return: 数据
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"找不到YAML文件: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
