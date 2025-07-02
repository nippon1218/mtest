#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF Fonts Module

This module registers fonts for PDF generation, including Chinese fonts.
"""

import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 定义字体路径
FONT_DIR = "/usr/share/fonts/truetype"

def register_fonts():
    """
    注册字体，包括中文字体
    
    Returns:
        dict: 字体映射字典
    """
    fonts = {}
    
    # 尝试注册常见的中文字体
    chinese_fonts = [
        # 微软雅黑
        {"name": "MSYaHei", "path": "msyh.ttf", "dirs": ["windows", "WindowsFonts", "chinese"]},
        # 宋体
        {"name": "SimSun", "path": "simsun.ttc", "dirs": ["windows", "WindowsFonts", "chinese"]},
        # 黑体
        {"name": "SimHei", "path": "simhei.ttf", "dirs": ["windows", "WindowsFonts", "chinese"]},
        # 文泉驿微米黑 (Linux常见中文字体)
        {"name": "WenQuanYi", "path": "wqy-microhei.ttc", "dirs": ["wqy", "wenquanyi"]},
        # 思源黑体
        {"name": "SourceHanSans", "path": "SourceHanSansSC-Regular.otf", "dirs": ["adobe", "source-han-sans"]},
        # Noto Sans CJK (Google字体)
        {"name": "NotoSansCJK", "path": "NotoSansCJK-Regular.ttc", "dirs": ["noto", "google-noto"]},
        # DejaVu Sans (Linux常见字体，有限的中文支持)
        {"name": "DejaVuSans", "path": "DejaVuSans.ttf", "dirs": ["dejavu", "ttf-dejavu"]},
        # Droid Sans Fallback (Android字体)
        {"name": "DroidSansFallback", "path": "DroidSansFallback.ttf", "dirs": ["droid", "android"]},
    ]
    
    # 尝试注册字体
    registered = False
    for font in chinese_fonts:
        for dir_name in font["dirs"]:
            potential_path = os.path.join(FONT_DIR, dir_name, font["path"])
            if os.path.exists(potential_path):
                try:
                    pdfmetrics.registerFont(TTFont(font["name"], potential_path))
                    fonts["chinese"] = font["name"]
                    print(f"成功注册中文字体: {font['name']} ({potential_path})")
                    registered = True
                    break
                except Exception as e:
                    print(f"注册字体 {font['name']} 失败: {e}")
        if registered:
            break
    
    # 如果没有找到中文字体，尝试使用系统默认字体
    if not registered:
        # 在Linux系统上查找可能的中文字体
        for root, dirs, files in os.walk(FONT_DIR):
            for file in files:
                if file.endswith(('.ttf', '.ttc', '.otf')) and any(keyword in file.lower() for keyword in ['chinese', 'cjk', 'han', 'zh', 'wqy', 'song', 'hei', 'ming']):
                    font_path = os.path.join(root, file)
                    try:
                        font_name = "ChineseFont"
                        pdfmetrics.registerFont(TTFont(font_name, font_path))
                        fonts["chinese"] = font_name
                        print(f"成功注册中文字体: {font_name} ({font_path})")
                        registered = True
                        break
                    except Exception as e:
                        print(f"注册字体失败: {e}")
            if registered:
                break
    
    # 如果仍然没有找到中文字体，使用Helvetica并警告用户
    if not registered:
        print("警告: 未找到中文字体，中文可能无法正确显示")
        fonts["chinese"] = "Helvetica"
    
    return fonts
