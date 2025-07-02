#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF Styles Module

This module contains functions for creating PDF styles.
"""

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from .pdf_fonts import register_fonts


def create_pdf_styles(config):
    """
    Create PDF styles based on configuration
    
    Args:
        config (dict): Configuration dictionary from YAML
        
    Returns:
        dict: Dictionary of styles
    """
    # 注册并获取字体
    fonts = register_fonts()
    chinese_font = fonts.get("chinese", "Helvetica")
    
    styles = getSampleStyleSheet()
    style_config = config['pdf']['style']
    
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=style_config['title_font_size'],
        alignment=1,  # Center
        spaceAfter=12,
        fontName=chinese_font
    )
    
    heading1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=style_config['heading1_font_size'],
        spaceAfter=10,
        fontName=chinese_font
    )
    
    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=style_config['heading2_font_size'],
        spaceAfter=8,
        fontName=chinese_font
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=style_config['normal_font_size'],
        spaceAfter=6,
        fontName=chinese_font
    )
    
    normal_bold_style = ParagraphStyle(
        'NormalBold',
        parent=styles['Normal'],
        fontSize=style_config['normal_font_size'],
        spaceAfter=6,
        fontName=chinese_font,
        fontWeight='bold'
    )
    
    # Add TOC styles
    styles.add(ParagraphStyle(
        name='TOCHeading1', 
        fontSize=14, 
        leading=16, 
        fontName=chinese_font,
        fontWeight='bold'
    ))
    
    styles.add(ParagraphStyle(
        name='TOCHeading2', 
        fontSize=12, 
        leading=14, 
        leftIndent=20, 
        fontName=chinese_font
    ))
    
    return {
        'title': title_style,
        'heading1': heading1_style,
        'heading2': heading2_style,
        'normal': normal_style,
        'normal_bold': normal_bold_style,
        'base': styles
    }
