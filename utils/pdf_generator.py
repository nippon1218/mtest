#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF Report Generation Utilities

This module contains functions for generating PDF reports with data visualizations.
"""

import tempfile
from datetime import datetime
import os
import yaml
import sys

# 确保正确处理中文字符
import locale
try:
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        print("警告: 无法设置中文区域设置，可能会影响中文显示")

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

from .pdf_styles import create_pdf_styles
from .pdf_document import PDFDocTemplate, header_footer_func
from .visualization import create_data_visualizations


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_consolidated_pdf_report(data_dict, config=None, pytest_info=None):
    """
    Create a professional consolidated PDF report with tables from all subdirectories
    
    Args:
        data_dict (dict): Dictionary of DataFrames containing test data by category
        config (dict): Configuration dictionary from YAML or None to load from default path
        pytest_info (dict): Dictionary containing pytest execution information
        
    Returns:
        str: Path to the generated PDF report
    """
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Import required modules here to avoid circular imports
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.platypus.tableofcontents import TableOfContents
    
    # Get output path from config
    output_path = config['pdf']['output']['consolidated_report']
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate data visualizations
    chart_files = create_data_visualizations(data_dict, config)
    
    # Define page size and margins
    page_config = config['pdf']['page']
    page_width, page_height = A4
    margin_left = page_config['margin_left']
    margin_right = page_config['margin_right']
    margin_top = page_config['margin_top']
    margin_bottom = page_config['margin_bottom']
    content_width = page_width - margin_left - margin_right
    content_height = page_height - margin_top - margin_bottom
    
    # Create the document template
    doc = PDFDocTemplate(output_path, config)
    
    # Get header and footer functions
    header_func, footer_func = header_footer_func(config)
    
    # Create frames and page templates
    from reportlab.platypus.frames import Frame
    from reportlab.platypus.doctemplate import PageTemplate
    
    frame = Frame(margin_left, margin_bottom, content_width, content_height, id='normal')
    
    # Create page templates with header and footer
    main_template = PageTemplate(id='main', frames=[frame], 
                                onPage=lambda canvas, doc: (header_func(canvas, doc), footer_func(canvas, doc)))
    
    # Add page templates to document
    doc.addPageTemplates([main_template])
    
    # Create content list
    elements = []
    
    # Get styles
    styles = create_pdf_styles(config)
    report_config = config['report']
    
    # Add cover page
    elements.append(Paragraph(report_config['title'], styles['title']))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated: {timestamp}", styles['normal']))
    elements.append(Spacer(1, 0.5*inch))
    
    # Add summary information
    total_tests = sum(len(df) for df in data_dict.values())
    elements.append(Paragraph(f"Total Test Cases: {total_tests}", styles['normal']))
    elements.append(Paragraph(f"Test Categories: {len(data_dict)}", styles['normal']))
    
    # Add pytest execution information if available
    if pytest_info:
        elements.append(Spacer(1, 0.25*inch))
        # Use normal style instead of heading to avoid outline/bookmark issues
        elements.append(Paragraph("Pytest Execution Summary", styles['normal_bold']))
        
        # Create a table for pytest information
        pytest_data = [
            ["Metric", "Value"],
            ["Exit Status", str(pytest_info.get('exit_status', 'N/A'))],
            ["Total Tests", str(pytest_info.get('tests_total', 'N/A'))],
            ["Passed Tests", str(pytest_info.get('tests_passed', 'N/A'))],
            ["Failed Tests", str(pytest_info.get('tests_failed', 'N/A'))]
        ]
        
        # Calculate success rate if possible
        tests_total = pytest_info.get('tests_total', 0)
        tests_passed = pytest_info.get('tests_passed', 0)
        if tests_total > 0:
            success_rate = f"{(tests_passed/tests_total*100):.1f}%"
            pytest_data.append(["Success Rate", success_rate])
        
        # Create and style the table
        pytest_table = Table(pytest_data, colWidths=[2*inch, 1.5*inch])
        pytest_table_style = TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            
            # Content styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),   # Left align first column
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),  # Center align second column
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            
            # Grid styling
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
        ])
        
        # Add color to highlight pass/fail status
        if 'tests_failed' in pytest_info and pytest_info['tests_failed'] > 0:
            pytest_table_style.add('TEXTCOLOR', (1, 4), (1, 4), colors.red)  # Failed tests in red
        if 'tests_passed' in pytest_info:
            pytest_table_style.add('TEXTCOLOR', (1, 3), (1, 3), colors.green)  # Passed tests in green
        
        pytest_table.setStyle(pytest_table_style)
        elements.append(pytest_table)
    
    elements.append(Spacer(1, 0.5*inch))
    
    # Add data visualizations
    if chart_files:
        elements.append(Paragraph(report_config['visualization_title'], styles['heading1']))
        elements.append(Spacer(1, 0.25*inch))
        
        for chart_file in chart_files:
            # Add chart description
            chart_name = os.path.splitext(os.path.basename(chart_file))[0]
            if "category_distribution" in chart_name:
                elements.append(Paragraph("Test Distribution by Category", styles['heading2']))
                elements.append(Paragraph("The following chart shows the distribution of tests across different categories:", styles['normal']))
            elif "status_by_category" in chart_name:
                elements.append(Paragraph("Test Results by Status and Category", styles['heading2']))
                elements.append(Paragraph("The following chart shows the pass/fail distribution for each test category:", styles['normal']))
            elif "execution_time" in chart_name:
                elements.append(Paragraph("Test Execution Time Analysis", styles['heading2']))
                elements.append(Paragraph("The following chart shows the execution time distribution for tests in each category:", styles['normal']))
            
            # Add the chart image
            img = Image(chart_file, width=450, height=300)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 0.25*inch))
        
        elements.append(PageBreak())
    
    # Add automatic Table of Contents
    toc = TableOfContents()
    toc.levelStyles = [
        styles['base']['TOCHeading1'],
        styles['base']['TOCHeading2']
    ]
    
    elements.append(Paragraph(report_config['toc_title'], styles['heading1']))
    elements.append(toc)
    elements.append(PageBreak())
    
    # Add section for each category
    elements.append(Paragraph(report_config['subtitle'], styles['heading1']))
    
    # Get style configuration
    style_config = config['pdf']['style']
    
    # Process each section
    for section_name, dataframe in data_dict.items():
        # 获取目录映射配置
        directory_mappings = report_config.get('directory_mappings', {})
        
        # 获取当前目录的标题和描述，如果没有配置则使用默认值
        dir_config = directory_mappings.get(section_name.lower(), directory_mappings.get('default', {}))
        
        # 使用配置的标题，如果没有则使用目录名
        section_title = dir_config.get('title', f"{section_name.upper()}")
        section_description = dir_config.get('description', f"Number of tests: {len(dataframe)}")
        
        # 添加标题和描述
        elements.append(Paragraph(section_title, styles['heading2']))
        elements.append(Paragraph(f"{section_description}", styles['normal']))
        elements.append(Paragraph(f"Number of tests: {len(dataframe)}", styles['normal']))
        
        # Prepare table data
        table_data = [dataframe.columns.tolist()]
        for i, row in dataframe.iterrows():
            table_data.append(row.tolist())
        
        # Create table
        table = Table(table_data, repeatRows=1)
        
        # Add table style with academic look
        table_style = TableStyle([
            # Header styling - more subtle for academic look
            ('BACKGROUND', (0, 0), (-1, 0), getattr(colors, style_config['table_header_color'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), getattr(colors, style_config['table_header_text_color'])),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),  # Center align headers only
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),  # Times for more academic look
            ('FONTSIZE', (0, 0), (-1, 0), 11),  # Slightly smaller for academic style
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            
            # Content styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),  # Clean white background
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),  # Left align data for readability
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),  # Times for academic look
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Grid styling - more subtle
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.grey),  # Lighter inner grid
            ('BOX', (0, 0), (-1, -1), 0.5, colors.black),  # Slightly heavier outer border
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),  # Emphasize header separation
        ])
        
        # Add subtle alternating row colors for readability
        for i in range(1, len(table_data)):
            if i % 2 == 0:
                table_style.add('BACKGROUND', (0, i), (-1, i), colors.whitesmoke)  # Very light gray
        
        table.setStyle(table_style)
        elements.append(table)
        elements.append(Spacer(1, 0.25*inch))
        
        # Add page break between sections
        if section_name != list(data_dict.keys())[-1]:
            elements.append(PageBreak())
    
    # Build PDF document with automatic bookmarks and TOC
    doc.multiBuild(elements)
    
    return output_path
