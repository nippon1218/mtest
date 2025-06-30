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
import tempfile
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

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
    summary_file = Path("./report/test_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"   摘要报告已保存到: {summary_file}")
    
    # 处理result目录中的CSV文件并生成PDF报告
    process_csv_to_pdf()
    
def process_csv_to_pdf():
    """Organize CSV files from result directory into a single PDF table"""
    print("   Processing CSV files and generating PDF table...")
    
    # Get result directory path
    result_dir = Path("./result")
    if not result_dir.exists():
        print("   Warning: Result directory does not exist")
        return
    
    # Get all subdirectories
    subdirs = [d for d in result_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print("   Warning: No subdirectories found in result directory")
        return
    
    # Dictionary to store dataframes by subdirectory
    all_data = {}
    
    # Process each subdirectory
    for subdir in subdirs:
        # Get all CSV files in the subdirectory
        csv_files = list(subdir.glob("*.csv"))
        if not csv_files:
            print(f"   Warning: No CSV files found in {subdir.name} directory")
            continue
        
        print(f"   Processing {len(csv_files)} CSV files in {subdir.name} directory")
        
        # Read and merge CSV files
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Add filename as source column
                df['Source'] = csv_file.stem
                # Add directory name as category column
                df['Category'] = subdir.name
                dfs.append(df)
            except Exception as e:
                print(f"   Warning: Failed to read {csv_file}: {e}")
        
        if not dfs:
            print(f"   Warning: No valid CSV files in {subdir.name} directory")
            continue
        
        # Merge all dataframes for this subdirectory
        all_data[subdir.name] = pd.concat(dfs, ignore_index=True)
    
    # If no data was collected, return
    if not all_data:
        print("   Warning: No data collected from any subdirectory")
        return
    
    # Generate a single PDF file with all data
    pdf_path = result_dir / "consolidated_test_report.pdf"
    create_consolidated_pdf_report(all_data, pdf_path)
    
    print(f"   PDF report generated: {pdf_path}")

def create_pdf_report(dataframe, output_path, title):
    """Create a professional PDF table report"""
    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Create content list
    elements = []
    
    # Add styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1,  # Center
        spaceAfter=12
    )
    
    # Add title
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"{title.upper()} TEST RESULTS REPORT", title_style))
    elements.append(Paragraph(f"Generated: {timestamp}", styles["Normal"]))
    elements.append(Spacer(1, 0.25*inch))
    
    # Prepare table data
    # Add headers
    table_data = [dataframe.columns.tolist()]
    # Add data rows
    for i, row in dataframe.iterrows():
        table_data.append(row.tolist())
    
    # Create table
    table = Table(table_data, repeatRows=1)
    
    # Add table style
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ])
    
    # Add alternating row colors
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            table_style.add('BACKGROUND', (0, i), (-1, i), colors.lightgrey)
    
    table.setStyle(table_style)
    elements.append(table)
    
    # Add footer
    elements.append(Spacer(1, 0.5*inch))
    footer_text = f"This report was automatically generated by the Test Automation System - {title} Results Summary"
    elements.append(Paragraph(footer_text, styles["Italic"]))
    
    # Build PDF document
    doc.build(elements)

def create_data_visualizations(data_dict, report_dir):
    """Create data visualizations for the PDF report"""
    # Create a directory for charts if it doesn't exist
    charts_dir = Path(report_dir) / "charts"
    charts_dir.mkdir(exist_ok=True, parents=True)
    
    chart_files = []
    
    # 1. Create pie chart of test results by category
    category_counts = {name: len(df) for name, df in data_dict.items()}
    
    if category_counts:
        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            category_counts.values(), 
            labels=category_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            explode=[0.05] * len(category_counts),
            colors=plt.cm.Paired(np.linspace(0, 1, len(category_counts)))
        )
        
        # Style the chart
        plt.setp(autotexts, size=10, weight="bold")
        ax.set_title('Test Distribution by Category', fontsize=14, fontweight='bold')
        
        # Save the chart
        pie_chart_path = charts_dir / "category_distribution.png"
        plt.tight_layout()
        plt.savefig(pie_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_files.append(pie_chart_path)
    
    # 2. Create bar chart of pass/fail counts if status column exists
    status_data = {}
    for category, df in data_dict.items():
        # Check if there's a status or result column
        status_col = None
        for col in df.columns:
            if col.lower() in ['status', 'result', 'outcome', 'pass/fail', 'passed']:
                status_col = col
                break
        
        if status_col:
            status_counts = df[status_col].value_counts()
            status_data[category] = status_counts
    
    if status_data:
        # Prepare data for grouped bar chart
        categories = list(status_data.keys())
        statuses = set()
        for counts in status_data.values():
            statuses.update(counts.index)
        statuses = list(statuses)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.8 / len(statuses)
        opacity = 0.8
        
        for i, status in enumerate(statuses):
            counts = [status_data[cat].get(status, 0) for cat in categories]
            x = np.arange(len(categories))
            rects = ax.bar(x + i*bar_width, counts, bar_width,
                           alpha=opacity, label=status)
            
            # Add count labels on top of bars
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate(f'{height}',
                               xy=(rect.get_x() + rect.get_width()/2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom')
        
        # Add labels and legend
        ax.set_xlabel('Test Categories')
        ax.set_ylabel('Number of Tests')
        ax.set_title('Test Results by Category and Status', fontsize=14, fontweight='bold')
        ax.set_xticks(x + bar_width * (len(statuses) - 1) / 2)
        ax.set_xticklabels(categories)
        ax.legend()
        
        # Save the chart
        status_chart_path = charts_dir / "status_by_category.png"
        plt.tight_layout()
        plt.savefig(status_chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_files.append(status_chart_path)
    
    # 3. Create execution time chart if time data exists
    time_data = {}
    for category, df in data_dict.items():
        # Check if there's a time or duration column
        time_col = None
        for col in df.columns:
            if any(t in col.lower() for t in ['time', 'duration', 'elapsed', 'execution']):
                time_col = col
                break
        
        if time_col and pd.api.types.is_numeric_dtype(df[time_col]):
            time_data[category] = df[time_col].describe()
    
    if time_data:
        # Create box plot for execution times
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for box plot
        box_data = []
        labels = []
        for category, df in data_dict.items():
            for col in df.columns:
                if any(t in col.lower() for t in ['time', 'duration', 'elapsed', 'execution']):
                    if pd.api.types.is_numeric_dtype(df[col]):
                        box_data.append(df[col].values)
                        labels.append(category)
                        break
        
        if box_data:
            ax.boxplot(box_data, labels=labels, patch_artist=True)
            ax.set_title('Test Execution Time Distribution by Category', fontsize=14, fontweight='bold')
            ax.set_ylabel('Time (seconds)')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save the chart
            time_chart_path = charts_dir / "execution_time.png"
            plt.tight_layout()
            plt.savefig(time_chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            chart_files.append(time_chart_path)
    
    return chart_files

def create_consolidated_pdf_report(data_dict, output_path):
    """Create a professional consolidated PDF report with tables from all subdirectories"""
    # Import additional ReportLab components for advanced features
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Image, Paragraph, Table, TableStyle
    from reportlab.platypus.tableofcontents import TableOfContents
    from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate, NextPageTemplate
    from reportlab.platypus.frames import Frame
    from reportlab.pdfgen.canvas import Canvas
    
    # Generate data visualizations
    report_dir = Path(output_path).parent
    chart_files = create_data_visualizations(data_dict, report_dir)
    
    # Define page size and margins
    page_width, page_height = A4
    margin_left = margin_right = 72
    margin_top = margin_bottom = 72
    content_width = page_width - margin_left - margin_right
    content_height = page_height - margin_top - margin_bottom
    
    # Create a custom document template with header and footer
    class PDFDocTemplate(BaseDocTemplate):
        def __init__(self, filename, **kw):
            self.allowSplitting = 0
            BaseDocTemplate.__init__(self, filename, **kw)
            self.pageinfo = "Test Results Report"
            
        def afterFlowable(self, flowable):
            """Register TOC entries and bookmarks"""
            if flowable.__class__.__name__ == 'Paragraph':
                text = flowable.getPlainText()
                style = flowable.style.name
                if style == 'Heading1':
                    self.canv.bookmarkPage(text)
                    self.canv.addOutlineEntry(text, text, 0, 0)
                    self.notify('TOCEntry', (0, text, self.page, text))
                elif style == 'Heading2':
                    self.canv.bookmarkPage(text)
                    self.canv.addOutlineEntry(text, text, 1, 0)
                    self.notify('TOCEntry', (1, text, self.page, text))
    
    # Create the document template
    doc = PDFDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=margin_right,
        leftMargin=margin_left,
        topMargin=margin_top,
        bottomMargin=margin_bottom
    )
    
    # Define header and footer functions
    def header(canvas, doc):
        canvas.saveState()
        # Draw a header line
        canvas.setStrokeColor(colors.darkblue)
        canvas.setLineWidth(1)
        canvas.line(margin_left, page_height - 40, page_width - margin_right, page_height - 40)
        
        # Add header text
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(margin_left, page_height - 30, "Test Automation System")
        
        # Add date on the right
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(page_width - margin_right, page_height - 30, 
                             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        canvas.restoreState()
    
    def footer(canvas, doc):
        canvas.saveState()
        # Draw a footer line
        canvas.setStrokeColor(colors.darkblue)
        canvas.setLineWidth(1)
        canvas.line(margin_left, 50, page_width - margin_right, 50)
        
        # Add page number
        canvas.setFont("Helvetica", 9)
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.drawRightString(page_width - margin_right, 30, text)
        
        # Add footer text
        canvas.setFont("Helvetica", 8)  # Use regular Helvetica instead of Italic
        canvas.drawString(margin_left, 30, "This report was automatically generated by the Test Automation System")
        canvas.restoreState()
    
    # Create frames and page templates
    frame = Frame(margin_left, margin_bottom, content_width, content_height, id='normal')
    
    # Create page templates with header and footer
    main_template = PageTemplate(id='main', frames=[frame], onPage=lambda canvas, doc: (header(canvas, doc), footer(canvas, doc)))
    
    # Add page templates to document
    doc.addPageTemplates([main_template])
    
    # Create content list
    elements = []
    
    # Add styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=1,  # Center
        spaceAfter=12
    )
    heading1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=10
    )
    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8
    )
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        fontName='Helvetica'
    )
    
    # Add cover page
    elements.append(Paragraph("CONSOLIDATED TEST RESULTS REPORT", title_style))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated: {timestamp}", normal_style))
    elements.append(Spacer(1, 0.5*inch))
    
    # Add summary information
    total_tests = sum(len(df) for df in data_dict.values())
    elements.append(Paragraph(f"Total Test Cases: {total_tests}", normal_style))
    elements.append(Paragraph(f"Test Categories: {len(data_dict)}", normal_style))
    elements.append(Spacer(1, 0.5*inch))
    
    # Add data visualizations
    if chart_files:
        elements.append(Paragraph("TEST RESULTS VISUALIZATION", heading1_style))
        elements.append(Spacer(1, 0.25*inch))
        
        for chart_file in chart_files:
            # Add chart description
            chart_name = chart_file.stem
            if "category_distribution" in chart_name:
                elements.append(Paragraph("Test Distribution by Category", heading2_style))
                elements.append(Paragraph("The following chart shows the distribution of tests across different categories:", normal_style))
            elif "status_by_category" in chart_name:
                elements.append(Paragraph("Test Results by Status and Category", heading2_style))
                elements.append(Paragraph("The following chart shows the pass/fail distribution for each test category:", normal_style))
            elif "execution_time" in chart_name:
                elements.append(Paragraph("Test Execution Time Analysis", heading2_style))
                elements.append(Paragraph("The following chart shows the execution time distribution for tests in each category:", normal_style))
            
            # Add the chart image
            img = Image(str(chart_file), width=450, height=300)
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 0.25*inch))
        
        elements.append(PageBreak())
    
    # Add automatic Table of Contents
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(name='TOCHeading1', fontSize=14, leading=16, fontName='Helvetica-Bold'),
        ParagraphStyle(name='TOCHeading2', fontSize=12, leading=14, leftIndent=20, fontName='Helvetica')
    ]
    
    elements.append(Paragraph("TABLE OF CONTENTS", heading1_style))
    elements.append(toc)
    elements.append(PageBreak())
    
    # Add section for each category
    elements.append(Paragraph("TEST RESULTS BY CATEGORY", heading1_style))
    
    # Process each section
    for section_name, dataframe in data_dict.items():
        # Add section header as a bookmark for TOC
        elements.append(Paragraph(f"{section_name.upper()}", heading2_style))
        elements.append(Paragraph(f"Number of tests: {len(dataframe)}", normal_style))
        
        # Prepare table data
        table_data = [dataframe.columns.tolist()]
        for i, row in dataframe.iterrows():
            table_data.append(row.tolist())
        
        # Create table
        table = Table(table_data, repeatRows=1)
        
        # Add table style
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ])
        
        # Add alternating row colors
        for i in range(1, len(table_data)):
            if i % 2 == 0:
                table_style.add('BACKGROUND', (0, i), (-1, i), colors.lightgrey)
        
        table.setStyle(table_style)
        elements.append(table)
        elements.append(Spacer(1, 0.25*inch))
        
        # Add page break between sections
        if section_name != list(data_dict.keys())[-1]:
            elements.append(PageBreak())
    
    # Build PDF document with automatic bookmarks and TOC
    doc.multiBuild(elements)

def process_test_reports(report_dir):
    """Process test reports"""
    print("\n2. Processing test reports...")
    
    report_path = Path(report_dir)
    if not report_path.exists():
        print("   Warning: Report directory does not exist")
        return
    
    # Count report files
    json_files = list(report_path.glob("*.json"))
    print(f"   Found {len(json_files)} JSON report files")
    
    # Additional report processing logic can be added here
    # For example: Parse allure reports, generate HTML summaries, etc.
    
    # Backup reports to timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"./report/backup/report_{timestamp}")
    
    if json_files:
        backup_dir.mkdir(parents=True, exist_ok=True)
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
    notification_log = Path("./report/notifications.log")
    notification_log.parent.mkdir(parents=True, exist_ok=True)
    
    with open(notification_log, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "SUCCESS" if exit_status == 0 else "FAILED"
        f.write(f"[{timestamp}] {status} - Total:{tests_total}, Passed:{tests_passed}, Failed:{tests_failed}\n")
    
    print(f"   Notification log written to: {notification_log}")

def cleanup_temp_files():
    """Clean up temporary files"""
    print("\n4. Cleaning up temporary files...")
    
    # Clean pytest cache
    cache_dir = Path(".pytest_cache")
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print("   Pytest cache cleaned")
        except Exception as e:
            print(f"   Failed to clean pytest cache: {e}")
    
    # Clean Python cache
    pycache_dirs = list(Path(".").rglob("__pycache__"))
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
