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
import json
import shutil
import pandas as pd
import glob
from datetime import datetime
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
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

def create_consolidated_pdf_report(data_dict, output_path):
    """Create a professional consolidated PDF report with tables from all subdirectories"""
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
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=14,
        alignment=1,  # Center
        spaceAfter=10
    )
    
    # Add main title
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph("CONSOLIDATED TEST RESULTS REPORT", title_style))
    elements.append(Paragraph(f"Generated: {timestamp}", styles["Normal"]))
    elements.append(Spacer(1, 0.5*inch))
    
    # Add table of contents
    toc_data = [["Section", "Tests", "Page"]]
    page_counter = 1  # Start page counter (approximate)
    
    # Calculate approximate pages for TOC
    for section_name, df in data_dict.items():
        rows = len(df)
        # Rough estimate: 20 rows per page
        pages = max(1, rows // 20)
        toc_data.append([section_name.upper(), str(rows), str(page_counter)])
        page_counter += pages
    
    # Create TOC table
    toc_table = Table(toc_data, repeatRows=1)
    toc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(Paragraph("TABLE OF CONTENTS", subtitle_style))
    elements.append(toc_table)
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("DETAILED TEST RESULTS", subtitle_style))
    
    # Process each section
    for section_name, dataframe in data_dict.items():
        # Add section header
        elements.append(Spacer(1, 0.25*inch))
        elements.append(Paragraph(f"SECTION: {section_name.upper()}", subtitle_style))
        
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
    
    # Add footer
    elements.append(Spacer(1, 0.5*inch))
    footer_text = "This report was automatically generated by the Test Automation System"
    elements.append(Paragraph(footer_text, styles["Italic"]))
    
    # Build PDF document
    doc.build(elements)

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
