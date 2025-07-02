#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Visualization Module

This module contains functions for creating data visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def create_data_visualizations(data_dict, config):
    """
    Create data visualizations for the PDF report
    
    Args:
        data_dict (dict): Dictionary of DataFrames containing test data by category
        config (dict): Configuration dictionary from YAML
        
    Returns:
        list: List of paths to generated chart images
    """
    # Create a directory for charts if it doesn't exist
    charts_dir = config['pdf']['output']['charts_directory']
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir, exist_ok=True)
    
    chart_files = []
    chart_config = config['charts']
    dpi = chart_config['dpi']
    
    # 1. Create pie chart of test results by category
    category_counts = {name: len(df) for name, df in data_dict.items()}
    
    if category_counts:
        pie_config = chart_config['pie_chart']
        fig, ax = plt.subplots(figsize=tuple(pie_config['figsize']))
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
        ax.set_title(pie_config['title'], fontsize=14, fontweight='bold')
        
        # Save the chart
        pie_chart_path = os.path.join(charts_dir, pie_config['filename'])
        plt.tight_layout()
        plt.savefig(pie_chart_path, dpi=dpi, bbox_inches='tight')
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
        status_config = chart_config['status_chart']
        # Prepare data for grouped bar chart
        categories = list(status_data.keys())
        statuses = set()
        for counts in status_data.values():
            statuses.update(counts.index)
        statuses = list(statuses)
        
        # Create figure
        fig, ax = plt.subplots(figsize=tuple(status_config['figsize']))
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
        ax.set_title(status_config['title'], fontsize=14, fontweight='bold')
        ax.set_xticks(x + bar_width * (len(statuses) - 1) / 2)
        ax.set_xticklabels(categories)
        ax.legend()
        
        # Save the chart
        status_chart_path = os.path.join(charts_dir, status_config['filename'])
        plt.tight_layout()
        plt.savefig(status_chart_path, dpi=dpi, bbox_inches='tight')
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
        time_config = chart_config['time_chart']
        # Create box plot for execution times
        fig, ax = plt.subplots(figsize=tuple(time_config['figsize']))
        
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
            ax.set_title(time_config['title'], fontsize=14, fontweight='bold')
            ax.set_ylabel('Time (seconds)')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save the chart
            time_chart_path = os.path.join(charts_dir, time_config['filename'])
            plt.tight_layout()
            plt.savefig(time_chart_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            chart_files.append(time_chart_path)
    
    return chart_files
