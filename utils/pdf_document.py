#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF Document Module

This module contains classes and functions for PDF document generation.
"""

from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus.doctemplate import BaseDocTemplate


class PDFDocTemplate(BaseDocTemplate):
    """Custom document template with header and footer"""
    def __init__(self, filename, config, **kw):
        self.allowSplitting = 0
        self.config = config
        page_config = config['pdf']['page']
        BaseDocTemplate.__init__(self, filename, 
                                pagesize=A4,
                                rightMargin=page_config['margin_right'],
                                leftMargin=page_config['margin_left'],
                                topMargin=page_config['margin_top'],
                                bottomMargin=page_config['margin_bottom'],
                                **kw)
        self.pageinfo = config['report']['header_text']
        
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


def header_footer_func(config):
    """Create header and footer functions"""
    page_config = config['pdf']['page']
    report_config = config['report']
    margin_left = page_config['margin_left']
    margin_right = page_config['margin_right']
    page_width, page_height = A4
    
    def header(canvas, doc):
        canvas.saveState()
        # Draw a header line
        canvas.setStrokeColor(colors.darkblue)
        canvas.setLineWidth(1)
        canvas.line(margin_left, page_height - 40, page_width - margin_right, page_height - 40)
        
        # Add header text
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(margin_left, page_height - 30, report_config['header_text'])
        
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
        canvas.drawString(margin_left, 30, report_config['footer_text'])
        canvas.restoreState()
    
    return header, footer
