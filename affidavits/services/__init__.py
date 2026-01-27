"""
Affidavit Express - Services Package

This package contains service modules for:
- ai_service: OpenAI-powered drafting and QA
- pdf_service: ReportLab PDF generation
- notification_service: Email notifications
"""

from .ai_service import draft_affidavit, qa_check, process_request
from .pdf_service import generate_affidavit_pdf
from .notification_service import send_approval_notification

__all__ = [
    'draft_affidavit',
    'qa_check', 
    'process_request',
    'generate_affidavit_pdf',
    'send_approval_notification',
]
