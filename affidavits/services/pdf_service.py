"""
Affidavit Express - PDF Service

PDF generation for affidavit documents.
Supports two rendering methods:
- ReportLab: Programmatic PDF building (original method)
- xhtml2pdf: HTML to PDF conversion (new method for AI-generated HTML)

Creates professional, print-ready PDFs with request codes.
"""

import io
import os
import logging
from datetime import datetime
from django.conf import settings
from django.core.files.base import ContentFile

logger = logging.getLogger(__name__)

# ReportLab imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not installed. PDF generation will not work.")

# xhtml2pdf imports
try:
    from xhtml2pdf import pisa
    XHTML2PDF_AVAILABLE = True
except ImportError:
    XHTML2PDF_AVAILABLE = False
    logger.warning("xhtml2pdf not installed. HTML to PDF conversion will not work.")


def get_custom_styles():
    """Create custom paragraph styles for the affidavit PDF."""
    styles = getSampleStyleSheet()
    
    # Title style
    styles.add(ParagraphStyle(
        name='AffidavitTitle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=20,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    ))
    
    # Subtitle / Request Code
    styles.add(ParagraphStyle(
        name='RequestCode',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_RIGHT,
        textColor=colors.grey,
        spaceAfter=10
    ))
    
    # Body text
    styles.add(ParagraphStyle(
        name='AffidavitBody',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceBefore=6,
        spaceAfter=6,
        leading=14,
        fontName='Helvetica'
    ))
    
    # Section header
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        fontName='Helvetica-Bold',
        spaceBefore=15,
        spaceAfter=8
    ))
    
    # Signature line label
    styles.add(ParagraphStyle(
        name='SignatureLabel',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_LEFT,
        spaceBefore=30
    ))
    
    # Footer
    styles.add(ParagraphStyle(
        name='Footer',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.grey
    ))
    
    return styles


def generate_affidavit_pdf(request_obj, save_to_model=True, commissioner=None) -> dict:
    """
    Generate a PDF for the given affidavit request.
    
    This function intelligently chooses between HTML-based generation (for AI-generated HTML content)
    and ReportLab-based generation (for plain text content).
    
    Args:
        request_obj: Request model instance
        save_to_model: If True, saves the PDF to request_obj.pdf_file
        commissioner: Optional User instance with PDF preferences
    
    Returns:
        dict: {
            'success': bool,
            'pdf_bytes': bytes (the PDF content),
            'filename': str,
            'error': str (if success is False)
        }
    """
    # Get the content - prefer final_text, fallback to draft_text
    content_text = request_obj.final_text or request_obj.draft_text
    
    # Check if content is HTML (AI-generated content is HTML)
    is_html_content = content_text and (
        '<p>' in content_text or 
        '<ol>' in content_text or 
        '<strong>' in content_text or
        '</p>' in content_text
    )
    
    if is_html_content and XHTML2PDF_AVAILABLE:
        # Use HTML-based PDF generation for AI-generated content
        logger.info(f"Using HTML-based PDF generation for {request_obj.request_code}")
        return generate_affidavit_pdf_from_html(
            request_obj=request_obj,
            html_content=content_text,
            save_to_model=save_to_model,
            commissioner=commissioner
        )
    
    # Fallback to ReportLab for plain text content
    if not REPORTLAB_AVAILABLE:
        return {
            'success': False,
            'pdf_bytes': None,
            'filename': None,
            'error': 'ReportLab is not installed'
        }
    
    try:
        # Get commissioner preferences if available
        prefs = {}
        if commissioner and hasattr(commissioner, 'pdf_preferences'):
            prefs = commissioner.pdf_preferences or {}
        
        # Page size from preferences
        page_size_name = prefs.get('page_size', 'letter')
        page_size = A4 if page_size_name == 'a4' else letter
        
        # Signature spacing
        sig_spacing = prefs.get('signature_spacing', 'normal')
        sig_space_map = {'compact': 20, 'normal': 30, 'expanded': 50}
        sig_spacer = sig_space_map.get(sig_spacing, 30)
        
        # Create buffer for PDF
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=page_size,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        styles = get_custom_styles()
        
        # Build document content
        story = []
        
        # Commissioner letterhead (if enabled)
        letterhead = prefs.get('letterhead', {})
        if letterhead.get('enabled') and letterhead.get('text'):
            story.append(Paragraph(
                letterhead['text'].replace('\n', '<br/>'),
                ParagraphStyle(
                    name='Letterhead',
                    parent=styles['Normal'],
                    fontSize=10,
                    alignment=TA_CENTER,
                    spaceAfter=20
                )
            ))
            story.append(HRFlowable(
                width="100%",
                thickness=1,
                color=colors.black,
                spaceBefore=5,
                spaceAfter=15
            ))
        
        # Header with request code
        story.append(Paragraph(
            f"Reference: {request_obj.request_code}",
            styles['RequestCode']
        ))
        
        # Title
        story.append(Paragraph(
            f"AFFIDAVIT",
            styles['AffidavitTitle']
        ))
        
        # Affidavit Type subtitle
        story.append(Paragraph(
            f"({request_obj.affidavit_type.name})",
            styles['AffidavitTitle']
        ))
        
        story.append(Spacer(1, 20))
        
        # Horizontal line
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.black,
            spaceBefore=5,
            spaceAfter=20
        ))
        
        # Main content - use final_text if available, otherwise draft_text
        content_text = request_obj.final_text or request_obj.draft_text
        
        if content_text:
            # Split content into paragraphs and add to story
            paragraphs = content_text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    # Handle section headers (lines that are all caps or end with :)
                    if para.strip().isupper() or para.strip().endswith(':'):
                        story.append(Paragraph(
                            para.strip(),
                            styles['SectionHeader']
                        ))
                    else:
                        story.append(Paragraph(
                            para.strip().replace('\n', '<br/>'),
                            styles['AffidavitBody']
                        ))
        
        story.append(Spacer(1, sig_spacer))
        
        # Signature section
        story.append(HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.grey,
            spaceBefore=20,
            spaceAfter=10
        ))
        
        # Deponent signature block
        story.append(Paragraph("SWORN/AFFIRMED before me at", styles['AffidavitBody']))
        story.append(Spacer(1, 10))
        
        # Create signature table
        sig_data = [
            ['_' * 40, '', '_' * 40],
            ['City/Town', '', 'Province/State'],
            ['', '', ''],
            ['this _____ day of _____________, 20____', '', ''],
            ['', '', ''],
            ['', '', ''],
            ['_' * 40, '', '_' * 40],
            ['Commissioner / Notary Public', '', 'Deponent Signature'],
        ]
        
        # Add commission number if preference enabled
        if prefs.get('show_commission_number') and commissioner:
            commission_num = getattr(commissioner, 'commission_number', '')
            if commission_num:
                sig_data.append(['', '', ''])
                sig_data.append([f'Commission #: {commission_num}', '', ''])
        
        sig_table = Table(sig_data, colWidths=[2.5*inch, 0.5*inch, 2.5*inch])
        sig_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        
        story.append(sig_table)
        
        story.append(Spacer(1, 30))
        
        # Footer with request code and generation date
        story.append(HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.grey,
            spaceBefore=20,
            spaceAfter=5
        ))
        
        # Custom footer or default
        custom_footer = prefs.get('custom_footer', '')
        if custom_footer:
            footer_text = f"{custom_footer} | Reference: {request_obj.request_code}"
        else:
            footer_text = (
                f"Document Reference: {request_obj.request_code} | "
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                f"Affidavit Express"
            )
        story.append(Paragraph(footer_text, styles['Footer']))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        # Generate filename
        filename = f"affidavit_{request_obj.request_code}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        # Save to model if requested
        if save_to_model:
            request_obj.pdf_file.save(
                filename,
                ContentFile(pdf_bytes),
                save=True
            )
        
        logger.info(f"Successfully generated PDF for {request_obj.request_code}")
        
        return {
            'success': True,
            'pdf_bytes': pdf_bytes,
            'filename': filename,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return {
            'success': False,
            'pdf_bytes': None,
            'filename': None,
            'error': str(e)
        }


# =============================================================================
# xhtml2pdf Functions - HTML to PDF Conversion
# =============================================================================

def get_default_pdf_css() -> str:
    """Return default CSS for affidavit PDFs."""
    return """
    @page {
        size: letter;
        margin: 1in;
    }
    
    body {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.4;
        color: #000;
    }
    
    h1 {
        font-size: 16pt;
        text-align: center;
        margin-bottom: 20pt;
        font-weight: bold;
    }
    
    h2 {
        font-size: 12pt;
        margin-top: 15pt;
        margin-bottom: 8pt;
        font-weight: bold;
    }
    
    p {
        text-align: justify;
        margin-bottom: 10pt;
    }
    
    ol {
        margin-left: 20pt;
        margin-bottom: 10pt;
    }
    
    li {
        margin-bottom: 8pt;
    }
    
    .header {
        text-align: center;
        font-weight: bold;
        margin-bottom: 15pt;
    }
    
    .request-code {
        text-align: right;
        font-size: 9pt;
        color: #666;
        margin-bottom: 10pt;
    }
    
    .signature-block {
        margin-top: 40pt;
        page-break-inside: avoid;
    }
    
    .signature-line {
        border-bottom: 1px solid #000;
        width: 250pt;
        margin-top: 30pt;
        margin-bottom: 5pt;
    }
    
    .signature-label {
        font-size: 10pt;
    }
    
    .attestation {
        margin-top: 30pt;
        font-style: italic;
    }
    
    .footer {
        font-size: 8pt;
        text-align: center;
        color: #666;
        margin-top: 30pt;
    }
    
    .draft-watermark {
        text-align: center;
        font-size: 14pt;
        color: red;
        font-weight: bold;
        margin-bottom: 10pt;
    }
    """


def wrap_html_for_pdf(html_content: str, request_code: str = '', is_draft: bool = False) -> str:
    """
    Wrap HTML content with proper structure and CSS for PDF generation.
    
    Args:
        html_content: The HTML content to wrap
        request_code: Optional request code to display
        is_draft: If True, adds draft watermark
    
    Returns:
        Complete HTML document ready for PDF conversion
    """
    css = get_default_pdf_css()
    
    draft_watermark = ""
    if is_draft:
        draft_watermark = '<div class="draft-watermark">*** DRAFT - NOT FOR OFFICIAL USE ***</div>'
    
    request_code_html = ""
    if request_code:
        request_code_html = f'<div class="request-code">Reference: {request_code}</div>'
    
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
    {css}
    </style>
</head>
<body>
    {draft_watermark}
    {request_code_html}
    {html_content}
</body>
</html>"""


def generate_pdf_from_html(
    html_content: str,
    request_code: str = '',
    is_draft: bool = False,
    custom_css: str = ''
) -> dict:
    """
    Generate a PDF from HTML content using xhtml2pdf.
    
    Args:
        html_content: HTML content to convert
        request_code: Optional request code for header
        is_draft: If True, adds draft watermark
        custom_css: Optional custom CSS to append
    
    Returns:
        dict: {
            'success': bool,
            'pdf_bytes': bytes,
            'filename': str,
            'error': str (if success is False)
        }
    """
    if not XHTML2PDF_AVAILABLE:
        return {
            'success': False,
            'pdf_bytes': None,
            'filename': None,
            'error': 'xhtml2pdf is not installed'
        }
    
    try:
        # Wrap HTML with proper structure
        full_html = wrap_html_for_pdf(html_content, request_code, is_draft)
        
        # Add custom CSS if provided
        if custom_css:
            full_html = full_html.replace('</style>', f'{custom_css}</style>')
        
        # Create PDF
        result = io.BytesIO()
        
        # Convert HTML to PDF
        pisa_status = pisa.CreatePDF(
            io.BytesIO(full_html.encode('utf-8')),
            dest=result
        )
        
        if pisa_status.err:
            return {
                'success': False,
                'pdf_bytes': None,
                'filename': None,
                'error': f'PDF generation failed with {pisa_status.err} errors'
            }
        
        pdf_bytes = result.getvalue()
        result.close()
        
        # Generate filename
        prefix = 'draft' if is_draft else 'affidavit'
        code_part = f'_{request_code}' if request_code else ''
        filename = f"{prefix}{code_part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        logger.info(f"Successfully generated PDF from HTML ({len(pdf_bytes)} bytes)")
        
        return {
            'success': True,
            'pdf_bytes': pdf_bytes,
            'filename': filename,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Error generating PDF from HTML: {e}")
        return {
            'success': False,
            'pdf_bytes': None,
            'filename': None,
            'error': str(e)
        }


def generate_affidavit_pdf_from_html(
    request_obj,
    html_content: str,
    save_to_model: bool = True,
    commissioner=None
) -> dict:
    """
    Generate a PDF from HTML content for a specific request.
    
    Args:
        request_obj: Request model instance
        html_content: HTML content of the affidavit
        save_to_model: If True, saves the PDF to request_obj.pdf_file
        commissioner: Optional User instance with PDF preferences
    
    Returns:
        dict with success, pdf_bytes, filename, error
    """
    try:
        # Get commissioner preferences if available
        custom_css = ""
        if commissioner and hasattr(commissioner, 'pdf_preferences'):
            prefs = commissioner.pdf_preferences or {}
            
            # Page size
            page_size = prefs.get('page_size', 'letter')
            if page_size == 'a4':
                custom_css += "@page { size: a4; }\n"
            
            # Custom footer
            if prefs.get('custom_footer'):
                # This would be handled differently in production
                pass
        
        # Generate PDF
        result = generate_pdf_from_html(
            html_content=html_content,
            request_code=request_obj.request_code,
            is_draft=False,
            custom_css=custom_css
        )
        
        if not result['success']:
            return result
        
        # Update filename with request code
        filename = f"affidavit_{request_obj.request_code}_{datetime.now().strftime('%Y%m%d')}.pdf"
        result['filename'] = filename
        
        # Save to model if requested
        if save_to_model and result['pdf_bytes']:
            request_obj.pdf_file.save(
                filename,
                ContentFile(result['pdf_bytes']),
                save=True
            )
        
        logger.info(f"Successfully generated PDF from HTML for {request_obj.request_code}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating affidavit PDF from HTML: {e}")
        return {
            'success': False,
            'pdf_bytes': None,
            'filename': None,
            'error': str(e)
        }


def generate_preview_pdf(draft_text: str, affidavit_type_name: str, request_code: str) -> dict:
    """
    Generate a preview/watermarked PDF for review purposes.
    
    Args:
        draft_text: The draft text to render
        affidavit_type_name: Name of the affidavit type
        request_code: The request code
    
    Returns:
        dict with success, pdf_bytes, filename, error
    """
    if not REPORTLAB_AVAILABLE:
        return {
            'success': False,
            'pdf_bytes': None,
            'filename': None,
            'error': 'ReportLab is not installed'
        }
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        styles = get_custom_styles()
        story = []
        
        # DRAFT watermark notice
        story.append(Paragraph(
            "*** DRAFT - NOT FOR OFFICIAL USE ***",
            ParagraphStyle(
                name='Watermark',
                parent=styles['Normal'],
                fontSize=14,
                alignment=TA_CENTER,
                textColor=colors.red,
                fontName='Helvetica-Bold'
            )
        ))
        
        story.append(Spacer(1, 10))
        
        # Request code
        story.append(Paragraph(
            f"Reference: {request_code}",
            styles['RequestCode']
        ))
        
        # Title
        story.append(Paragraph("AFFIDAVIT", styles['AffidavitTitle']))
        story.append(Paragraph(f"({affidavit_type_name})", styles['AffidavitTitle']))
        
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        story.append(Spacer(1, 20))
        
        # Content
        if draft_text:
            paragraphs = draft_text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    story.append(Paragraph(
                        para.strip().replace('\n', '<br/>'),
                        styles['AffidavitBody']
                    ))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            f"DRAFT - {request_code} - Generated {datetime.now().strftime('%Y-%m-%d')}",
            styles['Footer']
        ))
        
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        filename = f"draft_{request_code}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        return {
            'success': True,
            'pdf_bytes': pdf_bytes,
            'filename': filename,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Error generating preview PDF: {e}")
        return {
            'success': False,
            'pdf_bytes': None,
            'filename': None,
            'error': str(e)
        }
