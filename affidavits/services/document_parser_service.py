"""
Affidavit Express - Document Parser Service

Converts uploaded Word (.docx) and PDF documents to HTML for AI processing.
Uses:
- mammoth: Word to HTML conversion (preserves structure, minimal styling)
- PyPDF2: PDF text extraction
- python-docx: Additional Word document processing if needed
"""

import io
import re
import logging
from datetime import datetime
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)

# Lazy imports to avoid import errors if packages not installed
MAMMOTH_AVAILABLE = False
PYPDF2_AVAILABLE = False
DOCX_AVAILABLE = False

try:
    import mammoth
    MAMMOTH_AVAILABLE = True
except ImportError:
    logger.warning("mammoth not installed. Word to HTML conversion will not work.")

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    logger.warning("PyPDF2 not installed. PDF text extraction will not work.")

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    logger.warning("python-docx not installed. Advanced Word processing will not work.")


def get_file_extension(filename: str) -> str:
    """Extract file extension from filename."""
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower()
    return ''


def convert_docx_to_html(file_bytes: bytes, filename: str = 'document.docx') -> Tuple[bool, str, Optional[str]]:
    """
    Convert a Word document (.docx) to HTML using mammoth.
    
    Args:
        file_bytes: The raw bytes of the Word document
        filename: Original filename for logging
    
    Returns:
        Tuple of (success, html_content, error_message)
    """
    if not MAMMOTH_AVAILABLE:
        return False, '', 'mammoth library not installed'
    
    try:
        # Create file-like object from bytes
        file_obj = io.BytesIO(file_bytes)
        
        # Convert using mammoth
        result = mammoth.convert_to_html(file_obj)
        html_content = result.value
        
        # Log any conversion messages/warnings
        if result.messages:
            for message in result.messages:
                logger.info(f"Mammoth conversion message for {filename}: {message}")
        
        # Clean up the HTML
        html_content = clean_html(html_content)
        
        logger.info(f"Successfully converted {filename} to HTML ({len(html_content)} chars)")
        return True, html_content, None
        
    except Exception as e:
        logger.error(f"Error converting {filename} to HTML: {e}")
        return False, '', str(e)


def extract_text_from_pdf(file_bytes: bytes, filename: str = 'document.pdf') -> Tuple[bool, str, Optional[str]]:
    """
    Extract text from a PDF file using PyPDF2.
    
    Note: This extracts plain text. For preserving structure,
    the text is wrapped in basic HTML.
    
    Args:
        file_bytes: The raw bytes of the PDF document
        filename: Original filename for logging
    
    Returns:
        Tuple of (success, html_content, error_message)
    """
    if not PYPDF2_AVAILABLE:
        return False, '', 'PyPDF2 library not installed'
    
    try:
        # Create file-like object from bytes
        file_obj = io.BytesIO(file_bytes)
        
        # Read PDF
        reader = PdfReader(file_obj)
        
        # Extract text from all pages
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        full_text = '\n\n'.join(text_parts)
        
        # Convert plain text to basic HTML
        html_content = text_to_html(full_text)
        
        logger.info(f"Successfully extracted text from {filename} ({len(reader.pages)} pages, {len(html_content)} chars)")
        return True, html_content, None
        
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        return False, '', str(e)


def text_to_html(text: str) -> str:
    """
    Convert plain text to structured HTML.
    Attempts to detect paragraphs, numbered lists, and headers.
    """
    lines = text.split('\n')
    html_parts = []
    
    in_list = False
    current_paragraph = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            # Empty line - close any open paragraph
            if current_paragraph:
                html_parts.append(f"<p>{' '.join(current_paragraph)}</p>")
                current_paragraph = []
            if in_list:
                html_parts.append("</ol>")
                in_list = False
            continue
        
        # Check for numbered list items (1. 2. 3. etc)
        numbered_match = re.match(r'^(\d+)[\.\)]\s+(.+)', line)
        if numbered_match:
            if not in_list:
                if current_paragraph:
                    html_parts.append(f"<p>{' '.join(current_paragraph)}</p>")
                    current_paragraph = []
                html_parts.append("<ol>")
                in_list = True
            html_parts.append(f"<li>{numbered_match.group(2)}</li>")
            continue
        
        # Check for potential headers (ALL CAPS lines, short lines)
        if line.isupper() and len(line) < 80:
            if current_paragraph:
                html_parts.append(f"<p>{' '.join(current_paragraph)}</p>")
                current_paragraph = []
            if in_list:
                html_parts.append("</ol>")
                in_list = False
            html_parts.append(f"<h2>{line}</h2>")
            continue
        
        # Regular text - add to current paragraph
        current_paragraph.append(line)
    
    # Close any remaining elements
    if current_paragraph:
        html_parts.append(f"<p>{' '.join(current_paragraph)}</p>")
    if in_list:
        html_parts.append("</ol>")
    
    return '\n'.join(html_parts)


def clean_html(html: str) -> str:
    """
    Clean up HTML content - remove empty tags, normalize whitespace.
    """
    # Remove empty paragraphs
    html = re.sub(r'<p>\s*</p>', '', html)
    
    # Remove multiple consecutive whitespaces
    html = re.sub(r'\s+', ' ', html)
    
    # Add newlines after block elements for readability
    html = re.sub(r'(</(?:p|h[1-6]|li|ol|ul|div)>)', r'\1\n', html)
    
    # Trim whitespace
    html = html.strip()
    
    return html


def parse_document(file_bytes: bytes, filename: str) -> Dict:
    """
    Parse a document (Word or PDF) and convert to HTML.
    
    Args:
        file_bytes: The raw bytes of the document
        filename: Original filename (used to determine file type)
    
    Returns:
        dict: {
            'success': bool,
            'html_content': str,
            'filename': str,
            'file_type': str,
            'parsed_at': str (ISO timestamp),
            'error': str (if success is False)
        }
    """
    extension = get_file_extension(filename)
    
    result = {
        'success': False,
        'html_content': '',
        'filename': filename,
        'file_type': extension,
        'parsed_at': datetime.utcnow().isoformat() + 'Z',
        'error': None
    }
    
    if extension in ['docx', 'doc']:
        if extension == 'doc':
            # .doc files need conversion - mammoth only handles .docx
            result['error'] = 'Legacy .doc format not supported. Please convert to .docx'
            return result
        
        success, html, error = convert_docx_to_html(file_bytes, filename)
        result['success'] = success
        result['html_content'] = html
        result['error'] = error
        
    elif extension == 'pdf':
        success, html, error = extract_text_from_pdf(file_bytes, filename)
        result['success'] = success
        result['html_content'] = html
        result['error'] = error
        
    else:
        result['error'] = f'Unsupported file type: .{extension}. Supported: .docx, .pdf'
    
    return result


def parse_multiple_documents(files: List[Tuple[bytes, str]]) -> List[Dict]:
    """
    Parse multiple documents and return their HTML content.
    
    Args:
        files: List of tuples (file_bytes, filename)
    
    Returns:
        List of parse results
    """
    results = []
    for file_bytes, filename in files:
        result = parse_document(file_bytes, filename)
        results.append(result)
    
    return results


def extract_template_fields(html_content: str) -> List[str]:
    """
    Attempt to extract likely template field names from HTML content.
    Looks for patterns like:
    - {{field_name}}
    - [FIELD_NAME]
    - _________ (underlines suggesting blanks)
    - Names/dates that appear to be placeholders
    
    Args:
        html_content: The HTML content to analyze
    
    Returns:
        List of detected field names/patterns
    """
    fields = []
    
    # Find {{mustache}} style placeholders
    mustache_matches = re.findall(r'\{\{(\w+)\}\}', html_content)
    fields.extend(mustache_matches)
    
    # Find [BRACKET] style placeholders
    bracket_matches = re.findall(r'\[([A-Z_]+)\]', html_content)
    fields.extend([m.lower() for m in bracket_matches])
    
    # Find underline patterns (5+ underscores)
    if re.search(r'_{5,}', html_content):
        fields.append('__blank_field__')
    
    return list(set(fields))  # Remove duplicates


def combine_templates_to_master(parsed_documents: List[Dict]) -> str:
    """
    Combine multiple parsed documents into a single reference template.
    Used when admin uploads multiple examples - AI will use this combined
    content to understand the affidavit structure.
    
    Args:
        parsed_documents: List of parse results from parse_multiple_documents
    
    Returns:
        Combined HTML string with document separators
    """
    successful_docs = [doc for doc in parsed_documents if doc['success']]
    
    if not successful_docs:
        return ''
    
    combined_parts = []
    for i, doc in enumerate(successful_docs, 1):
        combined_parts.append(f'<!-- EXAMPLE DOCUMENT {i}: {doc["filename"]} -->')
        combined_parts.append(doc['html_content'])
        combined_parts.append(f'<!-- END DOCUMENT {i} -->')
        combined_parts.append('')
    
    return '\n'.join(combined_parts)
