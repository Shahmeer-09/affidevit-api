"""
Affidavit Express - Notification Service

Email notification service for request status updates.
Uses HTML templates for professional emails with plain text fallbacks.
"""

import logging
from django.conf import settings
from django.core.mail import send_mail, EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags

logger = logging.getLogger(__name__)

# Site URL for links in emails
SITE_URL = getattr(settings, 'SITE_URL', 'http://localhost:3000')


def send_email_with_template(
    subject: str,
    template_name: str,
    context: dict,
    recipient_email: str
) -> dict:
    """
    Send an email using an HTML template with plain text fallback.
    
    Args:
        subject: Email subject line
        template_name: Name of the template file (without path)
        context: Template context dictionary
        recipient_email: Recipient email address
    
    Returns:
        dict: {'success': bool, 'error': str or None}
    """
    try:
        # Add site URL to context
        context['site_url'] = SITE_URL
        
        # Try to render HTML template
        try:
            html_content = render_to_string(f'emails/{template_name}', context)
            plain_content = strip_tags(html_content)
        except Exception as template_error:
            logger.warning(f"Template not found, using plain text: {template_error}")
            html_content = None
            plain_content = _get_fallback_content(template_name, context)
        
        # Create email message
        email = EmailMultiAlternatives(
            subject=subject,
            body=plain_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[recipient_email]
        )
        
        # Attach HTML version if available
        if html_content:
            email.attach_alternative(html_content, "text/html")
        
        # Send the email
        email.send(fail_silently=False)
        
        logger.info(f"Email sent successfully to {recipient_email}: {subject}")
        return {'success': True, 'error': None}
        
    except Exception as e:
        logger.error(f"Failed to send email to {recipient_email}: {e}")
        return {'success': False, 'error': str(e)}


def _get_fallback_content(template_name: str, context: dict) -> str:
    """Generate plain text fallback content when template is not available."""
    
    if 'approval' in template_name:
        return f"""
Hello {context.get('user_name', 'User')},

Great news! Your affidavit request has been approved and is ready for commissioning.

Request Details:
- Reference Code: {context.get('request_code', 'N/A')}
- Affidavit Type: {context.get('affidavit_type', 'N/A')}

What's Next:
1. Visit any affiliated commissioner
2. Provide your reference code: {context.get('request_code', 'N/A')}
3. The commissioner will verify and stamp your document

Important: Please bring valid identification when visiting the commissioner.

Thank you for using Affidavit Express!

Best regards,
The Affidavit Express Team
"""
    
    elif 'clarification' in template_name:
        return f"""
Hello {context.get('user_name', 'User')},

We need a bit more information to complete your affidavit request.

Reference Code: {context.get('request_code', 'N/A')}
Affidavit Type: {context.get('affidavit_type', 'N/A')}

Question:
{context.get('clarification_question', 'Please provide additional information.')}

Please log in to your account and provide the requested information to continue with your request.

Thank you,
The Affidavit Express Team
"""
    
    elif 'completion' in template_name:
        return f"""
Hello {context.get('user_name', 'User')},

Congratulations! Your affidavit has been successfully commissioned!

Request Details:
- Reference Code: {context.get('request_code', 'N/A')}
- Affidavit Type: {context.get('affidavit_type', 'N/A')}
- Completed: {context.get('completed_at', 'N/A')}

Thank you for using Affidavit Express!

Best regards,
The Affidavit Express Team
"""
    
    elif 'ticket_created' in template_name:
        return f"""
Hello {context.get('user_name', 'User')},

We have received your support ticket.

Ticket Details:
- Ticket ID: #{context.get('ticket_id', 'N/A')}
- Subject: {context.get('subject', 'N/A')}
- Status: {context.get('status', 'Open')}

We will get back to you as soon as possible.

Best regards,
The Affidavit Express Team
"""

    elif 'ticket_reply' in template_name:
        return f"""
Hello {context.get('user_name', 'User')},

There is a new update on your ticket #{context.get('ticket_id', 'N/A')}.

Update from {context.get('sender_name', 'Support')}:
"{context.get('message_preview', 'New message received.')}"

Please log in to your dashboard to view the full conversation and reply.

Best regards,
The Affidavit Express Team
"""

    else:
        return f"Thank you for using Affidavit Express. Reference: {context.get('request_code', 'N/A')}"


def send_approval_notification(request_obj) -> dict:
    """
    Send email notification to user when their request is approved.
    
    Args:
        request_obj: Request model instance
    
    Returns:
        dict: {'success': bool, 'error': str or None}
    """
    user = request_obj.user
    
    if not user.email:
        logger.warning(f"No email for user {user.username}, skipping notification")
        return {'success': False, 'error': 'User has no email address'}
    
    context = {
        'user_name': user.get_full_name() or user.username,
        'request_code': request_obj.request_code,
        'request_id': request_obj.id,
        'affidavit_type': request_obj.affidavit_type.name,
        'approved_at': request_obj.approved_at.strftime('%B %d, %Y at %I:%M %p') if request_obj.approved_at else '',
    }
    
    result = send_email_with_template(
        subject=f"Your Affidavit is Ready - {request_obj.request_code}",
        template_name='approval_notification.html',
        context=context,
        recipient_email=user.email
    )
    
    if result['success']:
        logger.info(f"Sent approval notification for {request_obj.request_code} to {user.email}")
    
    return result


def send_clarification_notification(request_obj) -> dict:
    """
    Send email notification when clarification is needed from user.
    
    Args:
        request_obj: Request model instance
    
    Returns:
        dict: {'success': bool, 'error': str or None}
    """
    user = request_obj.user
    
    if not user.email:
        return {'success': False, 'error': 'User has no email address'}
    
    context = {
        'user_name': user.get_full_name() or user.username,
        'request_code': request_obj.request_code,
        'request_id': request_obj.id,
        'affidavit_type': request_obj.affidavit_type.name,
        'clarification_question': request_obj.clarification_question,
    }
    
    result = send_email_with_template(
        subject=f"Action Required - {request_obj.request_code}",
        template_name='clarification_notification.html',
        context=context,
        recipient_email=user.email
    )
    
    if result['success']:
        logger.info(f"Sent clarification notification for {request_obj.request_code}")
    
    return result


def send_completion_notification(request_obj, stamp=None) -> dict:
    """
    Send email notification when request is marked complete by commissioner.
    
    Args:
        request_obj: Request model instance
        stamp: Stamp model instance (optional)
    
    Returns:
        dict: {'success': bool, 'error': str or None}
    """
    user = request_obj.user
    
    if not user.email:
        return {'success': False, 'error': 'User has no email address'}
    
    # Get completion details
    completed_at = ''
    commissioner_name = ''
    
    if stamp:
        completed_at = stamp.stamped_at.strftime('%B %d, %Y at %I:%M %p') if stamp.stamped_at else ''
        if stamp.commissioner:
            commissioner_name = stamp.commissioner.get_full_name() or stamp.commissioner.username
    elif request_obj.completed_at:
        completed_at = request_obj.completed_at.strftime('%B %d, %Y at %I:%M %p')
    
    context = {
        'user_name': user.get_full_name() or user.username,
        'request_code': request_obj.request_code,
        'request_id': request_obj.id,
        'affidavit_type': request_obj.affidavit_type.name,
        'completed_at': completed_at,
        'commissioner_name': commissioner_name,
    }
    
    result = send_email_with_template(
        subject=f"Affidavit Completed - {request_obj.request_code}",
        template_name='completion_notification.html',
        context=context,
        recipient_email=user.email
    )
    
    if result['success']:
        logger.info(f"Sent completion notification for {request_obj.request_code}")
    
    return result


def send_submission_notification(request_obj) -> dict:
    """
    Send email notification when user submits a new request.
    
    Args:
        request_obj: Request model instance
    
    Returns:
        dict: {'success': bool, 'error': str or None}
    """
    user = request_obj.user
    
    if not user.email:
        return {'success': False, 'error': 'User has no email address'}
    
    context = {
        'user_name': user.get_full_name() or user.username,
        'request_code': request_obj.request_code,
        'request_id': request_obj.id,
        'affidavit_type': request_obj.affidavit_type.name,
        'submitted_at': request_obj.submitted_at.strftime('%B %d, %Y at %I:%M %p') if request_obj.submitted_at else '',
    }
    
    # Use a simple message for submission confirmation
    plain_content = f"""
Hello {context['user_name']},

Thank you for submitting your affidavit request!

Request Details:
- Reference Code: {context['request_code']}
- Affidavit Type: {context['affidavit_type']}
- Submitted: {context['submitted_at']}

What happens next:
1. Our system will review your submission
2. You'll receive an email when your document is ready
3. Then visit any affiliated commissioner to complete the process

You can track your request status using your reference code: {context['request_code']}

Thank you for using Affidavit Express!

Best regards,
The Affidavit Express Team
"""
    
    try:
        from django.core.mail import EmailMultiAlternatives
        
        email = EmailMultiAlternatives(
            subject=f"Request Received - {request_obj.request_code}",
            body=plain_content,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[user.email]
        )
        email.send(fail_silently=False)
        
        logger.info(f"Sent submission notification for {request_obj.request_code}")
        return {'success': True, 'error': None}
        
    except Exception as e:
        logger.error(f"Failed to send submission notification: {e}")
        return {'success': False, 'error': str(e)}


def send_ticket_created_notification(ticket_obj) -> dict:
    """
    Send notification to user when a ticket is successfully created.
    """
    user = ticket_obj.user
    
    if not user.email:
        return {'success': False, 'error': 'User has no email address'}
        
    context = {
        'user_name': user.get_full_name() or user.username,
        'ticket_id': ticket_obj.id,
        'subject': ticket_obj.subject,
        'status': ticket_obj.get_status_display(),
    }
    
    result = send_email_with_template(
        subject=f"Support Ticket Created - #{ticket_obj.id}",
        template_name='ticket_created_notification.html',
        context=context,
        recipient_email=user.email
    )
    
    if result['success']:
        logger.info(f"Sent ticket created notification for #{ticket_obj.id} to {user.email}")
        
    return result


def send_ticket_reply_notification(ticket_obj, message_obj) -> dict:
    """
    Notify user of an update or response to their ticket.
    """
    user = ticket_obj.user
    
    if not user.email:
        return {'success': False, 'error': 'User has no email address'}
        
    context = {
        'user_name': user.get_full_name() or user.username,
        'ticket_id': ticket_obj.id,
        'status': ticket_obj.get_status_display(),
        'sender_name': message_obj.sender.get_full_name() or message_obj.sender.username,
        'message_preview': message_obj.message[:100] + ('...' if len(message_obj.message) > 100 else '')
    }
    
    result = send_email_with_template(
        subject=f"Update on your Ticket - #{ticket_obj.id}",
        template_name='ticket_reply_notification.html',
        context=context,
        recipient_email=user.email
    )
    
    if result['success']:
        logger.info(f"Sent ticket reply notification for #{ticket_obj.id} to {user.email}")
        
    return result


def send_otp_email(email: str, otp_code: str, user_name: str = None) -> dict:
    """
    Send OTP verification code via email.
    
    Args:
        email: Recipient email address
        otp_code: 6-digit OTP code
        user_name: User's name for personalization
        
    Returns:
        dict: {'success': bool, 'error': str or None}
    """
    context = {
        'user_name': user_name or 'there',
        'otp_code': otp_code,
    }
    
    result = send_email_with_template(
        subject=f"Your Verification Code: {otp_code}",
        template_name='otp_verification.html',
        context=context,
        recipient_email=email
    )
    
    if result['success']:
        logger.info(f"Sent OTP email to {email}")
    else:
        logger.error(f"Failed to send OTP email to {email}: {result.get('error')}")
        
    return result


def send_welcome_email(
    email: str, 
    temp_password: str, 
    reset_link: str, 
    user_name: str = None,
    phone_number: str = None
) -> dict:
    """
    Send welcome email with temporary password and password reset link.
    
    Args:
        email: Recipient email address
        temp_password: Temporary password
        reset_link: Link to password reset page
        user_name: User's name for personalization
        phone_number: User's phone number (optional)
        
    Returns:
        dict: {'success': bool, 'error': str or None}
    """
    context = {
        'user_name': user_name or 'there',
        'temp_password': temp_password,
        'reset_link': reset_link,
        'email': email,
        'phone_number': phone_number,
    }
    
    result = send_email_with_template(
        subject="Welcome to Affidavit Express - Your Account Details",
        template_name='welcome_account.html',
        context=context,
        recipient_email=email
    )
    
    if result['success']:
        logger.info(f"Sent welcome email to {email}")
    else:
        logger.error(f"Failed to send welcome email to {email}: {result.get('error')}")
        
    return result

