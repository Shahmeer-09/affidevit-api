"""
Affidavit Express - Notification Service

Email notification service for request status updates.
Uses HTML templates for professional emails with plain text fallbacks.
Also includes unified dispatcher for multi-channel notifications (email + SMS/WhatsApp).
"""

import logging
from django.conf import settings
from django.core.mail import send_mail, EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags

from ..constants.notification_messages import (
    APPOINTMENT_MESSAGES,
    REQUEST_STATUS_MESSAGES,
    COMMISSIONER_BALANCE_MESSAGES,
    COMMISSIONER_ACCOUNT_MESSAGES,
    USER_ACCOUNT_MESSAGES,
    EMAIL_SUBJECTS,
)

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

View and download your affidavit here:
{context.get('site_url', '')}/request/{context.get('request_id', '')}

Thank you for using Affidavit Express! Need another affidavit? Visit us anytime.

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

    elif 'commissioner_approved' in template_name:
        return f"""
Hello {context.get('commissioner_name', 'Commissioner')},

Congratulations! Your Affidavit Express commissioner account has been approved.

You can now log in and start accepting affidavit requests:
{context.get('site_url', '')}/login

Welcome to the team!

Best regards,
The Affidavit Express Team
"""

    elif 'user_welcome' in template_name:
        return f"""
Hello {context.get('user_name', 'there')},

Welcome to Affidavit Express! Your account is now active.

Start your first affidavit request here:
{context.get('site_url', '')}/affidavit-types

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


def send_commissioner_approved_notification(commissioner, temp_password: str = None) -> dict:
    """
    Send welcome email + SMS to a commissioner when an admin approves their account.
    Includes temporary password so the commissioner can log in immediately.

    Args:
        commissioner: User model instance (role=COMMISSIONER)
        temp_password: Temporary password set by the approval flow

    Returns:
        dict: {'email': result, 'sms': result}
    """
    results = {'email': None, 'sms': None}

    commissioner_name = commissioner.get_full_name() or commissioner.username
    reset_link = f"{SITE_URL}/reset-password?email={commissioner.email}" if commissioner.email else f"{SITE_URL}/reset-password"

    # --- Email ---
    if commissioner.email:
        context = {
            'commissioner_name': commissioner_name,
            'email': commissioner.email,
            'temp_password': temp_password or '',
            'reset_link': reset_link,
            'site_url': SITE_URL,
        }
        email_result = send_email_with_template(
            subject=EMAIL_SUBJECTS["COMMISSIONER_APPROVED"],
            template_name='commissioner_approved.html',
            context=context,
            recipient_email=commissioner.email,
        )
        results['email'] = email_result
        if email_result['success']:
            logger.info(f"Sent commissioner approval email to {commissioner.email}")
        else:
            logger.error(f"Failed to send commissioner approval email: {email_result.get('error')}")
    else:
        results['email'] = {'success': False, 'error': 'Commissioner has no email address'}

    # --- SMS ---
    if commissioner.phone_number:
        from .twilio_service import TwilioService

        if temp_password:
            sms_message = COMMISSIONER_ACCOUNT_MESSAGES["ACCOUNT_APPROVED_WITH_PASSWORD"].format(
                commissioner_name=commissioner_name,
                temp_password=temp_password,
                reset_link=reset_link,
                site_url=SITE_URL,
            )
        else:
            sms_message = COMMISSIONER_ACCOUNT_MESSAGES["ACCOUNT_APPROVED"].format(
                commissioner_name=commissioner_name,
                site_url=SITE_URL,
            )
        sms_result = TwilioService.send_notification_message(
            commissioner.phone_number, sms_message
        )
        results['sms'] = sms_result
        if sms_result.get('success'):
            logger.info(f"Sent commissioner approval SMS to {commissioner.phone_number}")
        else:
            logger.warning(f"Failed to send commissioner approval SMS: {sms_result.get('error')}")
    else:
        results['sms'] = {'success': False, 'error': 'Commissioner has no phone number'}

    return results

def send_appointment_booked_notifications(request_obj, slot_obj) -> dict:
    """
    Send notifications to both user and commissioner when appointment is booked.
    
    Args:
        request_obj: Request model instance
        slot_obj: CommissionerSlot model instance
        
    Returns:
        dict: {'user': result, 'commissioner': result}
    """
    from .twilio_service import TwilioService
    
    results = {'user': None, 'commissioner': None}
    
    user = request_obj.user
    commissioner = slot_obj.commissioner
    
    # Format slot datetime
    slot_date = slot_obj.start_time.strftime('%B %d, %Y')
    slot_time = slot_obj.start_time.strftime('%I:%M %p')
    
    # Notify user
    if user.phone_number:
        user_message = APPOINTMENT_MESSAGES["BOOKED_USER"].format(
            commissioner_name=commissioner.get_full_name() or commissioner.username,
            slot_date=slot_date,
            slot_time=slot_time,
            request_code=request_obj.request_code
        )
        results['user'] = TwilioService.send_notification_message(
            user.phone_number, user_message
        )
        logger.info(f"Sent appointment booked notification to user {user.username}")
    
    # Notify commissioner
    if commissioner.phone_number:
        commissioner_message = APPOINTMENT_MESSAGES["BOOKED_COMMISSIONER"].format(
            user_name=user.get_full_name() or user.username,
            slot_date=slot_date,
            slot_time=slot_time,
            affidavit_type=request_obj.affidavit_type.name,
            request_code=request_obj.request_code
        )
        results['commissioner'] = TwilioService.send_notification_message(
            commissioner.phone_number, commissioner_message
        )
        logger.info(f"Sent appointment booked notification to commissioner {commissioner.username}")
    
    return results


def send_user_welcome_notification(user, temp_password: str = None) -> dict:
    """
    Send welcome email + SMS to a newly registered user after their OTP is verified.
    Uses welcome_account.html which shows a temp password section if provided.

    Args:
        user: User model instance (role=PUBLIC)
        temp_password: Optional temporary password (shown in email if provided)

    Returns:
        dict: {'email': result, 'sms': result}
    """
    results = {'email': None, 'sms': None}

    user_name = user.get_full_name() or user.first_name or user.username
    reset_link = f"{SITE_URL}/reset-password?email={user.email}" if user.email else f"{SITE_URL}/reset-password"

    # --- Email ---
    if user.email:
        context = {
            'user_name': user_name,
            'email': user.email,
            'phone_number': user.phone_number or '',
            'temp_password': temp_password or '',
            'reset_link': reset_link,
            'site_url': SITE_URL,
        }
        email_result = send_email_with_template(
            subject="Welcome to Affidavit Express — Your Account is Active",
            template_name='welcome_account.html',
            context=context,
            recipient_email=user.email,
        )
        results['email'] = email_result
        if email_result['success']:
            logger.info(f"Sent user welcome email to {user.email}")
        else:
            logger.error(f"Failed to send user welcome email: {email_result.get('error')}")
    else:
        results['email'] = {'success': False, 'error': 'User has no email address'}

    # --- SMS ---
    if user.phone_number:
        from .twilio_service import TwilioService

        if temp_password:
            sms_message = USER_ACCOUNT_MESSAGES["ACCOUNT_CREATED_WITH_PASSWORD"].format(
                user_name=user_name,
                temp_password=temp_password,
                reset_link=reset_link,
                site_url=SITE_URL,
            )
        else:
            sms_message = USER_ACCOUNT_MESSAGES["ACCOUNT_CREATED"].format(
                user_name=user_name,
                site_url=SITE_URL,
            )
        sms_result = TwilioService.send_notification_message(user.phone_number, sms_message)
        results['sms'] = sms_result
        if sms_result.get('success'):
            logger.info(f"Sent user welcome SMS to {user.phone_number}")
        else:
            logger.warning(f"Failed to send user welcome SMS: {sms_result.get('error')}")
    else:
        results['sms'] = {'success': False, 'error': 'User has no phone number'}

    return results


def send_completion_sms(request_obj, stamp=None) -> dict:
    """
    Send SMS to user when their affidavit is completed by the commissioner.

    Args:
        request_obj: Request model instance
        stamp: Stamp model instance (optional, for commissioner name)

    Returns:
        dict: {'success': bool, 'error': str or None}
    """
    from .twilio_service import TwilioService

    user = request_obj.user

    if not user.phone_number:
        return {'success': False, 'error': 'User has no phone number'}

    commissioner_name = ''
    if stamp and stamp.commissioner:
        commissioner_name = stamp.commissioner.get_full_name() or stamp.commissioner.username
    elif request_obj.commissioner:
        commissioner_name = request_obj.commissioner.get_full_name() or request_obj.commissioner.username

    message = USER_ACCOUNT_MESSAGES["AFFIDAVIT_COMPLETED"].format(
        user_name=user.get_full_name() or user.first_name or user.username,
        request_code=request_obj.request_code,
        commissioner_name=commissioner_name or 'your commissioner',
        site_url=SITE_URL,
        request_id=request_obj.id,
    )

    result = TwilioService.send_notification_message(user.phone_number, message)
    if result.get('success'):
        logger.info(f"Sent completion SMS to user {user.username} for {request_obj.request_code}")
    else:
        logger.warning(f"Failed to send completion SMS: {result.get('error')}")
    return result


def send_appointment_accepted_notification(request_obj, slot_obj) -> dict:
    """
    Send notification to user when commissioner accepts appointment.
    """
    from .twilio_service import TwilioService
    
    user = request_obj.user
    commissioner = slot_obj.commissioner
    
    slot_date = slot_obj.start_time.strftime('%B %d, %Y')
    slot_time = slot_obj.start_time.strftime('%I:%M %p')
    
    if not user.phone_number:
        return {'success': False, 'error': 'User has no phone number'}
    
    message = APPOINTMENT_MESSAGES["ACCEPTED_USER"].format(
        request_code=request_obj.request_code,
        commissioner_name=commissioner.get_full_name() or commissioner.username,
        slot_date=slot_date,
        slot_time=slot_time
    )
    
    result = TwilioService.send_notification_message(user.phone_number, message)
    logger.info(f"Sent appointment accepted notification to user {user.username}")
    return result


def send_appointment_rejected_notification(request_obj) -> dict:
    """
    Send notification to user when commissioner rejects appointment.
    """
    from .twilio_service import TwilioService
    
    user = request_obj.user
    
    if not user.phone_number:
        return {'success': False, 'error': 'User has no phone number'}
    
    message = APPOINTMENT_MESSAGES["REJECTED_USER"].format(
        request_code=request_obj.request_code
    )
    
    result = TwilioService.send_notification_message(user.phone_number, message)
    logger.info(f"Sent appointment rejected notification to user {user.username}")
    return result


def send_appointment_cancelled_by_commissioner_notification(request_obj) -> dict:
    """
    Send notification to user when commissioner cancels appointment.
    """
    from .twilio_service import TwilioService
    
    user = request_obj.user
    
    if not user.phone_number:
        return {'success': False, 'error': 'User has no phone number'}
    
    message = APPOINTMENT_MESSAGES["CANCELLED_BY_COMMISSIONER_USER"].format(
        request_code=request_obj.request_code
    )
    
    result = TwilioService.send_notification_message(user.phone_number, message)
    logger.info(f"Sent appointment cancelled notification to user {user.username}")
    return result


def send_user_withdrawn_notification(request_obj, slot_obj, commissioner) -> dict:
    """
    Send notification to commissioner when user withdraws appointment.
    """
    from .twilio_service import TwilioService
    
    user = request_obj.user
    
    if not commissioner.phone_number:
        return {'success': False, 'error': 'Commissioner has no phone number'}
    
    slot_date = slot_obj.start_time.strftime('%B %d, %Y') if slot_obj else 'N/A'
    slot_time = slot_obj.start_time.strftime('%I:%M %p') if slot_obj else 'N/A'
    
    message = APPOINTMENT_MESSAGES["WITHDRAWN_COMMISSIONER"].format(
        user_name=user.get_full_name() or user.username,
        request_code=request_obj.request_code,
        slot_date=slot_date,
        slot_time=slot_time
    )
    
    result = TwilioService.send_notification_message(commissioner.phone_number, message)
    logger.info(f"Sent user withdrawn notification to commissioner {commissioner.username}")
    return result


def send_request_approved_sms(request_obj) -> dict:
    """
    Send SMS/WhatsApp notification to user when request is approved.
    """
    from .twilio_service import TwilioService
    
    user = request_obj.user
    
    if not user.phone_number:
        return {'success': False, 'error': 'User has no phone number'}
    
    message = REQUEST_STATUS_MESSAGES["APPROVED_USER"].format(
        request_code=request_obj.request_code
    )
    
    result = TwilioService.send_notification_message(user.phone_number, message)
    logger.info(f"Sent approval SMS notification to user {user.username}")
    return result


def send_request_rejected_notification(request_obj, reason: str) -> dict:
    """
    Send notification to user when request is rejected.
    """
    from .twilio_service import TwilioService
    
    user = request_obj.user
    
    if not user.phone_number:
        return {'success': False, 'error': 'User has no phone number'}
    
    message = REQUEST_STATUS_MESSAGES["REJECTED_USER"].format(
        request_code=request_obj.request_code,
        rejection_reason=reason[:100] if reason else 'Not specified'
    )
    
    result = TwilioService.send_notification_message(user.phone_number, message)
    logger.info(f"Sent rejection notification to user {user.username}")
    return result


def send_commissioner_payout_added_message(commissioner, amount, request_code) -> str:
    """
    Generate message for commissioner about payout added to pending balance.
    Returns the message string (for use in API response).
    """
    message = COMMISSIONER_BALANCE_MESSAGES["PAYOUT_ADDED"].format(
        amount=amount,
        request_code=request_code
    )
    return message


def send_request_in_review_notification(request_obj) -> dict:
    """
    Send notification to user when request enters review.
    """
    from .twilio_service import TwilioService
    
    user = request_obj.user
    
    if not user.phone_number:
        return {'success': False, 'error': 'User has no phone number'}
    
    message = REQUEST_STATUS_MESSAGES["ENTERED_REVIEW_USER"].format(
        request_code=request_obj.request_code
    )
    
    result = TwilioService.send_notification_message(user.phone_number, message)
    logger.info(f"Sent in-review notification to user {user.username}")
    return result


def send_review_queue_notification_to_reviewers(request_obj) -> dict:
    """
    Send notification to all active reviewers when a request enters review queue.
    """
    from .twilio_service import TwilioService
    from ..models import User

    reviewers = User.objects.filter(
        role=User.Role.REVIEWER,
        is_active=True,
    ).exclude(phone_number__isnull=True).exclude(phone_number='')

    message = REQUEST_STATUS_MESSAGES["ENTERED_REVIEW_REVIEWER"].format(
        request_code=request_obj.request_code,
        affidavit_type=request_obj.affidavit_type.name,
    )

    sent_count = 0
    failed_reviewers = []

    for reviewer in reviewers:
        result = TwilioService.send_notification_message(reviewer.phone_number, message)
        if result.get('success'):
            sent_count += 1
        else:
            failed_reviewers.append(reviewer.username)

    logger.info(
        "Sent review-queue notifications for %s to %s reviewers (failed=%s)",
        request_obj.request_code,
        sent_count,
        len(failed_reviewers),
    )

    return {
        'success': sent_count > 0,
        'sent_count': sent_count,
        'failed_reviewers': failed_reviewers,
    }

