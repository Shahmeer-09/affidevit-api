"""
Twilio Service for WhatsApp and SMS messaging.

Supports direct messaging (not Verify API) for OTP and notifications.
Tries WhatsApp first, then falls back to SMS if WhatsApp fails.
"""

import os
from twilio.rest import Client
import logging

logger = logging.getLogger(__name__)


class TwilioService:
    @staticmethod
    def _normalize_phone(value: str) -> str:
        """Normalize phone/env values by removing an optional whatsapp: prefix."""
        normalized = (value or '').strip()
        if normalized.lower().startswith('whatsapp:'):
            normalized = normalized.split(':', 1)[1].strip()
        return normalized

    @staticmethod
    def _whatsapp_address(value: str) -> str:
        """Return a Twilio-compatible whatsapp:<E164> address."""
        return f"whatsapp:{TwilioService._normalize_phone(value)}"

    @staticmethod
    def get_client():
        """Get Twilio client instance from a single env source of truth."""
        try:
            use_test = os.getenv('TWILIO_USE_TEST_CREDENTIALS', 'false').lower() == 'true'
            account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            
            if not account_sid or not auth_token:
                logger.warning("Twilio credentials not found in environment variables")
                return None

            if use_test:
                logger.info("Using Twilio TEST mode with unified credentials env keys")
                
            return Client(account_sid, auth_token)
        except Exception as e:
            logger.error(f"Error initializing Twilio client: {e}")
            return None

    @staticmethod
    def send_otp_message(phone_number: str, otp_code: str) -> dict:
        """
        Send OTP via WhatsApp first, fallback to SMS if WhatsApp fails.

        You can change the preferred first channel by setting:
        - TWILIO_PREFERRED_CHANNEL=sms  (tries SMS first, then WhatsApp)
        - TWILIO_PREFERRED_CHANNEL=whatsapp (tries WhatsApp first, then SMS)
        
        Args:
            phone_number: Phone number with country code (e.g., +923086989618)
            otp_code: 6-digit OTP code
            
        Returns:
            dict: {'success': bool, 'channel': 'whatsapp'|'sms'|None, 'error': str|None}
        """
        client = TwilioService.get_client()
        if not client:
            return {'success': False, 'channel': None, 'error': 'Twilio client not configured'}
        
        use_test = os.getenv('TWILIO_USE_TEST_CREDENTIALS', 'false').lower() == 'true'
        preferred = (os.getenv('TWILIO_PREFERRED_CHANNEL') or 'whatsapp').strip().lower()
        
        whatsapp_number = TwilioService._normalize_phone(
            os.getenv('TWILIO_WHATSAPP_NUMBER', '+14155238886')
        )
        sms_number = '+15005550006' if use_test else os.getenv('TWILIO_SMS_NUMBER')
        
        message_body = f"Your Affidavit Express verification code is: {otp_code}\n\nThis code expires in 10 minutes. Do not share this code with anyone."

        if (os.getenv('TWILIO_DEV_MODE') or '').strip().lower() == 'true':
            logger.info(
                f"[TWILIO_DEV_MODE] OTP to={phone_number} preferred={preferred} body={message_body}"
            )
            return {
                'success': True,
                'channel': 'dev_mode',
                'error': None,
                'sid': 'dev_mode_mock_sid',
            }

        def _try_whatsapp() -> dict:
            whatsapp_from = TwilioService._whatsapp_address(whatsapp_number)
            whatsapp_to = TwilioService._whatsapp_address(phone_number)
            logger.info(
                f"Attempting to send OTP to {phone_number} via WhatsApp... from={whatsapp_from} to={whatsapp_to}"
            )
            message = client.messages.create(
                from_=whatsapp_from,
                body=message_body,
                to=whatsapp_to
            )
            logger.info(f"WhatsApp OTP sent successfully to {phone_number}, SID: {message.sid}, Status: {message.status}")
            return {'success': True, 'channel': 'whatsapp', 'error': None, 'sid': message.sid}

        def _try_sms() -> dict:
            if not sms_number:
                return {'success': False, 'channel': 'sms', 'error': 'SMS number not configured'}
            logger.info(f"Attempting to send OTP to {phone_number} via SMS...")
            message = client.messages.create(
                from_=sms_number,
                body=message_body,
                to=phone_number
            )
            logger.info(f"SMS OTP sent successfully to {phone_number}, SID: {message.sid}, Status: {message.status}")
            return {'success': True, 'channel': 'sms', 'error': None, 'sid': message.sid}

        first, second = (_try_sms, _try_whatsapp) if preferred == 'sms' else (_try_whatsapp, _try_sms)

        try:
            return first()
        except Exception as first_error:
            logger.warning(f"Primary channel '{preferred}' failed for {phone_number}: {first_error}")

        try:
            return second()
        except Exception as second_error:
            logger.error(f"Both channels failed for {phone_number}: {second_error}")
            return {'success': False, 'channel': None, 'error': str(second_error)}

    @staticmethod
    def send_welcome_message(phone_number: str, temp_password: str, reset_link: str) -> dict:
        """
        Send welcome message with temporary password and password reset link.
        Tries WhatsApp first, fallback to SMS.

        You can change the preferred first channel by setting:
        - TWILIO_PREFERRED_CHANNEL=sms
        - TWILIO_PREFERRED_CHANNEL=whatsapp
        
        Args:
            phone_number: Phone number with country code
            temp_password: Temporary password for the account
            reset_link: Link to password reset page
            
        Returns:
            dict: {'success': bool, 'channel': 'whatsapp'|'sms'|None, 'error': str|None}
        """
        client = TwilioService.get_client()
        if not client:
            return {'success': False, 'channel': None, 'error': 'Twilio client not configured'}
        
        use_test = os.getenv('TWILIO_USE_TEST_CREDENTIALS', 'false').lower() == 'true'
        preferred = (os.getenv('TWILIO_PREFERRED_CHANNEL') or 'whatsapp').strip().lower()
        
        whatsapp_number = os.getenv('TWILIO_WHATSAPP_NUMBER', '+14155238886')
        sms_number = '+15005550006' if use_test else os.getenv('TWILIO_SMS_NUMBER')
        
        message_body = f"""Welcome to Affidavit Express!

Your account has been created successfully.

Temporary Password: {temp_password}

Please change your password using this link:
{reset_link}

For security, we recommend changing your password immediately."""

        if (os.getenv('TWILIO_DEV_MODE') or '').strip().lower() == 'true':
            logger.info(
                f"[TWILIO_DEV_MODE] WELCOME to={phone_number} preferred={preferred} body={message_body}"
            )
            return {
                'success': True,
                'channel': 'dev_mode',
                'error': None,
                'sid': 'dev_mode_mock_sid',
            }
        
        def _try_whatsapp() -> dict:
            whatsapp_from = TwilioService._whatsapp_address(whatsapp_number)
            whatsapp_to = TwilioService._whatsapp_address(phone_number)
            logger.info(
                f"Sending welcome message to {phone_number} via WhatsApp... from={whatsapp_from} to={whatsapp_to}"
            )
            message = client.messages.create(
                from_=whatsapp_from,
                body=message_body,
                to=whatsapp_to
            )
            logger.info(f"WhatsApp welcome sent to {phone_number}, SID: {message.sid}")
            return {'success': True, 'channel': 'whatsapp', 'error': None, 'sid': message.sid}

        def _try_sms() -> dict:
            if not sms_number:
                return {'success': False, 'channel': 'sms', 'error': 'SMS number not configured'}
            logger.info(f"Sending welcome message to {phone_number} via SMS...")
            message = client.messages.create(
                from_=sms_number,
                body=message_body,
                to=phone_number
            )
            logger.info(f"SMS welcome sent to {phone_number}, SID: {message.sid}")
            return {'success': True, 'channel': 'sms', 'error': None, 'sid': message.sid}

        first, second = (_try_sms, _try_whatsapp) if preferred == 'sms' else (_try_whatsapp, _try_sms)

        try:
            result = first()
            if result.get('success'):
                return result
        except Exception as first_error:
            logger.warning(f"Primary channel '{preferred}' failed for {phone_number}: {first_error}")

        try:
            return second()
        except Exception as second_error:
            logger.error(f"Both channels failed for {phone_number}: {second_error}")
            return {'success': False, 'channel': None, 'error': str(second_error)}

    @staticmethod
    def send_verification_token(phone_number):
        """
        Legacy method using Twilio Verify API.
        Send a verification code to the specified phone number.
        Tries WhatsApp first, then falls back to SMS.
        """
        client = TwilioService.get_client()
        service_sid = os.getenv('TWILIO_VERIFY_SERVICE_SID')
        
        if not client or not service_sid:
            return {'success': False, 'error': 'Twilio Verify service not configured'}
            
        try:
            logger.info(f"Attempting to send OTP to {phone_number} via WhatsApp (Verify API)...")
            verification = client.verify \
                .v2 \
                .services(service_sid) \
                .verifications \
                .create(to=phone_number, channel='whatsapp')
                
            logger.info(f"Successfully initiated OTP via WhatsApp to {phone_number}, status: {verification.status}")
            return {'success': True, 'status': verification.status}
        except Exception as e:
            logger.warning(f"Failed to send OTP via WhatsApp to {phone_number}: {e}. Falling back to SMS.")
            try:
                verification = client.verify \
                    .v2 \
                    .services(service_sid) \
                    .verifications \
                    .create(to=phone_number, channel='sms')
                
                logger.info(f"Successfully initiated OTP via SMS to {phone_number}, status: {verification.status}")
                return {'success': True, 'status': verification.status}
            except Exception as sms_e:
                logger.error(f"Failed to send OTP via SMS to {phone_number} as well: {sms_e}")
                return {'success': False, 'error': str(sms_e)}

    @staticmethod
    def check_verification_token(phone_number, code):
        """
        Legacy method using Twilio Verify API.
        Verify the code provided by the user.
        """
        client = TwilioService.get_client()
        service_sid = os.getenv('TWILIO_VERIFY_SERVICE_SID')
        
        if not client or not service_sid:
            return {'success': False, 'error': 'Twilio Verify service not configured'}
            
        try:
            verification_check = client.verify \
                .v2 \
                .services(service_sid) \
                .verification_checks \
                .create(to=phone_number, code=code)
            
            if verification_check.status == 'approved':
                return {'success': True}
            else:
                return {'success': False, 'error': 'Invalid verification code'}
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return {'success': False, 'error': str(e)}
