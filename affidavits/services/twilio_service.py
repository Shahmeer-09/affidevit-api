import os
from twilio.rest import Client
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class TwilioService:
    @staticmethod
    def get_client():
        try:
            account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            
            if not account_sid or not auth_token:
                logger.warning("Twilio credentials not found in environment variables")
                return None
                
            return Client(account_sid, auth_token)
        except Exception as e:
            logger.error(f"Error initializing Twilio client: {e}")
            return None

    @staticmethod
    def send_verification_token(phone_number):
        """
        Send a verification code to the specified phone number.
        Tries WhatsApp first, then falls back to SMS.
        """
        client = TwilioService.get_client()
        service_sid = os.getenv('TWILIO_VERIFY_SERVICE_SID')
        
        if not client or not service_sid:
            return {'success': False, 'error': 'Twilio service not configured'}
            
        try:
            # First, try to send via WhatsApp
            logger.info(f"Attempting to send OTP to {phone_number} via WhatsApp...")
            verification = client.verify \
                .v2 \
                .services(service_sid) \
                .verifications \
                .create(to=phone_number, channel='whatsapp')
                
            logger.info(f"Successfully initiated OTP via WhatsApp to {phone_number}, status: {verification.status}")
            return {'success': True, 'status': verification.status}
        except Exception as e:
            # If WhatsApp fails, log it and fall back to SMS
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
        Verify the code provided by the user.
        """
        client = TwilioService.get_client()
        service_sid = os.getenv('TWILIO_VERIFY_SERVICE_SID')
        
        if not client or not service_sid:
            return {'success': False, 'error': 'Twilio service not configured'}
            
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
