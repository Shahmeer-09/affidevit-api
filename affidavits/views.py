"""
Affidavit Express - API Views

RESTful API endpoints for all user roles:
- Public: Decision tree, request creation, auto-save, submission
- Commissioner: Retrieve by code, mark complete, friction reports
- Reviewer: Review queue, approve/reject, edit tracking
- Admin: Policy management, dashboard, learning export
"""

import json
import os
from datetime import timedelta
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.db.models import Count, Q, F, Avg
from django.db import transaction
from django.http import HttpResponse
from rest_framework import viewsets, generics, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny, IsAuthenticated

from .models import (
    User, AffidavitType, DecisionTreeNode, Request,
    Stamp, FrictionReport, ReviewerEdit, RequestEvent, AIRun, PaymentLog,
    SiteSettings, Ticket, TicketMessage, TicketAttachment, CommissionerSlot,
    ReviewerFeedback, SubmitFeedback
)
from .serializers import (
    UserSerializer, UserRegistrationSerializer, CommissionerSerializer,
    CommissionerPublicSerializer, ReviewerSerializer,
    CreateStaffUserSerializer, UpdateStaffUserSerializer,
    AffidavitTypeSerializer, AffidavitTypeListSerializer,
    AffidavitTypePolicyUpdateSerializer,
    DecisionTreeNodeSerializer, DecisionTreeNodeChildSerializer,
    DecisionTreeAnswerSerializer,
    RequestCreateSerializer, RequestPatchSerializer, RequestSubmitSerializer,
    RequestListSerializer, RequestDetailSerializer, RequestReviewerSerializer,
    StampSerializer, MarkCompleteSerializer,
    FrictionReportSerializer, FrictionReportCreateSerializer,
    ReviewerEditSerializer, ReviewerEditCreateSerializer, ApproveRequestSerializer,
    ReviewerFeedbackSerializer, ReviewerFeedbackCreateSerializer, SubmitFeedbackSerializer,
    ConfidenceDashboardSerializer, LearningExportSerializer,
    PaymentLogSerializer, CommissionerPaymentSummarySerializer, MarkAsPaidSerializer,
    TicketSerializer, TicketDetailSerializer, TicketMessageSerializer, TicketAttachmentSerializer,
    CommissionerSlotSerializer,
    GuestSignupStartSerializer, GuestSignupVerifySerializer
)
from .authentication import (
    IsCommissioner, IsReviewer, IsAdminUser, IsSuperUser,
    IsOwnerOrAdmin, IsCommissionerOrReviewerOrAdmin
)
from .services import process_request, generate_affidavit_pdf
from .services.ai_service import validate_inputs_before_submission, translate_to_english, draft_affidavit, refine_template_section, refine_user_instruction
from .services.notification_service import (
    send_approval_notification, 
    send_clarification_notification,
    send_completion_notification,
    send_ticket_created_notification,
    send_ticket_reply_notification,
    send_otp_email,
    send_welcome_email
)
from .services.twilio_service import TwilioService
from .services.dashboard_service import (
    get_dashboard_data,
    get_weekly_learning_report,
    get_type_trend_data
)
from .tasks import (
    process_request_async,
    generate_pdf_async,
    send_notification_async
)


# ---------------------------------------------------------------------------
# Helper: filter answers to only include visible (non-hidden) fields
# ---------------------------------------------------------------------------

def _is_field_visible(field: dict, answers: dict, schema: list) -> bool:
    """Return True if a field should be shown given current answers (mirrors frontend shouldShowQuestion)."""
    show_if = field.get('show_if')
    if not show_if:
        return True

    parent_id = show_if.get('field', '')
    required_value = show_if.get('value')

    parent_answer = answers.get(parent_id)
    if parent_answer is None:
        for q in schema:
            qid = q.get('id') or q.get('field_name', '')
            if qid == parent_id or q.get('field_name') == parent_id:
                parent_answer = answers.get(qid)
                break

    if not required_value or required_value == '':
        return parent_answer is not None and parent_answer != '' and parent_answer is not False

    if isinstance(parent_answer, list):
        check = required_value if isinstance(required_value, list) else [required_value]
        return any(v in parent_answer for v in check)

    if isinstance(required_value, list):
        return parent_answer in required_value

    return parent_answer == required_value


def _filter_visible_answers(answers: dict, intake_schema: list) -> dict:
    """Return a copy of answers containing only keys for visible fields."""
    visible_ids = set()
    for field in intake_schema:
        fid = field.get('id') or field.get('field_name', '')
        if _is_field_visible(field, answers, intake_schema):
            visible_ids.add(fid)
    # Always keep internal keys (start with _)
    return {k: v for k, v in answers.items() if k in visible_ids or k.startswith('_')}


class ReviewerFeedbackCreateView(generics.CreateAPIView):
    """Reviewer-only endpoint to add minimal feedback for a request."""

    serializer_class = ReviewerFeedbackCreateSerializer
    permission_classes = [IsAuthenticated, IsReviewer]

    def create(self, request, *args, **kwargs):
        request_obj = get_object_or_404(Request, pk=kwargs.get('pk'))

        serializer = self.get_serializer(
            data={
                **request.data,
                'request': request_obj.id,
            }
        )
        serializer.is_valid(raise_exception=True)
        feedback = serializer.save()

        return Response(
            ReviewerFeedbackSerializer(feedback, context={'request': request}).data,
            status=status.HTTP_201_CREATED,
        )


class AdminReviewerFeedbackListView(generics.ListAPIView):
    """Admin endpoint to view reviewer feedback logs with filtering and search."""

    serializer_class = ReviewerFeedbackSerializer
    permission_classes = [IsAdminUser]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['request__request_code', 'reviewer__username', 'reviewer__email', 'message']
    ordering_fields = ['created_at', 'category']
    ordering = ['-created_at']

    def get_queryset(self):
        queryset = ReviewerFeedback.objects.select_related('request', 'reviewer').all()

        category = self.request.query_params.get('category')
        if category:
            queryset = queryset.filter(category=category)

        reviewer_id = self.request.query_params.get('reviewer')
        if reviewer_id:
            queryset = queryset.filter(reviewer_id=reviewer_id)

        request_code = self.request.query_params.get('request_code')
        if request_code:
            queryset = queryset.filter(request__request_code__icontains=request_code.strip())

        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')
        if start_date:
            queryset = queryset.filter(created_at__date__gte=start_date)
        if end_date:
            queryset = queryset.filter(created_at__date__lte=end_date)

        return queryset.order_by('-created_at')


class SubmitFeedbackView(generics.CreateAPIView):
    """Generic endpoint for admin/site feedback submissions."""

    serializer_class = SubmitFeedbackSerializer
    permission_classes = [AllowAny]

    def perform_create(self, serializer):
        user = self.request.user if self.request.user.is_authenticated else None
        serializer.save(user=user)



# =============================================================================
# Authentication Views
# =============================================================================

class UserRegistrationView(generics.CreateAPIView):
    """Public endpoint for user registration."""
    
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [AllowAny]
    
    def create(self, request, *args, **kwargs):
        import random
        from django.utils import timezone
        from rest_framework_simplejwt.tokens import RefreshToken
        
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        # Generate 6-digit OTP, store on user model, keep user inactive until verified
        otp_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        user.otp_code = otp_code
        user.otp_created_at = timezone.now()
        user.is_active = False
        user.save()
        
        # Send OTP via email
        send_otp_email(
            email=user.email,
            otp_code=otp_code,
            user_name=user.first_name or user.username,
        )
        
        return Response({
            'message': 'Registration successful. Please check your email for a verification code.',
            'otp_sent': True,
            'user_id': str(user.id),
        }, status=status.HTTP_201_CREATED)


class GuestAuthView(viewsets.ViewSet):
    """
    ViewSet for handling guest signup flows (invisible signup).
    """
    permission_classes = [AllowAny]

    @action(detail=False, methods=['post'])
    def start(self, request):
        """Start guest signup - Generate and send OTP via WhatsApp/SMS + Email."""
        import random
        
        serializer = GuestSignupStartSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        email = serializer.validated_data['email']
        full_name = serializer.validated_data['full_name']
        phone_number = serializer.validated_data.get('phone_number', '')
        
        # Generate 6-digit OTP
        otp_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        
        # Store OTP on the User model (reliable — no cache required).
        # Create an inactive placeholder user if they don't already exist.
        parts = full_name.split(' ', 1)
        first_name = parts[0]
        last_name = parts[1] if len(parts) > 1 else ''

        user_qs = User.objects.filter(email__iexact=email)
        if user_qs.exists():
            user = user_qs.order_by('-is_active', '-date_joined').first()
        else:
            from django.utils.crypto import get_random_string as _grs
            base_username = email.split('@')[0]
            username = base_username
            counter = 1
            while User.objects.filter(username=username).exists():
                username = f"{base_username}{counter}"
                counter += 1
            user = User.objects.create_user(
                username=username,
                email=email,
                password=_grs(16),
                first_name=first_name,
                last_name=last_name,
                phone_number=phone_number,
                role=User.Role.PUBLIC,
                is_active=False,  # activated after OTP verified
            )

        # Write OTP onto the user record
        user.otp_code = otp_code
        user.otp_created_at = timezone.now()
        user.save(update_fields=['otp_code', 'otp_created_at'])

        # Send OTP via Email (always)
        email_result = send_otp_email(
            email=email,
            otp_code=otp_code,
            user_name=full_name
        )
        
        # Send OTP via WhatsApp/SMS if phone number provided
        sms_result = {'success': False, 'channel': None, 'error': 'No phone number provided'}
        if phone_number:
            sms_result = TwilioService.send_otp_message(
                phone_number=phone_number,
                otp_code=otp_code
            )
        
        return Response({
            "message": "Verification code sent successfully.",
            "email_sent": email_result.get('success', False),
            "sms_sent": sms_result.get('success', False),
            "sms_channel": sms_result.get('channel'),  # 'whatsapp' or 'sms'
        })

    @action(detail=False, methods=['post'])
    def verify(self, request):
        """Verify OTP, Create Account, Create Request, Mark Paid."""
        from rest_framework_simplejwt.tokens import RefreshToken
        from django.db import IntegrityError

        serializer = GuestSignupVerifySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data['email']
        otp = serializer.validated_data['otp']
        full_name = serializer.validated_data['full_name']
        phone_number = serializer.validated_data.get('phone_number', '')
        affidavit_type_id = serializer.validated_data['affidavit_type_id']
        answers_json = serializer.validated_data['answers_json']
        draft_text = serializer.validated_data.get('draft_text', '')

        # 1. Verify OTP from User model (stored in start step)
        from django.utils import timezone as tz
        try:
            user_qs = User.objects.filter(email__iexact=email)
            if not user_qs.exists():
                return Response({"otp": "OTP expired or not found. Please request a new code."}, status=status.HTTP_400_BAD_REQUEST)
            cached_user = user_qs.order_by('-is_active', '-date_joined').first()

            if not cached_user.otp_code:
                return Response({"otp": "OTP expired or not found. Please request a new code."}, status=status.HTTP_400_BAD_REQUEST)

            # Check 10-minute expiry
            if cached_user.otp_created_at:
                age_seconds = (tz.now() - cached_user.otp_created_at).total_seconds()
                if age_seconds > 600:
                    cached_user.otp_code = ''
                    cached_user.otp_created_at = None
                    cached_user.save(update_fields=['otp_code', 'otp_created_at'])
                    return Response({"otp": "OTP has expired. Please request a new code."}, status=status.HTTP_400_BAD_REQUEST)

            if cached_user.otp_code != otp:
                return Response({"otp": "Invalid OTP code."}, status=status.HTTP_400_BAD_REQUEST)

            # OTP verified — clear it
            cached_user.otp_code = ''
            cached_user.otp_created_at = None
            cached_user.save(update_fields=['otp_code', 'otp_created_at'])

            # Use phone from the user record if not provided in this request
            if not phone_number and cached_user.phone_number:
                phone_number = cached_user.phone_number

        except Exception as e:
            logger.error(f"Guest OTP verification error: {e}")
            return Response({"otp": "OTP expired or not found. Please request a new code."}, status=status.HTTP_400_BAD_REQUEST)

        # 2. Get or Create User
        is_new_user = False
        temp_password = None

        allow_duplicate_phones = (os.getenv('ALLOW_DUPLICATE_PHONE_NUMBERS') or '').strip().lower() == 'true'
        
        try:
            user = User.objects.get(email__iexact=email)
            # Update existing user info if needed
            if not user.is_active:
                user.is_active = True
            # Update name if missing
            if not user.first_name and full_name:
                parts = full_name.split(' ', 1)
                user.first_name = parts[0]
                if len(parts) > 1:
                    user.last_name = parts[1]
            if phone_number and user.phone_number != phone_number:
                if (not allow_duplicate_phones) and User.objects.exclude(id=user.id).filter(phone_number=phone_number).exists():
                    return Response(
                        {'phone_number': 'This phone number is already in use. Please use a different phone number or sign in.'},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
                if not user.phone_number:
                    user.phone_number = phone_number
            try:
                user.save()
            except IntegrityError:
                return Response(
                    {'detail': 'Could not update user due to a conflicting email/phone. Please try a different email/phone or sign in.'},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        except User.DoesNotExist:
            if User.objects.filter(email__iexact=email).exists():
                return Response(
                    {'email': 'An account with this email already exists. Please sign in instead.'},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if (not allow_duplicate_phones) and phone_number and User.objects.filter(phone_number=phone_number).exists():
                return Response(
                    {'phone_number': 'An account with this phone number already exists. Please sign in instead.'},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            # Create new user
            username = email.split('@')[0]
            # Ensure unique username
            counter = 1
            base_username = username
            while User.objects.filter(username=username).exists():
                username = f"{base_username}{counter}"
                counter += 1
            
            parts = full_name.split(' ', 1)
            first_name = parts[0]
            last_name = parts[1] if len(parts) > 1 else ''

            # Generate temporary password
            temp_password = get_random_string(12)
            is_new_user = True

            try:
                user = User.objects.create_user(
                    username=username,
                    email=email,
                    password=temp_password,
                    first_name=first_name,
                    last_name=last_name,
                    phone_number=phone_number,
                    role=User.Role.PUBLIC,
                    is_active=True
                )
            except IntegrityError:
                existing_email = User.objects.filter(email__iexact=email).exists()
                existing_phone = (not allow_duplicate_phones) and bool(phone_number) and User.objects.filter(phone_number=phone_number).exists()
                if existing_email:
                    return Response(
                        {'email': 'An account with this email already exists. Please sign in instead.'},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
                if existing_phone:
                    return Response(
                        {'phone_number': 'An account with this phone number already exists. Please sign in instead.'},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
                return Response(
                    {'detail': 'Could not create account due to a conflicting email/phone. Please try again.'},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # 3. Create Request
        affidavit_type = get_object_or_404(AffidavitType, id=affidavit_type_id)
        
        # OTP verification only creates/authenticates the user and request.
        # Payment and affidavit processing happen in later explicit steps.
        has_draft = bool(draft_text and len(draft_text) > 50)
        initial_status = Request.Status.DRAFT
            
        request_obj = Request.objects.create(
            user=user,
            affidavit_type=affidavit_type,
            answers_json=answers_json,
            draft_text=draft_text if draft_text else '',
            policy_version_used=affidavit_type.policy_version,
            prompt_version_used=affidavit_type.prompt_pack_version,
            template_version_used=affidavit_type.template_version,
            status=initial_status,
            is_paid=False,
            user_paid_at=None,
        )

        # 5. Send Welcome Message for new users
        if is_new_user and temp_password:
            # Build password reset link
            from django.conf import settings
            site_url = getattr(settings, 'SITE_URL', 'http://localhost:3000')
            reset_link = f"{site_url}/reset-password?email={email}"
            
            # Send welcome email with temp password
            send_welcome_email(
                email=email,
                temp_password=temp_password,
                reset_link=reset_link,
                user_name=full_name,
                phone_number=phone_number
            )
            
            # Send welcome message via WhatsApp/SMS if phone number provided
            if phone_number:
                TwilioService.send_welcome_message(
                    phone_number=phone_number,
                    temp_password=temp_password,
                    reset_link=reset_link
                )

        # 6. Generate Tokens
        refresh = RefreshToken.for_user(user)
        
        # 7. Next step is payment for the new flow
        next_step = 'payment_required'

        return Response({
            'user': UserSerializer(user).data,
            'request': RequestDetailSerializer(request_obj, context={'request': request}).data,
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'is_new_user': is_new_user,
            'next_step': next_step,
        }, status=status.HTTP_201_CREATED)



class VerifyOTPView(APIView):
    """
    Verify OTP for email or phone number verification.
    Checks model-stored OTP first (email flow), then falls back to Twilio (phone flow).
    """
    permission_classes = [AllowAny]
    
    def post(self, request):
        from django.utils import timezone
        from rest_framework_simplejwt.tokens import RefreshToken
        
        user_id = request.data.get('user_id')
        code = request.data.get('code')
        
        if not user_id or not code:
            return Response(
                {'error': 'User ID and Code are required'},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        user = get_object_or_404(User, pk=user_id)
        
        # --- Email OTP path (stored on user model) ---
        if user.otp_code:
            # OTP valid for 10 minutes
            if user.otp_created_at:
                age_seconds = (timezone.now() - user.otp_created_at).total_seconds()
                if age_seconds > 600:
                    return Response(
                        {'error': 'Verification code has expired. Please register again to get a new code.'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            if user.otp_code != code:
                return Response(
                    {'error': 'Invalid verification code.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Correct code — clear OTP
            user.otp_code = ''
            user.otp_created_at = None

            # Commissioners stay inactive pending admin approval; only PUBLIC users get activated
            if user.role == User.Role.COMMISSIONER:
                user.save()
                return Response({
                    'success': True,
                    'message': 'Email verified. Your application is pending admin approval. You will be notified once approved.',
                })

            user.is_active = True
            user.save()

            refresh = RefreshToken.for_user(user)
            return Response({
                'success': True,
                'message': 'Email verified successfully.',
                'user': UserSerializer(user).data,
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            })
        
        # --- Phone OTP fallback (Twilio Verify) ---
        if not user.phone_number:
            return Response(
                {'error': 'No verification code found for this account. Please register again.'},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        from .services.twilio_service import TwilioService
        result = TwilioService.check_verification_token(user.phone_number, code)
        
        if result['success']:
            user.is_phone_verified = True
            
            if user.role == User.Role.PUBLIC:
                user.is_active = True
                user.save()
                
                refresh = RefreshToken.for_user(user)
                return Response({
                    'success': True,
                    'message': 'Phone verified successfully.',
                    'user': UserSerializer(user).data,
                    'refresh': str(refresh),
                    'access': str(refresh.access_token),
                })
            else:
                user.save()
                return Response({
                    'success': True,
                    'message': 'Phone verified successfully. Waiting for admin approval.'
                })
        else:
            return Response(
                {'error': result.get('error', 'Invalid verification code')},
                status=status.HTTP_400_BAD_REQUEST
            )


class CommissionerRegistrationView(APIView):
    """
    Public endpoint for commissioner self-registration.
    Commissioners can sign up with their details, availability, and profile image.
    """
    
    permission_classes = [AllowAny]
    
    def post(self, request):
        from .serializers import CommissionerRegistrationSerializer, CommissionerSerializer
        
        import random
        from django.utils import timezone

        serializer = CommissionerRegistrationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        # Generate OTP for email verification; keep inactive until both OTP verified + admin approval
        otp_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        user.otp_code = otp_code
        user.otp_created_at = timezone.now()
        user.is_active = False
        user.save()

        # Send OTP via email
        send_otp_email(
            email=user.email,
            otp_code=otp_code,
            user_name=user.first_name or user.username,
        )

        return Response({
            'message': 'Registration submitted. Please verify your email, then wait for admin approval.',
            'otp_sent': True,
            'user_id': str(user.id),
        }, status=status.HTTP_201_CREATED)


class UserProfileView(generics.RetrieveUpdateAPIView):
    """Get or update current user's profile."""
    
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]
    
    def get_object(self):
        return self.request.user


class PasswordResetRequestView(APIView):
    """
    Request a password reset email.
    Always returns success to prevent email enumeration.
    """
    
    permission_classes = [AllowAny]
    
    def post(self, request):
        email = request.data.get('email', '').lower().strip()
        
        if not email:
            return Response(
                {'error': 'Email is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            user = User.objects.get(email__iexact=email, is_active=True)
            
            # Generate password reset token
            from django.contrib.auth.tokens import default_token_generator
            from django.utils.http import urlsafe_base64_encode
            from django.utils.encoding import force_bytes
            from django.conf import settings
            from django.core.mail import send_mail
            from django.template.loader import render_to_string
            
            token = default_token_generator.make_token(user)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            
            reset_url = f"{settings.SITE_URL}/reset-password/{uid}/{token}"
            
            # Send email using HTML template
            subject = 'Reset Your Password - Affidavit Express'
            
            html_message = render_to_string('emails/password_reset.html', {
                'user_name': user.first_name or user.username,
                'reset_url': reset_url,
            })
            
            try:
                send_mail(
                    subject,
                    '',  # Plain text message (empty since we're using html_message)
                    settings.DEFAULT_FROM_EMAIL,
                    [user.email],
                    fail_silently=False,
                    html_message=html_message,
                )
            except Exception as e:
                # Log the error but don't expose it
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to send password reset email: {e}")
        
        except User.DoesNotExist:
            # Don't reveal that the user doesn't exist
            pass
        
        # Always return success to prevent email enumeration
        return Response({
            'success': True,
            'message': 'If an account exists with this email, you will receive password reset instructions.'
        })


class PasswordResetConfirmView(APIView):
    """
    Confirm password reset with token and set new password.
    """
    
    permission_classes = [AllowAny]
    
    def post(self, request):
        from django.contrib.auth.tokens import default_token_generator
        from django.utils.http import urlsafe_base64_decode
        
        uid = request.data.get('uid', '')
        token = request.data.get('token', '')
        new_password = request.data.get('new_password', '')
        
        if not all([uid, token, new_password]):
            return Response(
                {'error': 'Missing required fields'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if len(new_password) < 8:
            return Response(
                {'error': 'Password must be at least 8 characters'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            user_id = urlsafe_base64_decode(uid).decode()
            user = User.objects.get(pk=user_id, is_active=True)
            
            if not default_token_generator.check_token(user, token):
                return Response(
                    {'error': 'Invalid or expired reset link'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            user.set_password(new_password)
            user.save()
            
            return Response({
                'success': True,
                'message': 'Password has been reset successfully. You can now log in with your new password.'
            })
            
        except (User.DoesNotExist, ValueError, TypeError):
            return Response(
                {'error': 'Invalid reset link'},
                status=status.HTTP_400_BAD_REQUEST
            )


# =============================================================================
# Decision Tree Views (Public - Story 1.1)
# =============================================================================

class DecisionTreeRootView(generics.ListAPIView):
    """Get root nodes of the decision tree (starting questions)."""
    
    serializer_class = DecisionTreeNodeSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        return DecisionTreeNode.objects.filter(
            parent_node__isnull=True,
            is_active=True
        ).order_by('order')


class DecisionTreeNodeView(generics.RetrieveAPIView):
    """Get a specific decision tree node with its children."""
    
    serializer_class = DecisionTreeNodeSerializer
    permission_classes = [AllowAny]
    queryset = DecisionTreeNode.objects.filter(is_active=True)


class DecisionTreeTraverseView(APIView):
    """
    Traverse the decision tree based on user's answer.
    Returns the next node or the final affidavit type result.
    """
    
    permission_classes = [AllowAny]
    
    def post(self, request):
        serializer = DecisionTreeAnswerSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        node_id = serializer.validated_data['node_id']
        answer_value = serializer.validated_data['answer_value']
        
        # Find the child node matching the answer
        try:
            current_node = DecisionTreeNode.objects.get(id=node_id, is_active=True)
        except DecisionTreeNode.DoesNotExist:
            return Response(
                {'error': 'Node not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Find child with matching answer
        next_node = current_node.children.filter(
            answer_value=answer_value,
            is_active=True
        ).first()
        
        if not next_node:
            return Response(
                {'error': 'Invalid answer for this question'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check if this is a leaf node (has result)
        if next_node.is_leaf:
            return Response({
                'is_complete': True,
                'affidavit_type': AffidavitTypeListSerializer(
                    next_node.result_affidavit_type
                ).data
            })
        
        # Return the next question
        return Response({
            'is_complete': False,
            'node': DecisionTreeNodeSerializer(next_node).data
        })


# =============================================================================
# Affidavit Type Views
# =============================================================================

class AffidavitTypeListView(generics.ListAPIView):
    """List all active affidavit types (public)."""
    
    serializer_class = AffidavitTypeListSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        return AffidavitType.objects.filter(is_active=True)


class AffidavitTypeDetailView(generics.RetrieveAPIView):
    """Get details of an affidavit type including intake schema."""
    
    serializer_class = AffidavitTypeSerializer
    permission_classes = [AllowAny]
    queryset = AffidavitType.objects.filter(is_active=True)


# =============================================================================
# Request Views (Public - Stories 1.2, 1.3, 1.4)
# =============================================================================

class RequestCreateView(generics.CreateAPIView):
    """Create a new affidavit request."""
    
    serializer_class = RequestCreateSerializer
    permission_classes = [IsAuthenticated]
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context


class RequestDetailView(generics.RetrieveAPIView):
    """Get request details (owner only)."""
    
    serializer_class = RequestDetailSerializer
    permission_classes = [IsAuthenticated, IsOwnerOrAdmin]
    
    def get_queryset(self):
        user = self.request.user
        if user.role == 'admin':
            return Request.objects.all()
        return Request.objects.filter(user=user)


class RequestPatchView(generics.UpdateAPIView):
    """
    Auto-save intake form answers (Story 1.2 - Resilient Intake).
    Accepts PATCH requests to update answers_json.
    """
    
    serializer_class = RequestPatchSerializer
    permission_classes = [IsAuthenticated, IsOwnerOrAdmin]
    http_method_names = ['patch']
    
    def get_queryset(self):
        return Request.objects.filter(user=self.request.user)


class RequestDeleteView(APIView):
    """
    Delete a request (owner only).
    Only allows deletion of requests that haven't been completed/approved.
    """
    
    permission_classes = [IsAuthenticated]
    
    def delete(self, request, pk):
        request_obj = get_object_or_404(
            Request,
            pk=pk,
            user=request.user
        )
        
        # Only allow deletion of certain statuses
        allowed_statuses = [
            Request.Status.DRAFT,
            Request.Status.SUBMITTED,
            Request.Status.DRAFT_READY,
            Request.Status.NEEDS_CLARIFICATION,
            Request.Status.NEEDS_REVIEW,
            Request.Status.REJECTED,
        ]
        
        if request_obj.status not in allowed_statuses:
            return Response(
                {'error': 'Cannot delete a completed or approved request.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        request_code = request_obj.request_code
        request_obj.delete()
        
        return Response({
            'success': True,
            'message': f'Request {request_code} has been deleted.'
        })


class ValidateRequestInputView(APIView):
    """
    Validate user input BEFORE submission.
    This endpoint is called by the frontend to check if all fields are valid
    before allowing the user to submit their request.
    
    The AI validates each field and returns a list of invalid fields
    with explanations and examples of valid values.
    
    This replaces the old clarification-during-drafting flow.
    """
    
    permission_classes = [AllowAny]  # Anyone can validate their inputs
    
    def post(self, request):
        answers_json = request.data.get('answers_json', {})
        affidavit_type_id = request.data.get('affidavit_type_id')
        
        if not affidavit_type_id:
            return Response(
                {'error': 'affidavit_type_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not answers_json:
            return Response(
                {'error': 'answers_json is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get the affidavit type for template context
        affidavit_type = get_object_or_404(AffidavitType, pk=affidavit_type_id)
        
        # ===== DEBUG LOGGING =====
        import logging
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info("[VALIDATE_VIEW] Starting validation request")
        logger.info(f"[VALIDATE_VIEW] ORIGINAL answers_json: {json.dumps(answers_json, ensure_ascii=False)}")
        logger.info("=" * 80)
        # ===== END DEBUG LOGGING =====
        
        # STEP 0: Strip out answers for fields hidden by show_if conditions
        intake_schema = affidavit_type.intake_schema or []
        visible_answers = _filter_visible_answers(answers_json, intake_schema)

        # STEP 1: Translate non-English content to English BEFORE validation
        translated_answers = translate_to_english(visible_answers)
        
        # ===== DEBUG LOGGING =====
        logger.info("=" * 80)
        logger.info(f"[VALIDATE_VIEW] TRANSLATED answers_json: {json.dumps(translated_answers, ensure_ascii=False)}")
        logger.info(f"[VALIDATE_VIEW] Translation changed data: {answers_json != translated_answers}")
        logger.info("=" * 80)
        # ===== END DEBUG LOGGING =====
        
        # STEP 2: Run pre-submission validation on translated data
        validation_result = validate_inputs_before_submission(
            answers_json=translated_answers,
            template_html=affidavit_type.template_html,
            affidavit_type_name=affidavit_type.name,
            validation_rules=affidavit_type.validation_rules or []
        )
        
        # ===== DEBUG LOGGING =====
        logger.info("=" * 80)
        logger.info(f"[VALIDATE_VIEW] VALIDATION RESULT: {validation_result}")
        logger.info("=" * 80)
        # ===== END DEBUG LOGGING =====

        # STEP 3: Return validation result (draft generated separately via submit/Celery)
        return Response({
            'success': True,
            'all_valid': validation_result.get('all_valid', True),
            'invalid_fields': validation_result.get('invalid_fields', {}),
            'validation_notes': validation_result.get('validation_notes', []),
            'field_checks': validation_result.get('field_checks', []),
        })


class RequestSubmitView(APIView):
    """
    Submit a request for AI processing.
    Triggers async processing via Celery task.
    """
    
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        request_obj = get_object_or_404(
            Request, 
            pk=pk, 
            user=request.user
        )

        # Require payment before AI processing
        if not request_obj.is_paid:
            return Response(
                {'error': 'Payment is required before processing can begin.'},
                status=status.HTTP_402_PAYMENT_REQUIRED
            )
        
        # Handle submission from DRAFT_READY -> NEEDS_REVIEW (after appointment booked)
        # OR if draft already exists (prevent regeneration)
        if request_obj.status == Request.Status.DRAFT_READY or (request_obj.draft_text and len(request_obj.draft_text) > 50):
            request_obj.status = Request.Status.NEEDS_REVIEW
            request_obj.submitted_at = timezone.now() # Update timestamp or keep original? Keeping update to show recent activity
            request_obj.save()
            
            RequestEvent.objects.create(
                request=request_obj,
                action=RequestEvent.Action.SUBMITTED,
                actor=request.user,
                actor_role=request.user.role,
                details={'message': 'Submitted for review (Draft already exists)'}
            )

            # Notify user and reviewers that request entered review queue
            try:
                from .services.notification_service import (
                    send_request_in_review_notification,
                    send_review_queue_notification_to_reviewers,
                )

                send_request_in_review_notification(request_obj)
                send_review_queue_notification_to_reviewers(request_obj)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to send in-review notifications for {request_obj.request_code}: {e}")
            
            return Response({
                'status': request_obj.status,
                'request_code': request_obj.request_code,
                'message': 'Your request has been submitted for review.'
            })
        
        # Validate submission for initial draft
        serializer = RequestSubmitSerializer(
            instance=request_obj, 
            data={},
            context={'request': request}
        )
        serializer.is_valid(raise_exception=True)
        
        # Update submitted timestamp
        request_obj.submitted_at = timezone.now()
        request_obj.status = Request.Status.SUBMITTED
        request_obj.save()
        
        # Log the event
        RequestEvent.objects.create(
            request=request_obj,
            action=RequestEvent.Action.SUBMITTED,
            actor=request.user,
            actor_role=request.user.role,
            ip_address=self._get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', '')[:500]
        )
        
        # Queue async AI processing
        task = process_request_async.delay(request_obj.id)
        
        return Response({
            'status': request_obj.status,
            'request_code': request_obj.request_code,
            'task_id': task.id,
            'message': 'Your request has been submitted and is being processed. This may take a moment.'
        })
    
    def _get_client_ip(self, request):
        """Extract client IP from request."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR')


class SelectCommissionerView(APIView):
    """
    Allow user to select a commissioner for their approved request.
    """
    
    permission_classes = [IsAuthenticated]
    
    def patch(self, request, pk):
        request_obj = get_object_or_404(
            Request, 
            pk=pk, 
            user=request.user
        )

        # Lock commissioner changes once an appointment has been accepted by commissioner.
        if (
            hasattr(request_obj, 'appointment_slot')
            and request_obj.appointment_slot.appointment_status
            == CommissionerSlot.AppointmentStatus.ACCEPTED
        ):
            return Response(
                {'error': 'Commissioner selection is locked after appointment confirmation.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        # Ensure status is valid for selection
        if request_obj.status not in [
            Request.Status.DRAFT_READY,
            Request.Status.NEEDS_REVIEW,
            Request.Status.APPROVED,
        ]:
             return Response(
                {'error': 'Request is not ready for commissioner selection.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Require payment before selecting commissioner
        if not request_obj.is_paid:
            return Response(
                {'error': 'Payment required before selecting a commissioner.'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        commissioner_id = request.data.get('commissioner_id')
        
        # Allow withdrawing commissioner selection (set to null or 0)
        if commissioner_id is None or commissioner_id == 0:
            old_commissioner = request_obj.commissioner
            
            # Release any booked slot
            if hasattr(request_obj, 'appointment_slot'):
                slot = request_obj.appointment_slot
                slot.is_booked = False
                slot.request = None
                slot.save()

            request_obj.commissioner = None
            request_obj.save()
            
            # Log the withdrawal
            RequestEvent.objects.create(
                request=request_obj,
                action=RequestEvent.Action.COMMISSIONER_CHANGED,
                actor=request.user,
                actor_role=request.user.role,
                details={
                    'withdrawn': True,
                    'previous_commissioner_id': old_commissioner.id if old_commissioner else None,
                    'previous_commissioner_name': old_commissioner.get_full_name() if old_commissioner else None
                }
            )
            
            # Notify previous commissioner about withdrawal
            if old_commissioner:
                from .services.notification_service import send_user_withdrawn_notification
                try:
                    send_user_withdrawn_notification(request_obj, slot if hasattr(request_obj, 'appointment_slot') else None, old_commissioner)
                except Exception as e:
                    logger.warning(f"Failed to send withdrawal notification: {e}")
            
            serializer = RequestDetailSerializer(request_obj)
            return Response(serializer.data)
        
        # Validate commissioner exists and is active
        try:
            commissioner = User.objects.get(
                id=commissioner_id,
                role=User.Role.COMMISSIONER,
                is_active=True
            )
        except User.DoesNotExist:
            return Response(
                {'error': 'Invalid commissioner'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Assign commissioner
        old_commissioner = request_obj.commissioner
        request_obj.commissioner = commissioner
        request_obj.save()
        
        # Log the event
        RequestEvent.objects.create(
            request=request_obj,
            action=RequestEvent.Action.COMMISSIONER_CHANGED,
            actor=request.user,
            actor_role=request.user.role,
            details={
                'commissioner_id': commissioner_id, 
                'commissioner_name': commissioner.get_full_name(),
                'previous_commissioner_id': old_commissioner.id if old_commissioner else None,
                'previous_commissioner_name': old_commissioner.get_full_name() if old_commissioner else None
            }
        )
        
        serializer = RequestDetailSerializer(request_obj)
        return Response(serializer.data)


class MarkPaidView(APIView):
    """
    Mark a request as paid by the user (fake payment for now).
    """
    
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        request_obj = get_object_or_404(
            Request, 
            pk=pk, 
            user=request.user
        )
        
        # Only allow payment for certain statuses
        allowed_statuses = [
            Request.Status.DRAFT,
            Request.Status.SUBMITTED,
            Request.Status.DRAFT_READY,
            Request.Status.NEEDS_REVIEW,
            Request.Status.APPROVED,
            Request.Status.COMPLETED
        ]
        if request_obj.status not in allowed_statuses:
            return Response(
                {'error': 'Cannot process payment for this request status'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Already paid check
        if request_obj.is_paid:
            return Response(
                {'error': 'This request has already been paid'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Mark as paid
        request_obj.is_paid = True
        request_obj.user_paid_at = timezone.now()
        request_obj.save(update_fields=['is_paid', 'user_paid_at'])
        
        # Log the payment event
        RequestEvent.objects.create(
            request=request_obj,
            action=RequestEvent.Action.PAYMENT_CONFIRMED,
            actor=request.user,
            actor_role=request.user.role,
            details={'action': 'payment_completed', 'amount': '50.00', 'currency': 'TTD'}
        )
        
        # Check affidavit type mode for post-payment routing
        affidavit_type = request_obj.affidavit_type
        is_review_mode = (affidavit_type.default_mode == 'review_first') or not affidavit_type.is_instant_mode

        # Determine if this is the user's very first paid request.
        # First-time users always see the Thank You page and click the button to trigger AI.
        # Returning users skip Thank You and AI triggers immediately.
        has_previous_paid = Request.objects.filter(
            user=request.user,
            is_paid=True
        ).exclude(pk=request_obj.pk).exists()
        is_first_time = not has_previous_paid

        # If first-time user — don't trigger AI yet regardless of affidavit mode.
        # ThankYouPage button calls /submit/ which starts generation.
        if is_first_time:
            next_step = 'thank_you'
            message = 'Payment confirmed. Click Continue to generate your affidavit.'

        # If review-first mode and status is DRAFT_READY, submit for review
        elif is_review_mode and request_obj.status == Request.Status.DRAFT_READY:
            request_obj.status = Request.Status.NEEDS_REVIEW
            request_obj.submitted_at = timezone.now()
            request_obj.save(update_fields=['status', 'submitted_at'])
            
            RequestEvent.objects.create(
                request=request_obj,
                action=RequestEvent.Action.SUBMITTED,
                actor=request.user,
                actor_role=request.user.role,
                details={'message': 'Submitted for review after payment (review-mode affidavit type)'}
            )
            
            # Notify user and reviewers
            try:
                from .services.notification_service import (
                    send_request_in_review_notification,
                    send_review_queue_notification_to_reviewers,
                )
                send_request_in_review_notification(request_obj)
                send_review_queue_notification_to_reviewers(request_obj)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to send review notifications for {request_obj.request_code}: {e}")
            
            next_step = 'review_queue'
            message = 'Payment confirmed. Your request has been submitted for review.'
        
        elif affidavit_type.default_mode == 'review_first' and request_obj.status == Request.Status.DRAFT:
            # Returning user, review_first, DRAFT — auto-queue AI now
            request_obj.status = Request.Status.SUBMITTED
            request_obj.submitted_at = timezone.now()
            request_obj.save(update_fields=['status', 'submitted_at'])

            RequestEvent.objects.create(
                request=request_obj,
                action=RequestEvent.Action.SUBMITTED,
                actor=request.user,
                actor_role=request.user.role,
                details={'message': 'Auto-submitted for AI processing after payment (review_first, returning user)'}
            )

            from .tasks import process_request_async
            process_request_async.delay(request_obj.id)

            next_step = 'review_queue'
            message = 'Payment confirmed. Your affidavit is being prepared and will be sent for professional review.'

        else:
            if request_obj.status == Request.Status.DRAFT:
                # Returning user, standard/instant — auto-trigger AI immediately
                request_obj.status = Request.Status.SUBMITTED
                request_obj.submitted_at = timezone.now()
                request_obj.save(update_fields=['status', 'submitted_at'])

                RequestEvent.objects.create(
                    request=request_obj,
                    action=RequestEvent.Action.SUBMITTED,
                    actor=request.user,
                    actor_role=request.user.role,
                    details={'message': 'Auto-submitted for AI generation after payment (returning user)'}
                )

                from .tasks import process_request_async
                process_request_async.delay(request_obj.id)

                next_step = 'scheduling'
                message = 'Payment confirmed. Your affidavit is being generated.'
            else:
                # Already past DRAFT (APPROVED, COMPLETED, etc.)
                next_step = 'select_commissioner'
                message = 'Payment confirmed successfully'
        
        serializer = RequestDetailSerializer(request_obj)
        return Response({
            'success': True,
            'message': message,
            'request': serializer.data,
            'next_step': next_step,
        })


class RequestByCodeView(APIView):
    """
    Retrieve a request by its code (for commissioners and admins).
    Story 2.1 - Universal Request Retrieval.
    - Commissioners can only access requests assigned to them (and not completed ones)
    - Admins can access any request for support purposes
    - Supports lookup by User Last Name via query param `?lastname=Smith`
    """
    
    permission_classes = [IsCommissionerOrReviewerOrAdmin]
    
    def get(self, request, code=None):
        # Handle search by last name if 'code' is a special keyword 'search' or omitted
        # But this view is defined as /lookup/<str:code>/ in urls.py
        # So we check if the 'code' param looks like a request code or we need to handle search differently.
        # However, the user asked to search via user last name.
        # Since the URL pattern is fixed, we might need a separate endpoint or overload this one.
        # A better approach for "RequestByCodeView" is to strictly handle codes.
        # But if the user enters a last name in the frontend search box, we need to handle it.
        
        # Let's check if the input `code` matches a request code format (e.g. starts with AFF-)
        # or if we should treat it as a search term.
        
        query = code.strip()
        
        # Try finding by exact request code first
        request_obj = Request.objects.filter(request_code=query.upper()).first()
        
        if not request_obj:
            # If not found by code, try finding by user's name (first, last, full)
            # Only for commissioners to find their assigned requests
            if request.user.role == 'commissioner':
                # Base filter: Assigned to this commissioner OR Unassigned, not completed, and PAID
                base_qs = Request.objects.filter(
                    Q(commissioner=request.user) | Q(commissioner__isnull=True),
                    is_paid=True
                ).exclude(status=Request.Status.COMPLETED)
                
                # Search strategy 1: Direct match on fields (First, Last, Email)
                q_direct = Q(user__last_name__icontains=query) | \
                           Q(user__first_name__icontains=query) | \
                           Q(user__email__icontains=query)
                
                requests = base_qs.filter(q_direct).order_by('-created_at')
                
                if requests.exists():
                    request_obj = requests.first()
                else:
                    # Search strategy 2: Split terms (e.g. "Ali I" -> "Ali" AND "I")
                    # This handles "First Last" or "Last First" input
                    terms = query.split()
                    if len(terms) > 1:
                        # Construct a query where EACH term must be present in EITHER first or last name (or email)
                        q_terms = None
                        for term in terms:
                            term_q = Q(user__last_name__icontains=term) | \
                                     Q(user__first_name__icontains=term) | \
                                     Q(user__email__icontains=term)
                            
                            if q_terms is None:
                                q_terms = term_q
                            else:
                                q_terms &= term_q
                        
                        if q_terms:
                            requests = base_qs.filter(q_terms).order_by('-created_at')
                            if requests.exists():
                                request_obj = requests.first()

        if not request_obj:
            return Response(
                {'error': f'No request found for "{query}". Please check the code or last name.'},
                status=404
            )
        
        # Check payment status before proceeding
        # Users must pay before their affidavit is visible to commissioners
        if not request_obj.is_paid and request.user.role == 'commissioner':
            return Response(
                {'error': 'This request has not been paid for by the user. Please ask them to complete payment first.'},
                status=403
            )
        
        # Admins and reviewers can view any request (no assignment check)
        if request.user.role not in ['admin', 'reviewer']:
            # Commissioners: Cannot access completed requests
            if request_obj.status == Request.Status.COMPLETED:
                return Response(
                    {'error': 'This request has been completed and notarized. You can no longer access it.'},
                    status=403
                )
            
            # Commissioners: Check if request has an assigned commissioner
            if request_obj.commissioner:
                # Only the assigned commissioner can access this request
                if request_obj.commissioner.id != request.user.id:
                    return Response(
                        {'error': f'This request is assigned to {request_obj.commissioner.get_full_name() or request_obj.commissioner.username}. You cannot access it.'},
                        status=403
                    )
        
        # Attempt to acquire lock (skip for admins/reviewers viewing for support)
        if request.user.role == 'commissioner':
            commissioner = request.user
            success, locked_by = request_obj.acquire_lock(commissioner)
            
            response_data = RequestDetailSerializer(request_obj).data
            
            if not success:
                response_data['lock_warning'] = f"Currently being viewed by {locked_by.get_full_name() or locked_by.username}"
                response_data['can_takeover'] = True
            
            return Response(response_data)
        else:
            # Admins/reviewers just view without locking
            response_data = RequestDetailSerializer(request_obj).data
            return Response(response_data)


class RequestTakeoverView(APIView):
    """Force takeover of a locked request (Story 2.1)."""
    
    permission_classes = [IsCommissioner]
    
    def post(self, request, code):
        request_obj = get_object_or_404(Request, request_code=code.upper())
        
        previous_holder = request_obj.force_takeover(request.user)
        
        return Response({
            'success': True,
            'message': f"Taken over from {previous_holder.username if previous_holder else 'no one'}",
            'request': RequestDetailSerializer(request_obj).data
        })


class MyRequestsView(generics.ListAPIView):
    """List current user's requests."""
    
    serializer_class = RequestListSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Request.objects.filter(user=self.request.user)


class RequestStatusView(APIView):
    """
    Check the status of a request (poll for async processing completion).
    Used by frontend to poll after submission.
    Includes AI processing logs for transparency.
    """
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request, pk):
        request_obj = get_object_or_404(
            Request,
            pk=pk,
            user=request.user
        )
        
        # Get AI runs for this request
        from .models import AIRun
        ai_runs = AIRun.objects.filter(request=request_obj).order_by('created_at')
        
        ai_logs = []
        for run in ai_runs:
            ai_logs.append({
                'id': run.id,
                'node_type': run.node_type,
                'node_type_display': run.get_node_type_display(),
                'status': run.status,
                'model_name': run.model_name,
                'prompt_tokens': run.prompt_tokens,
                'completion_tokens': run.completion_tokens,
                'total_tokens': run.total_tokens,
                'latency_ms': run.latency_ms,
                'error_message': run.error_message if run.status == 'failed' else None,
                'created_at': run.created_at.isoformat(),
            })
        
        # Determine processing step
        processing_step = self._get_processing_step(request_obj, ai_logs)
        
        return Response({
            'request_code': request_obj.request_code,
            'status': request_obj.status,
            'qa_passed': request_obj.qa_passed,
            'qa_flags': request_obj.qa_flags_json,
            'clarification_question': request_obj.clarification_question,
            'pdf_url': request_obj.pdf_url or None,
            'updated_at': request_obj.updated_at.isoformat(),
            'is_processing': request_obj.status == Request.Status.SUBMITTED,
            'message': self._get_status_message(request_obj.status),
            'ai_logs': ai_logs,
            'processing_step': processing_step,
        })
    
    def _get_processing_step(self, request_obj, ai_logs):
        """Determine current processing step for UI display."""
        if request_obj.status == Request.Status.SUBMITTED:
            if not ai_logs:
                return {'step': 1, 'label': 'Preparing request...', 'progress': 10}
            
            has_draft = any(log['node_type'] == 'draft' for log in ai_logs)
            has_qa = any(log['node_type'] == 'qa' for log in ai_logs)
            
            if not has_draft:
                return {'step': 2, 'label': 'Generating AI draft...', 'progress': 30}
            elif not has_qa:
                return {'step': 3, 'label': 'Running quality checks...', 'progress': 60}
            else:
                return {'step': 4, 'label': 'Finalizing...', 'progress': 90}
        
        return {'step': 5, 'label': 'Complete', 'progress': 100}
    
    def _get_status_message(self, status):
        messages = {
            Request.Status.DRAFT: 'Your request is in draft mode.',
            Request.Status.SUBMITTED: 'Your request is being processed...',
            Request.Status.DRAFT_READY: 'Draft is ready for your review.',
            Request.Status.NEEDS_CLARIFICATION: 'Please answer the additional question.',
            Request.Status.NEEDS_REVIEW: 'Your request is being reviewed by our team.',
            Request.Status.APPROVED: 'Your affidavit is approved and ready!',
            Request.Status.COMPLETED: 'Your affidavit has been completed.',
            Request.Status.REJECTED: 'Your request has been rejected.',
        }
        return messages.get(status, 'Processing...')


class DevApproveView(APIView):
    """
    Developer endpoint to quickly approve a request for testing.
    Only works in DEBUG mode or for admin users.
    """
    
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        from django.conf import settings
        
        # Only allow in DEBUG mode or for admin/superuser
        if not settings.DEBUG and not request.user.is_superuser and request.user.role != 'admin':
            return Response(
                {'error': 'This endpoint is only available in development mode or for admins.'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        request_obj = get_object_or_404(Request, pk=pk)
        
        # Can approve from several states
        if request_obj.status not in [
            Request.Status.NEEDS_REVIEW, 
            Request.Status.DRAFT_READY,
            Request.Status.SUBMITTED
        ]:
            return Response(
                {'error': f'Cannot approve request in status: {request_obj.status}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Update to approved
        request_obj.status = Request.Status.APPROVED
        request_obj.approved_at = timezone.now()
        request_obj.save()
        
        # Log the event
        RequestEvent.objects.create(
            request=request_obj,
            action=RequestEvent.Action.APPROVED,
            actor=request.user,
            actor_role=request.user.role,
            details={'dev_approve': True}
        )
        
        return Response({
            'status': request_obj.status,
            'message': 'Request approved (dev mode)',
            'request_code': request_obj.request_code
        })


class ClarificationResponseView(APIView):
    """
    User responds to a clarification request.
    Re-triggers AI processing with the clarification.
    """
    
    permission_classes = [IsAuthenticated]
    
    def post(self, request, pk):
        request_obj = get_object_or_404(
            Request, 
            pk=pk, 
            user=request.user
        )
        
        if request_obj.status != Request.Status.NEEDS_CLARIFICATION:
            return Response(
                {'error': 'This request does not need clarification.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        response_text = request.data.get('response', '').strip()
        if not response_text:
            return Response(
                {'error': 'Response text is required.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Store the clarification response in user_edits_json
        user_edits = request_obj.user_edits_json or {}
        clarifications = user_edits.get('clarifications', [])
        clarifications.append({
            'question': request_obj.clarification_question,
            'response': response_text,
            'timestamp': timezone.now().isoformat()
        })
        user_edits['clarifications'] = clarifications
        request_obj.user_edits_json = user_edits
        
        # Clear the clarification question and re-submit
        request_obj.clarification_question = ''  # Set to empty string instead of None
        request_obj.status = Request.Status.SUBMITTED
        request_obj.save()
        
        # Log the event
        RequestEvent.objects.create(
            request=request_obj,
            action=RequestEvent.Action.SUBMITTED,
            actor=request.user,
            actor_role=request.user.role,
            details={'clarification_response': response_text}
        )
        
        # Re-trigger AI processing
        task = process_request_async.delay(request_obj.id)
        
        return Response({
            'status': request_obj.status,
            'message': 'Clarification received. Processing your request.',
            'task_id': task.id
        })


# =============================================================================
# Commissioner Views (Stories 2.1, 2.2, 2.3)
# =============================================================================

class CommissionerAssignedRequestsView(generics.ListAPIView):
    """
    List requests assigned to the current commissioner.
    These are requests where users selected this commissioner.
    """
    
    serializer_class = RequestListSerializer
    permission_classes = [IsCommissioner]
    
    def get_queryset(self):
        # Get requests where this commissioner is assigned (excluding completed ones)
        # Show APPROVED (ready to notarize) and DRAFT_READY only
        # Once notarized (COMPLETED), requests are removed from commissioner's view
        return Request.objects.filter(
            commissioner=self.request.user,
            status__in=[Request.Status.APPROVED, Request.Status.DRAFT_READY]
        ).order_by('-created_at')


class MarkCompleteView(APIView):
    """
    Mark a request as completed after stamping (Story 2.2).
    Creates a Stamp record and freezes the request.
    Only the assigned commissioner can complete the request.
    """
    
    permission_classes = [IsCommissioner]
    
    @transaction.atomic
    def post(self, request, pk):
        request_obj = get_object_or_404(Request, pk=pk)
        
        # Check if assigned to this commissioner
        if request_obj.commissioner and request_obj.commissioner.id != request.user.id:
            return Response(
                {'error': 'This request is assigned to another commissioner.'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        serializer = MarkCompleteSerializer(
            data=request.data,
            context={'request_obj': request_obj}
        )
        serializer.is_valid(raise_exception=True)
        
        commissioner = request.user
        
        # Get global payout amount from site settings
        payout_amount = SiteSettings.get_payout_amount()
        
        # Create stamp record
        stamp = Stamp.objects.create(
            request=request_obj,
            commissioner=commissioner,
            payout_amount=payout_amount,
            notes=serializer.validated_data.get('notes', '')
        )
        
        # Update request status
        request_obj.status = Request.Status.COMPLETED
        request_obj.completed_at = timezone.now()
        request_obj.release_lock()
        request_obj.save()
        
        # Send completion notification to user
        send_completion_notification(request_obj, stamp)
        
        # Generate payout message for commissioner
        from .services.notification_service import send_commissioner_payout_added_message
        payout_message = send_commissioner_payout_added_message(
            commissioner, payout_amount, request_obj.request_code
        )
        
        return Response({
            'success': True,
            'message': 'Request marked as completed',
            'stamp': StampSerializer(stamp).data,
            'payout_message': payout_message
        })


class FrictionReportCreateView(generics.CreateAPIView):
    """
    Report an issue with a document (Story 2.3).
    Logs friction event for dashboard tracking.
    """
    
    serializer_class = FrictionReportCreateSerializer
    permission_classes = [IsCommissioner]
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context


class CommissionerStampsView(generics.ListAPIView):
    """List stamps/completions for the current commissioner."""
    
    serializer_class = StampSerializer
    permission_classes = [IsCommissioner]
    
    def get_queryset(self):
        return Stamp.objects.filter(commissioner=self.request.user)


class CommissionerPDFPreferencesView(APIView):
    """
    Get or update commissioner's PDF formatting preferences.
    """
    
    permission_classes = [IsCommissioner]
    
    def get(self, request):
        """Get current PDF preferences."""
        prefs = request.user.pdf_preferences or {}
        return Response({
            'pdf_preferences': prefs,
            'defaults': {
                'letterhead': {'enabled': False, 'text': ''},
                'page_size': 'letter',
                'signature_spacing': 'normal',
                'show_commission_number': True,
                'custom_footer': ''
            }
        })
    
    def patch(self, request):
        """Update PDF preferences."""
        from .serializers import CommissionerPDFPreferencesSerializer
        
        serializer = CommissionerPDFPreferencesSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Merge with existing preferences
        current_prefs = request.user.pdf_preferences or {}
        current_prefs.update(serializer.validated_data)
        
        request.user.pdf_preferences = current_prefs
        request.user.save(update_fields=['pdf_preferences'])
        
        return Response({
            'success': True,
            'pdf_preferences': current_prefs
        })


# =============================================================================
# Reviewer Views (Stories 3.1, 3.2, 3.3)
# =============================================================================

class ReviewQueueView(generics.ListAPIView):
    """
    Prioritized review queue (Story 3.1).
    Shows requests needing review, sorted by newest first (LIFO) or risk score.
    """
    
    serializer_class = RequestReviewerSerializer
    permission_classes = [IsReviewer]
    
    def get_queryset(self):
        # Sort by newest created first (descending order)
        return Request.objects.filter(
            status=Request.Status.NEEDS_REVIEW
        ).order_by('-created_at')


class ReviewDetailView(generics.RetrieveAPIView):
    """
    Get detailed view of a request for review (Story 3.2).
    Includes QA flags for highlighting issues.
    """
    
    serializer_class = RequestDetailSerializer
    permission_classes = [IsReviewer]
    queryset = Request.objects.filter(status=Request.Status.NEEDS_REVIEW)


class ApproveRequestView(APIView):
    """
    Approve a request after review (Story 3.3).
    Updates status, sends notification, logs edits.
    Tracks significant edits for learning loop.
    """
    
    permission_classes = [IsReviewer]
    
    @transaction.atomic
    def post(self, request, pk):
        request_obj = get_object_or_404(
            Request, 
            pk=pk,
            status=Request.Status.NEEDS_REVIEW
        )
        
        serializer = ApproveRequestSerializer(
            data=request.data,
            context={'request_obj': request_obj}
        )
        serializer.is_valid(raise_exception=True)
        
        reviewer = request.user
        final_text = serializer.validated_data.get('final_text')
        
        # Log edit if text was changed
        if final_text and final_text != request_obj.draft_text:
            # Check if edit is significant (> 10% change)
            original_len = len(request_obj.draft_text)
            edit_len = len(final_text)
            change_ratio = abs(edit_len - original_len) / max(original_len, 1)
            
            is_significant = change_ratio > 0.10 or self._has_substantial_changes(
                request_obj.draft_text, final_text
            )
            
            ReviewerEdit.objects.create(
                request=request_obj,
                reviewer=reviewer,
                original_text=request_obj.draft_text,
                edited_text=final_text,
                issue_type=serializer.validated_data.get('issue_type', 'other'),
                issue_description=serializer.validated_data.get('issue_description', ''),
                ai_flag_accepted=True
            )
            
            request_obj.final_text = final_text
            request_obj.draft_edited_significantly = is_significant
            request_obj.user_edits_json = {
                'reviewer': reviewer.username,
                'change_ratio': round(change_ratio, 2),
                'timestamp': timezone.now().isoformat()
            }
        else:
            request_obj.final_text = request_obj.draft_text
            request_obj.draft_edited_significantly = False
        
        # Save feedback entries (learning loop)
        feedback_entries = serializer.validated_data.get('feedback_entries', [])
        auto_feedback_pairs = serializer.validated_data.get('auto_feedback_pairs', [])
        if feedback_entries or auto_feedback_pairs:
            from .services.feedback_service import save_feedback_entries, save_auto_feedback_pairs
            if feedback_entries:
                save_feedback_entries(request_obj, reviewer, feedback_entries)
            if auto_feedback_pairs:
                save_auto_feedback_pairs(request_obj, reviewer, auto_feedback_pairs)
        
        # Update status
        request_obj.status = Request.Status.APPROVED
        request_obj.approved_at = timezone.now()
        request_obj.save()
        
        # Log event
        RequestEvent.objects.create(
            request=request_obj,
            action=RequestEvent.Action.APPROVED,
            actor=reviewer,
            actor_role='reviewer',
            details={
                'edited': bool(final_text and final_text != request_obj.draft_text),
                'edit_significant': request_obj.draft_edited_significantly
            }
        )
        
        # Generate PDF async
        generate_pdf_async.delay(request_obj.id)
        
        # Send notification async (email)
        send_notification_async.delay('approval', request_obj.id)
        
        # Send SMS/WhatsApp notification
        from .services.notification_service import send_request_approved_sms
        try:
            send_request_approved_sms(request_obj)
        except Exception as e:
            logger.warning(f"Failed to send approval SMS notification: {e}")
        
        return Response({
            'success': True,
            'message': 'Request approved and user notified',
            'request_code': request_obj.request_code,
            'pdf_generating': True
        })
    
    def _has_substantial_changes(self, original: str, edited: str) -> bool:
        """
        Check for substantial content changes beyond simple typos.
        Uses simple word-level comparison.
        """
        original_words = set(original.lower().split())
        edited_words = set(edited.lower().split())
        
        # Check symmetric difference (words added or removed)
        diff = original_words.symmetric_difference(edited_words)
        total = len(original_words.union(edited_words))
        
        return len(diff) / max(total, 1) > 0.15


class RejectRequestView(APIView):
    """Reject a request (with reason)."""
    
    permission_classes = [IsReviewer]
    
    def post(self, request, pk):
        request_obj = get_object_or_404(
            Request,
            pk=pk,
            status=Request.Status.NEEDS_REVIEW
        )
        
        reviewer = request.user
        reason = request.data.get('reason', '')
        
        request_obj.status = Request.Status.REJECTED
        request_obj.qa_flags_json.append({
            'type': 'rejection',
            'description': reason,
            'reviewer': reviewer.username,
            'timestamp': timezone.now().isoformat()
        })
        request_obj.save()
        
        # Log event for reject action
        RequestEvent.objects.create(
            request=request_obj,
            action=RequestEvent.Action.REJECTED,
            actor=reviewer,
            actor_role='reviewer',
            details={
                'reason': reason,
                'rejected_at': timezone.now().isoformat()
            }
        )
        
        # Send rejection notification to user
        from .services.notification_service import send_request_rejected_notification
        try:
            send_request_rejected_notification(request_obj, reason)
        except Exception as e:
            logger.warning(f"Failed to send rejection notification: {e}")
        
        return Response({
            'success': True,
            'message': 'Request rejected'
        })


class OverrideAIFlagView(APIView):
    """
    Handle AI flag feedback (Story 3.2).
    Reviewer marks whether AI was correct or incorrect.
    Tracks feedback for learning loop analysis.
    """
    
    permission_classes = [IsReviewer]
    
    def post(self, request, pk):
        request_obj = get_object_or_404(Request, pk=pk)
        flag_index = request.data.get('flag_index')
        override_reason = request.data.get('reason', '')
        ai_correct = request.data.get('ai_correct', False)  # New field
        
        if flag_index is None or flag_index >= len(request_obj.qa_flags_json):
            return Response(
                {'error': 'Invalid flag index'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Mark flag as reviewed
        request_obj.qa_flags_json[flag_index]['overridden'] = True
        request_obj.qa_flags_json[flag_index]['overridden_by'] = request.user.username
        request_obj.qa_flags_json[flag_index]['override_reason'] = override_reason
        request_obj.qa_flags_json[flag_index]['ai_correct'] = ai_correct  # Track if AI was right
        
        # Track feedback for learning loop
        if not ai_correct:
            # AI was wrong - this is a false positive
            request_obj.qa_overridden = True
            if override_reason:
                request_obj.override_notes = f"{request_obj.override_notes}\nFalse positive: {override_reason}".strip()
        else:
            # AI was correct - the issue was real
            if override_reason:
                request_obj.override_notes = f"{request_obj.override_notes}\nAI correct: {override_reason}".strip()
        
        request_obj.save()
        
        # Log the event
        action_type = 'ai_correct' if ai_correct else 'ai_wrong'
        RequestEvent.objects.create(
            request=request_obj,
            action=RequestEvent.Action.EDITED,
            actor=request.user,
            actor_role='reviewer',
            details={
                'action_type': action_type,
                'flag_index': flag_index,
                'reason': override_reason,
                'ai_correct': ai_correct
            }
        )
        
        # Log for learning
        ReviewerEdit.objects.create(
            request=request_obj,
            reviewer=request.user,
            original_text=str(request_obj.qa_flags_json[flag_index]),
            edited_text='AI CORRECT' if ai_correct else 'AI WRONG - FALSE POSITIVE',
            issue_type=ReviewerEdit.IssueType.OTHER,
            issue_description=f'AI {"correct" if ai_correct else "wrong"}: {override_reason}',
            ai_flag_accepted=ai_correct
        )
        
        return Response({
            'success': True,
            'message': f'Flag marked as AI {"correct" if ai_correct else "incorrect"}'
        })


class ReviewerStatsView(APIView):
    """
    Get reviewer statistics for dashboard.
    Returns counts and metrics for the current reviewer.
    """

    permission_classes = [IsReviewer]

    def get(self, request):
        reviewer = request.user
        today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)

        pending_count = Request.objects.filter(
            status=Request.Status.NEEDS_REVIEW
        ).count()

        # Get today's stats from RequestEvent
        today_events = RequestEvent.objects.filter(
            actor=reviewer,
            created_at__gte=today_start
        )

        # Count reviews by action type
        approved_today = today_events.filter(
            action=RequestEvent.Action.APPROVED
        ).count()

        rejected_today = today_events.filter(
            action=RequestEvent.Action.REJECTED
        ).count()

        # Clarification requests (status changed to needs_clarification)
        clarification_today = Request.objects.filter(
            status=Request.Status.NEEDS_CLARIFICATION,
            updated_at__gte=today_start
        ).count()

        reviewed_today = approved_today + rejected_today + clarification_today

        # Calculate average review time (from created_at to approved_at)
        avg_time = Request.objects.filter(
            status=Request.Status.APPROVED,
            approved_at__gte=today_start
        ).annotate(
            review_time=F('approved_at') - F('created_at')
        ).aggregate(
            avg_time=Avg('review_time')
        )['avg_time']

        avg_minutes = 0
        if avg_time:
            avg_minutes = round(avg_time.total_seconds() / 60, 1)

        # Calculate approval rate (last 7 days)
        week_ago = timezone.now() - timedelta(days=7)
        week_approved = Request.objects.filter(
            status=Request.Status.APPROVED,
            approved_at__gte=week_ago
        ).count()
        week_total = Request.objects.filter(
            status__in=[Request.Status.APPROVED, Request.Status.REJECTED],
            updated_at__gte=week_ago
        ).count()

        approval_rate = round((week_approved / week_total * 100), 1) if week_total > 0 else 100

        return Response({
            'pending_count': pending_count,
            'reviewed_today': reviewed_today,
            'approved_today': approved_today,
            'rejected_today': rejected_today,
            'clarification_today': clarification_today,
            'avg_review_time_minutes': avg_minutes,
            'approval_rate': approval_rate
        })


class RequestClarificationView(APIView):
    """
    Request clarification from user (Story 3.2).
    Sets status to needs_clarification and stores the question.
    """
    
    permission_classes = [IsReviewer]
    
    def post(self, request, pk):
        request_obj = get_object_or_404(
            Request,
            pk=pk,
            status=Request.Status.NEEDS_REVIEW
        )
        
        question = request.data.get('question', '').strip()
        if not question:
            return Response(
                {'error': 'Clarification question is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Update request status and store question
        request_obj.status = Request.Status.NEEDS_CLARIFICATION
        request_obj.clarification_question = question
        request_obj.save()
        
        # Log event
        RequestEvent.objects.create(
            request=request_obj,
            action=RequestEvent.Action.EDITED,
            actor=request.user,
            actor_role='reviewer',
            details={
                'action_type': 'clarification_requested',
                'question': question
            }
        )
        
        # Send notification to user
        send_notification_async.delay('clarification', request_obj.id)
        
        return Response({
            'success': True,
            'message': 'Clarification request sent to user',
            'request_code': request_obj.request_code
        })


# =============================================================================
# Admin Views (Stories 4.1, 4.2, 4.3)
# =============================================================================

class AffidavitTypePolicyView(generics.UpdateAPIView):
    """
    Update policy for an affidavit type (Story 4.1).
    Increments version automatically.
    """
    
    serializer_class = AffidavitTypePolicyUpdateSerializer
    permission_classes = [IsAdminUser]
    queryset = AffidavitType.objects.all()


class ValidationRulesView(APIView):
    """
    GET/PUT validation rules for a specific affidavit type.
    Rules are stored as a JSON list on AffidavitType.validation_rules.
    """
    permission_classes = [IsAdminUser]

    def get(self, request, pk):
        try:
            affidavit_type = AffidavitType.objects.get(pk=pk)
        except AffidavitType.DoesNotExist:
            return Response({'error': 'Affidavit type not found'}, status=404)
        return Response({
            'affidavit_type_id': affidavit_type.id,
            'affidavit_type_name': affidavit_type.name,
            'validation_rules': affidavit_type.validation_rules or []
        })

    def put(self, request, pk):
        try:
            affidavit_type = AffidavitType.objects.get(pk=pk)
        except AffidavitType.DoesNotExist:
            return Response({'error': 'Affidavit type not found'}, status=404)

        rules = request.data.get('validation_rules', [])
        if not isinstance(rules, list):
            return Response({'error': 'validation_rules must be a list'}, status=400)

        # Basic validation of each rule
        valid_types = {'comparison', 'required_if', 'disallow_contains'}
        valid_operators = {'gte', 'lte', 'gt', 'lt', 'eq', 'ne'}
        valid_compare_as = {'number', 'date', 'string'}
        valid_join_with = {'AND', 'OR'}
        valid_contains_mode = {'contains', 'regex'}

        for idx, rule in enumerate(rules):
            if not isinstance(rule, dict):
                return Response({'error': f'Rule at index {idx} must be an object'}, status=400)
            rule_type = rule.get('type', '')
            if rule_type not in valid_types:
                return Response({'error': f'Rule at index {idx} has invalid type "{rule_type}". Valid: {list(valid_types)}'}, status=400)
            if not rule.get('message'):
                return Response({'error': f'Rule at index {idx} is missing a message'}, status=400)

            if rule_type == 'comparison':
                comparisons = rule.get('comparisons')
                if comparisons is not None:
                    if not isinstance(comparisons, list) or len(comparisons) == 0:
                        return Response({'error': f'Comparison rule at index {idx} comparisons must be a non-empty list'}, status=400)
                    for c_idx, clause in enumerate(comparisons):
                        if not isinstance(clause, dict):
                            return Response({'error': f'Comparison rule at index {idx}, clause {c_idx} must be an object'}, status=400)
                        if not clause.get('left_field') or not clause.get('right_field'):
                            return Response({'error': f'Comparison rule at index {idx}, clause {c_idx} requires left_field and right_field'}, status=400)
                        if clause.get('operator', 'gte') not in valid_operators:
                            return Response({'error': f'Comparison rule at index {idx}, clause {c_idx} has invalid operator'}, status=400)
                        compare_as = clause.get('compare_as', rule.get('compare_as', 'number'))
                        if compare_as not in valid_compare_as:
                            return Response({'error': f'Comparison rule at index {idx}, clause {c_idx} has invalid compare_as'}, status=400)
                        if c_idx > 0:
                            join_with = str(clause.get('join_with', 'AND')).strip().upper()
                            if join_with not in valid_join_with:
                                return Response({'error': f'Comparison rule at index {idx}, clause {c_idx} has invalid join_with (use AND/OR)'}, status=400)
                else:
                    if not rule.get('primary_field') or not rule.get('secondary_field'):
                        return Response({'error': f'Comparison rule at index {idx} requires primary_field and secondary_field'}, status=400)
                    if rule.get('operator', 'gte') not in valid_operators:
                        return Response({'error': f'Rule at index {idx} has invalid operator'}, status=400)
                    if rule.get('compare_as', 'number') not in valid_compare_as:
                        return Response({'error': f'Rule at index {idx} has invalid compare_as'}, status=400)

            elif rule_type == 'required_if':
                if not rule.get('condition_field') or not rule.get('required_field'):
                    return Response({'error': f'Required_if rule at index {idx} requires condition_field and required_field'}, status=400)

            elif rule_type == 'disallow_contains':
                if not rule.get('field') and not rule.get('primary_field'):
                    return Response({'error': f'Disallow_contains rule at index {idx} requires field'}, status=400)
                if not rule.get('pattern'):
                    return Response({'error': f'Disallow_contains rule at index {idx} requires pattern'}, status=400)
                mode = rule.get('mode', 'contains')
                if mode not in valid_contains_mode:
                    return Response({'error': f'Disallow_contains rule at index {idx} has invalid mode (use contains/regex)'}, status=400)

        affidavit_type.validation_rules = rules
        affidavit_type.save(update_fields=['validation_rules', 'updated_at'])

        return Response({
            'affidavit_type_id': affidavit_type.id,
            'validation_rules': affidavit_type.validation_rules
        })


class PlaceholderMappingView(APIView):
    """
    Template-First Intake Builder: Placeholder ↔ Question Mapping.
    
    GET  - Extract placeholders from template_html, compare against intake_schema,
           return audit report with mapped/unmapped/orphaned status.
    PUT  - Save placeholder_mapping and optionally auto-create missing questions.
    """
    permission_classes = [IsAdminUser]

    # Auto-computed placeholders that don't need a question
    AUTO_COMPUTED = {'calculated_age', 'current_date', 'current_year', 'current_month', 'current_day'}
    # Questions that feed an auto-computed field — exempt from orphan flagging
    # when their derived auto placeholder appears in the template.
    AUTO_COMPUTED_SOURCES = {'calculated_age': 'date_of_birth'}

    @staticmethod
    def _extract_placeholders(template_html: str) -> list:
        """Extract unique {{placeholder}} tokens from template HTML, preserving order."""
        import re
        return list(dict.fromkeys(re.findall(r'\{\{(\w+)\}\}', template_html or '')))

    def _build_audit(self, affidavit_type):
        """Build the full audit report comparing placeholders vs questions."""
        placeholders = self._extract_placeholders(affidavit_type.template_html)
        questions = affidavit_type.intake_schema or []
        mapping = affidavit_type.placeholder_mapping or {}

        # Build question lookup by id and field_name
        q_by_id = {}
        for q in questions:
            qid = q.get('id', '')
            q_by_id[qid] = q
            fname = q.get('field_name', '')
            if fname and fname != qid:
                q_by_id[fname] = q

        # Build audit entries for each placeholder
        entries = []
        mapped_question_ids = set()
        for ph in placeholders:
            is_auto = ph in self.AUTO_COMPUTED
            mapped_qid = mapping.get(ph, '')

            # Try to resolve: explicit mapping → same-name question → auto
            resolved_q = None
            if mapped_qid:
                resolved_q = q_by_id.get(mapped_qid)
            if not resolved_q and not is_auto:
                resolved_q = q_by_id.get(ph)

            if is_auto:
                status = 'auto'
                entry = {
                    'placeholder': ph,
                    'status': status,
                    'mapped_question_id': None,
                    'question_label': None,
                    'question_type': None,
                    'note': 'Auto-computed by system',
                }
            elif resolved_q:
                status = 'mapped'
                qid = resolved_q.get('id', '')
                mapped_question_ids.add(qid)
                entry = {
                    'placeholder': ph,
                    'status': status,
                    'mapped_question_id': qid,
                    'question_label': resolved_q.get('label', ''),
                    'question_type': resolved_q.get('type', 'text'),
                    'note': None,
                }
            else:
                status = 'unmapped'
                entry = {
                    'placeholder': ph,
                    'status': status,
                    'mapped_question_id': None,
                    'question_label': None,
                    'question_type': None,
                    'note': 'No intake question collects this data',
                }
            entries.append(entry)

        # Find orphaned questions (exist in intake_schema but no placeholder uses them)
        all_question_ids = {q.get('id', '') for q in questions}
        used_question_ids = mapped_question_ids | {
            ph for ph in placeholders if ph in all_question_ids
        }
        # Also exempt source questions whose auto-computed derivative is used in the template
        for auto_ph, source_qid in self.AUTO_COMPUTED_SOURCES.items():
            if auto_ph in placeholders:
                used_question_ids.add(source_qid)
        orphaned = []
        for q in questions:
            qid = q.get('id', '')
            if qid and qid not in used_question_ids:
                orphaned.append({
                    'question_id': qid,
                    'question_label': q.get('label', ''),
                    'question_type': q.get('type', 'text'),
                    'note': 'Question exists but no template placeholder uses it',
                })

        # Summary counts
        mapped_count = sum(1 for e in entries if e['status'] == 'mapped')
        unmapped_count = sum(1 for e in entries if e['status'] == 'unmapped')
        auto_count = sum(1 for e in entries if e['status'] == 'auto')

        return {
            'affidavit_type_id': affidavit_type.id,
            'affidavit_type_name': affidavit_type.name,
            'total_placeholders': len(placeholders),
            'summary': {
                'mapped': mapped_count,
                'unmapped': unmapped_count,
                'auto_computed': auto_count,
                'orphaned_questions': len(orphaned),
            },
            'entries': entries,
            'orphaned_questions': orphaned,
            'placeholder_mapping': mapping,
            'questions': [
                {'id': q.get('id', ''), 'label': q.get('label', ''), 'type': q.get('type', 'text')}
                for q in questions
            ],
        }

    def get(self, request, pk):
        try:
            affidavit_type = AffidavitType.objects.get(pk=pk)
        except AffidavitType.DoesNotExist:
            return Response({'error': 'Affidavit type not found'}, status=status.HTTP_404_NOT_FOUND)

        return Response(self._build_audit(affidavit_type))

    def put(self, request, pk):
        """
        Save placeholder_mapping and optionally auto-create questions for unmapped placeholders.
        
        Body:
          {
            "placeholder_mapping": {"full_name": "full_name", ...},
            "auto_create_questions": ["property_description", ...]  // optional
          }
        """
        try:
            affidavit_type = AffidavitType.objects.get(pk=pk)
        except AffidavitType.DoesNotExist:
            return Response({'error': 'Affidavit type not found'}, status=status.HTTP_404_NOT_FOUND)

        mapping = request.data.get('placeholder_mapping')
        if mapping is not None:
            if not isinstance(mapping, dict):
                return Response({'error': 'placeholder_mapping must be a dict'}, status=status.HTTP_400_BAD_REQUEST)
            affidavit_type.placeholder_mapping = mapping

        # Auto-create questions for specified unmapped placeholders
        auto_create = request.data.get('auto_create_questions', [])
        created_questions = []
        if auto_create and isinstance(auto_create, list):
            existing_ids = {q.get('id', '') for q in (affidavit_type.intake_schema or [])}
            schema = list(affidavit_type.intake_schema or [])

            for placeholder_id in auto_create:
                if not isinstance(placeholder_id, str) or not placeholder_id.strip():
                    continue
                pid = placeholder_id.strip()
                if pid in existing_ids or pid in self.AUTO_COMPUTED:
                    continue

                # Generate a sensible label from the placeholder id
                label = ' '.join(word.capitalize() for word in pid.split('_'))
                new_q = {
                    'id': pid,
                    'label': label,
                    'type': 'text',
                    'required': True,
                    'placeholder': '',
                    'help_text': f'Enter your {label.lower()}.',
                }
                schema.append(new_q)
                existing_ids.add(pid)
                created_questions.append(pid)

                # Also add to mapping
                if isinstance(affidavit_type.placeholder_mapping, dict):
                    affidavit_type.placeholder_mapping[pid] = pid

            if created_questions:
                affidavit_type.intake_schema = schema

        affidavit_type.save(update_fields=['placeholder_mapping', 'intake_schema', 'updated_at'])

        audit = self._build_audit(affidavit_type)
        audit['created_questions'] = created_questions
        return Response(audit)


class TemplateLivePreviewView(APIView):
    """
    Template Fill Preview: Accept sample answers and return the filled template HTML (deterministic).
    POST /admin/affidavit-types/<pk>/template-preview/
    Body: { "sample_answers": { "full_name": "John Smith", ... } }
    """
    permission_classes = [IsAdminUser]

    def post(self, request, pk):
        from .services.ai_service import pre_fill_template

        try:
            affidavit_type = AffidavitType.objects.get(pk=pk)
        except AffidavitType.DoesNotExist:
            return Response({'error': 'Affidavit type not found'}, status=status.HTTP_404_NOT_FOUND)

        template = affidavit_type.template_html or ''
        if not template:
            return Response({'error': 'No template HTML found for this type'}, status=status.HTTP_400_BAD_REQUEST)

        sample_answers = request.data.get('sample_answers', {})
        if not isinstance(sample_answers, dict):
            return Response({'error': 'sample_answers must be a dict'}, status=status.HTTP_400_BAD_REQUEST)

        filled_html = pre_fill_template(
            template_html=template,
            answers_json=sample_answers,
            placeholder_mapping=affidavit_type.placeholder_mapping or {},
            intake_schema=affidavit_type.intake_schema or [],
        )

        # Count what got replaced vs what's still pending
        import re
        remaining = re.findall(r'\{\{(\w+)\}\}', filled_html)

        return Response({
            'filled_html': filled_html,
            'remaining_placeholders': list(dict.fromkeys(remaining)),
            'total_placeholders': len(re.findall(r'\{\{(\w+)\}\}', template)),
            'filled_count': len(re.findall(r'\{\{(\w+)\}\}', template)) - len(set(remaining)),
        })


class AIDraftPreviewView(APIView):
    """
    AI Draft Preview: Run the full draft_affidavit flow with sample answers.
    Returns the actual AI-generated affidavit (same as production flow).
    POST /admin/affidavit-types/<pk>/ai-draft-preview/
    Body: { "sample_answers": { "full_name": "John Smith", ... } }
    """
    permission_classes = [IsAdminUser]

    def post(self, request, pk):
        from .services.ai_service import draft_affidavit
        import time

        try:
            affidavit_type = AffidavitType.objects.get(pk=pk)
        except AffidavitType.DoesNotExist:
            return Response({'error': 'Affidavit type not found'}, status=status.HTTP_404_NOT_FOUND)

        sample_answers = request.data.get('sample_answers', {})
        if not isinstance(sample_answers, dict):
            return Response({'error': 'sample_answers must be a dict'}, status=status.HTTP_400_BAD_REQUEST)

        # Validate we have enough data to draft
        if not sample_answers:
            return Response({'error': 'sample_answers cannot be empty'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            start_time = time.time()
            
            # Run the full AI drafting flow (same as production)
            draft_result = draft_affidavit(
                answers_json=sample_answers,
                policy_json=affidavit_type.policy_json or {},
                affidavit_type_name=affidavit_type.name,
                scenario_library=affidavit_type.scenario_library or [],
                template_html=affidavit_type.template_html or '',
                disallowed_phrases=affidavit_type.disallowed_phrases or [],
                placeholder_mapping=affidavit_type.placeholder_mapping or {},
                intake_schema=affidavit_type.intake_schema or [],
                affidavit_type_id=affidavit_type.id,
            )
            
            elapsed_time = time.time() - start_time

            if not draft_result.get('success'):
                return Response({
                    'error': draft_result.get('error', 'Draft generation failed'),
                    'elapsed_time': elapsed_time,
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({
                'draft_html': draft_result.get('draft_html', ''),
                'warnings': draft_result.get('warnings', []),
                'model_used': draft_result.get('model_used', 'unknown'),
                'elapsed_time': elapsed_time,
                'prompt_tokens': draft_result.get('prompt_tokens', 0),
                'completion_tokens': draft_result.get('completion_tokens', 0),
                'total_tokens': draft_result.get('total_tokens', 0),
            })

        except Exception as e:
            logger.error(f"AI draft preview failed: {e}")
            return Response({
                'error': f'Draft generation error: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FieldSuggestionsView(APIView):
    """
    Smart field suggestions for template editor.
    
    GET /admin/affidavit-types/<pk>/field-suggestions/
    Query params:
      - selected_text: text user selected
      - context_before: text before selection (up to 200 chars)
      - context_after: text after selection (up to 200 chars)
    
    Returns:
      {
        "ai_suggestions": [...],           // AI-powered suggestions based on context
        "unused_universal_fields": [...],  // UNIVERSAL_FIELDS not yet in schema
        "unused_common_fields": [...],     // Common FIELD_DEFAULTS not yet in schema
        "existing_fields": [...]            // Already in schema
      }
    """
    permission_classes = [IsAdminUser]

    def get(self, request, pk):
        from .services.policy_generator_service import FIELD_DEFAULTS, UNIVERSAL_FIELDS, FIELD_ALIASES
        from .services.ai_service import get_openai_client
        
        try:
            affidavit_type = AffidavitType.objects.get(pk=pk)
        except AffidavitType.DoesNotExist:
            return Response({'error': 'Affidavit type not found'}, status=status.HTTP_404_NOT_FOUND)

        existing_ids = {q.get('id', '').lower() for q in (affidavit_type.intake_schema or [])}
        
        # AI-powered suggestions if context provided
        ai_suggestions = []
        selected_text = request.query_params.get('selected_text', '').strip()
        context_before = request.query_params.get('context_before', '').strip()
        context_after = request.query_params.get('context_after', '').strip()
        
        if selected_text and (context_before or context_after):
            # Build available fields context
            available_fields = []
            for field_id, defaults in FIELD_DEFAULTS.items():
                if field_id not in existing_ids:
                    available_fields.append(f"- {field_id}: {defaults.get('label', field_id)}")
            
            available_fields_text = "\n".join(available_fields[:30]) if available_fields else "No predefined fields available"
            
            prompt = f"""Analyze this template text and suggest the most appropriate field to replace the selected text.

CONTEXT BEFORE: {context_before[-200:]}
SELECTED TEXT: "{selected_text}"
CONTEXT AFTER: {context_after[:200]}

AVAILABLE PREDEFINED FIELDS:
{available_fields_text}

INSTRUCTIONS:
1. Suggest the BEST field_id from available fields, OR create a new one if none fit
2. Consider the semantic meaning and context
3. Return ONLY JSON with top 3 suggestions:

[
  {{
    "field_id": "property_address",
    "confidence": 0.95,
    "label": "Property Address",
    "type": "text",
    "reason": "Context indicates physical location details"
  }},
  ...
]"""

            try:
                client = get_openai_client()
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500,
                )
                
                import json
                suggestions_text = response.choices[0].message.content.strip()
                # Extract JSON from markdown code blocks if present
                if '```json' in suggestions_text:
                    suggestions_text = suggestions_text.split('```json')[1].split('```')[0].strip()
                elif '```' in suggestions_text:
                    suggestions_text = suggestions_text.split('```')[1].split('```')[0].strip()
                
                ai_suggestions = json.loads(suggestions_text)
                
                # Enrich with help_text from FIELD_DEFAULTS if available
                for sug in ai_suggestions:
                    field_id = sug.get('field_id', '')
                    canonical = FIELD_ALIASES.get(field_id, field_id)
                    if canonical in FIELD_DEFAULTS:
                        defaults = FIELD_DEFAULTS[canonical]
                        sug['help_text'] = defaults.get('help_text', '')
                        if not sug.get('type'):
                            sug['type'] = defaults.get('type', 'text')
                    sug['category'] = 'ai_suggested'
                    
            except Exception as e:
                logger.warning(f"AI suggestion failed: {e}")
                # Fallback to basic suggestion
                suggested_id = selected_text.lower().replace(' ', '_').replace('-', '_')
                suggested_id = ''.join(c for c in suggested_id if c.isalnum() or c == '_')
                if suggested_id and not suggested_id[0].isdigit():
                    ai_suggestions = [{
                        'field_id': suggested_id,
                        'confidence': 0.5,
                        'label': selected_text.title(),
                        'type': 'text',
                        'reason': 'Basic text-based suggestion',
                        'category': 'ai_suggested',
                    }]
        
        # Universal fields not yet added
        unused_universal = []
        for uf in UNIVERSAL_FIELDS:
            if uf['id'].lower() not in existing_ids:
                unused_universal.append({
                    'id': uf['id'],
                    'label': uf['label'],
                    'type': uf['type'],
                    'help_text': uf.get('help_text', ''),
                    'category': 'universal',
                })
        
        # Common fields from FIELD_DEFAULTS not yet added
        unused_common = []
        for field_id, defaults in FIELD_DEFAULTS.items():
            if field_id not in existing_ids:
                unused_common.append({
                    'id': field_id,
                    'label': defaults.get('label', field_id.replace('_', ' ').title()),
                    'type': defaults.get('type', 'text'),
                    'help_text': defaults.get('help_text', ''),
                    'category': 'common',
                })
        
        # Existing fields
        existing_fields = [
            {
                'id': q.get('id', ''),
                'label': q.get('label', ''),
                'type': q.get('type', 'text'),
            }
            for q in (affidavit_type.intake_schema or [])
        ]
        
        return Response({
            'ai_suggestions': ai_suggestions,
            'unused_universal_fields': unused_universal,
            'unused_common_fields': unused_common[:20],  # Limit to top 20
            'existing_fields': existing_fields,
        })

    def post(self, request, pk):
        """
        Generate a single field config from a natural language description.
        Uses template_documents as AI reference context.

        POST body: { "description": "applicant's full name", "context_html": "..." }
        Returns:   { "field_id", "label", "type", "help_text", "placeholder", "reason" }
        """
        import re as _re
        import json as _json
        from .services.ai_service import get_openai_client

        try:
            affidavit_type = AffidavitType.objects.get(pk=pk)
        except AffidavitType.DoesNotExist:
            return Response({'error': 'Affidavit type not found'}, status=status.HTTP_404_NOT_FOUND)

        description = request.data.get('description', '').strip()
        context_html = request.data.get('context_html', '').strip()
        full_template = request.data.get('full_template', '').strip()

        if not description:
            return Response({'error': 'description is required'}, status=status.HTTP_400_BAD_REQUEST)

        existing_ids = {q.get('id', '').lower() for q in (affidavit_type.intake_schema or [])}

        # Build context from uploaded template_documents (up to 3 docs, 2000 chars each)
        template_docs = affidavit_type.template_documents or []
        docs_context = ''
        if template_docs:
            snippets = []
            for doc in template_docs[:3]:
                html = doc.get('html_content', '')
                if html:
                    text = _re.sub(r'<[^>]+>', ' ', html)
                    text = _re.sub(r'\s+', ' ', text).strip()[:2000]
                    snippets.append(f"=== {doc.get('filename', 'Document')} ===\n{text}")
            docs_context = '\n\n'.join(snippets)

        existing_context = ', '.join(sorted(existing_ids)) if existing_ids else 'none'

        if '[INSERTION_POINT]' not in context_html and '[REPLACE_START]' not in context_html:
            context_html += ' [INSERTION_POINT]'

        # Strip HTML tags from full_template for contradiction check
        full_template_text = _re.sub(r'<[^>]+>', ' ', full_template)
        full_template_text = _re.sub(r'\s+', ' ', full_template_text).strip()[:4000]

        prompt = f"""You are an expert legal document drafter assisting in creating dynamic affidavit templates.

Your goal is to suggest a smart field definition that seamlessly integrates into the existing sentence structure, WITHOUT creating any logical contradictions.

FIELD REQUEST: "{description}"
{"" if not context_html else "TEMPLATE SNIPPET (The text around the insertion point):\n" + context_html[:1000]}
{"" if not full_template_text else "FULL TEMPLATE (read this to detect contradictions anywhere in the document):\n" + full_template_text}
{"" if not docs_context else "REFERENCE DOCUMENTS (Style guide/Context):\n" + docs_context}
EXISTING FIELD IDs (do not reuse): {existing_context}

UNDERSTANDING THE MARKERS:
- '[INSERTION_POINT]' = the exact cursor position. The field will be INSERTED here. You must supply prefix/suffix so the sentence reads correctly.
- '[REPLACE_START]...[REPLACE_END]' = the user SELECTED this text to give you context about WHERE the field should go. See REPLACE rules below.

⚠️  LOGICAL CONTRADICTION CHECK — READ THIS FIRST:
Before generating anything, scan the ENTIRE template snippet for negation phrases near the insertion point:
- Phrases like: "I have no", "I do not have", "I don't have", "no other", "none", "nothing", "not any", "save for", "except for" that refer to the SAME concept as the requested field.
- If such a phrase EXISTS near the marker, blindly adding the field would create a contradiction.
  Example of BAD output: snippet says "I have no other property" and field {{other_property}} is inserted immediately after → the document now says "I have no other property [value]" which makes no sense.
- When a contradiction is detected, you MUST restructure the affected sentence in prefix/suffix so the contradiction is removed:
  - Remove or neutralise the negating phrase by incorporating it into the prefix/suffix.
  - Example fix: change "...I have no other property. Except for [INSERTION_POINT]..." so that prefix="Other property owned: " and suffix=".", which removes the contradiction.
  - Or, if the selected text itself IS the negating phrase ([REPLACE_START]I have no other property[REPLACE_END]), replace the whole phrase with "{{other_property_description}}".
- Set "contradiction_resolved" to true and briefly explain in "reason" what you changed.

CRITICAL INSTRUCTIONS:
1. Read the FULL template snippet including surrounding context. Run the contradiction check above first.
2. Determine whether this is a REPLACE (markers exist) or INSERT operation.
3. For REPLACE — TWO CASES:
   a. SHORT selection (a single value, name, date, number — a few words): The selection is exactly what becomes the field. Replace just that value; prefix/suffix are usually empty because the sentence already flows.
      Example: "[REPLACE_START]John Smith[REPLACE_END] of 15 Queen St" → field_id="full_name", prefix="", suffix="" ✅
   b. LONG selection (a full sentence, clause, or list item — like "That there is no dispute in the ownership of the Land"):
      The user selected the whole sentence to give you CONTEXT, NOT to delete the entire sentence.
      You MUST keep the static legal language and only make the VARIABLE PART a placeholder.
      Put the static text before the variable part in prefix, and static text after it in suffix.
      Example: "[REPLACE_START]That there is no dispute in the ownership of the Land[REPLACE_END]" with request "land description" →
        field_id="land_description", prefix="That there is no dispute in the ownership of ", suffix="" ✅
      NEVER produce prefix="" suffix="" when the selection is a full sentence — that would delete all the legal text. ❌
4. For INSERT:
   - Generate 'prefix' and 'suffix' so the field fits grammatically AND logically.
   - The resulting sentence must make complete logical sense — no self-contradictions.
5. Respect the tone and style of the REFERENCE DOCUMENTS if provided.
6. Return ONLY valid JSON — no markdown, no extra text.

Response Format:
{{
  "field_id": "snake_case_id",
  "label": "Human Readable Label",
  "type": "text|textarea|date|email|phone|number|select",
  "help_text": "Brief instruction for the user",
  "placeholder": "Example value",
  "prefix": "Text before the field (resolves any contradiction if needed)",
  "suffix": "Text after the field",
  "contradiction_resolved": false,
  "reason": "Explanation of choice and any contradiction resolution"
}}

Rules:
- field_id: lowercase snake_case, unique.
- type: choose specific types (date, phone, number) over generic text where possible.
- prefix/suffix: MUST ensure grammatical AND logical correctness.
  - If the previous word has no trailing space, start prefix with a space.
  - If the next word assumes a separator, include it in suffix.
  - For REPLACE operations: prefer empty strings unless needed to fix grammar/logic.
"""

        try:
            client = get_openai_client()
            response = client.chat.completions.create(
                model='gpt-4o',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.2,
                max_tokens=600,
            )
            result_text = response.choices[0].message.content.strip()
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()

            field_config = _json.loads(result_text)

            # Ensure uniqueness
            base_id = field_config.get('field_id', 'new_field')
            field_id = base_id
            counter = 1
            while field_id in existing_ids:
                field_id = f'{base_id}_{counter}'
                counter += 1
            field_config['field_id'] = field_id

            return Response(field_config)

        except Exception as e:
            logger.warning(f'AI field generation failed: {e}')
            # Fallback: derive from description
            fallback_id = _re.sub(r'[^a-z0-9_]', '', description.lower().replace(' ', '_'))[:40]
            if not fallback_id or fallback_id[0].isdigit():
                fallback_id = 'field_' + fallback_id
            base_id = fallback_id
            counter = 1
            while fallback_id in existing_ids:
                fallback_id = f'{base_id}_{counter}'
                counter += 1
            return Response({
                'field_id': fallback_id,
                'label': description.title(),
                'type': 'text',
                'help_text': f'Please enter {description.lower()}',
                'placeholder': '',
                'reason': 'AI generation failed; derived from description',
            })


class AtomicFieldInsertionView(APIView):
    """
    Template-First Field Creation: Atomically insert placeholder into template + create question + update mapping.
    
    POST /admin/affidavit-types/<pk>/insert-field/
    Body:
      {
        "mode": "replace" | "insert",  // replace text or insert at position
        "field_id": "ownership_proof_number",
        "field_config": {
          "label": "Ownership Proof Number",
          "type": "text",
          "required": true,
          "help_text": "...",
          "placeholder": "..."
        },
        // For replace mode:
        "target_text": "John Smith",  // exact text to replace
        // For insert mode:
        "insert_position": 123,  // character offset in template_html
        "insert_context": "before" | "after" | "replace"  // how to insert relative to position
      }
    """
    permission_classes = [IsAdminUser]

    def post(self, request, pk):
        import re
        from .services.policy_generator_service import FIELD_DEFAULTS, FIELD_ALIASES

        try:
            affidavit_type = AffidavitType.objects.get(pk=pk)
        except AffidavitType.DoesNotExist:
            return Response({'error': 'Affidavit type not found'}, status=status.HTTP_404_NOT_FOUND)

        mode = request.data.get('mode', 'insert')
        field_id = request.data.get('field_id', '').strip().lower()
        field_config = request.data.get('field_config', {})

        # Validate field_id format
        if not field_id or not re.match(r'^[a-z][a-z0-9_]*$', field_id):
            return Response({
                'error': 'field_id must be lowercase snake_case (letters, numbers, underscores)'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Check if field_id already exists in intake_schema
        existing_ids = {q.get('id', '').lower() for q in (affidavit_type.intake_schema or [])}
        if field_id in existing_ids:
            return Response({
                'error': f'Field "{field_id}" already exists in intake schema'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Check if placeholder already exists in template
        template = affidavit_type.template_html or ''
        existing_placeholders = set(re.findall(r'\{\{(\w+)\}\}', template))
        if field_id in existing_placeholders:
            return Response({
                'error': f'Placeholder {{{{field_id}}}} already exists in template'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Build placeholder
        placeholder_text = f'{{{{{field_id}}}}}'

        # Modify template based on mode
        new_template = template
        if mode == 'replace':
            target_text = request.data.get('target_text', '')
            if not target_text:
                return Response({'error': 'target_text required for replace mode'}, status=status.HTTP_400_BAD_REQUEST)
            if target_text not in template:
                return Response({'error': f'target_text "{target_text}" not found in template'}, status=status.HTTP_400_BAD_REQUEST)
            # Replace first occurrence only
            new_template = template.replace(target_text, placeholder_text, 1)
        elif mode == 'insert':
            insert_position = request.data.get('insert_position')
            if insert_position is None:
                return Response({'error': 'insert_position required for insert mode'}, status=status.HTTP_400_BAD_REQUEST)
            if not (0 <= insert_position <= len(template)):
                return Response({'error': 'insert_position out of range'}, status=status.HTTP_400_BAD_REQUEST)
            new_template = template[:insert_position] + placeholder_text + template[insert_position:]
        else:
            return Response({'error': 'mode must be "replace" or "insert"'}, status=status.HTTP_400_BAD_REQUEST)

        # Build question from field_config + smart defaults
        canonical_id = FIELD_ALIASES.get(field_id, field_id)
        defaults = FIELD_DEFAULTS.get(canonical_id, {})

        new_question = {
            'id': field_id,
            'label': field_config.get('label') or defaults.get('label') or field_id.replace('_', ' ').title(),
            'type': field_config.get('type') or defaults.get('type', 'text'),
            'required': field_config.get('required', True),
            'placeholder': field_config.get('placeholder') or defaults.get('placeholder', ''),
            'help_text': field_config.get('help_text') or defaults.get('help_text', f"Enter your {field_config.get('label', field_id).lower()}."),
        }

        # Add validation if provided
        if field_config.get('validation'):
            new_question['validation'] = field_config['validation']
        elif defaults.get('validation'):
            new_question['validation'] = defaults['validation']

        # Append to intake_schema
        schema = list(affidavit_type.intake_schema or [])
        new_question['order'] = len(schema) + 1
        schema.append(new_question)

        # Update placeholder_mapping
        mapping = dict(affidavit_type.placeholder_mapping or {})
        mapping[field_id] = field_id

        # Atomic save
        affidavit_type.template_html = new_template
        affidavit_type.intake_schema = schema
        affidavit_type.placeholder_mapping = mapping
        affidavit_type.increment_policy_version()
        affidavit_type.save()

        return Response({
            'success': True,
            'field_id': field_id,
            'placeholder': placeholder_text,
            'question': new_question,
            'template_updated': True,
            'mapping_updated': True,
        })


class ConfidenceDashboardView(APIView):
    """
    Confidence dashboard showing volume vs review load (Story 4.2).
    Highlights candidates for instant mode promotion.
    Uses the new dashboard_service for comprehensive metrics.
    """
    
    permission_classes = [IsAdminUser]
    
    def get(self, request):
        # Get full dashboard data from service
        dashboard_data = get_dashboard_data()
        return Response(dashboard_data)


class FullConfidenceDashboardView(APIView):
    """
    Full confidence dashboard with learning loop metrics.
    Includes tier-based thresholds, scenario analysis, and promotion status.
    """
    
    permission_classes = [IsAdminUser]
    
    def get(self, request):
        dashboard_data = get_dashboard_data()
        return Response(dashboard_data)


class TypeTrendView(APIView):
    """
    Get daily trend data for a specific affidavit type.
    """
    
    permission_classes = [IsAdminUser]
    
    def get(self, request, pk):
        days = int(request.query_params.get('days', 30))
        trend_data = get_type_trend_data(pk, days)
        return Response(trend_data)


class WeeklyLearningReportView(APIView):
    """
    Generate weekly learning report for AI improvement.
    Analyzes overrides, edits, and new scenarios.
    """
    
    permission_classes = [IsAdminUser]
    
    def get(self, request):
        report = get_weekly_learning_report()
        return Response(report)


class CostDashboardView(APIView):
    """
    AI cost analytics dashboard.
    Shows spending by model, node type, and affidavit type.
    """
    
    permission_classes = [IsAdminUser]
    
    def get(self, request):
        from .services.dashboard_service import get_cost_dashboard
        days = int(request.query_params.get('days', 30))
        cost_data = get_cost_dashboard(days)
        return Response(cost_data)


class LearningSuggestionsView(APIView):
    """
    Automated learning suggestions for AI improvement.
    Provides prioritized recommendations based on recent patterns.
    """
    
    permission_classes = [IsAdminUser]
    
    def get(self, request):
        from .services.dashboard_service import generate_learning_suggestions
        days = int(request.query_params.get('days', 7))
        suggestions = generate_learning_suggestions(days)
        return Response(suggestions)


class LearningExportView(APIView):
    """
    Export learning data for AI improvement (Story 4.3).
    Groups reviewer edits by issue type.
    """
    
    permission_classes = [IsAdminUser]
    
    def get(self, request):
        # Get date range from query params
        days = int(request.query_params.get('days', 7))
        since = timezone.now() - timezone.timedelta(days=days)
        
        # Get edits grouped by issue type
        edits = ReviewerEdit.objects.filter(created_at__gte=since)
        
        # Group by issue type
        grouped = {}
        for edit in edits:
            issue_type = edit.issue_type
            if issue_type not in grouped:
                grouped[issue_type] = {
                    'issue_type': edit.get_issue_type_display(),
                    'count': 0,
                    'examples': []
                }
            grouped[issue_type]['count'] += 1
            if len(grouped[issue_type]['examples']) < 5:  # Limit examples
                grouped[issue_type]['examples'].append(
                    ReviewerEditSerializer(edit).data
                )
        
        return Response({
            'period_days': days,
            'total_edits': edits.count(),
            'by_issue_type': list(grouped.values())
        })


class FrictionDashboardView(APIView):
    """View friction reports for monitoring."""
    
    permission_classes = [IsAdminUser]
    
    def get(self, request):
        reports = FrictionReport.objects.select_related(
            'request', 
            'commissioner'
        ).order_by('-created_at')[:50]
        
        return Response(FrictionReportSerializer(reports, many=True).data)


class PromoteToInstantModeView(APIView):
    """Promote an affidavit type to instant mode."""
    
    permission_classes = [IsAdminUser]
    
    def post(self, request, pk):
        aff_type = get_object_or_404(AffidavitType, pk=pk)
        aff_type.is_instant_mode = True
        aff_type.save()
        
        return Response({
            'success': True,
            'message': f'{aff_type.name} is now in instant mode'
        })


# =============================================================================
# PDF Download View
# =============================================================================

class DownloadPDFView(APIView):
    """Download the PDF for an approved request."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request, pk):
        # Get request based on user role
        user = request.user
        
        if user.role in ['commissioner', 'reviewer', 'admin']:
            request_obj = get_object_or_404(Request, pk=pk)
            # For commissioners, check if assigned to them (unless admin/reviewer)
            if user.role == 'commissioner' and request_obj.commissioner:
                if request_obj.commissioner.id != user.id:
                    return Response(
                        {'error': 'This request is assigned to another commissioner.'},
                        status=status.HTTP_403_FORBIDDEN
                    )
        else:
            request_obj = get_object_or_404(Request, pk=pk, user=user)
        
        # Always regenerate PDF to ensure latest content and styling is used
        if request_obj.status not in [Request.Status.APPROVED, Request.Status.COMPLETED, Request.Status.NEEDS_REVIEW, Request.Status.DRAFT_READY]:
            if not request_obj.pdf_file:
                return Response(
                    {'error': 'PDF not available for this request'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        else:
            # Need draft_text or final_text to generate PDF
            if not request_obj.draft_text and not request_obj.final_text:
                return Response(
                    {'error': 'No draft available to generate PDF'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            result = generate_affidavit_pdf(request_obj)
            if not result['success']:
                return Response(
                    {'error': result['error']},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        # Return PDF file
        response = HttpResponse(
            request_obj.pdf_file.read(),
            content_type='application/pdf'
        )
        response['Content-Disposition'] = (
            f'attachment; filename="affidavit_{request_obj.request_code}.pdf"'
        )
        return response


class DownloadWordView(APIView):
    """Download the affidavit as a Word document for editing."""
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request, pk):
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        import io
        import html
        import re
        from bs4 import BeautifulSoup
        
        # Get request based on user role
        user = request.user
        
        if user.role in ['commissioner', 'reviewer', 'admin']:
            request_obj = get_object_or_404(Request, pk=pk)
            # For commissioners, check if assigned to them (unless admin/reviewer)
            if user.role == 'commissioner' and request_obj.commissioner:
                if request_obj.commissioner.id != user.id:
                    return Response(
                        {'error': 'This request is assigned to another commissioner.'},
                        status=status.HTTP_403_FORBIDDEN
                    )
        else:
            request_obj = get_object_or_404(Request, pk=pk, user=user)
        
        # Check if draft exists
        if not request_obj.draft_text:
            return Response(
                {'error': 'No draft available for this request'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create Word document
        doc = Document()
        
        # Set default font for document
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Times New Roman'
        font.size = Pt(12)
        
        # Add reference code at top right
        ref_para = doc.add_paragraph()
        ref_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        ref_run = ref_para.add_run(f'Reference: {request_obj.request_code}')
        ref_run.font.size = Pt(10)
        ref_run.font.italic = True
        
        # Parse HTML content
        draft_text = request_obj.draft_text
        soup = BeautifulSoup(draft_text, 'html.parser')
        
        def add_formatted_paragraph(doc, element, is_list_item=False, list_number=None):
            """Add a paragraph with proper formatting from HTML element."""
            para = doc.add_paragraph()
            
            if is_list_item and list_number:
                # Add numbered list item
                run = para.add_run(f'{list_number}.\t')
                run.font.name = 'Times New Roman'
                run.font.size = Pt(12)
            
            # Process child elements for formatting
            for child in element.children if hasattr(element, 'children') else [element]:
                if isinstance(child, str):
                    text = child.strip()
                    if text:
                        run = para.add_run(text)
                        run.font.name = 'Times New Roman'
                        run.font.size = Pt(12)
                elif child.name == 'strong' or child.name == 'b':
                    run = para.add_run(child.get_text())
                    run.bold = True
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(12)
                elif child.name == 'em' or child.name == 'i':
                    run = para.add_run(child.get_text())
                    run.italic = True
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(12)
                elif child.name == 'u':
                    run = para.add_run(child.get_text())
                    run.underline = True
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(12)
                elif hasattr(child, 'get_text'):
                    # Recursively get text from nested elements
                    text = child.get_text()
                    if text.strip():
                        run = para.add_run(text)
                        run.font.name = 'Times New Roman'
                        run.font.size = Pt(12)
                        # Check if parent or self has bold/italic
                        if child.name in ['strong', 'b'] or child.find_parent(['strong', 'b']):
                            run.bold = True
                        if child.name in ['em', 'i'] or child.find_parent(['em', 'i']):
                            run.italic = True
            
            para.paragraph_format.space_after = Pt(6)
            return para
        
        # Process each element
        list_counter = 0
        for element in soup.children:
            if isinstance(element, str):
                text = element.strip()
                if text:
                    para = doc.add_paragraph(text)
                    para.runs[0].font.name = 'Times New Roman'
                    para.runs[0].font.size = Pt(12)
                continue
                
            if not hasattr(element, 'name'):
                continue
                
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Heading
                heading = doc.add_paragraph()
                heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
                run = heading.add_run(element.get_text().strip())
                run.bold = True
                run.font.name = 'Times New Roman'
                run.font.size = Pt(14 if element.name in ['h1', 'h2'] else 12)
                heading.paragraph_format.space_before = Pt(12)
                heading.paragraph_format.space_after = Pt(6)
                
            elif element.name == 'p':
                add_formatted_paragraph(doc, element)
                
            elif element.name in ['ul', 'ol']:
                # List
                list_counter = 0
                for li in element.find_all('li', recursive=False):
                    list_counter += 1
                    if element.name == 'ol':
                        add_formatted_paragraph(doc, li, is_list_item=True, list_number=list_counter)
                    else:
                        para = doc.add_paragraph(style='List Bullet')
                        for child in li.children:
                            if isinstance(child, str):
                                text = child.strip()
                                if text:
                                    run = para.add_run(text)
                                    run.font.name = 'Times New Roman'
                                    run.font.size = Pt(12)
                            elif child.name in ['strong', 'b']:
                                run = para.add_run(child.get_text())
                                run.bold = True
                                run.font.name = 'Times New Roman'
                                run.font.size = Pt(12)
                            elif hasattr(child, 'get_text'):
                                run = para.add_run(child.get_text())
                                run.font.name = 'Times New Roman'
                                run.font.size = Pt(12)
                
            elif element.name == 'br':
                doc.add_paragraph()
                
            elif element.name == 'hr':
                doc.add_paragraph('_' * 50)
                
            else:
                # Default: just add text
                text = element.get_text().strip()
                if text:
                    para = doc.add_paragraph(text)
                    for run in para.runs:
                        run.font.name = 'Times New Roman'
                        run.font.size = Pt(12)
        
        # Save to bytes
        file_buffer = io.BytesIO()
        doc.save(file_buffer)
        file_buffer.seek(0)
        
        # Return Word document
        response = HttpResponse(
            file_buffer.read(),
            content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
        response['Content-Disposition'] = (
            f'attachment; filename="affidavit_{request_obj.request_code}.docx"'
        )
        return response


# =============================================================================
# Public Commissioner Listing (for Landing Page)
# =============================================================================

class PublicCommissionerListView(generics.ListAPIView):
    """
    Public endpoint to list featured commissioners for landing page.
    Shows commissioners with is_featured=True.
    """
    
    serializer_class = CommissionerPublicSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        return User.objects.filter(
            role=User.Role.COMMISSIONER,
            is_active=True,
            is_featured=True
        ).order_by('first_name', 'last_name')


# =============================================================================
# Admin Staff Management Views
# =============================================================================

class AdminCommissionerListView(generics.ListCreateAPIView):
    """
    Admin endpoint to list and create commissioners.
    """
    
    permission_classes = [IsAdminUser]
    
    def get_serializer_class(self):
        if self.request.method == 'POST':
            return CreateStaffUserSerializer
        return CommissionerSerializer
    
    def get_queryset(self):
        return User.objects.filter(
            role=User.Role.COMMISSIONER
        ).order_by('-date_joined')
    
    def perform_create(self, serializer):
        serializer.save(role=User.Role.COMMISSIONER)


class AdminCommissionerDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    Admin endpoint to view, update, or delete a commissioner.
    """
    
    permission_classes = [IsAdminUser]
    
    def get_serializer_class(self):
        if self.request.method in ['PUT', 'PATCH']:
            return UpdateStaffUserSerializer
        return CommissionerSerializer
    
    def get_queryset(self):
        return User.objects.filter(role=User.Role.COMMISSIONER)


class AdminReviewerListView(generics.ListCreateAPIView):
    """
    Admin endpoint to list and create reviewers.
    """
    
    permission_classes = [IsAdminUser]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['username', 'email', 'first_name', 'last_name']
    ordering_fields = ['date_joined', 'username', 'last_name']
    ordering = ['-date_joined']
    
    def get_serializer_class(self):
        if self.request.method == 'POST':
            return CreateStaffUserSerializer
        return ReviewerSerializer
    
    def get_queryset(self):
        return User.objects.filter(
            role=User.Role.REVIEWER
        ).order_by('-date_joined')
    
    def perform_create(self, serializer):
        serializer.save(role=User.Role.REVIEWER)


class AdminReviewerDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    Admin endpoint to view, update, or delete a reviewer.
    """
    
    permission_classes = [IsAdminUser]
    
    def get_serializer_class(self):
        if self.request.method in ['PUT', 'PATCH']:
            return UpdateStaffUserSerializer
        return ReviewerSerializer
    
    def get_queryset(self):
        return User.objects.filter(role=User.Role.REVIEWER)


# =============================================================================
# Admin Commissioner Payment Views
# =============================================================================

class AdminCommissionerPaymentSummaryView(generics.RetrieveAPIView):
    """
    Get commissioner details with payment summary (amount to pay, banking details).
    """
    
    permission_classes = [IsAdminUser]
    serializer_class = CommissionerPaymentSummarySerializer
    
    def get_queryset(self):
        return User.objects.filter(role=User.Role.COMMISSIONER)


class AdminCommissionerPaymentHistoryView(generics.ListAPIView):
    """
    List payment history for a specific commissioner.
    """
    
    permission_classes = [IsAdminUser]
    serializer_class = PaymentLogSerializer
    
    def get_queryset(self):
        commissioner_id = self.kwargs.get('pk')
        return PaymentLog.objects.filter(
            commissioner_id=commissioner_id
        ).order_by('-paid_at')


class AdminMarkCommissionerPaidView(APIView):
    """
    Mark all unpaid stamps for a commissioner as paid.
    Creates a PaymentLog entry and updates all unpaid stamps.
    """
    
    permission_classes = [IsAdminUser]
    
    @transaction.atomic
    def post(self, request, pk):
        # Get commissioner
        commissioner = get_object_or_404(User, pk=pk, role=User.Role.COMMISSIONER)
        
        # Validate request data
        serializer = MarkAsPaidSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Get all unpaid stamps for this commissioner
        unpaid_stamps = Stamp.objects.filter(
            commissioner=commissioner,
            paid=False
        )
        
        if not unpaid_stamps.exists():
            return Response(
                {'detail': 'No unpaid stamps found for this commissioner.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Calculate total amount
        from django.db.models import Sum
        total_amount = unpaid_stamps.aggregate(total=Sum('payout_amount'))['total'] or 0
        stamps_count = unpaid_stamps.count()
        
        # Create payment log
        payment_log = PaymentLog.objects.create(
            commissioner=commissioner,
            amount_paid=total_amount,
            stamps_count=stamps_count,
            paid_by=request.user,
            payment_reference=serializer.validated_data.get('payment_reference', ''),
            payment_method=serializer.validated_data.get('payment_method', ''),
            notes=serializer.validated_data.get('notes', '')
        )
        
        # Mark all stamps as paid
        now = timezone.now()
        unpaid_stamps.update(paid=True, paid_at=now)
        
        return Response({
            'detail': f'Successfully marked {stamps_count} stamps as paid.',
            'payment_log': PaymentLogSerializer(payment_log).data,
            'total_amount': str(total_amount),
            'stamps_count': stamps_count
        }, status=status.HTTP_200_OK)


class AdminAllPaymentLogsView(generics.ListAPIView):
    """
    List all payment logs across all commissioners.
    """
    
    permission_classes = [IsAdminUser]
    serializer_class = PaymentLogSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['commissioner__username', 'commissioner__email', 'payment_reference']
    ordering_fields = ['paid_at', 'amount_paid']
    ordering = ['-paid_at']
    
    def get_queryset(self):
        return PaymentLog.objects.all()


class AdminGenerateSlotsView(APIView):
    """
    Superuser-only endpoint to manually generate/refresh availability slots
    for a specific commissioner. Use this as a fallback when the Celery Beat
    cron job did not run.
    """
    permission_classes = [IsAuthenticated, IsSuperUser]

    def post(self, request, pk):
        from .services.slot_service import generate_slots_for_commissioner

        commissioner = get_object_or_404(User, pk=pk, role=User.Role.COMMISSIONER)

        # Allow caller to specify how many days ahead to generate (default 14, max 60)
        try:
            days = int(request.data.get('days', 14))
            days = max(1, min(days, 60))
        except (TypeError, ValueError):
            days = 14

        slots_created = generate_slots_for_commissioner(commissioner, days=days, cleanup=True)

        return Response({
            'success': True,
            'message': (
                f'Generated {slots_created} slot(s) for '
                f'{commissioner.get_full_name() or commissioner.username} '
                f'({days}-day window).'
            ),
            'slots_created': slots_created,
            'commissioner_id': pk,
        }, status=status.HTTP_200_OK)


# =============================================================================
# Commissioner Slot Views
# =============================================================================

class CommissionerSlotsView(APIView):
    """
    List available time slots for a specific commissioner on a given date.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        from django.utils.dateparse import parse_date
        from django.utils import timezone
        import datetime
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        
        commissioner = get_object_or_404(User, pk=pk, role=User.Role.COMMISSIONER)
        
        date_str = request.query_params.get('date')
        if not date_str:
            return Response(
                {'error': 'Date parameter is required (YYYY-MM-DD)'},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        try:
            target_date = parse_date(date_str)
            if not target_date:
                raise ValueError("Invalid date format")
        except ValueError:
            return Response(
                {'error': 'Invalid date format. Use YYYY-MM-DD'},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Filter slots for the specific date in COMMISSIONER'S TIMEZONE
        # Standard Django __date filter uses UTC (or server time), which causes
        # slots from the previous night (UTC) to appear in today's list for UTC-4 (Trinidad).
        
        tz_name = commissioner.availability.get('timezone', 'America/Port_of_Spain')
        try:
            local_tz = ZoneInfo(tz_name)
        except Exception:
            local_tz = datetime.timezone.utc

        # Create range covering the full day in local time
        start_local = datetime.datetime.combine(target_date, datetime.time.min).replace(tzinfo=local_tz)
        # End of day should be up to 23:59:59
        end_local = datetime.datetime.combine(target_date, datetime.time.max).replace(tzinfo=local_tz)
        
        # Convert to UTC for DB querying
        start_utc = start_local.astimezone(datetime.timezone.utc)
        end_utc = end_local.astimezone(datetime.timezone.utc)

        slots = CommissionerSlot.objects.filter(
            commissioner=commissioner,
            start_time__gte=start_utc,
            start_time__lte=end_utc
        ).order_by('start_time')
        
        serializer = CommissionerSlotSerializer(slots, many=True)
        return Response(serializer.data)


class CommissionerBookedSlotsView(generics.ListAPIView):
    """
    List all booked upcoming slots for the current commissioner.
    Used for the "My Schedule" page.
    """
    permission_classes = [IsCommissioner]
    serializer_class = CommissionerSlotSerializer
    
    def get_queryset(self):
        # Show all future booked slots, ordered by time
        return CommissionerSlot.objects.filter(
            commissioner=self.request.user,
            is_booked=True,
            start_time__gte=timezone.now() - timezone.timedelta(hours=24) # Include recent past (24h) for context
        ).select_related('request', 'request__user', 'request__affidavit_type').order_by('start_time')


class BookSlotView(APIView):
    """
    Book a specific time slot for an affidavit request.
    Atomic transaction to prevent double booking.
    """
    permission_classes = [IsAuthenticated]

    @transaction.atomic
    def post(self, request, pk):
        slot = get_object_or_404(CommissionerSlot, pk=pk)
        
        request_id = request.data.get('request_id')
        if not request_id:
            return Response(
                {'error': 'request_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        request_obj = get_object_or_404(Request, pk=request_id, user=request.user)
        
        # Check if already booked
        if slot.is_booked:
            return Response(
                {'error': 'This slot is already booked.'},
                status=status.HTTP_409_CONFLICT
            )
            
        # Check if request already has a slot, release it if so
        if hasattr(request_obj, 'appointment_slot'):
            old_slot = request_obj.appointment_slot
            old_slot.is_booked = False
            old_slot.request = None
            old_slot.save()
            
        # Book the new slot
        slot.is_booked = True
        slot.request = request_obj
        slot.save()
        
        # Assign commissioner to request
        request_obj.commissioner = slot.commissioner
        
        # Update status to NEEDS_REVIEW if it was DRAFT_READY
        if request_obj.status == Request.Status.DRAFT_READY:
            request_obj.status = Request.Status.NEEDS_REVIEW
            
        request_obj.save()
        
        # Log event
        RequestEvent.objects.create(
            request=request_obj,
            action=RequestEvent.Action.COMMISSIONER_CHANGED,
            actor=request.user,
            actor_role=request.user.role,
            details={
                'action': 'booked_slot',
                'slot_id': slot.id,
                'slot_time': slot.start_time.isoformat(),
                'commissioner': slot.commissioner.username
            }
        )
        
        # Send notifications to user and commissioner
        from .services.notification_service import send_appointment_booked_notifications
        try:
            send_appointment_booked_notifications(request_obj, slot)
        except Exception as e:
            logger.warning(f"Failed to send appointment booked notifications: {e}")
        
        return Response({
            'success': True,
            'message': 'Appointment booked successfully',
            'slot': CommissionerSlotSerializer(slot).data
        })


# =============================================================================
# Commissioner Appointment Decision Views
# =============================================================================

class CommissionerAcceptSlotView(APIView):
    """
    Commissioner accepts a booked appointment.
    Sends confirmation notification to user.
    """
    permission_classes = [IsCommissioner]

    @transaction.atomic
    def post(self, request, pk):
        slot = get_object_or_404(
            CommissionerSlot, 
            pk=pk, 
            commissioner=request.user,
            is_booked=True
        )
        
        if slot.appointment_status == CommissionerSlot.AppointmentStatus.ACCEPTED:
            return Response(
                {'error': 'Appointment already accepted'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if slot.appointment_status in [
            CommissionerSlot.AppointmentStatus.REJECTED,
            CommissionerSlot.AppointmentStatus.CANCELLED_BY_COMMISSIONER,
            CommissionerSlot.AppointmentStatus.CANCELLED_BY_USER
        ]:
            return Response(
                {'error': 'Cannot accept a cancelled or rejected appointment'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Accept the appointment
        slot.accept()
        
        # Log event
        if slot.request:
            RequestEvent.objects.create(
                request=slot.request,
                action=RequestEvent.Action.COMMISSIONER_CHANGED,
                actor=request.user,
                actor_role=request.user.role,
                details={
                    'action': 'appointment_accepted',
                    'slot_id': slot.id,
                    'slot_time': slot.start_time.isoformat()
                }
            )
            
            # Send notification to user
            from .services.notification_service import send_appointment_accepted_notification
            try:
                send_appointment_accepted_notification(slot.request, slot)
            except Exception as e:
                logger.warning(f"Failed to send appointment accepted notification: {e}")
        
        return Response({
            'success': True,
            'message': 'Appointment accepted',
            'slot': CommissionerSlotSerializer(slot).data
        })


class CommissionerRejectSlotView(APIView):
    """
    Commissioner rejects a booked appointment.
    Releases the slot and notifies user to rebook.
    """
    permission_classes = [IsCommissioner]

    @transaction.atomic
    def post(self, request, pk):
        slot = get_object_or_404(
            CommissionerSlot, 
            pk=pk, 
            commissioner=request.user,
            is_booked=True
        )
        
        if slot.appointment_status in [
            CommissionerSlot.AppointmentStatus.REJECTED,
            CommissionerSlot.AppointmentStatus.CANCELLED_BY_COMMISSIONER,
            CommissionerSlot.AppointmentStatus.CANCELLED_BY_USER
        ]:
            return Response(
                {'error': 'Appointment already cancelled or rejected'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        reason = request.data.get('reason', '')
        request_obj = slot.request
        
        # Reject the appointment (releases slot and clears request link)
        slot.reject(reason)
        
        # Clear commissioner from request and revert status
        if request_obj:
            request_obj.commissioner = None
            if request_obj.status == Request.Status.NEEDS_REVIEW:
                request_obj.status = Request.Status.DRAFT_READY
            request_obj.save()
            
            # Log event
            RequestEvent.objects.create(
                request=request_obj,
                action=RequestEvent.Action.COMMISSIONER_CHANGED,
                actor=request.user,
                actor_role=request.user.role,
                details={
                    'action': 'appointment_rejected',
                    'slot_id': slot.id,
                    'reason': reason
                }
            )
            
            # Send notification to user
            from .services.notification_service import send_appointment_rejected_notification
            try:
                send_appointment_rejected_notification(request_obj)
            except Exception as e:
                logger.warning(f"Failed to send appointment rejected notification: {e}")
        
        return Response({
            'success': True,
            'message': 'Appointment rejected. User has been notified to select a new slot.'
        })


class CommissionerCancelSlotView(APIView):
    """
    Commissioner cancels a previously accepted appointment.
    Releases the slot and notifies user to rebook.
    """
    permission_classes = [IsCommissioner]

    @transaction.atomic
    def post(self, request, pk):
        slot = get_object_or_404(
            CommissionerSlot, 
            pk=pk, 
            commissioner=request.user,
            is_booked=True
        )
        
        if slot.appointment_status in [
            CommissionerSlot.AppointmentStatus.REJECTED,
            CommissionerSlot.AppointmentStatus.CANCELLED_BY_COMMISSIONER,
            CommissionerSlot.AppointmentStatus.CANCELLED_BY_USER
        ]:
            return Response(
                {'error': 'Appointment already cancelled or rejected'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        reason = request.data.get('reason', '')
        request_obj = slot.request
        
        # Cancel the appointment
        slot.cancel_by_commissioner(reason)
        
        # Clear commissioner from request and revert status
        if request_obj:
            request_obj.commissioner = None
            if request_obj.status == Request.Status.NEEDS_REVIEW:
                request_obj.status = Request.Status.DRAFT_READY
            request_obj.save()
            
            # Log event
            RequestEvent.objects.create(
                request=request_obj,
                action=RequestEvent.Action.COMMISSIONER_CHANGED,
                actor=request.user,
                actor_role=request.user.role,
                details={
                    'action': 'appointment_cancelled_by_commissioner',
                    'slot_id': slot.id,
                    'reason': reason
                }
            )
            
            # Send notification to user
            from .services.notification_service import send_appointment_cancelled_by_commissioner_notification
            try:
                send_appointment_cancelled_by_commissioner_notification(request_obj)
            except Exception as e:
                logger.warning(f"Failed to send appointment cancelled notification: {e}")
        
        return Response({
            'success': True,
            'message': 'Appointment cancelled. User has been notified to reschedule.'
        })


# =============================================================================
# Admin Affidavit Type CRUD Views
# =============================================================================

class AdminAffidavitTypeListView(generics.ListCreateAPIView):
    """
    Admin endpoint to list and create affidavit types.
    Includes full details with intake_schema.
    """
    
    permission_classes = [IsAdminUser]
    queryset = AffidavitType.objects.all().order_by('name')
    
    def get_permissions(self):
        if self.request.method == 'POST':
            return [IsSuperUser()]
        return [IsAdminUser()]
    
    def get_serializer_class(self):
        from .serializers import AffidavitTypeAdminSerializer
        return AffidavitTypeAdminSerializer


class AdminAffidavitTypeDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    Admin endpoint to view, update, or delete an affidavit type.
    """
    
    permission_classes = [IsAdminUser]
    queryset = AffidavitType.objects.all()
    
    def get_permissions(self):
        if self.request.method in ['PUT', 'PATCH', 'DELETE']:
            return [IsSuperUser()]
        return [IsAdminUser()]
    
    def get_serializer_class(self):
        from .serializers import AffidavitTypeAdminSerializer
        return AffidavitTypeAdminSerializer
    
    def destroy(self, request, *args, **kwargs):
        """
        Delete or deactivate the type.
        If there are requests using this type, deactivate instead of deleting.
        """
        instance = self.get_object()
        
        # Check if there are requests using this type
        request_count = Request.objects.filter(affidavit_type=instance).count()
        
        if request_count > 0:
            # Deactivate instead of delete
            instance.is_active = False
            instance.save()
            return Response({
                'message': f'Type deactivated (has {request_count} requests)',
                'deactivated': True
            })
        
        # Safe to delete
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)


class AdminAffidavitTypeDuplicateView(APIView):
    """
    Duplicate an affidavit type with all its settings.
    """
    
    permission_classes = [IsSuperUser]
    
    def post(self, request, pk):
        from .serializers import AffidavitTypeAdminSerializer
        
        original = get_object_or_404(AffidavitType, pk=pk)
        
        # Create a copy
        new_type = AffidavitType.objects.create(
            name=f"{original.name} (Copy)",
            description=original.description,
            tier=original.tier,
            default_mode=original.default_mode,
            confidence_status='learning',  # Reset confidence
            enabled_on_homepage=False,  # Don't enable on homepage by default
            policy_json=original.policy_json,
            policy_version=1,  # Reset version
            prompt_pack_version=original.prompt_pack_version,
            template_version=original.template_version,
            intake_schema=original.intake_schema,
            scenario_library=original.scenario_library,
            is_active=False,  # Start as inactive
        )
        
        serializer = AffidavitTypeAdminSerializer(new_type)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class RefineTemplateView(APIView):
    """
    AI-powered template refinement endpoint.
    Admin describes what to make dynamic; AI adds {{placeholder}} tokens,
    optionally learning from real saved affidavit drafts in DB.

    POST /admin/types/<pk>/refine-template/
    Body: { "instruction": "...", "current_template": "..." }
    """

    permission_classes = [IsAdminUser]

    def post(self, request, pk):
        affidavit_type = get_object_or_404(AffidavitType, pk=pk)

        instruction = request.data.get('instruction', '').strip()
        current_template = (
            request.data.get('current_template', '').strip()
            or affidavit_type.template_html
            or ''
        )

        if not instruction:
            return Response(
                {'error': 'Instruction is required'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not current_template:
            return Response(
                {'error': 'No template to refine. Please create a template first.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        result = refine_template_section(
            current_template=current_template,
            instruction=instruction,
            affidavit_type_id=pk,
        )

        http_status = status.HTTP_200_OK if result.get('success') else status.HTTP_500_INTERNAL_SERVER_ERROR
        return Response(result, status=http_status)


class RefineInstructionView(APIView):
    """
    AI-powered prompt refinement for template refinement instructions.
    Takes a rough user instruction and returns a clear, specific one.

    POST /admin/types/<pk>/refine-instruction/
    Body: { "raw_instruction": "...", "current_template": "..." }
    """

    permission_classes = [IsAdminUser]

    def post(self, request, pk):
        affidavit_type = get_object_or_404(AffidavitType, pk=pk)

        raw_instruction = request.data.get('raw_instruction', '').strip()
        current_template = (
            request.data.get('current_template', '').strip()
            or affidavit_type.template_html
            or ''
        )

        if not raw_instruction:
            return Response(
                {'error': 'Instruction is required'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        result = refine_user_instruction(
            raw_instruction=raw_instruction,
            current_template=current_template,
        )

        http_status = status.HTTP_200_OK if result.get('success') else status.HTTP_500_INTERNAL_SERVER_ERROR
        return Response(result, status=http_status)


class ValidateAffidavitConfigView(APIView):
    """
    Validate template ↔ intake_schema mapping completeness for an affidavit type.
    Returns errors (unmapped placeholders) and warnings (orphaned questions).

    GET /admin/affidavit-types/<pk>/validate-config/
    """

    permission_classes = [IsAdminUser]

    def get(self, request, pk):
        from .services.policy_generator_service import validate_template_mapping

        affidavit_type = get_object_or_404(AffidavitType, pk=pk)

        report = validate_template_mapping(
            template_html=affidavit_type.template_html or '',
            intake_schema=affidavit_type.intake_schema or [],
            placeholder_mapping=affidavit_type.placeholder_mapping or {},
        )
        report['affidavit_type_id'] = affidavit_type.id
        report['affidavit_type_name'] = affidavit_type.name

        return Response(report)


# =============================================================================
# Admin Document Upload & Policy Generation Views
# =============================================================================

class AdminDocumentUploadView(APIView):
    """
    Upload Word/PDF documents for template extraction.
    Converts documents to HTML and stores them for an affidavit type.
    """
    
    permission_classes = [IsAdminUser]
    
    def post(self, request, pk):
        from .serializers import DocumentUploadSerializer, DocumentParseResultSerializer
        from .services.document_parser_service import parse_document
        
        affidavit_type = get_object_or_404(AffidavitType, pk=pk)
        
        # Handle single or multiple files
        files = request.FILES.getlist('files', [])
        if not files:
            # Try single file field
            single_file = request.FILES.get('file')
            if single_file:
                files = [single_file]
        
        if not files:
            return Response(
                {'error': 'No files provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        results = []
        successful_docs = []
        
        for file in files:
            # Validate file
            serializer = DocumentUploadSerializer(data={'file': file})
            if not serializer.is_valid():
                results.append({
                    'success': False,
                    'filename': file.name,
                    'error': serializer.errors
                })
                continue
            
            # Parse the document
            file_bytes = file.read()
            parse_result = parse_document(file_bytes, file.name)
            results.append(parse_result)
            
            if parse_result['success']:
                successful_docs.append({
                    'filename': parse_result['filename'],
                    'html_content': parse_result['html_content'],
                    'file_type': parse_result['file_type'],
                    'uploaded_at': parse_result['parsed_at']
                })
        
        # Store successful documents in the affidavit type
        if successful_docs:
            current_docs = affidavit_type.template_documents or []
            current_docs.extend(successful_docs)
            affidavit_type.template_documents = current_docs
            affidavit_type.save()
        
        return Response({
            'message': f'Processed {len(files)} file(s), {len(successful_docs)} successful',
            'results': results,
            'total_documents': len(affidavit_type.template_documents or [])
        })


class AdminDocumentListView(APIView):
    """
    List and manage uploaded template documents for an affidavit type.
    """
    
    permission_classes = [IsAdminUser]
    
    def get(self, request, pk):
        """List all uploaded documents for this type."""
        affidavit_type = get_object_or_404(AffidavitType, pk=pk)
        
        documents = affidavit_type.template_documents or []
        
        # Return without full HTML content for list view
        docs_summary = []
        for doc in documents:
            docs_summary.append({
                'filename': doc.get('filename'),
                'file_type': doc.get('file_type'),
                'uploaded_at': doc.get('uploaded_at'),
                'content_length': len(doc.get('html_content', ''))
            })
        
        return Response({
            'affidavit_type_id': pk,
            'affidavit_type_name': affidavit_type.name,
            'documents': docs_summary,
            'total': len(docs_summary)
        })
    
    def delete(self, request, pk):
        """Delete a specific document by filename."""
        affidavit_type = get_object_or_404(AffidavitType, pk=pk)
        filename = request.data.get('filename')
        
        if not filename:
            return Response(
                {'error': 'filename is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        documents = affidavit_type.template_documents or []
        original_count = len(documents)
        
        # Filter out the document to delete
        documents = [d for d in documents if d.get('filename') != filename]
        
        if len(documents) == original_count:
            return Response(
                {'error': f'Document "{filename}" not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        affidavit_type.template_documents = documents
        affidavit_type.save()
        
        return Response({
            'message': f'Deleted "{filename}"',
            'remaining_documents': len(documents)
        })


class AdminGeneratePolicyView(APIView):
    """
    Use AI to generate policy from uploaded template documents.
    Now runs asynchronously to prevent request timeouts in production.
    """
    
    permission_classes = [IsAdminUser]
    
    def post(self, request, pk):
        from .tasks import generate_policy_async
        
        affidavit_type = get_object_or_404(AffidavitType, pk=pk)
        
        # Get uploaded documents
        documents = affidavit_type.template_documents or []
        if not documents:
            return Response(
                {'error': 'No template documents uploaded. Please upload 2-3 example affidavits first.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Extract HTML content from documents
        html_examples = [doc.get('html_content', '') for doc in documents if doc.get('html_content')]
        
        if not html_examples:
            return Response(
                {'error': 'No valid HTML content found in uploaded documents'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get additional context from request
        additional_context = request.data.get('additional_context', '')
        
        # Get existing intake_schema to merge intelligently
        existing_questions = affidavit_type.intake_schema or []
        
        # Trigger async task
        task = generate_policy_async.delay(
            affidavit_type_id=affidavit_type.id,
            html_examples=html_examples,
            additional_context=additional_context,
            existing_questions=existing_questions
        )
        
        return Response({
            'task_id': task.id,
            'status': 'processing',
            'message': 'Policy generation started. Use task_id to poll for results.'
        }, status=status.HTTP_202_ACCEPTED)


class AdminPolicyTaskStatusView(APIView):
    """
    Check status of policy generation task.
    """
    
    permission_classes = [IsAdminUser]
    
    def get(self, request, task_id):
        from celery.result import AsyncResult
        from rest_framework import serializers
        from .services.policy_generator_service import (
            convert_detected_fields_to_intake_schema,
            build_policy_json_from_generation,
            auto_generate_placeholder_mapping,
            build_scenarios_from_identified,
            validate_template_mapping,
        )
        
        task = AsyncResult(task_id)
        
        if task.state == 'PENDING':
            return Response({
                'status': 'pending',
                'message': 'Task is waiting to be processed'
            })
        elif task.state == 'STARTED':
            return Response({
                'status': 'processing',
                'message': 'Policy generation in progress'
            })
        elif task.state == 'SUCCESS':
            result = task.result
            
            if not result.get('success'):
                return Response({
                    'status': 'failed',
                    'error': result.get('error', 'Unknown error')
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Get auto_save flag from query params
            auto_save = request.query_params.get('auto_save', 'false').lower() == 'true'
            affidavit_type_id = request.query_params.get('affidavit_type_id')
            
            if auto_save and affidavit_type_id:
                try:
                    affidavit_type = AffidavitType.objects.get(id=affidavit_type_id)
                    existing_questions = affidavit_type.intake_schema or []
                    
                    # Update the affidavit type with generated content
                    affidavit_type.template_html = result['template_html']
                    affidavit_type.disallowed_phrases = result['disallowed_phrases']
                    
                    # Merge detected fields with existing intake_schema
                    if result['detected_fields']:
                        new_fields = convert_detected_fields_to_intake_schema(
                            result['detected_fields']
                        )
                        existing_ids = {q.get('id', '').lower() for q in existing_questions}
                        existing_labels = {q.get('label', '').lower() for q in existing_questions}
                        
                        fields_to_add = []
                        for field in new_fields:
                            field_id = field.get('id', '').lower()
                            field_label = field.get('label', '').lower()
                            if field_id not in existing_ids and field_label not in existing_labels:
                                field['order'] = len(existing_questions) + len(fields_to_add) + 1
                                fields_to_add.append(field)
                        
                        if fields_to_add:
                            affidavit_type.intake_schema = existing_questions + fields_to_add
                            result['new_fields_added'] = len(fields_to_add)
                        else:
                            result['new_fields_added'] = 0
                    
                    # Build and save policy_json
                    policy_json = build_policy_json_from_generation(result)
                    if policy_json:
                        affidavit_type.policy_json = {
                            **affidavit_type.policy_json,
                            **policy_json
                        }

                    # Store scenario_branches in policy_json for drafter use
                    scenario_branches = result.get('scenario_branches', {})
                    if scenario_branches:
                        affidavit_type.policy_json = {
                            **(affidavit_type.policy_json or {}),
                            'scenario_branches': scenario_branches,
                        }
                        result['scenario_branches_stored'] = len(scenario_branches)
                    
                    # Auto-generate placeholder_mapping from template + final schema
                    final_schema = affidavit_type.intake_schema or []
                    generated_mapping = auto_generate_placeholder_mapping(
                        affidavit_type.template_html or '',
                        final_schema,
                    )
                    if generated_mapping:
                        existing_mapping = affidavit_type.placeholder_mapping or {}
                        # Merge: keep existing manual overrides, fill in new auto-mappings
                        for ph, qid in generated_mapping.items():
                            if ph not in existing_mapping:
                                existing_mapping[ph] = qid
                        affidavit_type.placeholder_mapping = existing_mapping
                        result['auto_mapped_placeholders'] = len(generated_mapping)

                    # Merge AI-identified scenarios into scenario_library
                    identified = result.get('identified_scenarios', [])
                    if identified:
                        affidavit_type.scenario_library = build_scenarios_from_identified(
                            identified,
                            affidavit_type.scenario_library or [],
                        )
                        result['scenarios_added'] = len(identified)

                    affidavit_type.increment_policy_version()
                    affidavit_type.save()
                    result['saved'] = True

                    # Post-save validation report
                    validation_report = validate_template_mapping(
                        template_html=affidavit_type.template_html or '',
                        intake_schema=affidavit_type.intake_schema or [],
                        placeholder_mapping=affidavit_type.placeholder_mapping or {},
                    )
                    result['config_validation'] = validation_report
                except AffidavitType.DoesNotExist:
                    result['saved'] = False
                    result['save_error'] = 'Affidavit type not found'
                except serializers.ValidationError as e:
                    result['saved'] = False
                    result['save_error'] = f'Validation error: {str(e)}'
                    result['validation_details'] = e.detail if hasattr(e, 'detail') else str(e)
                except Exception as e:
                    result['saved'] = False
                    result['save_error'] = f'Error saving: {str(e)}'
            else:
                result['saved'] = False
            
            return Response({
                'status': 'completed',
                'result': result
            })
        elif task.state == 'FAILURE':
            return Response({
                'status': 'failed',
                'error': str(task.info)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response({
                'status': task.state.lower(),
                'message': f'Task state: {task.state}'
            })


class AdminDisallowedPhrasesView(APIView):
    """
    Manage disallowed phrases for an affidavit type.
    """
    
    permission_classes = [IsAdminUser]
    
    def get(self, request, pk):
        """Get current disallowed phrases and suggestions."""
        from .services.policy_generator_service import suggest_disallowed_phrases
        
        affidavit_type = get_object_or_404(AffidavitType, pk=pk)
        
        current_phrases = affidavit_type.disallowed_phrases or []
        suggestions = suggest_disallowed_phrases(current_phrases)
        
        return Response({
            'current_phrases': current_phrases,
            'suggestions': suggestions
        })
    
    def put(self, request, pk):
        """Update disallowed phrases."""
        from .serializers import DisallowedPhrasesSerializer
        
        affidavit_type = get_object_or_404(AffidavitType, pk=pk)
        
        serializer = DisallowedPhrasesSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        affidavit_type.disallowed_phrases = serializer.validated_data['phrases']
        affidavit_type.save()
        
        return Response({
            'message': 'Disallowed phrases updated',
            'phrases': affidavit_type.disallowed_phrases
        })
    
    def post(self, request, pk):
        """Add phrases to the list."""
        affidavit_type = get_object_or_404(AffidavitType, pk=pk)
        
        new_phrases = request.data.get('phrases', [])
        if not isinstance(new_phrases, list):
            new_phrases = [new_phrases]
        
        current = set(affidavit_type.disallowed_phrases or [])
        current.update(p.strip() for p in new_phrases if p.strip())
        
        affidavit_type.disallowed_phrases = list(current)
        affidavit_type.save()
        
        return Response({
            'message': f'Added {len(new_phrases)} phrase(s)',
            'phrases': affidavit_type.disallowed_phrases
        })


class AdminAIBaseInstructionView(APIView):
    """
    Manage the global AI base instruction (singleton).
    """
    
    permission_classes = [IsAdminUser]
    
    def get(self, request):
        """Get the current active AI base instruction."""
        from .serializers import AIBaseInstructionSerializer
        from .models import AIBaseInstruction
        
        instruction = AIBaseInstruction.get_active()
        serializer = AIBaseInstructionSerializer(instruction)
        
        return Response(serializer.data)
    
    def put(self, request):
        """Update the AI base instruction."""
        from .serializers import AIBaseInstructionSerializer
        from .models import AIBaseInstruction
        
        instruction = AIBaseInstruction.get_active()
        
        serializer = AIBaseInstructionSerializer(
            instruction, 
            data=request.data, 
            partial=True,
            context={'request': request}
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        
        return Response(serializer.data)
    
    def post(self, request):
        """Create a new version of the AI base instruction."""
        from .serializers import AIBaseInstructionSerializer
        from .models import AIBaseInstruction
        
        # Increment version based on existing
        current = AIBaseInstruction.get_active()
        try:
            major, minor = current.version.split('.')
            new_version = f"{major}.{int(minor) + 1}"
        except:
            new_version = '1.1'
        
        data = {
            **request.data,
            'version': new_version,
            'is_active': True
        }
        
        serializer = AIBaseInstructionSerializer(
            data=data,
            context={'request': request}
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        
        return Response(serializer.data, status=status.HTTP_201_CREATED)


# =============================================================================
# Admin Decision Tree Views
# =============================================================================

class AdminDecisionTreeNodeListView(generics.ListCreateAPIView):
    """
    Admin endpoint to list all decision tree nodes or create new ones.
    GET: List all nodes (for admin tree management)
    POST: Create a new node
    """
    
    permission_classes = [IsSuperUser]
    
    def get_serializer_class(self):
        from .serializers import AdminDecisionTreeNodeSerializer, AdminDecisionTreeNodeCreateSerializer
        if self.request.method == 'POST':
            return AdminDecisionTreeNodeCreateSerializer
        return AdminDecisionTreeNodeSerializer
    
    def get_queryset(self):
        queryset = DecisionTreeNode.objects.all().order_by('parent_node', 'order')
        
        # Filter by parent_node
        parent = self.request.query_params.get('parent')
        if parent:
            if parent == 'root':
                queryset = queryset.filter(parent_node__isnull=True)
            else:
                queryset = queryset.filter(parent_node_id=parent)
        
        # Filter by is_active
        is_active = self.request.query_params.get('is_active')
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active.lower() == 'true')
        
        return queryset


class AdminDecisionTreeNodeDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    Admin endpoint to retrieve, update, or delete a decision tree node.
    """
    
    permission_classes = [IsSuperUser]
    queryset = DecisionTreeNode.objects.all()
    
    def get_serializer_class(self):
        from .serializers import AdminDecisionTreeNodeSerializer, AdminDecisionTreeNodeCreateSerializer
        if self.request.method in ['PUT', 'PATCH']:
            return AdminDecisionTreeNodeCreateSerializer
        return AdminDecisionTreeNodeSerializer


class AdminDecisionTreeQuestionsView(generics.ListAPIView):
    """
    Get all question nodes (nodes that can be parents).
    Used for dropdown selection when creating new nodes.
    """
    
    permission_classes = [IsSuperUser]
    
    def get(self, request):
        from .serializers import AdminDecisionTreeNodeSerializer
        
        # Get all non-leaf nodes (nodes without result_affidavit_type)
        question_nodes = DecisionTreeNode.objects.filter(
            result_affidavit_type__isnull=True,
            is_active=True
        ).order_by('parent_node', 'order')
        
        serializer = AdminDecisionTreeNodeSerializer(question_nodes, many=True)
        return Response(serializer.data)


class AdminAffidavitTypeDecisionNodesView(APIView):
    """
    Get all decision tree nodes that lead to a specific affidavit type.
    Used in the "Discovery" tab of the affidavit type edit page.
    """
    
    permission_classes = [IsSuperUser]
    
    def get(self, request, pk):
        """Get all paths leading to this affidavit type."""
        try:
            affidavit_type = AffidavitType.objects.get(pk=pk)
        except AffidavitType.DoesNotExist:
            return Response(
                {'error': 'Affidavit type not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Get all leaf nodes pointing to this type
        result_nodes = DecisionTreeNode.objects.filter(
            result_affidavit_type=affidavit_type,
            is_active=True
        )
        
        paths = []
        for node in result_nodes:
            # Build the path from root to this node
            path = []
            current = node
            while current:
                path.insert(0, {
                    'id': current.id,
                    'question_text': current.question_text if current.parent_node else 'Root Question',
                    'answer_value': current.answer_value,
                    'is_root': current.parent_node is None
                })
                current = current.parent_node
            
            paths.append({
                'node_id': node.id,
                'answer_label': node.answer_value,
                'help_text': node.help_text,
                'order': node.order,
                'path': path
            })
        
        return Response({
            'affidavit_type_id': affidavit_type.id,
            'affidavit_type_name': affidavit_type.name,
            'paths': paths
        })
    
    def post(self, request, pk):
        """Create a new decision tree node pointing to this affidavit type."""
        from .serializers import AdminDecisionTreeNodeCreateSerializer
        
        try:
            affidavit_type = AffidavitType.objects.get(pk=pk)
        except AffidavitType.DoesNotExist:
            return Response(
                {'error': 'Affidavit type not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Add the affidavit type to the request data
        data = {
            **request.data,
            'result_affidavit_type': pk
        }
        
        serializer = AdminDecisionTreeNodeCreateSerializer(data=data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        
        return Response(serializer.data, status=status.HTTP_201_CREATED)


# =============================================================================
# Site Settings Views
# =============================================================================

class SiteSettingsView(APIView):
    """
    Get or update global site settings.
    Only admins can view and modify settings.
    """
    
    permission_classes = [IsAdminUser]
    
    def get(self, request):
        """Get current site settings."""
        from .serializers import SiteSettingsSerializer
        
        settings = SiteSettings.get_settings()
        serializer = SiteSettingsSerializer(settings)
        return Response(serializer.data)
    
    def patch(self, request):
        """Update site settings."""
        from .serializers import SiteSettingsSerializer
        
        settings = SiteSettings.get_settings()
        serializer = SiteSettingsSerializer(settings, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        
        # Track who made the update
        settings.updated_by = request.user
        serializer.save()
        
        return Response({
            'success': True,
            'message': 'Settings updated successfully',
            'settings': serializer.data
        })


# =============================================================================
# Admin Request Listing by Type (for viewing all generated affidavits)
# =============================================================================

class AdminTypeRequestsView(generics.ListAPIView):
    """
    Admin endpoint to list all requests for a specific affidavit type.
    Shows requests of all statuses: draft, submitted, in review, approved, completed, rejected.
    Includes rejection reasons from reviewers and friction reports from commissioners.
    """
    
    permission_classes = [IsAdminUser]
    serializer_class = RequestDetailSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['request_code', 'user__username', 'user__email']
    ordering_fields = ['created_at', 'updated_at', 'status', 'submitted_at', 'approved_at', 'completed_at']
    ordering = ['-created_at']
    
    def get_queryset(self):
        type_id = self.kwargs.get('pk')
        affidavit_type = get_object_or_404(AffidavitType, pk=type_id)
        
        # Get status filter from query params
        status_filter = self.request.query_params.get('status', None)
        
        queryset = Request.objects.filter(affidavit_type=affidavit_type)
        
        if status_filter and status_filter != 'all':
            queryset = queryset.filter(status=status_filter)
        
        # Prefetch related data for better performance
        return queryset.select_related('user', 'affidavit_type', 'commissioner', 'locked_by')
    
    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        
        # Paginate
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            data = serializer.data
            
            # Enhance with rejection/friction info
            for item in data:
                request_obj = Request.objects.get(id=item['id'])
                
                # Get rejection reason from reviewer (stored in qa_flags_json)
                reviewer_rejection = None
                if request_obj.status == 'rejected':
                    for flag in request_obj.qa_flags_json:
                        if isinstance(flag, dict) and flag.get('type') == 'rejection':
                            reviewer_rejection = {
                                'reason': flag.get('description', ''),
                                'reviewer': flag.get('reviewer', ''),
                                'timestamp': flag.get('timestamp', '')
                            }
                            break
                
                # Get friction reports from commissioners
                friction_reports = []
                for fr in request_obj.friction_reports.all():
                    friction_reports.append({
                        'id': fr.id,
                        'reason': fr.reason,
                        'commissioner': fr.commissioner.get_full_name() or fr.commissioner.username,
                        'created_at': fr.created_at.isoformat(),
                        'is_resolved': fr.is_resolved,
                        'resolution_notes': fr.resolution_notes
                    })
                
                item['reviewer_rejection'] = reviewer_rejection
                item['friction_reports'] = friction_reports
            
            return self.get_paginated_response(data)
        
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)


# =============================================================================
# Ticket Views
# =============================================================================

class TicketViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing support tickets.
    Users see their own; Admins see all.
    """
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        user = self.request.user
        if user.role == 'admin':
            return Ticket.objects.all().select_related('user', 'request').prefetch_related('messages')
        return Ticket.objects.filter(user=user).select_related('user', 'request').prefetch_related('messages')
        
    def get_serializer_class(self):
        if self.action in ['retrieve', 'update', 'partial_update']:
            return TicketDetailSerializer
        return TicketSerializer
        
    def perform_create(self, serializer):
        ticket = serializer.save(user=self.request.user)
        
        # Create attachment if provided
        files = self.request.FILES.getlist('files')
        for file in files:
            TicketAttachment.objects.create(ticket=ticket, file=file)
            
        # Send notification and mark email sent time
        try:
            send_ticket_created_notification(ticket)
            ticket.last_email_sent_at = timezone.now()
            ticket.save(update_fields=['last_email_sent_at'])
        except Exception as e:
            # Log error but don't fail the request
            print(f"Failed to send ticket notification: {e}")
            
    @action(detail=True, methods=['post'])
    def reply(self, request, pk=None):
        ticket = self.get_object()
        message_text = request.data.get('message')
        
        if not message_text:
            return Response({'error': 'Message is required'}, status=status.HTTP_400_BAD_REQUEST)
            
        is_internal = request.data.get('is_internal', False) if request.user.role == 'admin' else False
        
        message = TicketMessage.objects.create(
            ticket=ticket,
            sender=request.user,
            message=message_text,
            is_internal=is_internal
        )
        
        # Update ticket updated_at timestamp
        ticket.save()
        
        # Send notification with 5-minute debouncing
        # Only send if: admin reply to user, not internal note, and email cooldown passed
        if request.user.role == 'admin' and request.user != ticket.user and not is_internal:
            from datetime import timedelta
            should_send_email = False
            
            if ticket.last_email_sent_at is None:
                # First admin reply - always send
                should_send_email = True
            else:
                # Check if 5 minutes have passed since last email
                time_since_last = timezone.now() - ticket.last_email_sent_at
                if time_since_last >= timedelta(seconds=30):
                    should_send_email = True
            
            if should_send_email:
                try:
                    send_ticket_reply_notification(ticket, message)
                    ticket.last_email_sent_at = timezone.now()
                    ticket.save(update_fields=['last_email_sent_at'])
                except Exception as e:
                    print(f"Failed to send ticket reply notification: {e}")
        
        return Response(TicketMessageSerializer(message).data, status=status.HTTP_201_CREATED)
        
    @action(detail=True, methods=['post'])
    def status(self, request, pk=None):
        if request.user.role != 'admin':
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)
            
        ticket = self.get_object()
        old_status = ticket.status
        new_status = request.data.get('status')
        priority = request.data.get('priority')
        
        if new_status:
            ticket.status = new_status
            if new_status in [Ticket.Status.RESOLVED, Ticket.Status.CLOSED]:
                ticket.resolved_at = timezone.now()
                
        if priority:
            ticket.priority = priority
            
        ticket.save()
        
        # Send email notification on status changes (always, bypass debounce)
        if new_status and new_status != old_status:
            try:
                # Create a status change message for email context
                status_message = TicketMessage(
                    ticket=ticket,
                    sender=request.user,
                    message=f"Ticket status changed to: {ticket.get_status_display()}"
                )
                send_ticket_reply_notification(ticket, status_message)
                ticket.last_email_sent_at = timezone.now()
                ticket.save(update_fields=['last_email_sent_at'])
            except Exception as e:
                print(f"Failed to send status change notification: {e}")
        
        return Response(TicketDetailSerializer(ticket).data)
