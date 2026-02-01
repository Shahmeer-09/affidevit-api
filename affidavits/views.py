"""
Affidavit Express - API Views

RESTful API endpoints for all user roles:
- Public: Decision tree, request creation, auto-save, submission
- Commissioner: Retrieve by code, mark complete, friction reports
- Reviewer: Review queue, approve/reject, edit tracking
- Admin: Policy management, dashboard, learning export
"""

import json
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.db.models import Count, Q, F
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
    SiteSettings
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
    ConfidenceDashboardSerializer, LearningExportSerializer,
    PaymentLogSerializer, CommissionerPaymentSummarySerializer, MarkAsPaidSerializer
)
from .authentication import (
    IsCommissioner, IsReviewer, IsAdminUser, 
    IsOwnerOrAdmin, IsCommissionerOrReviewerOrAdmin
)
from .services import process_request, generate_affidavit_pdf
from .services.ai_service import validate_inputs_before_submission, translate_to_english
from .services.notification_service import (
    send_approval_notification, 
    send_clarification_notification,
    send_completion_notification
)
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


# =============================================================================
# Authentication Views
# =============================================================================

class UserRegistrationView(generics.CreateAPIView):
    """Public endpoint for user registration."""
    
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [AllowAny]
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        # Generate JWT tokens for the new user
        from rest_framework_simplejwt.tokens import RefreshToken
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'user': UserSerializer(user).data,
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }, status=status.HTTP_201_CREATED)


class CommissionerRegistrationView(APIView):
    """
    Public endpoint for commissioner self-registration.
    Commissioners can sign up with their details, availability, and profile image.
    """
    
    permission_classes = [AllowAny]
    
    def post(self, request):
        from .serializers import CommissionerRegistrationSerializer, CommissionerSerializer
        
        serializer = CommissionerRegistrationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        # Don't auto-login - require admin approval first
        return Response({
            'message': 'Your request has been sent! Once approved by admin, you will be able to login.',
            'email': user.email,
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
        
        # STEP 1: Translate non-English content to English BEFORE validation
        translated_answers = translate_to_english(answers_json)
        
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
            affidavit_type_name=affidavit_type.name
        )
        
        # ===== DEBUG LOGGING =====
        logger.info("=" * 80)
        logger.info(f"[VALIDATE_VIEW] VALIDATION RESULT: {validation_result}")
        logger.info("=" * 80)
        # ===== END DEBUG LOGGING =====
        
        return Response({
            'success': True,
            'all_valid': validation_result.get('all_valid', True),
            'invalid_fields': validation_result.get('invalid_fields', {}),
            'validation_notes': validation_result.get('validation_notes', []),
            'field_checks': validation_result.get('field_checks', [])
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
        
        # Validate submission
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
        
        # Only allow selecting commissioner for approved requests
        if request_obj.status != Request.Status.APPROVED:
            return Response(
                {'error': 'Can only select commissioner for approved requests'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        commissioner_id = request.data.get('commissioner_id')
        
        # Allow withdrawing commissioner selection (set to null)
        if commissioner_id is None:
            old_commissioner = request_obj.commissioner
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
        
        # Log the event
        RequestEvent.objects.create(
            request=request_obj,
            action=RequestEvent.Action.STATUS_CHANGED,
            actor=request.user,
            actor_role=request.user.role,
            details={'action': 'payment_completed', 'amount': '50.00', 'currency': 'TTD'}
        )
        
        serializer = RequestDetailSerializer(request_obj)
        return Response({
            'success': True,
            'message': 'Payment confirmed successfully',
            'request': serializer.data
        })


class RequestByCodeView(APIView):
    """
    Retrieve a request by its code (for commissioners and admins).
    Story 2.1 - Universal Request Retrieval.
    - Commissioners can only access requests assigned to them (and not completed ones)
    - Admins can access any request for support purposes
    """
    
    permission_classes = [IsCommissionerOrReviewerOrAdmin]
    
    def get(self, request, code):
        request_obj = get_object_or_404(Request, request_code=code.upper())
        
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
        
        # Send completion notification
        send_completion_notification(request_obj, stamp)
        
        return Response({
            'success': True,
            'message': 'Request marked as completed',
            'stamp': StampSerializer(stamp).data
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
    Shows requests needing review, sorted by oldest first.
    """
    
    serializer_class = RequestReviewerSerializer
    permission_classes = [IsReviewer]
    
    def get_queryset(self):
        return Request.objects.filter(
            status=Request.Status.NEEDS_REVIEW
        ).order_by('created_at')


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
        
        # Send notification async
        send_notification_async.delay('approval', request_obj.id)
        
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
        from django.db.models import Count, Avg
        from datetime import timedelta
        
        today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        reviewer = request.user
        
        # Get pending review count
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
        
        # Check if PDF exists
        if not request_obj.pdf_file:
            # Generate PDF if not exists (allow NEEDS_REVIEW for draft preview)
            if request_obj.status not in [Request.Status.APPROVED, Request.Status.COMPLETED, Request.Status.NEEDS_REVIEW, Request.Status.DRAFT_READY]:
                return Response(
                    {'error': 'PDF not available for this request'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Need draft_text to generate PDF
            if not request_obj.draft_text:
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
    
    def get_serializer_class(self):
        from .serializers import AffidavitTypeAdminSerializer
        return AffidavitTypeAdminSerializer


class AdminAffidavitTypeDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    Admin endpoint to view, update, or delete an affidavit type.
    """
    
    permission_classes = [IsAdminUser]
    queryset = AffidavitType.objects.all()
    
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
    
    permission_classes = [IsAdminUser]
    
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
        from .services.policy_generator_service import (
            convert_detected_fields_to_intake_schema,
            build_policy_json_from_generation
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
                    
                    affidavit_type.increment_policy_version()
                    affidavit_type.save()
                    result['saved'] = True
                except AffidavitType.DoesNotExist:
                    result['saved'] = False
                    result['save_error'] = 'Affidavit type not found'
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
    
    permission_classes = [IsAdminUser]
    
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
    
    permission_classes = [IsAdminUser]
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
    
    permission_classes = [IsAdminUser]
    
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
    
    permission_classes = [IsAdminUser]
    
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

