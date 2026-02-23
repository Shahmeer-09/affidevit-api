"""
Affidavit Express - URL Configuration

API endpoints organized by user role:
- /api/auth/ - Authentication (JWT)
- /api/public/ - Public endpoints (decision tree, affidavit types)
- /api/requests/ - Request management
- /api/commissioner/ - Commissioner endpoints
- /api/reviewer/ - Reviewer endpoints
- /api/admin/ - Admin endpoints
"""

from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from .authentication import CustomTokenObtainPairView
from .views import (
    # Auth
    UserRegistrationView, UserProfileView, CommissionerRegistrationView,
    PasswordResetRequestView, PasswordResetConfirmView, VerifyOTPView,
    # Decision Tree
    DecisionTreeRootView, DecisionTreeNodeView, DecisionTreeTraverseView,
    # Affidavit Types
    AffidavitTypeListView, AffidavitTypeDetailView,
    # Requests
    RequestCreateView, RequestDetailView, RequestPatchView, RequestDeleteView,
    RequestSubmitView, SelectCommissionerView, MarkPaidView, RequestByCodeView, RequestTakeoverView,
    MyRequestsView, DownloadPDFView, DownloadWordView, RequestStatusView,
    DevApproveView, ClarificationResponseView, ValidateRequestInputView,
    # Commissioner
    MarkCompleteView, FrictionReportCreateView, CommissionerStampsView,
    CommissionerPDFPreferencesView, CommissionerAssignedRequestsView,
    # Reviewer
    ReviewQueueView, ReviewDetailView, ApproveRequestView,
    RejectRequestView, OverrideAIFlagView, ReviewerStatsView, RequestClarificationView,
    ReviewerFeedbackCreateView,
    SubmitFeedbackView,
    # Admin
    AffidavitTypePolicyView, ValidationRulesView, PlaceholderMappingView, TemplateLivePreviewView, AIDraftPreviewView, 
    FieldSuggestionsView, AtomicFieldInsertionView, ConfidenceDashboardView, LearningExportView,
    FrictionDashboardView, PromoteToInstantModeView,
    FullConfidenceDashboardView, TypeTrendView, WeeklyLearningReportView,
    CostDashboardView, LearningSuggestionsView,
    # Admin Staff Management
    AdminCommissionerListView, AdminCommissionerDetailView,
    AdminReviewerListView, AdminReviewerDetailView,
    AdminCommissionerPaymentSummaryView, AdminCommissionerPaymentHistoryView,
    AdminMarkCommissionerPaidView, AdminAllPaymentLogsView,
    AdminGenerateSlotsView,
    AdminReviewerFeedbackListView,
    # Admin Affidavit Type CRUD
    AdminAffidavitTypeListView, AdminAffidavitTypeDetailView,
    AdminAffidavitTypeDuplicateView, AdminTypeRequestsView,
    ValidateAffidavitConfigView,
    # Admin Document Upload & Policy Generation
    AdminDocumentUploadView, AdminDocumentListView,
    AdminGeneratePolicyView, AdminPolicyTaskStatusView, AdminDisallowedPhrasesView,
    AdminAIBaseInstructionView, RefineTemplateView, RefineInstructionView,
    # Admin Decision Tree
    AdminDecisionTreeNodeListView, AdminDecisionTreeNodeDetailView,
    AdminDecisionTreeQuestionsView, AdminAffidavitTypeDecisionNodesView,
    # Admin Site Settings
    SiteSettingsView,
    # Public
    PublicCommissionerListView,
    # Tickets
    TicketViewSet,
    CommissionerSlotsView, BookSlotView, CommissionerBookedSlotsView,
    CommissionerAcceptSlotView, CommissionerRejectSlotView, CommissionerCancelSlotView,
    GuestAuthView
)

app_name = 'affidavits'

urlpatterns = [
    # ==========================================================================
    # Authentication Endpoints
    # ==========================================================================
    path('auth/register/', UserRegistrationView.as_view(), name='register'),
    path('auth/register/commissioner/', CommissionerRegistrationView.as_view(), name='register_commissioner'),
    path('auth/guest-signup/start/', GuestAuthView.as_view({'post': 'start'}), name='guest_signup_start'),
    path('auth/guest-signup/verify/', GuestAuthView.as_view({'post': 'verify'}), name='guest_signup_verify'),
    path('auth/verify-otp/', VerifyOTPView.as_view(), name='verify_otp'),
    path('auth/token/', CustomTokenObtainPairView.as_view(), name='token_obtain'),
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/profile/', UserProfileView.as_view(), name='profile'),
    path('auth/password-reset/', PasswordResetRequestView.as_view(), name='password_reset'),
    path('auth/password-reset/confirm/', PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    
    # ==========================================================================
    # Public Endpoints - Decision Tree (Story 1.1)
    # ==========================================================================
    path('decision-tree/', DecisionTreeRootView.as_view(), name='decision_tree_root'),
    path('decision-tree/<int:pk>/', DecisionTreeNodeView.as_view(), name='decision_tree_node'),
    path('decision-tree/traverse/', DecisionTreeTraverseView.as_view(), name='decision_tree_traverse'),
    
    # ==========================================================================
    # Public Endpoints - Affidavit Types
    # ==========================================================================
    path('affidavit-types/', AffidavitTypeListView.as_view(), name='affidavit_type_list'),
    path('affidavit-types/<int:pk>/', AffidavitTypeDetailView.as_view(), name='affidavit_type_detail'),
    
    # ==========================================================================
    # Public Endpoints - Commissioners (for Landing Page)
    # ==========================================================================
    path('commissioners/', PublicCommissionerListView.as_view(), name='public_commissioners'),
    
    # ==========================================================================
    # Request Endpoints (Stories 1.2, 1.3, 1.4)
    # ==========================================================================
    path('requests/', RequestCreateView.as_view(), name='request_create'),
    path('requests/my/', MyRequestsView.as_view(), name='my_requests'),
    path('requests/validate-input/', ValidateRequestInputView.as_view(), name='validate_request_input'),
    path('requests/<int:pk>/', RequestDetailView.as_view(), name='request_detail'),
    path('requests/<int:pk>/save/', RequestPatchView.as_view(), name='request_save'),
    path('requests/<int:pk>/submit/', RequestSubmitView.as_view(), name='request_submit'),
    path('requests/<int:pk>/select-commissioner/', SelectCommissionerView.as_view(), name='select_commissioner'),
    path('requests/<int:pk>/mark-paid/', MarkPaidView.as_view(), name='mark_paid'),
    path('requests/<int:pk>/status/', RequestStatusView.as_view(), name='request_status'),
    path('requests/<int:pk>/pdf/', DownloadPDFView.as_view(), name='request_pdf'),
    path('requests/<int:pk>/word/', DownloadWordView.as_view(), name='request_word'),
    path('requests/<int:pk>/dev-approve/', DevApproveView.as_view(), name='request_dev_approve'),
    path('requests/<int:pk>/clarification/', ClarificationResponseView.as_view(), name='request_clarification'),
    path('requests/<int:pk>/delete/', RequestDeleteView.as_view(), name='request_delete'),
    
    # ==========================================================================
    # Commissioner Endpoints (Stories 2.1, 2.2, 2.3)
    # ==========================================================================
    path('commissioner/my-requests/', CommissionerAssignedRequestsView.as_view(), name='commissioner_assigned_requests'),
    path('commissioner/lookup/<str:code>/', RequestByCodeView.as_view(), name='commissioner_lookup'),
    path('commissioner/takeover/<str:code>/', RequestTakeoverView.as_view(), name='commissioner_takeover'),
    path('commissioner/complete/<int:pk>/', MarkCompleteView.as_view(), name='commissioner_complete'),
    path('commissioner/report/', FrictionReportCreateView.as_view(), name='friction_report'),
    path('commissioner/stamps/', CommissionerStampsView.as_view(), name='commissioner_stamps'),
    path('commissioner/schedule/', CommissionerBookedSlotsView.as_view(), name='commissioner_schedule'),
    path('commissioner/pdf-preferences/', CommissionerPDFPreferencesView.as_view(), name='pdf_preferences'),
    path('commissioners/<int:pk>/slots/', CommissionerSlotsView.as_view(), name='commissioner_slots'),
    path('slots/<int:pk>/book/', BookSlotView.as_view(), name='book_slot'),
    path('slots/<int:pk>/accept/', CommissionerAcceptSlotView.as_view(), name='slot_accept'),
    path('slots/<int:pk>/reject/', CommissionerRejectSlotView.as_view(), name='slot_reject'),
    path('slots/<int:pk>/cancel/', CommissionerCancelSlotView.as_view(), name='slot_cancel'),
    
    # ==========================================================================
    # Reviewer Endpoints (Stories 3.1, 3.2, 3.3)
    # ==========================================================================
    path('reviewer/queue/', ReviewQueueView.as_view(), name='review_queue'),
    path('reviewer/stats/', ReviewerStatsView.as_view(), name='reviewer_stats'),
    path('reviewer/<int:pk>/', ReviewDetailView.as_view(), name='review_detail'),
    path('reviewer/<int:pk>/approve/', ApproveRequestView.as_view(), name='review_approve'),
    path('reviewer/<int:pk>/reject/', RejectRequestView.as_view(), name='review_reject'),
    path('reviewer/<int:pk>/clarify/', RequestClarificationView.as_view(), name='review_clarify'),
    path('reviewer/<int:pk>/override-flag/', OverrideAIFlagView.as_view(), name='override_flag'),
    path('reviewer/<int:pk>/feedback/', ReviewerFeedbackCreateView.as_view(), name='review_feedback'),
    
    # ==========================================================================
    # Admin Endpoints (Stories 4.1, 4.2, 4.3)
    # ==========================================================================
    path('admin/affidavit-types/<int:pk>/policy/', AffidavitTypePolicyView.as_view(), name='policy_update'),
    path('admin/affidavit-types/<int:pk>/validation-rules/', ValidationRulesView.as_view(), name='validation_rules'),
    path('admin/affidavit-types/<int:pk>/placeholder-mapping/', PlaceholderMappingView.as_view(), name='placeholder_mapping'),
    path('admin/affidavit-types/<int:pk>/template-preview/', TemplateLivePreviewView.as_view(), name='template_preview'),
    path('admin/affidavit-types/<int:pk>/ai-draft-preview/', AIDraftPreviewView.as_view(), name='ai_draft_preview'),
    path('admin/affidavit-types/<int:pk>/field-suggestions/', FieldSuggestionsView.as_view(), name='field_suggestions'),
    path('admin/affidavit-types/<int:pk>/insert-field/', AtomicFieldInsertionView.as_view(), name='insert_field'),
    path('admin/affidavit-types/<int:pk>/promote/', PromoteToInstantModeView.as_view(), name='promote_instant'),
    path('admin/dashboard/', ConfidenceDashboardView.as_view(), name='confidence_dashboard'),
    path('admin/dashboard/full/', FullConfidenceDashboardView.as_view(), name='full_confidence_dashboard'),
    path('admin/dashboard/costs/', CostDashboardView.as_view(), name='cost_dashboard'),
    path('admin/dashboard/type/<int:pk>/trend/', TypeTrendView.as_view(), name='type_trend'),
    path('admin/learning-export/', LearningExportView.as_view(), name='learning_export'),
    path('admin/learning-report/', WeeklyLearningReportView.as_view(), name='weekly_learning_report'),
    path('admin/learning-suggestions/', LearningSuggestionsView.as_view(), name='learning_suggestions'),
    path('admin/friction/', FrictionDashboardView.as_view(), name='friction_dashboard'),
    
    # ==========================================================================
    # Admin Staff Management Endpoints
    # ==========================================================================
    path('admin/commissioners/', AdminCommissionerListView.as_view(), name='admin_commissioner_list'),
    path('admin/commissioners/<int:pk>/', AdminCommissionerDetailView.as_view(), name='admin_commissioner_detail'),
    path('admin/commissioners/<int:pk>/payment-summary/', AdminCommissionerPaymentSummaryView.as_view(), name='admin_commissioner_payment_summary'),
    path('admin/commissioners/<int:pk>/payment-history/', AdminCommissionerPaymentHistoryView.as_view(), name='admin_commissioner_payment_history'),
    path('admin/commissioners/<int:pk>/mark-paid/', AdminMarkCommissionerPaidView.as_view(), name='admin_commissioner_mark_paid'),
    path('admin/commissioners/<int:pk>/generate-slots/', AdminGenerateSlotsView.as_view(), name='admin_commissioner_generate_slots'),
    path('admin/payment-logs/', AdminAllPaymentLogsView.as_view(), name='admin_all_payment_logs'),
    path('admin/reviewers/', AdminReviewerListView.as_view(), name='admin_reviewer_list'),
    path('admin/reviewers/<int:pk>/', AdminReviewerDetailView.as_view(), name='admin_reviewer_detail'),
    path('admin/reviewer-feedback/', AdminReviewerFeedbackListView.as_view(), name='admin_reviewer_feedback'),
    path('admin/feedback/', SubmitFeedbackView.as_view(), name='admin_feedback'),
    
    # ==========================================================================
    # Admin Affidavit Type CRUD Endpoints
    # ==========================================================================
    path('admin/types/', AdminAffidavitTypeListView.as_view(), name='admin_type_list'),
    path('admin/types/<int:pk>/', AdminAffidavitTypeDetailView.as_view(), name='admin_type_detail'),
    path('admin/types/<int:pk>/duplicate/', AdminAffidavitTypeDuplicateView.as_view(), name='admin_type_duplicate'),
    path('admin/types/<int:pk>/requests/', AdminTypeRequestsView.as_view(), name='admin_type_requests'),
    path('admin/affidavit-types/<int:pk>/validate-config/', ValidateAffidavitConfigView.as_view(), name='validate_config'),
    
    # ==========================================================================
    # Admin Document Upload & Policy Generation Endpoints
    # ==========================================================================
    path('admin/types/<int:pk>/upload-documents/', AdminDocumentUploadView.as_view(), name='admin_upload_documents'),
    path('admin/types/<int:pk>/documents/', AdminDocumentListView.as_view(), name='admin_documents_list'),
    path('admin/types/<int:pk>/generate-policy/', AdminGeneratePolicyView.as_view(), name='admin_generate_policy'),
    path('admin/policy-task/<str:task_id>/', AdminPolicyTaskStatusView.as_view(), name='admin_policy_task_status'),
    path('admin/types/<int:pk>/refine-template/', RefineTemplateView.as_view(), name='admin_refine_template'),
    path('admin/types/<int:pk>/refine-instruction/', RefineInstructionView.as_view(), name='admin_refine_instruction'),
    path('admin/types/<int:pk>/disallowed-phrases/', AdminDisallowedPhrasesView.as_view(), name='admin_disallowed_phrases'),
    path('admin/ai-instruction/', AdminAIBaseInstructionView.as_view(), name='admin_ai_instruction'),
    
    # ==========================================================================
    # Admin Decision Tree Endpoints
    # ==========================================================================
    path('admin/decision-tree/', AdminDecisionTreeNodeListView.as_view(), name='admin_decision_tree_list'),
    path('admin/decision-tree/<int:pk>/', AdminDecisionTreeNodeDetailView.as_view(), name='admin_decision_tree_detail'),
    path('admin/decision-tree/questions/', AdminDecisionTreeQuestionsView.as_view(), name='admin_decision_tree_questions'),
    path('admin/types/<int:pk>/decision-nodes/', AdminAffidavitTypeDecisionNodesView.as_view(), name='admin_type_decision_nodes'),
    
    # ==========================================================================
    # Admin Site Settings Endpoints
    # ==========================================================================
    path('admin/settings/', SiteSettingsView.as_view(), name='admin_site_settings'),
    
    # ==========================================================================
    # Ticket Endpoints
    # ==========================================================================
    path('tickets/', TicketViewSet.as_view({'get': 'list', 'post': 'create'}), name='ticket_list'),
    path('tickets/<int:pk>/', TicketViewSet.as_view({'get': 'retrieve'}), name='ticket_detail'),
    path('tickets/<int:pk>/reply/', TicketViewSet.as_view({'post': 'reply'}), name='ticket_reply'),
    path('tickets/<int:pk>/status/', TicketViewSet.as_view({'post': 'status'}), name='ticket_status'),
]
