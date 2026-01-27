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
    UserRegistrationView, UserProfileView,
    PasswordResetRequestView, PasswordResetConfirmView,
    # Decision Tree
    DecisionTreeRootView, DecisionTreeNodeView, DecisionTreeTraverseView,
    # Affidavit Types
    AffidavitTypeListView, AffidavitTypeDetailView,
    # Requests
    RequestCreateView, RequestDetailView, RequestPatchView, RequestDeleteView,
    RequestSubmitView, SelectCommissionerView, RequestByCodeView, RequestTakeoverView,
    MyRequestsView, DownloadPDFView, DownloadWordView, RequestStatusView,
    DevApproveView, ClarificationResponseView,
    # Commissioner
    MarkCompleteView, FrictionReportCreateView, CommissionerStampsView,
    CommissionerPDFPreferencesView, CommissionerAssignedRequestsView,
    # Reviewer
    ReviewQueueView, ReviewDetailView, ApproveRequestView,
    RejectRequestView, OverrideAIFlagView, ReviewerStatsView, RequestClarificationView,
    # Admin
    AffidavitTypePolicyView, ConfidenceDashboardView, LearningExportView,
    FrictionDashboardView, PromoteToInstantModeView,
    FullConfidenceDashboardView, TypeTrendView, WeeklyLearningReportView,
    CostDashboardView, LearningSuggestionsView,
    # Admin Staff Management
    AdminCommissionerListView, AdminCommissionerDetailView,
    AdminReviewerListView, AdminReviewerDetailView,
    # Admin Affidavit Type CRUD
    AdminAffidavitTypeListView, AdminAffidavitTypeDetailView,
    AdminAffidavitTypeDuplicateView,
    # Admin Document Upload & Policy Generation
    AdminDocumentUploadView, AdminDocumentListView,
    AdminGeneratePolicyView, AdminDisallowedPhrasesView,
    AdminAIBaseInstructionView,
    # Admin Decision Tree
    AdminDecisionTreeNodeListView, AdminDecisionTreeNodeDetailView,
    AdminDecisionTreeQuestionsView, AdminAffidavitTypeDecisionNodesView,
    # Public
    PublicCommissionerListView,
)

app_name = 'affidavits'

urlpatterns = [
    # ==========================================================================
    # Authentication Endpoints
    # ==========================================================================
    path('auth/register/', UserRegistrationView.as_view(), name='register'),
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
    path('requests/<int:pk>/', RequestDetailView.as_view(), name='request_detail'),
    path('requests/<int:pk>/save/', RequestPatchView.as_view(), name='request_save'),
    path('requests/<int:pk>/submit/', RequestSubmitView.as_view(), name='request_submit'),
    path('requests/<int:pk>/select-commissioner/', SelectCommissionerView.as_view(), name='select_commissioner'),
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
    path('commissioner/pdf-preferences/', CommissionerPDFPreferencesView.as_view(), name='pdf_preferences'),
    
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
    
    # ==========================================================================
    # Admin Endpoints (Stories 4.1, 4.2, 4.3)
    # ==========================================================================
    path('admin/affidavit-types/<int:pk>/policy/', AffidavitTypePolicyView.as_view(), name='policy_update'),
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
    path('admin/reviewers/', AdminReviewerListView.as_view(), name='admin_reviewer_list'),
    path('admin/reviewers/<int:pk>/', AdminReviewerDetailView.as_view(), name='admin_reviewer_detail'),
    
    # ==========================================================================
    # Admin Affidavit Type CRUD Endpoints
    # ==========================================================================
    path('admin/types/', AdminAffidavitTypeListView.as_view(), name='admin_type_list'),
    path('admin/types/<int:pk>/', AdminAffidavitTypeDetailView.as_view(), name='admin_type_detail'),
    path('admin/types/<int:pk>/duplicate/', AdminAffidavitTypeDuplicateView.as_view(), name='admin_type_duplicate'),
    
    # ==========================================================================
    # Admin Document Upload & Policy Generation Endpoints
    # ==========================================================================
    path('admin/types/<int:pk>/upload-documents/', AdminDocumentUploadView.as_view(), name='admin_upload_documents'),
    path('admin/types/<int:pk>/documents/', AdminDocumentListView.as_view(), name='admin_documents_list'),
    path('admin/types/<int:pk>/generate-policy/', AdminGeneratePolicyView.as_view(), name='admin_generate_policy'),
    path('admin/types/<int:pk>/disallowed-phrases/', AdminDisallowedPhrasesView.as_view(), name='admin_disallowed_phrases'),
    path('admin/ai-instruction/', AdminAIBaseInstructionView.as_view(), name='admin_ai_instruction'),
    
    # ==========================================================================
    # Admin Decision Tree Endpoints
    # ==========================================================================
    path('admin/decision-tree/', AdminDecisionTreeNodeListView.as_view(), name='admin_decision_tree_list'),
    path('admin/decision-tree/<int:pk>/', AdminDecisionTreeNodeDetailView.as_view(), name='admin_decision_tree_detail'),
    path('admin/decision-tree/questions/', AdminDecisionTreeQuestionsView.as_view(), name='admin_decision_tree_questions'),
    path('admin/types/<int:pk>/decision-nodes/', AdminAffidavitTypeDecisionNodesView.as_view(), name='admin_type_decision_nodes'),
]
