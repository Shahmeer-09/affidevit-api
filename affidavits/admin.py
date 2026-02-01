"""
Affidavit Express - Django Admin Configuration

Admin interface for managing all models with custom displays,
filters, search, and actions.
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.html import format_html
from .models import (
    User, AffidavitType, DecisionTreeNode, 
    Request, Stamp, FrictionReport, ReviewerEdit,
    RequestEvent, AIRun, AIBaseInstruction, PaymentLog
)


@admin.register(AIBaseInstruction)
class AIBaseInstructionAdmin(admin.ModelAdmin):
    """Admin for global AI base instruction (singleton)."""
    
    list_display = [
        'version', 'is_active', 'instruction_preview', 
        'updated_by', 'updated_at', 'created_at'
    ]
    list_filter = ['is_active', 'created_at', 'updated_at']
    search_fields = ['instruction_text', 'version']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']
    
    fieldsets = (
        (None, {
            'fields': ('version', 'is_active')
        }),
        ('Instruction', {
            'fields': ('instruction_text',),
            'description': 'This instruction is used globally for ALL affidavit types as the base system prompt.'
        }),
        ('Metadata', {
            'fields': ('updated_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def instruction_preview(self, obj):
        preview = obj.instruction_text[:100]
        return preview + '...' if len(obj.instruction_text) > 100 else preview
    instruction_preview.short_description = 'Instruction Preview'
    
    def save_model(self, request, obj, form, change):
        """Set the updated_by field to current user."""
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)
    
    def has_delete_permission(self, request, obj=None):
        """Prevent deletion if it's the only active instruction."""
        if obj and obj.is_active:
            active_count = AIBaseInstruction.objects.filter(is_active=True).count()
            return active_count > 1
        return True


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    """Custom User admin with role management."""
    
    list_display = [
        'username', 'email', 'role', 'first_name', 'last_name', 
        'is_active', 'is_featured', 'date_joined','commission_number', 'payout_rate'
    ]
    list_filter = ['role', 'is_active', 'is_staff', 'is_featured', 'date_joined']
    search_fields = ['username', 'email', 'first_name', 'last_name']
    ordering = ['-date_joined']
    
    fieldsets = BaseUserAdmin.fieldsets + (
        ('Role & Profile', {
            'fields': ('role', 'phone_number', 'profile_image', 'bio', 'is_featured')
        }),
        ('Commissioner Info', {
            'fields': ('commission_number', 'payout_rate'),
            'classes': ('collapse',)
        }),
    )
    
    add_fieldsets = BaseUserAdmin.add_fieldsets + (
        ('Role', {
            'fields': ('role',)
        }),
    )


@admin.register(AffidavitType)
class AffidavitTypeAdmin(admin.ModelAdmin):
    """Admin for affidavit types with policy versioning and tier management."""
    
    list_display = [
        'name', 'tier', 'default_mode', 'confidence_status_badge', 
        'is_active', 'enabled_on_homepage', 'template_count', 
        'disallowed_phrases_count', 'request_count', 'updated_at'
    ]
    list_filter = ['is_active', 'tier', 'default_mode', 'confidence_status', 'enabled_on_homepage']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['name']
    
    fieldsets = (
        (None, {
            'fields': ('name', 'description', 'is_active', 'enabled_on_homepage')
        }),
        ('Tier & Mode', {
            'fields': ('tier', 'default_mode', 'confidence_status')
        }),
        ('AI Configuration', {
            'fields': ('template_documents', 'disallowed_phrases', 'policy_json'),
            'description': 'Template documents are uploaded via API. System instructions stored in policy_json.system_prompt'
        }),
        ('Intake & Scenarios', {
            'fields': ('intake_schema', 'scenario_library'),
            'classes': ('collapse',),
            'description': 'Questions shown to users and example scenarios for testing.'
        }),
        ('Advanced', {
            'fields': ('template_html',),
            'classes': ('collapse',),
            'description': 'AI-extracted HTML template from uploaded documents.'
        }),
        ('Versioning', {
            'fields': ('policy_version', 'prompt_pack_version', 'template_version', 
                       'created_at', 'updated_at'),
        }),
    )
    
    def request_count(self, obj):
        return obj.requests.count()
    request_count.short_description = 'Requests'
    
    def template_count(self, obj):
        docs = obj.template_documents or []
        return len(docs)
    template_count.short_description = 'Templates'
    
    def disallowed_phrases_count(self, obj):
        phrases = obj.disallowed_phrases or []
        return len(phrases)
    disallowed_phrases_count.short_description = 'Disallowed'
    
    def confidence_status_badge(self, obj):
        colors = {
            'red': '#dc3545',
            'yellow': '#ffc107',
            'green': '#28a745',
        }
        text_colors = {
            'red': 'white',
            'yellow': 'black',
            'green': 'white',
        }
        color = colors.get(obj.confidence_status, '#6c757d')
        text_color = text_colors.get(obj.confidence_status, 'white')
        return format_html(
            '<span style="background-color: {}; color: {}; padding: 3px 8px; '
            'border-radius: 3px; font-weight: bold;">{}</span>',
            color,
            text_color,
            obj.get_confidence_status_display()
        )
    confidence_status_badge.short_description = 'Confidence'
    
    actions = ['promote_to_instant', 'demote_to_review']
    
    @admin.action(description='Promote to Instant Mode')
    def promote_to_instant(self, request, queryset):
        queryset.update(default_mode='instant')
    
    @admin.action(description='Demote to Review Mode')
    def demote_to_review(self, request, queryset):
        queryset.update(default_mode='review')


@admin.register(DecisionTreeNode)
class DecisionTreeNodeAdmin(admin.ModelAdmin):
    """Admin for decision tree nodes."""
    
    list_display = [
        'id', 'question_preview', 'parent_node', 'answer_value',
        'result_affidavit_type', 'order', 'is_active'
    ]
    list_filter = ['is_active', 'result_affidavit_type']
    search_fields = ['question_text', 'answer_value']
    ordering = ['parent_node', 'order']
    raw_id_fields = ['parent_node', 'result_affidavit_type']
    
    def question_preview(self, obj):
        return obj.question_text[:50] + '...' if len(obj.question_text) > 50 else obj.question_text
    question_preview.short_description = 'Question'


@admin.register(Request)
class RequestAdmin(admin.ModelAdmin):
    """Admin for affidavit requests with full workflow visibility."""
    
    list_display = [
        'request_code', 'affidavit_type', 'user', 'status_badge',
        'locked_by', 'created_at', 'updated_at'
    ]
    list_filter = ['status', 'affidavit_type', 'created_at']
    search_fields = ['request_code', 'user__username', 'user__email']
    readonly_fields = [
        'request_code', 'policy_version_used', 'created_at', 
        'updated_at', 'submitted_at', 'approved_at', 'completed_at'
    ]
    raw_id_fields = ['user', 'affidavit_type', 'locked_by']
    ordering = ['-created_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('request_code', 'affidavit_type', 'user', 'status')
        }),
        ('Content', {
            'fields': ('answers_json', 'draft_text', 'final_text'),
            'classes': ('collapse',)
        }),
        ('QA & Clarification', {
            'fields': ('clarification_question', 'qa_flags_json', 'policy_version_used'),
            'classes': ('collapse',)
        }),
        ('Locking', {
            'fields': ('locked_by', 'locked_at'),
            'classes': ('collapse',)
        }),
        ('Files & Timestamps', {
            'fields': ('pdf_file', 'created_at', 'updated_at', 
                      'submitted_at', 'approved_at', 'completed_at'),
            'classes': ('collapse',)
        }),
    )
    
    def status_badge(self, obj):
        colors = {
            'draft': 'gray',
            'submitted': 'blue',
            'needs_clarification': 'orange',
            'needs_review': 'yellow',
            'approved': 'green',
            'completed': 'darkgreen',
            'rejected': 'red',
        }
        color = colors.get(obj.status, 'gray')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; '
            'border-radius: 3px;">{}</span>',
            color,
            obj.get_status_display()
        )
    status_badge.short_description = 'Status'
    
    actions = ['mark_needs_review', 'mark_approved']
    
    @admin.action(description='Mark selected as Needs Review')
    def mark_needs_review(self, request, queryset):
        queryset.update(status=Request.Status.NEEDS_REVIEW)
    
    @admin.action(description='Mark selected as Approved')
    def mark_approved(self, request, queryset):
        from django.utils import timezone
        queryset.update(status=Request.Status.APPROVED, approved_at=timezone.now())


@admin.register(Stamp)
class StampAdmin(admin.ModelAdmin):
    """Admin for commissioner stamp records."""
    
    list_display = [
        'request', 'commissioner', 'payout_amount', 'stamped_at'
    ]
    list_filter = ['stamped_at', 'commissioner']
    search_fields = ['request__request_code', 'commissioner__username']
    readonly_fields = ['stamped_at']
    raw_id_fields = ['request', 'commissioner']
    ordering = ['-stamped_at']
    date_hierarchy = 'stamped_at'


@admin.register(FrictionReport)
class FrictionReportAdmin(admin.ModelAdmin):
    """Admin for friction reports."""
    
    list_display = [
        'request', 'commissioner', 'reason_preview', 
        'is_resolved', 'created_at'
    ]
    list_filter = ['is_resolved', 'created_at', 'commissioner']
    search_fields = ['request__request_code', 'reason']
    readonly_fields = ['created_at']
    raw_id_fields = ['request', 'commissioner']
    ordering = ['-created_at']
    
    def reason_preview(self, obj):
        return obj.reason[:50] + '...' if len(obj.reason) > 50 else obj.reason
    reason_preview.short_description = 'Reason'
    
    actions = ['mark_resolved']
    
    @admin.action(description='Mark selected as Resolved')
    def mark_resolved(self, request, queryset):
        from django.utils import timezone
        queryset.update(is_resolved=True, resolved_at=timezone.now())


@admin.register(ReviewerEdit)
class ReviewerEditAdmin(admin.ModelAdmin):
    """Admin for reviewer edits (learning loop)."""
    
    list_display = [
        'request', 'reviewer', 'issue_type', 
        'ai_flag_accepted', 'created_at'
    ]
    list_filter = ['issue_type', 'ai_flag_accepted', 'created_at', 'reviewer']
    search_fields = ['request__request_code', 'issue_description']
    readonly_fields = ['created_at']
    raw_id_fields = ['request', 'reviewer']
    ordering = ['-created_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Reference', {
            'fields': ('request', 'reviewer', 'created_at')
        }),
        ('Edit Details', {
            'fields': ('original_text', 'edited_text', 'issue_type', 
                      'issue_description', 'ai_flag_accepted')
        }),
    )


@admin.register(RequestEvent)
class RequestEventAdmin(admin.ModelAdmin):
    """Admin for request audit trail."""
    
    list_display = [
        'request', 'action', 'actor', 'actor_role',
        'ip_address', 'created_at'
    ]
    list_filter = ['action', 'actor_role', 'created_at']
    search_fields = ['request__request_code', 'actor__username', 'ip_address']
    readonly_fields = ['request', 'action', 'actor', 'actor_role', 
                       'ip_address', 'user_agent', 'details', 'created_at']
    raw_id_fields = ['request', 'actor']
    ordering = ['-created_at']
    date_hierarchy = 'created_at'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(PaymentLog)
class PaymentLogAdmin(admin.ModelAdmin):
    """Admin for commissioner payment logs."""
    
    list_display = [
        'id', 'commissioner', 'amount_paid', 'stamps_count',
        'payment_method', 'payment_reference', 'paid_by', 'paid_at'
    ]
    list_filter = ['payment_method', 'paid_at', 'paid_by']
    search_fields = ['commissioner__username', 'commissioner__email', 'payment_reference']
    readonly_fields = ['paid_at']
    raw_id_fields = ['commissioner', 'paid_by']
    ordering = ['-paid_at']
    date_hierarchy = 'paid_at'
    
    fieldsets = (
        ('Payment Info', {
            'fields': ('commissioner', 'amount_paid', 'stamps_count')
        }),
        ('Payment Details', {
            'fields': ('payment_method', 'payment_reference', 'notes')
        }),
        ('Admin', {
            'fields': ('paid_by', 'paid_at')
        }),
    )

@admin.register(AIRun)
class AIRunAdmin(admin.ModelAdmin):
    """Admin for AI API call logs."""
    
    list_display = [
        'request', 'node_type', 'status', 'model_name',
        'total_tokens', 'latency_ms', 'estimated_cost_usd', 'created_at'
    ]
    list_filter = ['node_type', 'status', 'model_name', 'created_at']
    search_fields = ['request__request_code']
    readonly_fields = [
        'request', 'node_type', 'status', 'model_name', 'prompt_version',
        'prompt_tokens', 'completion_tokens', 'total_tokens',
        'latency_ms', 'input_json', 'output_json', 'raw_response',
        'error_message', 'estimated_cost_usd', 'created_at'
    ]
    raw_id_fields = ['request']
    ordering = ['-created_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('request', 'node_type', 'status', 'model_name', 'prompt_version')
        }),
        ('Token Usage', {
            'fields': ('prompt_tokens', 'completion_tokens', 'total_tokens', 'estimated_cost_usd')
        }),
        ('Performance', {
            'fields': ('latency_ms',)
        }),
        ('Data', {
            'fields': ('input_json', 'output_json', 'raw_response', 'error_message'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at',)
        }),
    )
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False
