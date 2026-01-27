"""
Affidavit Express - Data Models

This module defines all the core models for the Affidavit Express system:
- User: Extended Django user with role-based access
- AIBaseInstruction: Global AI instruction (singleton)
- AffidavitType: Types of affidavits with versioned policies
- DecisionTreeNode: "Help Me Choose" decision tree logic
- Request: Core affidavit request with workflow states
- Stamp: Commissioner completion records for payout
- FrictionReport: Commissioner-reported issues
- ReviewerEdit: Learning loop for AI improvement
"""

import random
import string
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone


class AIBaseInstruction(models.Model):
    """
    Singleton model for global AI base instruction.
    This instruction is used for ALL affidavit types as the base system prompt.
    Only one active record should exist at a time.
    """
    
    instruction_text = models.TextField(
        help_text="Global base instruction for AI drafting (used for all types)"
    )
    version = models.CharField(
        max_length=20,
        default='1.0',
        help_text="Version identifier for tracking changes"
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Only one instruction can be active at a time"
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey(
        'User',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='ai_instruction_updates'
    )
    
    class Meta:
        db_table = 'ai_base_instructions'
        verbose_name = 'AI Base Instruction'
        verbose_name_plural = 'AI Base Instructions'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"AI Instruction v{self.version} ({'Active' if self.is_active else 'Inactive'})"
    
    def save(self, *args, **kwargs):
        # Ensure only one active instruction exists
        if self.is_active:
            AIBaseInstruction.objects.filter(is_active=True).exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)
    
    @classmethod
    def get_active(cls):
        """Get the current active instruction or create default."""
        instruction = cls.objects.filter(is_active=True).first()
        if not instruction:
            instruction = cls.objects.create(
                instruction_text=cls.get_default_instruction(),
                version='1.0',
                is_active=True
            )
        return instruction
    
    @staticmethod
    def get_default_instruction():
        """Return the default base instruction for AI drafting."""
        return """You are an expert legal document drafter specializing in affidavits for Trinidad and Tobago.

Your task is to create formal, legally-sound affidavit documents based on information provided.

CRITICAL GUIDELINES:
1. Use clear, formal legal language appropriate for statutory declarations
2. Include proper affidavit structure:
   - Header (Republic of Trinidad and Tobago, Statutory Declaration Act reference)
   - Declarant introduction with full name, age, address, and ID
   - Numbered factual statements
   - Declaration of truth with legal consequences acknowledgment
   - Signature block for declarant
   - Commissioner of Affidavits attestation section
3. Be precise with dates, names, addresses, and all facts
4. Include appropriate legal declarations as per Trinidad and Tobago law
5. Format for easy reading and commissioning
6. NEVER include false or misleading statements
7. Leave signature lines and date fields blank for completion
8. Output clean HTML format suitable for PDF generation

Always maintain professional tone and absolute legal accuracy."""


class User(AbstractUser):
    """
    Extended User model with role-based access control.
    Roles determine which portal/features the user can access.
    """
    
    class Role(models.TextChoices):
        PUBLIC = 'public', 'Public User'
        COMMISSIONER = 'commissioner', 'Commissioner'
        REVIEWER = 'reviewer', 'Reviewer'
        ADMIN = 'admin', 'Admin'
    
    role = models.CharField(
        max_length=20,
        choices=Role.choices,
        default=Role.PUBLIC,
        db_index=True
    )
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    
    # Profile image (used for commissioners on landing page)
    profile_image = models.ImageField(
        upload_to='profiles/',
        blank=True,
        null=True,
        help_text="Profile photo for public display"
    )
    bio = models.TextField(
        blank=True,
        help_text="Short bio for public display (commissioners)"
    )
    
    # Organization/Firm (for commissioners)
    organization = models.CharField(
        max_length=200,
        blank=True,
        null=True,
        help_text="Organization or law firm name for commissioner"
    )
    
    # Commissioner-specific fields
    commission_number = models.CharField(max_length=50, blank=True, null=True)
    commission_expiry = models.DateField(blank=True, null=True)
    payout_rate = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        default=0.00,
        help_text="Amount paid per completed affidavit"
    )
    is_featured = models.BooleanField(
        default=False,
        help_text="Show this commissioner on the landing page"
    )
    
    # Commissioner PDF preferences
    pdf_preferences = models.JSONField(
        default=dict,
        blank=True,
        help_text="Commissioner-specific PDF formatting preferences"
    )
    # Example pdf_preferences structure:
    # {
    #   "letterhead": {"enabled": true, "text": "Commissioner Name\nAddress"},
    #   "page_size": "letter",  # or "a4"
    #   "signature_spacing": "normal",  # "compact", "normal", "expanded"
    #   "show_commission_number": true,
    #   "custom_footer": "Custom footer text"
    # }
    
    class Meta:
        db_table = 'users'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
    
    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"
    
    @property
    def is_commissioner(self):
        return self.role == self.Role.COMMISSIONER
    
    @property
    def is_reviewer(self):
        return self.role == self.Role.REVIEWER
    
    @property
    def is_admin_user(self):
        return self.role == self.Role.ADMIN
    
    def save(self, *args, **kwargs):
        """
        Override save to sync role with Django's is_staff/is_superuser.
        
        Priority:
        1. If is_superuser=True (e.g., from createsuperuser), force role to Admin
        2. Otherwise, sync is_staff/is_superuser based on role
        """
        # Handle Django's createsuperuser command - it sets is_superuser before save
        if self.is_superuser:
            self.role = self.Role.ADMIN
            self.is_staff = True
        # Sync permissions based on role
        elif self.role == self.Role.ADMIN:
            self.is_superuser = True
            self.is_staff = True
        elif self.role == self.Role.REVIEWER:
            self.is_superuser = False
            self.is_staff = True
        else:
            self.is_superuser = False
            self.is_staff = False
        
        super().save(*args, **kwargs)


class AffidavitType(models.Model):
    """
    Defines types of affidavits with tiered policies and versioned configurations.
    """
    
    class Tier(models.TextChoices):
        LOW = 'low', 'Low Variability'
        MEDIUM = 'medium', 'Medium Variability'
        HIGH_PRECISION = 'high_precision', 'High Precision Required'
        SENSITIVE = 'sensitive', 'Sensitive/High-Risk'
    
    class DefaultMode(models.TextChoices):
        INSTANT = 'instant', 'Instant Draft'
        REVIEW_FIRST = 'review_first', 'Review First'
        INTAKE_ONLY = 'intake_only', 'Intake Only'
    
    class ConfidenceStatus(models.TextChoices):
        LEARNING = 'learning', 'Learning'
        CONTROLLED = 'controlled', 'Controlled'
        CONFIDENT = 'confident', 'Confident'
    
    # Basic info
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    
    # Tier determines volume thresholds
    tier = models.CharField(
        max_length=20,
        choices=Tier.choices,
        default=Tier.MEDIUM
    )
    
    # Visibility & behavior
    enabled_on_homepage = models.BooleanField(
        default=False,
        help_text="Show prominently on homepage"
    )
    default_mode = models.CharField(
        max_length=20,
        choices=DefaultMode.choices,
        default=DefaultMode.REVIEW_FIRST
    )
    
    # Current confidence status (derived from dashboard metrics)
    confidence_status = models.CharField(
        max_length=20,
        choices=ConfidenceStatus.choices,
        default=ConfidenceStatus.LEARNING
    )
    
    # Versioned configurations
    policy_json = models.JSONField(
        default=dict,
        help_text="Type-specific rules for drafting + QA"
    )
    policy_version = models.PositiveIntegerField(default=1)
    
    prompt_pack_version = models.PositiveIntegerField(
        default=1,
        help_text="Version of the prompt pack (system prompt + examples)"
    )
    template_version = models.PositiveIntegerField(
        default=1,
        help_text="Version of the PDF template"
    )
    
    # Intake schema
    intake_schema = models.JSONField(
        default=list,
        help_text="JSON schema defining required intake form fields"
    )
    
    # Scenario library for this type
    scenario_library = models.JSONField(
        default=list,
        help_text="Known scenario patterns with tags"
    )
    
    # Template HTML (extracted from uploaded documents)
    template_html = models.TextField(
        blank=True,
        default='',
        help_text="HTML template for AI drafting, extracted from uploaded examples"
    )
    
    # Uploaded template documents for reference/learning
    template_documents = models.JSONField(
        default=list,
        help_text="List of uploaded template documents [{filename, html_content, uploaded_at}]"
    )
    # Example structure:
    # [
    #   {
    #     "filename": "example_affidavit_1.docx",
    #     "html_content": "<html>...</html>",
    #     "uploaded_at": "2026-01-24T10:00:00Z",
    #     "file_type": "docx"
    #   }
    # ]
    
    # Disallowed phrases (admin-editable list)
    disallowed_phrases = models.JSONField(
        default=list,
        help_text="List of phrases AI should never use in drafts"
    )
    # Example: ["I think", "maybe", "approximately", "I believe"]
    
    # Legacy field for backward compatibility
    is_instant_mode = models.BooleanField(
        default=False,
        help_text="If True, approved requests skip reviewer queue"
    )
    
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'affidavit_types'
        verbose_name = 'Affidavit Type'
        verbose_name_plural = 'Affidavit Types'
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} (v{self.policy_version})"
    
    @property
    def min_volume_threshold(self):
        """Get volume threshold based on tier."""
        thresholds = {
            self.Tier.LOW: 30,
            self.Tier.MEDIUM: 50,
            self.Tier.HIGH_PRECISION: 70,
            self.Tier.SENSITIVE: 90,
        }
        return thresholds.get(self.tier, 50)
    
    def increment_policy_version(self):
        self.policy_version += 1
        self.save(update_fields=['policy_version', 'updated_at'])
    
    def increment_prompt_version(self):
        self.prompt_pack_version += 1
        self.save(update_fields=['prompt_pack_version', 'updated_at'])
    
    def increment_template_version(self):
        self.template_version += 1
        self.save(update_fields=['template_version', 'updated_at'])


class DecisionTreeNode(models.Model):
    """
    Decision tree for "Help Me Choose" feature.
    Each node is a question with possible answers leading to child nodes or results.
    """
    
    question_text = models.TextField(
        blank=True,
        default='',
        help_text="The simple, non-legal question to ask the user (required for question nodes, optional for result nodes)"
    )
    
    # Help text shown below the question
    help_text = models.TextField(
        blank=True,
        null=True,
        help_text="Optional explanation shown to users below the question"
    )
    
    # Parent node (null for root nodes)
    parent_node = models.ForeignKey(
        'self',
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='children'
    )
    
    # The answer value that leads to this node from the parent
    answer_value = models.CharField(
        max_length=200,
        blank=True,
        null=True,
        help_text="The answer choice from parent that leads here"
    )
    
    # If this is a leaf node, link to the resulting affidavit type
    result_affidavit_type = models.ForeignKey(
        AffidavitType,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='decision_nodes'
    )
    
    # Ordering for display
    order = models.PositiveIntegerField(default=0)
    
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'decision_tree_nodes'
        verbose_name = 'Decision Tree Node'
        verbose_name_plural = 'Decision Tree Nodes'
        ordering = ['order']
    
    def __str__(self):
        if self.result_affidavit_type:
            return f"Result: {self.result_affidavit_type.name}"
        return f"Q: {self.question_text[:50]}..."
    
    @property
    def is_leaf(self):
        """Check if this is a leaf node (has a result)."""
        return self.result_affidavit_type is not None
    
    def get_answer_choices(self):
        """Get all possible answer choices (child nodes)."""
        return self.children.filter(is_active=True).order_by('order')


class Request(models.Model):
    """
    Core model for affidavit requests.
    Tracks the full lifecycle from intake to completion with learning loop data.
    """
    
    class Status(models.TextChoices):
        DRAFT = 'draft', 'Draft (Intake in Progress)'
        SUBMITTED = 'submitted', 'Submitted (Awaiting AI Draft)'
        DRAFT_READY = 'draft_ready', 'Draft Ready'
        NEEDS_CLARIFICATION = 'needs_clarification', 'Needs Clarification'
        NEEDS_REVIEW = 'needs_review', 'Needs Human Review'
        APPROVED = 'approved', 'Approved (Ready for Commissioner)'
        COMPLETED = 'completed', 'Completed (Stamped)'
        REJECTED = 'rejected', 'Rejected'
    
    # Unique request code (e.g., AFF-XXXX-Y)
    request_code = models.CharField(
        max_length=20,
        unique=True,
        db_index=True,
        editable=False
    )
    
    # Relationships
    affidavit_type = models.ForeignKey(
        AffidavitType,
        on_delete=models.PROTECT,
        related_name='requests'
    )
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='requests'
    )
    
    # Assigned commissioner (selected by user before download)
    commissioner = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='assigned_requests',
        limit_choices_to={'role': 'commissioner'}
    )
    
    # Intake data (auto-saved during form filling)
    answers_json = models.JSONField(
        default=dict,
        help_text="User's answers to intake form questions"
    )
    
    # AI-generated content
    draft_json = models.JSONField(
        default=dict,
        help_text="Structured draft with clause selections"
    )
    draft_text = models.TextField(
        blank=True,
        help_text="AI-generated draft of the affidavit"
    )
    final_text = models.TextField(
        blank=True,
        help_text="Final approved text (may be edited by reviewer)"
    )
    
    # Status and workflow
    status = models.CharField(
        max_length=30,
        choices=Status.choices,
        default=Status.DRAFT,
        db_index=True
    )
    
    # QA results
    clarification_question = models.TextField(
        blank=True,
        default='',
        help_text="Specific question to ask user if needs_clarification"
    )
    qa_flags_json = models.JSONField(
        default=list,
        help_text="List of QA issues flagged by AI"
    )
    qa_passed = models.BooleanField(
        default=False,
        help_text="True if QA check passed without issues"
    )
    
    # === Scenario tracking ===
    scenario_tags = models.JSONField(
        default=list,
        help_text="Detected scenario patterns for this request"
    )
    new_scenario_flag = models.BooleanField(
        default=False,
        help_text="True if this doesn't match known scenarios"
    )
    
    # === Override tracking (for learning loop) ===
    qa_overridden = models.BooleanField(
        default=False,
        help_text="Human changed the QA routing decision"
    )
    draft_edited_significantly = models.BooleanField(
        default=False,
        help_text="Reviewer changed wording beyond minor formatting"
    )
    override_notes = models.TextField(
        blank=True,
        help_text="Why the override was made"
    )
    
    # === Version snapshot (for learning loop) ===
    policy_version_used = models.PositiveIntegerField(default=1)
    prompt_version_used = models.PositiveIntegerField(default=1)
    template_version_used = models.PositiveIntegerField(default=1)
    
    # User behavior tracking
    user_edits_json = models.JSONField(
        default=list,
        help_text="Track which questions user changed/backtracked"
    )
    time_to_complete_seconds = models.PositiveIntegerField(
        null=True, 
        blank=True,
        help_text="Total time from start to approval"
    )
    
    # Locking for commissioner access
    locked_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='locked_requests'
    )
    locked_at = models.DateTimeField(null=True, blank=True)
    
    # PDF storage
    pdf_url = models.URLField(blank=True)
    pdf_file = models.FileField(
        upload_to='affidavits/pdfs/',
        blank=True,
        null=True
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    submitted_at = models.DateTimeField(null=True, blank=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'requests'
        verbose_name = 'Request'
        verbose_name_plural = 'Requests'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['request_code']),
        ]
    
    def __str__(self):
        return f"{self.request_code} - {self.affidavit_type.name}"
    
    def save(self, *args, **kwargs):
        if not self.request_code:
            self.request_code = self.generate_request_code()
        
        # Snapshot versions on first save
        if not self.pk:
            self.policy_version_used = self.affidavit_type.policy_version
            self.prompt_version_used = self.affidavit_type.prompt_pack_version
            self.template_version_used = self.affidavit_type.template_version
        
        super().save(*args, **kwargs)
    
    @staticmethod
    def generate_request_code():
        """
        Generate a unique, non-sequential request code.
        Format: AFF-XXXX-Y where XXXX is 4 random alphanumeric and Y is a check character.
        """
        while True:
            # Generate 4 random alphanumeric characters
            chars = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            # Generate 1 check character
            check = random.choice(string.ascii_uppercase)
            code = f"AFF-{chars}-{check}"
            
            # Ensure uniqueness
            if not Request.objects.filter(request_code=code).exists():
                return code
    
    def is_lock_expired(self, timeout_minutes=15):
        """Check if the current lock has expired."""
        if not self.locked_at:
            return True
        expiry_time = self.locked_at + timezone.timedelta(minutes=timeout_minutes)
        return timezone.now() > expiry_time
    
    def acquire_lock(self, user):
        """Attempt to acquire lock for a commissioner."""
        if self.locked_by and not self.is_lock_expired():
            if self.locked_by != user:
                return False, self.locked_by
        
        self.locked_by = user
        self.locked_at = timezone.now()
        self.save(update_fields=['locked_by', 'locked_at'])
        return True, None
    
    def release_lock(self):
        """Release the lock on this request."""
        self.locked_by = None
        self.locked_at = None
        self.save(update_fields=['locked_by', 'locked_at'])
    
    def force_takeover(self, user):
        """Force takeover of lock by another commissioner."""
        previous_holder = self.locked_by
        self.locked_by = user
        self.locked_at = timezone.now()
        self.save(update_fields=['locked_by', 'locked_at'])
        return previous_holder


class Stamp(models.Model):
    """
    Records when a commissioner stamps/completes an affidavit.
    Used for payout calculations.
    """
    
    request = models.OneToOneField(
        Request,
        on_delete=models.CASCADE,
        related_name='stamp'
    )
    commissioner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='stamps',
        limit_choices_to={'role': User.Role.COMMISSIONER}
    )
    payout_amount = models.DecimalField(
        max_digits=10,
        decimal_places=2
    )
    stamped_at = models.DateTimeField(auto_now_add=True)
    
    # Optional notes
    notes = models.TextField(blank=True)
    
    class Meta:
        db_table = 'stamps'
        verbose_name = 'Stamp'
        verbose_name_plural = 'Stamps'
        ordering = ['-stamped_at']
    
    def __str__(self):
        return f"Stamp: {self.request.request_code} by {self.commissioner.username}"


class FrictionReport(models.Model):
    """
    Commissioner-reported issues with affidavit documents.
    Logs when a commissioner refuses to sign due to errors.
    """
    
    request = models.ForeignKey(
        Request,
        on_delete=models.CASCADE,
        related_name='friction_reports'
    )
    commissioner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='friction_reports',
        limit_choices_to={'role': User.Role.COMMISSIONER}
    )
    reason = models.TextField(
        help_text="Reason for refusal to sign"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Flag for tracking resolution
    is_resolved = models.BooleanField(default=False)
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolution_notes = models.TextField(blank=True)
    
    class Meta:
        db_table = 'friction_reports'
        verbose_name = 'Friction Report'
        verbose_name_plural = 'Friction Reports'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Friction: {self.request.request_code} - {self.reason[:50]}"


class ReviewerEdit(models.Model):
    """
    Tracks edits made by reviewers to AI-generated drafts.
    Used for the learning loop to improve AI prompts.
    """
    
    class IssueType(models.TextChoices):
        GRAMMAR = 'grammar', 'Grammar/Spelling'
        LEGAL_ERROR = 'legal_error', 'Legal Error'
        MISSING_INFO = 'missing_info', 'Missing Information'
        CONTRADICTION = 'contradiction', 'Contradiction'
        FORMATTING = 'formatting', 'Formatting Issue'
        INAPPROPRIATE = 'inappropriate', 'Inappropriate Content'
        OTHER = 'other', 'Other'
    
    request = models.ForeignKey(
        Request,
        on_delete=models.CASCADE,
        related_name='reviewer_edits'
    )
    reviewer = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='edits_made',
        limit_choices_to={'role': User.Role.REVIEWER}
    )
    
    # The original AI-generated text
    original_text = models.TextField()
    # The reviewer-corrected text
    edited_text = models.TextField()
    
    # Classification of the issue
    issue_type = models.CharField(
        max_length=30,
        choices=IssueType.choices,
        default=IssueType.OTHER
    )
    issue_description = models.TextField(
        blank=True,
        help_text="Optional detailed description of the issue"
    )
    
    # Whether the AI flag was correct
    ai_flag_accepted = models.BooleanField(
        default=True,
        help_text="True if reviewer confirmed AI's flag was valid"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'reviewer_edits'
        verbose_name = 'Reviewer Edit'
        verbose_name_plural = 'Reviewer Edits'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Edit: {self.request.request_code} - {self.get_issue_type_display()}"
    
    def get_diff(self):
        """
        Return a simple diff between original and edited text.
        For detailed diffs, use difflib in the service layer.
        """
        return {
            'original': self.original_text,
            'edited': self.edited_text,
            'issue_type': self.issue_type
        }


class RequestEvent(models.Model):
    """
    Audit trail for request actions.
    Logs every significant action on a request for compliance.
    """
    
    class Action(models.TextChoices):
        CREATED = 'created', 'Created'
        SUBMITTED = 'submitted', 'Submitted'
        DRAFT_GENERATED = 'draft_generated', 'Draft Generated'
        QA_COMPLETED = 'qa_completed', 'QA Completed'
        VIEWED = 'viewed', 'Viewed'
        EDITED = 'edited', 'Edited'
        APPROVED = 'approved', 'Approved'
        REJECTED = 'rejected', 'Rejected'
        COMPLETED = 'completed', 'Completed (Stamped)'
        LOCKED = 'locked', 'Locked'
        UNLOCKED = 'unlocked', 'Unlocked'
        PDF_GENERATED = 'pdf_generated', 'PDF Generated'
        PDF_DOWNLOADED = 'pdf_downloaded', 'PDF Downloaded'
        COMMISSIONER_OPENED = 'commissioner_opened', 'Commissioner Opened'
        COMMISSIONER_CHANGED = 'commissioner_changed', 'Commissioner Changed'
    
    request = models.ForeignKey(
        Request,
        on_delete=models.CASCADE,
        related_name='events'
    )
    action = models.CharField(
        max_length=30,
        choices=Action.choices,
        db_index=True
    )
    actor = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='request_events'
    )
    actor_role = models.CharField(max_length=20, blank=True)
    
    # Request metadata
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    
    # Additional context
    details = models.JSONField(
        default=dict,
        help_text="Additional action-specific details"
    )
    
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        db_table = 'request_events'
        verbose_name = 'Request Event'
        verbose_name_plural = 'Request Events'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['request', 'action']),
            models.Index(fields=['actor', 'created_at']),
        ]
    
    def __str__(self):
        actor_name = self.actor.username if self.actor else 'System'
        return f"{self.request.request_code}: {self.get_action_display()} by {actor_name}"


class AIRun(models.Model):
    """
    Logs AI API calls for monitoring, debugging, and learning loop.
    Tracks tokens, latency, and raw outputs.
    """
    
    class NodeType(models.TextChoices):
        DRAFT = 'draft', 'Draft Node'
        QA = 'qa', 'QA Node'
        CLARIFICATION = 'clarification', 'Clarification Node'
    
    class Status(models.TextChoices):
        SUCCESS = 'success', 'Success'
        FAILED = 'failed', 'Failed'
        TIMEOUT = 'timeout', 'Timeout'
    
    request = models.ForeignKey(
        Request,
        on_delete=models.CASCADE,
        related_name='ai_runs'
    )
    node_type = models.CharField(
        max_length=20,
        choices=NodeType.choices
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.SUCCESS
    )
    
    # Model info
    model_name = models.CharField(max_length=50)
    prompt_version = models.PositiveIntegerField(default=1)
    
    # Token usage
    prompt_tokens = models.PositiveIntegerField(default=0)
    completion_tokens = models.PositiveIntegerField(default=0)
    total_tokens = models.PositiveIntegerField(default=0)
    
    # Performance
    latency_ms = models.PositiveIntegerField(
        default=0,
        help_text="Response time in milliseconds"
    )
    
    # Raw data for debugging
    input_json = models.JSONField(
        default=dict,
        help_text="Input sent to the AI"
    )
    output_json = models.JSONField(
        default=dict,
        help_text="Parsed output from AI"
    )
    raw_response = models.TextField(
        blank=True,
        default='',
        help_text="Raw response text from AI"
    )
    error_message = models.TextField(blank=True, default='')
    
    # Cost tracking
    estimated_cost_usd = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        default=0
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'ai_runs'
        verbose_name = 'AI Run'
        verbose_name_plural = 'AI Runs'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.request.request_code}: {self.get_node_type_display()} ({self.status})"
    
    def calculate_cost(self):
        """
        Calculate estimated cost based on token usage.
        Prices as of 2024 (update as needed).
        """
        # GPT-4o pricing (per 1M tokens)
        prices = {
            'gpt-4o': {'input': 2.50, 'output': 10.00},
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
        }
        
        model_prices = prices.get(self.model_name, prices['gpt-4o-mini'])
        input_cost = (self.prompt_tokens / 1_000_000) * model_prices['input']
        output_cost = (self.completion_tokens / 1_000_000) * model_prices['output']
        
        self.estimated_cost_usd = input_cost + output_cost
        return self.estimated_cost_usd
