"""
Affidavit Express - Serializers

DRF serializers for all models with validation and nested representations.
"""

import os

from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.utils import timezone
from .models import (
    User,
    AffidavitType,
    DecisionTreeNode,
    Request,
    ReviewerEdit,
    CommissionerSlot,
    ReviewerFeedback,
    SubmitFeedback,
    RequestEvent,
    AIRun,
    AIBaseInstruction,
    PaymentLog,
    SiteSettings,
    Ticket,
    TicketMessage,
    TicketAttachment,
    Stamp,
    FrictionReport,
)

User = get_user_model()


# =============================================================================
# User Serializers
# =============================================================================

class UserSerializer(serializers.ModelSerializer):
    """User serializer for profile updates - includes all user fields."""
    
    full_name = serializers.SerializerMethodField()
    profile_image_url = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'first_name', 'last_name', 
            'full_name', 'role', 'phone_number', 'is_superuser',
            # Commissioner-specific fields
            'commission_number', 'commission_expiry', 'organization',
            'bio', 'address', 'availability',
            'profile_image', 'profile_image_url',
            # Bank/Payment details
            'bank_name', 'bank_branch', 'bank_account_number',
            'bank_account_name', 'payment_preference',
            # PDF Preferences
            'pdf_preferences',
            # Appointment preferences
            'auto_accept_appointments',
        ]
        read_only_fields = ['id', 'role', 'profile_image_url']
    
    def get_full_name(self, obj):
        return obj.get_full_name() or obj.username
    
    def get_profile_image_url(self, obj):
        if obj.profile_image:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.profile_image.url)
            return obj.profile_image.url
        return None


class UserRegistrationSerializer(serializers.ModelSerializer):
    """Serializer for user registration."""
    
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = [
            'email', 'password', 'password_confirm',
            'first_name', 'last_name', 'phone_number'
        ]
    
    def validate(self, attrs):
        if attrs['password'] != attrs.pop('password_confirm'):
            raise serializers.ValidationError({
                'password_confirm': 'Passwords do not match.'
            })
        
        # Only block if an ACTIVE (verified) user already holds this email.
        # Inactive (unverified) accounts are reused by the view so users can retry OTP.
        email = attrs.get('email')
        if email and User.objects.filter(email__iexact=email, is_active=True).exists():
            raise serializers.ValidationError({
                'email': 'A user with this email already exists.'
            })
        
        return attrs
    
    def create(self, validated_data):
        # Auto-generate username from email
        email = validated_data['email']
        base_username = email.split('@')[0]
        username = base_username
        
        # Ensure unique username
        counter = 1
        while User.objects.filter(username=username).exists():
            username = f"{base_username}{counter}"
            counter += 1
        
        user = User.objects.create_user(
            username=username,
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', ''),
            phone_number=validated_data.get('phone_number', ''),
            role=User.Role.PUBLIC
        )
        return user


def _parse_expiry_date(value):
    """
    Parse a commission expiry date accepting both:
      - DD-MMM-YYYY (e.g. 15-Sep-2026)   ← user-facing format
      - YYYY-MM-DD  (e.g. 2026-09-15)    ← native date-picker fallback
    Returns a Python date object or None if blank/null.
    """
    import re
    from datetime import date, datetime

    if value is None or value == '':
        return None
    if hasattr(value, 'year'):
        return value  # already a date object

    value_str = str(value).strip()

    MONTH_MAP = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    }

    # Try DD-MMM-YYYY
    m = re.match(r'^(\d{1,2})-([A-Za-z]{3})-(\d{4})$', value_str)
    if m:
        day = int(m.group(1))
        month = MONTH_MAP.get(m.group(2).lower())
        year = int(m.group(3))
        if month is None:
            raise serializers.ValidationError(
                'Invalid month abbreviation. Use DD-MMM-YYYY format (e.g. 15-Sep-2026).'
            )
        try:
            return date(year, month, day)
        except ValueError:
            raise serializers.ValidationError(
                'Invalid date value. Use DD-MMM-YYYY format (e.g. 15-Sep-2026).'
            )

    # Fallback: try YYYY-MM-DD
    m2 = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', value_str)
    if m2:
        try:
            return datetime.strptime(value_str, '%Y-%m-%d').date()
        except ValueError:
            pass

    raise serializers.ValidationError(
        'Commission expiry date must be in DD-MMM-YYYY format (e.g. 15-Sep-2026).'
    )


class CommissionerRegistrationSerializer(serializers.ModelSerializer):
    """Serializer for commissioner self-registration with all required details."""

    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)
    profile_image = serializers.ImageField(required=False, allow_null=True)
    availability = serializers.JSONField(required=False, default=dict)
    # Accept DD-MMM-YYYY (e.g. 15-Sep-2026) OR YYYY-MM-DD (native date picker fallback)
    commission_expiry = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)

    class Meta:
        model = User
        fields = [
            'email', 'password', 'password_confirm',
            'first_name', 'last_name', 'phone_number',
            'profile_image', 'bio', 'organization', 'address',
            'commission_number', 'commission_expiry',
            'availability',
            # Bank/Payment details
            'bank_name', 'bank_branch', 'bank_account_number',
            'bank_account_name', 'payment_preference', 'payout_rate',
        ]

    def validate_commission_expiry(self, value):
        """Accept DD-MMM-YYYY or YYYY-MM-DD formats for commission expiry date."""
        return _parse_expiry_date(value)

    def validate(self, attrs):
        if attrs['password'] != attrs.pop('password_confirm'):
            raise serializers.ValidationError({
                'password_confirm': 'Passwords do not match.'
            })

        # Check for unique email (case-insensitive)
        # Allow re-registration only if the existing account is inactive AND still has an OTP
        # (i.e. never verified). Once OTP is verified, the slot is claimed.
        email = attrs.get('email')
        if email:
            existing = User.objects.filter(email__iexact=email).first()
            if existing:
                if existing.is_active:
                    raise serializers.ValidationError({
                        'email': 'A user with this email already exists.'
                    })
                elif not existing.otp_code:
                    # Inactive but OTP already verified — awaiting admin approval
                    raise serializers.ValidationError({
                        'email': 'Your application has already been submitted and is awaiting admin approval.'
                    })
                # else: inactive + has otp_code → unverified, view will reuse the record

        # Check for unique commission number
        commission_number = attrs.get('commission_number')
        if commission_number and User.objects.filter(commission_number=commission_number).exists():
            raise serializers.ValidationError({
                'commission_number': 'This commission number is already registered.'
            })

        # Check for unique bank account number
        bank_account_number = attrs.get('bank_account_number')
        if bank_account_number and User.objects.filter(bank_account_number=bank_account_number).exists():
            raise serializers.ValidationError({
                'bank_account_number': 'This bank account number is already registered.'
            })

        # Validate availability JSON structure if provided
        availability = attrs.get('availability', {})
        if availability:
            if not isinstance(availability, dict):
                raise serializers.ValidationError({
                    'availability': 'Availability must be a JSON object.'
                })

            # Validate recurring schedule if present
            recurring = availability.get('recurring', {})
            valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            for day, slots in recurring.items():
                if day not in valid_days:
                    raise serializers.ValidationError({
                        'availability': f'Invalid day: {day}. Must be one of {valid_days}'
                    })
                if not isinstance(slots, list):
                    raise serializers.ValidationError({
                        'availability': f'Slots for {day} must be a list of time ranges.'
                    })
                for slot in slots:
                    if not isinstance(slot, dict) or 'start' not in slot or 'end' not in slot:
                        raise serializers.ValidationError({
                            'availability': f'Each slot must have "start" and "end" times.'
                        })

        return attrs

    def create(self, validated_data):
        # Auto-generate username from email
        email = validated_data['email']
        base_username = email.split('@')[0]
        username = base_username

        # Ensure unique username
        counter = 1
        while User.objects.filter(username=username).exists():
            username = f"{base_username}{counter}"
            counter += 1

        # Extract profile image separately (handled by DRF's file upload)
        profile_image = validated_data.pop('profile_image', None)

        user = User.objects.create_user(
            username=username,
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', ''),
            phone_number=validated_data.get('phone_number', ''),
            role=User.Role.COMMISSIONER,
            bio=validated_data.get('bio', ''),
            organization=validated_data.get('organization', ''),
            address=validated_data.get('address', ''),
            commission_number=validated_data.get('commission_number', ''),
            commission_expiry=validated_data.get('commission_expiry'),
            availability=validated_data.get('availability', {}),
            # Bank/Payment details
            bank_name=validated_data.get('bank_name', ''),
            bank_branch=validated_data.get('bank_branch', ''),
            bank_account_number=validated_data.get('bank_account_number', ''),
            bank_account_name=validated_data.get('bank_account_name', ''),
            payment_preference=validated_data.get('payment_preference', 'bank_transfer'),
            is_featured=False,  # Admin must approve to feature
        )

        # Set profile image if provided
        if profile_image:
            user.profile_image = profile_image
            user.save()

        return user


class CommissionerSerializer(serializers.ModelSerializer):
    """Serializer for commissioner details including PDF preferences and bank details."""
    
    full_name = serializers.SerializerMethodField()
    profile_image_url = serializers.SerializerMethodField()
    is_email_verified = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'first_name', 'last_name', 'full_name',
            'phone_number', 'commission_number', 'commission_expiry', 'payout_rate', 
            'organization', 'address', 'bio', 'availability',
            'pdf_preferences', 'profile_image', 'profile_image_url', 'is_featured',
            'is_active', 'is_email_verified',
            # Bank/Payment details
            'bank_name', 'bank_branch', 'bank_account_number',
            'bank_account_name', 'payment_preference',
        ]
        read_only_fields = ['id', 'profile_image_url', 'is_email_verified']
    
    def get_is_email_verified(self, obj):
        """OTP verified = otp_code is empty/blank. Still has otp_code = not yet verified."""
        return not bool(obj.otp_code)
    
    def get_full_name(self, obj):
        return obj.get_full_name() or obj.username
    
    def get_profile_image_url(self, obj):
        if obj.profile_image:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.profile_image.url)
            return obj.profile_image.url
        return None


class CommissionerPublicSerializer(serializers.ModelSerializer):
    """Public serializer for commissioners shown on landing page."""
    
    full_name = serializers.SerializerMethodField()
    profile_image_url = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = [
            'id', 'first_name', 'last_name', 'full_name',
            'commission_number', 'profile_image_url', 'bio', 'organization',
            'availability', 'address',
        ]
    
    def get_full_name(self, obj):
        return obj.get_full_name() or obj.username
    
    def get_profile_image_url(self, obj):
        if obj.profile_image:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.profile_image.url)
            return obj.profile_image.url
        return None


class ReviewerSerializer(serializers.ModelSerializer):
    """Serializer for reviewer details (admin use only)."""
    
    full_name = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'first_name', 'last_name', 
            'full_name', 'is_active', 'date_joined'
        ]
        read_only_fields = ['id', 'date_joined']
    
    def get_full_name(self, obj):
        return obj.get_full_name() or obj.username


class CreateStaffUserSerializer(serializers.ModelSerializer):
    """Serializer for admin to create commissioners or reviewers."""
    
    password = serializers.CharField(write_only=True, min_length=8)
    role = serializers.ChoiceField(choices=['commissioner', 'reviewer'])
    # Accept DD-MMM-YYYY (e.g. 15-Sep-2026) OR YYYY-MM-DD (native date picker fallback)
    commission_expiry = serializers.CharField(required=False, allow_blank=True, allow_null=True)

    def validate_commission_expiry(self, value):
        return _parse_expiry_date(value)
    
    class Meta:
        model = User
        fields = [
            'username', 'email', 'password', 'first_name', 'last_name',
            'phone_number', 'role', 'profile_image', 'bio', 'organization',
            # Commissioner-specific fields
            'commission_number', 'commission_expiry', 'payout_rate', 'is_featured'
        ]
    
    def validate(self, attrs):
        role = attrs.get('role')
        if role == 'commissioner':
            if not attrs.get('commission_number'):
                raise serializers.ValidationError({
                    'commission_number': 'Commission number is required for commissioners.'
                })
        return attrs
    
    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        return user


class UpdateStaffUserSerializer(serializers.ModelSerializer):
    """Serializer for admin to update commissioners or reviewers."""
    
    password = serializers.CharField(write_only=True, min_length=8, required=False)
    # Accept DD-MMM-YYYY (e.g. 15-Sep-2026) OR YYYY-MM-DD (native date picker fallback)
    commission_expiry = serializers.CharField(required=False, allow_blank=True, allow_null=True)

    def validate_commission_expiry(self, value):
        return _parse_expiry_date(value)
    
    class Meta:
        model = User
        fields = [
            'email', 'first_name', 'last_name', 'phone_number',
            'profile_image', 'bio', 'organization', 'is_active', 'password',
            # Commissioner-specific fields
            'commission_number', 'commission_expiry', 'payout_rate', 'is_featured'
        ]
    
    def update(self, instance, validated_data):
        password = validated_data.pop('password', None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password:
            instance.set_password(password)
        instance.save()
        return instance


class CommissionerPDFPreferencesSerializer(serializers.Serializer):
    """Serializer for updating commissioner PDF preferences."""
    
    letterhead = serializers.DictField(required=False, help_text="Letterhead config: {enabled: bool, text: str}")
    page_size = serializers.ChoiceField(choices=['letter', 'a4'], required=False, default='letter')
    signature_spacing = serializers.ChoiceField(choices=['compact', 'normal', 'expanded'], required=False, default='normal')
    show_commission_number = serializers.BooleanField(required=False, default=True)
    custom_footer = serializers.CharField(required=False, allow_blank=True, max_length=200)


class PaymentLogSerializer(serializers.ModelSerializer):
    """Serializer for payment log entries."""
    
    commissioner_name = serializers.SerializerMethodField()
    paid_by_name = serializers.SerializerMethodField()
    
    class Meta:
        model = PaymentLog
        fields = [
            'id', 'commissioner', 'commissioner_name', 'amount_paid', 
            'stamps_count', 'paid_by', 'paid_by_name', 'payment_reference',
            'payment_method', 'notes', 'paid_at'
        ]
        read_only_fields = ['id', 'paid_at', 'commissioner_name', 'paid_by_name']
    
    def get_commissioner_name(self, obj):
        return obj.commissioner.get_full_name() or obj.commissioner.username
    
    def get_paid_by_name(self, obj):
        if obj.paid_by:
            return obj.paid_by.get_full_name() or obj.paid_by.username
        return None


class CommissionerPaymentSummarySerializer(serializers.ModelSerializer):
    """Serializer for commissioner with payment summary."""
    
    full_name = serializers.SerializerMethodField()
    profile_image_url = serializers.SerializerMethodField()
    amount_to_pay = serializers.SerializerMethodField()
    unpaid_stamps_count = serializers.SerializerMethodField()
    total_earned = serializers.SerializerMethodField()
    total_paid = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'first_name', 'last_name', 'full_name',
            'phone_number', 'commission_number', 'payout_rate', 
            'organization', 'address', 'bio', 'profile_image_url', 'is_featured',
            # Bank/Payment details
            'bank_name', 'bank_branch', 'bank_account_number',
            'bank_account_name', 'payment_preference',
            # Payment summary
            'amount_to_pay', 'unpaid_stamps_count', 'total_earned', 'total_paid',
        ]
        read_only_fields = ['id', 'profile_image_url', 'amount_to_pay', 
                          'unpaid_stamps_count', 'total_earned', 'total_paid']
    
    def get_full_name(self, obj):
        return obj.get_full_name() or obj.username
    
    def get_profile_image_url(self, obj):
        if obj.profile_image:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.profile_image.url)
            return obj.profile_image.url
        return None
    
    def get_amount_to_pay(self, obj):
        """Calculate total unpaid amount from stamps."""
        from django.db.models import Sum
        unpaid = obj.stamps.filter(paid=False).aggregate(total=Sum('payout_amount'))
        return str(unpaid['total'] or 0)
    
    def get_unpaid_stamps_count(self, obj):
        """Count unpaid stamps."""
        return obj.stamps.filter(paid=False).count()
    
    def get_total_earned(self, obj):
        """Total lifetime earnings from all stamps."""
        from django.db.models import Sum
        total = obj.stamps.aggregate(total=Sum('payout_amount'))
        return str(total['total'] or 0)
    
    def get_total_paid(self, obj):
        """Total amount paid out."""
        from django.db.models import Sum
        paid = obj.payment_logs.aggregate(total=Sum('amount_paid'))
        return str(paid['total'] or 0)


class MarkAsPaidSerializer(serializers.Serializer):
    """Serializer for marking commissioner stamps as paid."""
    
    payment_reference = serializers.CharField(max_length=100, required=False, allow_blank=True)
    payment_method = serializers.CharField(max_length=50, required=False, allow_blank=True)
    notes = serializers.CharField(required=False, allow_blank=True)


# =============================================================================
# Affidavit Type Serializers
# =============================================================================

class AffidavitTypeSerializer(serializers.ModelSerializer):
    """Full affidavit type serializer including policy and tier info."""
    
    min_volume_threshold = serializers.ReadOnlyField()
    intake_schema = serializers.SerializerMethodField()
    
    class Meta:
        model = AffidavitType
        fields = [
            'id', 'name', 'description', 'policy_json',
            'tier', 'default_mode', 'confidence_status', 'enabled_on_homepage',
            'policy_version', 'prompt_pack_version', 'template_version',
            'min_volume_threshold', 'intake_schema', 'scenario_library',
            'validation_rules',
            'is_active', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'min_volume_threshold', 'created_at', 'updated_at']
    
    def get_intake_schema(self, obj):
        """
        Return intake_schema with T&T validation rules applied.
        This ensures all fields have proper validation even if saved before the feature.
        """
        from .services.policy_generator_service import upgrade_intake_schema_with_tt_validation
        
        if not obj.intake_schema:
            return []
        
        # Apply T&T validation rules to all fields
        return upgrade_intake_schema_with_tt_validation(obj.intake_schema)


class AffidavitTypeListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing affidavit types."""
    
    intake_schema_count = serializers.SerializerMethodField()
    confidence_status = serializers.SerializerMethodField()
    
    class Meta:
        model = AffidavitType
        fields = ['id', 'name', 'description', 'tier', 'default_mode', 
                  'enabled_on_homepage', 'is_active', 'intake_schema_count', 'confidence_status']
    
    def get_intake_schema_count(self, obj):
        """Return count of questions in intake schema."""
        return len(obj.intake_schema) if obj.intake_schema else 0
    
    def get_confidence_status(self, obj):
        """Return confidence status based on volume."""
        if hasattr(obj, 'volume') and obj.volume >= obj.min_volume_threshold:
            return 'confident'
        return 'learning'


class AffidavitTypePolicyUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating policy JSON (increments version)."""
    
    class Meta:
        model = AffidavitType
        fields = ['policy_json', 'prompt_pack_version', 'template_version', 'scenario_library']
    
    def update(self, instance, validated_data):
        if 'policy_json' in validated_data:
            instance.policy_json = validated_data['policy_json']
            instance.policy_version += 1
        if 'prompt_pack_version' in validated_data:
            instance.prompt_pack_version = validated_data['prompt_pack_version']
        if 'template_version' in validated_data:
            instance.template_version = validated_data['template_version']
        if 'scenario_library' in validated_data:
            instance.scenario_library = validated_data['scenario_library']
        instance.save()
        return instance


class AffidavitTypeAdminSerializer(serializers.ModelSerializer):
    """
    Full CRUD serializer for admin management of affidavit types.
    Includes all fields for creating and updating types with intake_schema.
    """
    
    min_volume_threshold = serializers.ReadOnlyField()
    questions_count = serializers.SerializerMethodField()
    template_documents_count = serializers.SerializerMethodField()
    
    class Meta:
        model = AffidavitType
        fields = [
            'id', 'name', 'description', 
            'tier', 'default_mode', 'confidence_status', 
            'enabled_on_homepage', 'is_active',
            'policy_version', 'prompt_pack_version', 'template_version',
            'min_volume_threshold', 
            'intake_schema', 'scenario_library', 'policy_json',
            'template_html', 'template_documents', 'disallowed_phrases',
            'validation_rules', 'placeholder_mapping',
            'questions_count', 'template_documents_count',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'min_volume_threshold', 'questions_count', 'template_documents_count', 'created_at', 'updated_at']
    
    def get_questions_count(self, obj):
        """Count questions in intake_schema."""
        return len(obj.intake_schema) if obj.intake_schema else 0
    
    def get_template_documents_count(self, obj):
        """Count uploaded template documents."""
        return len(obj.template_documents) if obj.template_documents else 0
    
    def validate_intake_schema(self, value):
        """Validate intake_schema structure."""
        if not isinstance(value, list):
            raise serializers.ValidationError("intake_schema must be a list")
        
        required_keys = {'id', 'type', 'label', 'required'}
        valid_types = {'text', 'textarea', 'select', 'radio', 'checkbox', 'date', 'number', 'email', 'phone'}
        
        for idx, question in enumerate(value):
            if not isinstance(question, dict):
                raise serializers.ValidationError(f"Question at index {idx} must be an object")
            
            missing = required_keys - set(question.keys())
            if missing:
                raise serializers.ValidationError(
                    f"Question at index {idx} missing required keys: {missing}"
                )
            
            if question.get('type') not in valid_types:
                raise serializers.ValidationError(
                    f"Question at index {idx} has invalid type: {question.get('type')}. "
                    f"Valid types: {valid_types}"
                )
            
            # Validate options for select/radio/checkbox types
            if question.get('type') in ('select', 'radio', 'checkbox'):
                if not question.get('options'):
                    raise serializers.ValidationError(
                        f"Question at index {idx} of type {question['type']} requires 'options'"
                    )
            
            # Validate conditional/show_if structure
            if 'show_if' in question:
                show_if = question['show_if']
                if not isinstance(show_if, dict) or 'field' not in show_if or 'value' not in show_if:
                    raise serializers.ValidationError(
                        f"Question at index {idx} has invalid 'show_if' structure. "
                        "Expected: {field: string, value: string|array}"
                    )
        
        return value

    def update(self, instance, validated_data):
        """
        Override update to auto-sync placeholder_mapping when
        intake_schema OR template_html changes.
        """
        from .services.policy_generator_service import (
            sync_mapping_after_question_change,
            auto_generate_placeholder_mapping,
        )

        new_schema = validated_data.get('intake_schema')
        new_template = validated_data.get('template_html')
        explicit_mapping = 'placeholder_mapping' in validated_data

        # Case 1: intake_schema changed → sync mappings (remove stale, add new)
        if new_schema is not None:
            old_schema = instance.intake_schema or []
            synced_mapping = sync_mapping_after_question_change(
                template_html=new_template or instance.template_html or '',
                old_schema=old_schema,
                new_schema=new_schema,
                placeholder_mapping=instance.placeholder_mapping or {},
            )
            if not explicit_mapping:
                validated_data['placeholder_mapping'] = synced_mapping

        # Case 2: template_html changed → re-generate mapping for new placeholders
        if new_template is not None and new_template != (instance.template_html or ''):
            final_schema = new_schema if new_schema is not None else (instance.intake_schema or [])
            base_mapping = validated_data.get('placeholder_mapping', instance.placeholder_mapping or {})
            generated = auto_generate_placeholder_mapping(new_template, final_schema)
            # Merge: keep existing manual overrides, fill in new auto-mappings
            merged = dict(base_mapping)
            for ph, qid in generated.items():
                if ph not in merged:
                    merged[ph] = qid
            if not explicit_mapping:
                validated_data['placeholder_mapping'] = merged

        return super().update(instance, validated_data)


# =============================================================================
# Decision Tree Serializers
# =============================================================================

class DecisionTreeNodeSerializer(serializers.ModelSerializer):
    """Serializer for decision tree nodes with nested children."""
    
    children = serializers.SerializerMethodField()
    result_affidavit_type = AffidavitTypeListSerializer(read_only=True)
    
    class Meta:
        model = DecisionTreeNode
        fields = [
            'id', 'question_text', 'answer_value', 'result_affidavit_type',
            'is_leaf', 'order', 'children'
        ]
    
    def get_children(self, obj):
        children = obj.get_answer_choices()
        return DecisionTreeNodeChildSerializer(children, many=True).data


class DecisionTreeNodeChildSerializer(serializers.ModelSerializer):
    """Lightweight child node serializer (no deep nesting)."""
    
    result_affidavit_type_id = serializers.IntegerField(
        source='result_affidavit_type.id', 
        read_only=True,
        allow_null=True
    )
    result_affidavit_type_name = serializers.CharField(
        source='result_affidavit_type.name',
        read_only=True,
        allow_null=True
    )
    
    class Meta:
        model = DecisionTreeNode
        fields = [
            'id', 'question_text', 'answer_value', 'is_leaf',
            'result_affidavit_type_id', 'result_affidavit_type_name'
        ]


class DecisionTreeAnswerSerializer(serializers.Serializer):
    """Serializer for traversing the decision tree."""
    
    node_id = serializers.IntegerField()
    answer_value = serializers.CharField()


# =============================================================================
# Admin Decision Tree Serializers
# =============================================================================

class AdminDecisionTreeNodeSerializer(serializers.ModelSerializer):
    """Admin serializer for managing decision tree nodes."""
    
    parent_node_id = serializers.IntegerField(
        source='parent_node.id', 
        read_only=True, 
        allow_null=True
    )
    parent_question = serializers.CharField(
        source='parent_node.question_text',
        read_only=True,
        allow_null=True
    )
    result_affidavit_type_id = serializers.IntegerField(
        source='result_affidavit_type.id',
        read_only=True,
        allow_null=True
    )
    result_affidavit_type_name = serializers.CharField(
        source='result_affidavit_type.name',
        read_only=True,
        allow_null=True
    )
    children_count = serializers.SerializerMethodField()
    
    class Meta:
        model = DecisionTreeNode
        fields = [
            'id', 'question_text', 'help_text', 'answer_value',
            'parent_node_id', 'parent_question',
            'result_affidavit_type_id', 'result_affidavit_type_name',
            'order', 'is_active', 'is_leaf', 'children_count',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'is_leaf']
    
    def get_children_count(self, obj):
        return obj.children.filter(is_active=True).count()


class AdminDecisionTreeNodeCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating/updating decision tree nodes."""
    
    parent_node = serializers.PrimaryKeyRelatedField(
        queryset=DecisionTreeNode.objects.all(),
        required=False,
        allow_null=True
    )
    result_affidavit_type = serializers.PrimaryKeyRelatedField(
        queryset=AffidavitType.objects.all(),
        required=False,
        allow_null=True
    )
    
    class Meta:
        model = DecisionTreeNode
        fields = [
            'id', 'question_text', 'help_text', 'answer_value',
            'parent_node', 'result_affidavit_type', 'order', 'is_active'
        ]
        read_only_fields = ['id']
    
    def validate(self, data):
        # A node should either have children (be a question) or link to an affidavit type (be a result)
        # This is soft validation - we allow flexibility during editing
        result_type = data.get('result_affidavit_type')
        parent = data.get('parent_node')
        answer_value = data.get('answer_value')
        
        # If linking to an affidavit type (leaf node), should have parent and answer
        if result_type:
            if not parent:
                raise serializers.ValidationError(
                    "Result nodes must have a parent node."
                )
            if not answer_value:
                raise serializers.ValidationError(
                    "Result nodes must have an answer_value (the option label users click)."
                )
        
        return data


class AdminDecisionTreePathSerializer(serializers.Serializer):
    """Serializer showing the path to an affidavit type in the decision tree."""
    
    affidavit_type_id = serializers.IntegerField()
    affidavit_type_name = serializers.CharField()
    path = serializers.ListField(
        child=serializers.DictField()
    )


# =============================================================================
# Guest Auth Serializers
# =============================================================================

class GuestSignupStartSerializer(serializers.Serializer):
    """Serializer for starting guest signup (sending OTP)."""
    email = serializers.EmailField()
    full_name = serializers.CharField(max_length=150)
    phone_number = serializers.CharField(max_length=50, required=False, allow_blank=True)

    def validate(self, attrs):
        email = attrs.get('email')
        phone_number = (attrs.get('phone_number') or '').strip()

        allow_duplicate_phones = (os.getenv('ALLOW_DUPLICATE_PHONE_NUMBERS') or '').strip().lower() == 'true'

        errors = {}

        if email and User.objects.filter(email__iexact=email).exists():
            errors['email'] = 'An account with this email already exists. Please sign in instead.'

        if (not allow_duplicate_phones) and phone_number and User.objects.filter(phone_number=phone_number).exists():
            errors['phone_number'] = 'An account with this phone number already exists. Please sign in instead.'

        if errors:
            raise serializers.ValidationError(errors)

        attrs['phone_number'] = phone_number
        return attrs


class GuestSignupVerifySerializer(serializers.Serializer):
    """Serializer for verifying guest OTP and creating account."""
    email = serializers.EmailField()
    otp = serializers.CharField(max_length=6)
    full_name = serializers.CharField(max_length=150)
    phone_number = serializers.CharField(max_length=50, required=False, allow_blank=True)
    affidavit_type_id = serializers.IntegerField()
    answers_json = serializers.JSONField()
    draft_text = serializers.CharField(required=False, allow_blank=True)


# =============================================================================
# Request Serializers
# =============================================================================

class RequestCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a new request (public intake start)."""
    
    class Meta:
        model = Request
        fields = ['id', 'request_code', 'affidavit_type', 'answers_json', 'draft_text', 'status']
        read_only_fields = ['id', 'request_code', 'status']
    
    def create(self, validated_data):
        user = self.context['request'].user
        affidavit_type = validated_data['affidavit_type']
        draft_text = validated_data.get('draft_text', '') or ''
        has_ready_draft = len(draft_text) > 50
        
        request_obj = Request.objects.create(
            user=user,
            affidavit_type=affidavit_type,
            answers_json=validated_data.get('answers_json', {}),
            draft_text=draft_text,
            policy_version_used=affidavit_type.policy_version,
            prompt_version_used=affidavit_type.prompt_pack_version,
            template_version_used=affidavit_type.template_version,
            status=Request.Status.DRAFT_READY if has_ready_draft else Request.Status.DRAFT
        )
        return request_obj


class RequestPatchSerializer(serializers.ModelSerializer):
    """Serializer for auto-saving intake form answers (PATCH)."""
    
    commissioner_id = serializers.IntegerField(required=False, write_only=True)
    
    class Meta:
        model = Request
        fields = ['answers_json', 'draft_text', 'commissioner_id']
    
    def validate_commissioner_id(self, value):
        """Validate commissioner exists and has the right role."""
        from .models import User
        try:
            commissioner = User.objects.get(id=value, role='commissioner')
            return value
        except User.DoesNotExist:
            raise serializers.ValidationError("Invalid commissioner selected.")
    
    def validate(self, attrs):
        # Commissioner can be set in APPROVED status
        if 'commissioner_id' in attrs:
            if self.instance and self.instance.status != Request.Status.APPROVED:
                raise serializers.ValidationError(
                    "Commissioner can only be selected when request is approved."
                )
            return attrs
        
        # Only allow patching answers if request is in DRAFT or NEEDS_CLARIFICATION status
        if self.instance and self.instance.status not in [
            Request.Status.DRAFT, 
            Request.Status.NEEDS_CLARIFICATION
        ]:
            raise serializers.ValidationError(
                "Cannot modify answers after submission."
            )
        return attrs
    
    def update(self, instance, validated_data):
        # Handle commissioner_id separately
        commissioner_id = validated_data.pop('commissioner_id', None)
        if commissioner_id:
            from .models import User
            instance.commissioner = User.objects.get(id=commissioner_id)

        draft_text = validated_data.get('draft_text')
        if (
            isinstance(draft_text, str)
            and len(draft_text) > 50
            and instance.status == Request.Status.DRAFT
        ):
            instance.status = Request.Status.DRAFT_READY

        instance = super().update(instance, validated_data)

        if commissioner_id:
            instance.save(update_fields=['commissioner'])
        return instance


class RequestSubmitSerializer(serializers.Serializer):
    """Serializer for submitting a request for AI drafting."""

    # ------------------------------------------------------------------
    # Helper: mirror frontend shouldShowQuestion() so conditional fields
    # that are hidden are never flagged as "missing".
    # ------------------------------------------------------------------
    @staticmethod
    def _is_field_visible(field: dict, answers: dict, schema: list) -> bool:
        """Return True if the field should be shown given current answers."""
        show_if = field.get('show_if')
        if not show_if:
            return True

        parent_field_id = show_if.get('field', '')
        required_value = show_if.get('value')

        # Resolve parent answer — try direct key first, then scan schema
        parent_answer = answers.get(parent_field_id)
        if parent_answer is None:
            for q in schema:
                qid = q.get('id') or q.get('field_name', '')
                if qid == parent_field_id or q.get('field_name') == parent_field_id:
                    parent_answer = answers.get(qid)
                    break

        # If no required value specified, field shows when parent has *any* value
        if not required_value or required_value == '':
            return parent_answer is not None and parent_answer != '' and parent_answer is not False

        # Array answer — check intersection
        if isinstance(parent_answer, list):
            check_values = required_value if isinstance(required_value, list) else [required_value]
            return any(v in parent_answer for v in check_values)

        # required_value is array — check membership
        if isinstance(required_value, list):
            return parent_answer in required_value

        return parent_answer == required_value

    def validate(self, attrs):
        request_obj = self.instance
        if request_obj.status not in [
            Request.Status.DRAFT, 
            Request.Status.NEEDS_CLARIFICATION
        ]:
            raise serializers.ValidationError(
                "Request has already been submitted."
            )
        
        # Validate required fields based on affidavit type schema
        intake_schema = request_obj.affidavit_type.intake_schema
        answers = request_obj.answers_json
        
        missing_fields = []
        for field in intake_schema:
            # Get field identifier - use 'id' or fallback to 'field_name'
            field_id = field.get('id') or field.get('field_name')
            field_label = field.get('label', field_id)
            
            # Skip fields hidden by show_if conditions
            if not self._is_field_visible(field, answers, intake_schema):
                continue

            # Check if field is required and has a value
            if field.get('required', False):
                answer = answers.get(field_id)
                # Consider empty strings, empty arrays, and None as missing
                if answer is None or answer == '' or (isinstance(answer, list) and len(answer) == 0):
                    missing_fields.append(field_label)
        
        if missing_fields:
            raise serializers.ValidationError({
                'answers_json': f"Missing required fields: {', '.join(missing_fields)}"
            })

        # Enforce complex validation rules from the Validation tab (runs
        # pure Python — no OpenAI call, no extra cost).
        from .services.ai_service import apply_validation_rules
        rules = request_obj.affidavit_type.validation_rules or []
        if rules:
            invalid_fields, _ = apply_validation_rules(answers, rules)
            if invalid_fields:
                raise serializers.ValidationError({
                    field_id: reason
                    for field_id, reason in invalid_fields.items()
                })

        return attrs


class RequestListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing requests."""
    
    affidavit_type = AffidavitTypeListSerializer(read_only=True)
    user_name = serializers.SerializerMethodField()
    time_waiting = serializers.SerializerMethodField()
    
    class Meta:
        model = Request
        fields = [
            'id', 'request_code', 'affidavit_type', 'user_name',
            'status', 'time_waiting', 'created_at', 'updated_at'
        ]
    
    def get_user_name(self, obj):
        return obj.user.get_full_name() or obj.user.username
    
    def get_time_waiting(self, obj):
        """Calculate time waiting in human-readable format."""
        if obj.status in [Request.Status.COMPLETED, Request.Status.REJECTED]:
            return None
        
        delta = timezone.now() - obj.created_at
        hours = delta.total_seconds() / 3600
        
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)} minutes"
        elif hours < 24:
            return f"{int(hours)} hours"
        else:
            return f"{int(hours / 24)} days"


class RequestDetailSerializer(serializers.ModelSerializer):
    """Full request detail serializer with learning loop fields."""
    
    affidavit_type = AffidavitTypeListSerializer(read_only=True)
    user = UserSerializer(read_only=True)
    locked_by = UserSerializer(read_only=True)
    commissioner = serializers.SerializerMethodField()
    is_locked = serializers.SerializerMethodField()
    lock_holder_name = serializers.SerializerMethodField()
    appointment_date = serializers.DateTimeField(
        source='appointment_slot.start_time', 
        read_only=True, 
        allow_null=True
    )
    appointment_slot = serializers.SerializerMethodField()
    
    class Meta:
        model = Request
        fields = [
            'id', 'request_code', 'affidavit_type', 'user', 'commissioner',
            'answers_json', 'draft_text', 'draft_json', 'final_text', 'status',
            'clarification_question', 'qa_passed', 'qa_flags_json',
            'scenario_tags', 'new_scenario_flag',
            'qa_overridden', 'draft_edited_significantly', 'override_notes',
            'policy_version_used', 'prompt_version_used', 'template_version_used',
            'user_edits_json', 'time_to_complete_seconds', 'pdf_url',
            'locked_by', 'locked_at', 'is_locked', 'lock_holder_name',
            'pdf_file', 'is_paid', 'user_paid_at', 'created_at', 'updated_at', 'submitted_at',
            'approved_at', 'completed_at', 'appointment_date', 'appointment_slot'
        ]
        read_only_fields = [
            'id', 'request_code', 'policy_version_used', 
            'prompt_version_used', 'template_version_used',
            'created_at', 'updated_at'
        ]
    
    def get_commissioner(self, obj):
        if obj.commissioner:
            profile_image_url = None
            if obj.commissioner.profile_image:
                request = self.context.get('request')
                if request:
                    profile_image_url = request.build_absolute_uri(obj.commissioner.profile_image.url)
                else:
                    profile_image_url = obj.commissioner.profile_image.url

            return {
                'id': obj.commissioner.id,
                'first_name': obj.commissioner.first_name,
                'last_name': obj.commissioner.last_name,
                'full_name': obj.commissioner.get_full_name() or f"{obj.commissioner.first_name} {obj.commissioner.last_name}",
                'profile_image_url': profile_image_url
            }
        return None
    
    def get_is_locked(self, obj):
        if not obj.locked_by:
            return False
        return not obj.is_lock_expired()
    
    def get_lock_holder_name(self, obj):
        if obj.locked_by and not obj.is_lock_expired():
            return obj.locked_by.get_full_name() or obj.locked_by.username
        return None
    
    def get_appointment_slot(self, obj):
        if hasattr(obj, 'appointment_slot') and obj.appointment_slot:
            slot = obj.appointment_slot
            return {
                'id': slot.id,
                'commissioner': slot.commissioner_id,
                'start_time': slot.start_time.isoformat() if slot.start_time else None,
                'is_booked': slot.is_booked,
                'appointment_status': slot.appointment_status,
                'decision_at': slot.decision_at.isoformat() if slot.decision_at else None,
                'decision_reason': slot.decision_reason or '',
            }
        return None


class RequestReviewerSerializer(serializers.ModelSerializer):
    """Serializer for reviewer queue with QA flags highlighted."""
    
    affidavit_type_name = serializers.CharField(
        source='affidavit_type.name', 
        read_only=True
    )
    time_waiting = serializers.SerializerMethodField()
    risk_flags = serializers.SerializerMethodField()
    
    class Meta:
        model = Request
        fields = [
            'id', 'request_code', 'affidavit_type_name', 
            'time_waiting', 'risk_flags', 'status', 'created_at'
        ]
    
    def get_time_waiting(self, obj):
        delta = timezone.now() - obj.created_at
        hours = delta.total_seconds() / 3600
        
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)} min"
        elif hours < 24:
            return f"{int(hours)}h"
        else:
            return f"{int(hours / 24)}d"
    
    def get_risk_flags(self, obj):
        """Extract risk flag summaries from qa_flags_json."""
        flags = obj.qa_flags_json or []
        return [f.get('type', 'Unknown') for f in flags if isinstance(f, dict)]


# =============================================================================
# Stamp Serializers
# =============================================================================

class StampSerializer(serializers.ModelSerializer):
    """Serializer for stamp records."""
    
    request_code = serializers.CharField(source='request.request_code', read_only=True)
    commissioner_name = serializers.SerializerMethodField()
    
    class Meta:
        model = Stamp
        fields = [
            'id', 'request', 'request_code', 'commissioner', 
            'commissioner_name', 'payout_amount', 'stamped_at', 'notes'
        ]
        read_only_fields = ['id', 'stamped_at']
    
    def get_commissioner_name(self, obj):
        return obj.commissioner.get_full_name() or obj.commissioner.username


class MarkCompleteSerializer(serializers.Serializer):
    """Serializer for marking a request as completed by commissioner."""
    
    notes = serializers.CharField(required=False, allow_blank=True)
    final_text = serializers.CharField(required=False, allow_blank=True)
    
    def validate(self, attrs):
        request_obj = self.context.get('request_obj')
        
        if request_obj.status not in [Request.Status.APPROVED, Request.Status.DRAFT_READY]:
            raise serializers.ValidationError(
                "Only approved or draft-ready requests can be marked as completed."
            )
        
        if hasattr(request_obj, 'stamp'):
            raise serializers.ValidationError(
                "This request has already been completed."
            )
        
        return attrs


# =============================================================================
# Friction Report Serializers
# =============================================================================

class FrictionReportSerializer(serializers.ModelSerializer):
    """Serializer for friction reports."""
    
    request_code = serializers.CharField(source='request.request_code', read_only=True)
    commissioner_name = serializers.SerializerMethodField()
    
    class Meta:
        model = FrictionReport
        fields = [
            'id', 'request', 'request_code', 'commissioner', 
            'commissioner_name', 'reason', 'is_resolved', 
            'resolved_at', 'resolution_notes', 'created_at'
        ]
        read_only_fields = ['id', 'commissioner', 'created_at']
    
    def get_commissioner_name(self, obj):
        return obj.commissioner.get_full_name() or obj.commissioner.username


class FrictionReportCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a friction report."""
    
    class Meta:
        model = FrictionReport
        fields = ['request', 'reason']
    
    def create(self, validated_data):
        commissioner = self.context['request'].user
        return FrictionReport.objects.create(
            commissioner=commissioner,
            **validated_data
        )


# =============================================================================
# Reviewer Edit Serializers
# =============================================================================

class ReviewerEditSerializer(serializers.ModelSerializer):
    """Serializer for reviewer edits."""
    
    request_code = serializers.CharField(source='request.request_code', read_only=True)
    reviewer_name = serializers.SerializerMethodField()
    diff = serializers.SerializerMethodField()
    
    class Meta:
        model = ReviewerEdit
        fields = [
            'id', 'request', 'request_code', 'reviewer', 'reviewer_name',
            'original_text', 'edited_text', 'issue_type', 'issue_description',
            'ai_flag_accepted', 'diff', 'created_at'
        ]
        read_only_fields = ['id', 'reviewer', 'created_at']
    
    def get_reviewer_name(self, obj):
        return obj.reviewer.get_full_name() or obj.reviewer.username
    
    def get_diff(self, obj):
        return obj.get_diff()


class ReviewerEditCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a reviewer edit."""
    
    class Meta:
        model = ReviewerEdit
        fields = [
            'request', 'original_text', 'edited_text', 
            'issue_type', 'issue_description', 'ai_flag_accepted'
        ]
    
    def create(self, validated_data):
        reviewer = self.context['request'].user
        return ReviewerEdit.objects.create(
            reviewer=reviewer,
            **validated_data
        )


class ReviewerFeedbackSerializer(serializers.ModelSerializer):
    """Serializer for listing reviewer feedback."""

    request_code = serializers.CharField(source='request.request_code', read_only=True)
    reviewer_name = serializers.SerializerMethodField()

    class Meta:
        model = ReviewerFeedback
        fields = [
            'id', 'request', 'request_code', 'reviewer', 'reviewer_name',
            'category', 'feedback_target', 'message',
            'original_snippet', 'revised_snippet', 'summary',
            'is_active', 'times_seen', 'created_at',
        ]
        read_only_fields = ['id', 'reviewer', 'is_active', 'times_seen', 'created_at']

    def get_reviewer_name(self, obj):
        return obj.reviewer.get_full_name() or obj.reviewer.username


class ReviewerFeedbackCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating reviewer feedback (manual notes or auto-detected diffs)."""

    class Meta:
        model = ReviewerFeedback
        fields = ['request', 'category', 'message', 'feedback_target', 'original_snippet', 'revised_snippet']

    def validate(self, attrs):
        message = (attrs.get('message') or '').strip()
        orig = (attrs.get('original_snippet') or '').strip()
        rev = (attrs.get('revised_snippet') or '').strip()
        # Must have either a message or both snippets
        if not message and not (orig and rev):
            raise serializers.ValidationError(
                'Provide either a feedback message or both original_snippet and revised_snippet.'
            )
        if message and (len(message) < 10 or len(message) > 300):
            raise serializers.ValidationError('Feedback message must be 10–300 characters.')
        return attrs

    def create(self, validated_data):
        reviewer = self.context['request'].user
        request_obj = validated_data['request']

        # Compute contextual fingerprint + scenario tags
        from .services.feedback_service import build_answer_fingerprint, _compute_fingerprint_and_tags
        fingerprint, tags = _compute_fingerprint_and_tags(request_obj)

        return ReviewerFeedback.objects.create(
            reviewer=reviewer,
            affidavit_type=request_obj.affidavit_type,
            answer_fingerprint=fingerprint,
            scenario_tags=tags,
            **validated_data
        )


class SubmitFeedbackSerializer(serializers.ModelSerializer):
    """Serializer for generic admin feedback submissions."""

    user = serializers.HiddenField(default=serializers.CurrentUserDefault())

    class Meta:
        model = SubmitFeedback
        fields = ['id', 'user', 'subject', 'message', 'email', 'created_at']
        read_only_fields = ['id', 'created_at']


class ApproveRequestSerializer(serializers.Serializer):
    """Serializer for approving a request."""
    
    final_text = serializers.CharField(required=False)
    issue_type = serializers.ChoiceField(
        choices=ReviewerEdit.IssueType.choices,
        required=False
    )
    issue_description = serializers.CharField(required=False, allow_blank=True)
    feedback_entries = serializers.ListField(
        child=serializers.DictField(),
        required=False,
        default=list,
        help_text='Optional manual feedback notes: [{category, message, feedback_target}]'
    )
    auto_feedback_pairs = serializers.ListField(
        child=serializers.DictField(),
        required=False,
        default=list,
        help_text='Auto-detected diff pairs from reviewer edit: [{original_snippet, revised_snippet, feedback_target}]'
    )
    
    def validate_feedback_entries(self, value):
        valid_cats = {c[0] for c in ReviewerFeedback.Category.choices}
        valid_targets = {t[0] for t in ReviewerFeedback.FeedbackTarget.choices}
        for entry in value:
            if not entry.get('message', '').strip():
                raise serializers.ValidationError('Each feedback entry must have a message.')
            msg = entry['message'].strip()
            if len(msg) < 10 or len(msg) > 300:
                raise serializers.ValidationError('Each feedback message must be 10–300 characters.')
            cat = entry.get('category', 'other')
            if cat not in valid_cats:
                raise serializers.ValidationError(f'Invalid category: {cat}')
            target = entry.get('feedback_target', 'drafter')
            if target not in valid_targets:
                raise serializers.ValidationError(f'Invalid feedback_target: {target}')
        return value

    def validate_auto_feedback_pairs(self, value):
        valid_targets = {t[0] for t in ReviewerFeedback.FeedbackTarget.choices} | {'skip'}
        for pair in value:
            target = pair.get('feedback_target', 'drafter')
            if target not in valid_targets:
                raise serializers.ValidationError(f'Invalid feedback_target: {target}')
        return value

    def validate(self, attrs):
        request_obj = self.context.get('request_obj')
        
        if request_obj.status != Request.Status.NEEDS_REVIEW:
            raise serializers.ValidationError(
                "Only requests needing review can be approved."
            )
        
        return attrs


# =============================================================================
# Dashboard Serializers
# =============================================================================

class ConfidenceDashboardSerializer(serializers.Serializer):
    """Serializer for confidence dashboard statistics."""
    
    affidavit_type_id = serializers.IntegerField()
    affidavit_type_name = serializers.CharField()
    total_requests = serializers.IntegerField()
    needs_review_count = serializers.IntegerField()
    needs_clarification_count = serializers.IntegerField()
    review_load_percentage = serializers.FloatField()
    is_instant_mode = serializers.BooleanField()
    is_promotion_candidate = serializers.BooleanField()


class LearningExportSerializer(serializers.Serializer):
    """Serializer for learning export data."""
    
    issue_type = serializers.CharField()
    count = serializers.IntegerField()
    examples = ReviewerEditSerializer(many=True)


# =============================================================================
# Audit Trail Serializers
# =============================================================================

class RequestEventSerializer(serializers.ModelSerializer):
    """Serializer for request audit events."""
    
    actor_name = serializers.SerializerMethodField()
    action_display = serializers.CharField(source='get_action_display', read_only=True)
    
    class Meta:
        model = RequestEvent
        fields = [
            'id', 'request', 'action', 'action_display', 
            'actor', 'actor_name', 'actor_role',
            'ip_address', 'details', 'created_at'
        ]
        read_only_fields = fields
    
    def get_actor_name(self, obj):
        if obj.actor:
            return obj.actor.get_full_name() or obj.actor.username
        return 'System'


class AIRunSerializer(serializers.ModelSerializer):
    """Serializer for AI run logs."""
    
    node_type_display = serializers.CharField(source='get_node_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = AIRun
        fields = [
            'id', 'request', 'node_type', 'node_type_display',
            'status', 'status_display', 'model_name', 'prompt_version',
            'prompt_tokens', 'completion_tokens', 'total_tokens',
            'latency_ms', 'estimated_cost_usd', 'created_at'
        ]
        read_only_fields = fields


# =============================================================================
# AI Base Instruction Serializers
# =============================================================================

class AIBaseInstructionSerializer(serializers.ModelSerializer):
    """Serializer for AI base instruction (singleton)."""
    
    updated_by_name = serializers.SerializerMethodField()
    
    class Meta:
        model = AIBaseInstruction
        fields = [
            'id', 'instruction_text', 'version', 'is_active',
            'created_at', 'updated_at', 'updated_by', 'updated_by_name'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'updated_by', 'updated_by_name']
    
    def get_updated_by_name(self, obj):
        if obj.updated_by:
            return obj.updated_by.get_full_name() or obj.updated_by.username
        return None
    
    def create(self, validated_data):
        # Set the user who created/updated
        request = self.context.get('request')
        if request and hasattr(request, 'user'):
            validated_data['updated_by'] = request.user
        return super().create(validated_data)
    
    def update(self, instance, validated_data):
        # Set the user who updated
        request = self.context.get('request')
        if request and hasattr(request, 'user'):
            validated_data['updated_by'] = request.user
        return super().update(instance, validated_data)


# =============================================================================
# Document Upload Serializers
# =============================================================================

class DocumentUploadSerializer(serializers.Serializer):
    """Serializer for document upload (Word/PDF to HTML)."""
    
    file = serializers.FileField(
        help_text="Word (.docx) or PDF (.pdf) file to upload"
    )
    
    def validate_file(self, value):
        # Check file extension
        filename = value.name.lower()
        if not (filename.endswith('.docx') or filename.endswith('.pdf')):
            raise serializers.ValidationError(
                "Only .docx and .pdf files are supported"
            )
        
        # Check file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if value.size > max_size:
            raise serializers.ValidationError(
                f"File size exceeds maximum of 10MB"
            )
        
        return value


class DocumentParseResultSerializer(serializers.Serializer):
    """Serializer for document parse result."""
    
    success = serializers.BooleanField()
    html_content = serializers.CharField()
    filename = serializers.CharField()
    file_type = serializers.CharField()
    parsed_at = serializers.CharField()
    error = serializers.CharField(allow_null=True)


class PolicyGenerationRequestSerializer(serializers.Serializer):
    """Serializer for policy generation request."""
    
    affidavit_type_id = serializers.IntegerField()
    additional_context = serializers.CharField(
        required=False, 
        allow_blank=True,
        help_text="Additional instructions for policy generation"
    )


class PolicyGenerationResultSerializer(serializers.Serializer):
    """Serializer for policy generation result."""
    
    success = serializers.BooleanField()
    template_html = serializers.CharField()
    detected_fields = serializers.ListField(child=serializers.DictField())
    disallowed_phrases = serializers.ListField(child=serializers.CharField())
    required_sections = serializers.ListField(child=serializers.CharField())
    validation_rules = serializers.ListField(child=serializers.DictField())
    analysis_notes = serializers.CharField(allow_blank=True)
    error = serializers.CharField(allow_null=True)


class DisallowedPhrasesSerializer(serializers.Serializer):
    """Serializer for managing disallowed phrases."""
    
    phrases = serializers.ListField(
        child=serializers.CharField(max_length=100),
        help_text="List of phrases AI should never use"
    )
    
    def validate_phrases(self, value):
        # Remove duplicates and empty strings
        cleaned = list(set(p.strip() for p in value if p.strip()))
        return cleaned


# =============================================================================
# Site Settings Serializers
# =============================================================================

class SiteSettingsSerializer(serializers.ModelSerializer):
    """Serializer for global site settings."""
    
    class Meta:
        model = SiteSettings
        fields = [
            'id', 'default_payout_amount', 
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


# =============================================================================
# Ticket Serializers
# =============================================================================

class TicketAttachmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = TicketAttachment
        fields = ['id', 'file', 'uploaded_at']
        read_only_fields = ['uploaded_at']

class TicketMessageSerializer(serializers.ModelSerializer):
    sender_name = serializers.CharField(source='sender.get_full_name', read_only=True)
    sender_role = serializers.CharField(source='sender.role', read_only=True)
    sender_avatar = serializers.SerializerMethodField()
    
    class Meta:
        model = TicketMessage
        fields = ['id', 'sender', 'sender_name', 'sender_role', 'sender_avatar', 'message', 'created_at', 'is_internal']
        read_only_fields = ['id', 'sender', 'created_at']
        
    def get_sender_avatar(self, obj):
        if obj.sender.profile_image:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.sender.profile_image.url)
            return obj.sender.profile_image.url
        return None

class TicketSerializer(serializers.ModelSerializer):
    user_name = serializers.CharField(source='user.get_full_name', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    priority_display = serializers.CharField(source='get_priority_display', read_only=True)
    category_display = serializers.CharField(source='get_category_display', read_only=True)
    
    class Meta:
        model = Ticket
        fields = [
            'id', 'user', 'user_name', 'request', 'subject', 'description', 
            'category', 'category_display', 'status', 'status_display', 
            'priority', 'priority_display', 'created_at', 'updated_at', 'resolved_at'
        ]
        read_only_fields = ['id', 'user', 'created_at', 'updated_at', 'resolved_at']

class TicketDetailSerializer(TicketSerializer):
    messages = TicketMessageSerializer(many=True, read_only=True)
    attachments = TicketAttachmentSerializer(many=True, read_only=True)
    
    class Meta(TicketSerializer.Meta):
        fields = TicketSerializer.Meta.fields + ['messages', 'attachments']


class CommissionerSlotSerializer(serializers.ModelSerializer):
    """Serializer for commissioner availability slots."""
    
    request_details = serializers.SerializerMethodField()
    
    class Meta:
        model = CommissionerSlot
        fields = ['id', 'commissioner', 'start_time', 'is_booked', 'appointment_status', 'decision_at', 'decision_reason', 'request_details']
        read_only_fields = ['id', 'commissioner', 'is_booked', 'appointment_status', 'decision_at', 'decision_reason']
        
    def get_request_details(self, obj):
        # Allow request details to be shown if booked OR if user is commissioner viewing their own schedule
        request = self.context.get('request')
        
        if obj.is_booked and obj.request:
            # Security check: Only show details if:
            # 1. User is the commissioner who owns the slot
            # 2. User is the client who made the request
            # 3. User is an admin
            if request and (
                request.user == obj.commissioner or 
                request.user == obj.request.user or 
                request.user.is_staff
            ):
                return {
                    'request_code': obj.request.request_code,
                    'client_name': obj.request.user.get_full_name() or obj.request.user.username,
                    'affidavit_type': obj.request.affidavit_type.name,
                    'status': obj.request.status
                }
        return None
