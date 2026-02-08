"""
Affidavit Express - Authentication Module

Custom JWT authentication with role-based claims and permission classes.
"""

from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import permissions, serializers
from django.contrib.auth import authenticate, get_user_model

User = get_user_model()


class EmailTokenObtainPairSerializer(TokenObtainPairSerializer):
    """
    Custom JWT serializer that uses email instead of username.
    Also adds user role to token claims and supports "remember me".
    """
    username_field = 'email'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove 'username' field and add 'email' field
        self.fields.pop('username', None)
        self.fields['email'] = serializers.EmailField()
        self.fields['remember_me'] = serializers.BooleanField(required=False, default=False)
    
    def validate(self, attrs):
        email = attrs.get('email')
        password = attrs.get('password')
        remember_me = attrs.get('remember_me', False)
        
        if email and password:
            # Use our EmailBackend for authentication
            user = authenticate(
                request=self.context.get('request'),
                username=email,  # Our backend treats this as email
                password=password
            )
            
            if not user:
                raise serializers.ValidationError(
                    'No active account found with the given credentials'
                )
            
            if not user.is_active:
                raise serializers.ValidationError('User account is disabled.')
            
            # Check if commissioner is approved (featured field used as approval flag)
            if user.role == 'commissioner' and not user.is_featured:
                raise serializers.ValidationError('Your account is pending admin approval. You will be notified once approved.')
            
            # Generate tokens with extended lifetime if remember_me is True
            refresh = self.get_token(user)
            
            # Extend token lifetime if "remember me" is checked
            if remember_me:
                from datetime import timedelta
                # Access token: 7 days instead of 60 minutes
                refresh.access_token.set_exp(lifetime=timedelta(days=7))
                # Refresh token: 30 days instead of 7 days
                refresh.set_exp(lifetime=timedelta(days=30))
            
            return {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'role': user.role,
                    'phone': user.phone_number,
                    'is_superuser': user.is_superuser,
                },
                'remember_me': remember_me,
            }
        
        raise serializers.ValidationError('Must include "email" and "password".')
    
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        
        # Add custom claims
        token['role'] = user.role
        token['username'] = user.username
        token['email'] = user.email
        token['full_name'] = user.get_full_name() or user.username
        
        # Add commissioner-specific claims
        if user.role == 'commissioner':
            token['commission_number'] = user.commission_number
        
        return token


class CustomTokenObtainPairView(TokenObtainPairView):
    """Custom token obtain view using email-based serializer."""
    serializer_class = EmailTokenObtainPairSerializer


# =============================================================================
# Permission Classes
# =============================================================================

class IsPublicUser(permissions.BasePermission):
    """
    Allow access only to public users (or higher roles).
    Essentially allows any authenticated user.
    """
    message = "Authentication required."
    
    def has_permission(self, request, view):
        return request.user and request.user.is_authenticated


class IsCommissioner(permissions.BasePermission):
    """
    Allow access only to commissioners (or admins).
    """
    message = "Commissioner access required."
    
    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        return request.user.role in ['commissioner', 'admin']


class IsReviewer(permissions.BasePermission):
    """
    Allow access only to reviewers (or admins).
    """
    message = "Reviewer access required."
    
    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        return request.user.role in ['reviewer', 'admin']


class IsAdminUser(permissions.BasePermission):
    """
    Allow access only to admin users.
    """
    message = "Admin access required."
    
    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        return request.user.role == 'admin'


class IsSuperUser(permissions.BasePermission):
    """
    Allow access only to superusers.
    Used for super-admin only actions (decision tree + affidavit type management).
    """
    message = "Superuser access required."
    
    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        return bool(request.user.is_superuser)


class IsOwnerOrAdmin(permissions.BasePermission):
    """
    Object-level permission: allow access only to the owner or admins.
    Used for Request objects where users can only see their own requests.
    """
    message = "You do not have permission to access this resource."
    
    def has_object_permission(self, request, view, obj):
        if request.user.role == 'admin':
            return True
        
        # Check if the object has a 'user' attribute
        if hasattr(obj, 'user'):
            return obj.user == request.user
        
        return False


class IsCommissionerOrReviewerOrAdmin(permissions.BasePermission):
    """
    Allow access to commissioners, reviewers, or admins.
    Used for endpoints that staff members can access.
    """
    message = "Staff access required."
    
    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        return request.user.role in ['commissioner', 'reviewer', 'admin']


class AllowAny(permissions.BasePermission):
    """
    Allow any access (authenticated or not).
    Used for public endpoints like decision tree.
    """
    
    def has_permission(self, request, view):
        return True
