"""
Custom authentication backend for email-based login.
"""

from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend

User = get_user_model()


class EmailBackend(ModelBackend):
    """
    Authenticate using email address only.
    """
    
    def authenticate(self, request, username=None, password=None, **kwargs):
        if username is None or password is None:
            return None
        
        try:
            # Treat 'username' parameter as email
            user = User.objects.get(email__iexact=username)
        except User.DoesNotExist:
            # Run the default password hasher once to reduce timing attacks
            User().set_password(password)
            return None
        
        # Check password and active status
        if user.check_password(password) and self.user_can_authenticate(user):
            return user
        
        return None
