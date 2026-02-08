"""
Signal handlers for affidavits app.
"""

from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import User
from .services.slot_service import generate_slots_for_commissioner
import logging

logger = logging.getLogger(__name__)

@receiver(post_save, sender=User)
def handle_commissioner_save(sender, instance, created, **kwargs):
    """
    Automatically generate slots when a commissioner is created or updated.
    """
    if instance.role == User.Role.COMMISSIONER:
        # If new commissioner created, generate slots immediately
        if created:
            logger.info(f"New commissioner created: {instance.username}. Generating initial slots.")
            generate_slots_for_commissioner(instance)
            
        # If existing commissioner updated, check if we need to regenerate
        # For now, we regenerate on every save to be safe (it's idempotent via get_or_create)
        # Ideally, we would check if 'availability' field changed using __init__ tracking
        else:
            # We don't want to regenerate on every single login or minor update
            # But since we don't have field tracking easily available without a mixin,
            # we will regenerate. The service is fast if slots exist.
            # Optimization: Only if update_fields is None (full save) or contains 'availability'
            update_fields = kwargs.get('update_fields')
            if update_fields is None or 'availability' in update_fields:
                logger.info(f"Commissioner {instance.username} updated. Refreshing slots.")
                # Pass cleanup=True to remove old slots that might no longer match new availability
                generate_slots_for_commissioner(instance, cleanup=True)
