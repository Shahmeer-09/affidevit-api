"""
Management command to generate commissioner availability slots.
Runs nightly via Celery or manually.
"""

from django.core.management.base import BaseCommand
from affidavits.models import User
from affidavits.services.slot_service import generate_slots_for_commissioner

class Command(BaseCommand):
    help = 'Generate daily time slots for commissioners based on their availability'

    def add_arguments(self, parser):
        parser.add_argument(
            '--refresh',
            action='store_true',
            help='Clean up existing unbooked future slots before regenerating',
        )

    def handle(self, *args, **options):
        cleanup = options['refresh']
        self.stdout.write(f"Generating commissioner slots (cleanup={cleanup})...")
        
        commissioners = User.objects.filter(role=User.Role.COMMISSIONER)
        total_count = 0
        
        for commissioner in commissioners:
            count = generate_slots_for_commissioner(commissioner, cleanup=cleanup)
            total_count += count
        
        self.stdout.write(self.style.SUCCESS(f'Successfully generated {total_count} new slots.'))
