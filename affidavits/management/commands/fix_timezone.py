from django.core.management.base import BaseCommand
from django.utils import timezone
from affidavits.models import Request, CommissionerSlot


class Command(BaseCommand):
    help = 'Fix appointment times to display in local timezone'

    def handle(self, *args, **options):
        # Get the specific request
        try:
            request = Request.objects.get(id=105)
            
            if hasattr(request, 'appointment_slot'):
                slot = request.appointment_slot
                self.stdout.write(f'Current slot time (UTC): {slot.start_time}')
                self.stdout.write(f'Current slot time (local): {slot.start_time.astimezone()}')
                
                # The issue is that the time was stored as UTC but we want it to display as local
                # Since we changed the timezone setting, we need to adjust the stored time
                # The slot shows 2:00 PM UTC but should be 10:00 AM local (4 hour difference for Trinidad)
                
                # Convert UTC time to local timezone equivalent
                # Trinidad is UTC-4, so we need to subtract 4 hours from the UTC time
                from datetime import timedelta
                local_time = slot.start_time - timedelta(hours=4)
                
                self.stdout.write(f'Will update to: {local_time}')
                
                # Update the slot time
                slot.start_time = local_time
                slot.save()
                
                self.stdout.write(self.style.SUCCESS('Successfully updated appointment time to local timezone'))
            else:
                self.stdout.write(self.style.WARNING('No appointment slot found for request 105'))
                
        except Request.DoesNotExist:
            self.stdout.write(self.style.ERROR('Request 105 not found'))
