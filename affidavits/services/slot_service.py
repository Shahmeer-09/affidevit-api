"""
Service for managing commissioner availability slots.
"""

import datetime
import logging
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    # Fallback for older python versions if needed, though 3.12 has it
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from django.utils import timezone
from affidavits.models import User, CommissionerSlot

logger = logging.getLogger(__name__)

def parse_time_string(time_str: str) -> datetime.time:
    """
    Parse time string in various formats (24h or 12h AM/PM).
    Supports: "09:00", "17:00", "9:00 AM", "5:00 PM", "9 AM", "5 PM"
    """
    time_str = time_str.strip().upper()
    formats = [
        '%H:%M',        # 09:00, 17:00
        '%I:%M %p',     # 09:00 AM, 05:00 PM
        '%I %p',        # 9 AM, 5 PM
        '%I:%M%p',      # 9:00AM
    ]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(time_str, fmt).time()
        except ValueError:
            continue
            
    # Fallback: try split logic if simple integer (e.g. "9", "17")
    try:
        if ':' not in time_str and ' ' not in time_str:
            hour = int(time_str)
            if 0 <= hour <= 23:
                return datetime.time(hour, 0)
    except ValueError:
        pass

    raise ValueError(f"Could not parse time string: {time_str}")

def generate_slots_for_commissioner(commissioner: User, days: int = 14, cleanup: bool = False) -> int:
    """
    Generate availability slots for a commissioner for the next N days.
    
    Args:
        commissioner: The commissioner User instance
        days: Number of days to generate slots for (default: 14)
        cleanup: If True, delete existing unbooked future slots before generating (default: False)
        
    Returns:
        int: Number of new slots created
    """
    if not commissioner.is_commissioner:
        logger.warning(f"User {commissioner.username} is not a commissioner. Skipping slot generation.")
        return 0
        
    # Get availability from commissioner profile or use default
    # Structure: {
    #   "recurring": {"monday": [{"start": "09:00", "end": "17:00"}], ...},
    #   "timezone": "America/Port_of_Spain"
    # }
    availability = commissioner.availability or {}
    recurring = availability.get('recurring', {})
    
    # Determine Commissioner's Timezone
    # Default to Trinidad (America/Port_of_Spain) if not set, as this is a TT app
    tz_name = availability.get('timezone', 'America/Port_of_Spain')
    try:
        commissioner_tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        logger.warning(f"Unknown timezone {tz_name} for {commissioner.username}, defaulting to UTC")
        commissioner_tz = datetime.timezone.utc

    if cleanup:
        # Determine start of TODAY in commissioner's timezone
        now_local = timezone.now().astimezone(commissioner_tz)
        today_local = now_local.date()
        start_of_day_local = datetime.datetime.combine(today_local, datetime.time.min).replace(tzinfo=commissioner_tz)
        cleanup_start_utc = start_of_day_local.astimezone(datetime.timezone.utc)
        
        deleted_count, _ = CommissionerSlot.objects.filter(
            commissioner=commissioner,
            is_booked=False,
            request__isnull=True,
            start_time__gte=cleanup_start_utc
        ).delete()
        logger.info(f"Cleaned up {deleted_count} stale slots for {commissioner.username} starting from {cleanup_start_utc}")

    # Use commissioner's timezone so slot dates align with their local day
    today = timezone.now().astimezone(commissioner_tz).date()
    count = 0
    
    # Default to Mon-Fri 9-5 if no specific availability set
    use_defaults = not recurring
    
    logger.info(f"Generating slots for {commissioner.username} for next {days} days in {tz_name}")
    
    for i in range(days):
        # Calculate date in Commissioner's timezone
        # We start from 'today' but should respect the timezone
        # For simplicity, we iterate dates from today.
        current_date = today + datetime.timedelta(days=i)
        weekday_name = current_date.strftime('%A').lower()  # monday, tuesday, etc.
        
        # Determine working hours for this day
        day_slots = []
        
        if use_defaults:
            # Skip weekends (5=Sat, 6=Sun)
            if current_date.weekday() >= 5:
                continue
            day_slots = [{'start': '09:00', 'end': '17:00'}]
        else:
            # Check custom availability
            day_slots = recurring.get(weekday_name, [])
            
        # Process each time range for the day
        for time_range in day_slots:
            try:
                start_str = time_range.get('start', '09:00')
                end_str = time_range.get('end', '17:00')
                
                # Parse times using robust parser
                start_time_obj = parse_time_string(start_str)
                end_time_obj = parse_time_string(end_str)
                
                # Create naive datetime objects
                start_dt_naive = datetime.datetime.combine(current_date, start_time_obj)
                end_dt_naive = datetime.datetime.combine(current_date, end_time_obj)
                
                # Handle overnight slots (e.g. 10 PM to 2 AM) - add day to end time
                if end_dt_naive <= start_dt_naive:
                    end_dt_naive += datetime.timedelta(days=1)
                
                # Localize to Commissioner's Timezone
                # This says "This naive time IS in Trinidad time"
                current_dt_aware = timezone.make_aware(start_dt_naive, commissioner_tz)
                end_dt_aware = timezone.make_aware(end_dt_naive, commissioner_tz)
                
                # Generate 30-min slots
                while current_dt_aware < end_dt_aware:
                    # Convert to UTC for storage (Django standard)
                    # Although Django handles aware datetimes automatically, ensuring conversion is safe
                    slot_start_utc = current_dt_aware.astimezone(datetime.timezone.utc)
                    slot_end_utc = (current_dt_aware + datetime.timedelta(minutes=30)).astimezone(datetime.timezone.utc)
                    
                    # Create slot if it doesn't exist
                    # We use get_or_create so we don't duplicate or overwrite booked slots
                    slot, created = CommissionerSlot.objects.get_or_create(
                        commissioner=commissioner,
                        start_time=slot_start_utc
                    )
                    
                    if created:
                        count += 1
                        
                    # Move to next 30 min slot
                    current_dt_aware += datetime.timedelta(minutes=30)
                    
            except ValueError as e:
                logger.error(f"Error parsing time range for {commissioner.username} on {weekday_name}: {e}")
                continue
                
    logger.info(f"Generated {count} new slots for {commissioner.username}")
    return count
