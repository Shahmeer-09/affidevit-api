"""
Centralized notification message templates.
All notification copy is maintained here for easy modification.
Use .format(**kwargs) to inject dynamic values.
"""

# ============================================================================
# APPOINTMENT MESSAGES
# ============================================================================
APPOINTMENT_MESSAGES = {
    # Sent to user when slot is booked
    "BOOKED_USER": (
        "Your appointment with {commissioner_name} has been scheduled for "
        "{slot_date} at {slot_time}. Request code: {request_code}. "
        "Please arrive on time with your documents."
    ),
    # Sent to commissioner when user books a slot
    "BOOKED_COMMISSIONER": (
        "New appointment scheduled: {user_name} has booked your slot on "
        "{slot_date} at {slot_time} for affidavit type '{affidavit_type}'. "
        "Request code: {request_code}."
    ),
    # Sent to user when commissioner accepts appointment
    "ACCEPTED_USER": (
        "Great news! Your appointment for request {request_code} has been "
        "confirmed by {commissioner_name}. See you on {slot_date} at {slot_time}."
    ),
    # Sent to user when commissioner rejects appointment
    "REJECTED_USER": (
        "Unfortunately, your appointment for request {request_code} was declined "
        "by the commissioner. Please select a new commissioner and time slot."
    ),
    # Sent to user when commissioner cancels appointment
    "CANCELLED_BY_COMMISSIONER_USER": (
        "Your appointment for request {request_code} was cancelled by the "
        "commissioner. Please select a new commissioner and reschedule."
    ),
    # Sent to commissioner when user withdraws/cancels
    "WITHDRAWN_COMMISSIONER": (
        "User {user_name} has withdrawn the appointment for request "
        "{request_code} scheduled on {slot_date} at {slot_time}."
    ),
    # Sent to commissioner when user cancels slot
    "CANCELLED_BY_USER_COMMISSIONER": (
        "Appointment cancelled: {user_name} has cancelled the appointment "
        "for request {request_code} on {slot_date} at {slot_time}."
    ),
}

# ============================================================================
# REQUEST STATUS MESSAGES
# ============================================================================
REQUEST_STATUS_MESSAGES = {
    # Sent to user when request enters review
    "ENTERED_REVIEW_USER": (
        "Your affidavit request {request_code} is now being reviewed. "
        "You will be notified once the review is complete."
    ),
    # Sent to reviewers when a request enters review queue
    "ENTERED_REVIEW_REVIEWER": (
        "New request {request_code} is now in the review queue. "
        "Affidavit type: {affidavit_type}."
    ),
    # Sent to user on approval
    "APPROVED_USER": (
        "Your affidavit request {request_code} has been approved! "
        "Please check your dashboard to view and download your document."
    ),
    # Sent to user on rejection
    "REJECTED_USER": (
        "Your affidavit request {request_code} was not approved. "
        "Reason: {rejection_reason}. Please review and resubmit if applicable."
    ),
    # Sent to user when clarification is needed
    "CLARIFICATION_USER": (
        "Clarification needed for your request {request_code}: {clarification_message}. "
        "Please update your submission."
    ),
    # Sent to user on completion (after commissioner stamps)
    "COMPLETED_USER": (
        "Your affidavit {request_code} has been completed and notarized by "
        "{commissioner_name}. You can now download the final document."
    ),
}

# ============================================================================
# COMMISSIONER BALANCE MESSAGES
# ============================================================================
COMMISSIONER_BALANCE_MESSAGES = {
    # Shown to commissioner after completing a request (stamp)
    "PAYOUT_ADDED": (
        "Rs. {amount} has been added to your pending payout balance for "
        "completing request {request_code}."
    ),
    # Summary message
    "PENDING_BALANCE_SUMMARY": (
        "Your current pending payout balance is Rs. {total_pending}."
    ),
}

# ============================================================================
# EMAIL SUBJECTS
# ============================================================================
EMAIL_SUBJECTS = {
    "APPOINTMENT_BOOKED_USER": "Appointment Scheduled - {request_code}",
    "APPOINTMENT_BOOKED_COMMISSIONER": "New Appointment - {request_code}",
    "APPOINTMENT_ACCEPTED_USER": "Appointment Confirmed - {request_code}",
    "APPOINTMENT_REJECTED_USER": "Appointment Declined - Action Required",
    "APPOINTMENT_CANCELLED_USER": "Appointment Cancelled - Action Required",
    "REQUEST_APPROVED": "Request Approved - {request_code}",
    "REQUEST_REJECTED": "Request Not Approved - {request_code}",
    "REQUEST_CLARIFICATION": "Clarification Needed - {request_code}",
    "REQUEST_COMPLETED": "Affidavit Completed - {request_code}",
    "REQUEST_IN_REVIEW": "Request Under Review - {request_code}",
}

# ============================================================================
# APPOINTMENT STATUS CHOICES (for model)
# ============================================================================
APPOINTMENT_STATUS_CHOICES = [
    ('pending', 'Pending'),
    ('accepted', 'Accepted'),
    ('rejected', 'Rejected'),
    ('cancelled_by_commissioner', 'Cancelled by Commissioner'),
    ('cancelled_by_user', 'Cancelled by User'),
]
