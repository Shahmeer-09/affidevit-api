"""
Affidavit Express - Policy Generator Service

Uses GPT-4o to analyze uploaded example affidavits and generate:
- Template HTML structure with {{field}} placeholders
- Policy JSON with validation rules
- Suggested disallowed phrases
- Few-shot examples for training
- Smart field validation for Trinidad & Tobago
"""

import json
import logging
import re
from typing import Dict, List, Optional
from django.conf import settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# Lazy import openai
openai_client = None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
    before_sleep=lambda retry_state: logger.warning(f"OpenAI API retry {retry_state.attempt_number}/3 in policy generator after error: {retry_state.outcome.exception()}")
)
def call_openai_with_retry(client, model, messages, **kwargs):
    """
    Wrapper for OpenAI API calls with retry logic and exponential back-off.
    
    Args:
        client: OpenAI client instance
        model: OpenAI model name
        messages: List of message dictionaries
        **kwargs: Additional parameters for the API call
        
    Returns:
        OpenAI API response
        
    Raises:
        Exception: If all retries are exhausted
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )

# =============================================================================
# TRINIDAD & TOBAGO SPECIFIC VALIDATION RULES
# =============================================================================

# Field patterns for smart detection and validation
TT_FIELD_RULES = {
    # National ID / Electoral ID - Trinidad & Tobago format (11 digits: YYYYMMDDXXX)
    'national_id': {
        'patterns': [
            'national_id', 'id_number', 'identification_number', 'id_card', 'national_identification',
            'electoral_id', 'electoral_identification', 'electoral_card', 'electoral_identification_card',
            'electoral_identification_card_number', 'eic', 'eic_number'
        ],
        'validation': {
            'pattern': r'^\d{11}$',
            'input_mode': 'numeric',
            'max_length': 11,
            'min_length': 11,
            'message': 'Enter valid Trinidad & Tobago Electoral ID (11 digits: YYYYMMDDXXX)'
        },
        'type': 'text',
        'placeholder': '19741104044',
        'help_text': 'Your 11-digit Trinidad & Tobago Electoral ID (includes your date of birth)'
    },
    # Phone number - Trinidad & Tobago format
    'phone': {
        'patterns': ['phone', 'telephone', 'mobile', 'contact_number', 'cell'],
        'validation': {
            'pattern': r'^(\+?1)?[-.\s]?868[-.\s]?\d{3}[-.\s]?\d{4}$',
            'input_mode': 'tel',
            'message': 'Enter valid Trinidad & Tobago phone number (868-XXX-XXXX)'
        },
        'type': 'phone',
        'placeholder': '868-123-4567'
    },
    # Name fields - text only, no numbers
    'name': {
        'patterns': ['full_name', 'first_name', 'last_name', 'middle_name', 'surname', 'given_name', 
                    'deponent_name', 'witness_name', 'applicant_name', 'name_of'],
        'validation': {
            'pattern': r'^[a-zA-Z\s\-\'\.]+$',
            'input_mode': 'text_only',
            'min_length': 2,
            'max_length': 100,
            'message': 'Name can only contain letters, spaces, hyphens, and apostrophes'
        },
        'type': 'text',
        'placeholder': 'John Michael Smith'
    },
    # Date of Birth - with age calculation, cannot be in future
    'date_of_birth': {
        'patterns': [
            'date_of_birth', 'dob', 'birth_date', 'birthdate',
            'birth', 'born_on', 'date_born', 'applicant_dob',
            'date_of_birth_of', 'dob_of', 'birth_day'
        ],
        'validation': {
            'max_date': 'today',
            'date_constraint': 'past_only',
            'message': 'Date of birth cannot be in the future'
        },
        'type': 'date',
        'computed_fields': ['age']  # Age will be calculated from this
    },
    # Age field - CONVERT to Date of Birth with calendar picker
    # Age will be auto-calculated from DOB
    'age': {
        'patterns': ['age', 'years_old', 'current_age'],
        'convert_to_dob': True,  # Flag to convert this field to date_of_birth
        'replacement_field': {
            'id': 'date_of_birth',
            'label': 'Date of Birth',
            'type': 'date',
            'help_text': 'Your age will be calculated automatically from your date of birth',
            'validation': {
                'max_date': 'today',
                'date_constraint': 'past_only',
                'message': 'Date of birth cannot be in the future'
            }
        },
        'type': 'date'  # Will be converted to date picker
    },
    # Declaration date (single combined field) - NO date constraint, user chooses freely
    'declaration_date': {
        'patterns': ['declaration_date', 'sworn_date', 'dated', 'date_declared', 'affirmed_date'],
        'validation': {},  # No past/future constraint — declaration date is the user's choice
        'type': 'date'
    },
    # --- Declaration date parts (declared_at section: City / Day / Month / Year) ---
    # These MUST use the declaration_ prefix so the system knows NOT to apply date constraints.
    # Users are free to enter any date when making/signing their declaration.
    'declaration_city': {
        'patterns': ['declaration_city'],
        'validation': {
            'pattern': r"^[a-zA-Z\s\-'\.]+$",
            'input_mode': 'text_only',
            'message': 'City can only contain letters'
        },
        'type': 'text',
        'placeholder': 'Port of Spain',
    },
    'declaration_day': {
        'patterns': ['declaration_day'],
        'validation': {
            'pattern': r'^([1-9]|[12]\d|3[01])$',
            'input_mode': 'numeric',
            'min': 1,
            'max': 31,
            'message': 'Enter a valid day (1-31)'
        },
        'type': 'number',
        'placeholder': '4',
    },
    'declaration_month': {
        'patterns': ['declaration_month'],
        'validation': {
            'pattern': r'^(January|February|March|April|May|June|July|August|September|October|November|December)$',
            'input_mode': 'text_only',
            'message': 'Please select a valid month'
        },
        'type': 'select',
        'placeholder': 'Select month',
        'options': [
            {'value': 'January', 'label': 'January'},
            {'value': 'February', 'label': 'February'},
            {'value': 'March', 'label': 'March'},
            {'value': 'April', 'label': 'April'},
            {'value': 'May', 'label': 'May'},
            {'value': 'June', 'label': 'June'},
            {'value': 'July', 'label': 'July'},
            {'value': 'August', 'label': 'August'},
            {'value': 'September', 'label': 'September'},
            {'value': 'October', 'label': 'October'},
            {'value': 'November', 'label': 'November'},
            {'value': 'December', 'label': 'December'}
        ],
    },
    'declaration_year': {
        'patterns': ['declaration_year'],
        'validation': {
            'pattern': r'^\d{4}$',
            'input_mode': 'numeric',
            'min': 1900,
            'message': 'Enter a valid 4-digit year'
        },
        'type': 'number',
        'placeholder': '2026',
    },
    # Event dates - typically past dates
    'event_date': {
        'patterns': ['incident_date', 'event_date', 'occurrence_date', 'date_of_incident', 'date_of_event'],
        'validation': {
            'max_date': 'today',
            'date_constraint': 'past_only',
            'message': 'Event date cannot be in the future'
        },
        'type': 'date'
    },
    # Address fields - text with numbers allowed
    'address': {
        'patterns': ['address', 'street_address', 'residential_address', 'home_address', 'mailing_address'],
        'validation': {
            'input_mode': 'text',
            'min_length': 5,
            'max_length': 200,
            'message': 'Please enter a valid address'
        },
        'type': 'textarea',
        'placeholder': '15 Queen Street, St. Augustine'
    },
    # City/Town - text only
    'city': {
        'patterns': ['city', 'town', 'village', 'municipality', 'location'],
        'validation': {
            'pattern': r'^[a-zA-Z\s\-\'\.]+$',
            'input_mode': 'text_only',
            'message': 'City/Town can only contain letters'
        },
        'type': 'text',
        'placeholder': 'Port of Spain'
    },
    # Month names - generic non-declaration month fields (e.g. event month)
    'month': {
        'patterns': ['month', 'month_name'],  # declaration_month handled by its own rule above
        'validation': {
            'pattern': r'^(January|February|March|April|May|June|July|August|September|October|November|December)$',
            'input_mode': 'text_only',
            'message': 'Please select a valid month',
            'check_future_date': True
        },
        'type': 'select',
        'placeholder': 'Select month',
        'options': [
            {'value': 'January', 'label': 'January'},
            {'value': 'February', 'label': 'February'},
            {'value': 'March', 'label': 'March'},
            {'value': 'April', 'label': 'April'},
            {'value': 'May', 'label': 'May'},
            {'value': 'June', 'label': 'June'},
            {'value': 'July', 'label': 'July'},
            {'value': 'August', 'label': 'August'},
            {'value': 'September', 'label': 'September'},
            {'value': 'October', 'label': 'October'},
            {'value': 'November', 'label': 'November'},
            {'value': 'December', 'label': 'December'}
        ]
    },
    # Year - generic non-declaration year fields (declaration_year handled by its own rule above)
    'year': {
        'patterns': ['year', 'year_of'],
        'validation': {
            'pattern': r'^\d{4}$',
            'input_mode': 'numeric',
            'min': 1900,
            'max_year_current': True,
            'check_future_date': True,
            'message': 'Enter a valid year (cannot be in the future)'
        },
        'type': 'number',
        'placeholder': '2026'
    },
    # Day - generic non-declaration day fields (declaration_day handled by its own rule above)
    'day': {
        'patterns': ['day', 'day_of'],
        'validation': {
            'pattern': r'^([1-9]|[12]\d|3[01])$',
            'input_mode': 'numeric',
            'min': 1,
            'max': 31,
            'check_future_date': True,
            'message': 'Enter a valid day (1-31)'
        },
        'type': 'number',
        'placeholder': '4'
    },
    # Email
    'email': {
        'patterns': ['email', 'email_address', 'e_mail'],
        'validation': {
            'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'input_mode': 'email',
            'message': 'Enter a valid email address'
        },
        'type': 'email',
        'placeholder': 'example@email.com'
    },
    # Occupation - text only
    'occupation': {
        'patterns': ['occupation', 'profession', 'job', 'employment', 'work'],
        'validation': {
            'pattern': r'^[a-zA-Z\s\-\'\.]+$',
            'input_mode': 'text_only',
            'min_length': 2,
            'message': 'Occupation can only contain letters'
        },
        'type': 'text',
        'placeholder': 'Teacher'
    },
    # Passport number
    'passport': {
        'patterns': ['passport', 'passport_number', 'passport_no'],
        'validation': {
            'pattern': r'^[A-Z]{2}\d{7}$',
            'input_mode': 'text',
            'message': 'Enter valid Trinidad & Tobago passport number (e.g., TB1234567)'
        },
        'type': 'text',
        'placeholder': 'TB1234567'
    },
    # Driver's permit
    'drivers_permit': {
        'patterns': ['drivers_permit', 'driving_permit', 'license_number', 'permit_number'],
        'validation': {
            'input_mode': 'text',
            'message': 'Enter valid driver\'s permit number'
        },
        'type': 'text',
        'placeholder': 'DL123456'
    }
}


def get_openai_client():
    """Get or create OpenAI client instance."""
    global openai_client
    if openai_client is None:
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    return openai_client


POLICY_GENERATION_SYSTEM_PROMPT = """You are an expert legal document analyst specializing in affidavits for Trinidad and Tobago.

Your task is to analyze example affidavit documents and generate a structured configuration for an AI drafting system.

You will be given 1-4 example affidavit documents in HTML format. Analyze them to:

1. **Identify the structure** - Common sections, ordering, required elements
2. **Extract field placeholders** - Names, dates, addresses, IDs, etc. that vary per affidavit
3. **Create a template** - HTML template with {{field_name}} placeholders
4. **Suggest disallowed phrases** - Words/phrases that should NOT appear in legal documents
5. **Define validation rules** - Required fields, format constraints

CRITICAL RULES - TEMPLATE MUST MATCH EXAMPLE FORMAT EXACTLY:

1. **Preserve Document Structure Exactly:**
   - If examples start with "REPUBLIC OF TRINIDAD AND TOBAGO:", the template MUST start the same way
   - DO NOT add extra headers like "AFFIDAVIT", "Self-Help Affidavit", or any title/subtitle NOT in examples
   - DO NOT modernize or "improve" the format - replicate it exactly
   - Maintain the exact ordering of sections as seen in examples

2. **HEADINGS MUST BE BOLD (CRITICAL):**
   - ALL document headings and titles MUST be wrapped in <strong> tags
   - Examples of headings that MUST be bold:
     * "REPUBLIC OF TRINIDAD AND TOBAGO:" → <strong>REPUBLIC OF TRINIDAD AND TOBAGO:</strong>
     * "IN THE MATTER OF THE STATUTORY DECLARATION ACT" → <strong>IN THE MATTER OF THE STATUTORY DECLARATION ACT</strong>
     * "CHAPTER 7: No 04" → <strong>CHAPTER 7: No 04</strong>
     * "AFFIDAVIT" or any title → <strong>AFFIDAVIT</strong>
   - Any text that appears as a header, title, or legal reference at the top of the document should be bold
   - The body text (declarations, statements) should NOT be bold
   - "Before me," and "Commissioner of Affidavits." can remain non-bold

3. **Commissioner/Attestation Section - CRITICAL:**
   - Copy the EXACT format from the examples for the commissioner section
   - If examples use simple format like:
     ```
     Declared at [Location]    )
     [City] this [day]         )
     Day of [Month], [Year].   ) --------------------------------
     Before me,
     Commissioner of Affidavits.
     ```
   - Then use EXACTLY that format, NOT a modernized signature block
   - DO NOT add City/Town, Province/State labeled fields if not in examples
   - DO NOT add "Commissioner / Notary Public" or "Deponent Signature" labels if not in examples
   - The commissioner section is for official stamp only - keep it simple as shown

4. **Placeholder Format:**
   - Use {{field_name}} format (double curly braces)
   - Field names should be snake_case (e.g., {{full_name}}, {{date_of_birth}})
   - Replace ONLY variable data with placeholders, preserve all static text exactly

5. **Content Preservation:**
   - Keep all legal language, references, and declarations exactly as in examples
   - Numbered statements should follow the same numbering style as examples
   - Preserve paragraph structure and formatting

6. **What to Extract as Fields:**
   - Personal details (name, age, address, ID numbers)
   - Dates (declaration date, relevant dates in the content)
   - Location information
   - Specific facts that vary per affidavit

=== TRINIDAD & TOBAGO SPECIFIC FIELD RULES (CRITICAL) ===

When detecting fields, apply these SMART VALIDATIONS for Trinidad & Tobago:

**IDENTITY FIELDS:**
- National ID / Electoral ID: Use id "electoral_id" or "national_id", type "text", MUST be exactly 11 digits (format: YYYYMMDDXXX)
- Electoral Identification Card Number = National ID = same format (11 digits, first 8 = DOB)
- Passport: Use id containing "passport", Trinidad format (2 letters + 7 digits, e.g., TB1234567)
- Driver's Permit: Use id "drivers_permit"

**NAME FIELDS (TEXT-ONLY - NO NUMBERS ALLOWED):**
- Any field containing: full_name, first_name, last_name, surname, deponent_name, witness_name
- Set input_mode: "text_only" in validation
- Pattern: letters, spaces, hyphens, apostrophes only

**AGE FIELD - CRITICAL CONVERSION:**
- NEVER ask for age as a number input!
- When the document shows "age X years" or asks for age:
  1. In template_html: Use {{calculated_age}} as placeholder (this will be auto-calculated)
  2. In detected_fields: Create a "date_of_birth" field with type "date" (calendar picker)
  3. The system will automatically calculate age from date_of_birth
- Example template: "I, {{full_name}}, age {{calculated_age}} years, of {{address}}..."
- Example field: {"id": "date_of_birth", "label": "Date of Birth", "type": "date", "help_text": "Your age will be calculated automatically"}

**DECLARATION DATE PARTS — CRITICAL NAMING (use declaration_ prefix, NO date constraints):**
- When the document has a "Declared at [City] this [Day] of [Month], [Year]" section:
  - City where declared → ID: "declaration_city"
  - Day of declaration  → ID: "declaration_day"
  - Month of declaration→ ID: "declaration_month"
  - Year of declaration → ID: "declaration_year"
- DO NOT set max_date, min_date, date_constraint, or check_future_date on these fields
- Declaration dates are the user's FREE CHOICE — they can be past, present, or future

**EVENT / PERSONAL DATE FIELDS (these DO get past constraints):**
- Event/incident dates: MUST have max_date: "today" (past events only)
- Date of Birth: MUST have max_date: "today" and date_constraint: "past_only"
- Set date_constraint: "past_only" or "past_or_today" appropriately

**LOCATION FIELDS (TEXT-ONLY for city/town):**
- City, Town, Village: input_mode "text_only" - no numbers
- Address: Allow both text and numbers (street addresses have numbers)

**NUMBER-ONLY FIELDS:**
- Day (1-31): input_mode "numeric"
- Year: input_mode "numeric", 4 digits
- Phone: input_mode "tel", Trinidad format 868-XXX-XXXX

**MONTH FIELDS:**
- If month is asked separately, use type "select" with month options
- Or input_mode "text_only" if free text

**VALIDATION OBJECT STRUCTURE:**
For each detected_field, include a "validation" object:
{
    "id": "field_id",
    "label": "Label",
    "type": "text|date|number|select|email|phone",
    "required": true,
    "validation": {
        "pattern": "regex pattern if applicable",
        "input_mode": "text_only|numeric|tel|email|text",
        "min_length": number,
        "max_length": number,
        "min": number (for numeric),
        "max": number (for numeric),
        "max_date": "today" (for dates that can't be future),
        "min_date": "today" (for dates that must be future),
        "date_constraint": "past_only|past_or_today|future_only",
        "message": "User-friendly error message"
    }
}

=== END TRINIDAD & TOBAGO RULES ===

**IMPORTANT - SCENARIO-AWARE CONDITIONAL FIELDS:**
When the example documents reveal MULTIPLE distinct scenarios (e.g. "self-owner" vs "tenant",
"individual" vs "company", "with children" vs "without children"), generate:

1. A **scenario selector** — a `select` field that lets the user pick their situation.
   - id: use a descriptive name like "ownership_type", "applicant_category", etc.
   - type: "select"
   - options: one option per detected scenario

2. **Scenario-specific fields** — fields that only apply to one scenario.
   - Add a `show_if` object: `{"field": "<selector_id>", "value": "<option_value>"}`
   - These fields will only appear when the user selects the matching scenario.
   - Mark scenario-specific required fields as `"required": true` — the system
     will automatically skip validation for hidden fields.

3. **Common fields** — fields that appear across ALL scenarios.
   - These have NO `show_if` and are always visible.

Example of a detected_field with show_if:
{
    "id": "landlord_name",
    "label": "Landlord's Full Name",
    "type": "text",
    "required": true,
    "placeholder": "Jane Doe",
    "help_text": "Enter the full name of your landlord",
    "show_if": {"field": "ownership_type", "value": "tenant"}
}

4. In `identified_scenarios`, list every scenario you detected.
5. In `scenario_branches`, provide a mapping from each scenario to its
   relevant template sections/paragraphs so the drafter knows which parts to include:
{
    "scenario_branches": {
        "self_owner": {
            "description": "Declarant owns the property themselves",
            "template_sections": ["ownership declaration paragraph"],
            "key_fields": ["property_description", "title_deed_number"]
        },
        "tenant": {
            "description": "Declarant is renting the property",
            "template_sections": ["tenancy declaration paragraph"],
            "key_fields": ["landlord_name", "lease_start_date", "monthly_rent"]
        }
    }
}

**Rules for show_if:**
- Only use `show_if` when there are CLEAR distinct scenarios in the examples
- The selector field itself must NEVER have show_if
- Keep common fields without show_if (name, DOB, address, electoral_id, etc.)
- A field can only depend on ONE selector — no nested conditionals

**CRITICAL - USE CANONICAL (STANDARD) FIELD IDs:**
When naming fields, ALWAYS use these standard canonical IDs instead of inventing new ones:
- For any name field: use "full_name" (NOT "name", "deponent_name", "applicant_name", etc.)
- For any address: use "address" (NOT "residential_address", "home_address", "current_address")
- For any ID card/number: use "electoral_id" (NOT "national_id", "id_number", "identification_number")
- For date of birth: use "date_of_birth" (NOT "dob", "birth_date")
- For phone: use "phone_number" (NOT "phone", "telephone", "mobile")
- For occupation: use "occupation" (NOT "profession", "job")
- For city/town: use "city" (NOT "town", "village", "municipality")
This ensures consistent mapping between template placeholders and intake questions.

**CROSS-DOCUMENT GENERALIZATION:**
When analyzing MULTIPLE example documents:
- Extract fields that are COMMON across all documents first (these are the core fields)
- Then extract fields specific to individual document variations
- A field that appears in ALL examples is more important than one in just one example
- Prefer GENERAL field names over document-specific ones (e.g., "property_description" not "lot_number_and_plan")
- The template should be a GENERALIZED version that works for ALL the examples, not a copy of one specific example

**CRITICAL - EVERY FIELD MUST HAVE LABEL, HELP TEXT, AND PLACEHOLDER:**
- Each detected_field MUST include:
  - label: User-friendly question/title (no jargon, Title Case, specific)
  - help_text: One short sentence telling the user exactly what to enter (and why, if useful)
  - placeholder: Realistic example value, formatted for Trinidad & Tobago
- If the source text is vague (e.g., "number", "details"), rewrite into a precise label (e.g., "Vehicle Registration Number", "Describe the incident in 2–3 sentences").
- Do NOT leave labels/help_text/placeholder empty. Fill them with clear guidance.

Examples of good label/help_text/placeholder:
- Date of Birth: label="Date of Birth", help_text="Select your date of birth; your age is calculated automatically.", placeholder="1990-06-14"
- Full Name: label="Full Name", help_text="Enter your full legal name as on your ID.", placeholder="John Michael Smith"
- Address: label="Residential Address", help_text="Enter your current residential address in Trinidad & Tobago.", placeholder="15 Queen Street, Port of Spain"
- Electoral ID: label="Electoral ID (11 digits)", help_text="11 digits in YYYYMMDDXXX format (first 8 = date of birth).", placeholder="19741104044"
- Phone: label="Phone Number", help_text="Enter a Trinidad & Tobago phone number (868-XXX-XXXX).", placeholder="868-123-4567"
- Email: label="Email Address", help_text="Enter your email to receive updates.", placeholder="your.email@example.com"

**CRITICAL - SELECT FIELDS MUST INCLUDE OPTIONS:**
For fields with type "select", you MUST include the "options" array.

Example for month field:
{
    "id": "declaration_month",
    "label": "Declaration Month",
    "type": "select",
    "required": true,
    "placeholder": "Select month",
    "help_text": "Month when this declaration is made",
    "options": [
        {"value": "January", "label": "January"},
        {"value": "February", "label": "February"},
        {"value": "March", "label": "March"},
        {"value": "April", "label": "April"},
        {"value": "May", "label": "May"},
        {"value": "June", "label": "June"},
        {"value": "July", "label": "July"},
        {"value": "August", "label": "August"},
        {"value": "September", "label": "September"},
        {"value": "October", "label": "October"},
        {"value": "November", "label": "November"},
        {"value": "December", "label": "December"}
    ]
}

Respond ONLY with a valid JSON object in this exact structure:
{
    "template_html": "<html template with {{placeholders}} - MUST match example format exactly. Use {{calculated_age}} for age, NOT a direct age input>",
    "detected_fields": [
        {
            "id": "date_of_birth",
            "label": "Date of Birth",
            "type": "date",
            "required": true,
            "placeholder": "",
            "help_text": "Your age will be calculated automatically from your date of birth",
            "validation": {
                "max_date": "today",
                "date_constraint": "past_only",
                "message": "Date of birth cannot be in the future"
            }
        },
        {
            "id": "full_name",
            "label": "Full Name",
            "type": "text",
            "required": true,
            "placeholder": "John Michael Smith",
            "help_text": "Enter your full legal name as it appears on your ID",
            "validation": {
                "input_mode": "text_only",
                "message": "Name can only contain letters"
            }
        },
        {
            "id": "address",
            "label": "Current Address",
            "type": "text",
            "required": true,
            "placeholder": "15 Queen Street, Port of Spain",
            "help_text": "Your current residential address in Trinidad & Tobago"
        },
        {
            "id": "electoral_id",
            "label": "Electoral ID Number",
            "type": "text",
            "required": true,
            "placeholder": "19741104044",
            "help_text": "Your 11-digit Trinidad & Tobago Electoral ID (format: YYYYMMDDXXX)",
            "validation": {
                "pattern": "^\\d{11}$",
                "input_mode": "numeric",
                "min_length": 11,
                "max_length": 11,
                "message": "Enter valid Trinidad & Tobago Electoral ID (11 digits)"
            }
        },
        {
            "id": "phone_number",
            "label": "Phone Number",
            "type": "text",
            "required": false,
            "placeholder": "868-123-4567",
            "help_text": "Your contact number including area code"
        }
    ],
    "required_sections": ["Section Name 1", "Section Name 2"],
    "disallowed_phrases": ["phrase 1", "phrase 2"],
    "validation_rules": [
        {
            "field": "field_id",
            "rule": "required|format|min_length|max_length",
            "value": "rule value if applicable",
            "message": "Error message"
        }
    ],
    "few_shot_example": {
        "input": {"full_name": "John Smith", "date_of_birth": "1985-03-15", "ownership_type": "self"},
        "output_html": "<example output with {{calculated_age}} showing computed age>"
    },
    "analysis_notes": "Brief notes about the affidavit type structure and identified scenarios",
    "format_warnings": ["Any deviations from standard format noted in examples"],
    "field_conversions": ["age -> date_of_birth (age will be calculated from DOB)"],
    "identified_scenarios": ["List of all unique scenarios detected across the example documents"],
    "scenario_branches": {
        "scenario_id": {
            "description": "What this scenario covers",
            "template_sections": ["Which paragraphs/sections apply to this scenario"],
            "key_fields": ["field_ids that are unique to this scenario"]
        }
    }
}"""


def generate_policy_from_examples(
    html_examples: List[str],
    affidavit_type_name: str,
    additional_context: str = "",
    existing_questions: List[Dict] = None,
    affidavit_type_id: int = None,
) -> Dict:
    """
    Analyze example affidavit documents and generate policy configuration.
    When existing questions are provided, merges intelligently - only adding new
    fields and suggesting modifications to existing ones.
    
    Args:
        html_examples: List of HTML content from parsed documents
        affidavit_type_name: Name of the affidavit type being configured
        additional_context: Any additional instructions from admin
        existing_questions: Current intake_schema questions (if any)
    
    Returns:
        dict: {
            'success': bool,
            'policy': {generated policy object},
            'template_html': str,
            'detected_fields': list,
            'disallowed_phrases': list,
            'error': str (if success is False)
        }
    """
    if not html_examples:
        return {
            'success': False,
            'policy': {},
            'template_html': '',
            'detected_fields': [],
            'disallowed_phrases': [],
            'error': 'No example documents provided'
        }
    
    try:
        client = get_openai_client()
        
        # Build the examples section
        examples_text = ""
        for i, html in enumerate(html_examples, 1):
            # Truncate if too long (to fit in context window)
            truncated = html[:15000] if len(html) > 15000 else html
            examples_text += f"\n\n=== EXAMPLE DOCUMENT {i} ===\n{truncated}\n=== END DOCUMENT {i} ==="
        
        # Build existing questions context if provided
        existing_questions_context = ""
        if existing_questions:
            existing_questions_context = f"""

=== EXISTING INTAKE QUESTIONS (DO NOT DUPLICATE) ===
The following questions already exist. DO NOT create duplicates.
Only suggest NEW fields that are not already covered, or suggest modifications if a field needs updating.

{json.dumps(existing_questions, indent=2)}

=== END EXISTING QUESTIONS ===

IMPORTANT:
- Review existing questions above before generating detected_fields
- Only include fields in detected_fields that are NEW (not already in existing questions)
- If an existing question needs modification, note it in analysis_notes instead
- Do not duplicate fields - match by id, label, or similar purpose
"""

        # Inject reviewer feedback for policy/template
        policy_feedback_section = ""
        if affidavit_type_id:
            try:
                from .feedback_service import get_policy_feedback
                policy_feedback_section = get_policy_feedback(affidavit_type_id)
                if policy_feedback_section:
                    logger.info(f"Injected {len(policy_feedback_section)} chars of policy feedback")
            except Exception as e:
                logger.warning(f"Failed to load policy feedback: {e}")

        user_prompt = f"""Analyze these example affidavit documents for: "{affidavit_type_name}"
{f"Additional context: {additional_context}" if additional_context else ""}
{existing_questions_context}
{examples_text}
{policy_feedback_section}
Generate the policy configuration JSON as specified.
Remember: Only include NEW fields in detected_fields that are not already covered by existing questions."""

        response = call_openai_with_retry(
            client=client,
            model=settings.OPENAI_QA_MODEL,  # Use GPT-4o for analysis
            messages=[
                {"role": "system", "content": POLICY_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=8000,  # Increased for more comprehensive question generation
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        
        logger.info(f"Successfully generated policy for '{affidavit_type_name}'")
        
        return {
            'success': True,
            'policy': result,
            'template_html': result.get('template_html', ''),
            'detected_fields': result.get('detected_fields', []),
            'disallowed_phrases': result.get('disallowed_phrases', []),
            'required_sections': result.get('required_sections', []),
            'validation_rules': result.get('validation_rules', []),
            'few_shot_example': result.get('few_shot_example'),
            'analysis_notes': result.get('analysis_notes', ''),
            'scenario_mapping': result.get('scenario_mapping', {}),
            'identified_scenarios': result.get('identified_scenarios', []),
            'scenario_branches': result.get('scenario_branches', {}),
            'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
            'completion_tokens': response.usage.completion_tokens if response.usage else 0,
            'error': None
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse policy generation response as JSON: {e}")
        return {
            'success': False,
            'policy': {},
            'template_html': '',
            'detected_fields': [],
            'disallowed_phrases': [],
            'error': f'AI response was not valid JSON: {e}'
        }
    except Exception as e:
        logger.error(f"Error generating policy: {e}")
        return {
            'success': False,
            'policy': {},
            'template_html': '',
            'detected_fields': [],
            'disallowed_phrases': [],
            'error': str(e)
        }


def suggest_disallowed_phrases(existing_phrases: List[str] = None) -> List[str]:
    """
    Get a list of commonly disallowed phrases for legal affidavits.
    Combines AI suggestions with standard legal document guidelines.
    
    Args:
        existing_phrases: Already configured phrases to exclude from suggestions
    
    Returns:
        List of suggested phrases
    """
    standard_phrases = [
        # Uncertain language
        "I think",
        "I believe",
        "I guess",
        "maybe",
        "probably",
        "possibly",
        "might be",
        "could be",
        "approximately",
        "around",
        "about",
        "roughly",
        "more or less",
        
        # Informal language
        "gonna",
        "wanna",
        "gotta",
        "kinda",
        "sorta",
        "a lot",
        "stuff",
        "things",
        "etc",
        "and so on",
        
        # Vague references
        "someone",
        "somewhere",
        "sometime",
        "something",
        "somehow",
        
        # Emotional/subjective
        "I feel",
        "in my opinion",
        "personally",
        "to be honest",
        
        # Informal contractions in formal docs
        "don't",
        "won't",
        "can't",
        "shouldn't",
        "wouldn't",
        "couldn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
    ]
    
    if existing_phrases:
        # Filter out already configured phrases
        existing_lower = [p.lower() for p in existing_phrases]
        return [p for p in standard_phrases if p.lower() not in existing_lower]
    
    return standard_phrases


def enhance_policy_with_ai(
    existing_policy: Dict,
    feedback: str,
    affidavit_type_name: str
) -> Dict:
    """
    Use AI to enhance or refine an existing policy based on admin feedback.
    
    Args:
        existing_policy: Current policy_json
        feedback: Admin's feedback/instructions for improvement
        affidavit_type_name: Name of the affidavit type
    
    Returns:
        dict with enhanced policy
    """
    try:
        client = get_openai_client()
        
        system_prompt = """You are a legal document configuration expert.
You will be given an existing affidavit policy configuration and feedback from an admin.
Update the policy according to the feedback while preserving valid existing configuration.
Respond with the complete updated policy JSON."""

        user_prompt = f"""Affidavit Type: {affidavit_type_name}

Current Policy:
{json.dumps(existing_policy, indent=2)}

Admin Feedback:
{feedback}

Provide the updated policy JSON:"""

        response = call_openai_with_retry(
            client=client,
            model=settings.OPENAI_QA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=8000,  # Increased for more comprehensive question generation
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        enhanced_policy = json.loads(result_text)
        
        return {
            'success': True,
            'policy': enhanced_policy,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Error enhancing policy: {e}")
        return {
            'success': False,
            'policy': existing_policy,
            'error': str(e)
        }


def convert_detected_fields_to_intake_schema(detected_fields: List[Dict]) -> List[Dict]:
    """
    Convert AI-detected fields to the intake_schema format used by the system.
    
    Pipeline:
      1. Reconcile & normalize (alias resolution, dedup, enrich with defaults)
      2. Ensure universal fields are present (full_name, DOB, address, electoral_id)
      3. Apply smart T&T validation rules & age→DOB conversion
    
    Args:
        detected_fields: Fields detected by generate_policy_from_examples
    
    Returns:
        List of intake_schema question objects with validation
    """
    # --- Step 1: Normalize & deduplicate ---
    normalized = reconcile_and_normalize_fields(detected_fields)
    logger.info(f"[FIELD_PIPELINE] After normalize: {len(detected_fields or [])} → {len(normalized)} fields")

    # --- Step 2: Ensure universal fields ---
    normalized = ensure_universal_fields(normalized)
    logger.info(f"[FIELD_PIPELINE] After universal inject: {len(normalized)} fields")

    # --- Step 3: Original T&T validation + age→DOB conversion ---
    intake_schema = []
    has_dob_field = False  # Track if we already have a DOB field
    
    type_mapping = {
        'text': 'text',
        'textarea': 'textarea',
        'date': 'date',
        'select': 'select',
        'number': 'number',
        'email': 'email',
        'phone': 'phone',
        'tel': 'phone',
    }
    
    # First pass: check if DOB already exists
    for field in normalized:
        field_id = field.get('id', '').lower()
        if _matches_field_pattern(field_id, TT_FIELD_RULES.get('date_of_birth', {}).get('patterns', [])):
            has_dob_field = True
            break
    
    for field in normalized:
        field_id = field.get('id', '').lower()
        field_label = field.get('label', field.get('id', ''))
        
        # Check if this is an age field that should be converted to DOB
        age_rules = TT_FIELD_RULES.get('age', {})
        if _matches_field_pattern(field_id, age_rules.get('patterns', [])) or \
           _matches_field_pattern(field_label.lower(), age_rules.get('patterns', [])):
            
            # Only convert if we don't already have a DOB field
            if not has_dob_field and age_rules.get('convert_to_dob'):
                replacement = age_rules.get('replacement_field', {})
                question = {
                    'id': replacement.get('id', 'date_of_birth'),
                    'label': replacement.get('label', 'Date of Birth'),
                    'type': 'date',
                    'required': field.get('required', True),
                    'placeholder': '',
                    'help_text': replacement.get('help_text', 'Your age will be calculated automatically'),
                    'validation': replacement.get('validation', {
                        'max_date': 'today',
                        'date_constraint': 'past_only',
                        'message': 'Date of birth cannot be in the future'
                    }),
                    '_converted_from': 'age',  # Mark that this was converted from age
                    'type_locked': True  # Prevent frontend from overriding backend type
                }
                intake_schema.append(question)
                has_dob_field = True
                continue  # Skip adding the original age field
        
        question = {
            'id': field.get('id', ''),
            'label': field_label,
            'type': type_mapping.get(field.get('type', 'text'), 'text'),
            'required': field.get('required', True),
            'placeholder': field.get('placeholder', ''),
            'help_text': field.get('help_text', ''),
        }

        # Preserve show_if conditional logic from AI-detected fields
        if field.get('show_if'):
            question['show_if'] = field['show_if']
        
        # Start with any validation from the AI
        validation = field.get('validation', {})
        
        # Apply smart T&T validation rules based on field ID/label
        # NOTE: type_locked is NOT set yet so smart rules can fix AI type errors
        # (e.g. date_of_birth returned as 'text' will be corrected to 'date')
        validation = apply_smart_validation(field_id, field_label, validation, question)
        
        # Lock type AFTER smart validation has had a chance to correct it
        question['type_locked'] = True
        
        # Add validation if we have any rules
        if validation:
            question['validation'] = validation
        
        # Add options if it's a select type
        if question['type'] == 'select':
            if 'options' in field:
                question['options'] = field['options']
            # Check if we should add month options
            elif _matches_field_pattern(field_id, TT_FIELD_RULES.get('month', {}).get('patterns', [])):
                question['options'] = TT_FIELD_RULES['month']['options']
        
        intake_schema.append(question)
    
    return intake_schema


def post_process_template_for_computed_fields(template_html: str, intake_schema: List[Dict]) -> str:
    """
    Post-process AI-generated template to replace DOB placeholders with
    {{calculated_age}} when the corresponding field has computed_fields: ['age'].
    
    The AI sometimes generates "born on {{date_of_birth}}" but when the DOB field
    is configured to auto-compute age, the template should show the computed age
    instead (e.g., "aged {{calculated_age}} years").
    
    Common patterns replaced:
      - "born on {{date_of_birth}}"  →  "aged {{calculated_age}} years"
      - "born {{date_of_birth}}"     →  "aged {{calculated_age}} years"
      - "date of birth {{date_of_birth}}" → "aged {{calculated_age}} years"
    
    Standalone {{date_of_birth}} that don't match these patterns are left as-is.
    
    Args:
        template_html: The AI-generated HTML template
        intake_schema: The processed intake schema with computed_fields
    
    Returns:
        Template HTML with DOB→age substitutions applied where appropriate
    """
    if not template_html or not intake_schema:
        return template_html or ''
    
    # Find DOB fields that have computed_fields: ['age']
    dob_field_ids = set()
    for q in intake_schema:
        if 'age' in (q.get('computed_fields') or []):
            dob_field_ids.add(q.get('id', ''))
    
    if not dob_field_ids:
        return template_html
    
    result = template_html
    for field_id in dob_field_ids:
        placeholder = '{{' + field_id + '}}'
        if placeholder not in result:
            continue
        
        # Replace "born on {{field_id}}" or "born {{field_id}}" patterns
        # These are the common AI-generated phrases we want to swap
        import re as _re
        patterns = [
            # "born on {{date_of_birth}}" → "aged {{calculated_age}} years"
            (_re.compile(r'born\s+on\s+\{\{' + _re.escape(field_id) + r'\}\}', _re.IGNORECASE),
             'aged {{calculated_age}} years'),
            # ", born {{date_of_birth}}" → ", aged {{calculated_age}} years"
            (_re.compile(r'born\s+\{\{' + _re.escape(field_id) + r'\}\}', _re.IGNORECASE),
             'aged {{calculated_age}} years'),
            # "date of birth {{date_of_birth}}" → "aged {{calculated_age}} years"
            (_re.compile(r'date\s+of\s+birth\s+\{\{' + _re.escape(field_id) + r'\}\}', _re.IGNORECASE),
             'aged {{calculated_age}} years'),
            # "date of birth: {{date_of_birth}}" → "age: {{calculated_age}} years"
            (_re.compile(r'date\s+of\s+birth\s*:\s*\{\{' + _re.escape(field_id) + r'\}\}', _re.IGNORECASE),
             'age: {{calculated_age}} years'),
        ]
        
        for pattern, replacement in patterns:
            result, count = pattern.subn(replacement, result)
            if count > 0:
                logger.info(f"[TEMPLATE_POSTPROCESS] Replaced DOB placeholder '{field_id}' with calculated_age")
                break
        else:
            # No phrase pattern matched — do a standalone swap
            # This catches {{date_of_birth}} appearing on its own
            result = result.replace(placeholder, '{{calculated_age}}')
            logger.info(f"[TEMPLATE_POSTPROCESS] Standalone DOB placeholder '{field_id}' → calculated_age")
    
    return result


def apply_smart_validation(field_id: str, field_label: str, existing_validation: Dict, question: Dict) -> Dict:
    """
    Apply smart Trinidad & Tobago specific validation based on field detection.

    IMPORTANT — declaration date parts (declaration_city, declaration_day, declaration_month,
    declaration_year) are handled with an early-return guard that:
      - Applies only their own format/input_mode rules
      - Strips ANY date constraints (max_date, date_constraint, check_future_date, etc.)
    This ensures users can freely enter any date when making their declaration.

    Args:
        field_id: The field ID (snake_case)
        field_label: Human readable label
        existing_validation: Any validation already set by AI
        question: The question dict (may be modified for type changes)

    Returns:
        Enhanced validation dict (declaration fields never get date constraints)
    """
    validation = existing_validation.copy() if existing_validation else {}
    field_lower = field_id.lower()
    label_lower = field_label.lower()

    # ── Declaration date parts: early-return guard ────────────────────────────
    # These fields use the declaration_ prefix specifically so we can detect them
    # here and skip the generic city/day/month/year rules (which would add
    # check_future_date or max_year_current via substring matching).
    DECLARATION_PARTS = {'declaration_city', 'declaration_day', 'declaration_month', 'declaration_year'}
    if field_lower in DECLARATION_PARTS:
        rule = TT_FIELD_RULES.get(field_lower, {})
        rule_validation = rule.get('validation', {})
        # Merge only format/input rules (existing takes precedence)
        for key, value in rule_validation.items():
            if key not in validation:
                validation[key] = value
        # Apply type / placeholder / options
        if 'type' in rule and not question.get('type_locked', False):
            if question.get('type', 'text') == 'text':
                question['type'] = rule['type']
        if not question.get('placeholder') and rule.get('placeholder'):
            question['placeholder'] = rule['placeholder']
        if rule.get('options'):
            question['options'] = rule['options']
        # Strip any date constraints — declaration dates are free choice
        for k in ('max_date', 'min_date', 'date_constraint', 'check_future_date', 'max_year_current'):
            validation.pop(k, None)
        logger.debug(f"[SMART_VALIDATION] '{field_id}' is a declaration part — date constraints excluded")
        return validation
    # ── End declaration guard ─────────────────────────────────────────────────

    # Check each rule set
    for rule_key, rules in TT_FIELD_RULES.items():
        patterns = rules.get('patterns', [])
        
        if _matches_field_pattern(field_lower, patterns) or _matches_field_pattern(label_lower, patterns):
            # Get the validation rules for this field type
            rule_validation = rules.get('validation', {})
            
            # Merge validations (existing takes precedence, but fill gaps)
            for key, value in rule_validation.items():
                if key not in validation:
                    validation[key] = value
            
            # Update field type if specified AND not locked by backend
            if 'type' in rules and not question.get('type_locked', False):
                current_type = question.get('type', 'text')
                # Only override if it's the default 'text' type (not user-set)
                if current_type == 'text':
                    new_type = rules['type']
                    if new_type in ['date', 'number', 'email', 'phone', 'select']:
                        question['type'] = new_type
            
            # Add placeholder if not set
            if not question.get('placeholder') and 'placeholder' in rules:
                question['placeholder'] = rules['placeholder']
            
            # Add options for select type
            if 'options' in rules:
                question['options'] = rules['options']
            
            # Handle field conversion suggestions (e.g., age -> date_of_birth)
            if 'convert_to' in rules:
                validation['_conversion_suggestion'] = rules['convert_to']
            
            # Transfer computed_fields metadata to the question for frontend logic
            if 'computed_fields' in rules:
                question['computed_fields'] = rules['computed_fields']

            break  # Use first matching rule
    
    return validation


def _matches_field_pattern(field_name: str, patterns: List[str]) -> bool:
    """
    Check if a field name matches any of the given patterns.
    
    Args:
        field_name: The field name to check (already lowercase)
        patterns: List of patterns to match against
    
    Returns:
        True if matches any pattern
    """
    field_name = field_name.lower().replace(' ', '_').replace('-', '_')
    
    for pattern in patterns:
        pattern_lower = pattern.lower()
        # Exact match
        if field_name == pattern_lower:
            return True
        # Contains match
        if pattern_lower in field_name:
            return True
        # Field contains pattern
        if field_name in pattern_lower:
            return True
    
    return False


def enhance_field_with_tt_rules(field: Dict) -> Dict:
    """
    Enhance a single field with Trinidad & Tobago specific validation rules.
    Can be used to upgrade existing intake_schema fields.
    
    Args:
        field: An intake_schema question dict
    
    Returns:
        Enhanced field dict with validation
    """
    field_id = field.get('id', '').lower()
    field_label = field.get('label', '').lower()
    
    # Debug logging for property_age field
    if 'property_age' in field_id or 'property age' in field_label.lower():
        logger.info(f"DEBUG: Processing property_age field - ID: {field_id}, Label: {field_label}, Type: {field.get('type')}, type_locked: {field.get('type_locked')}")
    
    # Preserve admin-defined type before smart rules run
    if 'type_locked' not in field:
        field['type_locked'] = True
    
    # Get or create validation
    validation = field.get('validation', {})
    
    # Apply smart validation (respect type_locked flag)
    validation = apply_smart_validation(field_id, field_label, validation, field)
    
    # Debug logging after smart validation
    if 'property_age' in field_id or 'property age' in field_label.lower():
        logger.info(f"DEBUG: After smart validation - Type: {field.get('type')}, type_locked: {field.get('type_locked')}")
    
    # Add validation if we have any
    if validation:
        field['validation'] = validation
    
    return field


def upgrade_intake_schema_with_tt_validation(intake_schema: List[Dict]) -> List[Dict]:
    """
    Upgrade an existing intake_schema with Trinidad & Tobago validation rules.
    Use this to enhance existing affidavit types with smart validation.
    
    Args:
        intake_schema: Existing list of intake questions
    
    Returns:
        Enhanced intake_schema with validation rules
    """
    return [enhance_field_with_tt_rules(field.copy()) for field in intake_schema]


def build_policy_json_from_generation(generation_result: Dict) -> Dict:
    """
    Build a complete policy_json from the AI generation result.
    
    Args:
        generation_result: Result from generate_policy_from_examples
    
    Returns:
        Complete policy_json ready to save to AffidavitType
    """
    if not generation_result.get('success'):
        return {}
    
    detected_fields = _apply_field_defaults(generation_result.get('detected_fields', []))

    policy = {
        'required_sections': generation_result.get('required_sections', []),
        'validation_rules': generation_result.get('validation_rules', []),
        'few_shot_examples': [],
        'detected_fields': detected_fields,
    }
    
    # Add few-shot example if generated
    few_shot = generation_result.get('few_shot_example')
    if few_shot:
        policy['few_shot_examples'].append(few_shot)
    
    return policy


# ---------------------------------------------------------------------------
# Helpers: enforce label/help_text/placeholder defaults for detected_fields
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FIELD_ALIASES: maps variant IDs to a canonical ID for deduplication
# ---------------------------------------------------------------------------
FIELD_ALIASES = {
    # name variants → full_name
    'name': 'full_name', 'deponent_name': 'full_name', 'applicant_name': 'full_name',
    'declarant_name': 'full_name', 'your_name': 'full_name', 'person_name': 'full_name',
    'first_name': 'full_name', 'surname': 'full_name',
    # address variants → address
    'residential_address': 'address', 'home_address': 'address',
    'current_address': 'address', 'street_address': 'address',
    'mailing_address': 'address',
    # ID variants → electoral_id
    'national_id': 'electoral_id', 'id_number': 'electoral_id',
    'identification_number': 'electoral_id', 'id_card': 'electoral_id',
    'eic': 'electoral_id', 'eic_number': 'electoral_id',
    # DOB variants → date_of_birth
    'dob': 'date_of_birth', 'birth_date': 'date_of_birth', 'birthdate': 'date_of_birth',
    # age variants (will be converted to DOB)
    'age': 'date_of_birth', 'years_old': 'date_of_birth', 'current_age': 'date_of_birth',
    # phone variants → phone_number
    'phone': 'phone_number', 'telephone': 'phone_number', 'mobile': 'phone_number',
    'contact_number': 'phone_number', 'cell': 'phone_number',
    # occupation variants → occupation
    'profession': 'occupation', 'job': 'occupation', 'employment': 'occupation',
    # city/town variants → city
    'town': 'city', 'village': 'city', 'municipality': 'city', 'location': 'city',
    # --- declaration date part variants → declaration_* ---
    # These are used for the "Declared at [City] this [Day] day of [Month] [Year]" block
    'declaration_at_city': 'declaration_city', 'declared_at_city': 'declaration_city',
    'signing_city': 'declaration_city', 'sworn_city': 'declaration_city',
    'sworn_day': 'declaration_day', 'signing_day': 'declaration_day',
    'declared_day': 'declaration_day', 'date_day': 'declaration_day',
    'sworn_month': 'declaration_month', 'signing_month': 'declaration_month',
    'declared_month': 'declaration_month', 'date_month': 'declaration_month',
    'sworn_year': 'declaration_year', 'signing_year': 'declaration_year',
    'declared_year': 'declaration_year', 'date_year': 'declaration_year',
}

# ---------------------------------------------------------------------------
# UNIVERSAL_FIELDS: fields that should appear in EVERY T&T affidavit
# They are injected if missing after AI field detection.
# ---------------------------------------------------------------------------
UNIVERSAL_FIELDS = [
    {
        'id': 'full_name', 'label': 'Full Name', 'type': 'text', 'required': True,
        'placeholder': 'John Michael Smith',
        'help_text': 'Enter your full legal name as on your ID.',
        'type_locked': True,
        'validation': {'input_mode': 'text_only', 'min_length': 2, 'max_length': 100,
                        'message': 'Name can only contain letters, spaces, hyphens, and apostrophes'},
    },
    {
        'id': 'date_of_birth', 'label': 'Date of Birth', 'type': 'date', 'required': True,
        'placeholder': '1990-06-14',
        'help_text': 'Select your date of birth; your age is calculated automatically.',
        'type_locked': True,
        'validation': {'max_date': 'today', 'date_constraint': 'past_only',
                        'message': 'Date of birth cannot be in the future'},
    },
    {
        'id': 'address', 'label': 'Current Address', 'type': 'textarea', 'required': True,
        'placeholder': '15 Queen Street, Port of Spain',
        'help_text': 'Enter your current residential address in Trinidad & Tobago.',
        'type_locked': True,
        'validation': {'min_length': 5, 'max_length': 200,
                        'message': 'Please enter a valid address'},
    },
    {
        'id': 'electoral_id', 'label': 'Electoral ID Number', 'type': 'text', 'required': True,
        'placeholder': '19741104044',
        'help_text': 'Your 11-digit Trinidad & Tobago Electoral ID (format: YYYYMMDDXXX).',
        'type_locked': True,
        'validation': {'pattern': r'^\d{11}$', 'input_mode': 'numeric',
                        'min_length': 11, 'max_length': 11,
                        'message': 'Enter valid Electoral ID (exactly 11 digits)'},
    },
]

# ---------------------------------------------------------------------------
# FIELD_DEFAULTS: proper label/help_text/placeholder for known fields
# ---------------------------------------------------------------------------
FIELD_DEFAULTS = {
    # --- core identity ---
    'full_name': {
        'label': 'Full Name', 'placeholder': 'John Michael Smith',
        'help_text': 'Enter your full legal name as on your ID.', 'type': 'text',
    },
    'date_of_birth': {
        'label': 'Date of Birth', 'placeholder': '1990-06-14',
        'help_text': 'Select your date of birth; your age is calculated automatically.', 'type': 'date',
    },
    'address': {
        'label': 'Current Address', 'placeholder': '15 Queen Street, Port of Spain',
        'help_text': 'Enter your current residential address in Trinidad & Tobago.', 'type': 'textarea',
    },
    'electoral_id': {
        'label': 'Electoral ID (11 digits)', 'placeholder': '19741104044',
        'help_text': '11 digits in YYYYMMDDXXX format (first 8 = date of birth).', 'type': 'text',
    },
    'national_id': {
        'label': 'National ID (11 digits)', 'placeholder': '19741104044',
        'help_text': '11 digits in YYYYMMDDXXX format (first 8 = date of birth).', 'type': 'text',
    },
    'occupation': {
        'label': 'Occupation', 'placeholder': 'Teacher',
        'help_text': 'Enter your current occupation or profession.', 'type': 'text',
    },
    'city': {
        'label': 'City / Town', 'placeholder': 'Port of Spain',
        'help_text': 'Enter your city or town in Trinidad & Tobago.', 'type': 'text',
    },
    # --- contact ---
    'phone_number': {
        'label': 'Phone Number', 'placeholder': '868-123-4567',
        'help_text': 'Enter a Trinidad & Tobago phone number (868-XXX-XXXX).', 'type': 'text',
    },
    'email': {
        'label': 'Email Address', 'placeholder': 'your.email@example.com',
        'help_text': 'Enter your email to receive updates.', 'type': 'email',
    },
    # --- ID documents ---
    'passport': {
        'label': 'Passport Number', 'placeholder': 'TB1234567',
        'help_text': '2 letters + 7 digits (e.g., TB1234567).', 'type': 'text',
    },
    'drivers_permit': {
        'label': "Driver's Permit Number", 'placeholder': 'DL123456',
        'help_text': "Enter the number from your driver's permit.", 'type': 'text',
    },
    # --- declaration date parts (no date validation — user-chosen date) ---
    'declaration_city': {
        'label': 'Declaration City', 'placeholder': 'Port of Spain',
        'help_text': 'City where the declaration is being made.', 'type': 'text',
    },
    'declaration_day': {
        'label': 'Declaration Day', 'placeholder': '14',
        'help_text': 'Day of the month this declaration is made (1–31).', 'type': 'number',
    },
    'declaration_month': {
        'label': 'Declaration Month', 'placeholder': 'February',
        'help_text': 'Month this declaration is made.', 'type': 'select',
    },
    'declaration_year': {
        'label': 'Declaration Year', 'placeholder': '2026',
        'help_text': 'Year this declaration is made (4-digit).', 'type': 'number',
    },
    # --- property / land ---
    'property_description': {
        'label': 'Property Description', 'placeholder': 'Lot 14, LP No. 52, Chaguanas',
        'help_text': 'Describe the property including lot number, plan number, and area.', 'type': 'textarea',
    },
    'land_ownership_details': {
        'label': 'Land Ownership Details', 'placeholder': 'Deed of Conveyance registered as...',
        'help_text': 'Enter details about how you acquired or own the land.', 'type': 'textarea',
    },
    'years_residing': {
        'label': 'Years Residing', 'placeholder': '15',
        'help_text': 'How many years have you lived at this address?', 'type': 'number',
    },
    'additional_property_details': {
        'label': 'Additional Property Details', 'placeholder': 'Bounded on the north by...',
        'help_text': 'Any extra property details (boundaries, dimensions, etc.).', 'type': 'textarea',
    },
    'ownership_disclaimer': {
        'label': 'Ownership Disclaimer', 'placeholder': 'I am the sole owner...',
        'help_text': 'Statement about your ownership status.', 'type': 'textarea',
    },
    # --- relationships / witnesses ---
    'witness_name': {
        'label': 'Witness Name', 'placeholder': 'Jane Marie Williams',
        'help_text': 'Full legal name of the witness.', 'type': 'text',
    },
    'relationship': {
        'label': 'Relationship to Deponent', 'placeholder': 'Mother',
        'help_text': 'Relationship between you and the other party (if applicable).', 'type': 'text',
    },
    # --- incident / event ---
    'incident_date': {
        'label': 'Date of Incident', 'placeholder': '2025-01-15',
        'help_text': 'When did the incident or event occur?', 'type': 'date',
    },
    'incident_description': {
        'label': 'Incident Description', 'placeholder': 'On the said date, I was...',
        'help_text': 'Describe the incident in detail.', 'type': 'textarea',
    },
    'reason': {
        'label': 'Reason / Purpose', 'placeholder': 'To establish ownership of...',
        'help_text': 'Why is this affidavit being made?', 'type': 'textarea',
    },
}


def _title_from_id(field_id: str) -> str:
    return ' '.join(part.capitalize() for part in field_id.split('_')) if field_id else ''


_DECLARATION_IDS = {k for k in FIELD_DEFAULTS if k.startswith('declaration_')}


def _apply_field_defaults(detected_fields: List[Dict]) -> List[Dict]:
    updated = []
    for field in detected_fields or []:
        field_id = field.get('id', '').strip()

        # Label-based declaration ID fix: if label says "Declaration Day" but id is "day",
        # promote id → "declaration_day" so it matches the template placeholder.
        if field_id and not field_id.startswith('declaration_'):
            label_lower = field.get('label', '').lower()
            if 'declaration' in label_lower:
                suffix = label_lower.replace('declaration', '').strip().replace(' ', '_').strip('_')
                proposed = f'declaration_{suffix}'
                if proposed in _DECLARATION_IDS:
                    field['id'] = proposed
                    field['field_name'] = proposed
                    field_id = proposed

        defaults = FIELD_DEFAULTS.get(field_id, {})

        # Label
        if not field.get('label'):
            field['label'] = defaults.get('label') or _title_from_id(field_id)

        # Placeholder
        if not field.get('placeholder'):
            field['placeholder'] = defaults.get('placeholder') or ''

        # Help text
        if not field.get('help_text'):
            # If type-specific default exists
            if defaults.get('help_text'):
                field['help_text'] = defaults['help_text']
            else:
                # Generic fallback
                field['help_text'] = f"Enter your {field.get('label', field_id).lower()}."

        # Type override from defaults (only if field has generic 'text' type)
        if field.get('type', 'text') == 'text' and defaults.get('type') and defaults['type'] != 'text':
            if not field.get('type_locked'):
                field['type'] = defaults['type']

        updated.append(field)

    return updated


def reconcile_and_normalize_fields(detected_fields: List[Dict]) -> List[Dict]:
    """
    Normalize AI-detected fields:
      1. Canonical ID via FIELD_ALIASES (e.g., 'name' → 'full_name')
      2. Dedup by canonical ID (first occurrence wins, merge labels)
      3. Enrich with FIELD_DEFAULTS
    """
    seen: Dict[str, Dict] = {}  # canonical_id → best field dict
    order = []  # preserve insertion order

    for field in detected_fields or []:
        raw_id = field.get('id', '').strip().lower().replace(' ', '_').replace('-', '_')
        if not raw_id:
            continue

        # Resolve canonical ID via alias table
        canonical = FIELD_ALIASES.get(raw_id, raw_id)

        # Label-based declaration ID promotion:
        # If AI gave id="day" / label="Declaration Day", promote to "declaration_day".
        # This fixes the case where the alias table doesn't have the exact variant.
        if not canonical.startswith('declaration_'):
            label_lower = field.get('label', '').lower()
            if 'declaration' in label_lower:
                # Derive suffix from the label: "declaration day" → "day" → "declaration_day"
                suffix = label_lower.replace('declaration', '').strip().replace(' ', '_').strip('_')
                proposed = f'declaration_{suffix}'
                if proposed in _DECLARATION_IDS:
                    canonical = proposed

        if canonical in seen:
            # Merge: keep richer label / help_text
            existing = seen[canonical]
            if not existing.get('help_text') and field.get('help_text'):
                existing['help_text'] = field['help_text']
            if not existing.get('placeholder') and field.get('placeholder'):
                existing['placeholder'] = field['placeholder']
            # Merge validation keys
            ev = existing.get('validation', {})
            fv = field.get('validation', {})
            for k, v in fv.items():
                if k not in ev:
                    ev[k] = v
            if ev:
                existing['validation'] = ev
        else:
            field['id'] = canonical          # rename to canonical
            field['field_name'] = canonical  # keep template placeholder in sync
            # Pull defaults
            defaults = FIELD_DEFAULTS.get(canonical, {})
            if not field.get('label'):
                field['label'] = defaults.get('label') or _title_from_id(canonical)
            if not field.get('placeholder'):
                field['placeholder'] = defaults.get('placeholder') or ''
            if not field.get('help_text'):
                field['help_text'] = defaults.get('help_text') or f"Enter your {field.get('label', canonical).lower()}."
            # Type from defaults if still generic text
            if field.get('type', 'text') == 'text' and defaults.get('type') and defaults['type'] != 'text':
                if not field.get('type_locked'):
                    field['type'] = defaults['type']

            # Preserve show_if — update the parent field reference to use
            # canonical ID if the parent was also aliased
            if field.get('show_if'):
                parent_ref = field['show_if'].get('field', '')
                parent_canonical = FIELD_ALIASES.get(
                    parent_ref.lower().replace(' ', '_').replace('-', '_'),
                    parent_ref,
                )
                field['show_if']['field'] = parent_canonical

            seen[canonical] = field
            order.append(canonical)

    return [seen[cid] for cid in order]


def ensure_universal_fields(fields: List[Dict]) -> List[Dict]:
    """
    Inject UNIVERSAL_FIELDS that are missing from the detected list.
    Universal fields are placed at the top of the list for consistent UX.
    """
    existing_ids = {f.get('id', '').lower() for f in fields}

    inject = []
    for uf in UNIVERSAL_FIELDS:
        if uf['id'].lower() not in existing_ids:
            inject.append(uf.copy())
            logger.info(f"[ENSURE_UNIVERSAL] Injected missing universal field: {uf['id']}")

    # Universal fields first, then rest
    if inject:
        return inject + fields
    return fields


def auto_generate_placeholder_mapping(
    template_html: str,
    intake_schema: List[Dict],
) -> Dict[str, str]:
    """
    Deterministically build a placeholder_mapping from template_html + intake_schema.
    For each {{placeholder}} in the template, tries to match to a question by:
      1. Exact ID match
      2. Alias resolution  (FIELD_ALIASES)
      3. Fuzzy label match (placeholder words ⊂ label words)
    """
    if not template_html:
        return {}

    placeholders = list(dict.fromkeys(re.findall(r'\{\{(\w+)\}\}', template_html)))
    q_by_id = {}
    q_by_alias = {}
    for q in intake_schema or []:
        qid = q.get('id', '')
        q_by_id[qid.lower()] = qid
        # Also register canonical alias → qid
        canonical = FIELD_ALIASES.get(qid.lower(), qid.lower())
        q_by_alias[canonical] = qid

    # Auto-computed placeholders that never need a question
    auto_computed = {'calculated_age', 'current_date', 'current_year', 'current_month', 'current_day'}

    mapping = {}
    for ph in placeholders:
        ph_lower = ph.lower()
        if ph_lower in auto_computed:
            continue

        # 1. Exact match
        if ph_lower in q_by_id:
            mapping[ph] = q_by_id[ph_lower]
            continue

        # 2. Alias resolution
        canonical = FIELD_ALIASES.get(ph_lower, ph_lower)
        if canonical in q_by_id:
            mapping[ph] = q_by_id[canonical]
            continue
        if canonical in q_by_alias:
            mapping[ph] = q_by_alias[canonical]
            continue

        # 3. Fuzzy: placeholder words ⊂ question label words
        ph_words = set(ph_lower.split('_'))
        for q in intake_schema or []:
            label_words = set(q.get('label', '').lower().replace('-', ' ').split())
            if ph_words and ph_words.issubset(label_words):
                mapping[ph] = q.get('id', '')
                break

    logger.info(f"[AUTO_MAPPING] Generated mapping for {len(mapping)}/{len(placeholders)} placeholders")
    return mapping


# ---------------------------------------------------------------------------
# P0: Validate template ↔ intake_schema mapping completeness
# ---------------------------------------------------------------------------

def validate_template_mapping(
    template_html: str,
    intake_schema: List[Dict],
    placeholder_mapping: Dict[str, str],
) -> Dict:
    """
    Validate that every {{placeholder}} in the template has a matching intake
    question and that no intake questions are orphaned (unused by any placeholder).

    Returns a report dict:
    {
        'valid': bool,
        'errors': [{'type': 'unmapped_placeholder', 'placeholder': str, 'message': str}, ...],
        'warnings': [{'type': 'orphaned_question', 'question_id': str, 'label': str, 'message': str}, ...],
        'info': {'total_placeholders': int, 'mapped': int, 'auto_computed': int, 'unmapped': int, 'orphaned': int},
    }
    """
    AUTO_COMPUTED = {'calculated_age', 'current_date', 'current_year', 'current_month', 'current_day'}
    # Questions that feed an auto-computed field — exempt from orphan flagging
    # when their derived auto placeholder appears in the template.
    AUTO_COMPUTED_SOURCES = {'calculated_age': 'date_of_birth'}

    placeholders = list(dict.fromkeys(re.findall(r'\{\{(\w+)\}\}', template_html or '')))
    q_ids = {q.get('id', '') for q in (intake_schema or [])}
    mapping = placeholder_mapping or {}

    errors: List[Dict] = []
    warnings: List[Dict] = []

    mapped_question_ids: set = set()
    mapped_count = 0
    auto_count = 0
    unmapped_count = 0

    for ph in placeholders:
        ph_lower = ph.lower()
        if ph_lower in AUTO_COMPUTED or ph in AUTO_COMPUTED:
            auto_count += 1
            continue

        # Resolve via explicit mapping → same-name question → alias
        resolved_qid = mapping.get(ph, '')
        if not resolved_qid and ph in q_ids:
            resolved_qid = ph
        if not resolved_qid:
            canonical = FIELD_ALIASES.get(ph_lower, ph_lower)
            if canonical in q_ids:
                resolved_qid = canonical

        if resolved_qid and resolved_qid in q_ids:
            mapped_count += 1
            mapped_question_ids.add(resolved_qid)
        else:
            unmapped_count += 1
            errors.append({
                'type': 'unmapped_placeholder',
                'placeholder': ph,
                'message': f'Template placeholder "{{{{{ph}}}}}" has no matching intake question.',
            })

    # Orphaned questions (exist in intake_schema but unused by any placeholder)
    orphaned = []
    for q in (intake_schema or []):
        qid = q.get('id', '')
        if qid and qid not in mapped_question_ids:
            # Also check if any placeholder directly matches this qid
            if qid not in placeholders and qid not in {mapping.get(p) for p in placeholders}:
                # Exempt source questions whose auto-computed derivative is used in the template
                is_auto_source = any(
                    auto_ph in placeholders and source_qid == qid
                    for auto_ph, source_qid in AUTO_COMPUTED_SOURCES.items()
                )
                if not is_auto_source:
                    orphaned.append(qid)
                    warnings.append({
                        'type': 'orphaned_question',
                        'question_id': qid,
                        'label': q.get('label', ''),
                        'message': f'Intake question "{qid}" ({q.get("label", "")}) is not used by any template placeholder.',
                    })

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'info': {
            'total_placeholders': len(placeholders),
            'mapped': mapped_count,
            'auto_computed': auto_count,
            'unmapped': unmapped_count,
            'orphaned': len(orphaned),
        },
    }


# ---------------------------------------------------------------------------
# P1: Sync placeholder_mapping when questions are added/removed/renamed
# ---------------------------------------------------------------------------

def sync_mapping_after_question_change(
    template_html: str,
    old_schema: List[Dict],
    new_schema: List[Dict],
    placeholder_mapping: Dict[str, str],
) -> Dict[str, str]:
    """
    After an admin edits intake_schema, reconcile placeholder_mapping:

    1. Remove mappings that point to deleted question IDs.
    2. If a deleted question ID appears as a placeholder in the template and
       there is a new question with a matching alias, re-map automatically.
    3. Auto-map any NEW questions to matching template placeholders.

    Returns updated placeholder_mapping (never mutates the original).
    """
    mapping = dict(placeholder_mapping or {})
    old_ids = {q.get('id', '') for q in (old_schema or [])}
    new_ids = {q.get('id', '') for q in (new_schema or [])}

    deleted_ids = old_ids - new_ids
    added_ids = new_ids - old_ids

    # Step 1: purge mappings that point to deleted questions
    stale_keys = [ph for ph, qid in mapping.items() if qid in deleted_ids]
    for key in stale_keys:
        del mapping[key]
        logger.info(f"[SYNC_MAPPING] Removed stale mapping: {key} → (deleted question)")

    # Step 2: for deleted question keys, try to re-resolve via alias to a new question
    placeholders = list(dict.fromkeys(re.findall(r'\{\{(\w+)\}\}', template_html or '')))
    for ph in placeholders:
        if ph in mapping:
            continue  # already mapped
        ph_lower = ph.lower()
        # Try alias resolution first
        canonical = FIELD_ALIASES.get(ph_lower, ph_lower)
        if canonical in new_ids:
            mapping[ph] = canonical
            logger.info(f"[SYNC_MAPPING] Re-mapped {ph} → {canonical} via alias")
            continue
        # Try exact match
        if ph in new_ids:
            mapping[ph] = ph
            logger.info(f"[SYNC_MAPPING] Auto-mapped {ph} → {ph} (exact)")
            continue

    # Step 3: auto-map added questions to matching template placeholders
    for qid in added_ids:
        qid_lower = qid.lower()
        for ph in placeholders:
            if ph in mapping:
                continue
            ph_lower = ph.lower()
            if ph_lower == qid_lower:
                mapping[ph] = qid
                logger.info(f"[SYNC_MAPPING] Auto-mapped new question {qid} → placeholder {ph}")
                break
            canonical = FIELD_ALIASES.get(ph_lower, ph_lower)
            if canonical == qid_lower:
                mapping[ph] = qid
                logger.info(f"[SYNC_MAPPING] Auto-mapped new question {qid} → placeholder {ph} (alias)")
                break

    return mapping


# ---------------------------------------------------------------------------
# P4: Convert AI-generated identified_scenarios to structured scenario_library
# ---------------------------------------------------------------------------

def build_scenarios_from_identified(
    identified_scenarios: List[str],
    existing_library: List[Dict],
) -> List[Dict]:
    """
    Convert the flat list of scenario names returned by the AI policy generator
    into structured scenario_library entries that `detect_scenario()` can use.

    For each scenario string the AI identified:
      - Derive an ID (snake_case)
      - Extract keywords from the scenario name
      - Skip if a scenario with the same ID already exists

    Returns a NEW list with existing entries preserved + new ones appended.
    """
    if not identified_scenarios:
        return existing_library or []

    library = list(existing_library or [])
    existing_ids = {s.get('id', '').lower() for s in library}

    for scenario_name in identified_scenarios:
        if not isinstance(scenario_name, str) or not scenario_name.strip():
            continue

        # Derive a stable ID
        scenario_id = re.sub(r'[^a-z0-9]+', '_', scenario_name.lower()).strip('_')
        if not scenario_id or scenario_id in existing_ids:
            continue

        # Extract keywords from the scenario description (words >= 3 chars)
        words = re.findall(r'[a-z]{3,}', scenario_name.lower())
        # Remove very common words
        stop_words = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'are', 'was', 'has', 'have', 'not'}
        keywords = [w for w in words if w not in stop_words]

        entry = {
            'id': scenario_id,
            'name': scenario_name.strip(),
            'keywords': keywords,
            'patterns': [],
            'keyword_threshold': max(1, len(keywords) // 2),
            'drafting_instructions': '',
            '_auto_generated': True,
        }
        library.append(entry)
        existing_ids.add(scenario_id)
        logger.info(f"[SCENARIO_BUILDER] Created scenario entry '{scenario_id}' with {len(keywords)} keywords")

    return library
