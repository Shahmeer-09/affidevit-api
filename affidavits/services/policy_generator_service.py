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
        'patterns': ['date_of_birth', 'dob', 'birth_date', 'birthdate'],
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
    # Declaration date - cannot be in future
    'declaration_date': {
        'patterns': ['declaration_date', 'sworn_date', 'dated', 'date_declared', 'affirmed_date'],
        'validation': {
            'max_date': 'today',
            'date_constraint': 'past_or_today',
            'message': 'Declaration date cannot be in the future'
        },
        'type': 'date'
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
    # Month names - with future date check for declaration contexts
    'month': {
        'patterns': ['month', 'month_name', 'declaration_month'],
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
    # Year - numeric only, prevent future years for declarations
    'year': {
        'patterns': ['year', 'year_of', 'declaration_year'],
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
    # Day - numeric only, with future date check for declaration contexts
    'day': {
        'patterns': ['day', 'day_of', 'declaration_day'],
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

**DATE FIELDS:**
- Declaration date: MUST have max_date: "today" (cannot be in future)
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

**IMPORTANT - GENERATE ALL FIELDS AS SIMPLE, FLAT LIST:**
- Create one field for EACH piece of information that appears in ANY of the example documents
- Do NOT use conditional logic (show_if) - create ALL fields as regular required/optional fields
- Users will fill in the fields that apply to their situation
- Empty/unused fields will be handled gracefully by the AI drafter
- This ensures we capture ALL possible information needs across all document variations
- The more fields you detect, the better - don't skip any information that varies between documents

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
    "identified_scenarios": ["List of all unique scenarios detected across the example documents"]
}"""


def generate_policy_from_examples(
    html_examples: List[str],
    affidavit_type_name: str,
    additional_context: str = "",
    existing_questions: List[Dict] = None
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

        user_prompt = f"""Analyze these example affidavit documents for: "{affidavit_type_name}"
{f"Additional context: {additional_context}" if additional_context else ""}
{existing_questions_context}
{examples_text}

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
    Applies smart Trinidad & Tobago specific validation rules automatically.
    Converts age fields to date_of_birth with calendar picker.
    
    Args:
        detected_fields: Fields detected by generate_policy_from_examples
    
    Returns:
        List of intake_schema question objects with validation
    """
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
    for field in detected_fields:
        field_id = field.get('id', '').lower()
        if _matches_field_pattern(field_id, TT_FIELD_RULES.get('date_of_birth', {}).get('patterns', [])):
            has_dob_field = True
            break
    
    for field in detected_fields:
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
            'type_locked': True  # Prevent frontend from overriding backend type
        }
        
        # Start with any validation from the AI
        validation = field.get('validation', {})
        
        # Apply smart T&T validation rules based on field ID/label
        validation = apply_smart_validation(field_id, field_label, validation, question)
        
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


def apply_smart_validation(field_id: str, field_label: str, existing_validation: Dict, question: Dict) -> Dict:
    """
    Apply smart Trinidad & Tobago specific validation based on field detection.
    
    Args:
        field_id: The field ID (snake_case)
        field_label: Human readable label
        existing_validation: Any validation already set by AI
        question: The question dict (may be modified for type changes)
    
    Returns:
        Enhanced validation dict
    """
    validation = existing_validation.copy() if existing_validation else {}
    field_lower = field_id.lower()
    label_lower = field_label.lower()
    
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

FIELD_DEFAULTS = {
    'full_name': {
        'label': 'Full Name',
        'placeholder': 'John Michael Smith',
        'help_text': 'Enter your full legal name as on your ID.'
    },
    'date_of_birth': {
        'label': 'Date of Birth',
        'placeholder': '1990-06-14',
        'help_text': 'Select your date of birth; your age is calculated automatically.'
    },
    'address': {
        'label': 'Residential Address',
        'placeholder': '15 Queen Street, Port of Spain',
        'help_text': 'Enter your current residential address in Trinidad & Tobago.'
    },
    'phone': {
        'label': 'Phone Number',
        'placeholder': '868-123-4567',
        'help_text': 'Enter a Trinidad & Tobago phone number (868-XXX-XXXX).'
    },
    'email': {
        'label': 'Email Address',
        'placeholder': 'your.email@example.com',
        'help_text': 'Enter your email to receive updates.'
    },
    'electoral_id': {
        'label': 'Electoral ID (11 digits)',
        'placeholder': '19741104044',
        'help_text': '11 digits in YYYYMMDDXXX format (first 8 = date of birth).'
    },
    'national_id': {
        'label': 'National ID (11 digits)',
        'placeholder': '19741104044',
        'help_text': '11 digits in YYYYMMDDXXX format (first 8 = date of birth).'
    },
    'passport': {
        'label': 'Passport Number',
        'placeholder': 'TB1234567',
        'help_text': '2 letters + 7 digits (e.g., TB1234567).'
    },
    'drivers_permit': {
        'label': "Driver's Permit Number",
        'placeholder': 'DL123456',
        'help_text': "Enter the number from your driver's permit."
    },
    'declaration_month': {
        'label': 'Declaration Month',
        'placeholder': 'February',
        'help_text': 'Select the month of declaration.'
    },
    'declaration_day': {
        'label': 'Declaration Day',
        'placeholder': '14',
        'help_text': 'Enter the day of the month (1–31).'
    },
    'declaration_year': {
        'label': 'Declaration Year',
        'placeholder': '2026',
        'help_text': 'Enter the 4-digit year.'
    },
}


def _title_from_id(field_id: str) -> str:
    return ' '.join(part.capitalize() for part in field_id.split('_')) if field_id else ''


def _apply_field_defaults(detected_fields: List[Dict]) -> List[Dict]:
    updated = []
    for field in detected_fields or []:
        field_id = field.get('id', '').strip()
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

        updated.append(field)

    return updated
