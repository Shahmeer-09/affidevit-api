"""
Affidavit Express - Policy Generator Service

Uses GPT-4o to analyze uploaded example affidavits and generate:
- Template HTML structure with {{field}} placeholders
- Policy JSON with validation rules
- Suggested disallowed phrases
- Few-shot examples for training
"""

import json
import logging
from typing import Dict, List, Optional
from django.conf import settings

logger = logging.getLogger(__name__)

# Lazy import openai
openai_client = None


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

2. **Commissioner/Attestation Section - CRITICAL:**
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

3. **Placeholder Format:**
   - Use {{field_name}} format (double curly braces)
   - Field names should be snake_case (e.g., {{full_name}}, {{date_of_birth}})
   - Replace ONLY variable data with placeholders, preserve all static text exactly

4. **Content Preservation:**
   - Keep all legal language, references, and declarations exactly as in examples
   - Numbered statements should follow the same numbering style as examples
   - Preserve paragraph structure and formatting

5. **What to Extract as Fields:**
   - Personal details (name, age, address, ID numbers)
   - Dates (declaration date, relevant dates in the content)
   - Location information
   - Specific facts that vary per affidavit

Respond ONLY with a valid JSON object in this exact structure:
{
    "template_html": "<html template with {{placeholders}} - MUST match example format exactly>",
    "detected_fields": [
        {
            "id": "field_id",
            "label": "Human Readable Label",
            "type": "text|textarea|date|select|number",
            "required": true,
            "placeholder": "Example value",
            "help_text": "What this field is for"
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
        "input": {"field_id": "example value"},
        "output_html": "<example output matching the exact template format>"
    },
    "analysis_notes": "Brief notes about the affidavit type structure",
    "format_warnings": ["Any deviations from standard format noted in examples"]
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

        response = client.chat.completions.create(
            model=settings.OPENAI_QA_MODEL,  # Use GPT-4o for analysis
            messages=[
                {"role": "system", "content": POLICY_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=4000,
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

        response = client.chat.completions.create(
            model=settings.OPENAI_QA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=4000,
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
    
    Args:
        detected_fields: Fields detected by generate_policy_from_examples
    
    Returns:
        List of intake_schema question objects
    """
    intake_schema = []
    
    type_mapping = {
        'text': 'text',
        'textarea': 'textarea',
        'date': 'date',
        'select': 'select',
        'number': 'number',
        'email': 'email',
        'phone': 'tel',
    }
    
    for field in detected_fields:
        question = {
            'id': field.get('id', ''),
            'label': field.get('label', field.get('id', '')),
            'type': type_mapping.get(field.get('type', 'text'), 'text'),
            'required': field.get('required', True),
            'placeholder': field.get('placeholder', ''),
            'help_text': field.get('help_text', ''),
        }
        
        # Add options if it's a select type
        if question['type'] == 'select' and 'options' in field:
            question['options'] = field['options']
        
        intake_schema.append(question)
    
    return intake_schema


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
    
    policy = {
        'required_sections': generation_result.get('required_sections', []),
        'validation_rules': generation_result.get('validation_rules', []),
        'few_shot_examples': [],
    }
    
    # Add few-shot example if generated
    few_shot = generation_result.get('few_shot_example')
    if few_shot:
        policy['few_shot_examples'].append(few_shot)
    
    return policy
