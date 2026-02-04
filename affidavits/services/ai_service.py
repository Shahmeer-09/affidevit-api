"""
Affidavit Express - AI Service Layer

Handles AI-powered drafting and quality assurance using OpenAI GPT models.
- Draft Node: GPT-4o-mini for generating affidavit drafts
- QA Node: GPT-4o for quality checks and validation
- Scenario Detection: Pattern matching against known scenarios
"""

import json
import logging
import time
from typing import Optional, List, Dict, Tuple
from django.conf import settings

logger = logging.getLogger(__name__)

# Lazy import openai to avoid import errors if not installed
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


def detect_scenario(answers_json: dict, scenario_library: List[dict]) -> Tuple[List[str], bool]:
    """
    Detect which scenarios match the user's answers.
    
    Args:
        answers_json: User's intake form answers
        scenario_library: List of scenario definitions from affidavit type
        
    Returns:
        Tuple of (matched_scenario_tags, is_new_scenario)
    """
    if not scenario_library:
        return [], True
    
    matched_tags = []
    answers_str = json.dumps(answers_json).lower()
    
    for scenario in scenario_library:
        scenario_id = scenario.get('id', '')
        keywords = scenario.get('keywords', [])
        patterns = scenario.get('patterns', [])
        
        # Check keyword matches
        keyword_matches = sum(1 for kw in keywords if kw.lower() in answers_str)
        keyword_threshold = scenario.get('keyword_threshold', len(keywords) // 2 + 1)
        
        # Check pattern matches (field-value pairs)
        pattern_matches = 0
        for pattern in patterns:
            field = pattern.get('field', '')
            values = pattern.get('values', [])
            field_value = str(answers_json.get(field, '')).lower()
            if any(v.lower() in field_value for v in values):
                pattern_matches += 1
        
        pattern_threshold = scenario.get('pattern_threshold', len(patterns) // 2 + 1) if patterns else 0
        
        # Match if either keywords or patterns meet threshold
        if keyword_matches >= keyword_threshold or (patterns and pattern_matches >= pattern_threshold):
            matched_tags.append(scenario_id)
    
    is_new_scenario = len(matched_tags) == 0
    return matched_tags, is_new_scenario


def build_few_shot_examples(few_shot_examples: List[dict], max_examples: int = 3) -> str:
    """
    Build few-shot examples section for the prompt.
    
    Args:
        few_shot_examples: List of example dicts with 'input' and 'output' keys
        max_examples: Maximum number of examples to include
        
    Returns:
        Formatted string with few-shot examples
    """
    if not few_shot_examples:
        return ""
    
    examples_text = "\n\n**EXAMPLES OF SIMILAR AFFIDAVITS:**\n"
    
    for i, example in enumerate(few_shot_examples[:max_examples]):
        example_input = example.get('input', {})
        example_output = example.get('output', '')
        scenario_tag = example.get('scenario', 'general')
        
        examples_text += f"\n--- Example {i + 1} ({scenario_tag}) ---\n"
        examples_text += f"Input: {json.dumps(example_input, indent=2)}\n"
        examples_text += f"Output:\n{example_output}\n"
    
    examples_text += "\n--- End of Examples ---\n"
    return examples_text


def select_relevant_examples(
    few_shot_examples: List[dict], 
    scenario_tags: List[str],
    max_examples: int = 3
) -> List[dict]:
    """
    Select the most relevant few-shot examples based on matched scenarios.
    
    Args:
        few_shot_examples: All available examples
        scenario_tags: Matched scenario tags for this request
        max_examples: Maximum examples to return
        
    Returns:
        List of most relevant examples
    """
    if not few_shot_examples:
        return []
    
    # Score examples by relevance
    scored_examples = []
    for example in few_shot_examples:
        example_scenarios = example.get('scenarios', [example.get('scenario', 'general')])
        if isinstance(example_scenarios, str):
            example_scenarios = [example_scenarios]
        
        # Calculate relevance score
        overlap = len(set(example_scenarios) & set(scenario_tags))
        score = overlap * 10 + (1 if 'general' in example_scenarios else 0)
        scored_examples.append((score, example))
    
    # Sort by score descending and take top N
    scored_examples.sort(key=lambda x: -x[0])
    return [ex for _, ex in scored_examples[:max_examples]]


def translate_to_english(answers_json: dict) -> dict:
    """
    Detect and translate non-English content in user answers to English.
    Handles mixed language input (e.g., "my house is old or ye khrab ho gya h").
    
    Args:
        answers_json: User's intake form answers (may contain non-English text)
    
    Returns:
        dict: Translated answers with all text in English
    """
    try:
        client = get_openai_client()
        
        logger.info("[TRANSLATE] Checking all answers for non-English content (including romanized foreign languages)")
        
        system_prompt = """You are a language translation assistant. Your job is to translate non-English text to English while preserving the meaning exactly.

**RULES:**
1. If text is already in English, leave it unchanged
2. If text is mixed (English + another language), translate ONLY the non-English parts
3. Preserve all factual information exactly - don't change numbers, names, or addresses
4. Return the translated answers in the SAME JSON structure
5. Common languages you may encounter: Urdu, Hindi, Arabic, Spanish, French, etc.

**EXAMPLES:**
Input: {"repairs": "my house is old or ye khrab ho gya h"}
Output: {"repairs": "my house is old or it has become damaged"}

Input: {"property": "dwelling house at 15 Queen Street"}
Output: {"property": "dwelling house at 15 Queen Street"}

Input: {"name": "Ali Ahmed"}
Output: {"name": "Ali Ahmed"}

**IMPORTANT:**
- Translate to natural, grammatically correct English
- Keep the same JSON field names
- Preserve numbers, dates, and proper nouns
- Return ONLY the JSON, no explanations"""

        user_prompt = f"""Translate any non-English text in this JSON to English:

{json.dumps(answers_json, indent=2)}

Return the translated JSON with the same structure. If a field is already in English, keep it exactly as is."""

        response = client.chat.completions.create(
            model=settings.OPENAI_DRAFT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        translated_json = json.loads(response.choices[0].message.content.strip())
        
        logger.info(f"[TRANSLATE] Successfully translated answers")
        logger.info(f"[TRANSLATE] Original: {json.dumps(answers_json, ensure_ascii=False)[:200]}")
        logger.info(f"[TRANSLATE] Translated: {json.dumps(translated_json)[:200]}")
        
        return translated_json
        
    except Exception as e:
        logger.error(f"Error in translation: {e}, using original answers")
        # If translation fails, return original answers
        return answers_json


def get_base_instruction() -> str:
    """Get the active AI base instruction from the database."""
    try:
        from ..models import AIBaseInstruction
        instruction = AIBaseInstruction.get_active()
        return instruction.instruction_text
    except Exception as e:
        logger.warning(f"Could not get base instruction from DB: {e}")
        return get_default_draft_system_prompt()


def draft_affidavit(
    answers_json: dict,
    policy_json: dict,
    affidavit_type_name: str,
    scenario_library: List[dict] = None,
    template_html: str = '',
    disallowed_phrases: List[str] = None
) -> dict:
    """
    Generate an affidavit draft using GPT-4o-mini.
    Uses scenario detection and few-shot examples for better drafts.
    Now outputs HTML format for PDF generation.
    
    Args:
        answers_json: User's intake form answers
        policy_json: Affidavit type policy with prompts, rules, and examples
        affidavit_type_name: Name of the affidavit type
        scenario_library: Optional scenario library for detection
        template_html: Optional HTML template with {{placeholders}}
        disallowed_phrases: Optional list of phrases to avoid
    
    Returns:
        dict: {
            'success': bool,
            'draft_text': str (the generated draft - now in HTML),
            'draft_html': str (alias for draft_text),
            'scenario_tags': list of matched scenario tags,
            'new_scenario': bool (true if no known scenario matched),
            'prompt_tokens': int,
            'completion_tokens': int,
            'total_tokens': int,
            'error': str (if success is False)
        }
    """
    try:
        client = get_openai_client()
        start_time = time.time()
        
        # STEP 0: Translate non-English content to English
        answers_json = translate_to_english(answers_json)
        
        # Get base instruction from database
        base_instruction = get_base_instruction()
        
        # ===== DEBUG LOGGING =====
        logger.info("=" * 80)
        logger.info(f"[DRAFT_AFFIDAVIT] Starting draft for: {affidavit_type_name}")
        logger.info(f"[DRAFT_AFFIDAVIT] template_html parameter length: {len(template_html) if template_html else 0}")
        logger.info(f"[DRAFT_AFFIDAVIT] template_html first 200 chars: {template_html[:200] if template_html else 'EMPTY'}")
        logger.info(f"[DRAFT_AFFIDAVIT] policy_json keys: {list(policy_json.keys()) if policy_json else 'None'}")
        if policy_json:
            policy_template = policy_json.get('template_html', '')
            logger.info(f"[DRAFT_AFFIDAVIT] policy_json.template_html length: {len(policy_template) if policy_template else 0}")
        logger.info(f"[DRAFT_AFFIDAVIT] disallowed_phrases: {disallowed_phrases}")
        # ===== END DEBUG LOGGING =====
        
        # Extract policy components
        template = template_html or policy_json.get('template_html', '') or policy_json.get('template', '')
        phrases_to_avoid = disallowed_phrases or policy_json.get('disallowed_phrases', [])
        required_sections = policy_json.get('required_sections', [])
        few_shot_examples = policy_json.get('few_shot_examples', [])
        
        # ===== MORE DEBUG LOGGING =====
        logger.info(f"[DRAFT_AFFIDAVIT] Final template length after resolution: {len(template) if template else 0}")
        logger.info(f"[DRAFT_AFFIDAVIT] Template starts with: {template[:300] if template else 'NO TEMPLATE'}")
        logger.info("=" * 80)
        # ===== END DEBUG LOGGING =====
        
        # Detect scenarios from user answers
        lib = scenario_library or policy_json.get('scenario_library', [])
        scenario_tags, is_new_scenario = detect_scenario(answers_json, lib)
        
        # Select relevant few-shot examples based on scenarios
        relevant_examples = select_relevant_examples(few_shot_examples, scenario_tags)
        logger.info(f"[DRAFT_AFFIDAVIT] Scenarios: {scenario_tags}, Relevant examples: {len(relevant_examples)}")
        examples_text = build_few_shot_examples(relevant_examples)
        
        # Build scenario context if matched
        scenario_context = ""
        if scenario_tags:
            scenario_context = f"\n**Detected Scenario(s):** {', '.join(scenario_tags)}\n"
            # Get scenario-specific instructions
            for scenario in lib:
                if scenario.get('id') in scenario_tags:
                    instructions = scenario.get('drafting_instructions', '')
                    if instructions:
                        scenario_context += f"**Scenario Instructions:** {instructions}\n"
        
        # Build the user prompt - now requesting HTML output
        template_section = ""
        template_format_instructions = ""
        if template:
            template_section = f"""
**TEMPLATE TO FOLLOW (CRITICAL - MATCH THIS FORMAT EXACTLY):**
{template[:5000] if len(template) > 5000 else template}
"""
            template_format_instructions = """
**TEMPLATE ADHERENCE RULES (CRITICAL):**
- Follow the template structure EXACTLY - do not add or remove sections
- DO NOT add headers like "AFFIDAVIT" or subtitles if not in the template
- Start the document EXACTLY as the template starts (e.g., "REPUBLIC OF TRINIDAD AND TOBAGO:")
- For the commissioner/attestation section, use the EXACT format from the template
- DO NOT modernize or "improve" the commissioner section format
- Replace ONLY the {{placeholder}} values with actual data
- Preserve all static text, legal language, and formatting exactly as shown
"""
        
        # Check if we have a calculated age from DOB validation
        calculated_age_note = ""
        if answers_json.get('_calculated_age') is not None:
            calculated_age = answers_json.get('_calculated_age')
            calculated_age_note = f"""
**IMPORTANT - USE CALCULATED AGE:**
The system has calculated the declarant's age from their date of birth as: {calculated_age} years old.
Use this calculated age ({calculated_age}) in the affidavit, NOT any age the user may have manually entered.
Do NOT include the date of birth in the affidavit - only include the age.
"""
        
        user_prompt = f"""
Generate a formal {affidavit_type_name} based on the following information provided by the applicant.

**OUTPUT FORMAT: Clean HTML suitable for PDF generation.**
Use semantic HTML tags: <p>, <ol>, <li>, <strong>, <sup>, etc.
Do NOT include <html>, <head>, or <body> tags - just the content.
{calculated_age_note}
**Applicant Information:**
{json.dumps(answers_json, indent=2)}
{scenario_context}
{template_section}
{template_format_instructions}
**CRITICAL: INTELLIGENT TEXT INTEGRATION**

User answers may be informal, fragmented, or grammatically incomplete. Your job is to:
1. **Rephrase** user answers into proper legal language
2. **Fix grammar** - don't copy-paste broken sentences
3. **Make it flow** - integrate user's info naturally into the template
4. **Preserve meaning** - keep all facts exactly as provided, just phrase them professionally

EXAMPLE OF WHAT NOT TO DO:
❌ User wrote: "house is old"
❌ Bad template fill: "making improvements to the house is old on land..."
✅ Correct: "making improvements to my old house on land..." OR "making improvements to the house, which is old, on land..."

EXAMPLE OF WHAT TO DO:
User wrote: "live there 5 year" → "resided there for 5 years"
User wrote: "car broke" → "the vehicle was damaged"
User wrote: "work at hospital" → "employed at the hospital"

**Transform user input into professional legal language while preserving facts.**

**REQUIREMENTS:**
1. {"FOLLOW THE TEMPLATE FORMAT EXACTLY as shown above." if template else "Use formal legal language appropriate for an affidavit in Trinidad and Tobago."}
2. Include all standard affidavit sections as per the template or Trinidad and Tobago format:
   - Header starting with "REPUBLIC OF TRINIDAD AND TOBAGO:" (NO extra title before this)
   - "IN THE MATTER OF THE STATUTORY DECLARATION ACT"
   - "CHAPTER 7: No 04"
   - Declarant introduction with full name, age, address, and ID number
   - Numbered factual statements (rephrase user answers into complete, grammatically correct sentences)
   - Declaration of truth with legal consequences acknowledgment
   - Commissioner attestation section (simple format: "Declared at...", "Before me, Commissioner of Affidavits.")
3. **Intelligently integrate** the user's details - don't copy-paste their raw text
4. Ensure the document is suitable for commissioning.
5. DO NOT add "SWORN/AFFIRMED before me at" with City/Town, Province/State fields - use the simple commissioner format.
6. If a calculated age (_calculated_age) is provided, use THAT age in the document, not any user-entered age field.

{f"**Required Sections:** {', '.join(required_sections)}" if required_sections else ""}

{f"**Do NOT include these phrases:** {', '.join(phrases_to_avoid)}" if phrases_to_avoid else ""}
{examples_text}
Generate the complete affidavit in HTML format:
"""
        
        # ===== LOG THE FULL PROMPT =====
        logger.info("=" * 80)
        logger.info("[DRAFT_AFFIDAVIT] FULL USER PROMPT:")
        logger.info(user_prompt)
        logger.info("=" * 80)
        logger.info("[DRAFT_AFFIDAVIT] SYSTEM PROMPT (base_instruction):")
        logger.info(base_instruction[:500] + "..." if len(base_instruction) > 500 else base_instruction)
        logger.info("=" * 80)
        # ===== END PROMPT LOG =====
        
        response = client.chat.completions.create(
            model=settings.OPENAI_DRAFT_MODEL,
            messages=[
                {"role": "system", "content": base_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )
        
        draft_html = response.choices[0].message.content.strip()
        
        # ===== LOG THE AI RESPONSE =====
        logger.info("=" * 80)
        logger.info("[DRAFT_AFFIDAVIT] AI RESPONSE (first 1000 chars):")
        logger.info(draft_html[:1000])
        logger.info("=" * 80)
        # ===== END RESPONSE LOG =====
        
        # Clean up the response - remove markdown code blocks if present
        if draft_html.startswith('```html'):
            draft_html = draft_html[7:]
        if draft_html.startswith('```'):
            draft_html = draft_html[3:]
        if draft_html.endswith('```'):
            draft_html = draft_html[:-3]
        draft_html = draft_html.strip()
        
        usage = response.usage
        
        logger.info(f"Successfully generated HTML draft for {affidavit_type_name} (scenarios: {scenario_tags})")
        
        return {
            'success': True,
            'draft_text': draft_html,  # Keep for backward compatibility
            'draft_html': draft_html,  # New field for HTML output
            'scenario_tags': scenario_tags,
            'new_scenario': is_new_scenario,
            'prompt_tokens': usage.prompt_tokens if usage else 0,
            'completion_tokens': usage.completion_tokens if usage else 0,
            'total_tokens': usage.total_tokens if usage else 0,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Error generating draft: {e}")
        return {
            'success': False,
            'draft_text': '',
            'scenario_tags': [],
            'new_scenario': True,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'error': str(e)
        }


def analyze_input_suitability(
    answers_json: dict,
    template_html: str,
    affidavit_type_name: str,
    form_fields: list = None
) -> dict:
    """
    DEPRECATED: This function is kept for backward compatibility.
    Use validate_inputs_before_submission() for the new pre-submission validation flow.
    
    This now just returns is_suitable=True to skip the clarification loop.
    Validation should happen BEFORE submission via the validate endpoint.
    """
    return {
        'is_suitable': True,
        'issues': [],
        'clarification_question': '',
        'clarification_example': '',
        'calculated_age': None,
        'resolved_fields': [],
        'needs_human_review': False,
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
    }


def validate_inputs_before_submission(
    answers_json: dict,
    template_html: str,
    affidavit_type_name: str
) -> dict:
    """
    Quick validation check BEFORE allowing user to submit.
    Only checks if inputs are valid - does NOT generate clarification questions.
    Used to block submission until user fixes all invalid fields.
    
    This is called from the frontend BEFORE submission to validate all fields.
    
    Args:
        answers_json: User's intake form answers
        template_html: The HTML template with placeholders
        affidavit_type_name: Name of the affidavit type
    
    Returns:
        dict: {
            'all_valid': bool,
            'invalid_fields': {'field_name': 'reason why invalid'},
            'validation_notes': [{'field': 'address', 'issue': 'Not valid', 'suggestion': '...'}]
        }
    """
    try:
        client = get_openai_client()
        
        if not template_html:
            return {'all_valid': True, 'invalid_fields': {}, 'validation_notes': []}
        
        system_prompt = """You are a strict field validator for legal documents in Trinidad and Tobago. Your job is to catch serious issues - gibberish, future dates, irrelevant answers, and logical contradictions. Return your response as JSON.

**CRITICAL RULES:**
1. BE LENIENT on spelling/grammar - The AI drafter will fix those
2. BE STRICT on future dates, gibberish, and contradictions - these MUST be flagged
3. DO NOT return "informational notes" - ONLY REAL ERRORS
4. If a value is VALID, do NOT include it in any error list
5. If house_age is given as a date, CALCULATE the age yourself

**FLAG THESE SERIOUS PROBLEMS:**

1. **FUTURE DATES (CRITICAL - MUST CHECK):**
   - Check declaration_month + declaration_year + declaration_day combinations
   - If the date is IN THE FUTURE compared to current date → INVALID
   - Example: Current date is February 4, 2026. User enters "March 2026" → FUTURE DATE ❌
   - Example: Current date is February 4, 2026. User enters "February 10, 2026" → FUTURE DATE ❌
   - Example: User enters "January 2026" → VALID ✅ (in the past)
   - Check ANY date-related fields (declaration_month, declaration_year, declaration_day, event dates)
   - Month names: January=1, February=2, March=3, April=4, May=5, June=6, July=7, August=8, September=9, October=10, November=11, December=12

2. **PURE GIBBERISH (keyboard mashing):**
   - Examples: "ihdhfihdbhb", "asdfasdfasdf", "fgdgsgsdvchdvudv", "jkljkljkl"
   - Random characters with NO recognizable words
   - Must be COMPLETELY meaningless

3. **TRAILING GIBBERISH (valid start, bad ending) - BE AGGRESSIVE:**
   - "15 Queen Street asdfasdf" → INVALID (gibberish at end)
   - "replacing windows hahahah i am happy" → INVALID (irrelevant at end)
   - "fixing the roof khdfbiewbfiwbf" → INVALID (gibberish at end)
   - "my property is 100sqm lololol" → INVALID (nonsense at end)
   - "valid address ????" → INVALID (junk at end)
   - "something !!!!!!!!" → INVALID (excessive punctuation)
   - "text here ......." → INVALID (trailing dots)
   - Look for: random chars, "haha", "lol", "????", "!!!!", ".....", keyboard mashing ANYWHERE in the text

4. **COMPLETELY IRRELEVANT ANSWERS:**
   - Asked about property, user says "i'm sick" or "i'm sad hahaha" → INVALID
   - Asked about address, user says "i don't like this" → INVALID
   - The answer has NOTHING to do with the question

5. **ACTUAL MATHEMATICAL CONTRADICTIONS:**
   - Residence duration > person's age → IMPOSSIBLE
   - Residence duration > house age → IMPOSSIBLE
   - Events before person was born → IMPOSSIBLE
   - ONLY flag when math is truly impossible

6. **IMPOSSIBLE VALUES:**
   - Age 250, Age -5 → INVALID
   - Negative durations → INVALID

7. **ID NUMBER FORMAT (Trinidad & Tobago) - 11 DIGITS WITH DOB:**
   - Electoral ID MUST be exactly 11 digits in format YYYYMMDDXXX
   - First 8 digits encode date of birth: YYYYMMDD (Year-Month-Day)
   - Last 3 digits are unique sequence number
   - Extract ONLY the digits from input (ignore spaces, dashes)
   - Count digits: if count == 11 → Check if valid format ✅
   - Count digits: if count != 11 → INVALID ❌
   - Example: "19741104044" → 11 digits, DOB=1974-11-04 → VALID ✅
   - Example: "1974-11-04-044" → 11 digits → VALID ✅  
   - Example: "123456789" → 9 digits → INVALID ❌ (old format)
   
   **CRITICAL - CROSS-VALIDATE WITH DATE OF BIRTH:**
   - If BOTH electoral_id AND date_of_birth fields are provided:
     * Extract year (chars 1-4), month (chars 5-6), day (chars 7-8) from electoral_id
     * Format as YYYY-MM-DD
     * Compare with the date_of_birth field value
     * ONLY FLAG if they are DIFFERENT (mismatch)
     * DO NOT FLAG if they MATCH - matching is CORRECT!
   - Example: electoral_id="19741104044" + date_of_birth="1974-11-04" → MATCH → DO NOT FLAG ✅ (this is correct!)
   - Example: electoral_id="19900614044" + date_of_birth="1990-06-14" → MATCH → DO NOT FLAG ✅ (this is correct!)
   - Example: electoral_id="19741104044" + date_of_birth="1980-05-15" → MISMATCH → FLAG ERROR ❌
   - ONLY return error when dates are DIFFERENT, NEVER when they match!
   - Error message (only for mismatch): "Your Electoral ID indicates birth date 1974-11-04, but you entered 1980-05-15 as Date of Birth"

**WHAT TO ACCEPT (DO NOT FLAG) - The drafter will fix these:**

✅ Spelling errors, informal language, incomplete but meaningful, vague but relevant answers

**OUTPUT FORMAT - INCLUDE HELPFUL SUGGESTIONS:**
{
    "all_valid": boolean,
    "contradictions_found": [
        {
            "type": "age_vs_residence" | "house_vs_residence" | "date_contradiction" | "other",
            "fields_involved": ["field1", "field2"],
            "explanation": "You said you're 30 years old but have lived here for 45 years. This is impossible - you cannot live somewhere longer than you've been alive.",
            "suggestion": "Please check: either your age/date of birth or your residence duration needs to be corrected. If you're 30, you could have lived there at most 30 years."
        }
    ],
    "field_checks": [
        {
            "field": "field_name",
            "value": "what user entered",
            "is_valid": true/false,
            "issue": "Clear explanation of what's wrong - null if valid",
            "suggestion": "Helpful suggestion of what they SHOULD enter - null if valid. Be specific and give examples.",
            "correct_format": "Show the correct format if applicable (e.g., '123456789' for ID)"
        }
    ]
}

**IMPORTANT - ONLY RETURN ACTUAL ERRORS:**
- If something is VALID, set all_valid=true and return EMPTY arrays
- DO NOT include "informational notes" or "observations" in the response
- DO NOT complain about data formats (dates vs numbers) - just calculate what you need
- If house_age is given as a date like "1994-07-04", calculate: current_year - 1994 = house age
- ONLY return items in contradictions_found or field_checks if there is an ACTUAL ERROR

Examples of what NOT to return:
❌ "House age is given as a date instead of years" - just calculate it!
❌ "This is valid but needs clarification" - if it's valid, don't return it!
❌ "The values are consistent" - good, then return all_valid=true with empty arrays

Examples of what TO return:
✅ "You lived here 50 years but are only 30 years old - impossible"
✅ "Electoral ID must be 11 digits, you entered 5"
✅ "This field contains gibberish: asdfghjkl"
"""

        from datetime import date
        current_year = date.today().year
        current_month = date.today().month
        current_day = date.today().day
        current_date = date.today().isoformat()
        current_month_name = date.today().strftime('%B')

        user_prompt = f"""
**AFFIDAVIT TYPE:** {affidavit_type_name}
**CURRENT DATE:** {current_date}
**CURRENT YEAR:** {current_year}
**CURRENT MONTH:** {current_month_name} (month number: {current_month})
**CURRENT DAY:** {current_day}

**MONTH NUMBER REFERENCE:**
January=1, February=2, March=3, April=4, May=5, June=6, July=7, August=8, September=9, October=10, November=11, December=12

**TEMPLATE (shows what each field is for):**
{template_html[:4000] if template_html else 'No template'}

**USER INPUT TO VALIDATE:**
{json.dumps(answers_json, indent=2)}

**CRITICAL INSTRUCTIONS - CHECK THESE IN ORDER:**

1. **FUTURE DATE CHECK (MOST IMPORTANT):**
   - Look for declaration_month, declaration_year, declaration_day fields
   - Convert month name to number (March=3, April=4, etc.)
   - TODAY'S DATE IS VALID - do NOT flag today as future!
   - Comparison rules:
     * If year > {current_year} → FUTURE DATE ❌
     * If year == {current_year} AND month > {current_month} → FUTURE DATE ❌
     * If year == {current_year} AND month == {current_month} AND day > {current_day} → FUTURE DATE ❌
     * If year == {current_year} AND month == {current_month} AND day == {current_day} → TODAY = VALID ✅
   - Examples with current date {current_month_name} {current_day}, {current_year}:
     * Month="March", Year="2026" → March=3 > {current_month} = FUTURE ❌
     * Month="February", Year="2026", Day=5 → Same month, but 5 > {current_day} = FUTURE ❌
     * Month="February", Year="2026", Day=4 → Same month, 4 == {current_day} = TODAY = VALID ✅
     * Month="February", Year="2026", Day=3 → Same month, 3 < {current_day} = PAST = VALID ✅
     * Month="January", Year="2026" → January=1 < {current_month} = PAST ✅

2. **TRAILING GIBBERISH CHECK:**
   - Scan EVERY text field for gibberish at the END
   - Look for: random characters, "haha", "lol", "????", "!!!!", ".....", keyboard mashing
   - Example: "15 Queen Street asdfg" → Has gibberish "asdfg" at end → INVALID
   - Example: "fixing roof hahaha" → Has "hahaha" at end → INVALID

3. **PURE GIBBERISH CHECK:**
   - Fields that are ENTIRELY meaningless: "asdfasdf", "jkljkl", etc.

4. **CONTRADICTION CHECK:**
   - Calculate age from date_of_birth if present
   - Calculate house age from build date if present
   - ONLY FLAG if residence > age OR residence > house_age

5. **ID FORMAT CHECK (11 DIGITS + DOB VALIDATION):**
   - Extract ONLY digits from the ID field (remove spaces, dashes, any non-digits)
   - Count the digits
   - If digit count == 11 → Extract first 8 digits as YYYYMMDD and validate date
   - If digit count != 11 → INVALID (flag: "Electoral ID must be 11 digits, you entered X digits")
   - Example: "19741104044" has 11 digits → Check if 19741104 is valid date → VALID
   - Example: "1974 11 04 044" has 11 digits → VALID
   - Example: "123456789" has 9 digits → INVALID (old format, must be 11)
   
   **CROSS-VALIDATE ELECTORAL ID WITH DATE OF BIRTH:**
   - If electoral_id field exists AND date_of_birth field exists:
     * Extract digits 1-4 (year), 5-6 (month), 7-8 (day) from electoral_id
     * Format as YYYY-MM-DD (e.g., "19741104" → "1974-11-04")
     * Compare with date_of_birth field value
     * ONLY flag if they are DIFFERENT!
     * If they MATCH → This is CORRECT, do NOT flag it!
   - Example: electoral_id="19900614044" + date_of_birth="1990-06-14" → MATCH → VALID, DO NOT FLAG ✅
   - Example: electoral_id="19741104044" + date_of_birth="1974-11-04" → MATCH → VALID, DO NOT FLAG ✅
   - Example: electoral_id="19741104044" + date_of_birth="1980-05-15" → MISMATCH → FLAG ERROR ❌
   - NEVER flag when DOB matches the ID! Only flag when they differ.

**BE STRICT** on future dates and gibberish - these MUST be caught.
**BE LENIENT** on spelling/grammar - the AI drafter fixes that.
"""

        response = client.chat.completions.create(
            model=settings.OPENAI_QA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        
        # Transform into simple format for frontend
        invalid_fields = {}
        validation_notes = []
        
        # Add contradictions to validation notes with high priority
        for contradiction in result.get('contradictions_found', []):
            for field in contradiction.get('fields_involved', []):
                if field not in invalid_fields:
                    invalid_fields[field] = contradiction.get('explanation', 'Logical contradiction detected')
            validation_notes.append({
                'type': 'contradiction',
                'contradiction_type': contradiction.get('type', 'other'),
                'fields': contradiction.get('fields_involved', []),
                'issue': contradiction.get('explanation', 'Logical contradiction detected'),
                'suggestion': contradiction.get('suggestion', 'Please review the related fields for consistency')
            })
        
        # Add individual field issues
        for check in result.get('field_checks', []):
            if not check.get('is_valid', True):
                field_name = check.get('field', 'unknown')
                if field_name not in invalid_fields:  # Don't overwrite contradiction messages
                    invalid_fields[field_name] = check.get('issue', 'Invalid value')
                validation_notes.append({
                    'type': 'field_error',
                    'field': field_name,
                    'value': check.get('value', ''),
                    'issue': check.get('issue', 'Invalid value'),
                    'suggestion': check.get('suggestion', ''),
                    'correct_format': check.get('correct_format', '')
                })
        
        all_valid = result.get('all_valid', len(invalid_fields) == 0)
        
        logger.info(f"Pre-submission validation for {affidavit_type_name}: "
                   f"all_valid={all_valid}, invalid_fields={list(invalid_fields.keys())}")
        
        return {
            'all_valid': all_valid,
            'invalid_fields': invalid_fields,
            'validation_notes': validation_notes,
            'field_checks': result.get('field_checks', [])
        }
        
    except Exception as e:
        logger.error(f"Error in pre-submission validation: {e}")
        
        # FALLBACK: Use pattern-based validation when AI is unavailable
        fallback_invalid_fields = {}
        fallback_notes = []
        
        import re
        
        # First, try to detect basic contradictions
        age = None
        residence_duration = None
        house_age = None
        date_of_birth = None
        
        # Extract numeric values for contradiction checking
        for field_name, field_value in answers_json.items():
            field_lower = field_name.lower()
            value_str = str(field_value).lower().strip()
            
            # Try to extract age
            if 'date_of_birth' in field_lower or 'dob' in field_lower:
                try:
                    from datetime import datetime, date as dt_date
                    dob = datetime.strptime(str(field_value), '%Y-%m-%d').date()
                    today = dt_date.today()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    date_of_birth = field_value
                except:
                    pass
            elif 'age' in field_lower and not 'house' in field_lower:
                try:
                    age = int(re.search(r'\d+', str(field_value)).group())
                except:
                    pass
            
            # Try to extract residence duration
            if 'residence' in field_lower or 'living' in field_lower or 'resided' in field_lower:
                try:
                    match = re.search(r'(\d+)\s*(?:years?|yrs?)', value_str)
                    if match:
                        residence_duration = int(match.group(1))
                except:
                    pass
            
            # Try to extract house age
            if 'house_age' in field_lower or 'property_age' in field_lower:
                try:
                    match = re.search(r'(\d+)\s*(?:years?|yrs?)', value_str)
                    if match:
                        house_age = int(match.group(1))
                except:
                    pass
        
        # Check for contradictions
        if age is not None and residence_duration is not None:
            if residence_duration > age:
                fallback_invalid_fields['residence_duration'] = f'You cannot have lived somewhere for {residence_duration} years if you are only {age} years old'
                fallback_notes.append({
                    'type': 'contradiction',
                    'fields': ['date_of_birth' if date_of_birth else 'age', 'residence_duration'],
                    'issue': f'Contradiction: You said you\'re {age} years old but have lived here for {residence_duration} years. This is impossible.',
                    'suggestion': f'Please correct either your age/date of birth or residence duration. If you\'re {age}, you could have lived there at most {age} years.'
                })
        
        if house_age is not None and residence_duration is not None:
            if residence_duration > house_age:
                fallback_invalid_fields['house_age'] = f'You cannot have lived in a house for {residence_duration} years if it is only {house_age} years old'
                fallback_notes.append({
                    'type': 'contradiction',
                    'fields': ['house_age', 'residence_duration'],
                    'issue': f'Contradiction: You said the house is {house_age} years old but you\'ve lived there for {residence_duration} years. The house didn\'t exist!',
                    'suggestion': f'Please check: if the house is {house_age} years old, you could have lived there at most {house_age} years.'
                })
        
        for field_name, field_value in answers_json.items():
            if not isinstance(field_value, str) or len(field_value.strip()) == 0:
                continue
            
            value = field_value.lower().strip()
            field_lower = field_name.lower()
            
            # Check ID number format (should be 11 digits for T&T Electoral ID: YYYYMMDDXXX)
            if 'electoral' in field_lower or 'national_id' in field_lower or 'id_number' in field_lower:
                digits_only = re.sub(r'\D', '', str(field_value))
                if len(digits_only) != 11:
                    fallback_invalid_fields[field_name] = f'Electoral ID must be exactly 11 digits (you entered {len(digits_only)})'
                    fallback_notes.append({
                        'type': 'field_error',
                        'field': field_name,
                        'value': field_value,
                        'issue': f'Trinidad & Tobago Electoral ID must be exactly 11 digits (format: YYYYMMDDXXX). You entered {len(digits_only)} digits.',
                        'suggestion': 'Enter your 11-digit Electoral ID. First 8 digits are your date of birth.',
                        'correct_format': '19901234567'
                    })
                    continue
            
            # Pattern 1: Pure keyboard mashing (same char repeated or random consonant clusters)
            if re.search(r'([qwrtypsdfghjklzxcvbnm])\1{4,}', value):
                fallback_invalid_fields[field_name] = 'Contains gibberish or keyboard mashing'
                fallback_notes.append({
                    'type': 'field_error',
                    'field': field_name,
                    'value': field_value,
                    'issue': 'This looks like random keyboard characters, not a real answer.',
                    'suggestion': 'Please enter a meaningful response. For example, if this is asking for a name, enter your full legal name.'
                })
                continue
            
            # Pattern 2: Random gibberish strings (low vowel ratio + mixed consonants)
            words = value.split()
            for word in words:
                if len(word) >= 6:
                    vowel_count = sum(1 for c in word if c in 'aeiou')
                    consonant_count = sum(1 for c in word if c.isalpha() and c not in 'aeiou')
                    
                    if consonant_count > 0:
                        vowel_ratio = vowel_count / (vowel_count + consonant_count)
                        unique_consonants = len(set(c for c in word if c.isalpha() and c not in 'aeiou'))
                        
                        if vowel_ratio < 0.2 and unique_consonants >= 5:
                            fallback_invalid_fields[field_name] = 'Contains gibberish text'
                            fallback_notes.append({
                                'type': 'field_error',
                                'field': field_name,
                                'value': field_value,
                                'issue': 'This contains what appears to be random characters.',
                                'suggestion': 'Please remove the gibberish and enter only meaningful text.'
                            })
                            break
            
            if field_name in fallback_invalid_fields:
                continue
            
            # Pattern 3: Trailing junk - laughs, emotional expressions
            if re.search(r'\s+(ha){2,}|lol+|yeah\s+m+\s+\w+|[.]{4,}', value):
                fallback_invalid_fields[field_name] = 'Contains irrelevant trailing content'
                fallback_notes.append({
                    'type': 'field_error',
                    'field': field_name,
                    'value': field_value,
                    'issue': 'Your answer has irrelevant content at the end (like "haha" or "lol").',
                    'suggestion': 'Please remove any jokes, laughs, or off-topic comments. Keep only the relevant information.'
                })
                continue
        
        # If we found issues with fallback validation, return them
        if fallback_invalid_fields:
            logger.info(f"Fallback validation caught {len(fallback_invalid_fields)} invalid fields")
            return {
                'all_valid': False,
                'invalid_fields': fallback_invalid_fields,
                'validation_notes': fallback_notes,
                'fallback_validation': True,
                'error': f'AI validation unavailable (using pattern matching): {str(e)}'
            }
        
        # If no obvious issues found and AI failed, return error instead of fail-open
        return {
            'all_valid': False,
            'invalid_fields': {'_system': 'Validation service temporarily unavailable'},
            'validation_notes': [{
                'type': 'system_error',
                'field': '_system',
                'value': '',
                'issue': f'Validation service error: {str(e)}. Please try again in a moment.',
                'suggestion': 'Please wait a moment and try submitting again.'
            }],
            'error': str(e)
        }


def qa_check(
    draft_text: str,
    answers_json: dict,
    policy_json: dict,
    affidavit_type_name: str,
    template_html: str = None,
    system_instructions: str = None,
    affidavit_instructions: str = None
) -> dict:
    """
    Perform quality assurance check on the AI-GENERATED DRAFT.
    
    **PURPOSE:** Validate AI's OUTPUT quality, NOT user's INPUT.
    
    This checks if the AI:
    - Followed the template structure correctly
    - Used proper legal language for Trinidad and Tobago
    - Didn't include disallowed phrases from policy/affidavit instructions
    - Maintained proper formatting and structure
    - Correctly incorporated user answers into the template
    - Followed system instructions and affidavit-specific instructions
    
    **NOT CHECKED HERE (handled by input_suitability checker in clarification phase):**
    - Missing user information
    - Contradictory user answers
    - Age/timeline impossibilities in user input
    
    Args:
        draft_text: The AI-generated draft text
        answers_json: User's original intake answers (for reference)
        policy_json: Affidavit type policy with validation rules
        affidavit_type_name: Name of the affidavit type
        template_html: Template for structure comparison (optional)
        system_instructions: Global system instructions for AI (optional)
        affidavit_instructions: Specific instructions for this affidavit type (optional)
    
    Returns:
        dict: {
            'status': 'approved' | 'needs_review',
            'issues': list of issue objects,
            'flags_json': list of flags for reviewer
        }
    """
    try:
        client = get_openai_client()
        
        # Extract policy components
        validation_rules = policy_json.get('validation_rules', [])
        disallowed_phrases = policy_json.get('disallowed_phrases', [])
        required_sections = policy_json.get('required_sections', [])
        legal_requirements = policy_json.get('legal_requirements', {})
        
        system_prompt = """You are a legal document quality assurance expert for Trinidad and Tobago affidavits.

**YOUR JOB:** Check if the AI-GENERATED DRAFT has REAL, SIGNIFICANT quality issues.

**CRITICAL: BE INTELLIGENT AND PRECISE - AVOID FALSE POSITIVES**

1. **HTML OUTPUT IS EXPECTED:**
   - The draft is in HTML format for PDF generation - this is CORRECT
   - DO NOT flag HTML tags like <p>, <strong>, <br>, <ol>, <li> - these are REQUIRED
   - Only flag if HTML is malformed or broken

2. **DISALLOWED PHRASES - EXACT MATCHING ONLY:**
   - You will receive a list of EXACT phrases to avoid (e.g., "I swear", "I promise")
   - ONLY flag if the draft contains these EXACT phrases or very close variations
   - "I am aware" is NOT the same as "I swear" - don't flag unrelated text
   - Be PRECISE: Check for actual matches, not similar-sounding words

3. **HALLUCINATION CHECK - CROSS-REFERENCE USER DATA:**
   - You will receive the user's answers
   - BEFORE flagging hallucination, CHECK if the data exists in user answers
   - Example: If user provided "city: Muzaffargarh", DO NOT flag it as hallucination
   - ONLY flag if AI invented facts that are NOT in user answers at all

4. **TEMPLATE COMPLIANCE - BE SPECIFIC:**
   - Don't flag "Entire document" - that's useless
   - If flagging template issues, quote the SPECIFIC section that's wrong
   - Only flag if sections are missing or structure is significantly different

5. **LEGAL LANGUAGE - FOCUS ON SUBSTANCE:**
   - Check if legal terminology is correct
   - Check if declarations are properly worded
   - Don't nitpick minor formatting preferences

6. **FORMATTING - MAJOR ISSUES ONLY:**
   - Flag inconsistent capitalization of names/places
   - Flag broken structure or missing signature blocks
   - Don't flag HTML tags or minor spacing

**DO NOT FLAG:**
- HTML tags (they're required for PDF generation)
- Data that came from user input (check answers_json first)
- Phrases that are NOT exact matches to disallowed list
- Minor stylistic preferences
- Vague issues like "entire document"

**ONLY FLAG REAL PROBLEMS:**
- Actually missing required sections
- Actually incorrect legal language
- Actually hallucinated facts not in user data
- Actually present disallowed phrases (exact matches)
- Actually broken formatting (not HTML structure)

Respond with JSON:
{
    "status": "approved" | "needs_review",
    "issues": [
        {
            "type": "template_deviation" | "legal_error" | "disallowed_phrase" | "formatting" | "hallucination" | "instruction_violation",
            "severity": "low" | "medium" | "high",
            "description": "Specific, actionable description of the REAL problem",
            "location": "Exact quote showing the problem (not 'entire document')",
            "suggestion": "How to fix this specific issue"
        }
    ],
    "summary": "Brief, factual summary"
}

**DEFAULT TO APPROVED:** If the draft is reasonable and functional, return "approved" even if not perfect.
Only return "needs_review" for SIGNIFICANT, REAL issues that would make the document unprofessional or incorrect.
"""
        
        # Parse disallowed phrases from multiple sources
        disallowed_list = []
        if disallowed_phrases:
            if isinstance(disallowed_phrases, list):
                disallowed_list = disallowed_phrases
            elif isinstance(disallowed_phrases, str):
                try:
                    disallowed_list = json.loads(disallowed_phrases)
                except:
                    disallowed_list = [p.strip() for p in disallowed_phrases.split(',') if p.strip()]

        user_prompt = f"""
Review this AI-generated {affidavit_type_name} draft for QUALITY:

**AI-GENERATED DRAFT:**
{draft_text}

**USER ANSWERS (for reference - check if AI used them correctly without adding extra info):**
{json.dumps(answers_json, indent=2)}

**CRITICAL: CHECK FOR CLARIFICATIONS**
If the USER ANSWERS above contain "PREVIOUS_CLARIFICATIONS":
- The user has CORRECTED original contradictory data after being asked for clarification
- PARSE the "User Answer" in the clarification to extract the FINAL CORRECT values
- When checking for hallucinations, use CLARIFIED values as the source of truth, NOT original fields
- Example: If original had age=22, years=25, but clarification says "I am 25 and lived for 3 years":
  → The AI should use age=25 in draft (this is CORRECT, not a hallucination)
  → The AI should use years=3 in draft (this is CORRECT, not a hallucination)
- DO NOT flag these as hallucinations - the AI correctly used the user's clarified corrections
- Only flag if AI used values that contradict BOTH the original data AND the clarifications

**SYSTEM INSTRUCTIONS (AI must follow these):**
{system_instructions if system_instructions else "Standard professional legal document generation"}

**AFFIDAVIT-SPECIFIC INSTRUCTIONS (AI must follow these):**
{affidavit_instructions if affidavit_instructions else "No specific instructions provided"}

**DISALLOWED PHRASES (AI must NOT use these EXACT phrases or close variations):**
{json.dumps(disallowed_list) if disallowed_list else "None specified"}

**CRITICAL INSTRUCTIONS FOR CHECKING:**

1. **HTML IS EXPECTED - DON'T FLAG IT:**
   - The draft uses HTML tags like <p>, <strong>, <ol>, <li> for PDF generation
   - This is CORRECT and REQUIRED - do not flag HTML tags as formatting issues

2. **DISALLOWED PHRASES - EXACT MATCH ONLY:**
   - Check if draft contains the EXACT phrases listed above
   - Example: If list has "I swear", only flag if draft has "I swear"
   - "I am aware" is NOT the same as "I swear" - don't flag unrelated text
   - Be precise: word-for-word match or very close variation only

3. **HALLUCINATION CHECK - CROSS-REFERENCE FIRST:**
   - BEFORE flagging hallucination, check if the data is in USER ANSWERS above
   - If user provided "city: Muzaffargarh", that's NOT a hallucination
   - If user provided "declaration_location: Bacolet Street", that's NOT a hallucination
   - ONLY flag if AI made up facts that are COMPLETELY ABSENT from user input

4. **BE SPECIFIC WITH LOCATIONS:**
   - Don't quote "Entire document" - that's useless for reviewers
   - Quote the EXACT problematic sentence or phrase
   - Give actionable, specific feedback

**YOUR TASK:**
1. Read the user answers to know what data the user provided
2. Read the draft and check for REAL, SIGNIFICANT issues only
3. Cross-reference: Is flagged "hallucination" actually from user data? If yes, DON'T FLAG
4. Check disallowed phrases: Is it an EXACT match? If no, DON'T FLAG
5. Default to "approved" unless there are REAL problems

Focus on QUALITY and ACCURACY. Avoid nitpicking minor issues.
"""
        
        response = client.chat.completions.create(
            model=settings.OPENAI_QA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        usage = response.usage
        
        logger.info(f"QA check completed for {affidavit_type_name}: {result.get('status')}")
        
        return {
            'status': result.get('status', 'needs_review'),
            'issues': result.get('issues', []),
            'flags_json': result.get('issues', []),
            'summary': result.get('summary', ''),
            'prompt_tokens': usage.prompt_tokens if usage else 0,
            'completion_tokens': usage.completion_tokens if usage else 0,
            'total_tokens': usage.total_tokens if usage else 0,
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse QA response as JSON: {e}")
        return {
            'status': 'needs_review',
            'issues': [{'type': 'system_error', 'description': 'QA check failed to parse'}],
            'flags_json': [],
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        }
    except Exception as e:
        logger.error(f"Error in QA check: {e}")
        return {
            'status': 'needs_review',
            'issues': [{'type': 'system_error', 'description': str(e)}],
            'flags_json': [],
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        }


def process_request(request_obj) -> dict:
    """
    Full AI processing pipeline for a request.
    1. Generate draft using GPT-4o-mini
    2. Run QA check using GPT-4o
    3. Update request status based on results
    
    Args:
        request_obj: Request model instance
    
    Returns:
        dict: Processing result with status and any issues
    """
    from ..models import Request
    
    affidavit_type = request_obj.affidavit_type
    
    # Prepare answers with any clarifications
    final_answers = request_obj.answers_json.copy() if request_obj.answers_json else {}
    user_edits = request_obj.user_edits_json or {}
    clarifications = user_edits.get('clarifications', [])
    
    if clarifications:
        # Format clarifications for the AI
        clarification_text = []
        for c in clarifications:
            clarification_text.append(f"Clarification Q: {c.get('question', '')}\nUser Answer: {c.get('response', '')}")
        
        final_answers['PREVIOUS_CLARIFICATIONS'] = "\n---\n".join(clarification_text)
    
    # ===== DEBUG LOGGING - PROCESS_REQUEST =====
    logger.info("=" * 80)
    logger.info(f"[PROCESS_REQUEST] Processing request: {request_obj.request_code}")
    logger.info(f"[PROCESS_REQUEST] Affidavit Type: {affidavit_type.name} (ID: {affidavit_type.id})")
    logger.info(f"[PROCESS_REQUEST] Clarifications found: {len(clarifications)}")
    logger.info(f"[PROCESS_REQUEST] affidavit_type.template_html length: {len(affidavit_type.template_html) if affidavit_type.template_html else 0}")
    logger.info(f"[PROCESS_REQUEST] affidavit_type.template_html first 300 chars: {affidavit_type.template_html[:300] if affidavit_type.template_html else 'EMPTY/NONE'}")
    logger.info(f"[PROCESS_REQUEST] affidavit_type.disallowed_phrases: {affidavit_type.disallowed_phrases}")
    logger.info(f"[PROCESS_REQUEST] affidavit_type.policy_json keys: {list(affidavit_type.policy_json.keys()) if affidavit_type.policy_json else 'None'}")
    logger.info("=" * 80)
    # ===== END DEBUG LOGGING =====
    
    # Step 1: Generate draft with new parameters
    draft_result = draft_affidavit(
        answers_json=final_answers,
        policy_json=affidavit_type.policy_json,
        affidavit_type_name=affidavit_type.name,
        scenario_library=affidavit_type.scenario_library,
        template_html=affidavit_type.template_html,
        disallowed_phrases=affidavit_type.disallowed_phrases
    )
    
    if not draft_result['success']:
        request_obj.status = Request.Status.NEEDS_REVIEW
        request_obj.save()
        return {
            'success': False,
            'status': 'needs_review',
            'error': draft_result['error']
        }
    
    # Save the draft
    request_obj.draft_text = draft_result['draft_text']
    
    # All requests go to human review (no QA flagging - reviewer reads completely)
    if request_obj.affidavit_type.is_instant_mode:
        # Instant mode: auto-approve
        request_obj.status = Request.Status.APPROVED
        request_obj.final_text = request_obj.draft_text
    else:
        # Normal mode: needs human review
        request_obj.status = Request.Status.NEEDS_REVIEW
    
    request_obj.save()
    
    return {
        'success': True,
        'status': request_obj.status,
        'draft_text': request_obj.draft_text,
        'issues': []
    }


def get_default_draft_system_prompt() -> str:
    """Return default system prompt for drafting if not specified in policy."""
    return """You are an expert legal document drafter specializing in affidavits.
Your task is to create formal, legally-sound affidavit documents based on information provided.

Guidelines:
1. Use clear, formal legal language
2. Include proper affidavit structure (heading, body, attestation)
3. Be precise with dates, names, and facts
4. Include appropriate legal declarations
5. Format for easy reading and notarization
6. Do not include any false or misleading statements
7. Leave signature lines and date fields for completion

Always maintain professional tone and legal accuracy."""
