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
    affidavit_type_name: str
) -> dict:
    """
    Validate user input against template requirements BEFORE draft generation.
    Checks if the user has provided enough specific information to fill the template.
    
    Args:
        answers_json: User's intake form answers
        template_html: The HTML template to be filled
        affidavit_type_name: Name of the affidavit type
        
    Returns:
        dict: {
            'is_suitable': bool,
            'issues': list of issue objects,
            'clarification_question': str (if is_suitable is False),
            'prompt_tokens': int,
            'completion_tokens': int,
            'total_tokens': int
        }
    """
    try:
        client = get_openai_client()
        
        # If no template, we can't really validate against it, so assume suitable
        if not template_html:
            return {
                'is_suitable': True,
                'issues': [],
                'clarification_question': '',
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }
            
        system_prompt = """You are an intelligent legal document validation expert with strong analytical skills.
Your task is to check if the user's input is LOGICALLY CONSISTENT and contains sufficient information to fill the provided affidavit template.

**CRITICAL: HANDLING CLARIFICATIONS - STRICT LOOP PREVENTION**

BEFORE flagging ANY issue:
1. Check if 'PREVIOUS_CLARIFICATIONS' field exists in user input
2. If it exists and contains ANY text, the user HAS ALREADY RESPONDED to a clarification
3. Parse the clarification to extract the FINAL CORRECT VALUES
4. UPDATE your analysis to use the clarified values as ABSOLUTE TRUTH
5. NEVER ask the same question in different words
6. NEVER ask for "confirmation" or "verification" of clarified data

**CLARIFICATION RESPONSE RULES:**
- If user said "I am 40" or "my age is 40" → AGE = 40 (FINAL, DO NOT QUESTION)
- If user said "3 years" or "I lived for 3 years" → DURATION = 3 (FINAL, DO NOT QUESTION)
- If user said "house is 50" or "50 years old" → HOUSE_AGE = 50 (FINAL, DO NOT QUESTION)

**MANDATORY LOOP PREVENTION:**
- If PREVIOUS_CLARIFICATIONS exists → User already answered → Return is_suitable=TRUE UNLESS there's a COMPLETELY NEW contradiction
- If you already asked about age vs residence → DO NOT ask again about age vs house age (these are related, use clarified age)
- If any value was clarified, treat it as THE TRUTH and check if OTHER unclarified fields contradict it
- Maximum 1 clarification request per unique logical contradiction

**WHEN TO RETURN is_suitable=TRUE:**
- PREVIOUS_CLARIFICATIONS exists AND no new contradictions found → STOP, APPROVE
- User provided ANY clarification response → ASSUME they fixed the issue, check for NEW issues only
- You've asked about this combination of fields before → STOP, APPROVE

**INTELLIGENT CONTRADICTION DETECTION**

You MUST analyze ALL answers together and check for logical impossibilities:

1. **AGE vs DURATION CONTRADICTIONS:**
   - If user says age is X years, they CANNOT have done anything for more than X years
   - Example: "I am 23 years old" + "I have lived at this address for 30 years" = IMPOSSIBLE
   - BUT if user clarified "my age is 40", then "25 years residence" is POSSIBLE - accept it!

2. **DATE OF BIRTH vs AGE CALCULATION:**
   - If DOB is provided, CALCULATE the actual age from it
   - If calculated age differs from stated age by more than 1 year, ask user which is correct
   - The CALCULATED age should be used in the affidavit if provided

3. **DATE OF BIRTH vs DURATION:**
   - Check ALL duration claims against the birth year

4. **EVENT DATES vs AGE:**
   - Marriage date, purchase date, employment start date - all must be AFTER birth date

5. **TIMELINE CONSISTENCY:**
   - Start dates must be before end dates

**CALCULATION INSTRUCTIONS:**
- Current year for calculations: Use the CURRENT DATE provided
- Age = Current Year - Birth Year
- Duration = Current Year - Start Year
- If user clarified a value, USE IT - don't recalculate or question it

You must respond with a JSON object in this exact format:
{
    "is_suitable": boolean,
    "issues": [
        {
            "field": "field_name_or_description",
            "type": "age_duration_contradiction" | "dob_age_mismatch" | "timeline_impossible" | "missing_info",
            "reason": "Clear explanation: 'User stated age 23, but claims 30 years residence. This is impossible.'",
            "found_values": {"stated_age": "23", "claimed_duration": "30 years"}
        }
    ],
    "clarification_question": "A detailed, context-aware question that EXPLAINS THE PROBLEM clearly with actual numbers from user input",
    "clarification_example": "Example answer: '40' or 'I am 40 years old' or '3 years'",
    "calculated_age": null or number (if DOB was provided, include the calculated age here)
}

**CRITICAL RULES FOR clarification_question:**
1. ALWAYS state the SPECIFIC contradiction with ACTUAL NUMBERS from user input
2. EXPLAIN why it's impossible (e.g., "You cannot have lived anywhere longer than you've been alive")
3. Provide TWO contextual examples showing both possible corrections:
   - Example if user wants to keep one value: "If you are actually 43 years old (and the 25 years is correct)..."
   - Example if user wants to correct the other: "If your age of 23 is correct, then residence should be 23 years or less"
4. Make it CRYSTAL CLEAR what the user needs to fix

Example of a GOOD clarification_question:
"You stated your age is 23 years old, but you claim to have lived at this address for 25 years. This is mathematically impossible - you cannot have lived anywhere longer than you've been alive.

Please provide the correct information:
- If you are actually older (to match 25 years residence): 'I am 45 years old'
- If your age 23 is correct: 'I have lived here for 3 years' (or any number ≤ 23)"

Rules:
1. "is_suitable" = TRUE if all data is internally consistent OR user has clarified the issues
2. "is_suitable" = FALSE only if there are NEW contradictions after considering clarifications
3. If user answered a clarification, ACCEPT it and validate other fields against it
4. NEVER ask the same question twice
5. Make clarification_question DETAILED and CONTEXT-AWARE with actual user numbers
6. If PREVIOUS_CLARIFICATIONS exists, those issues are RESOLVED - don't flag them again
"""

        # Get current date for calculations
        from datetime import date
        current_date = date.today()
        current_year = current_date.year
        current_date_str = current_date.isoformat()

        user_prompt = f"""
Validate this user input for LOGICAL CONSISTENCY:

**AFFIDAVIT TYPE:** {affidavit_type_name}

**CURRENT DATE:** {current_date_str}
**CURRENT YEAR:** {current_year}

**USER INPUT TO VALIDATE:**
{json.dumps(answers_json, indent=2)}

**YOUR TASK:**
1. **FIRST: Check if 'PREVIOUS_CLARIFICATIONS' exists**
   - If YES: Extract the user's corrected values from their response
   - Parse natural language answers (e.g., "I am 40" means age=40, "3 years" means duration=3)
   - Treat these as ABSOLUTE TRUTH - the user has already corrected the issue

2. **Build a "RESOLVED VALUES" dictionary:**
   - If user clarified age → resolved_age = extracted_value
   - If user clarified duration → resolved_duration = extracted_value
   - Use these RESOLVED values for ALL subsequent checks

3. **Check for NEW contradictions ONLY:**
   - Use resolved_age (if exists) instead of original age
   - Use resolved_duration (if exists) instead of original duration
   - ONLY flag issues that involve UNCLARIFIED fields

4. **CRITICAL DECISION LOGIC:**
   - If PREVIOUS_CLARIFICATIONS exists AND no new contradictions → is_suitable = TRUE (STOP THE LOOP)
   - If you detect the same type of issue as before (age vs duration) → is_suitable = TRUE (user already answered)
   - Only return is_suitable = FALSE for COMPLETELY NEW contradictions

**EXAMPLE:**
User originally said: age=23, years_residing=25
First clarification asked: "age vs residence contradiction"
User responded: "I am 40 and lived for 3 years"
→ RESOLVED: age=40, years_residing=3
→ NOW CHECK: Is 40 vs 3 valid? YES → is_suitable=TRUE
→ DO NOT ask about age/residence again, even if you see house_age=50
→ If house_age=50 contradicts NOTHING, return is_suitable=TRUE

Return is_suitable=TRUE if clarifications resolve all issues.
Return is_suitable=FALSE ONLY if there are NEW unresolved contradictions.
"""

        response = client.chat.completions.create(
            model=settings.OPENAI_QA_MODEL,  # Use QA model (GPT-4o) for better reasoning
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        usage = response.usage
        
        # Log with more detail about what was found
        calculated_age = result.get('calculated_age')
        issues_count = len(result.get('issues', []))
        logger.info(f"Input suitability check for {affidavit_type_name}: suitable={result.get('is_suitable')}, issues={issues_count}, calculated_age={calculated_age}")
        
        return {
            'is_suitable': result.get('is_suitable', True),
            'issues': result.get('issues', []),
            'clarification_question': result.get('clarification_question', ''),
            'clarification_example': result.get('clarification_example', ''),
            'calculated_age': calculated_age,  # Pass this to be used in draft instead of user-stated age
            'prompt_tokens': usage.prompt_tokens if usage else 0,
            'completion_tokens': usage.completion_tokens if usage else 0,
            'total_tokens': usage.total_tokens if usage else 0,
        }
        
    except Exception as e:
        logger.error(f"Error in input suitability check: {e}")
        # Fail open - assume suitable if check fails to avoid blocking users
        return {
            'is_suitable': True,
            'issues': [{'type': 'system_error', 'description': str(e)}],
            'clarification_question': '',
            'clarification_example': '',
            'calculated_age': None,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
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
        request_obj.qa_flags_json = [{
            'type': 'draft_error',
            'description': draft_result['error']
        }]
        request_obj.save()
        return {
            'success': False,
            'status': 'needs_review',
            'error': draft_result['error']
        }
    
    # Save the draft
    request_obj.draft_text = draft_result['draft_text']
    
    # Step 2: Run QA check (validates AI OUTPUT quality, not user input)
    # Get global system instructions if available
    from ..models import AIBaseInstruction
    try:
        system_instructions = AIBaseInstruction.get_active().instruction_text
    except Exception as e:
        logger.warning(f"Could not get base instruction for QA: {e}")
        system_instructions = None
    
    qa_result = qa_check(
        draft_text=draft_result['draft_text'],
        answers_json=final_answers,
        policy_json=request_obj.affidavit_type.policy_json,
        affidavit_type_name=request_obj.affidavit_type.name,
        template_html=affidavit_type.template_html,
        system_instructions=system_instructions,
        affidavit_instructions=affidavit_type.policy_json.get('ai_instructions', '')
    )
    
    # Step 3: Update request based on QA result
    request_obj.qa_flags_json = qa_result['flags_json']
    
    if qa_result['status'] == 'approved':
        # Check if affidavit type is in instant mode
        if request_obj.affidavit_type.is_instant_mode:
            request_obj.status = Request.Status.APPROVED
            request_obj.final_text = request_obj.draft_text
        else:
            # Even if approved, still needs human review unless instant mode
            request_obj.status = Request.Status.NEEDS_REVIEW
    
    else:  # needs_review (QA no longer returns needs_clarification)
        request_obj.status = Request.Status.NEEDS_REVIEW
    
    request_obj.save()
    
    return {
        'success': True,
        'status': request_obj.status,
        'draft_text': request_obj.draft_text,
        'issues': qa_result.get('issues', [])
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
