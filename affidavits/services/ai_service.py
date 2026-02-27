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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# Lazy import openai to avoid import errors if not installed
openai_client = None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
    before_sleep=lambda retry_state: logger.warning(f"OpenAI API retry {retry_state.attempt_number}/3 after error: {retry_state.outcome.exception()}")
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


# ---------------------------------------------------------------------------
# Helper: filter out answers for conditionally-hidden fields (show_if)
# ---------------------------------------------------------------------------

def _is_field_visible(field: dict, answers: dict, schema: list) -> bool:
    """Return True if a field should be shown given current answers."""
    show_if = field.get('show_if')
    if not show_if:
        return True

    parent_id = show_if.get('field', '')
    required_value = show_if.get('value')

    parent_answer = answers.get(parent_id)
    if parent_answer is None:
        for q in schema:
            qid = q.get('id') or q.get('field_name', '')
            if qid == parent_id or q.get('field_name') == parent_id:
                parent_answer = answers.get(qid)
                break

    if not required_value or required_value == '':
        return parent_answer is not None and parent_answer != '' and parent_answer is not False

    if isinstance(parent_answer, list):
        check = required_value if isinstance(required_value, list) else [required_value]
        return any(v in parent_answer for v in check)

    if isinstance(required_value, list):
        return parent_answer in required_value

    return parent_answer == required_value


def _filter_visible_answers(answers: dict, intake_schema: list) -> dict:
    """Return a copy of answers containing only keys for visible fields."""
    visible_ids = set()
    for field in intake_schema:
        fid = field.get('id') or field.get('field_name', '')
        if _is_field_visible(field, answers, intake_schema):
            visible_ids.add(fid)
    # Always keep internal keys (e.g. _calculated_age)
    return {k: v for k, v in answers.items() if k in visible_ids or k.startswith('_')}


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

        response = call_openai_with_retry(
            client=client,
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


def pre_fill_template(
    template_html: str,
    answers_json: dict,
    placeholder_mapping: Dict[str, str] = None,
    intake_schema: List[dict] = None,
) -> str:
    """
    Mechanically replace {{placeholders}} in the template with actual user answers.
    
    Resolution order for each placeholder:
      1. Explicit mapping  (placeholder_mapping[placeholder] -> question_id -> answer)
      2. Direct match      (answers_json[placeholder])
      3. Field-name match  (intake_schema field_name -> id -> answer)
    
    Placeholders that can't be resolved are left as-is so AI can still attempt them.
    
    Args:
        template_html: HTML template with {{placeholder}} tokens
        answers_json: User's intake form answers keyed by question id
        placeholder_mapping: Optional dict mapping placeholder -> question_id
        intake_schema: Optional intake schema list for field_name lookups
    
    Returns:
        Template with as many placeholders replaced as possible
    """
    import re
    if not template_html:
        return template_html

    mapping = placeholder_mapping or {}

    # Build a reverse lookup: field_name -> question_id from intake_schema
    field_name_to_id = {}
    if intake_schema:
        for q in intake_schema:
            fname = q.get('field_name', '')
            qid = q.get('id', '')
            if fname and fname != qid:
                field_name_to_id[fname] = qid

    def _resolve(placeholder: str) -> Optional[str]:
        # 1. Explicit mapping
        mapped_qid = mapping.get(placeholder, '')
        if mapped_qid:
            val = answers_json.get(mapped_qid)
            if val not in (None, ''):
                return str(val)

        # 2. Direct match
        val = answers_json.get(placeholder)
        if val not in (None, ''):
            return str(val)

        # 3. Field-name reverse lookup
        qid = field_name_to_id.get(placeholder, '')
        if qid:
            val = answers_json.get(qid)
            if val not in (None, ''):
                return str(val)

        return None

    placeholders = re.findall(r'\{\{(\w+)\}\}', template_html)
    filled = template_html
    replaced_count = 0

    for ph in dict.fromkeys(placeholders):  # unique, ordered
        resolved = _resolve(ph)
        if resolved is not None:
            filled = filled.replace('{{' + ph + '}}', resolved)
            replaced_count += 1

    logger.info(
        f"[PRE_FILL] Replaced {replaced_count}/{len(set(placeholders))} placeholders"
    )
    return filled


def draft_affidavit(
    answers_json: dict,
    policy_json: dict,
    affidavit_type_name: str,
    scenario_library: List[dict] = None,
    template_html: str = '',
    disallowed_phrases: List[str] = None,
    placeholder_mapping: Dict[str, str] = None,
    intake_schema: List[dict] = None,
    affidavit_type_id: int = None,
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

        # STEP 0.5: Strip answers for fields hidden by show_if so the
        # drafter only sees the user's visible answers.
        if intake_schema:
            answers_json = _filter_visible_answers(answers_json, intake_schema)
        
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
        
        # STEP 1: Pre-fill template mechanically — stored as debug reference only.
        # The AI receives the ORIGINAL template (with {{placeholder}} tokens) so its
        # context-aware rephrasing instructions can fire properly.  If we pre-fill first,
        # raw user input like "land is near uphill POS" gets pasted verbatim and the AI
        # then blindly reproduces it ("...situate at land is near uphill POS...") because
        # it is told to "follow the template exactly as-is".
        if template:
            prefilled_template = pre_fill_template(
                template_html=template,
                answers_json=answers_json,
                placeholder_mapping=placeholder_mapping,
                intake_schema=intake_schema,
            )
            logger.info(f"[DRAFT_AFFIDAVIT] Pre-fill reference (first 300): {prefilled_template[:300]}")
            # `template` is intentionally NOT reassigned — AI sees original {{placeholders}}
        
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

        # Inject scenario_branches guidance so the drafter knows which
        # template sections / paragraphs to include for the matched scenario.
        scenario_branches = policy_json.get('scenario_branches', {})
        if scenario_branches:
            # Figure out which branch the user selected by scanning answers
            # against the intake_schema selector fields (fields without show_if
            # that are of type select and whose id appears as a key in branches).
            matched_branch = None
            for branch_key, branch_info in scenario_branches.items():
                key_fields = branch_info.get('key_fields', [])
                # Check if any selector answer value matches this branch
                for q in (intake_schema or []):
                    qid = q.get('id', '')
                    if q.get('type') == 'select' and not q.get('show_if'):
                        user_val = answers_json.get(qid, '')
                        if isinstance(user_val, str):
                            # Match if the user's selection matches the branch key
                            val_normalized = user_val.lower().replace(' ', '_').replace('-', '_')
                            if val_normalized == branch_key or user_val == branch_key:
                                matched_branch = branch_info
                                break
                if matched_branch:
                    break

            if matched_branch:
                scenario_context += f"\n**Scenario Branch Details:**\n"
                scenario_context += f"- Description: {matched_branch.get('description', '')}\n"
                sections = matched_branch.get('template_sections', [])
                if sections:
                    scenario_context += f"- Relevant template sections: {', '.join(sections)}\n"
                scenario_context += (
                    "- ONLY include template paragraphs/sections relevant to this scenario.\n"
                    "- OMIT paragraphs that belong to other scenarios.\n"
                )
                logger.info(f"[DRAFT_AFFIDAVIT] Matched scenario branch: {matched_branch.get('description', 'N/A')}")
        
        # Build field context from intake_schema: help_text + placeholder + user value
        field_context = ""
        if intake_schema:
            field_hints = []
            for q in intake_schema:
                qid = q.get('id', q.get('field_name', ''))
                help_text = (q.get('help_text') or '').strip()
                placeholder = (q.get('placeholder') or '').strip()
                field_value = answers_json.get(qid, '')
                if not field_value:
                    continue
                hint_lines = [f"• {{{{{qid}}}}}: User entered → \"{field_value}\""]
                if help_text:
                    hint_lines.append(f"  ↳ Field asks for: {help_text}")
                if placeholder:
                    hint_lines.append(f"  ↳ Complete answer looks like: \"{placeholder}\"")
                if help_text or placeholder:
                    hint_lines.append(
                        "  ↳ ACTION: Do NOT use the raw input verbatim. "
                        "Read the field meaning + example above, then EXPAND the user's answer "
                        "into a complete, grammatically correct legal phrase that fits the template sentence."
                    )
                field_hints.append("\n".join(hint_lines))

            if field_hints:
                field_context = (
                    "╔══════════════════════════════════════════════════════════════╗\n"
                    "║  STEP 1 — FIELD MEANINGS (PROCESS THIS BEFORE THE TEMPLATE)  ║\n"
                    "╚══════════════════════════════════════════════════════════════╝\n"
                    "Each field below shows:\n"
                    "  • What the user typed (raw — may be short/incomplete/informal)\n"
                    "  • What the field ACTUALLY asks for (from help text)\n"
                    "  • What a COMPLETE answer looks like (from example/placeholder)\n\n"
                    "YOUR RULE FOR EVERY PLACEHOLDER:\n"
                    "  1. Look up the field in this list.\n"
                    "  2. Read the template sentence surrounding the placeholder.\n"
                    "  3. Use the field meaning + example to EXPAND the user's raw input into a\n"
                    "     full legal phrase — do NOT paste raw input verbatim.\n"
                    "  4. If the user gave just a name or single word, use the example format\n"
                    "     to complete the thought (e.g. 'my wife' + 'responsible for taxes' →\n"
                    "     'my wife, who is responsible for the payment of taxes on the said land').\n\n"
                    + "\n\n".join(field_hints)
                    + "\n"
                )

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
- **FOLLOW THE TEMPLATE EXACTLY AS IS.**
- Do not change, remove, or reorder any static text.
- Do not add headers like "AFFIDAVIT" or subtitles if not in the template.
- Start the document EXACTLY as the template starts (e.g., "REPUBLIC OF TRINIDAD AND TOBAGO:")
- For the commissioner/attestation section, use the EXACT format from the template.
- DO NOT modernize or "improve" the commissioner section format.
- Replace ONLY the {{placeholder}} values with actual data.
- Preserve all static text, legal language, and formatting exactly as shown.
- **STRICTLY FOLLOW THE TEMPLATE'S SPACING AND STRUCTURE.**

**HEADING & BOLD FORMATTING (CRITICAL):**
- Any standalone line that is a document title or heading (e.g. "STATUTORY DECLARATION", "REPUBLIC OF TRINIDAD AND TOBAGO", "SCHEDULE") MUST be wrapped in <strong> tags.
- If the template already has <strong> or <b> tags, preserve them exactly.
- Do NOT apply bold to regular paragraph text — only to clear headings/titles.

**NUMBERED AND BULLETED LISTS (CRITICAL):**
- If the template contains items prefixed with numbers (1. 2. 3.) or letters (a. b. c.), render them using <ol><li>...</li></ol> HTML tags.
- If the template contains bullet items (-, •), render them using <ul><li>...</li></ol> HTML tags.
- If the template uses "That I am...", "That I have...", "That this..." paragraph style (NOT numbered), preserve that paragraph format — do NOT convert those paragraphs to a numbered list.
- Never drop or merge list items — every numbered/bulleted item in the template must appear as its own <li>.
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
        
        user_prompt = f"""Generate a formal {affidavit_type_name}.

OUTPUT: Clean HTML for PDF. No <html>/<head>/<body>. Use <p>, <ol>, <li>, <strong>, <sup>.
{calculated_age_note}
{field_context}
╔══════════════════════════════════════════════════════════════╗
║  STEP 2 — RAW APPLICANT ANSWERS (do NOT paste verbatim)     ║
╚══════════════════════════════════════════════════════════════╝
These are the user's unedited inputs. Always cross-reference with STEP 1 field meanings
before using any value — raw input is often incomplete or informal.
{json.dumps(answers_json, indent=2)}

╔══════════════════════════════════════════════════════════════╗
║  STEP 3 — TEMPLATE (replace placeholders only)              ║
╚══════════════════════════════════════════════════════════════╝
{template_section}
{template_format_instructions}
╔══════════════════════════════════════════════════════════════╗
║  STEP 4 — PLACEHOLDER FILLING RULES                         ║
╚══════════════════════════════════════════════════════════════╝
For EVERY {{{{placeholder}}}} in the template:
  A. Find the field in STEP 1 and read its meaning + example.
  B. Read the template sentence around the placeholder.
  C. EXPAND the user's raw input using the field meaning + example so it:
     - Fills the full meaning the field asks for
     - Fits grammatically in the template sentence
     - Is written in proper legal English

GENERAL EXPANSION RULES (apply to ALL affidavit types):
  • Short name/person answer + field asking for responsibility/role →
    expand to "[person], who is responsible for [what the field says]"
    e.g. user: "my wife" | field: "responsible for taxes" →
    output: "my wife, who is responsible for the payment of taxes on the said land"

  • Raw verb phrase that would break the surrounding sentence →
    rephrase to a legal noun/gerund phrase
    e.g. user: "modifying my house" in "...process of {{{{X}}}} to the house..." →
    output: "effecting modifications"

  • Ownership answer that is a full sentence but placeholder sits after a preposition →
    convert to a grammatical noun phrase
    e.g. user: "my wife is the owner" in "...process of work to {{{{X}}}}, where..." →
    output: "my wife's property" or "a property belonging to my wife"

  • Incomplete duration → spell out in legal format
    e.g. "3 years" → "three (3) years"

  • Fix ALL capitalisation: names, streets, cities, months → Title Case
    e.g. "shahmeer" → "Shahmeer", "near uphill station" → "near Uphill Station"

  • Fix all spelling typos silently.

  • If _calculated_age is provided, use THAT age value — ignore any user-entered age field.

{scenario_context}

REQUIREMENTS:
1. {"Follow the template EXACTLY — replace only placeholders, touch nothing else." if template else "Use formal legal language for a Trinidad and Tobago affidavit."}
2. Do NOT add sections, headers, or text not in the template.
3. Preserve all static legal text, spacing, bold tags, and list structure exactly.
4. Do NOT add 'SWORN/AFFIRMED before me' blocks — use the template's commissioner section.

{f"Required sections: {', '.join(required_sections)}" if required_sections else ""}
{f"Do NOT use these phrases: {', '.join(phrases_to_avoid)}" if phrases_to_avoid else ""}
{examples_text}
Generate the complete affidavit HTML:
"""

        # Inject reviewer feedback from learning loop
        if affidavit_type_id:
            try:
                from .feedback_service import get_drafter_feedback
                feedback_section = get_drafter_feedback(affidavit_type_id)
                if feedback_section:
                    user_prompt += feedback_section
                    logger.info(f"[DRAFT_AFFIDAVIT] Injected {len(feedback_section)} chars of reviewer feedback")
            except Exception as e:
                logger.warning(f"[DRAFT_AFFIDAVIT] Failed to load reviewer feedback: {e}")
        
        # ===== LOG THE FULL PROMPT =====
        logger.info("=" * 80)
        logger.info("[DRAFT_AFFIDAVIT] FULL USER PROMPT:")
        logger.info(user_prompt)
        logger.info("=" * 80)
        logger.info("[DRAFT_AFFIDAVIT] SYSTEM PROMPT (base_instruction):")
        logger.info(base_instruction[:500] + "..." if len(base_instruction) > 500 else base_instruction)
        logger.info("=" * 80)
        # ===== END PROMPT LOG =====
        
        response = call_openai_with_retry(
            client=client,
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


def apply_validation_rules(answers_json: dict, rules: list) -> tuple:
    """
    Apply custom validation rules to user answers.
    
    Args:
        answers_json: User's intake form answers
        rules: List of validation rule objects from affidavit type settings
        
    Returns:
        tuple: (invalid_fields dict, validation_notes list)
    """
    import re
    from datetime import datetime, date as dt_date
    
    invalid_fields = {}
    validation_notes = []
    
    if not rules:
        return invalid_fields, validation_notes
    
    # Helper function to extract numeric value from text
    def extract_number(value_str: str, field_name: str = '') -> Optional[float]:
        """Extract numeric value from text, handling phrases like 'over 3 years', 'since 1990'.

        Special case: for DOB/birth date fields, convert date strings to age-in-years so they can be
        used in numeric comparison rules (e.g., residence_duration <= age).
        """
        if not value_str:
            return None
        
        value_str = str(value_str).lower().strip()

        field_lower = str(field_name or '').lower().strip()
        if any(kw in field_lower for kw in ['date_of_birth', 'dob', 'birth_date', 'birthdate']):
            try:
                # YYYY-MM-DD
                if re.match(r'^\d{4}-\d{2}-\d{2}$', value_str):
                    dob = datetime.strptime(value_str, '%Y-%m-%d').date()
                    today = dt_date.today()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    return float(age)
                # DD-MMM-YYYY (e.g. 23-Feb-2001)
                if re.match(r'^\d{1,2}-[A-Za-z]{3}-\d{4}$', value_str):
                    dob = datetime.strptime(value_str, '%d-%b-%Y').date()
                    today = dt_date.today()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    return float(age)
                # DD/MM/YYYY
                if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', value_str):
                    dob = datetime.strptime(value_str, '%d/%m/%Y').date()
                    today = dt_date.today()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    return float(age)
            except ValueError:
                pass
        
        # Try "since YYYY" pattern
        since_match = re.search(r'since\s+(\d{4})', value_str)
        if since_match:
            return float(dt_date.today().year - int(since_match.group(1)))
        
        # Try "built in YYYY" pattern
        built_match = re.search(r'(?:built\s+in|from)\s+(\d{4})', value_str)
        if built_match:
            return float(dt_date.today().year - int(built_match.group(1)))
        
        # Try "X years/yrs" pattern
        years_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', value_str)
        if years_match:
            return float(years_match.group(1))
        
        # Fall back to any number
        num_match = re.search(r'(\d+(?:\.\d+)?)', value_str)
        if num_match:
            return float(num_match.group(1))
        
        return None
    
    # Helper function to extract date value
    def extract_date(value_str: str) -> Optional[dt_date]:
        """Extract date from various formats."""
        if not value_str:
            return None
        
        value_str = str(value_str).strip()
        
        # Try DD-MMM-YYYY (e.g. 23-Feb-2001) — primary format after UI change
        if re.match(r'^\d{1,2}-[A-Za-z]{3}-\d{4}$', value_str):
            try:
                return datetime.strptime(value_str, '%d-%b-%Y').date()
            except ValueError:
                pass
        
        # Try YYYY-MM-DD
        if re.match(r'^\d{4}-\d{2}-\d{2}$', value_str):
            try:
                return datetime.strptime(value_str, '%Y-%m-%d').date()
            except ValueError:
                pass
        
        # Try DD/MM/YYYY
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', value_str):
            try:
                return datetime.strptime(value_str, '%d/%m/%Y').date()
            except ValueError:
                pass
        
        return None
    
    def evaluate_comparison_clause(clause: dict, default_compare_as: str = 'number') -> Optional[bool]:
        """Evaluate a single left_field OP right_field clause. Returns True/False, or None if cannot evaluate."""
        left_field = clause.get('left_field') or clause.get('primary_field')
        right_field = clause.get('right_field') or clause.get('secondary_field')
        operator = clause.get('operator', 'gte')
        compare_as = clause.get('compare_as') or default_compare_as

        if not left_field or not right_field:
            return None
        if left_field not in answers_json or right_field not in answers_json:
            return None

        left_value = answers_json.get(left_field)
        right_value = answers_json.get(right_field)

        if compare_as == 'number':
            left_num = extract_number(str(left_value), left_field)
            right_num = extract_number(str(right_value), right_field)
            if left_num is None or right_num is None:
                return None
            if operator == 'gte':
                return left_num >= right_num
            if operator == 'lte':
                return left_num <= right_num
            if operator == 'gt':
                return left_num > right_num
            if operator == 'lt':
                return left_num < right_num
            if operator == 'eq':
                return left_num == right_num
            if operator == 'ne':
                return left_num != right_num
            return None

        if compare_as == 'date':
            left_date = extract_date(str(left_value))
            right_date = extract_date(str(right_value))
            if left_date is None or right_date is None:
                return None
            if operator == 'gte':
                return left_date >= right_date
            if operator == 'lte':
                return left_date <= right_date
            if operator == 'gt':
                return left_date > right_date
            if operator == 'lt':
                return left_date < right_date
            if operator == 'eq':
                return left_date == right_date
            if operator == 'ne':
                return left_date != right_date
            return None

        if compare_as == 'string':
            left_s = str(left_value).strip().lower()
            right_s = str(right_value).strip().lower()
            if operator == 'eq':
                return left_s == right_s
            if operator == 'ne':
                return left_s != right_s
            if operator == 'contains':
                return right_s in left_s
            return None

        return None

    def combine_results(current: bool, next_value: bool, join_with: str) -> bool:
        join_with = str(join_with or 'AND').strip().upper()
        if join_with == 'OR':
            return current or next_value
        return current and next_value

    # Process each rule
    for rule in rules:
        rule_type = rule.get('type', '')
        message = rule.get('message', 'Validation rule failed')

        # Handle different rule types
        if rule_type == 'comparison':
            # Backwards compatible single-pair comparison
            comparisons = rule.get('comparisons')
            negate = bool(rule.get('negate', False))

            if isinstance(comparisons, list) and comparisons:
                # Advanced chained comparisons
                default_compare_as = rule.get('compare_as', 'number')
                combined_result: Optional[bool] = None
                fields_involved = []

                for idx, clause in enumerate(comparisons):
                    if not isinstance(clause, dict):
                        continue
                    left_field = clause.get('left_field')
                    right_field = clause.get('right_field')
                    if left_field and left_field not in fields_involved:
                        fields_involved.append(left_field)
                    if right_field and right_field not in fields_involved:
                        fields_involved.append(right_field)

                    clause_result = evaluate_comparison_clause(clause, default_compare_as=default_compare_as)
                    if clause_result is None:
                        continue

                    if combined_result is None:
                        combined_result = clause_result
                    else:
                        join_with = clause.get('join_with', 'AND')
                        combined_result = combine_results(combined_result, clause_result, join_with)

                # If we couldn't evaluate any clauses, skip
                if combined_result is None:
                    continue

                if negate:
                    combined_result = not combined_result

                rule_failed = not combined_result
                target_field = rule.get('target_field')
                if not target_field:
                    first_clause = comparisons[0] if isinstance(comparisons[0], dict) else {}
                    target_field = first_clause.get('left_field') or rule.get('primary_field', '')

                if rule_failed and target_field:
                    invalid_fields[target_field] = message
                    validation_notes.append({
                        'type': 'rule_violation',
                        'rule_type': 'comparison_chain',
                        'field': target_field,
                        'fields_involved': fields_involved,
                        'issue': message,
                        'suggestion': 'Please check the related fields for consistency.'
                    })

            else:
                primary_field = rule.get('primary_field', '')
                secondary_field = rule.get('secondary_field', '')
                target_field = rule.get('target_field', primary_field)
                operator = rule.get('operator', 'gte')
                compare_as = rule.get('compare_as', 'number')

                if not primary_field or not secondary_field:
                    continue
                if primary_field not in answers_json or secondary_field not in answers_json:
                    continue

                single_clause = {
                    'left_field': primary_field,
                    'right_field': secondary_field,
                    'operator': operator,
                    'compare_as': compare_as,
                }
                clause_result = evaluate_comparison_clause(single_clause, default_compare_as=compare_as)
                if clause_result is None:
                    continue
                if negate:
                    clause_result = not clause_result
                if not clause_result:
                    invalid_fields[target_field] = message
                    validation_notes.append({
                        'type': 'rule_violation',
                        'rule_type': 'comparison',
                        'field': target_field,
                        'fields_involved': [primary_field, secondary_field],
                        'issue': message,
                        'suggestion': f'Please check the values for {primary_field} and {secondary_field}.'
                    })

        elif rule_type == 'required_if':
            condition_field = rule.get('condition_field', '')
            condition_value = rule.get('condition_value', '')
            required_field = rule.get('required_field') or rule.get('primary_field', '')
            
            if condition_field not in answers_json:
                continue
            
            condition_actual = str(answers_json.get(condition_field, '')).strip().lower()
            condition_expected = str(condition_value).strip().lower()
            
            # If condition is met, check if required field is filled
            if condition_actual == condition_expected:
                required_actual = str(answers_json.get(required_field, '')).strip()
                if not required_actual:
                    invalid_fields[required_field] = message
                    validation_notes.append({
                        'type': 'rule_violation',
                        'rule_type': 'required_if',
                        'field': required_field,
                        'fields_involved': [condition_field, required_field],
                        'issue': message,
                        'suggestion': f'This field is required when {condition_field} is "{condition_value}".'
                    })

        elif rule_type == 'disallow_contains':
            field_name = rule.get('field') or rule.get('primary_field', '')
            pattern = rule.get('pattern', '')
            mode = rule.get('mode', 'contains')
            case_sensitive = bool(rule.get('case_sensitive', False))
            target_field = rule.get('target_field', field_name)

            if not field_name or not pattern:
                continue
            if field_name not in answers_json:
                continue

            field_value = str(answers_json.get(field_name, '') or '')
            haystack = field_value if case_sensitive else field_value.lower()
            needle = pattern if case_sensitive else str(pattern).lower()

            match_found = False
            if mode == 'regex':
                try:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    match_found = re.search(pattern, field_value, flags=flags) is not None
                except re.error:
                    continue
            else:
                match_found = needle in haystack

            if match_found and target_field:
                invalid_fields[target_field] = message
                validation_notes.append({
                    'type': 'rule_violation',
                    'rule_type': 'disallow_contains',
                    'field': target_field,
                    'fields_involved': [field_name],
                    'issue': message,
                    'suggestion': 'Please remove the disallowed text and try again.'
                })
    
    return invalid_fields, validation_notes


def generate_rule_summary_for_ai(rules: list) -> str:
    """
    Generate a human-readable summary of validation rules for the AI prompt.
    
    Args:
        rules: List of validation rule objects
        
    Returns:
        str: Formatted rule summary for AI
    """
    if not rules:
        return ""
    
    def op_to_text(op: str) -> str:
        mapping = {
            'gte': 'greater than or equal to',
            'lte': 'less than or equal to',
            'gt': 'greater than',
            'lt': 'less than',
            'eq': 'equal to',
            'ne': 'not equal to',
        }
        return mapping.get(op, op)

    summary_lines = ["\n**AFFIDAVIT-SPECIFIC VALIDATION RULES:**"]
    
    for idx, rule in enumerate(rules, 1):
        rule_type = rule.get('type', '')
        
        if rule_type == 'comparison':
            negate = bool(rule.get('negate', False))
            comparisons = rule.get('comparisons')
            if isinstance(comparisons, list) and comparisons:
                parts = []
                for c_idx, clause in enumerate(comparisons):
                    if not isinstance(clause, dict):
                        continue
                    left_field = clause.get('left_field', '')
                    right_field = clause.get('right_field', '')
                    operator = clause.get('operator', 'gte')
                    compare_as = clause.get('compare_as') or rule.get('compare_as', 'number')
                    clause_text = f"{left_field} must be {op_to_text(operator)} {right_field} (as {compare_as})"
                    if c_idx == 0:
                        parts.append(clause_text)
                    else:
                        join_with = str(clause.get('join_with', 'AND')).strip().upper()
                        parts.append(f"{join_with} {clause_text}")
                chain_text = " ".join(parts).strip()
                if negate:
                    chain_text = f"NOT ({chain_text})"
                summary_lines.append(f"{idx}. {chain_text}. Error: \"{rule.get('message', '')}\"")
            else:
                primary_field = rule.get('primary_field', '')
                secondary_field = rule.get('secondary_field', '')
                operator = rule.get('operator', 'gte')
                compare_as = rule.get('compare_as', 'number')
                base_text = f"{primary_field} must be {op_to_text(operator)} {secondary_field} (as {compare_as})"
                if negate:
                    base_text = f"NOT ({base_text})"
                summary_lines.append(f"{idx}. {base_text}. Error: \"{rule.get('message', '')}\"")
        
        elif rule_type == 'required_if':
            condition_field = rule.get('condition_field', '')
            condition_value = rule.get('condition_value', '')
            required_field = rule.get('required_field', '')
            message = rule.get('message', '')
            
            summary_lines.append(f"{idx}. If {condition_field} is \"{condition_value}\", then {required_field} is required. Error: \"{message}\"")
    
    summary_lines.append("\nWhen a rule is violated, return a field error with the specified message.")
    
    return "\n".join(summary_lines)


def validate_inputs_before_submission(
    answers_json: dict,
    template_html: str,
    affidavit_type_name: str,
    validation_rules: list = None
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
        
        # Generate rule summary for AI if validation rules are provided
        rule_summary = generate_rule_summary_for_ai(validation_rules or [])
        
        system_prompt = f"""You are a strict field validator for legal documents in Trinidad and Tobago. Your job is to catch serious issues - gibberish, future dates, irrelevant answers, and invalid ID formats. Return your response as JSON.

**CRITICAL RULES:**
1. BE LENIENT on spelling/grammar - The AI drafter will fix those
2. BE STRICT on future dates, gibberish, and invalid IDs - these MUST be flagged
3. DO NOT return "informational notes" or "observations" - ONLY REAL ERRORS
4. If a value is VALID, do NOT include it in any error list
5. If house_age is given as a date, CALCULATE the age yourself
6. **IGNORE future date checks for these specific fields:** "current_date", "declaration_day", "declaration_month", "declaration_year", "declaration_date" - these refer to the document date which is TODAY.
7. **ACCEPT TEXT PHRASES FOR ALL FIELDS** - Fields may be "Short Text" type. Users can write phrases like "for over 3 years", "about twenty-five years old", "since 1990", "more than 10 years" etc. NEVER flag a field just because it contains words instead of a pure number. The drafter will normalize formatting.
   - "for over 3 years" = extract 3 = VALID
   - "about 25 years" = extract 25 = VALID
   - "since 1990" = calculate current_year - 1990 = VALID
   - "more than a decade" = extract ~10 = VALID
   - "twenty-five (25) years old" = extract 25 = VALID
   - Do not reject values just because they contain words
8. **DO NOT ENFORCE DATA FORMATS** - If a field asks for duration/age/years, accept BOTH pure numbers ("3") AND text phrases ("for over 3 years"). The drafter will normalize the format.
{rule_summary}

**FLAG THESE SERIOUS PROBLEMS:**

1. **PURE GIBBERISH (keyboard mashing):**
   - Examples: "ihdhfihdbhb", "asdfasdfasdf", "fgdgsgsdvchdvudv", "jkljkljkl"
   - Random characters with NO recognizable words
   - Must be COMPLETELY meaningless

2. **TRAILING GIBBERISH (valid start, bad ending) - BE AGGRESSIVE:**
   - "15 Queen Street asdfasdf" = INVALID (gibberish at end)
   - "replacing windows hahahah i am happy" = INVALID (irrelevant at end)
   - "fixing the roof khdfbiewbfiwbf" = INVALID (gibberish at end)
   - "my property is 100sqm lololol" = INVALID (nonsense at end)
   - "valid address ????" = INVALID (junk at end)
   - "something !!!!!!!!" = INVALID (excessive punctuation)
   - "text here ......." = INVALID (trailing dots)
   - Look for: random chars, "haha", "lol", "????", "!!!!", ".....", keyboard mashing ANYWHERE in the text

3. **COMPLETELY IRRELEVANT ANSWERS:**
   - The answer has NOTHING to do with the question being asked
   - User enters jokes, complaints, or off-topic content instead of answering

4. **DATE REASONABLENESS (NO CROSS-FIELD CONTRADICTIONS):**
   - Flag truly impossible single-field values (e.g., date_of_birth in the future)
   - Do NOT attempt cross-field numeric contradiction checking; those are enforced by server-side validation rules.

5. **IMPOSSIBLE VALUES:**
   - Unreasonable ages (e.g., 250, -5) = INVALID
   - Negative durations or amounts = INVALID

6. **ID NUMBER FORMAT (Trinidad & Tobago) - 11 DIGITS WITH DOB:**
   - Electoral ID MUST be exactly 11 digits in format YYYYMMDDXXX
   - First 8 digits encode date of birth: YYYYMMDD (Year-Month-Day)
   - Last 3 digits are unique sequence number
   - Extract ONLY the digits from input (ignore spaces, dashes)
   - Count digits: if count == 11 = Check if valid format
   - Count digits: if count != 11 = INVALID
   - Example: "19741104044" = 11 digits, DOB=1974-11-04 = VALID
   - Example: "1974-11-04-044" = 11 digits = VALID
   - Example: "123456789" = 9 digits = INVALID (old format)
   
   **CRITICAL - CROSS-VALIDATE WITH DATE OF BIRTH:**
   - If BOTH electoral_id AND date_of_birth fields are provided:
     * Extract year (chars 1-4), month (chars 5-6), day (chars 7-8) from electoral_id
     * Format as YYYY-MM-DD
     * Compare with the date_of_birth field value
     * ONLY FLAG if they are DIFFERENT (mismatch)
     * DO NOT FLAG if they MATCH - matching is CORRECT!
   - Example: electoral_id="19741104044" + date_of_birth="1974-11-04" = MATCH = DO NOT FLAG (this is correct!)
   - Example: electoral_id="19900614044" + date_of_birth="1990-06-14" = MATCH = DO NOT FLAG (this is correct!)
   - Example: electoral_id="19741104044" + date_of_birth="1980-05-15" = MISMATCH = FLAG ERROR
   - ONLY return error when dates are DIFFERENT, NEVER when they match!
   - Error message (only for mismatch): "Your Electoral ID indicates birth date 1974-11-04, but you entered 1980-05-15 as Date of Birth"

**WHAT TO ACCEPT (DO NOT FLAG) - The drafter will fix these:**

✅ Spelling errors, informal language, incomplete but meaningful, vague but relevant answers

**OUTPUT FORMAT - INCLUDE HELPFUL SUGGESTIONS:**
{{
    "all_valid": boolean,
    "field_checks": [
        {{
            "field": "field_name",
            "value": "what user entered",
            "is_valid": true/false,
            "issue": "Clear explanation of what's wrong - null if valid",
            "suggestion": "Helpful suggestion of what they SHOULD enter - null if valid. Be specific and give examples.",
            "correct_format": "Show the correct format if applicable (e.g., '123456789' for ID)"
        }}
    ]
}}

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

1. **TRAILING GIBBERISH CHECK:**
   - Scan EVERY text field for gibberish at the END
   - Look for: random characters, "haha", "lol", "????", "!!!!", ".....", keyboard mashing
   - Example: "15 Queen Street asdfg" = Has gibberish "asdfg" at end = INVALID
   - Example: "fixing roof hahaha" = Has "hahaha" at end = INVALID

2. **PURE GIBBERISH CHECK:**
   - Fields that are ENTIRELY meaningless: "asdfasdf", "jkljkl", etc.

3. **ID FORMAT CHECK (SUPPORTS MULTIPLE ID TYPES):**
   Trinidad & Tobago accepts THREE types of ID documents:
   
   a) **Electoral ID (11 digits):**
      - Extract ONLY digits (remove spaces, dashes)
      - If digit count == 11 = VALID
      - First 8 digits are DOB (YYYYMMDD), last 3 are sequence
      - Example: "19900909098" = 11 digits = VALID
      - Example: "1990 09 09 098" = 11 digits = VALID
   
   b) **Driver's Permit (8 digits):**
      - Extract ONLY digits
      - If digit count == 8 = VALID
      - Example: "12345678" = 8 digits = VALID
   
   c) **Passport (2 letters + 6 digits):**
      - Format: Two uppercase letters followed by 6 digits
      - Examples: "TA123456", "TB987654", "TC456789" = VALID
   
   **VALIDATION RULES:**
   - If format matches ANY of the above = VALID, DO NOT FLAG
   - If format matches NONE = INVALID, flag with helpful message
   - For Electoral ID, cross-validate DOB with date_of_birth field if present
   
   **CROSS-VALIDATE ELECTORAL ID WITH DATE OF BIRTH (only for 11-digit IDs):**
   - If electoral_id has 11 digits AND date_of_birth field exists:
     * Extract digits 1-4 (year), 5-6 (month), 7-8 (day) from electoral_id
     * Compare with date_of_birth field value
     * ONLY flag if they are DIFFERENT!
   - Example: electoral_id="19900614044" + date_of_birth="1990-06-14" = MATCH = VALID
   - Example: electoral_id="19741104044" + date_of_birth="1980-05-15" = MISMATCH = FLAG

**BE STRICT** on future dates and gibberish - these MUST be caught.
**BE LENIENT** on spelling/grammar - the AI drafter fixes that.
"""

        response = call_openai_with_retry(
            client=client,
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

        def _ai_check_is_allowed(check: dict) -> bool:
            """Allow only general, non-hallucination-prone AI validations.

            Cross-field numeric contradictions are enforced by server-side validation rules.
            """
            if not isinstance(check, dict):
                return False
            if check.get('is_valid', True):
                return False

            field_name = str(check.get('field', '') or '').lower()
            issue = str(check.get('issue', '') or '').lower()
            suggestion = str(check.get('suggestion', '') or '').lower()
            text = f"{issue} {suggestion}"

            # ID-related checks (format or DOB mismatch)
            if any(k in field_name for k in ['electoral', 'national_id', 'id_number', 'passport', 'permit', 'driver']):
                return True
            if any(k in text for k in ['electoral id', 'passport', 'permit', 'driver', 'id format', 'digits', '11 digits', '8 digits', 'date of birth', 'dob', 'birth date', 'mismatch']):
                return True

            # Future/invalid date checks
            if any(k in text for k in ['future date', 'in the future', 'after today', 'cannot be in the future']):
                return True

            # Gibberish / irrelevant checks
            if any(k in text for k in ['gibberish', 'keyboard', 'mashing', 'nonsense', 'irrelevant', 'off-topic', 'not related']):
                return True

            return False

        filtered_ai_field_checks = []
        for check in result.get('field_checks', []):
            if not _ai_check_is_allowed(check):
                continue
            filtered_ai_field_checks.append(check)
            field_name = check.get('field', 'unknown')
            if field_name not in invalid_fields:
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
        
        # PYTHON-BASED DATE AND ID VALIDATION OVERRIDE
        # Run reliable checking to CORRECT any AI hallucinations
        import re
        from datetime import datetime
        
        today = datetime.now().date()
        date_patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # DD/MM/YYYY or DD-MM-YYYY
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY/MM/DD or YYYY-MM-DD
            r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{4})',  # DD Month YYYY
        ]
        month_names = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                       'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        
        # Track fields to remove (AI hallucinations we're correcting)
        fields_to_remove = []
        
        # === MULTI-FIELD DATE VALIDATION ===
        # (Removed future date check for declaration dates as per user requirement)
        
        # Fields to explicitly ignore for future date checks (they are typically today's date)
        ignore_future_check_fields = ['current_date', 'declaration_day', 'declaration_month', 'declaration_year', 'declaration_date']

        for field_name, field_value in answers_json.items():
            if not isinstance(field_value, str):
                continue
            
            field_lower = field_name.lower()
            value_lower = field_value.lower().strip()
            
            # === EXPLICIT IGNORE FOR DECLARATION DATES ===
            if any(ignore_kw in field_lower for ignore_kw in ignore_future_check_fields):
                # If AI flagged this field, remove it - we trust these are today's date
                if field_name in invalid_fields:
                    logger.info(f"Python override: Removing AI error for {field_name} (declaration/current date field)")
                    fields_to_remove.append(field_name)
                continue
            
            # === ID VALIDATION OVERRIDE ===
            # Check if AI flagged an ID field - verify with Python
            if any(kw in field_lower for kw in ['electoral', 'national_id', 'id_number', 'passport', 'permit', 'driver']):
                value_upper = str(field_value).strip().upper()
                is_valid_id = False
                is_electoral_id = False
                digits_only = None
                
                # Check Passport format: 2 letters + 6 digits
                if re.match(r'^[A-Z]{2}\d{6}$', value_upper):
                    is_valid_id = True
                else:
                    # Check numeric formats (Electoral ID: 11 digits, Driver's Permit: 8 digits)
                    digits_only = re.sub(r'\D', '', str(field_value))
                    if len(digits_only) == 11:
                        is_valid_id = True
                        is_electoral_id = True
                    elif len(digits_only) == 8:
                        is_valid_id = True
                
                # Cross-validate Electoral ID with DOB if present
                if is_electoral_id and digits_only and len(digits_only) == 11:
                    # Extract DOB from Electoral ID (first 8 digits: YYYYMMDD)
                    try:
                        id_year = int(digits_only[0:4])
                        id_month = int(digits_only[4:6])
                        id_day = int(digits_only[6:8])
                        id_dob = datetime(id_year, id_month, id_day).date()
                        
                        # Check if there's a separate DOB field
                        dob_field = None
                        for dob_key in ['date_of_birth', 'dob', 'birth_date', 'birthdate']:
                            if dob_key in answers_json:
                                dob_field = answers_json[dob_key]
                                break
                        
                        if dob_field:
                            # Parse the DOB field — supports DD-MMM-YYYY, YYYY-MM-DD, DD/MM/YYYY
                            parsed_dob = None
                            dob_str = str(dob_field).strip()
                            
                            # Try DD-MMM-YYYY format (primary UI format e.g. 23-Feb-2001)
                            if re.match(r'^\d{1,2}-[A-Za-z]{3}-\d{4}$', dob_str):
                                try:
                                    parsed_dob = datetime.strptime(dob_str, '%d-%b-%Y').date()
                                except ValueError:
                                    pass
                            
                            # Try YYYY-MM-DD format
                            if not parsed_dob and re.match(r'^\d{4}-\d{2}-\d{2}$', dob_str):
                                try:
                                    parsed_dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
                                except ValueError:
                                    pass
                            
                            # Try DD/MM/YYYY format
                            if not parsed_dob and re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', dob_str):
                                try:
                                    parsed_dob = datetime.strptime(dob_str, '%d/%m/%Y').date()
                                except ValueError:
                                    pass
                            
                            # Compare DOBs
                            if parsed_dob:
                                if parsed_dob != id_dob:
                                    # DOB mismatch - this is a REAL error
                                    error_msg = f'Electoral ID indicates birth date {id_dob.strftime("%Y-%m-%d")}, but Date of Birth field shows {parsed_dob.strftime("%Y-%m-%d")}'
                                    if field_name not in invalid_fields:
                                        invalid_fields[field_name] = error_msg
                                        all_valid = False
                                        logger.info(f"Python validation: Electoral ID DOB mismatch - ID: {id_dob}, DOB field: {parsed_dob}")
                                    is_valid_id = False  # Don't remove AI error if there's a mismatch
                                else:
                                    # DOBs MATCH - this is correct, remove any AI error about mismatch
                                    if field_name in invalid_fields:
                                        ai_error = invalid_fields.get(field_name, '').lower()
                                        if 'birth' in ai_error or 'dob' in ai_error or 'date of birth' in ai_error:
                                            logger.info(f"Python override: Removing AI hallucination for Electoral ID {field_name} - DOBs match correctly (ID: {id_dob}, DOB: {parsed_dob})")
                                            fields_to_remove.append(field_name)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse Electoral ID DOB: {e}")
                
                # If valid ID format and no DOB mismatch, but AI flagged it, remove the AI error
                if is_valid_id and field_name in invalid_fields:
                    # Only remove if it's a format error, not a DOB mismatch
                    ai_error = invalid_fields.get(field_name, '').lower()
                    if 'format' in ai_error or 'digit' in ai_error or 'invalid' in ai_error:
                        logger.info(f"Python override: Removing AI hallucination for valid ID field {field_name}")
                        fields_to_remove.append(field_name)
                # If invalid ID and AI didn't catch it, add error
                elif not is_valid_id and field_name not in invalid_fields:
                    invalid_fields[field_name] = f'Invalid ID format. Accepted: Electoral ID (11 digits), Driver\'s Permit (8 digits), or Passport (2 letters + 6 digits)'
                    all_valid = False
                continue
            
            # === DATE VALIDATION OVERRIDE ===
            # Skip DOB fields (always in the past)
            if 'birth' in field_lower or 'dob' in field_lower:
                continue
            
            # Check fields that contain dates
            if any(kw in field_lower for kw in ['date', 'when', 'occurred', 'incident', 'event', 'declaration', 'year']):
                parsed_date = None
                
                # Try DD-MMM-YYYY pattern (e.g. 23-Feb-2001) — primary UI format
                if not parsed_date:
                    match = re.search(r'(\d{1,2})-(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*-(\d{4})', value_lower, re.IGNORECASE)
                    if match:
                        try:
                            day = int(match.group(1))
                            month = month_names.get(match.group(2).lower()[:3], 0)
                            year = int(match.group(3))
                            if month > 0:
                                parsed_date = datetime(year, month, day).date()
                        except (ValueError, KeyError):
                            pass

                # Try Month DD, YYYY pattern (e.g., "January 06, 2026", "January 6, 2026")
                match = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{1,2}),?\s+(\d{4})', value_lower, re.IGNORECASE)
                if match:
                    try:
                        month = month_names.get(match.group(1).lower()[:3], 0)
                        day = int(match.group(2))
                        year = int(match.group(3))
                        if month > 0:
                            parsed_date = datetime(year, month, day).date()
                    except (ValueError, KeyError):
                        pass
                
                # Try DD Month YYYY pattern (e.g., "06 January 2026")
                if not parsed_date:
                    match = re.search(r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{4})', value_lower, re.IGNORECASE)
                    if match:
                        try:
                            day = int(match.group(1))
                            month = month_names.get(match.group(2).lower()[:3], 0)
                            year = int(match.group(3))
                            if month > 0:
                                parsed_date = datetime(year, month, day).date()
                        except (ValueError, KeyError):
                            pass
                
                # Try DD/MM/YYYY pattern
                if not parsed_date:
                    match = re.search(date_patterns[0], field_value)
                    if match:
                        try:
                            day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                            parsed_date = datetime(year, month, day).date()
                        except ValueError:
                            pass
                
                # Try YYYY-MM-DD pattern
                if not parsed_date:
                    match = re.search(date_patterns[1], field_value)
                    if match:
                        try:
                            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                            parsed_date = datetime(year, month, day).date()
                        except ValueError:
                            pass
                
                if parsed_date:
                    is_future = parsed_date > today
                    ai_flagged = field_name in invalid_fields
                    ai_says_future = ai_flagged and any(kw in invalid_fields.get(field_name, '').lower() for kw in ['future', 'ahead', 'not yet', 'hasn\'t happened', 'current date'])
                    
                    if is_future and not ai_flagged:
                        # Date IS future but AI missed it - add error
                        invalid_fields[field_name] = f'Date cannot be in the future. You entered {field_value}, but today is {today.strftime("%B %d, %Y")}'
                        validation_notes.append({
                            'type': 'field_error',
                            'field': field_name,
                            'value': field_value,
                            'issue': f'Future date detected: {field_value} is after today ({today.strftime("%B %d, %Y")})',
                            'suggestion': f'Please enter a date on or before {today.strftime("%B %d, %Y")}'
                        })
                        all_valid = False
                    elif not is_future and ai_says_future:
                        # Date is NOT future but AI hallucinated - REMOVE the error
                        logger.info(f"Python override: Removing AI hallucination for valid date field {field_name} (value: {field_value}, parsed: {parsed_date}, today: {today})")
                        fields_to_remove.append(field_name)
        
        # Remove AI hallucinations
        for field_name in fields_to_remove:
            if field_name in invalid_fields:
                del invalid_fields[field_name]
            # Also remove from validation_notes
            validation_notes[:] = [note for note in validation_notes if note.get('field') != field_name]

        # Also remove from returned AI field_checks (so UI doesn't show stale AI errors)
        filtered_field_checks = []
        for check in filtered_ai_field_checks:
            try:
                if check.get('field') in fields_to_remove:
                    continue
                filtered_field_checks.append(check)
            except AttributeError:
                continue
        
        # Recalculate all_valid after removing hallucinations
        all_valid = len(invalid_fields) == 0
        
        # Apply custom validation rules (per-affidavit rules)
        if validation_rules:
            rule_invalid_fields, rule_notes = apply_validation_rules(answers_json, validation_rules)
            # Merge rule violations into invalid_fields (don't overwrite existing errors)
            for field, message in rule_invalid_fields.items():
                if field not in invalid_fields:
                    invalid_fields[field] = message
            # Merge rule notes
            validation_notes.extend(rule_notes)
            # Recalculate all_valid
            all_valid = len(invalid_fields) == 0
        
        logger.info(f"Pre-submission validation for {affidavit_type_name}: "
                   f"all_valid={all_valid}, invalid_fields={list(invalid_fields.keys())}")
        
        return {
            'all_valid': all_valid,
            'invalid_fields': invalid_fields,
            'validation_notes': validation_notes,
            'field_checks': filtered_field_checks
        }
        
    except Exception as e:
        logger.error(f"Error in pre-submission validation: {e}")
        
        # FALLBACK: Use pattern-based validation when AI is unavailable
        fallback_invalid_fields = {}
        fallback_notes = []
        
        import re
        
        # Apply per-affidavit custom validation rules FIRST (replaces all hardcoded contradiction checks)
        # This is the generic rule engine that handles any field comparisons defined in affidavit settings
        if validation_rules:
            rule_invalid_fields, rule_notes = apply_validation_rules(answers_json, validation_rules)
            fallback_invalid_fields.update(rule_invalid_fields)
            fallback_notes.extend(rule_notes)
        
        # Generic checks that apply to ALL affidavit types (gibberish, future dates, ID format)
        for field_name, field_value in answers_json.items():
            if not isinstance(field_value, str) or len(field_value.strip()) == 0:
                continue
            
            value = field_value.lower().strip()
            field_lower = field_name.lower()
            
            # Check ID number format - supports multiple Trinidad & Tobago ID types:
            # 1. Electoral ID: 11 digits (YYYYMMDDXXX)
            # 2. Driver's Permit: 8 digits
            # 3. Passport: 2 letters + 6 digits (e.g., TA123456)
            if 'electoral' in field_lower or 'national_id' in field_lower or 'id_number' in field_lower or 'passport' in field_lower or 'permit' in field_lower or 'driver' in field_lower:
                value_upper = str(field_value).strip().upper()
                
                # Check for Passport format: 2 letters + 6 digits (e.g., TA123456, TB987654)
                if re.match(r'^[A-Z]{2}\d{6}$', value_upper):
                    continue  # Valid passport format
                
                # Check for numeric ID formats (Electoral ID or Driver's Permit)
                digits_only = re.sub(r'\D', '', str(field_value))
                
                # Valid formats: 11 digits (Electoral ID) or 8 digits (Driver's Permit)
                if len(digits_only) == 11 or len(digits_only) == 8:
                    continue  # Valid ID format
                
                # Invalid format - provide helpful error message
                fallback_invalid_fields[field_name] = f'Invalid ID format. Accepted formats: Electoral ID (11 digits), Driver\'s Permit (8 digits), or Passport (2 letters + 6 digits like TA123456)'
                fallback_notes.append({
                    'type': 'field_error',
                    'field': field_name,
                    'value': field_value,
                    'issue': f'ID format not recognized. You entered: "{field_value}"',
                    'suggestion': 'Enter one of: Electoral ID (11 digits, e.g., 19900909098), Driver\'s Permit (8 digits), or Passport (2 letters + 6 digits, e.g., TA123456)',
                    'correct_format': 'Electoral: 19900909098, Driver: 12345678, Passport: TA123456'
                })
                continue
            
            # Date validation - check for future dates
            if any(kw in field_lower for kw in ['date', 'when', 'occurred', 'incident', 'event']) and 'birth' not in field_lower and 'dob' not in field_lower and not any(ign in field_lower for ign in ['declaration', 'current_date']):
                from datetime import datetime
                today = datetime.now().date()
                date_patterns = [
                    (r'(\d{1,2})-(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*-(\d{4})', 'dmy_dash'),  # DD-MMM-YYYY (primary)
                    (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', 'dmy'),  # DD/MM/YYYY
                    (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', 'ymd'),  # YYYY-MM-DD
                    (r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{4})', 'dmy_text'),
                ]
                month_names = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                               'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
                
                parsed_date = None
                for pattern, fmt in date_patterns:
                    match = re.search(pattern, field_value, re.IGNORECASE)
                    if match:
                        try:
                            if fmt == 'dmy_dash':  # DD-MMM-YYYY (e.g. 23-Feb-2001)
                                day = int(match.group(1))
                                month = month_names.get(match.group(2).lower()[:3], 0)
                                year = int(match.group(3))
                            elif fmt == 'dmy':
                                day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                            elif fmt == 'ymd':
                                year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                            else:  # dmy_text
                                day = int(match.group(1))
                                month = month_names.get(match.group(2).lower()[:3], 0)
                                year = int(match.group(3))
                            if month > 0:
                                parsed_date = datetime(year, month, day).date()
                                break
                        except ValueError:
                            pass
                
                if parsed_date and parsed_date > today:
                    fallback_invalid_fields[field_name] = f'Date cannot be in the future. You entered {field_value}, but today is {today.strftime("%B %d, %Y")}'
                    fallback_notes.append({
                        'type': 'field_error',
                        'field': field_name,
                        'value': field_value,
                        'issue': f'Future date detected: {field_value} is after today ({today.strftime("%B %d, %Y")})',
                        'suggestion': f'Please enter a date on or before {today.strftime("%B %d, %Y")}'
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

2. **IGNORE SPELLING/CAPITALIZATION DIFFERENCES (CRITICAL):**
   - **DO NOT COMPARE** the draft against user input for spelling, capitalization, or grammar.
   - **ASSUME THE DRAFT IS CORRECT** and the user input was sloppy.
   - **NEVER FLAG** "Bacolet Street" vs "bacolet street".
   - **NEVER FLAG** "Mother" vs "motther".
   - **NEVER FLAG** "Port of Spain" vs "pos".
   - **NEVER FLAG** "John Doe" vs "john doe".
   - The User Input is ONLY provided to check if *facts* are missing. It is NOT a spell-check reference.
   - If the draft says "Mother" and user said "motther", this is **PERFECT**. DO NOT FLAG.
   - If the draft says "123 Main St" and user said "123 main street", this is **PERFECT**. DO NOT FLAG.

3. **DISALLOWED PHRASES - EXACT MATCH, NON-BOILERPLATE ONLY:**
   - You will receive a list of EXACT phrases to avoid (e.g., "I swear", "I promise").
   - ONLY flag if the draft contains these EXACT phrases in AI-authored narrative content.
   - **CRITICAL: NEVER apply disallowed-phrase checks to standard legal boilerplate** — i.e., the statutory declaration attestation clause ("...conscientiously believing the same to be true... false in fact..."), signature blocks, standard T&T legal formulas, or any text that is clearly part of the affidavit template structure. These are fixed legal clauses required by law — the AI did not author them and they MUST stay.
   - "I am aware" is NOT the same as "I swear" — don't flag unrelated text.

4. **HALLUCINATION CHECK - ONLY CONCRETE INVENTED FACTS:**
   - **THE DRAFTER ALWAYS REPHRASES USER INPUT INTO PROPER LEGAL LANGUAGE — THIS IS EXPECTED AND CORRECT.**
   - Do NOT flag something just because it is "not clearly stated in user input" or is worded differently.
   - ONLY flag if the AI invented a **specific concrete fact** — a proper name, a specific number, a specific date, or a specific address — that is **entirely absent from ALL user input fields** (not just unclear or paraphrased).
   - Example: User said "Car", Draft says "Toyota Corolla" → IS a hallucination (invented specific model).
   - Example: User said "my two flats in POS", Draft says "two flats on POS" → NOT a hallucination (rephrased).
   - Example: User said "Car", Draft says "vehicle" → NOT a hallucination (synonym).
   - Example: User said "bacolet st", Draft says "Bacolet Street" → NOT a hallucination (formatting).
   - If the fact appears ANYWHERE in the user answers (even loosely), do NOT flag it as a hallucination.

5. **TEMPLATE COMPLIANCE - BE SPECIFIC:**
   - Flag if the AI added headers like "AFFIDAVIT" that are not in the template.
   - Flag if the AI added extra sections not in the template.
   - Flag if the AI removed critical legal sections.

**DO NOT FLAG:**
- **ANY** spelling difference between draft and user input.
- **ANY** capitalization difference between draft and user input.
- **ANY** rephrasing or legal normalisation (e.g., "live there" → "reside at", "my two flats in POS" → "two flats on POS").
- Content that is "not clearly stated in user input" — the drafter always rephrases and infers legal language.
- Disallowed phrases found inside standard legal boilerplate / statutory declaration clauses.
- HTML tags.

**ONLY FLAG REAL PROBLEMS:**
- Template violations (wrong structure, extra headers).
- Legal errors (missing attestation, missing signature block).
- Made-up facts (names/dates that don't exist in input).
- Disallowed phrases (exact matches).
- Broken structure (missing required legal sections).

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

**USER ANSWERS (reference only — the drafter ALWAYS rephrases these into proper legal language; that is expected and correct):**
{json.dumps(answers_json, indent=2)}

**CRITICAL: CHECK FOR CLARIFICATIONS**
If the USER ANSWERS above contain "PREVIOUS_CLARIFICATIONS":
- The user has CORRECTED original contradictory data after being asked for clarification
- PARSE the "User Answer" in the clarification to extract the FINAL CORRECT values
- When checking for hallucinations, use CLARIFIED values as the source of truth, NOT original fields
 - Example: If original had age=22, years=25, but clarification says "I am 25 and lived for 3 years":
   = The AI should use age=25 in draft (this is CORRECT, not a hallucination)
   = The AI should use years=3 in draft (this is CORRECT, not a hallucination)
 - DO NOT flag these as hallucinations - the AI correctly used the user's clarified corrections
 - Only flag if AI used values that contradict BOTH the original data AND the clarifications

**SYSTEM INSTRUCTIONS (AI must follow these):**
{system_instructions if system_instructions else "Standard professional legal document generation"}

**AFFIDAVIT-SPECIFIC INSTRUCTIONS (AI must follow these):**
{affidavit_instructions if affidavit_instructions else "No specific instructions provided"}

**DISALLOWED PHRASES (apply ONLY to AI-authored narrative content, NOT to template boilerplate or statutory clauses):**
{json.dumps(disallowed_list) if disallowed_list else "None specified"}

**CRITICAL INSTRUCTIONS FOR CHECKING:**

1. **HTML IS EXPECTED - DON'T FLAG IT:**
   - The draft uses HTML tags like <p>, <strong>, <ol>, <li> for PDF generation
   - This is CORRECT and REQUIRED - do not flag HTML tags as formatting issues

2. **DISALLOWED PHRASES - EXACT MATCH, NON-BOILERPLATE ONLY:**
   - Check if the AI-authored narrative content contains the EXACT phrases listed above.
   - **DO NOT apply this check to standard legal boilerplate/template language.** Boilerplate includes: the statutory declaration attestation clause ("...conscientiously believing the same to be true and in accordance with the Statutory Declaration Act... if there is any statement in this Declaration which is false in fact..."), signature/attestation blocks, and any standard T&T legal formulas from the template. These clauses are legally required and must not be flagged.
   - Only flag if the AI inserted a disallowed phrase into the affidavit's substantive narrative body (not in boilerplate).
   - "I am aware" is NOT the same as "I swear" — don't flag unrelated text.

3. **HALLUCINATION CHECK - ONLY ENTIRELY ABSENT CONCRETE FACTS:**
   - **THE AI DRAFTER ALWAYS REPHRASES, REFORMATS, AND NORMALISES USER INPUT — THIS IS BY DESIGN. DO NOT FLAG THIS.**
   - BEFORE flagging a hallucination, search ALL user answer fields for ANY mention of the fact.
   - ONLY flag if a **specific concrete fact** (a proper name, number, date, or address) is **completely absent from every user input field**.
   - If user provided ANY mention — even loosely worded — that's NOT a hallucination.
   - "Not clearly stated in user input" is NOT sufficient to flag — the drafter fills in legal language from context.
   - If user provided "city: Muzaffargarh", that's NOT a hallucination.
   - If user provided "declaration_location: Bacolet Street", that's NOT a hallucination.
   - ONLY flag if the AI invented facts with NO basis in any user input field.

4. **BE SPECIFIC WITH LOCATIONS:**
   - Don't quote "Entire document" - that's useless for reviewers
   - Quote the EXACT problematic sentence or phrase
   - Give actionable, specific feedback

**YOUR TASK:**
1. Read the user answers to know what data the user provided.
2. Read the draft and check for REAL, SIGNIFICANT issues only.
3. Hallucination: Does the suspicious fact appear ANYWHERE in user answers (even loosely)? If YES → NOT a hallucination → DO NOT FLAG.
4. Disallowed phrases: Is the match inside the statutory declaration / attestation boilerplate? If YES → DO NOT FLAG (it is required template language). Is it an exact phrase match in the narrative body? If NO → DO NOT FLAG.
5. Default to "approved" unless there is a REAL, CONCRETE problem that would make the document legally incorrect or professionally unacceptable.

Focus on QUALITY and ACCURACY. When in doubt, APPROVE — false positives are more damaging than false negatives here.
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
        disallowed_phrases=affidavit_type.disallowed_phrases,
        placeholder_mapping=affidavit_type.placeholder_mapping,
        intake_schema=affidavit_type.intake_schema,
        affidavit_type_id=affidavit_type.id,
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
    
    # Step 2: Run QA Check
    # We run this for ALL requests to populate the flags, even if instant mode
    try:
        qa_result = qa_check(
            draft_text=request_obj.draft_text,
            answers_json=final_answers,
            policy_json=affidavit_type.policy_json,
            affidavit_type_name=affidavit_type.name,
            template_html=affidavit_type.template_html,
            disallowed_phrases=affidavit_type.disallowed_phrases
        )
        
        request_obj.qa_flags_json = qa_result.get('issues', [])
        request_obj.qa_passed = qa_result.get('status') == 'approved'
        
        # Log QA results
        logger.info(f"[PROCESS_REQUEST] QA Check Result: {qa_result.get('status')}")
        logger.info(f"[PROCESS_REQUEST] QA Issues Found: {len(request_obj.qa_flags_json)}")
        
    except Exception as e:
        logger.error(f"[PROCESS_REQUEST] QA Check failed: {e}")
        # Don't fail the request if QA fails, just log it
        request_obj.qa_flags_json = []
        request_obj.qa_passed = True
    
    # Determine routing based on affidavit type mode
    is_instant = request_obj.affidavit_type.is_instant_mode or request_obj.affidavit_type.default_mode == 'instant'
    if is_instant:
        # Instant mode: skip reviewer, mark as DRAFT_READY for commissioner
        request_obj.status = Request.Status.DRAFT_READY
    else:
        # Normal / review-first mode: needs human review
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


def refine_template_section(
    current_template: str,
    instruction: str,
    affidavit_type_id: int,
    max_examples: int = 5,
) -> dict:
    """
    AI-powered template refinement.
    Admin describes what to make dynamic; AI adds {{placeholder}} tokens.
    Learns patterns from real saved affidavit drafts stored in DB.
    """
    client = get_openai_client()
    if not client:
        return {'success': False, 'error': 'OpenAI not available'}

    try:
        import re
        from affidavits.models import Request as AffidavitRequest

        # Pull up to max_examples recent approved drafts for this type
        drafts = AffidavitRequest.objects.filter(
            affidavit_type_id=affidavit_type_id,
        ).exclude(final_text='').order_by('-created_at')[:max_examples]

        examples = [d.final_text[:2000] for d in drafts if d.final_text]

        examples_section = ''
        if examples:
            examples_numbered = '\n\n---\n\n'.join(
                f'Example {i + 1}:\n{ex}' for i, ex in enumerate(examples)
            )
            examples_section = f"""**REAL EXAMPLES FROM SAVED DOCUMENTS ({len(examples)} examples):**
These are actual approved affidavits — use them to understand which parts vary across cases:

{examples_numbered}

---
"""

        system_prompt = """You are an expert legal template engineer.
Your job is to make specific sections of an affidavit template dynamic by introducing {{placeholder}} tokens.

RULES:
1. Return ONLY the complete refined template HTML — no explanations, no markdown code blocks.
2. Keep ALL existing {{placeholder}} tokens exactly as they already are.
3. Only modify the specific part described in the instruction — leave everything else exactly the same.
4. New placeholder IDs must use snake_case and describe the data (e.g. {{land_owner_relationship}}, {{deceased_relative_name}}).
5. Make each placeholder as atomic as possible — one concept per placeholder.
6. Preserve all HTML tags, bold, ordered lists, and document structure.
7. Return valid HTML suitable for PDF generation.
8. ⚠️  LOGICAL CONSISTENCY — CRITICAL: Before adding any placeholder, scan the ENTIRE template for sentences that logically negate or contradict the concept being made dynamic.
   - Example violation: template has "I have no other property" (static) AND you add {{other_property}} nearby — this is a direct contradiction.
   - If you detect this: DO NOT just insert the placeholder next to a negating sentence. Instead, RESTRUCTURE the negating sentence so it can logically hold the dynamic value.
     e.g. Change "I have no other property. Except for [static text]" → "Other property (if any): {{other_property_description}}"
   - Ensure ALL references to the same concept in the template are either ALL static or ALL dynamic — never a mix where one part denies and another part lists.
9. When making a sentence dynamic, the final rendered sentence must make complete logical sense regardless of what value is filled in."""

        user_prompt = f"""**CURRENT TEMPLATE:**
{current_template}

{examples_section}
**REFINEMENT INSTRUCTION:**
{instruction}

Based on the examples (if any), identify which parts of the template text actually vary across different cases and replace them with appropriate {{{{placeholder}}}} tokens. Return the complete refined template HTML."""

        response = call_openai_with_retry(
            client,
            model='gpt-4o',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            max_tokens=6000,
            temperature=0.2,
        )

        refined = response.choices[0].message.content.strip()

        # Strip markdown code fences if AI added them
        if refined.startswith('```'):
            lines = refined.split('\n')
            start = 1 if lines[0].startswith('```') else 0
            end = len(lines) - 1 if lines[-1].strip() == '```' else len(lines)
            refined = '\n'.join(lines[start:end])

        # Report which placeholders are genuinely new
        original_phs = set(re.findall(r'\{\{(\w+)\}\}', current_template))
        new_phs = set(re.findall(r'\{\{(\w+)\}\}', refined))
        new_field_ids = list(new_phs - original_phs)

        # Generate help_text for each new field using surrounding context
        new_fields_meta = []
        for fid in new_field_ids:
            # Find context around the placeholder in the refined template
            pattern = re.compile(r'(.{0,80})\{\{' + re.escape(fid) + r'\}\}(.{0,80})', re.DOTALL)
            ctx_match = pattern.search(refined)
            surrounding = ''
            if ctx_match:
                surrounding = (ctx_match.group(1).strip() + ' [VALUE] ' + ctx_match.group(2).strip())
            # Clean HTML from surrounding
            surrounding = re.sub(r'<[^>]+>', ' ', surrounding).strip()
            label = fid.replace('_', ' ').title()
            help_text = f"Enter the {label.lower()} as it should appear in: {surrounding}" if surrounding else f"Enter the {label.lower()}"

            # Detect proper field type using TT smart validation rules
            from affidavits.services.policy_generator_service import (
                apply_smart_validation,
                TT_FIELD_RULES,
                _matches_field_pattern,
            )
            detected_type = 'text'
            detected_validation = {}
            computed_fields = None
            placeholder_text = ''
            question_stub = {'type': 'text'}  # mutable — smart validation may update it
            detected_validation = apply_smart_validation(fid, label, {}, question_stub)
            detected_type = question_stub.get('type', 'text')
            placeholder_text = question_stub.get('placeholder', '')
            computed_fields = question_stub.get('computed_fields')
            if question_stub.get('help_text'):
                help_text = question_stub['help_text']

            meta_entry = {
                'id': fid,
                'label': label,
                'help_text': help_text[:200],
                'type': detected_type,
            }
            if detected_validation:
                meta_entry['validation'] = detected_validation
            if placeholder_text:
                meta_entry['placeholder'] = placeholder_text
            if computed_fields:
                meta_entry['computed_fields'] = computed_fields
            if question_stub.get('options'):
                meta_entry['options'] = question_stub['options']

            new_fields_meta.append(meta_entry)

        # Post-process template: swap DOB placeholders → {{calculated_age}}
        # when a new DOB field with computed_fields: ['age'] is detected
        from affidavits.services.policy_generator_service import post_process_template_for_computed_fields
        refined = post_process_template_for_computed_fields(refined, new_fields_meta)

        logger.info(f"[REFINE_TEMPLATE] type={affidavit_type_id} examples={len(examples)} new_fields={new_field_ids}")

        return {
            'success': True,
            'refined_template': refined,
            'new_fields': new_field_ids,
            'new_fields_meta': new_fields_meta,
            'examples_used': len(examples),
        }

    except Exception as e:
        logger.error(f"[REFINE_TEMPLATE] Error: {e}")
        return {'success': False, 'error': str(e)}


def refine_user_instruction(raw_instruction: str, current_template: str) -> dict:
    """
    AI-powered prompt refinement.
    Takes a rough user instruction and transforms it into a clear,
    specific instruction suitable for template refinement.
    """
    client = get_openai_client()
    if not client:
        return {'success': False, 'error': 'OpenAI not available'}

    try:
        # Extract existing placeholders from template for context
        import re
        existing = re.findall(r'\{\{(\w+)\}\}', current_template)
        existing_str = ', '.join(f'{{{{{p}}}}}' for p in existing[:20]) if existing else 'None'

        # Extract first 500 chars of plain text from template for context
        plain = re.sub(r'<[^>]+>', ' ', current_template[:1000]).strip()
        plain = re.sub(r'\s+', ' ', plain)[:500]

        system_prompt = """You are a helpful assistant that improves user instructions for template refinement.
The user wants to make parts of a legal affidavit template dynamic by adding {{placeholder}} tokens.
Their instruction may be vague, misspelled, or unclear.

Your job:
1. Understand what they want to make dynamic
2. Rewrite their instruction to be clear, specific, and actionable
3. Suggest which specific text in the template should become placeholders
4. Use proper terminology

Return ONLY the improved instruction text — no explanations, no markdown, no prefixes like "Improved:".
Keep it under 200 words."""

        user_prompt = f"""**User's raw instruction:**
{raw_instruction}

**Template preview (first 500 chars):**
{plain}

**Already dynamic fields:** {existing_str}

Rewrite the user's instruction to be clear and specific for the AI template refiner."""

        response = call_openai_with_retry(
            client,
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            max_tokens=300,
            temperature=0.3,
        )

        refined_instruction = response.choices[0].message.content.strip()
        return {'success': True, 'refined_instruction': refined_instruction}

    except Exception as e:
        logger.error(f"[REFINE_INSTRUCTION] Error: {e}")
        return {'success': False, 'error': str(e)}
