"""
Feedback Service — Learning Loop
Retrieves and manages reviewer feedback for injection into AI prompts.
Supports both manual feedback notes and auto-detected edit diffs.
Auto-detected pairs get an AI-generated summary distilled as a one-liner lesson.

Per-request contextual matching:
  - build_answer_fingerprint() snapshots selector-field values from answers_json
  - score_feedback_relevance() uses Jaccard + exact-match to score similarity
  - get_drafter_feedback() splits results into HIGH-PRIORITY (similar requests)
    and GENERAL (type-level corrections) for two-tier prompt injection.
"""
import re
import difflib
import logging
from typing import Dict, List, Optional, Tuple

from django.conf import settings
from django.db.models import F

from affidavits.models import ReviewerFeedback

logger = logging.getLogger(__name__)

MAX_FEEDBACK_PER_PROMPT = 10
SNIPPET_MAX_LEN = 400  # chars per snippet in prompt

# Relevance score threshold — feedback scoring above this is "similar"
SIMILAR_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Diff Extraction
# ---------------------------------------------------------------------------

def _strip_tags(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    clean = re.sub(r'<[^>]+>', ' ', html or '')
    clean = re.sub(r'&nbsp;|&amp;|&lt;|&gt;', ' ', clean)
    return re.sub(r'\s+', ' ', clean).strip()


def _to_chunks(html: str) -> List[str]:
    """Split HTML into paragraph-level plain-text chunks."""
    parts = re.split(
        r'</?(?:p|li|tr|div|ol|ul|h[1-6])[^>]*>',
        html or '',
        flags=re.IGNORECASE,
    )
    chunks = [_strip_tags(c).strip() for c in parts]
    return [c for c in chunks if len(c) > 15]


def extract_sentence_diffs(
    original_html: str,
    revised_html: str,
    max_pairs: int = 10,
) -> List[dict]:
    """
    Extract changed paragraph pairs between original and revised HTML using difflib.
    Returns list of {original, revised} dicts representing before→after changes.
    """
    orig_chunks = _to_chunks(original_html)
    rev_chunks = _to_chunks(revised_html)

    if not orig_chunks or not rev_chunks:
        return []

    matcher = difflib.SequenceMatcher(None, orig_chunks, rev_chunks, autojunk=False)
    pairs = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace' and len(pairs) < max_pairs:
            orig_block = ' '.join(orig_chunks[i1:i2]).strip()
            rev_block = ' '.join(rev_chunks[j1:j2]).strip()
            if orig_block and rev_block:
                pairs.append({
                    'original': orig_block[:600],
                    'revised': rev_block[:600],
                })

    return pairs


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _compute_fingerprint_and_tags(request_obj) -> Tuple[dict, list]:
    """Compute answer_fingerprint and scenario_tags for a request."""
    fingerprint = build_answer_fingerprint(request_obj)
    tags = []
    try:
        from .ai_service import detect_scenario
        lib = []
        if request_obj.affidavit_type:
            policy = request_obj.affidavit_type.policy_json or {}
            lib = policy.get('scenario_library', [])
        if lib and request_obj.answers_json:
            tags, _ = detect_scenario(request_obj.answers_json, lib)
    except Exception as exc:
        logger.warning(f'[FEEDBACK] Could not detect scenarios: {exc}')
    return fingerprint, tags


def save_feedback(
    request_obj,
    reviewer,
    category: str,
    message: str,
    feedback_target: str = 'drafter',
) -> ReviewerFeedback:
    """Create a manual feedback note entry with contextual fingerprint."""
    fingerprint, tags = _compute_fingerprint_and_tags(request_obj)
    return ReviewerFeedback.objects.create(
        request=request_obj,
        reviewer=reviewer,
        affidavit_type=request_obj.affidavit_type,
        category=category,
        message=message,
        feedback_target=feedback_target,
        answer_fingerprint=fingerprint,
        scenario_tags=tags,
    )


def save_feedback_entries(
    request_obj,
    reviewer,
    entries: List[dict],
) -> List[ReviewerFeedback]:
    """
    Bulk-save manual feedback entries from a single approve action.
    Each entry: { category, message, feedback_target }
    """
    created = []
    for entry in entries:
        fb = save_feedback(
            request_obj=request_obj,
            reviewer=reviewer,
            category=entry.get('category', 'other'),
            message=entry.get('message', ''),
            feedback_target=entry.get('feedback_target', 'drafter'),
        )
        created.append(fb)
        logger.info(
            f"[FEEDBACK] Saved note #{fb.id} for {request_obj.request_code} "
            f"target={fb.feedback_target} category={fb.category}"
        )
    return created


# ---------------------------------------------------------------------------
# AI Summary Generation
# ---------------------------------------------------------------------------

def generate_feedback_summary(original: str, revised: str) -> Optional[str]:
    """
    Ask the AI to distill a one-liner lesson from a reviewer correction.
    E.g. "Use present perfect tense (have been living) instead of present (am living)."
    Returns None on failure so callers can proceed without a summary.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a legal document quality instructor. '
                        'A human reviewer corrected an AI-generated affidavit. '
                        'Distill ONE concise rule or lesson (max 60 words) that the AI '
                        'should remember to avoid the same mistake next time. '
                        'Be specific, actionable and start with a verb (e.g. "Use", "Avoid", "Ensure"). '
                        'Output only the lesson — no labels, no quotes, no extra text.'
                    ),
                },
                {
                    'role': 'user',
                    'content': (
                        f'ORIGINAL (AI wrote):\n{original[:400]}\n\n'
                        f'REVISED (reviewer corrected to):\n{revised[:400]}'
                    ),
                },
            ],
            max_tokens=100,
            temperature=0.3,
        )
        summary = response.choices[0].message.content.strip()
        return summary[:280] if summary else None
    except Exception as exc:
        logger.warning(f'[FEEDBACK] Summary generation failed: {exc}')
        return None


def save_auto_feedback_pairs(
    request_obj,
    reviewer,
    pairs: List[dict],
) -> List[ReviewerFeedback]:
    """
    Save auto-detected diff pairs (from reviewer edit classify dialog) as ReviewerFeedback.
    Each pair: { original_snippet, revised_snippet, feedback_target }
    Skip entries where feedback_target == 'skip'.
    Attempts to generate an AI summary for each saved pair.
    Automatically computes answer_fingerprint and scenario_tags from the request.
    """
    fingerprint, tags = _compute_fingerprint_and_tags(request_obj)
    created = []
    for pair in pairs:
        target = pair.get('feedback_target', 'drafter')
        if target == 'skip':
            continue
        orig = (pair.get('original_snippet') or '').strip()
        rev = (pair.get('revised_snippet') or '').strip()
        if not orig or not rev:
            continue

        summary = generate_feedback_summary(orig, rev)

        fb = ReviewerFeedback.objects.create(
            request=request_obj,
            reviewer=reviewer,
            affidavit_type=request_obj.affidavit_type,
            category='other',
            message='',
            original_snippet=orig[:600],
            revised_snippet=rev[:600],
            summary=summary,
            feedback_target=target,
            answer_fingerprint=fingerprint,
            scenario_tags=tags,
        )
        created.append(fb)
        logger.info(
            f'[FEEDBACK] Auto-saved pair #{fb.id} for {request_obj.request_code} '
            f'target={target} has_summary={bool(summary)}'
        )
    return created


# ---------------------------------------------------------------------------
# Answer Fingerprint & Relevance Scoring
# ---------------------------------------------------------------------------

# Field types that act as "selectors" — their finite-choice values define
# request similarity.  Free-text fields are excluded to avoid noise.
_SELECTOR_TYPES = {'select', 'radio', 'checkbox', 'dropdown', 'toggle', 'boolean'}


def build_answer_fingerprint(request_obj) -> Dict[str, object]:
    """
    Extract selector-field values from a request's answers_json.

    Only keeps fields whose intake_schema type is a selector type (select,
    radio, checkbox, etc.) — free-text inputs are too noisy for matching.
    Returns a dict like {"purpose": "sale", "parish": "St. George"}.
    """
    answers = request_obj.answers_json or {}
    if not answers:
        return {}

    intake_schema = []
    if request_obj.affidavit_type:
        intake_schema = request_obj.affidavit_type.intake_schema or []

    if not intake_schema:
        # No schema → can't determine selector fields, return empty
        return {}

    selector_ids = set()
    for field in intake_schema:
        ftype = (field.get('type') or '').lower()
        fid = field.get('id') or field.get('field_name', '')
        if ftype in _SELECTOR_TYPES and fid:
            selector_ids.add(fid)

    fingerprint = {}
    for key, value in answers.items():
        if key in selector_ids and value not in (None, '', [], {}):
            fingerprint[key] = value

    return fingerprint


def score_feedback_relevance(
    feedback_fp: Dict[str, object],
    current_fp: Dict[str, object],
    feedback_tags: List[str],
    current_tags: List[str],
) -> float:
    """
    Score how relevant a saved feedback record is to the current request.

    Uses a weighted blend of:
      - Jaccard similarity on fingerprint keys+values  (weight 0.6)
      - Exact-match ratio on scenario tags              (weight 0.4)

    Returns a float between 0.0 (no overlap) and 1.0 (identical).
    """
    fp_score = 0.0
    tag_score = 0.0

    # --- Fingerprint Jaccard ---
    if feedback_fp and current_fp:
        all_keys = set(feedback_fp.keys()) | set(current_fp.keys())
        if all_keys:
            matching = sum(
                1 for k in all_keys
                if str(feedback_fp.get(k, '')).lower() == str(current_fp.get(k, '')).lower()
            )
            fp_score = matching / len(all_keys)

    # --- Scenario tag overlap ---
    fb_set = set(feedback_tags) if feedback_tags else set()
    cur_set = set(current_tags) if current_tags else set()
    union = fb_set | cur_set
    if union:
        tag_score = len(fb_set & cur_set) / len(union)

    return round(0.6 * fp_score + 0.4 * tag_score, 4)


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------

def _format_entry(fb: ReviewerFeedback) -> str:
    """Format a single ReviewerFeedback entry as a prompt line."""
    if fb.original_snippet and fb.revised_snippet:
        orig = fb.original_snippet[:SNIPPET_MAX_LEN]
        rev = fb.revised_snippet[:SNIPPET_MAX_LEN]
        if fb.summary:
            return f'- {fb.summary}\n  (Example: "{orig}" → "{rev}")'        
        return f'- CORRECTION: "{orig}" → "{rev}"'
    elif fb.message:
        cat_label = fb.get_category_display()
        return f'- [{cat_label}] {fb.message}'
    return ''


def get_drafter_feedback(
    affidavit_type_id: int,
    answers_json: dict = None,
    scenario_tags: list = None,
    limit: int = MAX_FEEDBACK_PER_PROMPT,
) -> str:
    """
    Build a prompt section from active reviewer feedback targeted at the AI drafter.

    When answers_json / scenario_tags are provided, feedback is scored by
    contextual relevance and split into two tiers:
      1. HIGH-PRIORITY — corrections from similar past requests
      2. GENERAL — type-level corrections (still useful, lower weight)

    Falls back to simple recency ordering when no context is available.
    Returns empty string if no feedback exists.
    """
    entries = list(
        ReviewerFeedback.objects
        .filter(
            affidavit_type_id=affidavit_type_id,
            is_active=True,
            feedback_target__in=['drafter', 'both'],
        )
        .order_by('-created_at')[:limit * 3]  # fetch extra to score & re-rank
    )

    if not entries:
        return ''

    # --- Contextual scoring (when caller provides request context) ---
    has_context = bool(answers_json) or bool(scenario_tags)
    similar: List[Tuple[float, ReviewerFeedback]] = []
    general: List[ReviewerFeedback] = []

    if has_context:
        # Build a fingerprint from the *current* request's answers
        # (We don't have the request object here, so build inline from answers + schema.)
        # The caller passes the raw answers_json; the fingerprint on existing feedback
        # was built at save-time from the same schema.
        current_fp = {}
        if answers_json:
            # Build a minimal fingerprint from answers_json — all keys present.
            # The saved fingerprint only has selector keys, so non-selector keys
            # simply won't match, which is fine.
            current_fp = {k: v for k, v in answers_json.items() if v not in (None, '', [], {})}
        current_tags = scenario_tags or []

        for fb in entries:
            score = score_feedback_relevance(
                feedback_fp=fb.answer_fingerprint or {},
                current_fp=current_fp,
                feedback_tags=fb.scenario_tags or [],
                current_tags=current_tags,
            )
            if score >= SIMILAR_THRESHOLD:
                similar.append((score, fb))
            else:
                general.append(fb)

        # Sort similar by score desc, take top entries
        similar.sort(key=lambda x: x[0], reverse=True)
    else:
        # No context — everything goes into general (backwards-compatible)
        general = entries

    # Build formatted lines for each tier
    similar_lines = [_format_entry(fb) for _, fb in similar[:limit]]
    similar_lines = [l for l in similar_lines if l]

    remaining_slots = max(0, limit - len(similar_lines))
    general_lines = [_format_entry(fb) for fb in general[:remaining_slots]]
    general_lines = [l for l in general_lines if l]

    if not similar_lines and not general_lines:
        return ''

    # Track times_seen for all used entries
    used_ids = [fb.id for _, fb in similar[:limit]] + [fb.id for fb in general[:remaining_slots]]
    if used_ids:
        ReviewerFeedback.objects.filter(id__in=used_ids).update(times_seen=F('times_seen') + 1)

    # Build the prompt section
    parts = []

    if similar_lines:
        parts.append(
            '\n\n**REVIEWER FEEDBACK — SIMILAR PAST REQUESTS (HIGH PRIORITY):**\n'
            'These corrections come from requests very similar to the current one. '
            'Apply these lessons with high confidence:\n'
            + '\n'.join(similar_lines)
        )

    if general_lines:
        parts.append(
            '\n\n**REVIEWER FEEDBACK — GENERAL TYPE CORRECTIONS:**\n'
            'The following are general corrections made by human reviewers on previous drafts '
            'of this affidavit type. Apply these lessons to avoid the same mistakes:\n'
            + '\n'.join(general_lines)
        )

    return ''.join(parts) + '\n'


def get_policy_feedback(affidavit_type_id: int, limit: int = MAX_FEEDBACK_PER_PROMPT) -> str:
    """
    Build a prompt section from active reviewer feedback targeted at the policy/template generator.
    Returns empty string if no feedback exists.
    """
    entries = list(
        ReviewerFeedback.objects
        .filter(
            affidavit_type_id=affidavit_type_id,
            is_active=True,
            feedback_target__in=['policy', 'both'],
        )
        .order_by('-created_at')[:limit]
    )

    if not entries:
        return ''

    lines = [_format_entry(fb) for fb in entries]
    lines = [l for l in lines if l]

    if not lines:
        return ''

    entry_ids = [fb.id for fb in entries]
    ReviewerFeedback.objects.filter(id__in=entry_ids).update(times_seen=F('times_seen') + 1)

    return (
        '\n\n**REVIEWER FEEDBACK (apply these lessons to the generated policy/template):**\n'
        'Human reviewers have flagged the following issues on documents of this type. '
        'Adjust the template and policy to prevent them:\n'
        + '\n'.join(lines)
        + '\n'
    )

