"""
Feedback Service — Learning Loop
Retrieves and manages reviewer feedback for injection into AI prompts.
Supports both manual feedback notes and auto-detected edit diffs.
Auto-detected pairs get an AI-generated summary distilled as a one-liner lesson.
"""
import re
import difflib
import logging
from typing import List, Optional

from django.conf import settings
from django.db.models import F

from affidavits.models import ReviewerFeedback

logger = logging.getLogger(__name__)

MAX_FEEDBACK_PER_PROMPT = 10
SNIPPET_MAX_LEN = 400  # chars per snippet in prompt


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

def save_feedback(
    request_obj,
    reviewer,
    category: str,
    message: str,
    feedback_target: str = 'drafter',
) -> ReviewerFeedback:
    """Create a manual feedback note entry."""
    return ReviewerFeedback.objects.create(
        request=request_obj,
        reviewer=reviewer,
        affidavit_type=request_obj.affidavit_type,
        category=category,
        message=message,
        feedback_target=feedback_target,
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
    """
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
        )
        created.append(fb)
        logger.info(
            f'[FEEDBACK] Auto-saved pair #{fb.id} for {request_obj.request_code} '
            f'target={target} has_summary={bool(summary)}'
        )
    return created


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


def get_drafter_feedback(affidavit_type_id: int, limit: int = MAX_FEEDBACK_PER_PROMPT) -> str:
    """
    Build a prompt section from active reviewer feedback targeted at the AI drafter.
    Includes both auto-detected diffs and manual notes.
    Returns empty string if no feedback exists.
    """
    entries = list(
        ReviewerFeedback.objects
        .filter(
            affidavit_type_id=affidavit_type_id,
            is_active=True,
            feedback_target__in=['drafter', 'both'],
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
        '\n\n**REVIEWER FEEDBACK (IMPORTANT — learn from past corrections):**\n'
        'The following are real corrections made by human reviewers on previous drafts '
        'of this affidavit type. Apply these lessons to avoid the same mistakes:\n'
        + '\n'.join(lines)
        + '\n'
    )


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

