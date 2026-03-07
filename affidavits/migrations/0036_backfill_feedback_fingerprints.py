"""
Backfill answer_fingerprint and scenario_tags for existing ReviewerFeedback records.
"""
from django.db import migrations


def backfill_fingerprints(apps, schema_editor):
    """
    For every existing ReviewerFeedback with an empty fingerprint,
    compute the fingerprint from the linked request's answers_json
    and detect scenario tags from the affidavit type's policy_json.
    """
    ReviewerFeedback = apps.get_model('affidavits', 'ReviewerFeedback')

    # We cannot import the helper directly because model classes from
    # apps.get_model differ from the real ones.  Re-implement the logic inline.
    SELECTOR_TYPES = {'select', 'radio', 'checkbox', 'dropdown', 'toggle', 'boolean'}

    feedbacks = (
        ReviewerFeedback.objects
        .select_related('request', 'request__affidavit_type')
        .filter(answer_fingerprint={})
    )

    updated = 0
    for fb in feedbacks.iterator(chunk_size=200):
        request_obj = fb.request
        if not request_obj:
            continue

        # --- Build fingerprint ---
        answers = request_obj.answers_json or {}
        intake_schema = []
        affidavit_type = getattr(request_obj, 'affidavit_type', None)
        if affidavit_type:
            intake_schema = affidavit_type.intake_schema or []

        selector_ids = set()
        for field in intake_schema:
            ftype = (field.get('type') or '').lower()
            fid = field.get('id') or field.get('field_name', '')
            if ftype in SELECTOR_TYPES and fid:
                selector_ids.add(fid)

        fingerprint = {}
        for key, value in answers.items():
            if key in selector_ids and value not in (None, '', [], {}):
                fingerprint[key] = value

        # --- Detect scenario tags ---
        import json
        tags = []
        if affidavit_type:
            policy = affidavit_type.policy_json or {}
            lib = policy.get('scenario_library', [])
            if lib and answers:
                answers_str = json.dumps(answers).lower()
                for scenario in lib:
                    scenario_id = scenario.get('id', '')
                    keywords = scenario.get('keywords', [])
                    patterns = scenario.get('patterns', [])

                    kw_matches = sum(1 for kw in keywords if kw.lower() in answers_str)
                    kw_thresh = scenario.get('keyword_threshold', len(keywords) // 2 + 1)

                    pat_matches = 0
                    for pat in patterns:
                        field = pat.get('field', '')
                        values = pat.get('values', [])
                        fv = str(answers.get(field, '')).lower()
                        if any(v.lower() in fv for v in values):
                            pat_matches += 1
                    pat_thresh = scenario.get('pattern_threshold', len(patterns) // 2 + 1) if patterns else 0

                    if kw_matches >= kw_thresh or (patterns and pat_matches >= pat_thresh):
                        tags.append(scenario_id)

        fb.answer_fingerprint = fingerprint
        fb.scenario_tags = tags
        fb.save(update_fields=['answer_fingerprint', 'scenario_tags'])
        updated += 1

    if updated:
        print(f'[BACKFILL] Updated {updated} ReviewerFeedback records with fingerprints + tags')


def noop(apps, schema_editor):
    """Reverse migration is a no-op — fields remain but data is cleared on next save."""
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('affidavits', '0035_add_feedback_fingerprint'),
    ]

    operations = [
        migrations.RunPython(backfill_fingerprints, noop),
    ]
