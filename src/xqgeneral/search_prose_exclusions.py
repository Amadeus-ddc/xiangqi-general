"""Apply explicit prose-review exclusions without editing teacher responses."""
import hashlib
import re

from .explanations import parse_explanation

PROFILE = 'search_prose_exclusions_v1'
SUMMARY_FIELDS = ('prose_review_exclusion_profile', 'prose_review_excluded_training_labels',
                  'prose_review_reviewer', 'prose_review_source_hashes_verified',
                  'remaining_prose_semantics_verified_by_exclusions')


def exclude_reviewed_prose(train, rejected, review, *, queries_sha256, responses_sha256):
    """Only remove already valid labels; retain all original row values and gates."""
    required = {'profile', 'queries_sha256', 'responses_sha256', 'reviewer', 'exclusions'}
    if not isinstance(review, dict) or set(review) != required or review.get('profile') != PROFILE:
        raise ValueError('Use the explicit search prose-exclusion review schema')
    for key, expected in [('queries_sha256', queries_sha256), ('responses_sha256', responses_sha256)]:
        value = review[key]
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) or value != expected:
            raise ValueError('Prose review belongs to different raw queries or teacher responses')
    if not isinstance(review['reviewer'], str) or not review['reviewer'].strip():
        raise ValueError('Prose exclusion review needs its recorded reviewer')
    entries = review['exclusions']
    if not isinstance(entries, list) or not entries:
        raise ValueError('Prose exclusions must be a nonempty list; omit the option for no exclusions')
    by_id = {row['id']: row for row in train}
    if len(by_id) != len(train):
        raise ValueError('Native accepted label IDs must be unique before prose review')
    excluded = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) != {'id', 'explanation_sha256', 'reason'}:
            raise ValueError('Each prose exclusion needs only an ID, explanation hash and reason')
        query_id = item['id']
        if not isinstance(query_id, str) or query_id in excluded or query_id not in by_id:
            raise ValueError('Every excluded ID must identify one unique natively accepted training label')
        if not isinstance(item['reason'], str) or not item['reason'].strip():
            raise ValueError('Prose exclusions require their case-level reason')
        prose = parse_explanation(by_id[query_id]['answer'])['explanation']
        actual = hashlib.sha256(prose.encode('utf-8')).hexdigest()
        if item['explanation_sha256'] != actual:
            raise ValueError('Reviewed final explanation changed; preserve the old review and review again')
        excluded[query_id] = item
    kept = [row for row in train if row['id'] not in excluded]
    dropped = [{'id': row['id'], 'reason': 'Excluded by bound prose review: ' + excluded[row['id']]['reason']}
               for row in train if row['id'] in excluded]
    summary = dict(prose_review_exclusion_profile=PROFILE, prose_review_excluded_training_labels=len(dropped),
        prose_review_reviewer=review['reviewer'], prose_review_source_hashes_verified=True,
        remaining_prose_semantics_verified_by_exclusions=False)
    return kept, [*rejected, *dropped], summary
