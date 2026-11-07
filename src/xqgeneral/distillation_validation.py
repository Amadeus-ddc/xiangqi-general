"""Complete original lecture/course validation coverage for an explicit round."""
from collections import Counter
import hashlib
import json

from .curriculum_data import STAGES

LEGACY_VALIDATION_PROFILE = 'explanation_only_v1'
JOINT_VALIDATION_PROFILE = 'joint_explanation_and_course_replay_v1'


def joint_validation_coverage(rows, selected_ids, expected_counts):
    """Check actual validation IDs once, without model or expert construction."""
    stages = ['explanation', *STAGES]
    if (not isinstance(expected_counts, dict) or set(expected_counts) != set(stages) or
            any(type(expected_counts[s]) is not int or expected_counts[s] < 1 for s in stages)):
        raise ValueError('Joint validation requires actual positive coverage for every original stage')
    expected, counts = set(), Counter()
    metadata = rows.iter_metadata() if hasattr(rows, 'iter_metadata') else rows
    for row in metadata:
        identity, stage = row.get('id'), row.get('stage')
        if (not isinstance(identity, str) or not identity or identity in expected or stage not in stages or
                row.get('split', 'validation') != 'validation'):
            raise ValueError('Joint validation sources require unique original IDs and validation-only stages')
        expected.add(identity)
        counts[stage] += 1
    if (counts != expected_counts or not isinstance(selected_ids, list) or
            any(not isinstance(value, str) or not value for value in selected_ids) or
            len(selected_ids) != len(expected) or len(set(selected_ids)) != len(selected_ids) or
            set(selected_ids) != expected):
        raise ValueError('Joint validation selection must cover every original validation ID exactly once')
    return {'profile': JOINT_VALIDATION_PROFILE, 'validation_rows': len(expected),
        'validation_rows_by_stage': {stage: counts[stage] for stage in stages},
        'selected_validation_ids_sha256': hashlib.sha256(json.dumps(selected_ids,
            ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
        'all_original_validation_rows_covered_once': True, 'independent_test_used': False,
        'metric': 'token_nll_over_complete_original_validation'}
