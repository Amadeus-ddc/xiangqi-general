"""Select unchanged original coaching labels, without reading held-out answers."""
import hashlib
import heapq
import json
from pathlib import Path


def coaching_config(data, quota):
    if data is None:
        if type(quota) is not int or quota != 0:
            raise ValueError('Original coaching replay needs its explicit source and positive quota')
        return None
    if (not isinstance(data, (str, Path)) or not str(data) or
            type(quota) is not int or quota < 1):
        raise ValueError('Original coaching replay needs its explicit source and positive quota')
    return dict(coaching_data=str(Path(data).resolve()), coaching_train_rows=quota)


def original_training_subset(data, quota, seed, bound):
    from .oracle_policy_corpus import _basic, _lines
    from .reviewed_explanations import ENGINE_QUALIFIED_KIND, source_contract
    data = Path(data).resolve()
    proof = bound.json(data / 'manifest.json')
    if proof.get('status') != 'complete' or proof.get('kind') != ENGINE_QUALIFIED_KIND:
        raise ValueError('Coaching retention requires the explicitly qualified original dataset')
    counts = proof.get('verification', {}).get('by_split', {})
    if (set(counts) != {'train', 'validation', 'test'} or
            any(type(value) is not int or value < 1 for value in counts.values())):
        raise ValueError('Original coaching split coverage is missing')
    for section in ('inputs', 'outputs'):
        for name, identity in proof[section].items():
            bound.bind(name, identity)
    source_contract(proof, counts)
    for split in ('train', 'validation', 'test'):
        path = data / (split + '.jsonl')
        identity = proof['outputs'].get(str(path))
        if identity is None:
            raise ValueError('Original coaching completion must bind every exact source split')
        bound.bind(path, identity)
    queue, ids, features = [], set(), set()
    for pointer, raw, row in _lines(data / 'train.jsonl'):
        _basic(row, 'train', ['explanation'])
        if row['id'] in ids or row['feature_key'] in features:
            raise ValueError('Original coaching training identities must be unique')
        ids.add(row['id']); features.add(row['feature_key'])
        rank = int.from_bytes(hashlib.sha256(json.dumps([seed, 'explanation', row['id']],
            ensure_ascii=False, separators=(',', ':')).encode()).digest(), 'big')
        item = (-rank, pointer['byte_offset'], raw, row)
        if len(queue) < quota:
            heapq.heappush(queue, item)
        elif rank < -queue[0][0]:
            heapq.heapreplace(queue, item)
    if len(ids) != counts['train']:
        raise ValueError('Original coaching training coverage differs from its qualified producer')
    selected = [(raw, row) for _, _, raw, row in sorted(queue, key=lambda item: item[1])]
    return selected, dict(source_data=str(data), source_training_rows=counts['train'],
        requested_training_rows=quota, selected_training_rows=len(selected),
        selected_original_train_bytes_and_order_preserved=True,
        completed_original_semantic_and_engine_qualification_reopened=True,
        new_teacher_or_search_labels=0, validation_rows_added=0, test_rows_added=0,
        heldout_answers_parsed_or_tokenized=0)
