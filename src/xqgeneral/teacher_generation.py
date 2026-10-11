"""Bounded, explicit consolidation decoding and recorded batch sampling seeds."""
import hashlib
import json

from .sampling import SAMPLING_KEYS, checked_seed, sampling_options

PROFILE_KEYS = SAMPLING_KEYS | {'max_context_tokens', 'max_new_tokens', 'reasoning_effort'}
SEED_PROFILE = 'sha256_teacher_seed_and_ordered_batch_ids_v1'


def checked_budget(max_new_tokens, max_context_tokens=None):
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError('A positive integer generation budget is required')
    if max_context_tokens is not None and (type(max_context_tokens) is not int or
            max_context_tokens <= max_new_tokens):
        raise ValueError('Teacher context must be a positive integer larger than the output budget')


def generation_profile(value):
    if not isinstance(value, dict) or set(value) != PROFILE_KEYS:
        raise ValueError('Teacher generation requires explicit sampling, context, output and reasoning settings')
    checked_budget(value['max_new_tokens'], value['max_context_tokens'])
    if not isinstance(value['reasoning_effort'], str) or value['reasoning_effort'] not in {'low', 'medium', 'xhigh'}:
        raise ValueError('Unsupported teacher reasoning effort')
    return {**sampling_options({k: value[k] for k in SAMPLING_KEYS}),
            **{k: value[k] for k in ['max_context_tokens', 'max_new_tokens', 'reasoning_effort']}}


def checked_prompt_width(width, max_new_tokens, max_context_tokens):
    checked_budget(max_new_tokens, max_context_tokens)
    if max_context_tokens is not None and width >= max_context_tokens:
        raise ValueError('Teacher prompt leaves no output room in its context; no prompt was truncated')
    return min(max_new_tokens, max_context_tokens - width) if max_context_tokens is not None else max_new_tokens


def batch_seed(seed, ids):
    checked_seed(seed)
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Teacher sampling requires unique nonempty batch query IDs')
    body = {'profile': SEED_PROFILE, 'seed': seed, 'query_ids': ids}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big') & (2**63 - 1)


def checked_response(response, budget, context, profile=None):
    if context is None:
        return
    prompt, width, generated = (response.get(k) for k in
                                ['prompt_tokens', 'padded_prompt_tokens', 'generated_tokens'])
    effective = response.get('effective_max_new_tokens')
    if (type(prompt) is not int or prompt <= 0 or type(width) is not int or width < prompt or
            type(effective) is not int or type(generated) is not int or not 0 <= generated <= effective or
            type(response.get('hit_generation_limit')) is not bool or
            response['hit_generation_limit'] != (generated >= effective)):
        raise ValueError('Teacher response has invalid bounded token metadata')
    if (effective != checked_prompt_width(width, budget, context) or
            response.get('context_limited_output') is not (effective < budget)):
        raise ValueError('Teacher effective output budget differs from its context')
    if profile is not None:
        raw = response.get('raw_text')
        if not isinstance(raw, str):
            raise ValueError('Reasoning teacher response must preserve its raw text')
        finished = '</think>' in raw
        expected = raw.split('</think>', 1)[1].strip() if finished else raw.strip()
        if response.get('reasoning_completed') is not finished or response.get('text') != expected:
            raise ValueError('Teacher reasoning boundary or displayed response changed')
