"""Explicit, lossless JSON answer ordering for explanation supervision."""
import json

EXPLANATION_FIRST_PROFILE = 'explanation_before_move_json_v1'


def checked_answer_profile(profile):
    if profile is not None and (not isinstance(profile, str) or profile != EXPLANATION_FIRST_PROFILE):
        raise ValueError('Unknown explanation answer profile')
    return profile


def answer_profile_kwargs(config):
    profile = checked_answer_profile(config.get('explanation_answer_profile'))
    return {} if profile is None else {'explanation_answer_profile': profile}


def require_matching_answer_profiles(recipe, token_config):
    profile = checked_answer_profile(recipe.get('explanation_answer_profile'))
    if profile != checked_answer_profile(token_config.get('explanation_answer_profile')):
        raise ValueError('Explanation answer profile differs from its token preflight')
    return profile


def profiled_token_kind(base_kind, profile=None):
    checked_answer_profile(profile)
    return base_kind if profile is None else base_kind + '_' + profile


def checked_token_answer_kind(token):
    """Recognize typed new targets; older workers reject their distinct raw kind."""
    config = token.get('config', {})
    if not isinstance(config, dict):
        raise ValueError('Token preflight configuration must be an object')
    profile = checked_answer_profile(config.get('explanation_answer_profile'))
    kind = token.get('kind')
    suffix = '_' + EXPLANATION_FIRST_PROFILE
    typed = isinstance(kind, str) and kind.endswith(suffix)
    if profile is None:
        if typed:
            raise ValueError('Token kind requires its explanation answer profile declaration')
        return kind
    verification = token.get('verification', {})
    if (not typed or not isinstance(verification, dict)
            or checked_answer_profile(verification.get('explanation_answer_profile')) != profile):
        raise ValueError('Token kind or verification does not declare its explanation answer profile')
    return kind[:-len(suffix)]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON member in explanation answer')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('Nonfinite JSON constant in explanation answer: ' + value)


def _members(raw):
    if not isinstance(raw, str):
        raise ValueError('Explanation answer must be serialized JSON text')
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_constant)
    obj = decoder.decode(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get('explanation'), str) or not isinstance(obj.get('move'), str):
        raise ValueError('Explanation answer requires text explanation and move fields')

    def whitespace(index):
        while index < len(raw) and raw[index] in ' \t\r\n':
            index += 1
        return index

    start = whitespace(0)
    if raw[start] != '{':
        raise ValueError('Explanation answer must be a JSON object')
    index = whitespace(start + 1)
    prefix, parts, separators = raw[:index], [], []
    while True:
        key, end = decoder.raw_decode(raw, index)
        if not isinstance(key, str):
            raise ValueError('JSON member key must be text')
        colon = whitespace(end)
        if raw[colon] != ':':
            raise ValueError('Missing JSON member colon')
        _, value_end = decoder.raw_decode(raw, whitespace(colon + 1))
        parts.append((key, raw[index:value_end]))
        after = whitespace(value_end)
        if raw[after] == '}':
            suffix = raw[value_end:]
            break
        if raw[after] != ',':
            raise ValueError('Missing JSON member delimiter')
        index = whitespace(after + 1)
        separators.append(raw[value_end:index])
    if len(parts) != len(obj) or len(separators) != len(parts) - 1:
        raise ValueError('Explanation answer member coverage changed')
    return obj, prefix, parts, separators, suffix


def explanation_first_json(raw):
    before, prefix, parts, separators, suffix = _members(raw)
    ordered = [part for part in parts if part[0] == 'explanation']
    ordered += [part for part in parts if part[0] != 'explanation']
    result = prefix + ordered[0][1] + ''.join(
        separator + part[1] for separator, part in zip(separators, ordered[1:], strict=True)) + suffix
    after, after_prefix, after_parts, after_separators, after_suffix = _members(result)
    if (json.dumps(before, ensure_ascii=True, sort_keys=True) != json.dumps(after, ensure_ascii=True, sort_keys=True)
            or dict(parts) != dict(after_parts) or prefix != after_prefix or suffix != after_suffix
            or separators != after_separators or after_parts[0][0] != 'explanation'):
        raise ValueError('Explanation ordering changed JSON values, member bytes or whitespace')
    return result


def serialized_answer(record, profile=None):
    """Leave source records intact; only explicitly opted-in explanation targets change."""
    checked_answer_profile(profile)
    answer = record['answer']
    if profile is None or record.get('stage') != 'explanation':
        return answer
    return explanation_first_json(answer)
