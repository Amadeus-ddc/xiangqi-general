"""Explicit, process-scoped reuse of hashes for unchanged verified input bytes."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import re
import time

KIND = 'verified_immutable_inputs_v1'
ENVIRONMENT = 'XQGENERAL_VERIFIED_INPUT_RECEIPT'
MINIMUM_FILE_AGE_NS = 2_000_000_000
_ACTIVE = ContextVar('xqgeneral_verified_input_digests', default=None)


def _signature(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _settled(signature):
    # Filesystem timestamp precision can exceed its actual update resolution.
    # A rapid same-size rewrite can retain all five stat fields in one tick.
    return time.time_ns() - signature[4] >= MINIMUM_FILE_AGE_NS


def _binding(name, value):
    if (not isinstance(name, str) or not Path(name).is_absolute() or
            str(Path(name).resolve()) != name or not isinstance(value, dict) or
            set(value) != {'sha256', 'bytes', 'signature'} or
            not isinstance(value['sha256'], str) or re.fullmatch('[0-9a-f]{64}', value['sha256']) is None or
            type(value['bytes']) is not int or value['bytes'] < 0 or
            not isinstance(value['signature'], list) or len(value['signature']) != 5 or
            any(type(v) is not int for v in value['signature']) or value['signature'][2] != value['bytes']):
        raise ValueError('Invalid verified input hash, path or file signature')
    return {'sha256': value['sha256'], 'bytes': value['bytes'], 'signature': tuple(value['signature'])}


class VerifiedInputDigests:
    """A complete receipt plus fresh process-local hashes; never a content oracle."""
    def __init__(self, receipt):
        path = Path(receipt)
        before = _signature(path)
        raw = path.read_bytes()
        if _signature(path) != before:
            raise ValueError('Verified input receipt changed while reading')
        proof = json.loads(raw)
        if (not isinstance(proof, dict) or proof.get('status') != 'complete' or proof.get('kind') != KIND or
                not isinstance(proof.get('inputs'), dict) or not proof['inputs']):
            raise ValueError('Input digest reuse requires a complete nonempty verified receipt')
        self.bindings = {name: _binding(name, value) for name, value in proof['inputs'].items()}
        if any(not _settled(item['signature']) for item in self.bindings.values()):
            raise ValueError('Verified receipt contains a recently written or future-dated inode; publish settled byte hashes')
        self.receipt_sha256 = hashlib.sha256(raw).hexdigest()
        self.memo = {}
        self.reused_names = set()
        self.reused_calls = self.reused_bytes = self.fresh_calls = self.fresh_bytes = 0
        self.check_unchanged()

    def check_unchanged(self):
        for name, item in self.bindings.items():
            if _signature(name) != item['signature']:
                raise ValueError(f'Verified input file identity changed: {name}')

    def known(self, path):
        name = str(Path(path).resolve())
        current = _signature(name)
        item = self.bindings.get(name)
        if item is not None:
            if current != item['signature']:
                raise ValueError(f'Verified input file identity changed: {name}')
            result = item['sha256']
        else:
            prior = self.memo.get(name)
            if prior is None or prior[0] != current or not _settled(current):
                return None
            result = prior[1]
        self.reused_names.add(name)
        self.reused_calls += 1
        self.reused_bytes += current[2]
        return result

    def remember_fresh(self, path, signature, sha256):
        if _signature(path) != signature:
            raise ValueError(f'Input changed while hashing: {path}')
        self.memo[str(Path(path).resolve())] = signature, sha256
        self.fresh_calls += 1
        self.fresh_bytes += signature[2]

    def metadata(self):
        self.check_unchanged()
        return {'profile': KIND, 'receipt_sha256': self.receipt_sha256,
                'receipt_bound_files': len(self.bindings), 'reused_unique_files': len(self.reused_names),
                'reused_digest_calls': self.reused_calls, 'reused_digest_bytes': self.reused_bytes,
                'fresh_digest_calls': self.fresh_calls, 'fresh_digest_bytes': self.fresh_bytes,
                'all_receipt_file_signatures_checked': True,
                'minimum_verified_inode_age_ns': MINIMUM_FILE_AGE_NS,
                'reused_bytes_were_freshly_hashed_in_this_process': False}


def digest_with_verified_inputs(path):
    """Retain ordinary fresh hashing unless a caller explicitly opens a session."""
    active = _ACTIVE.get()
    if active is not None:
        known = active.known(path)
        if known is not None:
            return known
        before = _signature(path)
    with Path(path).open('rb') as handle:
        result = hashlib.file_digest(handle, 'sha256').hexdigest()
    if active is not None:
        active.remember_fresh(path, before, result)
    return result


@contextmanager
def verified_input_session(receipt=None):
    """Guard a single operation; restore the previous session even on failure."""
    if receipt is None:
        yield None
        return
    current = VerifiedInputDigests(receipt)
    token = _ACTIVE.set(current)
    try:
        yield current
        current.check_unchanged()
    finally:
        _ACTIVE.reset(token)


def input_session_from_environment():
    return verified_input_session(os.environ.get(ENVIRONMENT))


def input_digest_execution():
    current = _ACTIVE.get()
    return {} if current is None else {'input_digest_verification': current.metadata()}


def write_verified_input_receipt(path, bindings, *, origin=None):
    """Publish SHA256 plus signatures already established by a complete producer.

    Callers must supply signatures captured during actual successful byte hashing,
    never signatures newly attached to an unverified expected hash. All bindings
    are guarded again before and after publication.
    """
    checked = {name: _binding(name, value) for name, value in bindings.items()}
    if not checked:
        raise ValueError('A verified input receipt cannot be empty')
    for name, item in checked.items():
        if _signature(name) != item['signature']:
            raise ValueError(f'Verified producer input changed before publication: {name}')
    recent = {name: item for name, item in checked.items() if not _settled(item['signature'])}
    if recent:
        remaining = max(MINIMUM_FILE_AGE_NS - (time.time_ns() - item['signature'][4])
                        for item in recent.values()) / 1_000_000_000
        time.sleep(min(remaining + 0.01, MINIMUM_FILE_AGE_NS / 1_000_000_000 + 0.01))
        for name, item in recent.items():
            if _signature(name) != item['signature']:
                raise ValueError(f'Verified producer input changed before publication: {name}')
            if not _settled(item['signature']):
                raise ValueError('Verified producer inode timestamps have not settled')
            # Validate young byte hashes again after the timestamp collision
            # window, instead of certifying an undetectable intervening write.
            with Path(name).open('rb') as handle:
                actual = hashlib.file_digest(handle, 'sha256').hexdigest()
            if actual != item['sha256'] or _signature(name) != item['signature']:
                raise ValueError(f'Verified producer input changed before publication: {name}')
    receipt = {'status': 'complete', 'kind': KIND,
               'inputs': {name: {**item, 'signature': list(item['signature'])} for name, item in checked.items()}}
    if origin is not None:
        receipt['origin'] = origin
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + '.partial')
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    for name, item in checked.items():
        if _signature(name) != item['signature']:
            raise ValueError(f'Verified producer input changed during publication: {name}')
    temporary.replace(dest)
    return hashlib.sha256(dest.read_bytes()).hexdigest()
