import hashlib
import json
import os
from pathlib import Path

import pytest

from xqgeneral.evidence import digest, file_signature, manifest
from xqgeneral.verified_inputs import (ENVIRONMENT, input_digest_execution,
    verified_input_session, write_verified_input_receipt)


@pytest.fixture
def verified(tmp_path):
    data = tmp_path / 'features.pt'
    data.write_bytes(b'original feature bytes')
    value = {'sha256': digest(data), 'bytes': data.stat().st_size,
             'signature': list(file_signature(data))}
    receipt = tmp_path / 'receipt.json'
    write_verified_input_receipt(receipt, {str(data): value}, origin={'fixture': True})
    return data, receipt, value


def count_binary_reads(monkeypatch, data):
    calls = []
    original = Path.open

    def opened(path, mode='r', *args, **kwargs):
        if path.resolve() == data.resolve() and mode == 'rb':
            calls.append(path)
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', opened)
    return calls


def test_explicit_session_reuses_verified_bytes_without_binary_reads(verified, monkeypatch):
    data, receipt, bound = verified
    calls = count_binary_reads(monkeypatch, data)
    with verified_input_session(receipt):
        assert digest(data) == digest(data) == bound['sha256']
        meta = input_digest_execution()['input_digest_verification']
        assert meta['reused_digest_calls'] == 2 and meta['fresh_digest_calls'] == 0
        assert not meta['reused_bytes_were_freshly_hashed_in_this_process']
    assert not calls and input_digest_execution() == {}


def test_environment_does_not_change_unscoped_digest_behavior(verified, monkeypatch):
    data, receipt, bound = verified
    monkeypatch.setenv(ENVIRONMENT, str(receipt))
    calls = count_binary_reads(monkeypatch, data)
    assert digest(data) == digest(data) == bound['sha256']
    assert len(calls) == 2


@pytest.mark.parametrize('replacement', [False, True])
def test_same_size_change_with_original_mtime_rejects_receipt(verified, replacement, monkeypatch):
    data, receipt, bound = verified
    timestamp = data.stat().st_mtime_ns
    target = data.with_suffix('.replacement') if replacement else data
    target.write_bytes(b'x' * bound['bytes'])
    os.utime(target, ns=(timestamp, timestamp))
    if replacement:
        target.replace(data)
    assert data.stat().st_size == bound['bytes'] and data.stat().st_mtime_ns == timestamp
    assert list(file_signature(data)) != bound['signature']
    calls = count_binary_reads(monkeypatch, data)
    with pytest.raises(ValueError, match='identity changed'):
        with verified_input_session(receipt):
            digest(data)
    assert not calls


def test_mutation_during_session_is_rejected_and_failed_session_is_reset(verified):
    data, receipt, bound = verified
    with pytest.raises(ValueError, match='identity changed'):
        with verified_input_session(receipt):
            assert digest(data) == bound['sha256']
            data.write_bytes(b'x' * bound['bytes'])
            digest(data)
    assert input_digest_execution() == {}
    assert digest(data) == hashlib.sha256(data.read_bytes()).hexdigest() != bound['sha256']


def test_session_checks_unused_bound_file_at_normal_exit(verified):
    data, receipt, bound = verified
    with pytest.raises(ValueError, match='identity changed'):
        with verified_input_session(receipt):
            data.write_bytes(b'x' * bound['bytes'])
    assert input_digest_execution() == {}


def test_young_outputs_keep_fresh_hashing_across_rapid_mutation(verified, tmp_path, monkeypatch):
    _, receipt, _ = verified
    output = tmp_path / 'latest.pt'
    output.write_bytes(b'first checkpoint')
    calls = count_binary_reads(monkeypatch, output)
    with verified_input_session(receipt):
        first = digest(output)
        assert digest(output) == first
        output.write_bytes(b'next checkpoint!')
        second = digest(output)
        assert second != first and digest(output) == second
    assert len(calls) == 4


def test_manifest_declares_reused_and_fresh_hashes(verified, tmp_path):
    data, receipt, bound = verified
    output = tmp_path / 'output.json'
    output.write_text('{}')
    with verified_input_session(receipt):
        result = manifest('fixture', {}, [data], [output], {}, code={'fixture': True})
    assert result['inputs'][str(data)] == {'sha256': bound['sha256'], 'bytes': bound['bytes']}
    meta = result['runtime_execution']['input_digest_verification']
    assert meta['reused_digest_calls'] == 1 and meta['fresh_digest_calls'] == 1
    assert meta['receipt_sha256'] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert not meta['reused_bytes_were_freshly_hashed_in_this_process']


def test_unscoped_manifest_preserves_default_shape(verified):
    data, _, _ = verified
    assert 'runtime_execution' not in manifest('fixture', {}, [data], [], code={'fixture': True})


@pytest.mark.parametrize('change', [
    lambda value: value.update(status='incomplete'),
    lambda value: value.update(kind='different'),
    lambda value: value.update(inputs={}),
    lambda value: next(iter(value['inputs'].values())).update(sha256='invalid'),
    lambda value: next(iter(value['inputs'].values())).update(bytes=True),
    lambda value: next(iter(value['inputs'].values())).update(signature=[1, 2, 3]),
])
def test_incomplete_or_malformed_receipts_are_rejected(verified, change):
    _, receipt, _ = verified
    value = json.loads(receipt.read_text())
    change(value)
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        with verified_input_session(receipt):
            pass


def test_receipt_publisher_refuses_stale_verified_signatures(verified, tmp_path):
    data, _, bound = verified
    data.write_bytes(b'x' * bound['bytes'])
    with pytest.raises(ValueError, match='changed before publication'):
        write_verified_input_receipt(tmp_path / 'new.json', {str(data): bound})


def test_raw_receipt_cannot_enable_reuse_for_young_inode(tmp_path):
    data = tmp_path / 'young.pt'
    data.write_bytes(b'young feature bytes')
    receipt = tmp_path / 'raw-receipt.json'
    receipt.write_text(json.dumps({'status': 'complete', 'kind': 'verified_immutable_inputs_v1',
        'inputs': {str(data): {'sha256': digest(data), 'bytes': data.stat().st_size,
                             'signature': list(file_signature(data))}}}))
    with pytest.raises(ValueError, match='recently written'):
        with verified_input_session(receipt):
            digest(data)


def test_young_producer_bytes_are_rechecked_after_settling(tmp_path):
    data = tmp_path / 'young.pt'
    data.write_bytes(b'first')
    bound = {'sha256': digest(data), 'bytes': data.stat().st_size,
             'signature': list(file_signature(data))}
    # Both unchanged metadata and an ordinary detectable change must refuse
    # the old byte hash; timestamp equality is never treated as proof here.
    data.write_bytes(b'other')
    with pytest.raises(ValueError, match='changed before publication'):
        write_verified_input_receipt(tmp_path / 'receipt.json', {str(data): bound})


def test_nested_failure_restores_outer_receipt(verified, tmp_path):
    data, receipt, bound = verified
    other = tmp_path / 'other.pt'
    other.write_bytes(b'other features')
    second = tmp_path / 'second.json'
    write_verified_input_receipt(second, {str(other): {'sha256': digest(other),
        'bytes': other.stat().st_size, 'signature': list(file_signature(other))}})
    with verified_input_session(receipt):
        with pytest.raises(RuntimeError, match='body failed'):
            with verified_input_session(second):
                raise RuntimeError('body failed')
        assert digest(data) == bound['sha256']
        assert input_digest_execution()['input_digest_verification']['receipt_sha256'] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert input_digest_execution() == {}
