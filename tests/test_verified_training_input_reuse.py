"""Exercise input identity at the real initial/resumed training entrypoint."""
import json
from pathlib import Path
import sys
import time

import pytest

from xqgeneral import training
from xqgeneral.evidence import digest, file_signature


class IdentityBoundaryReached(Exception):
    pass


def test_initial_and_resumed_training_reuse_unchanged_verified_inputs(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    for split in ('train', 'validation'):
        (data / f'{split}.jsonl').write_text('{}\n')
    features = tmp_path / 'features.pt'
    features.write_bytes(b'bound immutable feature bytes\n' * 128)
    config = {'seed': 13, 'output': str(tmp_path / 'training'),
              'feature_path': str(features), 'data_path': str(data)}
    config_path = tmp_path / 'training.config.json'
    config_path.write_text(json.dumps(config))
    paths = training.training_input_paths(config)
    # The producer hashes stable inputs; fresh inode timestamps can collide
    # across rapid same-size writes on this server's filesystems.
    time.sleep(2.02)
    bindings = {str(Path(p).resolve()): {'sha256': digest(p), 'bytes': Path(p).stat().st_size,
                                      'signature': list(file_signature(p))} for p in paths}
    receipt = tmp_path / 'verified-inputs.json'
    receipt.write_text(json.dumps({'status': 'complete', 'kind': 'verified_immutable_inputs_v1',
                                   'inputs': bindings}))
    monkeypatch.setenv('XQGENERAL_VERIFIED_INPUT_RECEIPT', str(receipt))
    full_reads, consumed = [], []

    original_open = Path.open

    def counted_open(path, mode='r', *args, **kwargs):
        name = str(path.resolve())
        if name in bindings and mode == 'rb':
            full_reads.append(name)
        return original_open(path, mode, *args, **kwargs)

    def after_identity(path, **kwargs):
        consumed.append(kwargs['verified_hashes'])
        raise IdentityBoundaryReached

    monkeypatch.setattr(Path, 'open', counted_open)
    monkeypatch.setattr(training, 'open_feature_cache', after_identity)
    for resume in (False, True):
        argv = ['xqgeneral.train', '--config', str(config_path)]
        if resume:
            argv += ['--resume', str(tmp_path / 'latest.pt')]
        monkeypatch.setattr(sys, 'argv', argv)
        with pytest.raises(IdentityBoundaryReached):
            training.main()
    expected = {str(p): bindings[str(Path(p).resolve())]['sha256'] for p in paths}
    assert consumed == [expected, expected]
    assert all(list(file_signature(name)) == bound['signature'] for name, bound in bindings.items())
    assert full_reads == [], (
        'Initial/resumed training reread already verified immutable inputs: '
        f'{full_reads}; the producer already hashed these exact file identities')
