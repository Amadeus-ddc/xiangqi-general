"""Real OS flock and pipe checks, without GPU or model inference."""
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

fcntl = pytest.importorskip('fcntl')
HELPER = Path(__file__).resolve().parents[1] / 'scripts/match_startup_lease.py'
CHILD = "from pathlib import Path\nimport fcntl,json,sys,time\np=Path(sys.argv[1]); mode=sys.argv[2]\nprint('endpoint=https://example.invalid/help address=192.0.2.1:9999',flush=True)\nif mode=='failure':raise SystemExit(7)\nif mode=='completed':raise SystemExit(0)\nturns=[{'raw':'controlled existing turn'}] if mode=='resume' else []\ndef save():\n q=p/'progress/game-001.json';q.parent.mkdir(parents=True,exist_ok=True)\n t=q.with_suffix('.partial');t.write_text(json.dumps({'game':{'turns':turns}}));t.replace(q)\nsave()\nwith (p/'lease.lock').open('a+') as lock:\n try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n except BlockingIOError:pass\n else:raise AssertionError('startup lease was released before a new model turn')\n turns.append({'raw':'controlled new turn; not neural inference'});save()\n deadline=time.monotonic()+8\n while not (p/'release.json').exists():\n  if time.monotonic()>deadline:raise RuntimeError('controlled handshake did not release')\n  time.sleep(.01)\n fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n (p/'other-process-acquired.json').write_text(json.dumps({'acquired_after_new_turn':True}))\n"


@pytest.mark.parametrize('case', ['normal', 'resume', 'completed', 'failure'])
def test_startup_lease_tracks_new_raw_turns_and_releases_on_completion_or_failure(tmp_path, case):
    stream = runpy.run_path(str(HELPER))['stream_match_process']
    program = tmp_path / 'child.py'
    program.write_text(CHILD)
    if case == 'resume':
        progress = tmp_path / 'progress'; progress.mkdir()
        (progress / 'game-001.json').write_text(json.dumps({'game': {'turns': [{'raw': 'controlled existing turn'}]}}))
    releases = []
    def released(value):
        releases.append(value)
        (tmp_path / 'release.json').write_text(json.dumps(value))
    with (tmp_path / 'lease.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        command = [sys.executable, str(program), str(tmp_path), case]
        if case == 'failure':
            with pytest.raises(subprocess.CalledProcessError) as error:
                stream(command, tmp_path / 'log.txt', '', tmp_path / 'progress', lease, released, poll_seconds=.01)
            assert error.value.returncode == 7
        else:
            stream(command, tmp_path / 'log.txt', '', tmp_path / 'progress', lease, released, poll_seconds=.01)
        with (tmp_path / 'lease.lock').open('a+') as observer:
            fcntl.flock(observer, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert len(releases) == 1
    release = releases[0]
    if case in ['normal', 'resume']:
        assert json.loads((tmp_path / 'other-process-acquired.json').read_text())['acquired_after_new_turn']
        assert release['reason'] == 'new_raw_model_turn_observed'
        assert release['observed_raw_turns'] == release['previous_raw_turns'] + 1
    else:
        assert release['reason'] == 'child_exited_without_observed_new_raw_turn'
        assert release['child_returncode'] == (7 if case == 'failure' else 0)
    if case == 'resume':
        assert release['previous_raw_turns'] == 1 and release['observed_raw_turns'] == 2
    log = (tmp_path / 'log.txt').read_text()
    assert '[REDACTED_URL]' in log and '[REDACTED_ADDRESS]' in log
    assert 'https://example.invalid' not in log and '192.0.2.1' not in log
