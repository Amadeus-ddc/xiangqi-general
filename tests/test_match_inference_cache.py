"""Match CLI inference cache contracts; no throughput or chess-strength claim."""
import json
from pathlib import Path
import sys

import pytest
import torch

from test_inference_autocast_cache import controlled_predictor
from test_match_resume import Engine, Student, cli_arguments
from xqgeneral import evaluate_games, inference
from xqgeneral.evidence import load_jsonl


def players(monkeypatch, *, ignored=False):
    calls = []

    def student(*args, **options):
        value = Student()
        enabled = options.get('autocast_cache_enabled', False)
        value.config = {'autocast_cache_enabled': not enabled if ignored else enabled}
        calls.append(options)
        return value

    monkeypatch.setattr(inference, 'Predictor', student)
    monkeypatch.setattr(evaluate_games, 'Pikafish', Engine)
    return calls


@pytest.mark.parametrize('setting', [None, 'on', 'off'])
def test_optional_override_reaches_loader_and_completed_readback_never_reloads(tmp_path, monkeypatch, setting):
    argv, dest, artifacts = cli_arguments(tmp_path)
    original_checkpoint = artifacts['checkpoint'].read_bytes()
    calls = players(monkeypatch)
    if setting is not None:
        argv += ['--autocast-cache', setting]
    monkeypatch.setattr(sys, 'argv', argv)
    evaluate_games.main()
    expected = {} if setting is None else {'autocast_cache_enabled': setting == 'on'}
    assert calls == [expected]
    config = json.loads((dest / 'contract.json').read_text())['config']
    metrics = json.loads((dest / 'metrics.json').read_text())
    if setting is None:
        assert 'autocast_cache' not in config and 'inference_autocast_cache' not in metrics
    else:
        assert config['autocast_cache'] == setting
        assert metrics['inference_autocast_cache'] == {
            'requested_override': setting == 'on', 'observed_this_invocation': setting == 'on',
            'model_loaded_this_invocation': True, 'checkpoint_settings_modified': False}
    assert metrics['model_oracle_repairs'] == 0 and metrics['censored'] == 8
    assert artifacts['checkpoint'].read_bytes() == original_checkpoint
    saved = {p: p.read_bytes() for p in dest.rglob('*') if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError('Completed native replay loaded a player')

    monkeypatch.setattr(inference, 'Predictor', forbidden)
    monkeypatch.setattr(evaluate_games, 'Pikafish', forbidden)
    monkeypatch.setattr(sys, 'argv', [*argv, '--resume'])
    evaluate_games.main()
    assert all(p.read_bytes() == raw for p, raw in saved.items())


@pytest.mark.parametrize('before,after', [(None,'on'),(None,'off'),('on',None),('off',None),
                                         ('on','off'),('off','on')])
def test_cache_changes_rejected_before_loading_any_player(tmp_path, monkeypatch, before, after):
    argv, dest, _ = cli_arguments(tmp_path)
    players(monkeypatch)
    original = argv + (['--autocast-cache', before] if before is not None else [])
    monkeypatch.setattr(sys, 'argv', original)
    evaluate_games.main()

    def forbidden(*args, **kwargs):
        raise AssertionError('Changed match contract reached a model or engine')

    monkeypatch.setattr(inference, 'Predictor', forbidden)
    monkeypatch.setattr(evaluate_games, 'Pikafish', forbidden)
    changed = argv + (['--autocast-cache', after] if after is not None else [])
    monkeypatch.setattr(sys, 'argv', [*changed, '--resume'])
    with pytest.raises(ValueError, match='original input, source and evaluation contract'):
        evaluate_games.main()


@pytest.mark.parametrize('setting', ['on', 'off'])
def test_ignored_override_fails_before_engine_or_generation(tmp_path, monkeypatch, setting):
    argv, dest, _ = cli_arguments(tmp_path)
    players(monkeypatch, ignored=True)

    def forbidden(*args, **kwargs):
        raise AssertionError('Ignored inference cache option reached a game')

    monkeypatch.setattr(evaluate_games, 'Pikafish', forbidden)
    monkeypatch.setattr(sys, 'argv', [*argv, '--autocast-cache', setting])
    with pytest.raises(ValueError, match='cache override was not applied'):
        evaluate_games.main()
    assert not (dest / 'manifest.json').exists() and not (dest / 'progress').exists()


@pytest.mark.parametrize('setting', ['true', 'ON', '0'])
def test_invalid_option_never_reaches_output_or_model(tmp_path, monkeypatch, setting):
    argv, dest, _ = cli_arguments(tmp_path)
    monkeypatch.setattr(sys, 'argv', [*argv, '--autocast-cache', setting])
    with pytest.raises(SystemExit) as error:
        evaluate_games.main()
    assert error.value.code == 2 and not dest.exists()


@pytest.mark.parametrize('setting', ['on', 'off'])
def test_partial_resume_preserves_raw_prefix_and_same_cache_setting(tmp_path, monkeypatch, setting):
    argv, dest, _ = cli_arguments(tmp_path)
    argv += ['--autocast-cache', setting]
    calls = players(monkeypatch)
    original = evaluate_games.save_game_checkpoint

    def interrupted(path, spec, game):
        original(path, spec, game)
        if Path(path).name == 'game-002.json' and len(game['turns']) == 1:
            raise RuntimeError('saved cache-controlled interruption')

    monkeypatch.setattr(evaluate_games, 'save_game_checkpoint', interrupted)
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(RuntimeError, match='cache-controlled interruption'):
        evaluate_games.main()
    prefix = (dest / 'game-001.json').read_bytes()
    monkeypatch.setattr(evaluate_games, 'save_game_checkpoint', original)
    monkeypatch.setattr(sys, 'argv', [*argv, '--resume'])
    evaluate_games.main()
    assert calls == [{'autocast_cache_enabled': setting == 'on'}] * 2
    assert (dest / 'game-001.json').read_bytes() == prefix
    assert json.loads((dest / 'metrics.json').read_text())['saved_turns_reused'] == 5


@pytest.mark.parametrize('setting', ['on', 'off'])
def test_saved_complete_games_without_manifest_resume_without_loading_a_model(tmp_path, monkeypatch, setting):
    argv, dest, _ = cli_arguments(tmp_path)
    argv += ['--autocast-cache', setting]
    players(monkeypatch)
    original = evaluate_games.write_jsonl

    def interrupted(*args):
        raise RuntimeError('all games saved before final metrics')

    monkeypatch.setattr(evaluate_games, 'write_jsonl', interrupted)
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(RuntimeError, match='all games saved'):
        evaluate_games.main()
    saved = {p: p.read_bytes() for p in dest.glob('game-*.json')}
    assert len(saved) == 8 and not (dest / 'manifest.json').exists()

    def forbidden(*args, **kwargs):
        raise AssertionError('Closed games required another model load')

    monkeypatch.setattr(evaluate_games, 'write_jsonl', original)
    monkeypatch.setattr(inference, 'Predictor', forbidden)
    monkeypatch.setattr(evaluate_games, 'Pikafish', forbidden)
    monkeypatch.setattr(sys, 'argv', [*argv, '--resume'])
    evaluate_games.main()
    metrics = json.loads((dest / 'metrics.json').read_text())
    assert metrics['inference_autocast_cache'] == {
        'requested_override': setting == 'on', 'observed_this_invocation': None,
        'model_loaded_this_invocation': False, 'checkpoint_settings_modified': False}
    assert all(p.read_bytes() == value for p, value in saved.items())


@pytest.mark.parametrize('setting', ['on', 'off'])
def test_match_flag_reaches_actual_native_torch_generation_context_and_keeps_raw_forfeits(tmp_path, monkeypatch, setting):
    argv, dest, _ = cli_arguments(tmp_path)
    controlled, seen = controlled_predictor(setting != 'on')
    saved_config = dict(controlled.config, mode='text_lora')

    def generate(**kwargs):
        seen.append(torch.is_autocast_cache_enabled())
        assert kwargs['max_new_tokens'] == 768 and kwargs.get('do_sample') is False
        return torch.cat([kwargs['input_ids'], torch.full((len(kwargs['input_ids']), 1), 3)], dim=1)

    controlled.model.base.generate = generate
    monkeypatch.setattr(inference, 'load_checkpoint',
                        lambda *args: (controlled.model, controlled.tokenizer, saved_config))
    original = inference.Predictor

    def cpu_predictor(checkpoint, expert, **options):
        result = original(checkpoint, expert, device='cpu', **options)
        result.features = lambda record: [torch.zeros((1, 90, 4))]
        return result

    monkeypatch.setattr(inference, 'Predictor', cpu_predictor)
    monkeypatch.setattr(evaluate_games, 'Pikafish', Engine)
    monkeypatch.setattr(sys, 'argv', [*argv, '--autocast-cache', setting])
    evaluate_games.main()
    assert seen == [setting == 'on'] * 8
    assert saved_config['autocast_cache_enabled'] is (setting != 'on')
    metrics = json.loads((dest / 'metrics.json').read_text())
    assert metrics['raw_model_forfeits'] == 8 and metrics['model_oracle_repairs'] == 0
    games = load_jsonl(dest / 'games.jsonl')
    assert all(g['turns'][-1]['raw'] == 'raw model output' for g in games)
    assert all(g['reason'] == 'raw_model_invalid_move_forfeit' for g in games)
