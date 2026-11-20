"""Temporary initialization models must die before a GPU probe is launched."""
import gc
import weakref

import pytest

from test_sft_pipeline import install_runner, setup_data
from xqgeneral import modeling, sft_preflight
from xqgeneral.foundation_handoff import export_curriculum


@pytest.mark.parametrize('save_failure', [False, True])
def test_reference_model_cycles_released_before_workers_even_if_save_fails(
        tmp_path, monkeypatch, save_failure):
    curriculum, recipe, token = setup_data(tmp_path)
    runner = install_runner(monkeypatch)
    handoff = tmp_path / 'handoff'
    export_curriculum(curriculum, handoff)
    original_load = modeling.load_model
    references = []

    def load_with_hook_cycle(*args, **kwargs):
        model, tokenizer = original_load(*args, **kwargs)

        def inject(_module, arguments):
            # The real board injection hook also closes over its owning model.
            assert model.training
            return arguments

        model.register_forward_pre_hook(inject)
        references.append(weakref.ref(model))
        return model, tokenizer

    def checked_worker(command, **kwargs):
        if 'xqgeneral.sft_probe_worker' in command:
            assert references and all(reference() is None for reference in references), (
                'Temporary initialization model remains alive at GPU worker launch')
        return runner(command, **kwargs)

    monkeypatch.setattr(modeling, 'load_model', load_with_hook_cycle)
    monkeypatch.setattr(sft_preflight.subprocess, 'run', checked_worker)
    if save_failure:
        def failed_save(*_args, **_kwargs):
            raise OSError('controlled initial checkpoint save failure')
        monkeypatch.setattr(sft_preflight, 'atomic_checkpoint', failed_save)

    previously_enabled = gc.isenabled()
    gc.disable()
    try:
        if save_failure:
            with pytest.raises(OSError, match='controlled initial checkpoint save failure'):
                sft_preflight.run_preflight(handoff / 'adapter.pt', recipe, token, tmp_path / 'probe')
            assert not runner.jobs
        else:
            result = sft_preflight.run_preflight(
                handoff / 'adapter.pt', recipe, token, tmp_path / 'probe')
            assert result['actual_continuous_four_and_two_plus_two_updates_compared']
            assert result['exact_resume']['mismatched_fields'] == []
            assert len(runner.jobs) == 3
        assert references and all(reference() is None for reference in references)
        assert not gc.isenabled()
    finally:
        if previously_enabled:
            gc.enable()
        gc.collect()
