"""The real data preflight does not declare decoder weight files."""
import json
from pathlib import Path

import pytest

from test_sft_pipeline import install_runner, setup_data
from xqgeneral import sft_pipeline, sft_preflight
from xqgeneral.evidence import atomic_json, digest, manifest
from xqgeneral.foundation_handoff import export_curriculum


def explicit_base_fixture(tmp_path, layout='single'):
    curriculum, recipe_path, token_path = setup_data(tmp_path)
    foundation = json.loads((curriculum/'manifest.json').read_text())['config']
    model = Path(foundation['model_path'])
    for name in ['tokenizer_config.json', 'tokenizer.json']:
        atomic_json(model/name, {})
    if layout == 'sharded':
        (model/'model.safetensors').unlink()
        for name in ['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']:
            (model/name).write_bytes(name.encode())
        atomic_json(model/'model.safetensors.index.json', {'weight_map': {
            'first': 'model-00001-of-00002.safetensors', 'second': 'model-00002-of-00002.safetensors'}})
    # Match the actual data-only producer instead of injecting weights into it.
    token = json.loads(token_path.read_text())
    data_proof = Path(token['config']['foundation_preflight'])
    atomic_json(data_proof, manifest('recorded_engine_clean_latent_foundation_preflight', {},
        [Path(foundation['data_path'])/'validation.jsonl'], [], {}))
    token['inputs'][str(data_proof)] = {'sha256':digest(data_proof),'bytes':data_proof.stat().st_size}
    atomic_json(token_path, token)
    pinned = tmp_path/'pinned-decoder.json'
    atomic_json(pinned, manifest('pinned_local_decoder_assets', {
        'model_path':str(model), 'model_revision':foundation['model_revision']},
        sorted(p for p in model.iterdir() if p.is_file()), [], {'no_model_loaded':True}))
    recipe = json.loads(recipe_path.read_text())
    recipe['pinned_base_model_manifest'] = {'path':str(pinned),'sha256':digest(pinned)}
    atomic_json(recipe_path,recipe)
    return curriculum,recipe_path,token_path,pinned,model


@pytest.mark.parametrize('layout',['single','sharded'])
def test_data_only_foundation_uses_explicit_complete_pinned_decoder(tmp_path,monkeypatch,layout):
    curriculum,recipe,token,pinned,model=explicit_base_fixture(tmp_path,layout)
    runner=install_runner(monkeypatch)
    handoff=tmp_path/'handoff';export_curriculum(curriculum,handoff)
    probe=tmp_path/'probe'
    result=sft_preflight.run_preflight(handoff/'adapter.pt',recipe,token,probe)
    assert result['actual_continuous_four_and_two_plus_two_updates_compared']
    assert len(runner.jobs)==3
    contract=json.loads((probe/'contract.json').read_text())
    expected=json.loads(pinned.read_text())['inputs']
    assert contract['base_assets']==expected
    proof=json.loads((probe/'manifest.json').read_text())
    assert proof['inputs'][str(pinned)]['sha256']==digest(pinned)
    assert all(proof['inputs'][path]==binding for path,binding in expected.items())


@pytest.mark.parametrize('problem,match',[
    ('manifest_digest','Pinned decoder manifest'),('revision','selected decoder'),
    ('different_model','selected decoder'),('missing_shard_binding','complete local decoder'),
    ('changed_weights','completed recorded search source'),('extra_asset','complete local decoder'),
    ('legacy_conflict','foundation decoder binding')])
def test_invalid_explicit_decoder_rejected_before_model_or_worker(tmp_path,monkeypatch,problem,match):
    curriculum,recipe,token,pinned,model=explicit_base_fixture(tmp_path,'sharded')
    r=json.loads(recipe.read_text());p=json.loads(pinned.read_text())
    if problem=='manifest_digest':r['pinned_base_model_manifest']['sha256']='0'*64
    elif problem=='revision':p['config']['model_revision']='different'
    elif problem=='different_model':p['config']['model_path']=str(tmp_path/'other-model')
    elif problem=='missing_shard_binding':p['inputs'].pop(str(model/'model-00002-of-00002.safetensors'))
    elif problem=='changed_weights':(model/'model-00001-of-00002.safetensors').write_bytes(b'changed')
    elif problem=='extra_asset':
        extra=tmp_path/'unrelated';extra.write_bytes(b'other')
        p['inputs'][str(extra)]={'sha256':digest(extra),'bytes':extra.stat().st_size}
    elif problem=='legacy_conflict':
        t=json.loads(token.read_text());f=Path(t['config']['foundation_preflight']);old=json.loads(f.read_text())
        old['inputs'][str(model/'model-00001-of-00002.safetensors')]={'sha256':'0'*64,'bytes':1}
        atomic_json(f,old);t['inputs'][str(f)]={'sha256':digest(f),'bytes':f.stat().st_size};atomic_json(token,t)
    if problem not in ['manifest_digest','changed_weights','legacy_conflict']:
        atomic_json(pinned,p);r['pinned_base_model_manifest']['sha256']=digest(pinned)
    atomic_json(recipe,r)
    runner=install_runner(monkeypatch)
    handoff=tmp_path/'handoff';export_curriculum(curriculum,handoff)
    with pytest.raises(ValueError,match=match):
        sft_preflight.run_preflight(handoff/'adapter.pt',recipe,token,tmp_path/'probe')
    assert not runner.jobs and not (tmp_path/'probe').exists()


def test_pinned_decoder_cannot_change_between_probe_and_formal_sft(tmp_path,monkeypatch):
    curriculum,recipe,token,pinned,model=explicit_base_fixture(tmp_path)
    runner=install_runner(monkeypatch)
    original=runner.__call__
    def change_after_formal(command,**kwargs):
        value=original(command,**kwargs)
        if 'xqgeneral.sft' in command:
            (model/'model.safetensors').write_bytes(b'changed during formal controller')
        return value
    monkeypatch.setattr(sft_pipeline.subprocess,'run',change_after_formal)
    output=tmp_path/'pipeline'
    with pytest.raises(ValueError,match='completed recorded search source|inputs changed'):
        sft_pipeline.run_pipeline(curriculum,'owned-session',recipe,token,output)
    assert not (output/'manifest.json').exists()
    assert json.loads((output/'state.json').read_text())['status']=='failed'
