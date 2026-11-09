import json
from pathlib import Path

import pytest

from test_saved_explanation_scoring import preserved_generation, binding
from test_explanation_score_coverage import BoundOracle
from xqgeneral import clean_sft_evaluation as evaluation, score_saved_explanations as scoring, search_parent as admission
from xqgeneral.evidence import atomic_json, load_jsonl, write_jsonl
from xqgeneral.foundation_handoff import Artifacts


def recovered(tmp_path, monkeypatch):
    receipt, _ = preserved_generation(tmp_path)
    class Oracle(BoundOracle):
        def __init__(self, *args, **kwargs):super().__init__(1)
        def close(self):pass
    monkeypatch.setattr(scoring, 'Pikafish', Oracle)
    output = tmp_path / 'scores'
    scoring.run_scoring(receipt, output)
    r = json.loads(receipt.read_text());config = r['generation_config']
    core = Artifacts()
    for path in [str(Path(config['data'])/'validation.jsonl'), config['executable'], config['weights']]:
        core.add(path, binding(Path(path)))
    return receipt, output, load_jsonl(Path(config['data'])/'validation.jsonl'), dict(config, output=str(output)), core


def test_current_native_dispatch_reopens_all_recovered_raw_errors_without_model_queries(tmp_path, monkeypatch):
    receipt, output, rows, args, core = recovered(tmp_path, monkeypatch)
    # The legacy-only consumer still rejects a recovery manifest. The declared
    # validation dispatcher accepts it without changing that older contract.
    with pytest.raises(ValueError):evaluation.checked_raw(output, 'normal', rows, args, core)
    metrics, judged = admission.checked_raw(output, 'normal', rows, args, core)
    assert len(judged) == 2 and metrics['parse_valid_rate'] == .5
    assert metrics['oracle_scoring_coverage']['unscored_engine_queries'] == 1
    assert metrics['new_model_or_teacher_queries'] == 0


@pytest.mark.parametrize('damage', ['model','budget','output','source','raw','metrics','receipt_unbound','raw_origin_unbound'])
def test_recovered_evaluation_cannot_hide_changed_contracts_by_rebinding_outputs(tmp_path, monkeypatch, damage):
    receipt, output, rows, args, core = recovered(tmp_path, monkeypatch)
    proof = json.loads((output/'manifest.json').read_text())
    if damage == 'model':args['checkpoint'] = 'another-model'
    elif damage == 'budget':args['max_new_tokens'] += 1
    elif damage == 'output':args['output'] = 'another-output'
    elif damage == 'source':proof['code']['revision'] = 'foreign'
    elif damage == 'raw':
        path=output/'judged-predictions.jsonl';data=load_jsonl(path);data[0]['raw']='repaired'
        write_jsonl(path,data);proof['outputs'][str(path)]=binding(path)
    elif damage == 'metrics':
        path=output/'metrics.json';data=json.loads(path.read_text());data['first_move_legal_rate']=1
        atomic_json(path,data);proof['outputs'][str(path)]=binding(path);proof['verification']=data
    elif damage == 'receipt_unbound':proof['inputs'].pop(str(receipt))
    else:proof['inputs'].pop(json.loads(receipt.read_text())['raw_predictions']['path'])
    atomic_json(output/'manifest.json',proof)
    with pytest.raises(ValueError):admission.checked_raw(output,'normal',rows,args,core)


def native_descendant_with_recovered_normal(tmp_path, monkeypatch):
    import test_search_parent as native
    from xqgeneral.evidence import digest
    value = list(native.case(tmp_path, monkeypatch))
    original = Path(value[3]).parent/'normal'
    previous = json.loads((original/'manifest.json').read_text())
    config = dict(previous['config']);config.pop('output')
    data = Path(config['data'])/'validation.jsonl'
    receipt = tmp_path/'recovered-generation.json'
    atomic_json(receipt, {'kind':scoring.KIND,'status':'complete_raw_answers_only',
        'generation_config':config,'generation_code':previous['code'],
        'origin_artifacts':{str(original/'manifest.json'):binding(original/'manifest.json')},
        'raw_predictions':dict(path=str(original/'raw-predictions.jsonl'),**binding(original/'raw-predictions.jsonl')),
        'validation':dict(path=str(data),**binding(data)),
        'engine_artifacts':{config[k]:binding(Path(config[k])) for k in ('executable','weights')}})
    class NoAnalysis:
        def __init__(self,*args,**kwargs):pass
        def analyze(self,*args,**kwargs):raise AssertionError('Invalid original JSON needs no engine query')
        def close(self):pass
    monkeypatch.setattr(scoring,'Pikafish',NoAnalysis)
    destination = tmp_path/'recovered-normal'
    metrics = scoring.run_scoring(receipt,destination)
    proof_path = destination/'manifest.json';proof=json.loads(proof_path.read_text())
    evaluation_config = value[5]
    evaluation_config['reuse_raw_validation']={'normal':{'path':str(destination),'manifest_sha256':digest(proof_path)}}
    config_path = Path(value[4]['config']['config'])
    atomic_json(config_path,evaluation_config)
    value[4]['config']['config_sha256']=digest(config_path)
    value[4]['inputs'][str(proof_path)]=binding(proof_path)
    value[4]['outputs'].update(proof['outputs'])
    value[4]['verification']['raw_validation_by_memory']['normal']=metrics
    return value, destination


def test_complete_native_auxiliary_lineage_and_all_three_raw_memories_admit_recovered_normal(tmp_path,monkeypatch):
    import test_search_parent as native
    value,destination=native_descendant_with_recovered_normal(tmp_path,monkeypatch)
    calls=len(value[-1].commands)
    result=native.admit(value)
    assert result['every_original_paired_raw_answer_and_metric_reopened']
    assert result['full_blinded_review_raw_responses_and_rejections_reopened']
    assert result['completed_ancestor_counts_rechecked']
    assert len(value[-1].commands)==calls
    assert value[4]['verification']['raw_validation_by_memory']['normal']['parse_valid_rate']==0


@pytest.mark.parametrize('damage',['missing_recovered_manifest','changed_generation_actor'])
def test_native_descendant_gate_still_requires_complete_recovered_provenance(tmp_path,monkeypatch,damage):
    import test_search_parent as native
    value,destination=native_descendant_with_recovered_normal(tmp_path,monkeypatch)
    if damage=='missing_recovered_manifest':value[4]['inputs'].pop(str(destination/'manifest.json'))
    else:
        path=destination/'manifest.json';proof=json.loads(path.read_text())
        proof['config']['generation_config']['checkpoint']='another-checkpoint'
        atomic_json(path,proof);value[4]['inputs'][str(path)]=binding(path)
        value[5]['reuse_raw_validation']['normal']['manifest_sha256']=binding(path)['sha256']
    with pytest.raises(ValueError):native.admit(value)
