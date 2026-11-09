"""The typed unscored exception preserves the historical successful decoder."""
from pathlib import Path
import ast
import json

import pytest

from xqgeneral import search_protection as protection
from xqgeneral.evidence import atomic_json
from test_search_protection import setup
from test_oracle_policy_corpus import binding


LEGACY_ORACLE='2bed40736e2d0e4fe05405783a60bb85073ee92876f79aadf6863ea1c53edb19'


def legacy_corpus(tmp_path):
    args,counts,shared=setup(tmp_path)
    root=args['output'];path=root/'contract.json';contract=json.loads(path.read_text())
    contract['closure']['oracle.py']=LEGACY_ORACLE;atomic_json(path,contract)
    target=root/'manifest.json';proof=json.loads(target.read_text())
    proof['outputs'][str(path)]=binding(path)
    proof['code']['source_sha256']['src/xqgeneral/oracle.py']=LEGACY_ORACLE
    atomic_json(target,proof)
    return args,counts,shared


def test_real_protection_preparation_reopens_old_scored_references_after_typed_error_upgrade(tmp_path):
    args,counts,_=legacy_corpus(tmp_path);dest=tmp_path/'protection'
    result=protection.prepare(args['output'],dest)
    assert result['protected_positions']==counts['extended_protected_positions']
    proof=json.loads((dest/'manifest.json').read_text())
    compatibility=proof['config']['native_decoder_compatibility']
    assert compatibility['profile']=='exact_successful_decoder_typed_unscored_error_upgrade_v2'
    assert compatibility['AST_canonical_profile']=='omit_only_empty_type_params_v1'
    assert compatibility['producer_oracle_sha256']==LEGACY_ORACLE
    assert compatibility['all_other_closure_modules_byte_identical'] is True
    assert compatibility['artifact_source_code_executed'] is False
    assert protection.prepare(args['output'],dest,resume=True)==result


@pytest.mark.parametrize('damage',['unknown_oracle','other_module','missing_module','producer_identity'])
def test_decoder_upgrade_does_not_admit_other_source_changes(tmp_path,damage):
    args,_,_=legacy_corpus(tmp_path);root=args['output'];p=root/'contract.json';v=json.loads(p.read_text())
    if damage=='unknown_oracle':v['closure']['oracle.py']='0'*64
    elif damage=='other_module':v['closure']['rules.py']='0'*64
    elif damage=='missing_module':v['closure'].pop('rules.py')
    atomic_json(p,v);p=root/'manifest.json';v=json.loads(p.read_text())
    v['outputs'][str(root/'contract.json')]=binding(root/'contract.json');atomic_json(p,v)
    if damage=='producer_identity':
        v['code']['source_sha256']['src/xqgeneral/oracle.py']='0'*64;atomic_json(p,v)
    with pytest.raises(ValueError):protection.prepare(root,tmp_path/'protection')


@pytest.mark.parametrize('before,after',[
    ('ranks, scored = {}, {}','ranks, scored = {}, {"a0a1": {}}'),
    ('if best not in scored:','if False:'),
    ("class UnscoredBestMove(RuntimeError):","class UnscoredBestMove(ValueError):"),
    ("'requested_nodes': nodes","'requested_nodes': nodes + 1"),
])
def test_exact_AST_comparison_rejects_scoring_or_exception_semantic_changes(before,after):
    path=Path(protection.__file__).with_name('oracle.py');source=path.read_text()
    assert before in source
    assert protection._typed_unscored_upgrade(source) is True
    assert protection._typed_unscored_upgrade(source.replace(before,after,1)) is False


def test_same_verified_decoder_accepts_the_python311_AST_shape(monkeypatch):
    source=Path(protection.__file__).with_name('oracle.py').read_text()
    original_parse=ast.parse
    def python311_shape(*args,**kwargs):
        tree=original_parse(*args,**kwargs)
        for node in ast.walk(tree):
            if 'type_params' in node._fields and node.type_params==[]:
                node._fields=tuple(field for field in node._fields if field!='type_params')
                del node.type_params
        return tree
    monkeypatch.setattr(protection.ast,'parse',python311_shape)
    assert protection._typed_unscored_upgrade(source) is True


def test_real_generic_type_parameters_are_not_ignored():
    source=Path(protection.__file__).with_name('oracle.py').read_text()
    generic=source.replace('class UnscoredBestMove(RuntimeError):','class UnscoredBestMove[T](RuntimeError):')
    assert generic!=source
    assert protection._typed_unscored_upgrade(generic) is False
