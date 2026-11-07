"""Typed CPU token/cache and finite-budget gates for seven-stage auxiliary SFT."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import hashlib
import math
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory

from .distillation_preflight import completed_parent
from .evidence import atomic_json,code_identity,digest,manifest
from .explanation_preflight import token_chunk,token_initializer
from .extend_features import validate_cache
from .feature_store import FeatureStore,feature_paths
from .finite_training import FINITE_PROFILE,FULL_SOURCE_PROFILE,mixture_counts
from .foundation_preflight import bounded_results
from .oracle_policy_corpus import ALL_STAGES,KIND as CORPUS_KIND,PROFILE as CORPUS_PROFILE,_basic,_lines
from .oracle_policy_data import _binding
from .recorded_search_inputs import BoundInputs
from .sft import checked_parent,prepare_config
from .sft_preflight import pinned_decoder_assets
from .verified_inputs import input_digest_execution,input_session_from_environment,write_verified_input_receipt

KIND='latent_only_auxiliary_policy_data_preflight'
PROFILE='seven_stage_auxiliary_policy_tokens_v1'
VALIDATION_PROFILE='seven_stage_complete_original_validation_v1'
FILES=('token-lengths.jsonl','counts.json','contract.json','verified-inputs.json')
CLOSURE=('auxiliary_policy_preflight.py','oracle_policy_corpus.py','distillation_preflight.py',
    'explanation_preflight.py','sft_preflight.py','sft.py','training.py','feature_store.py',
    'extend_features.py','model_bundle.py','modeling.py','verified_inputs.py','evidence.py')


def _closure():
    hashes=code_identity()['source_sha256']
    return {name:hashes['src/xqgeneral/'+name] for name in CLOSURE}


def inherited_cache(features,saved,bound):
    """Allow multiple completed extensions while preserving exact parent tensors."""
    target=Path(saved['feature_path']).resolve();features=Path(features).resolve()
    if features==target:raise ValueError('Auxiliary inputs require their completed extended feature store')
    for path in [*feature_paths(features),*feature_paths(target)]:bound.bind(path)
    hashes={name:value['sha256'] for name,value in bound.artifacts.items()}
    cache=FeatureStore(features,verified_hashes=hashes);original=FeatureStore(target,verified_hashes=hashes)
    encoder=code_identity()['source_sha256']['src/xqgeneral/expert.py']
    if (cache.keys[:len(original.keys)]!=original.keys or cache.shards[:len(original.shards)]!=original.shards or
            cache.depths!=original.depths or cache.width!=original.width or
            any(item.proof.get('code',{}).get('source_sha256',{}).get('src/xqgeneral/expert.py')!=encoder
                for item in [cache,original]) or
            cache.proof['verification']['expert_sha256']!=original.proof['verification']['expert_sha256']):
        raise ValueError('The complete original feature-key/tensor prefix or expert identity changed')
    chain=[];current=features;seen=set()
    while current!=target:
        if current in seen:raise ValueError('Feature extension ancestry contains a cycle')
        seen.add(current);proof=bound.json(current)
        parent=proof.get('config',{}).get('base_cache');base_proof=proof.get('config',{}).get('base_proof')
        if (proof.get('status')!='complete' or not isinstance(parent,str) or parent!=base_proof or
                proof.get('verification',{}).get('expert_sha256')!=cache.proof['verification']['expert_sha256']):
            raise ValueError('Every feature ancestor must preserve its completed parent and expert')
        previous=Path(parent).resolve();bound.bind(previous,_binding(proof['inputs'],previous))
        chain.append(dict(feature_manifest=str(current),parent_manifest=str(previous)));current=previous
    validate_cache(cache,saved['expert_feature_depths']);cache.check_unchanged();original.check_unchanged()
    bound.unchanged()
    return cache,chain


def corpus_source(data,bound):
    proof=bound.json(data/'manifest.json');verified=proof.get('verification',{})
    required=['unchanged_source_row_bytes_retained','original_validation_row_bytes_preserved',
        'upstream_complete_native_history_proofs_reused','all_supervised_positions_and_mirrors_isolated',
        'no_new_policy_validation_target_enters_other_protected_positions']
    if (proof.get('kind')!=CORPUS_KIND or proof.get('status')!='complete' or
            proof.get('config',{}).get('profile')!=CORPUS_PROFILE or
            any(verified.get(name) is not True for name in required) or verified.get('test_rows')!=0 or
            verified.get('final_independent_test_rows_parsed_or_tokenized')!=0):
        raise ValueError('Use the actual completed seven-stage isolated corpus')
    for name,identity in proof['outputs'].items():bound.bind(name,identity)
    if (data/'test.jsonl').stat().st_size:raise ValueError('Auxiliary corpus must not contain final-test answers')
    counts={split:verified.get(split+'_rows_by_stage',{}) for split in ['train','validation']}
    if any(set(values)!=set(ALL_STAGES) or any(type(n) is not int or n<1 for n in values.values()) for values in counts.values()):
        raise ValueError('Every auxiliary stage requires actual positive train and validation rows')
    return proof,counts


def encoded_chunk(items):
    lengths=token_chunk([row for row,pointer in items])
    if len(lengths)!=len(items):raise ValueError('Native token worker returned incomplete coverage')
    return [dict(observed,stage=row['stage'],feature_key=row['feature_key'],source_row_sha256=pointer['raw_line_sha256'])
            for observed,(row,pointer) in zip(lengths,items)]


def prepare(data,source,parent_completion,features,pinned_base_model_manifest,output,*,max_tokens=2048,workers=8,resume=False):
    if type(max_tokens) is not int or max_tokens<1 or type(workers) is not int or not 1<=workers<=32:
        raise ValueError('Use positive native token limits and1..32 CPU workers')
    data,source,parent_completion,features,pinned_base_model_manifest,output=map(lambda p:Path(p).resolve(),
        [data,source,parent_completion,features,pinned_base_model_manifest,output])
    logical=dict(profile=PROFILE,data=str(data),source=str(source),parent_completion=str(parent_completion),
        features=str(features),pinned_base_model_manifest=str(pinned_base_model_manifest),max_tokens=max_tokens,closure=_closure())
    if any(output==path or output in path.parents for path in [data,source,parent_completion,features,pinned_base_model_manifest]):
        raise ValueError('Preflight output cannot replace an input or its parent')
    if output.exists():
        if not resume:raise FileExistsError('Preserve existing preflight; use completed --resume or a fresh output')
        return _reopen(logical,output,workers)
    if resume:raise FileNotFoundError('Completed preflight is required before --resume')
    return _build(logical,output,workers)


def _reopen(logical,output,workers):
    bound=BoundInputs();proof=bound.json(output/'manifest.json')
    if (proof.get('status')!='complete' or proof.get('kind')!=KIND or proof.get('config',{}).get('logical')!=logical):
        raise ValueError('Reopen requires the identical complete auxiliary input/code contract')
    for section in ['inputs','outputs']:
        for name,identity in proof[section].items():bound.bind(name,identity)
    with TemporaryDirectory(prefix='auxiliary-token-reopen-',dir=output.parent) as temporary:
        rebuilt=Path(temporary)/'preflight';counts=_build(logical,rebuilt,workers)
        if counts!=proof['verification'] or any(
                dict(sha256=digest(rebuilt/name),bytes=(rebuilt/name).stat().st_size)!=proof['outputs'][str(output/name)]
                for name in FILES[:3]):
            raise ValueError('Completed auxiliary preflight differs from full native re-encoding')
    bound.unchanged();return counts


def _build(logical,output,workers):
    bound=BoundInputs();data=Path(logical['data']);source=Path(logical['source'])
    corpus,expected=corpus_source(data,bound)
    saved,parent=completed_parent(source,logical['parent_completion'],bound)
    cache,chain=inherited_cache(logical['features'],saved,bound)
    reference_path=Path(logical['pinned_base_model_manifest'])
    reference=dict(path=str(reference_path),sha256=bound.bind(reference_path)['sha256'])
    assets,_=pinned_decoder_assets(saved,dict(pinned_base_model_manifest=reference),dict(inputs={}),bound)
    keys=set(cache.keys);seen=set();train_histories=set();observed=Counter();tokens=Counter();supervised=Counter();maximum=Counter()
    output.mkdir(parents=True)
    atomic_json(output/'contract.json',logical)
    def rows():
        for split in ['train','validation']:
            for pointer,raw,row in _lines(data/(split+'.jsonl')):
                _basic(row,split,ALL_STAGES)
                if row['id'] in seen:raise ValueError('Auxiliary token IDs repeat within or across splits')
                if row['feature_key'] not in keys:raise ValueError('A complete auxiliary history has no expert feature')
                seen.add(row['id'])
                if split=='train':train_histories.add(row['feature_key'])
                yield row,pointer
    with (output/'token-lengths.jsonl').open('x') as stream:
        def write(items):
            for item in items:
                split,stage=item['split'],item['stage'];key=split+'/'+stage
                if (type(item['tokens']) is not int or not 1<=item['tokens']<=logical['max_tokens'] or
                        type(item['supervised_tokens']) is not int or not 1<=item['supervised_tokens']<=item['tokens'] or
                        item['prompt_tokens']+item['supervised_tokens']!=item['tokens']):
                    raise ValueError('Actual native token lengths or supervision differ')
                observed[key]+=1;tokens[key]+=item['tokens'];supervised[key]+=item['supervised_tokens'];maximum[key]=max(maximum[key],item['tokens'])
                stream.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
        if workers==1:
            token_initializer(saved['model_path'],logical['max_tokens'])
            chunk=[]
            for item in rows():
                chunk.append(item)
                if len(chunk)==128:write(encoded_chunk(chunk));chunk=[]
            if chunk:write(encoded_chunk(chunk))
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                    initializer=token_initializer,initargs=(saved['model_path'],logical['max_tokens'])) as pool:
                for result in bounded_results(pool,encoded_chunk,rows(),workers,128):write(result)
    by_split={split:{stage:observed[split+'/'+stage] for stage in ALL_STAGES} for split in ['train','validation']}
    if by_split!=expected or len(train_histories)!=corpus['verification']['training_unique_history_contexts']:
        raise ValueError('Full auxiliary token/history coverage differs from completed corpus counts')
    counts=dict(rows_by_split_and_stage=by_split,rows_by_split={s:sum(c.values()) for s,c in by_split.items()},
        supervised_tokens_by_split_and_stage={s:{t:supervised[s+'/'+t] for t in ALL_STAGES} for s in by_split},
        total_tokens_by_split_and_stage={s:{t:tokens[s+'/'+t] for t in ALL_STAGES} for s in by_split},
        maximum_tokens_by_split_and_stage={s:{t:maximum[s+'/'+t] for t in ALL_STAGES} for s in by_split},
        training_unique_history_contexts=len(train_histories),feature_history_contexts=len(cache.keys),feature_ancestry=chain,
        token_limit=logical['max_tokens'],all_actual_corpus_rows_tokenized=True,
        all_train_validation_supervision_fits_without_truncation=True,all_corpus_feature_keys_present=True,
        full_decoder_parent_completion_verified=True,original_parent_feature_keys_shards_and_expert_preserved=True,
        upstream_native_history_and_isolation_proofs_reused=True,
        independent_test_rows_parsed_or_tokenized=0,independent_test_used_for_training_or_selection=False,
        student_teacher_or_engine_queries=0,new_SGD_updates=0,policy_or_coaching_gain_proven=False)
    atomic_json(output/'counts.json',counts);cache.check_unchanged();bound.unchanged()
    if _closure()!=logical['closure']:raise ValueError('Auxiliary preflight source changed during encoding')
    inputs=dict(bound.artifacts)
    for name in FILES[:3]:bound.bind(output/name)
    write_verified_input_receipt(output/'verified-inputs.json',
        {name:dict(value,signature=list(bound.signatures[name])) for name,value in bound.artifacts.items()},
        origin=dict(profile='producer_hashed_or_prior_verified_receipt_bound_inputs',execution=input_digest_execution(),
                    source_manifest=str(data/'manifest.json'),new_model_or_engine_queries=0))
    proof=manifest(KIND,dict(logical=logical,execution=dict(workers=workers)),(),[output/name for name in FILES],counts)
    proof['inputs']=inputs;proof['pinned_base_model_manifest']=reference
    proof['cache_header_only_input']=dict(path=logical['features'],**inputs[logical['features']])
    proof['runtime_execution']=input_digest_execution();bound.unchanged();atomic_json(output/'manifest.json',proof)
    return counts


def compile_recipe(token,template,output):
    """Compile full finite source coverage with author-style answer-token loss."""
    result=token.get('verification',{});logical=token.get('config',{}).get('logical',{})
    required=['all_actual_corpus_rows_tokenized','all_train_validation_supervision_fits_without_truncation',
        'all_corpus_feature_keys_present','full_decoder_parent_completion_verified','original_parent_feature_keys_shards_and_expert_preserved']
    if (token.get('status')!='complete' or token.get('kind')!=KIND or logical.get('profile')!=PROFILE or
            any(result.get(k) is not True for k in required) or result.get('independent_test_rows_parsed_or_tokenized')!=0):
        raise ValueError('Require the completed actual auxiliary token/cache/parent proof')
    source_counts=result['rows_by_split_and_stage']['train'];validation=result['rows_by_split_and_stage']['validation']
    if any(set(c)!=set(ALL_STAGES) or any(type(n) is not int or n<1 for n in c.values()) for c in [source_counts,validation]):
        raise ValueError('Formal auxiliary training needs actual seven-stage coverage')
    if (template.get('decoder_training')!='full' or template.get('trainable_parameter_dtype')!='float32' or
            template.get('ddp_world_size')!=4 or template.get('batch_size')!=32 or template.get('micro_batch_size')!=1 or
            template.get('loss_normalization')!='token' or template.get('training_data_profile')!=FINITE_PROFILE or
            template.get('source_coverage_profile')!=FULL_SOURCE_PROFILE):
        raise ValueError('Auxiliary SFT requires explicit four-worker full-source token-normalized training')
    passes=template.get('max_finite_passes')
    if type(passes) is not int or not 1<=passes<=4:raise ValueError('Declare one to four complete finite passes')
    total=sum(source_counts.values());mixture={s:source_counts[s]/total for s in ALL_STAGES}
    selected=mixture_counts(source_counts,mixture)
    if selected['selected_rows_by_stage']!=source_counts:raise ValueError('Every actual auxiliary source row must be selected')
    per_pass=math.ceil(total/template['batch_size'])
    recipe=dict(template,data_path=logical['data'],feature_path=logical['features'],stages=list(ALL_STAGES),mixture=mixture,
        replay_data_paths=[],validation_stages=list(ALL_STAGES),validation_examples=sum(validation.values()),
        validation_profile=VALIDATION_PROFILE,validation_rows_by_stage=validation,require_clean_foundation_handoff=False,
        epochs=passes,max_steps=per_pass*passes,min_steps=per_pass,max_tokens=result['token_limit'],
        pinned_base_model_manifest=token['pinned_base_model_manifest'])
    recipe['row_index_paths']={split:str(Path(output)/(split+'-index')) for split in ['train','validation']}
    source=Path(logical['source']);saved=checked_parent(source,False)
    config=prepare_config(saved,recipe,source,Path(output)/'sft')
    if config['steps']!=per_pass*passes:raise ValueError('Native SFT budget differs from all-row finite compilation')
    budget=dict(training_question_rows=total,training_unique_history_contexts=result['training_unique_history_contexts'],
        training_rows_by_stage=source_counts,validation_rows_by_stage=validation,steps_per_pass=per_pass,
        max_finite_passes=passes,max_steps=config['steps'],minimum_steps=config['min_steps'],
        distributed_tail_repeated_rows_per_pass=per_pass*template['batch_size']-total,
        all_actual_rows_selected_once_before_tail_padding=True,loss_normalization='token',
        validation_metric='token_nll_over_all_seven_stage_validation_rows',new_independent_validation_contexts=0,
        auxiliary_supervision_is_new_on_policy_search_round=False,actual_SGD_started=False)
    return recipe,config,budget


def validation_coverage(rows,selected_ids,expected):
    """Require every original seven-stage validation ID exactly once."""
    if set(expected)!=set(ALL_STAGES) or any(type(n) is not int or n<1 for n in expected.values()):
        raise ValueError('Complete seven-stage validation needs exact positive source counts')
    counts=Counter();identities=set()
    for row in rows.iter_metadata() if hasattr(rows,'iter_metadata') else rows:
        identity,stage=row.get('id'),row.get('stage')
        if (not isinstance(identity,str) or not identity or identity in identities or stage not in ALL_STAGES or
                row.get('split','validation')!='validation'):
            raise ValueError('Validation metadata changed original IDs, stages or protected ownership')
        identities.add(identity);counts[stage]+=1
    if (counts!=expected or not isinstance(selected_ids,list) or len(selected_ids)!=len(identities) or
            any(not isinstance(value,str) for value in selected_ids) or set(selected_ids)!=identities):
        raise ValueError('Native validation selection must cover every seven-stage ID once')
    return dict(profile=VALIDATION_PROFILE,rows_by_stage=dict(counts),validation_rows=len(identities),
        selected_ids_sha256=hashlib.sha256(json.dumps(selected_ids,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
        every_original_validation_id_covered_once=True,independent_test_used=False,
        metric='token_nll_over_complete_seven_stage_validation')


def auxiliary_parent(source,token,recipe,bound):
    """Admit this typed CPU proof to a fresh full-decoder GPU preflight."""
    logical=token.get('config',{}).get('logical',{});result=token.get('verification',{})
    required=['all_actual_corpus_rows_tokenized','all_train_validation_supervision_fits_without_truncation',
        'all_corpus_feature_keys_present','full_decoder_parent_completion_verified',
        'original_parent_feature_keys_shards_and_expert_preserved']
    if (token.get('status')!='complete' or token.get('kind')!=KIND or logical.get('profile')!=PROFILE or
            logical.get('source')!=str(source) or logical.get('data')!=recipe.get('data_path') or
            logical.get('features')!=recipe.get('feature_path') or
            recipe.get('require_clean_foundation_handoff') is not False or recipe.get('loss_normalization')!='token' or
            recipe.get('stages')!=list(ALL_STAGES) or recipe.get('validation_stages')!=list(ALL_STAGES) or
            recipe.get('validation_profile')!=VALIDATION_PROFILE or
            recipe.get('validation_rows_by_stage')!=result.get('rows_by_split_and_stage',{}).get('validation') or
            recipe.get('validation_examples')!=sum(result.get('rows_by_split_and_stage',{}).get('validation',{}).values()) or
            any(result.get(k) is not True for k in required) or result.get('independent_test_rows_parsed_or_tokenized')!=0):
        raise ValueError('Auxiliary GPU preflight requires its exact typed parent, corpus and token proof')
    return completed_parent(source,logical['parent_completion'],bound)[0]


def auxiliary_probe_data(recipe,token,output,*,train_examples=32):
    """Cover every stage and its longest row in bounded GPU capacity data."""
    proof=token.get('verification',{})
    if token.get('kind')!=KIND or token.get('status')!='complete':
        raise ValueError('Only the actual auxiliary token proof can select this GPU probe')
    observed={split:[] for split in ['train','validation']};identities=set()
    source=next(Path(p) for p in token['outputs'] if Path(p).name=='token-lengths.jsonl')
    for _,_,item in _lines(source):
        if (item.get('split') not in observed or item.get('stage') not in ALL_STAGES or
                not isinstance(item.get('id'),str) or item['id'] in identities):
            raise ValueError('Auxiliary probe token metadata changed its stage, split or identity')
        identities.add(item['id']);observed[item['split']].append(item)
    result={};output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for split,limit in [('train',train_examples),('validation',12)]:
        rows=observed[split];counts=Counter(row['stage'] for row in rows)
        if counts!=proof['rows_by_split_and_stage'][split] or limit<len(ALL_STAGES):
            raise ValueError('Auxiliary probe must observe every original row and retain all stages')
        ordered=sorted(rows,key=lambda row:(-row['tokens'],row['id']));chosen={}
        for stage in ALL_STAGES:
            first=next(row for row in ordered if row['stage']==stage);chosen[first['id']]=first
        for item in ordered:
            if len(chosen)==limit:break
            chosen.setdefault(item['id'],item)
        if len(chosen)!=limit:raise ValueError('The bounded auxiliary probe needs32 train and12 validation rows')
        selected=sorted(chosen.values(),key=lambda row:(-row['tokens'],row['id']));raws={}
        for pointer,raw,row in _lines(Path(recipe['data_path'])/(split+'.jsonl')):
            if row['id'] not in chosen:continue
            item=chosen[row['id']]
            if (row['split']!=split or row['stage']!=item['stage'] or row['feature_key']!=item['feature_key'] or
                    pointer['raw_line_sha256']!=item['source_row_sha256'] or row['id'] in raws):
                raise ValueError('Auxiliary GPU probe row differs from its exact tokenized source bytes')
            raws[row['id']]=raw
        if set(raws)!=set(chosen):raise ValueError('Auxiliary GPU probe is missing original source rows')
        with (output/(split+'.jsonl')).open('xb') as stream:
            for item in selected:stream.write(raws[item['id']])
        result[split]=dict(ids=[row['id'] for row in selected],maximum_tokens=selected[0]['tokens'],
            rows_by_stage=dict(Counter(row['stage'] for row in selected)),
            longest_row_from_every_stage_retained=True,actual_source_row_bytes_preserved=True,
            bounded_probe_is_full_training_coverage=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['data','source','parent-completion','features','pinned-base-model-manifest','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--max-tokens',type=int,default=2048);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--resume',action='store_true')
    with input_session_from_environment():print(json.dumps(prepare(**vars(parser.parse_args())),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
