"""Model-specific dependency inspection without constructing a decoder or expert."""
from pathlib import Path

from .evidence import file_signature
from .model_bundle import checkpoint_input_paths


def checkpoint_dependencies(checkpoint,bound,*,decoder_model=None,decoder_manifest=None,
                            execution=None,input_digest_receipt=None):
    """Verify portable inputs or a legacy checkpoint's pinned local dependencies.

    Legacy inference reads its base/tokenizer separately. Its completed native
    input-digest session can certify those dependencies even though the old
    gameplay input table lists only the trained checkpoint and expert.
    """
    path = Path(checkpoint)
    if path.is_dir():
        paths = checkpoint_input_paths(path)
        for asset in paths:
            bound.bind(asset)
        if decoder_model is not None and Path(decoder_model).resolve()!=(path/'decoder').resolve():
            raise ValueError('The explicit decoder differs from the portable checkpoint')
        return {'profile':'portable_inference_directory','complete_model_assets_verified':True,
            'model_or_expert_constructed':False,'assets':{str(p):bound.artifacts[str(p)] for p in paths}}
    if decoder_model is None or decoder_manifest is None or input_digest_receipt is None:
        raise ValueError('Legacy neural gameplay requires explicit decoder, pinned manifest and actual input receipt')
    bound.bind(path)
    import torch
    saved = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    config = dict(saved['config'])
    del saved
    reference = config.get('pinned_base_model_manifest')
    if (Path(config['model_path']).resolve()!=Path(decoder_model).resolve() or
            not isinstance(reference,dict) or set(reference)!={'path','sha256'} or
            Path(reference['path']).resolve()!=Path(decoder_manifest).resolve()):
        raise ValueError('Explicit decoder dependencies differ from the selected checkpoint metadata')
    from .sft_preflight import pinned_decoder_assets
    assets,_ = pinned_decoder_assets(config,config,{'inputs':{}},bound)
    receipt = bound.json(input_digest_receipt)
    from .verified_inputs import KIND
    runtime = (execution or {}).get('input_digest_verification',{})
    receipt_binding = bound.artifacts[str(input_digest_receipt)]
    if (receipt.get('kind')!=KIND or receipt.get('status')!='complete' or
            runtime.get('profile')!=KIND or runtime.get('receipt_sha256')!=receipt_binding['sha256'] or
            runtime.get('all_receipt_file_signatures_checked') is not True or
            runtime.get('receipt_bound_files')!=len(receipt.get('inputs',{}))):
        raise ValueError('Actual neural inference did not attest the supplied complete input-digest session')
    for name,binding in assets.items():
        item = receipt['inputs'].get(str(Path(name).resolve()),{})
        if ({k:item.get(k) for k in ['sha256','bytes']}!=binding or
                item.get('signature')!=list(file_signature(name))):
            raise ValueError('Actual neural input receipt did not preserve the pinned decoder dependency')
    bound.unchanged()
    return {'profile':'legacy_checkpoint_pinned_decoder_and_actual_digest_session',
        'complete_model_assets_verified':True,'model_or_expert_constructed':False,
        'assets':assets,'model_revision':config.get('model_revision'),
        'pinned_decoder_manifest_binding':bound.artifacts[reference['path']],
        'actual_input_digest_receipt_binding':receipt_binding,
        'actual_neural_decoder_dependency_file_signatures_preserved':True}
