"""Verify every full-precision teacher shard against the pinned official LFS hash."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi
from safetensors import safe_open
from fetch_teacher import REPO, REVISION, ROOT


def verify():
    dest = ROOT / 'models/qwen3.8-27b'
    source = json.loads((dest / 'source.json').read_text())
    if source.get('repository') != REPO or source.get('revision') != REVISION:
        raise ValueError('Teacher repository identity differs from the pinned contract')
    config = json.loads((dest / 'config.json').read_text())
    if config.get('quantization_config') is not None:
        raise ValueError('A quantized teacher configuration is not authorized')
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError('Official revision identity differs')
    shards = [s for s in info.siblings if s.rfilename.endswith('.safetensors')]
    if not shards:
        raise ValueError('Official full-precision shards are missing')
    proofs, dtypes = {}, {}
    for entry in shards:
        path = dest / entry.rfilename
        if not entry.lfs or path.stat().st_size != entry.lfs.size:
            raise ValueError(f'Incomplete official weight shard: {entry.rfilename}')
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual != entry.lfs.sha256:
            raise ValueError(f'Official weight hash differs: {entry.rfilename}')
        proofs[entry.rfilename] = {'bytes': path.stat().st_size, 'sha256': actual}
        with safe_open(path, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                dtype = handle.get_slice(key).get_dtype()
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
                if dtype not in {'BF16', 'F16', 'F32', 'F64'}:
                    raise ValueError(f'Non-floating parameter in the full teacher: {key}')
        print(json.dumps({'verified_shard': entry.rfilename}), flush=True)
    proof = {'repository': REPO, 'revision': REVISION, 'quantization': 'none',
             'total_weight_bytes': sum(v['bytes'] for v in proofs.values()),
             'all_official_lfs_sha256_matched': True, 'stored_tensor_dtypes': dtypes, 'shards': proofs}
    partial = dest / 'weights.manifest.json.partial'
    partial.write_text(json.dumps(proof, indent=2) + '\n')
    partial.replace(dest / 'weights.manifest.json')
    print(json.dumps({k: v for k, v in proof.items() if k != 'shards'}), flush=True)
    return proof


if __name__ == '__main__':
    verify()
