"""Download an optional pinned local teacher; never consult stored credentials."""
from pathlib import Path
import json
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'


def main():
    dest = ROOT / 'models/qwen3.8-27b'
    snapshot_download(REPO, revision=REVISION, local_dir=dest,
                      cache_dir=ROOT / 'models/hf-cache', token=False, max_workers=4,
                      allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', 'LICENSE', 'README.md'])
    (dest / 'source.json').write_text(json.dumps({'repository': REPO, 'revision': REVISION,
                                                'purpose': 'local search consolidation; not initial teacher or student'}, indent=2) + '\n')
    print(json.dumps({'teacher_download': 'complete', 'repository': REPO, 'revision': REVISION}), flush=True)


if __name__ == '__main__':
    main()
