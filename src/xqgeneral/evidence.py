"""Small, portable evidence records; no host identities or environment secrets."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def file_signature(path):
    """Guard the identity and timestamps of an input already hashed by the caller."""
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    partial.replace(path)


def iter_jsonl(path):
    """Read records in file order without holding the complete JSONL text in memory."""
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path):
    return list(iter_jsonl(path))


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    partial.replace(path)


def history_key(history):
    return hashlib.sha256(json.dumps(history).encode()).hexdigest()


def position_key(fen):
    return " ".join(fen.split()[:2])


def code_identity():
    source = Path(__file__).resolve().parent
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True)
    return {"revision": os.environ.get('XQGENERAL_CODE_REVISION') or (rev.stdout.strip() if rev.returncode == 0 else None),
            "source_sha256": {f'src/xqgeneral/{p.name}': digest(p) for p in sorted(source.glob('*.py'))}}


def manifest(kind, config, inputs=(), outputs=(), verification=None, code=None):
    def artifacts(paths):
        return {str(p): {"sha256": digest(p), "bytes": Path(p).stat().st_size} for p in paths}
    return {"schema_version": 1, "kind": kind, "status": "complete",
            "evidence_state": "reconstructed_baseline", "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": config, "code": code or code_identity(), "inputs": artifacts(inputs),
            "outputs": artifacts(outputs), "verification": verification or {}}
