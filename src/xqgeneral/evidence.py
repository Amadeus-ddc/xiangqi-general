"""Small, portable evidence records; no host identities or environment secrets."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    partial.replace(path)


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


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
    root = Path(__file__).resolve().parents[2]
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    files = sorted((root / "src/xqgeneral").glob("*.py"))
    return {"revision": rev.stdout.strip() if rev.returncode == 0 else None,
            "source_sha256": {str(p.relative_to(root)): digest(p) for p in files}}


def manifest(kind, config, inputs=(), outputs=(), verification=None):
    def artifacts(paths):
        return {str(p): {"sha256": digest(p), "bytes": Path(p).stat().st_size} for p in paths}
    return {"schema_version": 1, "kind": kind, "status": "complete",
            "evidence_state": "reconstructed_baseline", "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": config, "code": code_identity(), "inputs": artifacts(inputs),
            "outputs": artifacts(outputs), "verification": verification or {}}
