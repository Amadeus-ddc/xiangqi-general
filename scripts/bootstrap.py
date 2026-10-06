"""Download pinned public assets into this project, without changing other environments."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def log(**item):
    print(json.dumps(item, ensure_ascii=False), flush=True)


def download_weight(item):
    import requests
    name, digest_id = item
    url = f"https://github.com/official-pikafish/pxzero-networks/releases/download/px0_client/{digest_id}"
    dest = ROOT / "models" / name
    if not dest.exists():
        partial = dest.with_suffix(dest.suffix + ".partial")
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    handle.write(chunk)
        partial.replace(dest)
    sha = hashlib.file_digest(dest.open("rb"), "sha256").hexdigest()
    record = {"name": name, "url": url, "bytes": dest.stat().st_size, "sha256": sha}
    (ROOT / "models" / f"{name}.manifest.json").write_text(json.dumps(record, indent=2))
    log(weight=record)


def download_qwen():
    from huggingface_hub import snapshot_download
    snapshot_download(
        "Qwen/Qwen3-4B-Instruct-2507",
        revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        local_dir=ROOT / "models" / "qwen3-4b",
        cache_dir=ROOT / "models" / "hf-cache",
        token=False,
        max_workers=3,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE"],
    )
    log(qwen="download_complete")


def main():
    started = time.time()
    for name in ["models", "runs", "src/xqgeneral"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    tasks = [
        ("px0-latest.pb.gz", "5dd1a88c05f7d19f92a53c3c74c5b4cbee6e50d16841596382b22a980fe8c87d"),
        ("px0-large.pb.gz", "003be1c2c041be27b54a71636303276634b8ddbfd526342962f35e45d491c60f"),
    ]
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download_weight, task) for task in tasks]
        futures.append(pool.submit(download_qwen))
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                log(error=errors[-1])
    state = {"complete": not errors, "errors": errors, "seconds": time.time() - started}
    (ROOT / "runs/bootstrap.json").write_text(json.dumps(state, indent=2))
    log(bootstrap=state)


if __name__ == "__main__":
    main()
