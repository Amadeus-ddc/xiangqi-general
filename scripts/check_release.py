"""Check the publishable tree without reading account configuration or secrets."""
from pathlib import Path
import re
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    files = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout.split(b"\0")
    failures = []
    for raw in files:
        if not raw:
            continue
        name = raw.decode()
        path = root / name
        if not path.exists():  # Renames may be staged after this local check.
            continue
        if name.split('/')[0] in {"models", "data", "runs", "vendor", ".venv"}:
            failures.append(f"generated asset tracked: {name}")
        if path.stat().st_size > 2 * 1024 * 1024:
            failures.append(f"large file tracked: {name}")
        if path.suffix in {".pt", ".onnx", ".nnue", ".safetensors"}:
            failures.append(f"weight file tracked: {name}")
        content = path.read_text(errors="replace")
        for pattern in [r"gh[pousr]_[A-Za-z0-9]{25,}", r"github_pat_[A-Za-z0-9_]{30,}",
                        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
                        r"https?://[^\s/@]+:[^\s/@]+@"]:
            if re.search(pattern, content):
                failures.append(f"potential credential in {name}")
    required = ["LICENSE", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", "pyproject.toml",
                ".github/workflows/ci.yml", "README.md", "docs/MODEL_CARD.md"]
    failures += [f"missing release file: {name}" for name in required if not (root / name).is_file()]
    if failures:
        raise SystemExit('\n'.join(failures))
    print("Release tree verified: source and notices only; no generated weights or detected credentials")


if __name__ == "__main__":
    main()
