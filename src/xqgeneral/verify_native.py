from pathlib import Path
import json
import random
import subprocess
import numpy as np
from .expert import encode_history
from .rules import START_FEN, replay


def main():
    unique = {}
    for split in ["train", "validation"]:
        for line in Path(f"data/{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            unique.setdefault(tuple(row["moves"]), row)
    selected = random.Random(20261006).sample(list(unique.values()), 100)
    selected += [{"initial_fen": START_FEN, "moves": [], "history": [START_FEN]},
                 {"initial_fen": START_FEN, "moves": ["b0c2", "b9c7", "c2b0", "c7b9"],
                  "history": replay(START_FEN, ["b0c2", "b9c7", "c2b0", "c7b9"])}]
    failures = []
    for row in selected:
        result = subprocess.run(["build/native-reference/encoder", row["initial_fen"], *row["moves"]],
                                capture_output=True, check=True)
        native = np.frombuffer(result.stdout, dtype=np.float32).reshape(124, 10, 9)
        python = encode_history(row["history"]).numpy()
        delta = np.abs(native - python)
        if delta.max() > 1e-6:
            failures.append({"moves": row["moves"], "mismatched_planes": np.flatnonzero(delta.max(axis=(1,2)) > 1e-6).tolist(),
                             "max_error": float(delta.max())})
    summary = {"positions": len(selected), "mismatches": len(failures), "failures": failures,
               "source": "official Px0 C++ input encoder", "scope": "input planes; not full network output"}
    Path("runs/native_encoder_verification.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    assert not failures, "Python input encoding differs from native Px0"


if __name__ == "__main__":
    main()
