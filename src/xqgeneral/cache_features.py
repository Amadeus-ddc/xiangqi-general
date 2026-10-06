from pathlib import Path
import argparse
import hashlib
import json
import time
import torch
from .expert import FrozenPx0, FEATURE_DEPTHS, encode_history
from .evidence import atomic_json, history_key, load_jsonl, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/px0-latest.pb.gz")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="data/features.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Feature cache already exists; select a new output")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.time()
    expert = FrozenPx0(args.weights).cuda()
    inputs = [Path(args.data) / f"{s}.jsonl" for s in ["train", "validation", "test"]
              if (Path(args.data) / f"{s}.jsonl").exists()]
    rows = [row for path in inputs for row in load_jsonl(path)]
    unique = {}
    for record in rows:
        # Include every history FEN: roots with different histories are not interchangeable.
        key = history_key(record["history"])
        record["feature_key"] = key
        unique.setdefault(key, record["history"])
    keys = list(unique)
    chunks = [[] for _ in FEATURE_DEPTHS]
    wdl_chunks = []
    for offset in range(0, len(keys), args.batch_size):
        batch = keys[offset:offset + args.batch_size]
        planes = torch.stack([encode_history(unique[k]) for k in batch]).cuda()
        features, wdl = expert(planes)
        for dst, value in zip(chunks, features):
            assert torch.isfinite(value).all()
            dst.append(value.cpu().half())
        wdl_chunks.append(wdl.cpu())
    tensors = [torch.cat(level) for level in chunks]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"keys": keys, "features": tensors, "wdl": torch.cat(wdl_chunks),
                "depths": FEATURE_DEPTHS}, args.output)
    for split in [p.stem for p in inputs]:
        (Path(args.data) / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                     for r in rows if r["split"] == split))
    summary = {"roots": len(keys), "feature_shapes": [list(t.shape) for t in tensors],
               "depths": FEATURE_DEPTHS, "expert_dim": expert.dim, "expert_layers": expert.depth,
               "stored_expert_weight_elements": sum(b.numel() for b in expert.buffers()),
               "expert_trainable_parameters": sum(p.numel() for p in expert.parameters() if p.requires_grad),
               "feature_mean_std": [(float(t.float().mean()), float(t.float().std())) for t in tensors],
               "seconds": time.time() - started,
               "expert_sha256": hashlib.file_digest(Path(args.weights).open("rb"), "sha256").hexdigest()}
    report = args.report or str(Path(args.output).with_suffix(".manifest.json"))
    atomic_json(report, manifest("expert_feature_cache", vars(args), inputs + [args.weights], [args.output], summary))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
