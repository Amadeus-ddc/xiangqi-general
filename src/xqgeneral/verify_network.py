"""Compare every expert head and bridge feature against native Px0 ONNX."""
import argparse
import json
from pathlib import Path
import random
import re
import subprocess

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .evidence import atomic_json, digest, load_jsonl, manifest
from .expert import FEATURE_DEPTHS, FrozenPx0, encode_history
from .rules import START_FEN, replay


def policy_mapping(path):
    text = Path(path).read_text()
    body = text.split("kAttnPolicyMap", 1)[1].split("{", 1)[1].split("}", 1)[0]
    values = [int(v) for v in re.findall(r"-?\d+", body)]
    if len(values) != 8100:
        raise ValueError("The pinned Xiangqi policy map must cover 90 x 90 pairs")
    indices = np.empty(2062, dtype=np.int64)
    seen = set()
    for offset, value in enumerate(values):
        if value >= 0:
            if value in seen or value >= 2062:
                raise ValueError("Policy map is not one-to-one")
            indices[value] = offset
            seen.add(value)
    if len(seen) != 2062:
        raise ValueError("Incomplete policy map")
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/px0-latest.pb.gz")
    parser.add_argument("--reference", default="build/native-reference/px0.onnx")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="runs/native_network_verification.json")
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    graph = onnx.load(args.reference)
    names = {output for node in graph.graph.node for output in node.output}
    taps = []
    for depth in FEATURE_DEPTHS:
        name = f"/encoder{depth}/ln2"
        if name not in names:
            choices = [n for n in names if n.startswith(name)]
            if not choices:
                raise ValueError(f"Missing native feature output {name}")
            name = sorted(choices, key=len)[0]
        taps.append(name)
        graph.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [None, 512]))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    positions = {}
    for split in ["train", "validation"]:
        for row in load_jsonl(Path(args.data) / f"{split}.jsonl"):
            positions.setdefault(tuple(row["moves"]), row)
    selected = random.Random(20261006).sample(list(positions.values()), min(args.positions, len(positions)))
    selected += [{"initial_fen": START_FEN, "moves": [], "history": [START_FEN]},
                 {"initial_fen": START_FEN, "moves": ["b0c2", "b9c7", "c2b0", "c7b9"],
                  "history": replay(START_FEN, ["b0c2", "b9c7", "c2b0", "c7b9"])}]
    expert = FrozenPx0(args.weights).to(args.device)
    mapping = policy_mapping("vendor/px0/src/neural/tables/attention_policy_map.h")
    errors = {key: 0.0 for key in ["wdl", "policy", "moves_left", *taps]}
    failures = []
    output_names = [o.name for o in session.get_outputs()]
    for row in selected:
        # Independent native encoder supplies inputs to the independent native-exported graph.
        result = subprocess.run(["build/native-reference/encoder", row["initial_fen"], *row["moves"]],
                                capture_output=True, check=True)
        native_planes = np.frombuffer(result.stdout, dtype=np.float32).reshape(1, 124, 10, 9)
        planes = encode_history(row["history"]).unsqueeze(0)
        np.testing.assert_array_equal(native_planes, planes.numpy())
        native = dict(zip(output_names, session.run(None, {session.get_inputs()[0].name: native_planes})))
        features, heads = expert(planes.to(args.device), all_outputs=True)
        pairs = [("wdl", heads["wdl"], native["/output/wdl"]),
                 ("policy", heads["policy_matrix"].flatten(1)[:, mapping], native["/output/policy"]),
                 ("moves_left", heads["moves_left"], native["/output/mlh"])]
        for name, feature in zip(taps, features):
            if row["history"][-1].split()[1] == "b":
                feature = feature.reshape(1, 10, 9, 512).flip(1)
            pairs.append((name, feature.reshape(-1, 512), native[name].reshape(-1, 512)))
        for name, value, expected in pairs:
            actual = value.cpu().numpy()
            delta = float(np.max(np.abs(actual - expected)))
            errors[name] = max(errors[name], delta)
            atol = 2e-3 if name in ["policy", "moves_left"] else 2e-4
            if not np.allclose(actual, expected, rtol=2e-4, atol=atol):
                failures.append({"head": name, "moves": row["moves"], "max_absolute_error": delta})
    report = {"positions": len(selected), "mismatches": len(failures), "failures": failures,
              "max_absolute_errors": errors, "rtol": 2e-4, "wdl_features_atol": 2e-4,
              "policy_moves_left_atol": 2e-3, "all_policy_logits_compared": 2062,
              "expert_sha256": digest(args.weights), "reference_sha256": digest(args.reference),
              "reference": "pinned official C++ input encoder + unmodified C++ ONNX exporter + ONNX Runtime CPU",
              "scope": "all output heads plus four intermediate feature levels"}
    atomic_json(args.output, report)
    print(json.dumps(report), flush=True)
    if failures:
        raise RuntimeError("Native Px0 output parity failed")
    atomic_json(Path(args.output).with_suffix(".manifest.json"),
                manifest("native_network_parity", vars(args), [args.weights, args.reference], [args.output], report))


if __name__ == "__main__":
    main()
