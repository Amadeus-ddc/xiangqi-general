"""Build a small driver around the official Px0 exporter; no upstream edits."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/px0-latest.pb.gz")
    parser.add_argument("--output", default="build/native-reference/px0.onnx")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = root / "build/native-reference"
    build.mkdir(parents=True, exist_ok=True)
    upstream = root / "vendor/px0"
    for proto in ["net", "onnx"]:
        subprocess.run([sys.executable, str(upstream / "scripts/compile_proto.py"),
                        str(upstream / f"proto/{proto}.proto"),
                        f"--proto_path={upstream}", f"--cpp_out={build}"], check=True)
    (build / "build_id.h").write_text('#define BUILD_IDENTIFIER "xiangqi-general-reference"\n')
    sources = [upstream / "src" / name for name in [
        "neural/network_legacy.cc", "utils/weights_adapter.cc", "utils/protomessage.cc", "utils/logging.cc",
        "neural/loader.cc", "utils/commandline.cc",
        "neural/onnx/adapters.cc", "neural/onnx/builder.cc", "neural/onnx/converter.cc",
        "version.cc",
    ]] + [root / "scripts/export_native_onnx.cc"]
    command = ["g++", "-std=c++20", "-O2", "-DNDEBUG", "-DNO_PEXT", "-mavx2", "-mf16c",
               "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
               f"-I{upstream / 'src'}", f"-I{build}", f"-I{root / 'vendor/abseil-cpp'}"]
    command += [str(path) for path in sources] + ["-lz", "-o", str(build / "export-network")]
    subprocess.run(command, check=True)
    subprocess.run([str(build / "export-network"), args.weights, args.output], check=True, cwd=root)
    print(json.dumps({"built": True, "reference": "unchanged Px0 C++ exporter + ONNX CPU backend",
                      "output": args.output}), flush=True)


if __name__ == "__main__":
    main()
