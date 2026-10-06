from pathlib import Path
import subprocess
import json

ROOT = Path(__file__).resolve().parents[1]
build = ROOT / "build/native-reference"
build.mkdir(parents=True, exist_ok=True)
abseil = ROOT / "vendor/abseil-cpp"
if not Path("/usr/include/absl/cleanup/cleanup.h").exists() and not abseil.exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/abseil/abseil-cpp.git", str(abseil)], check=True)
subprocess.run(["python3", str(ROOT / "vendor/px0/scripts/compile_proto.py"),
                str(ROOT / "vendor/px0/proto/net.proto"),
                f"--proto_path={ROOT / 'vendor/px0'}", f"--cpp_out={build}"], check=True)
# Px0's board templates are defined in board.cc; use one translation unit so
# their official definitions are visible to position.cc without modifying them.
combined = build / "combined.cc"
parts = [ROOT / "vendor/px0/src" / source for source in [
    "chess/board.cc", "chess/position.cc", "neural/encoder.cc",
]] + [ROOT / "scripts/native_encoder.cc"]
combined.write_text("\n".join(f'#include "{path}"' for path in parts) + "\n")
sources = [combined] + [ROOT / "vendor/px0/src" / source for source in [
    "utils/logging.cc", "utils/protomessage.cc",
]]
command = ["g++", "-std=c++20", "-O2", "-DNDEBUG", "-DNO_PEXT", "-I" + str(ROOT / "vendor/px0/src"),
           "-I" + str(build), "-I" + str(abseil), "-pthread"] + [str(s) for s in sources] + ["-o", str(build / "encoder")]
subprocess.run(command, check=True)
print(json.dumps({"native_reference_built": True, "source": "unmodified official Px0 encoder"}))
