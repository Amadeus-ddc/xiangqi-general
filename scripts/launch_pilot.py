from pathlib import Path
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
required = [ROOT / "models/qwen3-4b" / name for name in [
    "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors", "tokenizer.json", "tokenizer_config.json",
]]
deadline = time.monotonic() + 900
report = ROOT / "runs/native_encoder_verification.json"
while not all(p.is_file() for p in required) or not report.exists():
    if not report.exists():
        active = subprocess.run(["tmux", "has-session", "-t", "xqgeneral-native-reference"],
                                capture_output=True).returncode == 0
        if not active:
            raise RuntimeError("Native encoding verification did not complete; inspect its build log")
    if time.monotonic() > deadline:
        raise TimeoutError("Weights or native verification are incomplete; no training launched")
    time.sleep(10)
assert json.loads(report.read_text())["mismatches"] == 0
memory = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, check=True)
if int(memory.stdout.strip()) > 1000:
    raise RuntimeError("GPU 0 is occupied; do not interfere with another job")
print(json.dumps({"event": "starting_pilot", "gpu": 0}), flush=True)
subprocess.run(["bash", "scripts/pilot.sh"], cwd=ROOT, check=True)
print(json.dumps({"event": "pilot_complete"}), flush=True)
