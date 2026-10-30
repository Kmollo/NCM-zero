"""Fetch only the pinned, public NCM implementation. Never resets an existing checkout."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
lock = json.loads((root / "upstream.lock.json").read_text())
dest = root / lock["path"]
if not dest.exists():
    subprocess.run(["git", "clone", lock["repository"], str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--detach", lock["revision"]], check=True)
actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
if actual != lock["revision"]:
    raise SystemExit(f"Existing upstream is {actual}; expected {lock['revision']}. Preserve your changes and select the pinned revision manually.")
print(f"NCM source ready at {dest} ({actual})")
