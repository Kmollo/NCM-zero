#!/bin/bash
# Show progress of the l7 run (tokenizer -> gate -> NCM -> evaluation).
#   bash code_scale/progress.sh          one snapshot
#   bash code_scale/progress.sh --watch  refresh every 30 s (Ctrl+C to quit)
cd "$(dirname "$0")/.."
show() {
  .venv/Scripts/python.exe - <<'EOF'
import json, os, time
R = "runs/code-scale-v1"
def last(path):
    try:
        rows = [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return []
    return rows
def bar(step, total, rows):
    pct = 100 * (step + 1) / total
    fill = int(pct / 5)
    eta = ""
    if rows and rows[-1]["elapsed_s"] > 0 and step > 0:
        rate = rows[-1]["elapsed_s"] / step
        left = (total - step) * rate
        eta = f"  ~{int(left // 60)} min left (done ~{time.strftime('%I:%M %p', time.localtime(time.time() + left))})"
    return f"[{'#' * fill}{'.' * (20 - fill)}] {pct:5.1f}%  step {step}/{total}{eta}"

print(time.strftime("%I:%M %p"), "\n")
codec = last(f"{R}/codec-l7/log.jsonl")
print("1. Tokenizer (codec-l7)")
if not codec:
    print("   not started")
else:
    print("  ", bar(codec[-1]["step"], 5000, codec))
    for r in [r for r in codec if "token_accuracy" in r]:
        print(f"     step {r['step']:>4}: {100*r['token_accuracy']:5.1f}% tokens, {100*r['chunk_exact']:5.1f}% exact chunks")
s = f"{R}/codec-l7/summary.json"
print("\n2. Gate (needs >= 99% tokens and >= 50% exact chunks)")
if os.path.exists(s):
    v = json.load(open(s))["val_reconstruction"]
    ok = v["token_accuracy"] >= 0.99 and v["chunk_exact"] >= 0.50
    print(f"   final: {100*v['token_accuracy']:.1f}% tokens, {100*v['chunk_exact']:.1f}% exact -> {'PASS' if ok else 'FAIL (NCM will not train)'}")
else:
    print("   waiting for tokenizer")
ncm = last(f"{R}/ncm-l7/log.jsonl")
print("\n3. NCM training (ncm-l7)")
print("   " + bar(ncm[-1]["step"], 1800, ncm) + f"  loss {ncm[-1]['loss']:.3f}" if ncm else "   not started")
print("\n4. Evaluation + scoring")
ev = f"{R}/eval-l7"
print("   files so far:", ", ".join(sorted(os.listdir(ev))) if os.path.isdir(ev) else "not started")
log = open("runs/code-scale-v1-pipeline-l7.log").read() if os.path.exists("runs/code-scale-v1-pipeline-l7.log") else ""
if "PIPELINE_L7_DONE" in log:
    print("\nALL DONE")
elif "STOPPED" in log:
    print("\nSTOPPED:", log.strip().splitlines()[-1])
EOF
}
if [ "$1" = "--watch" ]; then
  while true; do clear; show; sleep 30; done
else
  show
fi
