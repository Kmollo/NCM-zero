#!/bin/bash
# Progress of the Path A overnight run (code_scale/pipeline_path_a.sh).
#   bash code_scale/progress_a.sh          one snapshot
#   bash code_scale/progress_a.sh --watch  refresh every 30 s (Ctrl+C to quit)
cd "$(dirname "$0")/.."
show() {
  .venv/Scripts/python.exe - <<'EOF'
import json, os, time
R = "runs/code-scale-v1"
def rows(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return []
def load(path):
    try:
        return json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
def bar(r, total):
    step = r[-1]["step"]
    pct = 100 * (step + 1) / total
    eta = ""
    if step > 0:
        left = (total - step) * r[-1]["elapsed_s"] / step
        eta = f"  ~{int(left // 60)} min left (done ~{time.strftime('%I:%M %p', time.localtime(time.time() + left))})"
    return f"[{'#' * int(pct / 5)}{'.' * (20 - int(pct / 5))}] {pct:5.1f}%  step {step}/{total}{eta}"

print(time.strftime("%I:%M %p"), "\n")
for i, w in enumerate(("05", "01"), 1):
    c = f"codec-pp{w}"
    print(f"{i}. Tokenizer {c} (plan weight 0.{w[1]})")
    r = rows(f"{R}/{c}/log.jsonl")
    if not r:
        print("   not started\n"); continue
    print("  ", bar(r, 5000))
    last = r[-1]
    if "plan_top1" in last:
        print(f"   plan guesser picks the right chunk: {100*last['plan_top1']:.0f}% (chance 3%)")
    for e in [e for e in r if "token_accuracy" in e]:
        print(f"     step {e['step']:>4}: {100*e['token_accuracy']:5.1f}% tokens, {100*e['chunk_exact']:5.1f}% exact, codes used L0-2 {e.get('codes_used_l0_l2')}")
    s = load(f"{R}/{c}/summary.json")
    if s:
        v = s["val_reconstruction"]
        ok = v["token_accuracy"] >= 0.99 and v["chunk_exact"] >= 0.50
        print(f"   final: {100*v['token_accuracy']:.2f}% tokens, {100*v['chunk_exact']:.1f}% exact -> reconstruction {'PASS' if ok else 'FAIL'}")
    p = load(f"{R}/{c}/plan-probe.json")
    if p and p.get("A2_2_probe_bits_per_code"):
        l0 = p["A2_2_probe_bits_per_code"][0]
        print(f"   level-0 guessability: {l0['probe_mlp_gain']:.2f} bits (need >= 1.0; old tokenizer 0.13) -> {'PASS' if l0['probe_mlp_gain'] >= 1.0 else 'FAIL'}")
    if p and p.get("A2_3_stability"):
        print(f"   level-0 codes changed by a 1-token shift: {100*p['A2_3_stability'][0]['changed_by_one_token_shift']:.0f}% (old 100%, target <= 50%)")
    print()
d = load(f"{R}/path-a-decision.json")
print("3. Decision:", d["chosen"] if d else "waiting")
if d and d["chosen"] != "NONE":
    n = "ncm-" + d["chosen"][6:]
    e = "eval-" + d["chosen"][6:]
    r = rows(f"{R}/{n}/log.jsonl")
    print(f"\n4. NCM training ({n})")
    print("   " + (bar(r, 1800) + f"  loss {r[-1]['loss']:.3f}" if r else "not started"))
    s6 = load(f"{R}/{e}/step6.json")
    print("\n5. Step-6 test vs GPT")
    if s6 and "continuation" in s6:
        c = s6["continuation"]
        print(f"   similarity: NCM {c['ncm_sampled']['char_similarity']:.3f} vs GPT {c['ar_sampled']['char_similarity']:.3f}")
    if s6 and "gate_quality" in s6:
        print("  ", s6["gate_quality"])
    sp = load(f"{R}/{e}/speed.json")
    if sp:
        print(f"   speed vs GPT: batch 1 {sp.get('ncm_speedup_b1')}x, batch 32 {sp.get('ncm_speedup_b32')}x")
log = open(f"{R}/pipeline-path-a.log").read() if os.path.exists(f"{R}/pipeline-path-a.log") else ""
print("\nLog:")
print("\n".join("   " + l for l in log.strip().splitlines()[-6:]) if log else "   not started (starts 10:30 PM)")
if "PATH_A_DONE" in log:
    print("\nALL DONE")
EOF
}
if [ "$1" = "--watch" ]; then
  while true; do clear; show; sleep 30; done
else
  show
fi
