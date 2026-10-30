#!/bin/bash
# Progress of the Path B run (code_scale/pipeline_b.sh).  --watch refreshes every 30 s.
cd "$(dirname "$0")/.."
show() {
  echo "$(date '+%I:%M %p')"; echo
  for d in ncm-bc-l7:1800 codec-sc:5000 ncm-sc:1800; do
    n=${d%%:*}; t=${d#*:}; f=runs/code-scale-v1/$n/log.jsonl
    if [ -f $f ]; then
      .venv/Scripts/python.exe -c "
import json,time
r=[json.loads(l) for l in open('$f') if l.strip()][-1]; s=r['step']; t=$t
left=(t-s)*r['elapsed_s']/max(s,1)
print(f'$n: step {s}/{t} ({100*(s+1)/t:.0f}%), loss {r[\"loss\"]:.3f}, ~{left/60:.0f} min left')"
    else echo "$n: not started"; fi
  done
  echo; echo "Log:"; tail -8 runs/code-scale-v1/pipeline-path-b.log 2>/dev/null || echo "  not started (starts 3:00 PM)"
  [ -f runs/code-scale-v1/path-b-verdict.md ] && { echo; echo "Verdict so far:"; head -1 runs/code-scale-v1/path-b-verdict.md; }
}
if [ "$1" = "--watch" ]; then while true; do clear; show; sleep 30; done; else show; fi
