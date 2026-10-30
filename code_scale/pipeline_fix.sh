#!/bin/bash
# Restart-fix rerun: fixed tokenizer -> gate -> NCM -> evaluate (AR checkpoint reused; it doesn't use the tokenizer).
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
L=runs/code-scale-v1/logs
until [ -f runs/code-scale-v1/codec-restart-fix/summary.json ]; do sleep 30; done   # let the matched test finish first
$PY -m code_scale.train codec --steps 5000 --restart-fix --out codec-fix > $L/codec-fix.log 2>&1
export CODE_SCALE_CODEC=codec-fix CODE_SCALE_NCM=ncm-fix CODE_SCALE_EVAL=eval-fix
if ! $PY -m code_scale.train ncm --steps 1800 --out ncm-fix > $L/ncm-fix.log 2>&1; then
  echo "STOPPED: tokenizer gate or NCM training failed (see $L/ncm-fix.log)"; exit 2
fi
$PY -m code_scale.evaluate > $L/evaluate-fix.log 2>&1
for s in easy humaneval; do
  k=1,10; [ $s = humaneval ] && k=1,5
  for m in ncm ar; do
    $PY -m zero_baseline.code_bench score --suite $s --samples runs/code-scale-v1/eval-fix/$m-$s-samples.jsonl --out runs/code-scale-v1/eval-fix/$m-$s-score.json --k $k --workers 6 > $L/score-fix-$m-$s.log 2>&1
  done
done
echo PIPELINE_FIX_DONE
