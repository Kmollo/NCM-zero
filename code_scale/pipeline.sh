#!/bin/bash
# Overnight pipeline: codec -> NCM -> AR -> evaluate -> score. Stops on first failure.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
mkdir -p runs/code-scale-v1/logs
$PY -m code_scale.train codec --steps 5000 > runs/code-scale-v1/logs/codec.log 2>&1
$PY -m code_scale.train ncm --steps 1800 > runs/code-scale-v1/logs/ncm.log 2>&1
$PY -m code_scale.train ar --steps 1800 > runs/code-scale-v1/logs/ar.log 2>&1
$PY -m code_scale.evaluate > runs/code-scale-v1/logs/evaluate.log 2>&1
for m in ncm ar; do
  $PY -m zero_baseline.code_bench score --suite easy --samples runs/code-scale-v1/eval/$m-easy-samples.jsonl --out runs/code-scale-v1/eval/$m-easy-score.json --k 1,10 --workers 6 > runs/code-scale-v1/logs/score-$m-easy.log 2>&1
done
for m in ncm ar; do
  $PY -m zero_baseline.code_bench score --suite humaneval --samples runs/code-scale-v1/eval/$m-humaneval-samples.jsonl --out runs/code-scale-v1/eval/$m-humaneval-score.json --k 1,5 --workers 6 > runs/code-scale-v1/logs/score-$m-humaneval.log 2>&1
done
echo PIPELINE_DONE
