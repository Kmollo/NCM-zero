#!/bin/bash
# Path B run, 2026-10-29 (docs/ncm-next-plan.md, "Path B"). Needs no internet and no Claude.
# Started at 15:00 by the Windows scheduled task "NCM Path B" (via code_scale/keep_awake_run.py).
#   Run 1: NCM with block_causal attention on the existing tokenizer codec-l7   (one change vs ncm-l7)
#   Run 2: tokenizer with small coarse codebooks (codec-sc), then a normal NCM  (one change vs ncm-l7)
# Each run: train -> step-6 test -> speed -> scoring -> bits per chunk. Then decide.py writes the verdict.
# Progress: bash code_scale/progress_b.sh --watch
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
R=runs/code-scale-v1
L=$R/logs
LOG=$R/pipeline-path-b.log
mkdir -p $L
export CODE_SCALE_LEVELS=4,8,16,32,48,64,64 PYTHONUNBUFFERED=1
SMALL=64,128,256,512,2048,2048,2048
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }

# check_meta DIR KEY=VALUE...: the settings saved by train.py must match the plan (guards against
# a repeat of the 2026-10-28 weight bug). Prints the mismatch and returns 1 if any differs.
check_meta() {
  $PY - "$@" <<'EOF'
import json, sys
d, pairs = sys.argv[1], sys.argv[2:]
s = json.load(open(f"runs/code-scale-v1/{d}/summary.json"))
bad = [f"{k}: planned {v}, got {json.dumps(s.get(k))}" for k, v in (p.split("=", 1) for p in pairs)
       if json.dumps(s.get(k), separators=(",", ":")) != v]
print("; ".join(bad) if bad else "settings OK")
sys.exit(1 if bad else 0)
EOF
}

# A folder left by an unfinished attempt is moved aside, never silently reused.
fresh() {
  if [ -d $R/$1 ] && [ ! -f $R/$1/summary.json ]; then
    mv $R/$1 $R/$1-partial-$(date +%H%M); note "$1: moved an unfinished folder aside"
  fi
}

evaluate() {  # NCM_DIR EVAL_DIR
  local N=$1 E=$2
  export CODE_SCALE_NCM=$N CODE_SCALE_EVAL=$E
  note "$E: step-6 test"
  $PY -m code_scale.step6 > $L/step6-$E.log 2>&1 || note "$E: step6 FAILED (see $L/step6-$E.log)"
  note "$E: speed test"
  $PY -m code_scale.speed > $L/speed-$E.log 2>&1 || note "$E: speed FAILED (see $L/speed-$E.log)"
  note "$E: scoring"
  for s in easy humaneval; do
    k=1,10; [ $s = humaneval ] && k=1,5
    for m in ncm ar; do
      $PY -m zero_baseline.code_bench score --suite $s --samples $R/$E/$m-$s-samples.jsonl \
        --out $R/$E/$m-$s-score.json --k $k --workers 6 > $L/score-$E-$m-$s.log 2>&1 || note "$E: score $m $s FAILED"
    done
  done
  $PY -m code_scale.rescore_syntax $E > $L/rescore-$E.log 2>&1 || note "$E: rescore FAILED"
  note "$E: bits per chunk"
  $PY -m code_scale.plan_probe --gain-only > $L/bits-$N.log 2>&1 || note "$N: bits per chunk FAILED"
  $PY -m code_scale.decide > /dev/null 2>> $L/decide.log  # refresh the verdict after each run
}

note "START"

# ---- Run 1: block_causal NCM on codec-l7 ----
N=ncm-bc-l7
export CODE_SCALE_CODEC=codec-l7 CODE_SCALE_ATTN=block_causal
unset CODE_SCALE_CODEBOOKS
fresh $N
if [ -f $R/$N/summary.json ]; then
  note "$N already trained, skipping training"
else
  note "$N: NCM training (block_causal, tokenizer codec-l7), 1800 steps"
  $PY -m code_scale.train ncm --steps 1800 --out $N > $L/$N.log 2>&1 || note "$N: training FAILED (see $L/$N.log)"
fi
if [ -f $R/$N/summary.json ] && M=$(check_meta $N attn_pattern='"block_causal"' codec='"codec-l7"' steps=1800); then
  note "$N: $M"
  evaluate $N eval-bc-l7
else
  note "$N: STOPPED run 1 (${M:-no summary})"
fi

# ---- Run 2: small coarse codebooks, normal NCM ----
C=codec-sc
N=ncm-sc
export CODE_SCALE_CODEBOOKS=$SMALL CODE_SCALE_ATTN=block_diagonal
fresh $C
if [ -f $R/$C/summary.json ]; then
  note "$C already trained, skipping training"
else
  note "$C: tokenizer training (codebooks $SMALL), 5000 steps"
  $PY -m code_scale.train codec --steps 5000 --restart-fix --out $C > $L/$C.log 2>&1 || note "$C: training FAILED (see $L/$C.log)"
fi
if [ -f $R/$C/summary.json ] && M=$(check_meta $C codebooks="[$SMALL]" restart_fix=true plan_weight=0.0 steps=5000); then
  note "$C: $M; reconstruction $($PY -c "import json;v=json.load(open('$R/$C/summary.json'))['val_reconstruction'];print(f\"{100*v['token_accuracy']:.2f}% tokens, {100*v['chunk_exact']:.1f}% exact chunks\")")"
  export CODE_SCALE_CODEC=$C
  fresh $N
  if [ ! -f $R/$N/summary.json ]; then
    note "$N: NCM training (normal attention, tokenizer $C), 1800 steps (refused if the tokenizer failed its gate)"
    $PY -m code_scale.train ncm --steps 1800 --out $N > $L/$N.log 2>&1 || note "$N: training FAILED or refused (see $L/$N.log)"
  fi
  if [ -f $R/$N/summary.json ] && M=$(check_meta $N attn_pattern='"block_diagonal"' codec='"codec-sc"' codebooks="[$SMALL]" steps=1800); then
    note "$N: $M"
    evaluate $N eval-sc
  else
    note "$N: STOPPED run 2 (${M:-no summary})"
  fi
else
  note "$C: STOPPED run 2 (${M:-no summary})"
fi

$PY -m code_scale.decide > $L/decide.log 2>&1 || note "decide FAILED (see $L/decide.log)"
note "verdict: $(head -1 $R/path-b-verdict.md 2>/dev/null)"
note "PATH_B_DONE"
