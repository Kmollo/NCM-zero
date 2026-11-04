#!/bin/bash
# Idea 1b night (docs/ncm-next-plan.md, "Idea 1b"): upstream lm1b lookup encoder AND its 4-block tokenizer decoder.
# Needs no internet and no Claude. codec -> gate -> NCM -> bits per level -> step-6 test -> scoring -> verdict.
# Progress: tail runs/code-scale-v1/pipeline-idea-1b.log
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
R=runs/code-scale-v1
L=$R/logs
LOG=$R/pipeline-idea-1b.log
mkdir -p $L
export CODE_SCALE_LEVELS=4,8,16,32,48,64,64 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CODE_SCALE_ATTN=block_diagonal
export CODE_SCALE_ENC=lookup-group4 CODE_SCALE_DEC=4 IDEA1_TAG=4
unset CODE_SCALE_CODEBOOKS CODE_SCALE_REVEAL
C=codec-lk4
N=ncm-lk4
E=eval-lk4
note() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $LOG; }
finish() {
  note "IDEA_1B_DONE"
  $PY -m code_scale.decide_1 > $L/decide-1b.log 2>&1 || note "decide FAILED (see $L/decide-1b.log)"
  note "verdict: $(head -1 $R/idea-1b-verdict.md 2>/dev/null)"
  exit 0
}

note "START"
if [ ! -f $R/$C/summary.json ]; then
  if [ -d $R/$C ]; then mv $R/$C $R/$C-partial-$(date +%H%M); note "$C: moved an unfinished folder aside"; fi
  note "$C: tokenizer training (lookup-group4 encoder, 4-block decoder), 5000 steps"
  $PY -m code_scale.train codec --steps 5000 --restart-fix --out $C > $L/$C.log 2>&1 || note "$C: training FAILED"
fi
if ! $PY -c "from code_scale.train import require_codec_quality as q; q('$R/$C/summary.json')" 2>/dev/null; then
  note "STOPPED: $C failed the reconstruction gate"; finish
fi
note "$C: gate passed"

export CODE_SCALE_CODEC=$C CODE_SCALE_NCM=$N CODE_SCALE_EVAL=$E
if [ -d $R/$N ] && [ ! -f $R/$N/summary.json ]; then
  mv $R/$N $R/$N-partial-$(date +%H%M); note "$N: moved an unfinished folder aside"
fi
if [ ! -f $R/$N/summary.json ]; then
  note "$N: NCM training on $C, 1800 steps"
  $PY -m code_scale.train ncm --steps 1800 --out $N > $L/$N.log 2>&1 || note "$N: training FAILED (see $L/$N.log)"
fi
if ! $PY -m code_scale.check_ncm_1 $R/$N/summary.json $C lookup-group4 4 2>/dev/null; then
  note "STOPPED: $N missing or trained with other settings"; finish
fi
note "$N: settings OK; bits per level"
$PY -m code_scale.level_bits > $L/level-bits-$N.log 2>&1 || note "$N: bits FAILED (see $L/level-bits-$N.log)"
$PY -m code_scale.decide_1 > /dev/null 2>> $L/decide-1b.log
note "$E: step-6 test"
$PY -m code_scale.step6 > $L/step6-$E.log 2>&1 || note "$E: step6 FAILED (see $L/step6-$E.log)"
note "$E: scoring"
for s in easy humaneval; do
  k=1,10; [ $s = humaneval ] && k=1,5
  for m in ncm ar; do
    $PY -m zero_baseline.code_bench score --suite $s --samples $R/$E/$m-$s-samples.jsonl \
      --out $R/$E/$m-$s-score.json --k $k --workers 6 > $L/score-$E-$m-$s.log 2>&1 || note "$E: score $m $s FAILED"
  done
done
$PY -m code_scale.rescore_syntax $E > $L/rescore-$E.log 2>&1 || note "$E: rescore FAILED"
finish
