#!/bin/bash
# Strong-push check (docs/joint-training-plan.md, "Strong-push check"): lambdas 0 / 0.3 / 1.0 at tokenizer lr 5e-4.
# If one qualifies, the main run waits for runs/code-scale-v1/HOLD (19:30). Offline.
cd "$(dirname "$0")/.."
export JT_SET=jt2 JT_LAMS=0,0.3,1.0 CODEC_LR=5e-4 MAIN_WAIT_HOLD=1
exec bash code_scale/pipeline_j.sh
