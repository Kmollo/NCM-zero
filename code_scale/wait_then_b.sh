#!/bin/bash
# Wait until 15:00 today, then run the Path B pipeline (started early under keep_awake_run.py
# so the laptop stays awake while waiting).
cd "$(dirname "$0")/.."
echo "$(date '+%Y-%m-%d %H:%M:%S') waiting for 15:00" >> runs/code-scale-v1/pipeline-path-b.log
while [ "$(date +%H%M)" -lt 1500 ]; do sleep 20; done
exec bash code_scale/pipeline_b.sh
