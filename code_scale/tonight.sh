#!/bin/bash
# Wait until the HOLD time (23:00 local), then run the l7 layout pipeline.
cd "$(dirname "$0")/.."
until [ "$(date +%H%M)" -ge 2300 ]; do sleep 60; done
bash code_scale/pipeline_layout.sh
