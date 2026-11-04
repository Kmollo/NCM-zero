STOP: no lookup tokenizer passed the reconstruction gate

Rule (fixed before results): CONTINUE if the bits rule passes (extra bits over GPT <= 70% of ncm-l7's, with at least half of the required drop in levels 0-4, tokenizer gate held) or step-6 quality is PROGRESS/PASS; otherwise STOP.

## Pre-test (1,000 steps; codec-l7 had 94.0% tokens at step 1,000; qualify >= 85%)
- lookup-group4: 91.7% tokens, 4.3% exact chunks (step 999)
- lookup-layer: 91.9% tokens, 5.5% exact chunks (step 999)
