STOP: no lambda kept the rebuild and cut bits by 5% vs lambda 0

Rule (fixed before results): CONTINUE if the bits rule passes (extra bits over GPT <= 611.7, levels 0-4 drop >= 131.1, tokenizer gate held) or step-6 quality is PROGRESS/PASS; otherwise STOP.

## Pre-test (300 joint steps from codec-l7 + ncm-l7, lambdas 0, 0.01, 0.03, 0.1; joint generator's bits on held-out chunks)
- lambda 0: 99.65% tokens, 83.6% exact chunks, 1301.0 bits per chunk (per level 49, 116, 152, 244, 296, 141, 304); codes changed per level [0.5869, 0.3442, 0.2845, 0.1442, 0.303, 0.1382, 0.5747]
- lambda 0.01: 99.65% tokens, 83.8% exact chunks, 1304.2 bits per chunk (per level 49, 115, 152, 244, 295, 141, 307); codes changed per level [0.5837, 0.339, 0.2817, 0.1418, 0.3001, 0.1371, 0.5739]
- lambda 0.03: 99.64% tokens, 83.6% exact chunks, 1305.1 bits per chunk (per level 49, 116, 152, 244, 295, 141, 308); codes changed per level [0.5879, 0.3447, 0.283, 0.1469, 0.3043, 0.1379, 0.5771]
- lambda 0.1: 99.66% tokens, 84.3% exact chunks, 1297.8 bits per chunk (per level 49, 116, 152, 244, 293, 140, 304); codes changed per level [0.5925, 0.347, 0.29, 0.1457, 0.304, 0.1383, 0.5751]
- Pick (>= 99.5% tokens, >= 70% exact, bits <= 95% of lambda 0): none qualifies
