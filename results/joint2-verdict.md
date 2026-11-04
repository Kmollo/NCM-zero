STOP: no lambda kept the rebuild and cut bits by 5% vs lambda 0

Rule (fixed before results): CONTINUE if the bits rule passes (extra bits over GPT <= 611.7, levels 0-4 drop >= 131.1, tokenizer gate held) or step-6 quality is PROGRESS/PASS; otherwise STOP.

## Pre-test (300 joint steps from codec-l7 + ncm-l7, lambdas 0, 0.3, 1.0; joint generator's bits on held-out chunks)
- lambda 0: 99.10% tokens, 65.9% exact chunks, 1345.8 bits per chunk (per level 46, 110, 156, 251, 312, 149, 321); codes changed per level [0.6233, 0.3807, 0.3094, 0.1569, 0.3295, 0.1518, 0.6101]
- lambda 0.3: 99.08% tokens, 65.6% exact chunks, 1320.5 bits per chunk (per level 47, 112, 154, 252, 304, 148, 304); codes changed per level [0.616, 0.3993, 0.3431, 0.1863, 0.3551, 0.1625, 0.6171]
- lambda 1.0: 98.33% tokens, 43.5% exact chunks, 1420.6 bits per chunk (per level 50, 108, 147, 231, 341, 198, 346); codes changed per level [0.7, 0.6991, 0.6473, 0.4078, 0.5554, 0.3222, 0.7556]
- Pick (>= 99.5% tokens, >= 70% exact, bits <= 95% of lambda 0): none qualifies
