STOP: the last bet did not reach the progress bar; stop NCM for code at this scale.

Rule (fixed before results, last bet): CONTINUE if PASS (all three ratios >= 0.8) or PROGRESS (similarity >= 0.78 or a with-code syntax ratio >= 0.40); otherwise STOP.

## Path C: positions within every level see each other (3 passes per level)
- Cause measurement: GPT needs 250 bits per chunk; NCM needs 1,191 guessing blind, 1,186 with 3 passes, 1,184 with 8 passes
- 3 passes remove 5 of the 940 extra bits (1%) -> cause: the codes themselves (rule: at least half means blind guessing)
- Category: NO PROGRESS
- Similarity to the real code: NCM 0.236 vs GPT 0.331 = 0.71 of GPT (baseline 0.72; continue needs 0.78)
- Valid Python with code, HumanEval: NCM 0.6% vs GPT 6.0% = 0.10 (baseline 0.10; continue needs 0.40)
- Valid Python with code, easy tasks: NCM 1.1% vs GPT 9.2% = 0.12 (baseline 0.10; continue needs 0.40)
- Bits per chunk with nothing shown (first pass): 1,193 (baseline 1,127)
- Speed vs GPT: 0.59x one at a time, 0.08x 32 at once (21 model calls per chunk with a simple full-forward decoder; reported, does not decide)
