STOP recommended for the quick fixes: neither change moved quality. One bigger idea is untested (positions within a level seeing each other); try it only as a last bet.

Rule (fixed before results): PASS = all three ratios >= 0.8; PROGRESS = similarity >= 0.78 or a with-code syntax ratio >= 0.40; NO PROGRESS = similarity < 0.75 and both syntax ratios < 0.25; otherwise WEAK. Baseline (ncm-l7): similarity 0.72, HumanEval 0.10, easy 0.10, 1,127 bits per chunk.

## Run 1: levels see the levels above them (block_causal)  ->  NO PROGRESS
- Similarity to the real code: NCM 0.235 vs GPT 0.331 = 0.71 of GPT (baseline 0.72, need 0.8)
- Valid Python with code, HumanEval: NCM 0.9% vs GPT 7.6% = 0.11 (baseline 0.10, need 0.8)
- Valid Python with code, easy tasks: NCM 0.8% vs GPT 10.5% = 0.07 (baseline 0.10, need 0.8)
- Bits NCM must predict per chunk: 1,115 (baseline 1,127; lower is easier)
- Speed vs GPT: 2.39x one at a time, 0.24x 32 at once (Run 1 uses a simple, about 2x slower generator; speed does not decide tonight)

## Run 2: smaller plan codebooks (64/128/256/512)  ->  NO PROGRESS
- Tokenizer: 99.6% tokens, 84.5% exact chunks (passes the 99% / 50% gate)
- Similarity to the real code: NCM 0.240 vs GPT 0.331 = 0.73 of GPT (baseline 0.72, need 0.8)
- Valid Python with code, HumanEval: NCM 0.0% vs GPT 7.6% = 0.00 (baseline 0.10, need 0.8)
- Valid Python with code, easy tasks: NCM 1.1% vs GPT 10.5% = 0.10 (baseline 0.10, need 0.8)
- Bits NCM must predict per chunk: 1,008 (baseline 1,127; lower is easier)
- Speed vs GPT: 4.89x one at a time, 0.47x 32 at once
