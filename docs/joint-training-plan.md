# Joint training plan (fixed 2026-11-03 ~23:40, before any joint-training result)

## The problem it targets
The tokenizer is trained only to rebuild the text. Nothing in its training asks whether its codes can be guessed
from the 192 tokens before the chunk. Result: NCM needs 1,124 bits per 64-token chunk, GPT needs 250; level 0
costs 32.7 bits with the prefix vs 33.2 with no prefix at all. The lookup-encoder runs (ideas 1 and 1b) showed the
encoder's attention is needed to rebuild exact code, so we keep it and change what it is trained for.

## The change
Train the tokenizer and the generator at the same time, starting from the ones we have (codec-l7, ncm-l7).
The tokenizer gets a second goal next to rebuilding: pick codes the generator finds easy to guess.

How the signal reaches the encoder (today it cannot: residuals are built from `z.detach()`, quant.py:451):
1. Re-run the tokenizer's level-by-level choice with gradient: residual_k = z - (codes already chosen, levels < k),
   resampled to level k's length exactly as the quantizer does.
2. Turn its distances to the 2,048 codebook vectors into soft scores: q_k = softmax(-distance / tau_k).
   The nearest code gets the highest score, so argmax q_k equals the code actually chosen (a unit test checks this).
3. Guess term = sum over levels and positions of -sum_c q(c) log p_model(c), with the generator's probabilities
   p_model held fixed. Its gradient moves z toward codes the generator already expects. It only changes a choice
   when two codes are close (a near-tie), which is where the decoder barely cares (swap test: 96-99.9% of tokens
   unchanged when every code of a level is replaced by its runner-up).
4. tau_k per level = median gap between the nearest and second-nearest distance on codec-l7 (measured before any
   run), so near-ties are soft and clear wins stay hard.

Losses in one joint step (same 32 windows of 192 + 64 tokens for both parts):
- tokenizer: rebuild + vq loss (as now) + lambda x guess term;
- generator: its usual loss on the tokenizer's current hard codes. No generator gradient reaches the tokenizer
  except through the guess term (prefix embedding and level inputs stay detached).
Warm start from codec-l7 and ncm-l7, tokenizer learning rate 1e-4 (5x below its original peak) so codes drift slowly.

## Guard against cheating
- The easiest codes to guess are constant ones, which cannot rebuild the text. The reconstruction gate stays:
  >= 99% tokens and >= 50% exact chunks on 1,024 held-out chunks.
- The final test does not use the jointly trained generator (it has seen more training). After joint training the
  tokenizer is frozen and a fresh generator is trained exactly like ncm-l7 (same size, 1,800 steps, same data
  order), then measured. So a lower number means the codes themselves got easier to guess.
- The tokenizer still sees only the chunk; at test time codes come only from the generator reading the prefix.

## Steps
0. Build (about a day): code_scale/joint.py and a `joint` stage in train.py. Checks: argmax of the soft scores
   equals the real codes on codec-l7; with lambda = 0 the tokenizer trains as before; the guess term gives the
   encoder a non-zero gradient; whole pipeline dry-run offline (internet blocked) on throwaway folders.
   If 32 windows do not fit in 4 GB, use 2 x 16 with gradient accumulation (same effective batch).
1. Pre-test (about 2 h GPU): 300 joint steps for lambda = 0 (control: just more training), 0.01, 0.03, 0.1. For each: tokens rebuilt and exact chunks (1,024 held-out chunks), held-out guess bits of the joint
   generator (level_bits.py), share of codes that changed vs codec-l7.
   Pick rule: among lambda > 0 arms with >= 99.5% tokens and >= 70% exact chunks, the one with the lowest guess
   bits; it qualifies only if its bits are at least 5% below the lambda = 0 arm. None qualifies: STOP, no night run.
2. Main run (about 5.5 h, offline, keep-awake; chained right after the pre-test, so about 7.5 h in all): joint 1,500 steps with the picked lambda -> freeze tokenizer ->
   gate -> fresh generator 1,800 steps -> bits per level -> step-6 quality test -> scoring -> verdict.

## Pass rule (same numbers as ideas 1 and 1b, vs ncm-l7 measured by level_bits.py)
- PASS on bits if: extra bits over GPT <= 611.7 (30% fewer than 873.9), levels 0-4 together drop >= 131.1 bits,
  and the gate holds.
- Verdict: CONTINUE if bits PASS or step-6 quality reaches PROGRESS (similarity ratio >= 0.78, or a with-code
  syntax ratio >= 0.40); otherwise STOP. Also reported: share of codes changed, per-level bits.

## Risks
- Upstream notes that a straight-through gradient at levels above 0 diverges within ~100 steps. This term is
  different (soft scores on the residual, codebook held fixed), but instability is the main risk; the pre-test
  is there to catch it.
- Moving target: the generator chases codes that keep changing; low tokenizer learning rate and warm start limit this.
- Near-tie room may be too small to save ~260 bits; the swap test suggests up to ~530 bits sit in near-tie
  choices at levels 0, 1, 2 and 6, but how many can move without breaking the rebuild is unknown.

Chance of passing the bits rule: about 25% (my estimate). Cost: no money, no internet, laptop GPU only.

## Values fixed before any joint run (2026-11-03 ~23:40)
- tau per level (runs/code-scale-v1/joint-taus.json, codec-l7, 1,024 held-out chunks): 6726.9, 1.44, 3.64, 15.40,
  4.50, 22.34, 2.10 (level 0's distances are on a much larger scale because of its learned down-sampling).
- Check passed: on 256 held-out chunks the soft scores' top choice equals codec-l7's real codes at every position.
- Lambda grid changed from the draft (0.03 / 0.1 / 0.3) to **0, 0.01, 0.03, 0.1**: measured on codec-l7 + ncm-l7,
  the guess term's gradient on the encoder is 9.0x the rebuild + vq gradient (0.958 vs 0.106, median of 8 batches),
  so these lambdas give the guess push about 0.1x, 0.3x and 0.9x the rebuild signal; 0.3 would be 2.7x.
- Joint step: 5.2 s, 3.5 of 4 GB GPU memory at 32 windows (no accumulation needed).
- Code: code_scale/joint.py, code_scale/train_joint.py, code_scale/decide_j.py, code_scale/pipeline_j.sh.
  Verdict file runs/code-scale-v1/joint-verdict.md. Whole pipeline dry-run with internet blocked first.

## Strong-push check (fixed 2026-11-04 ~08:45, before any result of it)
**Why.** The pre-test above (STOP) showed the gentle push did nothing, not even to its own target (guess term
7.82 nats in all four arms): at tokenizer lr 1e-4 for 300 steps the encoder barely moved. This check asks whether
any push can make the codes cheaper to guess. Same code, start (codec-l7 + ncm-l7), 300 steps, data and taus.
Changes: tokenizer lr **5e-4** (its original peak) in every arm including the control, lambdas **0, 0.3, 1.0**
(guess push about 2.7x and 9x the rebuild signal). Folders codec-jt2-pre-*, ncm-jt2-pre-*;
verdict runs/code-scale-v1/joint2-verdict.md; log pipeline-joint2.log; code_scale/pipeline_j2.sh.

**Rule (same as the pre-test above).** Among lambda > 0 arms with >= 99.5% tokens and >= 70% exact chunks, the one
with the fewest held-out bits (joint generator, level_bits.py); it qualifies only if its bits are <= 95% of the
lambda 0 arm's. None qualifies: STOP joint training, and with it NCM for code at this scale. One qualifies: the
main run (1,500 joint steps at that lambda and lr 5e-4 -> frozen tokenizer -> gate -> fresh 1,800-step generator
-> bits rule and step-6 test, unchanged) starts at 19:30 tonight (runs/code-scale-v1/HOLD; deleting it starts it
sooner). Also reported: the training-log guess term per arm and the share of codes changed.
