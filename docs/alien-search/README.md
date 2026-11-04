# Alien idea search for NCM (2026-11-03): steps 0-3 done, top 3 picked, nothing trained

Goal: find a fix for the Path C finding. NCM needs 1,190.7 bits per 64-token chunk vs AR's 250.5, and
the extra ~940 bits are in the tokenizer's codes themselves ([../code-scale-v1-results.md](../code-scale-v1-results.md), Path C).

## What was done
0. Cloned [alejandrohdez00/alien-space-of-science](https://github.com/alejandrohdez00/alien-space-of-science)
   (MIT, commit a55d660) locally and downloaded its released models and data from Hugging Face
   (~480 MB: coherence + availability models, 273 idea atoms from 16,068 papers). No API keys, no paid calls.
1. Picked 36 of the 273 atoms related to the failure (VQ/RVQ, VAEs, information bottleneck, masked and
   order-free generation, STE/Gumbel, contrastive losses, locality hashing, code structure...). Scored every
   3-atom set containing >= 2 of them (156,450 sets) with Alien's two models; kept the top 60 at beta 0.7 and
   40 at beta 0.4 ([scripts/targeted_sample.py](scripts/targeted_sample.py), [alien_top100.json](alien_top100.json)).
2. Five Fable agents each turned 20 combinations into proposals, given the one-page
   [brief.md](brief.md); a sixth (control) got the brief only. 23 proposals in [proposals/](proposals/).
3. A Fable critic checked the claims against the code, ran two short CPU checks on `codec-l7`, grouped the
   proposals and ranked them ([proposals/critic.md](proposals/critic.md)). The lead (Opus) checked the key
   claims and reran the swap test.

## New facts found during screening (verified)
- **We deviated from upstream's text recipe.** All released NCM text tokenizers (lm1b, tinystories, xsum, owt)
  use a token-lookup encoder (`enc_layers=0`); ours has 2 bidirectional attention layers plus absolute
  positions (`code_scale/models.py` CODEC / build_codec). lm1b also drops the encoder position table.
- **No code-level loss can reach the encoder**: every residual is built from `z.detach()`
  (`upstream/ncm/models/quant.py:451`). This is why Path A's loss could only touch level 0.
- **The decoder ignores most code choices.** Replacing every code of a level with its second-nearest code
  leaves 99.86-99.93% of tokens unchanged at levels 0, 1, 2 and 99.71% at level 6 (level 5: 96.2%).
  Those four levels cost NCM ~532 bits per chunk, mostly for near-tie choices that do not change the output.
- So the pass test as first written is **gameable**: a 6-level codec without level 6 could pass by deletion.

## Top 3 to test (nothing has been trained)
| # | Idea | Change | Pre-test (< 30 min, existing checkpoints) | Test night | Critic's chance of passing |
|---|---|---|---|---|---|
| 1 | **Lookup, position-free tokenizer encoder** (upstream's own recipe) | `enc_layers=0`, `enc_positional=False`; arms: LayerNorm vs GroupNorm | 1,000-step codec vs codec-l7's log (94.0% tokens at step 1,000) | codec 5k steps + NCM 1.8k + bits, ~3.2 h | 30% |
| 2 | **Entropy-constrained RVQ**: encoder picks the code that is cheapest to predict among near-ties | assignment = distance + lambda x (-log prior), prior on the generator's own inputs; lambda only on near-tie levels 0,1,2,4,6; warm start from codec-l7 | fit the prior on existing codes (~15 min) | ~5 h | 25% |
| 3 | **Propose-and-verify re-encoding (RDOQ)**: re-encode data choosing the generator's preferred near-tie code when the decode is identical | no tokenizer retrain; EM-style: re-encode, retrain generator | 20-min check on existing generator + codec | ~5.5 h | 25% (bits); quality gain doubtful |

The three overlap (2 and 3 exploit the same near-tie freedom), so the chance that at least one passes is
roughly 40-50%, not the sum. Passing the bits test does not mean beating GPT; the step-6 quality test would
still decide that.

## Proposed pass rule for step 4 (to be fixed before any result)
A candidate passes only if all hold:
1. NCM extra bits over AR drop >= 30% (940 -> <= 660 per chunk), measured as in `code_scale/reveal_bits.py`
   (1 pass);
2. at least 140 of the removed bits come from levels 0-4 (so dropping or cheapening level 6 alone cannot pass);
3. tokenizer gate still holds: >= 99% tokens, >= 50% exact chunks on 1,024 held-out chunks;
4. reported alongside (does not decide): continuation similarity ratio vs AR on 256 windows.
Optional control arm: a 6-level codec `[4,8,16,32,48,64]`, to show what deletion alone gives.

## Did Alien help?
Partly. Two specific ideas came from Alien combinations and not from the control: propose-and-verify
re-encoding (from the speculative-decoding atom) and the lookup-encoder argument with the "levels 0-2 hold
only 248 bits" arithmetic. On the leading idea (entropy-constrained tokenizer) the control agent was as
good or better. No agent noticed on its own that the top idea is upstream's own recipe; that came from
checking the configs during screening.
