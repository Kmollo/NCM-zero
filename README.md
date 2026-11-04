# NCM(zero)

**Can a Next-Concept Model match a GPT-style model on Python code? At small scale, no. Here is
where the gap comes from.**

This repo is a small, pre-registered study of [Hazy Research's Next-Concept Model
(NCM)](https://github.com/HazyResearch/ncm) on real Python code. NCM does not predict text one token
at a time. A tokenizer compresses each 64-token chunk into multiscale codes, and a generator predicts
those codes, coarse to fine, in a few parallel steps. We compared it with a GPT-style
autoregressive (AR) model of about the same size, trained on the same data for the same number of
steps on one laptop GPU (Quadro T1000).

Every continue/stop rule was written down **before** its result came in
([docs/ncm-next-plan.md](docs/ncm-next-plan.md), [docs/joint-training-plan.md](docs/joint-training-plan.md)).
Seven fixes were tried; none passed its rule, so the verdict is **STOP: NCM for code at this scale**.

## Findings

1. **The upstream tokenizer had a dead-code restart bug.** Restarted codes were stored about
   1/threshold too large, so they never won and were restarted forever (reported upstream:
   [HazyResearch/ncm#16](https://github.com/HazyResearch/ncm/issues/16)). Fixing it, plus a
   7-level layout `[4, 8, 16, 32, 48, 64, 64]`, took the tokenizer from 65.7% token accuracy / 0%
   exact 64-token chunks to **99.72% / 88.1%**. Decoding the true codes now reproduces the text
   almost exactly (character similarity 0.996).
2. **With a near-perfect tokenizer, NCM still trails AR by a wide margin.** NCM reaches about
   **0.72** of AR's similarity to the real continuation and about **0.10** of AR's rate of valid
   Python that contains code.
3. **The generator cannot predict the coarse "plan".** Teacher-forced accuracy at level 0 is 1.6%
   (2,048 options). If the true codes for levels 0–3 are given, greedy similarity jumps from 0.135
   to 0.675, and the repetition loops disappear. NCM fills in detail well; it cannot choose the plan.
4. **Three structural fixes did not move quality:**

   | Run | Change | Similarity ratio | HumanEval valid-with-code ratio | Easy valid-with-code ratio | Bits per chunk |
   |---|---|---:|---:|---:|---:|
   | Baseline `ncm-l7` | — | 0.72 | 0.10 | 0.10 | 1,127 |
   | Path A `ncm-pp05` | tokenizer trained to make level 0 predictable | 0.72 | 0.13 | 0.12 | — |
   | Path B run 1 `ncm-bc-l7` | each level also sees the levels above it | 0.71 | 0.11 | 0.07 | 1,115 |
   | Path B run 2 `ncm-sc` | small coarse codebooks (64/128/256/512) | 0.73 | 0.00 | 0.10 | 1,008 |
   | Path C `ncm-rv7` | positions within a level see each other (3 passes) | 0.71 | 0.10 | 0.12 | 1,193 |

   Ratios are NCM ÷ AR (1.0 = parity). The continue bar was 0.78 for similarity or 0.40 for either
   syntax ratio; parity was 0.8 on all three.
5. **The extra bits are in the codes themselves, not in guessing blind.** The codes are a
   deterministic function of the text, so a perfect code model would need no more bits than the
   text itself. Measured on 2,048 held-out windows, teacher-forced, bits per 64-token chunk:

   | GPT (AR) | NCM, 1 pass (blind within a level) | NCM, 3 passes | NCM, 8 passes |
   |---:|---:|---:|---:|
   | 250.5 | 1,190.7 | 1,185.5 | 1,183.9 |

   Letting positions see the other codes in their level removes **5 of the 940 extra bits
   (0.6%)**. The pre-set rule needed at least half to blame blind guessing. So, at this scale, each
   code is hard to predict from the prefix even when every other code in its level is known.
6. **Speed.** NCM's single-sample advantage held in some runs: 3.2× AR at batch 1 under the
   fixed protocol (Path A). It is lost at batch 32 (0.47×), where the KV-cached AR is faster.
   Path C's multi-pass decoder was slower: 0.59× at batch 1, 0.08× at batch 32.
7. **Upstream's own text-tokenizer recipe does not carry over to code.** Our tokenizer encodes with
   2 attention layers plus positions; every released NCM text tokenizer uses a token-lookup encoder
   (`enc_layers=0`). An idea search with [Alien Space of Science](https://github.com/alejandrohdez00/alien-space-of-science)
   and Claude agents ([docs/alien-search/](docs/alien-search/)) ranked switching to it first. Lookup
   tokenizers stall near 96.6% tokens / ~21% exact chunks with either a 2-block or upstream's 4-block
   decoder (gate: 99% / 50%), so the attention encoder is what makes exact code reconstructable.
8. **Training the tokenizer and generator together does not make the codes guessable.** No signal about
   predictability reaches the encoder upstream (residuals are built from `z.detach()`). We added one:
   soft scores over the tokenizer's own code distances (their argmax is exactly its codes) carry the
   generator's cost back into the encoder ([code_scale/joint.py](code_scale/joint.py)). Starting from the
   working pair, 300 joint steps per setting, held-out bits of the joint generator:

   | Push strength (tokenizer lr) | Tokens rebuilt | Exact chunks | Bits per chunk |
   |---|---:|---:|---:|
   | 0 / 0.01 / 0.03 / 0.1 (1e-4) | 99.64–99.66% | 83.6–84.3% | 1,298–1,305 |
   | 0, control (5e-4) | 99.10% | 65.9% | 1,346 |
   | 0.3 (5e-4) | 99.08% | 65.6% | 1,321 |
   | 1.0 (5e-4) | 98.33% | 43.5% | 1,421 |

   A setting needed bits at least 5% below its control with the rebuild held. A gentle push changes
   nothing; a strong one lowers its own training target (7.77 → 7.41 nats) but not the held-out bits,
   and gives up exact reconstruction first.

## What this does and does not show

- It shows that at ~13M parameters and ~12M training tokens per model, NCM with this tokenizer
  learns Python far worse than a matched AR model. The bottleneck is that the codes are a
  hard-to-learn function of the text, not the tokenizer's accuracy or the generator's layout.
- It does **not** show that NCM fails at large scale. Upstream's reported results use much larger
  models; one seed and one small model each are a smoke-scale signal, not a capability claim.
  Both models score 0 pass@k on HumanEval at this size.
- At this size there is a trade we could not get around: a tokenizer that rebuilds exact code needs
  context, and its codes cost ~4.5× GPT's bits; making codes local (lookup encoder) or pushing them
  toward guessability (joint training) breaks exact reconstruction first.
- Untested: much larger models and data, where upstream reports NCM working on text.

## Repo layout

| Path | What |
|---|---|
| [docs/ncm-next-plan.md](docs/ncm-next-plan.md) | Paths A–C and ideas 1/1b: each change and its continue/stop rule, written before results |
| [docs/joint-training-plan.md](docs/joint-training-plan.md) | Joint training and the strong-push check, with their rules |
| [docs/alien-search/](docs/alien-search/) | Idea search: scored combinations, agent proposals, critic review, shortlist |
| [docs/code-scale-v1-results.md](docs/code-scale-v1-results.md) | Full results log in order, including deviations |
| [docs/code-scale-v1-plan.md](docs/code-scale-v1-plan.md) | Original study plan and step-6 gate |
| `code_scale/` | Models, training, evaluation, bits measurements (`reveal_bits.py`, `level_bits.py`), joint training (`joint.py`, `train_joint.py`), verdict scripts, overnight pipelines |
| `ncm_code/` | Thin adapter that loads the pinned upstream NCM modules |
| `zero_baseline/code_bench.py` | HumanEval / easy-task sampling scorer (sandboxed) |
| `evals/code-bench-v1/` | HumanEval (MIT) and MBPP (CC BY 4.0) task files with their licenses |
| `results/` | Small result files for every run: summaries, step-6 reports, scores, speed, bits, verdicts, pipeline logs |
| `data/code-pretrain-v1/` | Corpus provenance and manifest only (the corpus itself is not included) |
| `tests/` | Tests for the NCM variants, the plan loss and the joint-training code choice |

## Reproducing

Requirements: Python 3.10, a CUDA GPU, `pip install -r requirements.txt`, and Git Bash on Windows
(the pipelines are bash scripts). Model checkpoints and the training corpus are not included.

1. Fetch the pinned upstream NCM source (MIT, commit `c702742`):
   `python scripts/fetch_upstream.py` (pin in `upstream.lock.json`).
2. Build the corpus: `python -I scripts/prepare_code_corpus.py`. It downloads permissively licensed
   files from [codeparrot/codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean)
   at a pinned revision (see `data/code-pretrain-v1/provenance.json`). Its decontamination step also
   reads task files from the wider Zero project that are not in this repo; drop those paths if
   they are missing.
3. Run a path, for example Path C: `bash code_scale/pipeline_c.sh` (about 6 hours on a T1000). The
   scripts call `.venv/Scripts/python.exe`; change `PY=` on another OS. The earlier pipelines
   (`pipeline.sh`, `pipeline_layout.sh`, `pipeline_path_a.sh`, `pipeline_b.sh`) reproduce the
   earlier steps; `pipeline_1.sh`, `pipeline_1b.sh`, `pipeline_j.sh` and `pipeline_j2.sh` the
   tokenizer-recipe and joint-training runs (joint training needs `python -m code_scale.joint taus` first).
4. Tests: `python -m pytest tests`.

## Credits and license

- NCM model and tokenizer code: [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (MIT), not
  vendored here.
- HumanEval: OpenAI (MIT). MBPP: Google Research (CC BY 4.0).
- Idea-search scoring: [Alien Space of Science](https://github.com/alejandrohdez00/alien-space-of-science) (MIT);
  only its outputs are included (`docs/alien-search/alien_top100.json`).
- Training data: codeparrot/codeparrot-clean; each file keeps the license of its source repository.
- This repo's code and docs: MIT, see [LICENSE](LICENSE).

The experiments, code and write-ups were produced with Claude Code.
