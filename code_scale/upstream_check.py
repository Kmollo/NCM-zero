"""Phase A4: run the official TinyStories NCM two ways and compare.

  upstream: exactly like upstream scripts/eval_shard.py (fp32, decoder(z_levels[-1]).argmax)
  ours:     like code_scale.evaluate (fp16 autocast generate, codec.decode(codes).argmax)

If the official model writes coherent, non-looping stories both ways, our generation
harness is not the cause of the l7 repetition. Read-only; writes runs/code-scale-v1/upstream-check.json.

python -m code_scale.upstream_check
"""
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from code_scale.diagnose import rep4
from ncm_code.model import upstream

CK = Path("artifacts/upstream-tinystories")
OUT = Path("runs/code-scale-v1/upstream-check.json")


def main(n=64, bs=16):
    upstream()  # verifies the pinned upstream revision and puts it on sys.path
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    cfg = OmegaConf.load(CK / "config.yaml")
    cfg.tokenizer.vqvae.checkpoint_path = str(CK / cfg.checkpoints.tokenizer)
    cfg.tokenizer.pre_tokenizer.vocab_file_path = str(CK / cfg.checkpoints.bpe_vocab)
    cfg.tokenizer.pre_tokenizer.merges_file_path = str(CK / cfg.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    vq_model = tk.vqvae.cuda().eval()
    net = NextConceptTransformer(vqvae_local=vq_model, config=cfg.model).cuda()
    sd = torch.load(CK / cfg.checkpoints.generator, map_location="cpu", weights_only=True)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()
    l0_p = sd["inference"]["level0_unigram"].float()
    eos, pad = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
    g = cfg.generation
    L0 = vq_model.vq.levels[0]

    def text(ids):
        ids = ids[:ids.index(eos)] if eos in ids else ids
        return tk.pre_tokenizer.decode([t for t in ids if t not in (eos, pad)], skip_special_tokens=True), ids

    @torch.no_grad()
    def run(mode, seed):
        torch.manual_seed(seed)
        outs = []
        for _ in range(0, n, bs):
            l0 = torch.multinomial(l0_p, bs * L0, replacement=True).view(bs, L0).cuda()
            x = torch.full((bs, 1), eos, device="cuda")
            if mode == "upstream":
                codes = net.generate(l0, x, temperature=g.temperature, top_k=g.top_k, top_p=g.top_p)
                _, zl = vq_model.vq.idx_to_zhats(codes, return_zlevels=True)
                ids = vq_model.decoder(zl[-1]).argmax(-1)
            else:  # our decode path (codec.decode); fp32, because fp16 overflows on this model (see fp16 check)
                codes = net.generate(l0, x, temperature=g.temperature, top_k=g.top_k, top_p=g.top_p)
                ids = vq_model.decode(codes).argmax(-1)
            outs += ids.tolist()
        return outs

    report = {}
    for mode in ("upstream", "ours"):
        texts, reps = [], []
        for ids in run(mode, seed=42):
            t, kept = text(ids)
            texts.append(t)
            reps.append(rep4(kept))
        report[mode] = {"repetition": round(sum(reps) / len(reps), 4),
                        "mean_words": round(sum(len(t.split()) for t in texts) / len(texts), 1),
                        "examples": texts[:3]}
        print(json.dumps({mode: {k: v for k, v in report[mode].items() if k != "examples"}}), flush=True)

    # Are the two decode paths the same function? Compare logits on one batch of codes.
    with torch.no_grad():
        torch.manual_seed(0)
        l0 = torch.multinomial(l0_p, 4 * L0, replacement=True).view(4, L0).cuda()
        codes = net.generate(l0, torch.full((4, 1), eos, device="cuda"), temperature=1.0, top_k=0, top_p=1.0)
        _, zl = vq_model.vq.idx_to_zhats(codes, return_zlevels=True)
        a, b = vq_model.decoder(zl[-1]), vq_model.decode(codes)
        report["decode_paths_max_abs_diff"] = (a - b).abs().max().item()
        # Our evaluation generates under fp16 autocast. Does this model survive that?
        seen = {"nonfinite_levels": 0}
        raw = net._sample_level_raw

        def watch(logits, *args):
            seen["nonfinite_levels"] += int(not torch.isfinite(logits).all())
            return raw(torch.nan_to_num(logits.float()), *args)
        net._sample_level_raw = watch
        with torch.autocast("cuda", dtype=torch.float16):
            net.generate(l0, torch.full((4, 1), eos, device="cuda"), temperature=1.0, top_k=0, top_p=1.0)
        net._sample_level_raw = raw
        report["fp16_autocast_nonfinite_levels"] = f"{seen['nonfinite_levels']} of {len(vq_model.vq.levels) - 1}"
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"decode_paths_max_abs_diff": report["decode_paths_max_abs_diff"]}), flush=True)
    print("UPSTREAM_CHECK_DONE")


if __name__ == "__main__":
    main(*(int(v) for v in sys.argv[1:]))
