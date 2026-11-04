"""Joint training of tokenizer + generator (docs/joint-training-plan.md).

Warm start from a trained codec and NCM. Each step, on the same windows:
  tokenizer loss = rebuild + vq loss + lam * guess term   (guess term: code_scale/joint.py)
  generator loss = usual NCM loss on the tokenizer's current codes (never reaches the tokenizer)

python -m code_scale.train_joint --steps 300 --lam 0.1 --out-codec codec-jt-x --out-ncm ncm-jt-x
Writes RUNS/<out-codec>/{final.pt,summary.json,log.jsonl} (a normal codec dir) and RUNS/<out-ncm>/.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from code_scale import data
from code_scale.joint import BOUNDS, LEVELS, guess_term, load_taus, soft_codes
from code_scale.models import ATTN, CODEC, ENC, SHAPE, build_codec, build_ncm, count, ncm_logits, ncm_loss
from code_scale.train import DEV, RUNS, codec_eval, windows

P, T = SHAPE["prefix_len"], SHAPE["chunk"]


def lr_at(step, total, peak, warmup=50):
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


def load(vocab, pad, codec_dir, ncm_dir):
    codec = build_codec(vocab, restart_fix=True).to(DEV)
    codec.load_state_dict(torch.load(RUNS / codec_dir / "final.pt", map_location=DEV, weights_only=False)["model"])
    ncm = build_ncm(codec, pad).to(DEV)  # sets codec eval + no grad; undone below
    ncm.load_state_dict(torch.load(RUNS / ncm_dir / "final.pt", map_location=DEV, weights_only=False)["model"])
    return codec, ncm


@torch.no_grad()
def code_drift(codec, base, val, n=1024):
    x = windows(val, n, T, np.random.default_rng(123))
    a = torch.cat([codec.encode(x[i:i + 128]) for i in range(0, n, 128)])
    b = torch.cat([base.encode(x[i:i + 128]) for i in range(0, n, 128)])
    return [round(float((a[:, BOUNDS[l]:BOUNDS[l + 1]] != b[:, BOUNDS[l]:BOUNDS[l + 1]]).float().mean()), 4)
            for l in range(len(LEVELS))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--out-codec", required=True)
    ap.add_argument("--out-ncm", required=True)
    ap.add_argument("--from-codec", default="codec-l7")
    ap.add_argument("--from-ncm", default="ncm-l7")
    ap.add_argument("--taus", default="runs/code-scale-v1/joint-taus.json")
    ap.add_argument("--codec-lr", type=float, default=1e-4)
    ap.add_argument("--ncm-lr", type=float, default=3e-4)
    ap.add_argument("--accum", type=int, default=1, help="split the 32 windows into this many pieces")
    args = ap.parse_args()
    assert ENC == "attn" and CODEC["dec_layers"] == 2 and ATTN == "block_diagonal"
    tok, eos, pad = data.ids()
    vocab = tok.get_vocab_size()
    train, val = data.stream("train"), data.stream("val")
    taus = load_taus(args.taus)
    codec, ncm = load(vocab, pad, args.from_codec, args.from_ncm)
    base = build_codec(vocab).to(DEV)
    base.load_state_dict(codec.state_dict())
    base.eval().requires_grad_(False)
    codec.requires_grad_(True)
    out_c, out_n = RUNS / args.out_codec, RUNS / args.out_ncm
    out_c.mkdir(parents=True, exist_ok=True)
    out_n.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    batch = 32
    if os.environ.get("CODE_SCALE_TEST_BATCH"):  # dry runs only; real runs never set it
        batch = min(batch, int(os.environ["CODE_SCALE_TEST_BATCH"]))
    cp = [p for p in codec.parameters() if p.requires_grad]
    npar = [p for p in ncm.parameters() if p.requires_grad]
    opt_c = torch.optim.AdamW(cp, lr=args.codec_lr, betas=(0.9, 0.95), weight_decay=0.1)
    opt_n = torch.optim.AdamW(npar, lr=args.ncm_lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler()
    meta = {"stage": "joint", "steps": args.steps, "lam": args.lam, "batch": batch, "accum": args.accum,
            "from_codec": args.from_codec, "from_ncm": args.from_ncm, "codec_lr": args.codec_lr,
            "ncm_lr": args.ncm_lr, "taus": taus, "levels": SHAPE["levels"], "codebooks": SHAPE["codebooks"],
            "encoder": ENC, "dec_layers": CODEC["dec_layers"]}
    print(json.dumps(meta), flush=True)
    log = (out_c / "log.jsonl").open("a")
    started = time.time()
    piece = batch // args.accum
    for step in range(args.steps):
        codec.train()
        ncm.train()
        for g in opt_c.param_groups:
            g["lr"] = lr_at(step, args.steps, args.codec_lr)
        for g in opt_n.param_groups:
            g["lr"] = lr_at(step, args.steps, args.ncm_lr)
        w = windows(train, batch, P + T, rng)
        opt_c.zero_grad(set_to_none=True)
        opt_n.zero_grad(set_to_none=True)
        stats = {"rec": 0.0, "vq": 0.0, "guess": 0.0, "gen": 0.0}
        for a in range(args.accum):
            ww = w[a * piece:(a + 1) * piece]
            prefix, x = ww[:, :P], ww[:, P:]
            with torch.autocast("cuda", dtype=torch.float16):
                z = codec._encode_pre_quant(x)
                z_q, vq_loss, *_ = codec.vq(z)
                logits = codec.decoder(z_q)
                rec = F.cross_entropy(logits.float().reshape(-1, vocab), x.reshape(-1))
            codes, logq = soft_codes(codec.vq, z, taus)
            with torch.autocast("cuda", dtype=torch.float16):
                gen_logits = ncm_logits(ncm, codec, prefix, codes, pad)
                gen = ncm_loss(gen_logits, codes)
            guess = guess_term(logq, gen_logits)
            loss = (rec + vq_loss + args.lam * guess + gen) / args.accum
            scaler.scale(loss).backward()
            for k, v in (("rec", rec), ("vq", vq_loss), ("guess", guess), ("gen", gen)):
                stats[k] += float(v) / args.accum
        scaler.unscale_(opt_c)
        scaler.unscale_(opt_n)
        torch.nn.utils.clip_grad_norm_(cp, 1.0)
        torch.nn.utils.clip_grad_norm_(npar, 1.0)
        scaler.step(opt_c)
        scaler.step(opt_n)
        scaler.update()
        if step % 25 == 0 or step == args.steps - 1:
            row = {"step": step, **{k: round(v, 4) for k, v in stats.items()},
                   "guess_bits_per_chunk": round(stats["guess"] * sum(LEVELS) / math.log(2), 1),
                   "elapsed_s": round(time.time() - started, 1)}
            if step % 100 == 0 or step == args.steps - 1:
                codec.eval()
                row.update(codec_eval(codec, val, 256))
                codec.train()
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            if not math.isfinite(stats["rec"] + stats["guess"] + stats["gen"]):
                raise SystemExit("loss is not finite; stopping")
    codec.eval()
    torch.save({"model": codec.state_dict(), "step": args.steps}, out_c / "final.pt")
    secs = round(time.time() - started, 1)
    csum = {**meta, "stage": "codec", "joint": True, "params": count(codec), "vocab": vocab, "restart_fix": True,
            "train_seconds": secs, "val_reconstruction": codec_eval(codec, val),
            "codes_changed_vs_start": code_drift(codec, base, val)}
    (out_c / "summary.json").write_text(json.dumps(csum, indent=2))
    torch.save({"model": ncm.state_dict(), "step": args.steps}, out_n / "final.pt")
    nsum = {**meta, "stage": "ncm", "joint": True, "codec": args.out_codec, "attn_pattern": ATTN, "reveal": [],
            "passes": 1, "params": count(ncm), "vocab": vocab, "train_seconds": secs}
    (out_n / "summary.json").write_text(json.dumps(nsum, indent=2))
    print(json.dumps(csum), flush=True)


if __name__ == "__main__":
    main()
