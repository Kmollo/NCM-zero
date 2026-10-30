"""Train the NCM tokenizer (codec), the NCM generator, or the AR control.

python -m code_scale.train codec|ncm|ar --steps N
Checkpoints and logs go to runs/code-scale-v1/<stage>/.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from code_scale import data
from code_scale.models import ATTN, PASSES, REVEAL, SHAPE, random_shown, build_ar, build_codec, build_ncm, count, ncm_logits, ncm_loss
from code_scale.plan_loss import PlanPredictor, ar_prefix_features, info_nce, level0_latent

RUNS = Path("runs/code-scale-v1")
CODEC_DIR = os.environ.get("CODE_SCALE_CODEC", "codec")  # which trained tokenizer NCM uses
DEV = torch.device("cuda")
P, T = SHAPE["prefix_len"], SHAPE["chunk"]


def windows(stream, batch, length, rng):
    starts = rng.integers(0, len(stream) - length, size=batch)
    return torch.from_numpy(np.stack([stream[s:s + length].astype(np.int64) for s in starts])).to(DEV)


def lr_at(step, total, peak, warmup=200):
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


def load_codec(vocab):
    codec = build_codec(vocab).to(DEV)
    state = torch.load(RUNS / CODEC_DIR / "final.pt", map_location=DEV, weights_only=False)
    codec.load_state_dict(state["model"])
    codec.eval().requires_grad_(False)
    return codec


def require_codec_quality(summary_path):
    """A minimum reconstruction gate, not authorization for generator training.

    >= 99% token accuracy and >= 50% exact 64-token chunks (exact part relaxed
    from 99% by the user on 2026-10-26: 99% exact chunks needs ~99.98% per token).
    """
    path = Path(summary_path)
    if not path.exists():
        raise ValueError("Codec quality evidence missing; generator training refused")
    result = json.loads(path.read_text()).get("val_reconstruction", {})
    accuracy, exact = result.get("token_accuracy", -1), result.get("chunk_exact", -1)
    if (not isinstance(accuracy, (int, float)) or not isinstance(exact, (int, float))
            or not math.isfinite(accuracy) or not math.isfinite(exact)
            or not 0.99 <= accuracy <= 1 or not 0.50 <= exact <= 1
            or result.get("chunks", 0) < 512):
        raise ValueError("Codec failed reconstruction gate; generator training refused")


def wait_for_hold_release():
    """Keep the laptop usable: GPU-heavy NCM training waits while runs/code-scale-v1/HOLD
    exists and its ISO local time ("2026-10-26T23:00") has not passed. Delete the file to start now."""
    from datetime import datetime
    hold = RUNS / "HOLD"
    while hold.exists():
        until = datetime.fromisoformat(hold.read_text().strip())
        if datetime.now() >= until:
            break
        print(json.dumps({"held_until": until.isoformat()}), flush=True)
        time.sleep(60)


@torch.no_grad()
def codec_eval(codec, val, n=1024):
    rng = np.random.default_rng(123)
    x = windows(val, n, T, rng)
    codec.eval()
    out = []
    for i in range(0, n, 128):
        out.append(codec.decode(codec.encode(x[i:i + 128])).argmax(-1))
    rec = torch.cat(out)
    codes = torch.cat([codec.encode(x[i:i + 128]) for i in range(0, n, 128)])
    bounds = np.cumsum([0] + SHAPE["levels"])
    used = [int(codes[:, bounds[l]:bounds[l + 1]].unique().numel()) for l in range(3)]  # collapse watch
    return {"token_accuracy": (rec == x).float().mean().item(),
            "chunk_exact": (rec == x).all(1).float().mean().item(), "chunks": n, "codes_used_l0_l2": used}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["codec", "ncm", "ar"])
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--stop-at", type=int, default=None, help="end early but keep the --steps LR schedule")
    ap.add_argument("--restart-fix", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plan-weight", type=float, default=0.0,
                    help="codec only: weight of the level-0 predictability (InfoNCE) loss, Path A")
    ap.add_argument("--plan-batch", type=int, default=32,
                    help="windows per batch that get the plan loss (the AR prefix read costs ~1.2 s per 64)")
    args = ap.parse_args()
    if args.stage == "ncm":
        require_codec_quality(RUNS / CODEC_DIR / "summary.json")
        wait_for_hold_release()
    tok, eos, pad = data.ids()
    vocab = tok.get_vocab_size()
    train, val = data.stream("train"), data.stream("val")
    out = RUNS / (args.out or args.stage)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng({"codec": 1, "ncm": 2, "ar": 2}[args.stage])  # NCM and AR see identical windows

    planner = None
    if args.stage == "codec":
        model = build_codec(vocab, restart_fix=args.restart_fix).to(DEV)
        batch, peak = 64, 5e-4
        if args.plan_weight > 0:  # version 1: frozen AR control reads the prefix
            ar = build_ar(vocab).to(DEV)
            ar.load_state_dict(torch.load(RUNS / "ar/final.pt", map_location=DEV, weights_only=False)["model"])
            ar.eval().requires_grad_(False)
            planner = PlanPredictor(2 * ar.wte.weight.shape[1], SHAPE["levels"][0], model.vq.embed_dim).to(DEV)
    elif args.stage == "ncm":
        codec = load_codec(vocab)
        model = build_ncm(codec, pad).to(DEV)
        batch, peak = 32, 1e-3
    else:
        model = build_ar(vocab).to(DEV)
        batch, peak = 32, 1e-3
    params = [p for p in model.parameters() if p.requires_grad]
    if args.stage == "codec" and planner is not None:
        params += list(planner.parameters())
    opt = torch.optim.AdamW(params, lr=peak, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler()
    log = (out / "log.jsonl").open("a")
    started = time.time()
    meta = {"stage": args.stage, "steps": args.steps, "batch": batch, "params": count(model), "vocab": vocab}
    if args.stage in ("codec", "ncm"):
        meta.update(levels=SHAPE["levels"], codebooks=SHAPE["codebooks"])
    if args.stage == "ncm":
        meta.update(codec=CODEC_DIR, attn_pattern=ATTN, reveal=REVEAL, passes=PASSES if REVEAL else 1)
    if args.stage == "codec":
        meta.update(plan_weight=args.plan_weight, plan_batch=args.plan_batch if args.plan_weight > 0 else 0,
                    restart_fix=args.restart_fix, levels=SHAPE["levels"])
    print(json.dumps(meta), flush=True)

    for step in range(args.stop_at or args.steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, peak)
        if args.stage == "ncm":  # encode targets in fp32 so labels match inference-time codes
            w = windows(train, batch, P + T, rng)
            prefix, target = w[:, :P], w[:, P:]
            codes = codec.encode(target)
        with torch.autocast("cuda", dtype=torch.float16):
            if args.stage == "codec" and planner is not None:  # prefix + chunk windows
                w = windows(train, batch, P + T, rng)
                x = w[:, P:]
                z = model._encode_pre_quant(x)
                z_q, vq_loss, *_ = model.vq(z)
                logits = model.decoder(z_q)
                rec = F.cross_entropy(logits.float().reshape(-1, vocab), x.reshape(-1))
                pb = args.plan_batch
                plan, plan_acc = info_nce(planner(ar_prefix_features(ar, w[:pb, :P])),
                                          level0_latent(model, z[:pb]), planner.log_scale)
                loss = rec + vq_loss + args.plan_weight * plan
            elif args.stage == "codec":
                x = windows(train, batch, T, rng)
                logits, vq_loss, *_ = model(x)
                rec = F.cross_entropy(logits.float().reshape(-1, vocab), x.reshape(-1))
                loss = rec + vq_loss
            elif args.stage == "ncm":
                shown = random_shown(codes) if REVEAL else None
                loss = ncm_loss(ncm_logits(model, codec, prefix, codes, pad, shown=shown), codes, ignore=shown)
            else:
                w = windows(train, batch, P + T, rng)
                logits = model(w[:, :-1])
                loss = F.cross_entropy(logits.float().reshape(-1, vocab), w[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 50 == 0 or step == (args.stop_at or args.steps) - 1:
            rec_ = {"step": step, "loss": loss.item(), "lr": opt.param_groups[0]["lr"],
                    "elapsed_s": round(time.time() - started, 1)}
            if args.stage == "codec":
                rec_["rec_loss"] = rec.item()
                if planner is not None:
                    rec_.update(plan_loss=plan.item(), plan_top1=round(plan_acc.item(), 4))
                if step % 500 == 0 or step == (args.stop_at or args.steps) - 1:
                    rec_.update(codec_eval(model, val, 256))
            log.write(json.dumps(rec_) + "\n")
            log.flush()
            print(json.dumps(rec_), flush=True)
        if step and step % 1000 == 0:
            torch.save({"model": model.state_dict(), "step": step}, out / "latest.pt")

    final = {"model": model.state_dict(), "step": args.steps}
    if args.stage == "codec" and planner is not None:
        final["plan_predictor"] = planner.state_dict()
    torch.save(final, out / "final.pt")
    summary = {**meta, "train_seconds": round(time.time() - started, 1), "final_loss": loss.item()}
    if args.stage == "codec":
        summary["val_reconstruction"] = codec_eval(model, val)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
