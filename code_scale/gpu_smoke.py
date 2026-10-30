"""Step 1 gate: all three models train on the local GPU with fp16; measure throughput."""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from code_scale.models import SHAPE, build_ar, build_codec, build_ncm, count, ncm_logits, ncm_loss

VOCAB, PAD = 8192, 8191
OUT = Path("evals/code-scale-v1/gpu-smoke.json")


def timed(step, n=12, warm=3):
    for _ in range(warm):
        step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda")
    torch.manual_seed(0)
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "shape": SHAPE}
    P, T = SHAPE["prefix_len"], SHAPE["chunk"]

    codec = build_codec(VOCAB).to(dev)
    opt = torch.optim.AdamW([p for p in codec.parameters() if p.requires_grad], lr=3e-4)
    scaler = torch.amp.GradScaler()
    Bc = 64
    x = torch.randint(0, VOCAB, (Bc, T), device=dev)

    def codec_step():
        with torch.autocast("cuda", dtype=torch.float16):
            logits, vq_loss, *_ = codec(x)
            loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), x.reshape(-1)) + vq_loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.reset_peak_memory_stats()
    s = timed(codec_step)
    result["codec"] = {"params": count(codec), "batch": Bc, "sec_per_step": s,
                       "tokens_per_sec": Bc * T / s, "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
    del opt

    codec.eval()
    gen = build_ncm(codec, PAD).to(dev)
    opt = torch.optim.AdamW([p for p in gen.parameters() if p.requires_grad], lr=3e-4)
    Bn = 32
    prefix = torch.randint(0, VOCAB, (Bn, P), device=dev)
    target = torch.randint(0, VOCAB, (Bn, T), device=dev)
    codes = codec.encode(target)

    def ncm_step():
        gen.train()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = ncm_loss(ncm_logits(gen, codec, prefix, codes, PAD), codes)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.reset_peak_memory_stats()
    s = timed(ncm_step)
    gen_params = count(gen)
    result["ncm"] = {"params": gen_params, "batch": Bn, "sec_per_step": s,
                     "windows_per_sec": Bn / s, "tokens_per_sec": Bn * (P + T) / s,
                     "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}

    gen.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for bs in (1, 32):
            p = prefix[:1].expand(bs, -1).contiguous()
            s = timed(lambda: codec.decode(gen.generate(None, p, temperature=1.0, top_k=0, top_p=0.95)), n=5, warm=2)
            result["ncm"][f"generate_b{bs}_sec"] = s
            result["ncm"][f"generate_b{bs}_tokens_per_sec"] = bs * T / s
    del opt, gen

    ar = build_ar(VOCAB).to(dev)
    opt = torch.optim.AdamW(ar.parameters(), lr=3e-4)
    seq = torch.cat([prefix, target], dim=1)

    def ar_step():
        ar.train()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = ar(seq[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.reset_peak_memory_stats()
    s = timed(ar_step)
    result["ar"] = {"params": count(ar), "batch": Bn, "sec_per_step": s, "windows_per_sec": Bn / s,
                    "tokens_per_sec": Bn * (P + T) / s, "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
    ar.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for bs in (1, 32):
            p = prefix[:1].expand(bs, -1).contiguous()
            s = timed(lambda: ar.generate(p, T, temperature=1.0, top_p=0.95), n=3, warm=1)
            result["ar"][f"generate_b{bs}_sec"] = s
            result["ar"][f"generate_b{bs}_tokens_per_sec"] = bs * T / s

    result["passed"] = True
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
