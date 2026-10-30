"""Fixed-protocol generation speed test (docs/code-scale-v1-plan.md, amendment 2026-10-27).

Both models in one session, warm-up, then the median of 3 repeats of 10 generations of one
64-token chunk at batch 1 and batch 32 (temperature 0.8, top_p 0.95). GPU clocks, temperature
and power source are logged before and after. Run with the laptop idle and on AC power.

CODE_SCALE_LEVELS=... CODE_SCALE_CODEC=... CODE_SCALE_NCM=... CODE_SCALE_EVAL=... python -m code_scale.speed
"""
import json
import statistics
import subprocess
import time

import numpy as np
import torch

from code_scale import data
from code_scale.evaluate import OUT, ar_chunk, load_models, ncm_chunk
from code_scale.models import SHAPE
from code_scale.train import windows

P, T = SHAPE["prefix_len"], SHAPE["chunk"]


def machine():
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,utilization.gpu,power.draw",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    ac = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"],
                        capture_output=True, text=True).stdout.strip()
    return {"gpu": gpu, "battery_status": ac, "on_ac_power": ac == "2"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok, eos, pad = data.ids()
    val = data.stream("val")
    codec, ncm, ar = load_models(tok.get_vocab_size(), pad)
    report = {"levels": SHAPE["levels"], "before": machine(), "results": {}}
    rng = np.random.default_rng(9)
    for bs in (1, 32):
        ctx = windows(val, bs, P, rng)
        fns = {"ncm": lambda: ncm_chunk(ncm, codec, ctx, 0.8), "ar": lambda: ar_chunk(ar, ctx, 0.8)}
        for fn in fns.values():  # warm-up both before timing either
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        for name, fn in fns.items():
            rates = []
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    fn()
                torch.cuda.synchronize()
                rates.append(bs * T * 10 / (time.perf_counter() - t0))
            report["results"][f"{name}_b{bs}"] = {"median_tokens_per_sec": round(statistics.median(rates), 1),
                                                  "repeats": [round(r, 1) for r in rates]}
            print(json.dumps({f"{name}_b{bs}": report["results"][f"{name}_b{bs}"]}), flush=True)
        r = report["results"]
        report[f"ncm_speedup_b{bs}"] = round(r[f"ncm_b{bs}"]["median_tokens_per_sec"] / r[f"ar_b{bs}"]["median_tokens_per_sec"], 2)
    report["after"] = machine()
    (OUT / "speed.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k.startswith("ncm_speedup") or k in ("before", "after")}), flush=True)


if __name__ == "__main__":
    main()
