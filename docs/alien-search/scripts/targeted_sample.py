"""Targeted Alien sampling for NCM(zero): score every 3-atom set that contains >= 2 of our seed atoms
(the third atom can be anything, which supplies the outside idea), with Alien's released coherence
and availability models, then keep the top sets at beta 0.7 and 0.4.

Run from the Alien repo root:  python targeted_sample.py OUT.json
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from generation.sample import _zscore, score_sets_coherence
from generation.scoring import load_author_embeddings, load_availability_model, load_model, score_sets_alien

SEEDS = [8, 9, 12, 22, 23, 26, 27, 32, 33, 46, 49, 50, 57, 59, 60, 65, 66, 69, 80, 81, 101, 102, 111, 112,
         122, 123, 124, 139, 156, 185, 188, 189, 193, 199, 236, 253]
TOP = {0.7: 60, 0.4: 40}


def main():
    out = Path(sys.argv[1])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    coh, tm = load_model("models/coherence_model")
    coh.to(dev).eval()
    av, atm = load_availability_model("models/availability_model")
    av.to(dev).eval()
    emb, _ = load_author_embeddings("models/availability_model", device="cpu")
    emb = emb.to(dev)
    atoms = sorted(int(c) for c in tm.raw["cluster_to_token_id"])
    seeds = set(SEEDS) & set(atoms)
    assert len(seeds) == len(SEEDS), set(SEEDS) - set(atoms)
    sets = [s for s in combinations(atoms, 3) if len(seeds.intersection(s)) >= 2]
    sys.stderr.write(f"device {dev}; {len(sets):,} candidate sets\n")
    c = score_sets_coherence(sets, coh, tm, device=dev, batch_size=512)
    a = score_sets_alien(sets, av, atm, author_embeddings=emb, device=dev, batch_size=512, top_k_authors=10)
    zc, zu = _zscore(c), _zscore(-np.asarray(a))
    names = {int(k): v["name"] for k, v in json.load(open("papers/clusters_80.json", encoding="utf-8"))["clusters"].items()}
    picked, rows = set(), []
    for beta, n in TOP.items():
        score = (1 - beta) * zc + beta * zu
        for i in np.argsort(-score):
            if len([r for r in rows if r["beta"] == beta]) >= n:
                break
            if sets[i] in picked:
                continue
            picked.add(sets[i])
            rows.append({"beta": beta, "atoms": list(sets[i]), "score": round(float(score[i]), 3),
                         "z_coherence": round(float(zc[i]), 3), "z_unavailability": round(float(zu[i]), 3),
                         "names": [names[x] for x in sets[i]]})
    out.write_text(json.dumps({"n_candidates": len(sets), "seeds": SEEDS, "rows": rows}, indent=1), encoding="utf-8")
    sys.stderr.write(f"wrote {len(rows)} sets to {out}\n")


if __name__ == "__main__":
    main()
