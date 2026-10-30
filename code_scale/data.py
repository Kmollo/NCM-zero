"""BPE tokenizer training and packing the corpus into uint16 token streams."""
import json
import random
from pathlib import Path

import numpy as np

CORPUS = Path("data/code-pretrain-v1")
OUT = Path("runs/code-scale-v1/data")
VOCAB = 8192
SPECIALS = ["<|endoftext|>", "<|pad|>"]


def records(name):
    with (CORPUS / name).open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["content"]


def train_bpe():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    rng = random.Random(0)
    sample = (c for c in records("train.jsonl") if rng.random() < 0.3)  # ~120 MB
    tok.train_from_iterator(sample, trainer)
    OUT.mkdir(parents=True, exist_ok=True)
    tok.save(str(OUT / "bpe.json"))
    return tok


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(OUT / "bpe.json"))


def ids():
    tok = load_tokenizer()
    return tok, tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|pad|>")


def pack(split):
    tok, eos, _ = ids()
    path = OUT / f"{split}.bin"
    total = 0
    batch = []
    with path.open("wb") as f:
        def flush():
            nonlocal total
            for enc in tok.encode_batch(batch):
                arr = np.array(enc.ids + [eos], dtype=np.uint16)
                f.write(arr.tobytes())
                total += len(arr)
            batch.clear()
        for content in records(f"{split}.jsonl"):
            batch.append(content)
            if len(batch) == 512:
                flush()
        flush()
    return total


def roundtrip_check(n=300):
    """Exact text round-trip through BPE on val files (ByteLevel must be lossless)."""
    tok, _, _ = ids()
    ok = total = 0
    for content in records("val.jsonl"):
        total += 1
        ok += tok.decode(tok.encode(content).ids) == content
        if total == n:
            break
    return ok, total


def stream(split):
    return np.memmap(OUT / f"{split}.bin", dtype=np.uint16, mode="r")


def main():
    train_bpe()
    ok, total = roundtrip_check()
    counts = {s: pack(s) for s in ("train", "val")}
    report = {"vocab": VOCAB, "bpe_roundtrip_exact": f"{ok}/{total}", "tokens": counts}
    (OUT / "data-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
