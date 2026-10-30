"""Lossless byte inputs and provenance-aware, repository-grouped splits."""
import hashlib
import json
from pathlib import Path

PAD, EOS, VOCAB = 256, 257, 258
ALLOWED_LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "CC0-1.0"}


def encode(text, length):
    if not isinstance(text, str) or type(length) is not int or length < 1:
        raise ValueError('text must be a string and length a positive integer')
    ids = list(text.encode("utf-8")) + [EOS]
    if len(ids) > length:
        raise ValueError(f"Input needs {len(ids)} bytes including EOS; limit is {length}. No truncation was applied.")
    return ids + [PAD] * (length - len(ids))


def decode(ids):
    """Require one EOS followed only by PAD; malformed actions fail closed."""
    result = []
    ended = False
    for value in ids:
        if type(value) is not int:
            raise ValueError('byte tokens must be exact integers')
        if ended:
            if value != PAD:
                raise ValueError('only PAD may follow EOS')
            continue
        if value == EOS:
            ended = True
            continue
        if not 0 <= value < 256:
            raise ValueError(f"Invalid byte token: {value}")
        result.append(value)
    if not ended:
        raise ValueError('missing EOS')
    return bytes(result).decode("utf-8", errors="strict")


def read_records(path):
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("prompt", "completion", "repository", "license", "source"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{path}:{line_no}: missing nonempty {key}")
            if row["license"] not in ALLOWED_LICENSES:
                raise ValueError(f"{path}:{line_no}: license is outside the explicit allowlist")
            records.append(row)
    if not records:
        raise ValueError(f"{path}: empty dataset")
    return records


def prepare(source, output, prefix_length, output_length, seed=17, validation_fraction=0.2):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    rows = read_records(source)
    unique, seen = [], set()
    for row in rows:
        encode(row["prompt"], prefix_length)
        encode(row["completion"], output_length)
        # Deduplicate identical completions, even if their instructions differ.
        key = hashlib.sha256(row["completion"].encode()).hexdigest()
        if key not in seen:
            unique.append(row)
            seen.add(key)
    groups = sorted({r["repository"] for r in unique}, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    if len(groups) < 2:
        raise ValueError("At least two repository groups are required for a disjoint validation split")
    count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    held_out = set(groups[:count])
    splits = {"train": [], "validation": []}
    for row in unique:
        splits["validation" if row["repository"] in held_out else "train"].append(row)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty dataset directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "seed": seed, "prefix_length": prefix_length, "output_length": output_length,
                "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "duplicates_removed": len(rows) - len(unique), "splits": {}}
    for split, records in splits.items():
        content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        (output / f"{split}.jsonl").write_text(content, encoding="utf-8", newline="\n")
        manifest["splits"][split] = {"records": len(records), "sha256": hashlib.sha256(content.encode()).hexdigest(),
                                      "repositories": sorted({r["repository"] for r in records})}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
