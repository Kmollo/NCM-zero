"""Frozen code-generation benchmark (code-bench-v1) scored in CPython/WASI.

Suites: HumanEval (164), the MBPP sanitized test split (task ids 11-510) and a
small "easy" tier (the 20 shortest MBPP test tasks plus the held-out splits of
zero-composition-curriculum-v2). Every task prompt is a valid Python prefix, so
the scored program is always ``prompt + completion``; a completion may either
continue the prompt or restate the whole program.

Candidates never run in native Python. A batched guest first compiles unique
programs (compilation executes no candidate code); each syntax-valid unique
program then runs alone in a fresh guest from ``sandbox.run_python``, followed
by the task's tests in the same namespace. Like ``grading.py``, this is a
non-adversarial checker: WASI isolates the host, but a candidate sharing the
guest interpreter with its tests could forge an observation deliberately.

pass@k uses the unbiased estimator of Chen et al. 2021 (Codex paper).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ast
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import statistics
import sys
import threading
import time
from typing import Any, Callable, Iterable

from . import sandbox
from .sandbox import run_python


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "evals" / "code-bench-v1"
MANIFEST = BENCH / "frozen-manifest.json"
BENCH_VERSION = "code-bench-v1"
SUITES = ("humaneval", "mbpp", "easy")
DEFAULT_TIMEOUT = 20.0  # seconds per sample, including ~2-6 s guest startup
DEFAULT_K = (1, 10, 100)
EASY_MBPP_COUNT = 20
MBPP_TEST_IDS = range(11, 511)
CURRICULUM = ROOT / "data" / "zero-composition-curriculum-v2"
CURRICULUM_HELDOUT = ("unseen_constants.jsonl", "unseen_compositions.jsonl")
SYNTAX_CHUNK_FILES = 60
SYNTAX_CHUNK_BYTES = 1_500_000
PROGRAM_BYTES = sandbox.INPUT_LIMIT - 64 * 1024  # leave room for tests and entry

SOURCES = {
    "humaneval": {
        "name": "HumanEval (openai/human-eval)",
        "file": "raw/humaneval/HumanEval.jsonl.gz",
        "url": "https://raw.githubusercontent.com/openai/human-eval/463c980b59e818ace59f6f9803cd92c749ceae61/data/HumanEval.jsonl.gz",
        "repository": "https://github.com/openai/human-eval",
        "commit": "463c980b59e818ace59f6f9803cd92c749ceae61",
        "license": "MIT",
        "license_file": "raw/humaneval/LICENSE",
        "license_url": "https://github.com/openai/human-eval/blob/master/LICENSE",
        "citation": "Chen et al. 2021, Evaluating Large Language Models Trained on Code",
    },
    "mbpp": {
        "name": "MBPP sanitized (google-research/google-research/mbpp)",
        "file": "raw/mbpp/sanitized-mbpp.json",
        "url": "https://raw.githubusercontent.com/google-research/google-research/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp/sanitized-mbpp.json",
        "repository": "https://github.com/google-research/google-research/tree/master/mbpp",
        "commit": "f82046ba5aabbbb427dbfd38a254d26bff08b533",
        "license": "CC-BY-4.0",
        "license_file": "raw/mbpp/README.md",
        "license_url": "https://huggingface.co/datasets/google-research-datasets/mbpp",
        "citation": "Austin et al. 2021, Program Synthesis with Large Language Models",
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _docstring_prompt(text: str) -> str:
    """Wrap natural-language task text as a Python module docstring prefix."""
    body = text.strip().replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return '"""\n' + body + '\n"""\n'


# --- building the frozen task files ------------------------------------------------

def _humaneval_tasks() -> list[dict[str, Any]]:
    raw = gzip.decompress((BENCH / SOURCES["humaneval"]["file"]).read_bytes()).decode("utf-8")
    tasks = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tasks.append({
            "task_id": row["task_id"], "suite": "humaneval", "source": "humaneval",
            "prompt": row["prompt"], "entry_point": row["entry_point"],
            "canonical_completion": row["canonical_solution"],
            "test_code": row["test"] + f"\n\ncheck({row['entry_point']})\n",
        })
    return tasks


def _mbpp_entry_point(row: dict[str, Any]) -> str:
    names = [node.name for node in ast.walk(ast.parse(row["code"]))
             if isinstance(node, ast.FunctionDef)]
    tests = "\n".join(row["test_list"])
    used = [name for name in names if re.search(r"\b" + re.escape(name) + r"\s*\(", tests)]
    if not used:
        raise ValueError(f"Cannot identify MBPP entry point for task {row['task_id']}")
    return used[0]


def _mbpp_task(row: dict[str, Any], suite: str) -> dict[str, Any]:
    # bigcode-evaluation-harness style: description plus the first test as a docstring.
    return {
        "task_id": f"Mbpp/{row['task_id']}", "suite": suite, "source": "mbpp",
        "prompt": _docstring_prompt(row["prompt"] + "\n" + row["test_list"][0]),
        "description": row["prompt"], "tests": row["test_list"],
        "entry_point": _mbpp_entry_point(row),
        "canonical_completion": row["code"].rstrip() + "\n",
        "test_code": "\n".join(row["test_imports"] + row["test_list"]) + "\n",
    }


def _mbpp_rows() -> list[dict[str, Any]]:
    rows = json.loads((BENCH / SOURCES["mbpp"]["file"]).read_text(encoding="utf-8"))
    return sorted((row for row in rows if row["task_id"] in MBPP_TEST_IDS), key=lambda row: row["task_id"])


def _curriculum_tasks() -> tuple[list[dict[str, Any]], dict[str, str]]:
    tasks, hashes = [], {}
    for name in CURRICULUM_HELDOUT:
        path = CURRICULUM / name
        hashes[name] = sha256_file(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            asserts = "".join(f"assert solve({case['input']!r}) == {case['expected']!r}\n"
                              for case in row["tests"])
            tasks.append({
                "task_id": row["id"], "suite": "easy", "source": "zero-composition-curriculum-v2",
                "prompt": _docstring_prompt(row["prompt"]), "description": row["prompt"],
                "entry_point": "solve", "canonical_completion": row["completion"],
                "test_code": asserts,
            })
    return tasks, hashes


def build_tasks() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Deterministically derive all suites from the frozen raw files."""
    mbpp_rows = _mbpp_rows()
    easy_rows = sorted(mbpp_rows, key=lambda row: (len(row["code"].strip()), row["task_id"]))[:EASY_MBPP_COUNT]
    curriculum, curriculum_hashes = _curriculum_tasks()
    suites = {
        "humaneval": _humaneval_tasks(),
        "mbpp": [_mbpp_task(row, "mbpp") for row in mbpp_rows],
        "easy": [_mbpp_task(row, "easy") for row in sorted(easy_rows, key=lambda r: r["task_id"])] + curriculum,
    }
    for name, tasks in suites.items():
        ids = [task["task_id"] for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate task id in suite {name}")
        for task in tasks:
            ast.parse(task["prompt"])  # every prompt must be a valid Python prefix
    return suites, curriculum_hashes


def _jsonl(tasks: Iterable[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n" for task in tasks).encode("utf-8")


def build_manifest(suites: dict[str, list[dict[str, Any]]], curriculum_hashes: dict[str, str],
                   frozen_at: str) -> dict[str, Any]:
    sources = {}
    for key, source in SOURCES.items():
        sources[key] = {**source, "sha256": sha256_file(BENCH / source["file"]),
                        "license_file_sha256": sha256_file(BENCH / source["license_file"])}
    sources["zero-composition-curriculum-v2"] = {
        "name": "Zero composition curriculum v2, held-out splits",
        "path": "data/zero-composition-curriculum-v2",
        "files": curriculum_hashes,
        "manifest_sha256": sha256_file(CURRICULUM / "manifest.json"),
        "license": "Project-authored synthetic; redistribution license not assigned",
    }
    return {
        "benchmark": BENCH_VERSION, "status": "frozen", "frozen_at": frozen_at,
        "sources": sources,
        "suites": {
            name: {"file": f"tasks/{name}.jsonl", "tasks": len(tasks),
                   "sha256": sha256_bytes(_jsonl(tasks))}
            for name, tasks in suites.items()
        },
        "selection": {
            "humaneval": "All 164 HumanEval tasks.",
            "mbpp": "MBPP sanitized tasks whose task_id is in the official test range 11-510.",
            "easy": (f"The {EASY_MBPP_COUNT} MBPP sanitized test-split tasks with the shortest stripped "
                     "reference code (ties by task_id), plus every row of the curriculum's "
                     "unseen_constants and unseen_compositions held-out splits."),
        },
        "scoring": {
            "program": "prompt + completion, run as one module; then test_code in the same namespace",
            "prompt_format": {
                "humaneval": "original HumanEval prompt (imports, signature, docstring)",
                "mbpp": 'docstring """<description>\\n<first test assert>""" (bigcode-evaluation-harness style)',
                "easy": "MBPP tasks as above; curriculum tasks as a docstring of the original instruction",
            },
            "pass_at_k": "unbiased estimator 1 - C(n-c,k)/C(n,k) (Chen et al. 2021), averaged over tasks",
            "syntax_valid": "the whole program compiles in the guest CPython",
            "runtime": {"sandbox": "cpython-wasi-wasmtime", "wasmtime": sandbox.WASMTIME_VERSION,
                        "cpython_wasm_sha256": sandbox.CPYTHON_WASM_SHA256,
                        "memory_limit_bytes": sandbox.MEMORY_LIMIT, "fuel_limit": sandbox.FUEL_LIMIT},
            "default_timeout_seconds": DEFAULT_TIMEOUT,
        },
        "contamination": ("Training corpora must exclude these tasks: match on prompts, canonical "
                          "completions, and test asserts in tasks/*.jsonl."),
    }


def write_frozen(force: bool = False) -> dict[str, Any]:
    if MANIFEST.exists() and not force:
        raise FileExistsError("Benchmark is already frozen; use --check, or --force to rebuild")
    suites, curriculum_hashes = build_tasks()
    (BENCH / "tasks").mkdir(parents=True, exist_ok=True)
    for name, tasks in suites.items():
        (BENCH / "tasks" / f"{name}.jsonl").write_bytes(_jsonl(tasks))
    manifest = build_manifest(suites, curriculum_hashes, datetime.date.today().isoformat())
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def check_frozen() -> list[str]:
    """Return mismatches between frozen files and a fresh deterministic rebuild."""
    problems = []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    suites, curriculum_hashes = build_tasks()
    fresh = build_manifest(suites, curriculum_hashes, manifest["frozen_at"])
    if fresh != manifest:
        problems.append("frozen-manifest.json differs from a fresh rebuild")
    for name, info in manifest["suites"].items():
        path = BENCH / info["file"]
        if not path.is_file() or sha256_file(path) != info["sha256"]:
            problems.append(f"{info['file']} does not match its frozen hash")
    return problems


# --- loading ---------------------------------------------------------------------

def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_suite(name: str) -> list[dict[str, Any]]:
    """Load a suite after verifying its bytes against the frozen manifest."""
    if name not in SUITES:
        raise ValueError(f"Unknown suite {name!r}; choose from {', '.join(SUITES)}")
    info = load_manifest()["suites"][name]
    data = (BENCH / info["file"]).read_bytes()
    if sha256_bytes(data) != info["sha256"]:
        raise ValueError(f"{info['file']} changed since it was frozen; refusing to score")
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


# --- metrics ---------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one task with n samples, c correct (Chen et al. 2021)."""
    if not 0 <= c <= n or k < 1 or k > n:
        raise ValueError("Require 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


# --- guest programs --------------------------------------------------------------

_SYNTAX_ENTRY = '''import sys
NONCE = {nonce!r}
for index in range({count}):
    try:
        with open("/workspace/p%d.py" % index, encoding="utf-8") as handle:
            text = handle.read()
        compile(text, "candidate.py", "exec", dont_inherit=True)
    except BaseException as error:
        print(NONCE, index, "ERR", type(error).__name__, flush=True)
    else:
        print(NONCE, index, "OK", flush=True)
print(NONCE, "DONE", flush=True)
'''

_EXEC_ENTRY = '''import sys
NONCE = {nonce!r}

def _say(kind, error=None):
    detail = ""
    if error is not None:
        try:
            detail = type(error).__name__ + ": " + str(error)[:300]
        except BaseException:
            detail = "unprintable exception"
    print(NONCE, kind, detail, file=sys.stderr, flush=True)

with open("/workspace/candidate.py", encoding="utf-8") as handle:
    _program = handle.read()
with open("/workspace/bench_test.py", encoding="utf-8") as handle:
    _tests = handle.read()
try:
    _code = compile(_program, "candidate.py", "exec", dont_inherit=True)
except BaseException as error:
    _say("SYNTAX_ERROR", error)
    raise SystemExit(3)
_namespace = {{"__name__": "__candidate__", "__builtins__": __builtins__}}
try:
    exec(_code, _namespace)
except Exception as error:
    _say("CANDIDATE_ERROR", error)
    raise SystemExit(4)
try:
    exec(compile(_tests, "bench_test.py", "exec", dont_inherit=True), _namespace)
except AssertionError as error:
    _say("TEST_FAILED", error)
    raise SystemExit(1)
except Exception as error:
    _say("TEST_ERROR", error)
    raise SystemExit(2)
print(NONCE, "PASSED", flush=True)
'''

INFRA_ERRORS = {"runtime_unavailable", "worker_failed", "worker_start_failed", "sandbox_setup_failed"}
OUTCOMES = ("passed", "syntax_error", "candidate_error", "test_failed", "test_error", "timeout",
            "fuel_exhausted", "output_limit", "crash", "input_limit", "other_failure", "infra_error")


def _status_lines(text: str, nonce: str) -> list[list[str]]:
    return [line.split(" ", 2)[1:] for line in text.splitlines() if line.startswith(nonce + " ")]


def check_syntax(programs: list[str], timeout: float = 60) -> list[bool | None]:
    """Compile programs in batched guests. None means undetermined (run it fully)."""
    verdicts: list[bool | None] = [None] * len(programs)
    chunk: list[int] = []
    size = 0

    def flush() -> None:
        nonlocal chunk, size
        if not chunk:
            return
        nonce = "ZERO_SYNTAX_" + secrets.token_hex(12)
        files = {f"p{i}.py": programs[index] for i, index in enumerate(chunk)}
        result = run_python(_SYNTAX_ENTRY.format(nonce=nonce, count=len(chunk)), files, timeout=timeout)
        for parts in _status_lines(result.get("stdout", ""), nonce):
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) < len(chunk):
                verdicts[chunk[int(parts[0])]] = parts[1].startswith("OK")
        chunk, size = [], 0

    for index, program in enumerate(programs):
        length = len(program.encode("utf-8"))
        if length > PROGRAM_BYTES:
            continue  # left undetermined; execution reports input_limit
        if len(chunk) >= SYNTAX_CHUNK_FILES or size + length > SYNTAX_CHUNK_BYTES:
            flush()
        chunk.append(index)
        size += length
    flush()
    return verdicts


def run_program(program: str, test_code: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Run one program and its tests in a fresh guest; classify the outcome."""
    if len(program.encode("utf-8")) + len(test_code.encode("utf-8")) > PROGRAM_BYTES:
        return {"outcome": "input_limit", "passed": False, "syntax_valid": None,
                "wall_seconds": 0.0, "detail": "program exceeds sandbox input limit"}
    nonce = "ZERO_BENCH_" + secrets.token_hex(12)
    raw = run_python(_EXEC_ENTRY.format(nonce=nonce),
                     {"candidate.py": program, "bench_test.py": test_code}, timeout=timeout)
    stdout_status = [parts[0] for parts in _status_lines(raw.get("stdout", ""), nonce) if parts]
    stderr_status = _status_lines(raw.get("stderr", ""), nonce)
    kind = stderr_status[-1][0] if stderr_status and stderr_status[-1] else None
    detail = (stderr_status[-1][1] if stderr_status and len(stderr_status[-1]) > 1 else "")
    error = raw.get("error")
    syntax_valid = kind != "SYNTAX_ERROR"
    if error in INFRA_ERRORS:
        outcome, syntax_valid, detail = "infra_error", None, raw.get("stderr", "")[-300:]
    elif raw.get("timed_out"):
        outcome = "timeout"
    elif raw.get("output_limited"):
        outcome = "output_limit"
    elif error == "wasm_trap":
        trap = raw.get("trap", "")
        outcome = "fuel_exhausted" if "fuel" in trap.lower() else "crash"
        detail = trap[-300:]
    elif "PASSED" in stdout_status and raw.get("exit_code") == 0 and not error:
        outcome = "passed"
    elif kind == "SYNTAX_ERROR" and raw.get("exit_code") == 3:
        outcome = "syntax_error"
    elif kind == "CANDIDATE_ERROR" and raw.get("exit_code") == 4:
        outcome = "candidate_error"
    elif kind == "TEST_FAILED" and raw.get("exit_code") == 1:
        outcome = "test_failed"
    elif kind == "TEST_ERROR" and raw.get("exit_code") == 2:
        outcome = "test_error"
    else:
        outcome = "other_failure"  # e.g. SystemExit raised by the candidate before tests finished
        detail = detail or f"exit_code={raw.get('exit_code')} error={error}"
    return {"outcome": outcome, "passed": outcome == "passed", "syntax_valid": syntax_valid,
            "wall_seconds": raw.get("wall_seconds", 0.0), "detail": detail}


def evaluate(jobs: list[tuple[dict[str, Any], str]], timeout: float = DEFAULT_TIMEOUT,
             workers: int | None = None, progress: Callable[[int, int], None] | None = None
             ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Score (task, completion) jobs, deduplicating identical programs per task."""
    workers = workers or default_workers()
    keys, unique = [], {}
    for task, completion in jobs:
        program = task["prompt"] + completion
        try:
            program.encode("utf-8")
        except UnicodeEncodeError:
            program = None
        key = (task["task_id"], program)
        keys.append(key)
        if key not in unique:
            unique[key] = (task, program)
    results: dict[Any, dict[str, Any]] = {}
    pending = []
    for key, (task, program) in unique.items():
        if program is None:
            results[key] = {"outcome": "syntax_error", "passed": False, "syntax_valid": False,
                            "wall_seconds": 0.0, "detail": "completion is not encodable as UTF-8"}
        else:
            pending.append(key)
    # Batched compile pass: syntax errors never need their own guest.
    programs = [unique[key][1] for key in pending]
    chunks = [list(range(start, min(start + SYNTAX_CHUNK_FILES, len(programs))))
              for start in range(0, len(programs), SYNTAX_CHUNK_FILES)]
    verdicts: list[bool | None] = [None] * len(programs)

    def syntax_chunk(indices: list[int]) -> None:
        for index, verdict in zip(indices, check_syntax([programs[i] for i in indices])):
            verdicts[index] = verdict

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(syntax_chunk, chunks))
    to_run = []
    for key, verdict in zip(pending, verdicts):
        if verdict is False:
            results[key] = {"outcome": "syntax_error", "passed": False, "syntax_valid": False,
                            "wall_seconds": 0.0, "detail": "compile failed in guest"}
        else:
            to_run.append(key)
    done, lock = 0, threading.Lock()

    def execute(key: Any) -> None:
        nonlocal done
        task, program = unique[key]
        results[key] = run_program(program, task["test_code"], timeout)
        with lock:
            done += 1
            if progress:
                progress(done, len(to_run))

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(execute, to_run))
    stats = {"samples": len(jobs), "unique_programs": len(unique),
             "syntax_rejected_without_execution": len(unique) - len(to_run),
             "guest_executions": len(to_run)}
    return [dict(results[key]) for key in keys], stats


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 2))


# --- scoring ---------------------------------------------------------------------

def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 6) if values else None


def score(suite: str, samples: list[dict[str, Any]], k_values: Iterable[int] = DEFAULT_K,
          timeout: float = DEFAULT_TIMEOUT, workers: int | None = None, allow_partial: bool = False,
          progress: Callable[[int, int], None] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score samples ({task_id, completion, latency_s?, num_tokens?}) for one suite."""
    tasks = {task["task_id"]: task for task in load_suite(suite)}
    jobs = []
    for number, sample in enumerate(samples):
        if not isinstance(sample, dict) or not isinstance(sample.get("completion"), str):
            raise ValueError(f"Sample {number} needs a string 'completion'")
        if sample.get("task_id") not in tasks:
            raise ValueError(f"Sample {number} has unknown task_id {sample.get('task_id')!r} for suite {suite}")
        for field in ("latency_s", "num_tokens"):
            value = sample.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError(f"Sample {number} has invalid {field}")
        jobs.append((tasks[sample["task_id"]], sample["completion"]))
    covered = {sample["task_id"] for sample in samples}
    missing = sorted(set(tasks) - covered)
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} tasks have no samples (e.g. {missing[:3]}); pass allow_partial")
    started = time.monotonic()
    results, stats = evaluate(jobs, timeout=timeout, workers=workers, progress=progress)
    rows, per_task = [], {}
    for sample, result in zip(samples, results):
        task_id = sample["task_id"]
        entry = per_task.setdefault(task_id, {"n": 0, "c": 0})
        entry["n"] += 1
        entry["c"] += int(result["passed"])
        rows.append({"task_id": task_id, "sample_index": entry["n"] - 1, **result,
                     **{field: sample[field] for field in ("latency_s", "num_tokens") if field in sample}})
    min_n = min(entry["n"] for entry in per_task.values())
    pass_at = {}
    for k in sorted(set(k_values)):
        if k <= min_n:
            pass_at[f"pass@{k}"] = round(statistics.fmean(
                pass_at_k(entry["n"], entry["c"], k) for entry in per_task.values()), 6)
    outcomes = {name: 0 for name in OUTCOMES}
    for row in rows:
        outcomes[row["outcome"]] += 1
    syntax_known = [row["syntax_valid"] for row in rows if row["syntax_valid"] is not None]
    latencies = [float(s["latency_s"]) for s in samples if s.get("latency_s") is not None]
    rates = [s["num_tokens"] / s["latency_s"] for s in samples
             if s.get("num_tokens") is not None and s.get("latency_s")]
    report = {
        "result_type": "code_bench_score", "benchmark": BENCH_VERSION, "suite": suite,
        "suite_sha256": load_manifest()["suites"][suite]["sha256"],
        "valid": outcomes["infra_error"] == 0 and not missing,
        "tasks_in_suite": len(tasks), "tasks_scored": len(per_task), "tasks_missing": len(missing),
        "samples": len(rows), "samples_per_task": {"min": min_n, "max": max(e["n"] for e in per_task.values())},
        **pass_at,
        "pass_at_k_omitted": [k for k in sorted(set(k_values)) if k > min_n],
        "syntax_valid_rate": round(sum(syntax_known) / len(syntax_known), 6) if syntax_known else None,
        "sample_pass_rate": round(sum(row["passed"] for row in rows) / len(rows), 6) if rows else None,
        "outcomes": outcomes,
        "generation_latency_s": {"count": len(latencies), "median": _median(latencies),
                                 "mean": round(statistics.fmean(latencies), 6) if latencies else None},
        "generation_tokens_per_s": {"count": len(rates), "median": _median(rates)},
        "execution": {**stats, "timeout_s": timeout, "workers": workers or default_workers(),
                      "median_guest_wall_s": _median([r["wall_seconds"] for r in rows if r["wall_seconds"]]),
                      "scoring_wall_s": round(time.monotonic() - started, 2)},
        "per_task": per_task,
    }
    return report, rows


# --- self-check ------------------------------------------------------------------

def broken_candidates(task: dict[str, Any]) -> dict[str, str]:
    """Known-bad completions; each must be rejected by the task's tests."""
    if task["source"] == "humaneval":  # completions continue an open function body
        return {"empty_body": "", "pass": "    pass\n", "return_none": "    return None\n",
                "syntax_error": "    return (\n", "infinite_loop": "    while True:\n        pass\n"}
    head = f"def {task['entry_point']}(*args, **kwargs):\n"
    return {"empty_body": "", "pass": head + "    pass\n", "return_none": head + "    return None\n",
            "syntax_error": head + "    return (\n", "infinite_loop": head + "    while True:\n        pass\n"}


def self_check(suites: Iterable[str] = SUITES, timeout: float = DEFAULT_TIMEOUT,
               loop_tasks_per_suite: int = 2, loop_timeout: float = 8.0, workers: int | None = None,
               progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    estimator = {
        "n10_c0_k1": pass_at_k(10, 0, 1), "n10_c10_k5": pass_at_k(10, 10, 5),
        "n10_c3_k1": pass_at_k(10, 3, 1), "n5_c1_k2": pass_at_k(5, 1, 2),
    }
    estimator_ok = (estimator["n10_c0_k1"] == 0.0 and estimator["n10_c10_k5"] == 1.0
                    and abs(estimator["n10_c3_k1"] - 0.3) < 1e-12 and abs(estimator["n5_c1_k2"] - 0.4) < 1e-12)
    report: dict[str, Any] = {"result_type": "code_bench_self_check", "benchmark": BENCH_VERSION,
                              "manifest_sha256": sha256_file(MANIFEST),
                              "scorer_sha256": sha256_file(Path(__file__)),
                              "frozen_files_reproducible": not check_frozen(),
                              "estimator": {**estimator, "passed": estimator_ok},
                              "timeout_s": timeout, "suites": {}}
    for suite in suites:
        tasks = load_suite(suite)
        jobs, labels = [], []
        for task in tasks:
            jobs.append((task, task["canonical_completion"]))
            labels.append((task["task_id"], "canonical"))
            for kind, completion in broken_candidates(task).items():
                if kind != "infinite_loop":
                    jobs.append((task, completion))
                    labels.append((task["task_id"], kind))
        results, stats = evaluate(jobs, timeout=timeout, workers=workers, progress=progress)
        rows: dict[str, dict[str, Any]] = {}
        anomalies = []
        for (task_id, kind), result in zip(labels, results):
            row = rows.setdefault(task_id, {})
            row[kind] = result["outcome"]
            if kind == "canonical":
                row["canonical_wall_s"] = result["wall_seconds"]
            expected_ok = (result["passed"] if kind == "canonical" else
                           result["outcome"] == "syntax_error" if kind == "syntax_error" else
                           result["outcome"] in ("candidate_error", "test_failed", "test_error"))
            if not expected_ok:
                anomalies.append({"task_id": task_id, "candidate": kind, **result})
        loops = []
        for task in tasks[:loop_tasks_per_suite]:
            result = run_program(task["prompt"] + broken_candidates(task)["infinite_loop"],
                                 task["test_code"], loop_timeout)
            ok = result["outcome"] in ("timeout", "fuel_exhausted") and result["wall_seconds"] <= loop_timeout + 5
            loops.append({"task_id": task["task_id"], "passed": ok, **result})
        walls = [row["canonical_wall_s"] for row in rows.values()]
        summary = {"tasks": len(tasks),
                   "canonical_passed": sum(row["canonical"] == "passed" for row in rows.values())}
        for kind in ("empty_body", "pass", "return_none", "syntax_error"):
            summary[f"{kind}_rejected"] = sum(row[kind] != "passed" for row in rows.values())
        report["suites"][suite] = {
            "passed": not anomalies and all(loop["passed"] for loop in loops),
            "summary": summary, "anomalies": anomalies, "timeout_checks": loops,
            "canonical_wall_s": {"median": _median(walls), "max": max(walls)},
            "execution": stats, "tasks": rows,
        }
    report["passed"] = (estimator_ok and report["frozen_files_reproducible"]
                        and all(item["passed"] for item in report["suites"].values()))
    report["wall_s"] = round(time.monotonic() - started, 1)
    return report


# --- CLI -------------------------------------------------------------------------

def _read_samples(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _progress(label: str) -> Callable[[int, int], None]:
    last = [0.0]

    def report(done: int, total: int) -> None:
        now = time.monotonic()
        if done == total or now - last[0] > 10:
            last[0] = now
            print(f"[{label}] {done}/{total} guest runs", file=sys.stderr, flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m zero_baseline.code_bench", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="derive tasks/*.jsonl and the manifest from raw files")
    build.add_argument("--check", action="store_true", help="verify the frozen files reproduce exactly")
    build.add_argument("--force", action="store_true", help="overwrite an existing frozen manifest")
    sub.add_parser("list", help="show suites and task counts")
    prompts = sub.add_parser("prompts", help="write {task_id, prompt} JSONL for generation")
    prompts.add_argument("--suite", required=True, choices=SUITES)
    prompts.add_argument("--out", type=Path, required=True)
    scorer = sub.add_parser("score", help="score a samples JSONL file")
    scorer.add_argument("--suite", required=True, choices=SUITES)
    scorer.add_argument("--samples", type=Path, required=True,
                        help="JSONL lines: {task_id, completion, latency_s?, num_tokens?}")
    scorer.add_argument("--out", type=Path, required=True, help="report JSON path")
    scorer.add_argument("--per-sample", type=Path, help="optional per-sample JSONL path")
    scorer.add_argument("--k", default=",".join(map(str, DEFAULT_K)), help="comma-separated k values")
    scorer.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    scorer.add_argument("--workers", type=int, default=None)
    scorer.add_argument("--allow-partial", action="store_true")
    check = sub.add_parser("self-check", help="canonical solutions pass, broken candidates fail")
    check.add_argument("--suite", action="append", choices=SUITES, help="repeatable; default all")
    check.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    check.add_argument("--workers", type=int, default=None)
    check.add_argument("--out", type=Path, default=BENCH / "reports" / "self-check.json")
    args = parser.parse_args(argv)

    if args.command == "build":
        if args.check:
            problems = check_frozen()
            print(json.dumps({"reproducible": not problems, "problems": problems}, indent=2))
            return 1 if problems else 0
        manifest = write_frozen(force=args.force)
        print(json.dumps(manifest["suites"], indent=2))
        return 0
    if args.command == "list":
        print(json.dumps(load_manifest()["suites"], indent=2))
        return 0
    if args.command == "prompts":
        rows = [{"task_id": t["task_id"], "prompt": t["prompt"], "entry_point": t["entry_point"]}
                for t in load_suite(args.suite)]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(_jsonl(rows))
        print(f"wrote {len(rows)} prompts to {args.out}")
        return 0
    if args.command == "score":
        k_values = [int(k) for k in args.k.split(",") if k.strip()]
        report, rows = score(args.suite, _read_samples(args.samples), k_values, args.timeout,
                             args.workers, args.allow_partial, _progress("score"))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.per_sample:
            args.per_sample.parent.mkdir(parents=True, exist_ok=True)
            args.per_sample.write_bytes(_jsonl(rows))
        print(json.dumps({key: value for key, value in report.items() if key != "per_task"}, indent=2))
        return 0 if report["valid"] else 2
    report = self_check(args.suite or SUITES, args.timeout, workers=args.workers,
                        progress=_progress("self-check"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "wall_s": report["wall_s"],
                      **{name: {"passed": s["passed"], **s["summary"], "anomalies": len(s["anomalies"])}
                         for name, s in report["suites"].items()}}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
