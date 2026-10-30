"""Run candidate Python inside pinned CPython/WASI, never native Python.

The trusted host worker runs in a separate process. Its WASI guest receives only
read-only copied fixture files and the pinned standard library, no inherited
environment/stdio, and no networking or custom host functions. Wasmtime is a
software sandbox, not a VM; runtime defects remain outside this threat model.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".baseline-runtime"
WASMTIME_VERSION = "49.0.0"
CPYTHON_WASM_SHA256 = "9f58ad347987d0a23def7eefa5aa5e8ffaef529fba26979350c44376d6b8fae3"
MEMORY_LIMIT = 256 * 1024 * 1024
OUTPUT_LIMIT = 64 * 1024  # combined stdout and stderr, in bytes
INPUT_LIMIT = 2 * 1024 * 1024
FILE_LIMIT = 64
# CPython startup, Decimal imports and normal finalization exceed 2B fuel on
# CODE-010 even after its assertions pass. The wall deadline remains primary.
FUEL_LIMIT = 10_000_000_000
ENTRY = "_zero_entry.py"


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Sandbox path escapes its fixed root")
    return resolved


def _runtime_paths() -> dict[str, str]:
    distribution = RUNTIME / "cpython"
    wasm = sorted(distribution.rglob("*.wasm")) if distribution.is_dir() else []
    stdlibs = sorted(distribution.glob("**/lib/python3.*/encodings/__init__.py"))
    if len(wasm) != 1 or len(stdlibs) != 1:
        raise FileNotFoundError("Expected one pinned CPython WASI module and standard library under .baseline-runtime/cpython")
    python = RUNTIME / "python" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise FileNotFoundError("The isolated .baseline-runtime/python interpreter is missing")
    stdlib = stdlibs[0].parent.parent
    return {"python": str(_inside(python, RUNTIME)),
            "wasm": str(_inside(wasm[0], distribution)),
            "stdlib": str(_inside(stdlib, distribution))}


def runtime_status() -> dict:
    """Inspect local paths only; this does not install or execute a runtime."""
    try:
        return {"available": True, "wasmtime_required": WASMTIME_VERSION, **_runtime_paths()}
    except (OSError, ValueError) as exc:
        return {"available": False, "error": str(exc), "wasmtime_required": WASMTIME_VERSION}


def _validate(source: str, files: dict[str, str], timeout: float) -> None:
    if not isinstance(source, str) or not isinstance(files, dict):
        raise ValueError("Source and fixture files must be text")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("Timeout must be between zero and 60 seconds")
    if len(files) > FILE_LIMIT:
        raise ValueError("Too many fixture files")
    total = len(source.encode("utf-8"))
    normalized = set()
    for name, content in files.items():
        if not isinstance(name, str) or not isinstance(content, str) or not name:
            raise ValueError("Fixture names and contents must be strings")
        posix, windows = PurePosixPath(name), PureWindowsPath(name)
        if (posix.is_absolute() or windows.is_absolute() or windows.drive or "\\" in name
                or any(part in ("", ".", "..") for part in name.split("/"))
                or any(char in name for char in '\x00:<>|?*\"')
                or any(part.endswith((".", " ")) for part in posix.parts)
                or any(PureWindowsPath(part).is_reserved() for part in posix.parts)):
            raise ValueError("Fixture paths must be ordinary relative POSIX paths")
        key = name.casefold()
        if key == ENTRY.casefold() or key in normalized:
            raise ValueError("Duplicate or reserved fixture path")
        normalized.add(key)
        total += len(content.encode("utf-8"))
    if total > INPUT_LIMIT:
        raise ValueError("Sandbox input exceeds its byte limit")


def _base_result() -> dict:
    return {"exit_code": None, "stdout": "", "stderr": "", "timed_out": False,
            "output_limited": False, "sandbox": "cpython-wasi-wasmtime",
            "memory_limit_bytes": MEMORY_LIMIT, "fuel_limit": FUEL_LIMIT,
            "output_limit_bytes": OUTPUT_LIMIT}


def run_python(source: str, files: dict[str, str], timeout: float = 10) -> dict:
    """Execute source in a fresh WASI guest over a read-only copy of files.

    ``source`` runs as __main__ with /workspace as its cwd. Files use relative
    POSIX paths. Results are observations, not a trustworthy grading verdict.
    The deadline includes worker startup and compilation. No native fallback.
    """
    _validate(source, files, timeout)
    result = _base_result()
    started = time.monotonic()
    try:
        paths = _runtime_paths()
    except (OSError, ValueError) as exc:
        return {**result, "error": "runtime_unavailable", "stderr": str(exc)}
    scratch = _inside(RUNTIME / "sandbox-runs", RUNTIME)
    scratch.mkdir(parents=True, exist_ok=True)
    # TemporaryDirectory removes only a newly generated child of this exact root.
    with tempfile.TemporaryDirectory(prefix="invoke-", dir=scratch) as temporary:
        run = _inside(Path(temporary), scratch)
        workspace = run / "workspace"
        workspace.mkdir()
        for name, content in {**files, ENTRY: source}.items():
            target = _inside(workspace / name, workspace)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="")
        job = run / "job.json"
        job.write_text(json.dumps({"workspace": str(workspace), "timeout": timeout,
                                   "wasm": paths["wasm"], "stdlib": paths["stdlib"]}), encoding="utf-8")
        env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
        env.update({"TEMP": str(run), "TMP": str(run), "TMPDIR": str(run)})
        if os.name == "nt":
            # Python's platform.machine() (used by Wasmtime's DLL loader) needs
            # this non-secret host metadata on Windows. It never enters WASI.
            # Use the interpreter build tag if our caller also sanitized its
            # environment; platform.machine() alone can then return empty.
            env["PROCESSOR_ARCHITECTURE"] = {
                "win-amd64": "AMD64", "win-arm64": "ARM64"
            }.get(sysconfig.get_platform(), platform.machine())
        command = [paths["python"], "-I", "-B", str(Path(__file__).resolve()), "--worker", str(job)]
        try:
            completed = subprocess.run(command, cwd=run, env=env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            result.update(timed_out=True, error="wall_clock_limit", stderr="Sandbox wall-clock limit exceeded")
        except OSError as exc:
            result.update(error="worker_start_failed", stderr=str(exc))
        else:
            # Candidate output cannot write this pipe: it is captured by bounded WASI callbacks.
            try:
                payload = json.loads(completed.stdout)
                if not isinstance(payload, dict) or "exit_code" not in payload:
                    raise ValueError("Invalid worker result")
                result.update(payload)
            except (ValueError, UnicodeError):
                result.update(error="worker_failed", stderr=completed.stderr.decode("utf-8", "replace")[-4000:])
            if completed.returncode != 0:
                result.update(exit_code=None, error="worker_failed")
        result["wall_seconds"] = round(time.monotonic() - started, 4)
    return result


def _module(wasmtime, engine, wasm_path: Path):
    """Only deserialize caches created from the pinned, host-owned module."""
    digest = hashlib.sha256(wasm_path.read_bytes()).hexdigest()
    if digest != CPYTHON_WASM_SHA256:
        raise RuntimeError("Unexpected CPython WASI module digest; refusing execution")
    key = hashlib.sha256((digest + WASMTIME_VERSION + "fuel-epoch-opt0-v1").encode()).hexdigest()
    cache = _inside(RUNTIME / "wasmtime-cache", RUNTIME)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (key + ".cwasm")
    if target.is_file() and not target.is_symlink():
        try:
            return wasmtime.Module.deserialize_file(engine, str(target))
        except wasmtime.WasmtimeError:
            pass  # A runtime/architecture mismatch requires a fresh trusted compilation.
    module = wasmtime.Module.from_file(engine, str(wasm_path))
    # Publish atomically so concurrent test workers cannot observe partial native code.
    handle, temporary = tempfile.mkstemp(prefix=key + "-", suffix=".tmp", dir=cache)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(module.serialize())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return module


def _worker(job_path: Path) -> dict:
    from importlib.metadata import version
    if version("wasmtime") != WASMTIME_VERSION:
        raise RuntimeError("Unexpected Wasmtime version; refusing unverified runtime APIs")
    import wasmtime

    job_path = _inside(job_path, RUNTIME / "sandbox-runs")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    workspace = _inside(Path(job["workspace"]), job_path.parent)
    wasm_path = _inside(Path(job["wasm"]), RUNTIME / "cpython")
    stdlib = _inside(Path(job["stdlib"]), RUNTIME / "cpython")
    result = _base_result()
    config = wasmtime.Config()
    config.consume_fuel = True
    config.epoch_interruption = True
    config.cranelift_opt_level = "none"
    config.wasm_threads = False
    engine = wasmtime.Engine(config)
    module = _module(wasmtime, engine, wasm_path)
    # A fixed CPython WASIp1 module gets only standard WASI imports.
    if any(item.module != "wasi_snapshot_preview1" for item in module.imports):
        raise RuntimeError("Unexpected host imports in pinned WASI Python")
    store = wasmtime.Store(engine)
    store.set_limits(memory_size=MEMORY_LIMIT, instances=1, memories=1, tables=4,
                     table_elements=200_000)
    store.set_fuel(FUEL_LIMIT)
    store.set_epoch_deadline(1)
    wasi = wasmtime.WasiConfig()
    # CPython -I would discard this explicit guest-only PYTHONHOME. -S/-P remove
    # site and implicit cwd loading; the only additional import root is our copy.
    wasi.env = [("PYTHONHOME", "/python"), ("PYTHONHASHSEED", "0")]
    bootstrap = ("import os,runpy,sys; os.chdir('/workspace'); "
                 "sys.path.insert(0,'/workspace'); "
                 "runpy.run_path('/workspace/" + ENTRY + "',run_name='__main__')")
    wasi.argv = ["python", "-B", "-S", "-P", "-X", "utf8", "-c", bootstrap]
    wasi.preopen_dir(str(stdlib), "/python/lib/" + stdlib.name, fs_mutable=False)
    wasi.preopen_dir(str(workspace), "/workspace", fs_mutable=False)
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    used = 0

    def output(kind):
        def receive(data):
            nonlocal used
            remaining = max(0, OUTPUT_LIMIT - used)
            captured[kind].extend(data[:remaining])
            used += min(len(data), remaining)
            if len(data) > remaining:
                result["output_limited"] = True
                engine.increment_epoch()
                return -errno.EIO
            return len(data)
        return receive

    wasi.stdout_custom = output("stdout")
    wasi.stderr_custom = output("stderr")
    store.set_wasi(wasi)
    linker = wasmtime.Linker(engine)
    linker.define_wasi()
    expired = threading.Event()

    def deadline():
        expired.set()
        engine.increment_epoch()

    timer = threading.Timer(job["timeout"], deadline)
    timer.daemon = True
    timer.start()
    try:
        instance = linker.instantiate(store, module)
        instance.exports(store)["_start"](store)
        result["exit_code"] = 0
    except wasmtime.ExitTrap as exc:
        result["exit_code"] = exc.code
    except wasmtime.Trap as exc:
        result.update(exit_code=1, error="wasm_trap", trap=str(exc)[-2000:])
    finally:
        timer.cancel()
    if expired.is_set():
        result.update(exit_code=None, timed_out=True, error="wall_clock_limit")
    # Replacement characters can expand invalid UTF-8. Bound returned UTF-8
    # text as well as raw captured bytes, so malformed output cannot inflate it.
    remaining = OUTPUT_LIMIT
    for name, data in captured.items():
        encoded = bytes(data).decode("utf-8", "replace").encode("utf-8")
        if len(encoded) > remaining:
            result["output_limited"] = True
        result[name] = encoded[:remaining].decode("utf-8", "ignore")
        remaining -= len(result[name].encode("utf-8"))
    if result["output_limited"]:
        result.update(exit_code=1, error="output_limit")
    result["fuel_remaining"] = store.get_fuel()
    result["wasmtime_version"] = WASMTIME_VERSION
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Internal sandbox worker; use run_python()")
    try:
        payload = _worker(Path(sys.argv[2]))
    except Exception as exc:
        payload = {**_base_result(), "error": "sandbox_setup_failed", "stderr": str(exc)[-4000:]}
    print(json.dumps(payload, ensure_ascii=True), flush=True)
