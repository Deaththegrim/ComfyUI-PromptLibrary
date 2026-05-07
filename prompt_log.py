"""Optional JSONL prompt-log for the GrimmRibbity samplers.

Each entry is one line of compact JSON:
    {"ts": "2026-05-07T08:14:23Z", "sampler": "GrimmRibbitySamplerSDXL",
     "model": "checkpoints/anima.safetensors",
     "loras": [{"name": "Anima.safetensors", "strength": 0.85}, ...],
     "positive": "...", "negative": "...",
     "seed": 12345, "steps": 25, "cfg": 7.0,
     "sampler_name": "dpmpp_2m", "scheduler": "karras",
     "batch_size": 1, "denoise": 1.0}

Reads as much as it can from the workflow PROMPT trace (via civitai_save's
extract_workflow_metadata) and lets the sampler's runtime kwargs override.
JSONL was chosen so an overnight batch produces one growing file the user
can grep / pipe through `jq` instead of a directory full of per-image txts.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

try:
    import folder_paths
except ImportError:
    folder_paths = None

from .civitai_save import extract_workflow_metadata


_log = logging.getLogger(__name__)
_write_lock = threading.Lock()

_DEFAULT_SUBDIR = "prompt_logs"
_DEFAULT_BASENAME = "prompts.jsonl"


def default_log_path() -> Path:
    """Resolve the default log path: <output_dir>/prompt_logs/prompts.jsonl.
    Falls back to CWD if folder_paths isn't available (test env)."""
    if folder_paths is not None:
        try:
            base = Path(folder_paths.get_output_directory())
        except Exception:
            base = Path.cwd() / "output"
    else:
        base = Path.cwd() / "output"
    return base / _DEFAULT_SUBDIR / _DEFAULT_BASENAME


def resolve_log_path(user_path: str) -> Path:
    """Honour an absolute path verbatim, expand ~, and treat a relative path
    as rooted at ComfyUI's output directory. Empty string → default."""
    s = (user_path or "").strip()
    if not s:
        return default_log_path()
    p = Path(os.path.expanduser(s))
    if p.is_absolute():
        return p
    if folder_paths is not None:
        try:
            return Path(folder_paths.get_output_directory()) / p
        except Exception:
            pass
    return Path.cwd() / "output" / p


def build_record(
    *,
    sampler_node: str,
    prompt_trace: dict | None,
    runtime: dict[str, Any],
) -> dict:
    """Assemble one log record. Workflow trace fills in model/loras/positive/
    negative; runtime kwargs (the sampler's actual call args) override every
    other field — the runtime is authoritative because the workflow JSON can
    be one queue ahead/behind the executing call (cached primitives, etc.)."""
    record: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sampler": sampler_node,
    }

    trace = extract_workflow_metadata(prompt_trace) if prompt_trace else {}

    # Workflow-trace fields (best-effort).
    if "model_label" in trace:
        record["model"] = trace["model_label"]
    if "loras" in trace and trace["loras"]:
        record["loras"] = [
            {"name": name, "strength": float(strength)}
            for name, strength in trace["loras"]
        ]
    if "positive" in trace:
        record["positive"] = trace["positive"]
    if "negative" in trace:
        record["negative"] = trace["negative"]

    # Runtime kwargs are authoritative for sampler params.
    for key in ("seed", "steps", "cfg", "sampler_name", "scheduler",
                 "denoise", "batch_size"):
        v = runtime.get(key)
        if v is None:
            continue
        # Coerce to JSON-friendly scalars; tensors/objects skipped.
        if isinstance(v, (int, float, str, bool)):
            record[key] = v
        else:
            try:
                record[key] = float(v) if isinstance(v, float) else int(v)
            except (TypeError, ValueError):
                continue

    return record


def append_prompt_log(record: dict, log_path: Path) -> None:
    """Append one JSONL line. Creates parent dirs as needed. Errors are
    logged but never raised — a logging glitch should never break sampling.

    Atomicity: the line is encoded to bytes once, then written via a single
    os.write(fd, bytes) call inside a flock'd file. POSIX guarantees that
    a single write() of <PIPE_BUF bytes is atomic; for larger lines, fsync
    after the write ensures the line is durable before the next call.
    Without this, a process crash mid-write could leave a truncated JSON
    line that breaks subsequent reads of the file."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _log.warning("prompt_log: mkdir %s failed: %s", log_path.parent, e)
        return
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with _write_lock:
            # O_APPEND + single write() is atomic for line-sized writes on
            # POSIX. fsync() after ensures durability so a crash leaves
            # either no line or the complete line — never a truncation.
            # Owner-only mode (0o600) — prompt logs may contain sensitive
            # text. CodeQL py/overly-permissive-file flags 0o644.
            fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line)
                try:
                    os.fsync(fd)
                except OSError:
                    # fsync can fail on some filesystems (e.g. tmpfs); the
                    # write itself still went through.
                    pass
            finally:
                os.close(fd)
    except OSError as e:
        _log.warning("prompt_log: write %s failed: %s", log_path, e)
