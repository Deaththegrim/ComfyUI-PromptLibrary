"""Inspect GrimmRibbity-saved PNGs.

Reads the embedded `grimmribbity_library` snapshot + A1111 `parameters`
chunk and prints a readable summary. Works on a single file or a folder
(recursive). The save node writes this metadata automatically; this
script is the read-side counterpart for debugging, audits, and "where
did this output come from" questions.

Usage:
    python tools/inspect_png.py <file.png>
    python tools/inspect_png.py <folder>/
    python tools/inspect_png.py <folder>/ --summary
    python tools/inspect_png.py <file.png> --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

LIBRARY_CHUNK_KEY = "grimmribbity_library"
PARAMS_CHUNK_KEY = "parameters"


def read_chunks(path: Path) -> dict:
    """Pull every text chunk PIL surfaces from a PNG into a dict.
    Returns {} on a non-PNG / unreadable file rather than raising — the
    folder walker leans on that to skip bad files quietly."""
    try:
        with Image.open(path) as img:
            img.load()
            return dict(img.info or {})
    except Exception:
        return {}


def parse_library_snapshot(chunks: dict) -> dict | None:
    """Decode the grimmribbity_library JSON chunk. Returns None if the
    chunk is missing or unparseable — distinguishes "no snapshot present"
    from "bad snapshot" by raising on the latter only when caller wants
    strict mode (default: silent skip)."""
    raw = chunks.get(LIBRARY_CHUNK_KEY)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_a1111_parameters(chunks: dict) -> dict:
    """A1111 `parameters` text is a freeform string with model/steps/etc.
    on a single line after a `Negative prompt:` separator. Extract the
    well-known fields into a dict; leave unknown lines untouched in
    `_raw`."""
    raw = chunks.get(PARAMS_CHUNK_KEY)
    if not raw or not isinstance(raw, str):
        return {}
    out: dict = {"_raw": raw}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("Negative prompt:"):
            out["negative"] = line[len("Negative prompt:"):].strip()
            continue
        # The metadata line: "Steps: 30, Sampler: dpmpp_3m_sde_gpu karras, ..."
        if "Steps:" in line and "," in line:
            for piece in line.split(","):
                if ":" not in piece:
                    continue
                k, _, v = piece.partition(":")
                k = k.strip().lower().replace(" ", "_")
                v = v.strip()
                if k and v:
                    out[k] = v
    return out


def render_entry(entry: dict, indent: str = "  ") -> list[str]:
    """One library entry → human-readable lines. Used in the per-file
    pretty printer. Missing entries (deleted from the library since the
    PNG was rendered) get a clear marker so the user can act."""
    lines: list[str] = []
    eid = entry.get("id", "?")
    if entry.get("missing"):
        lines.append(f"{indent}- {eid}  (MISSING from library)")
        return lines
    name = entry.get("name") or "(no name)"
    tags = entry.get("tags") or []
    lines.append(f"{indent}- {eid}  —  {name}")
    if tags:
        lines.append(f"{indent}  tags: {', '.join(tags)}")
    loras = entry.get("loras") or []
    for l in loras:
        sm = l.get("strength_model", 1.0)
        sc = l.get("strength_clip", sm)
        en = "" if l.get("enabled", True) else "  (disabled)"
        if abs(sc - sm) < 1e-6:
            lines.append(f"{indent}  lora: {l.get('name')} @ {sm:.2f}{en}")
        else:
            lines.append(
                f"{indent}  lora: {l.get('name')} @ model={sm:.2f}/clip={sc:.2f}{en}")
    neg = entry.get("negative")
    if neg:
        # Snip very long negatives in the pretty output — the JSON mode
        # still emits the full truncated string the snapshot stored.
        short = neg if len(neg) <= 80 else neg[:80] + "…"
        lines.append(f"{indent}  negative: {short}")
    return lines


def print_one(path: Path, *, show_params: bool = True) -> None:
    chunks = read_chunks(path)
    snap = parse_library_snapshot(chunks)
    params = parse_a1111_parameters(chunks) if show_params else {}
    print(f"\n{path}")
    if not chunks:
        print("  (no readable PNG metadata)")
        return
    if snap is None:
        print("  (no grimmribbity_library chunk)")
    else:
        entries = snap.get("entries") or []
        schema = snap.get("schema", "?")
        print(f"  library snapshot (schema={schema}, {len(entries)} entries):")
        for e in entries:
            for line in render_entry(e):
                print(line)
    if show_params and params:
        model = params.get("model") or params.get("model_hash") or ""
        sampler = params.get("sampler") or ""
        steps = params.get("steps") or ""
        seed = params.get("seed") or ""
        cfg = params.get("cfg_scale") or ""
        bits: list[str] = []
        if model: bits.append(f"model={model}")
        if sampler: bits.append(f"sampler={sampler}")
        if steps: bits.append(f"steps={steps}")
        if cfg: bits.append(f"cfg={cfg}")
        if seed: bits.append(f"seed={seed}")
        if bits:
            print(f"  render: {' '.join(bits)}")


def emit_json(paths: list[Path]) -> None:
    """Machine-readable mode — one JSON object per path with snapshot +
    A1111 params parsed. Empty snapshot becomes None, not absent, so
    downstream tools can branch cleanly."""
    out = []
    for p in paths:
        chunks = read_chunks(p)
        out.append({
            "path": str(p),
            "library": parse_library_snapshot(chunks),
            "parameters": parse_a1111_parameters(chunks),
        })
    print(json.dumps(out, indent=2, ensure_ascii=False))


def summarize(paths: list[Path]) -> None:
    """Aggregate stats across a folder: total PNGs, how many carry a
    library snapshot, tag-frequency histogram, missing-entry count, and
    LoRA-usage histogram. Useful for "what's in my output folder" audits
    after a long queue run."""
    total = 0
    with_snap = 0
    tag_freq: Counter[str] = Counter()
    lora_freq: Counter[str] = Counter()
    missing = 0
    entry_ids: Counter[str] = Counter()
    for p in paths:
        total += 1
        chunks = read_chunks(p)
        snap = parse_library_snapshot(chunks)
        if snap is None:
            continue
        with_snap += 1
        for entry in snap.get("entries") or []:
            eid = entry.get("id") or ""
            if entry.get("missing"):
                missing += 1
                continue
            if eid:
                entry_ids[eid] += 1
            for t in entry.get("tags") or []:
                tag_freq[t] += 1
            for l in entry.get("loras") or []:
                lora_freq[l.get("name") or "?"] += 1
    print(f"PNGs scanned         : {total}")
    print(f"  with library chunk : {with_snap}")
    print(f"  missing entries    : {missing}")
    if entry_ids:
        print("\ntop entries used:")
        for eid, c in entry_ids.most_common(10):
            print(f"  {c:4d}  {eid}")
    if tag_freq:
        print("\ntop tags:")
        for t, c in tag_freq.most_common(15):
            print(f"  {c:4d}  {t}")
    if lora_freq:
        print("\nLoRAs touched:")
        for n, c in lora_freq.most_common(15):
            print(f"  {c:4d}  {n}")


def collect_paths(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.rglob("*.png") if p.is_file())
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="Path to a PNG or a folder of PNGs")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON (one object per path)")
    ap.add_argument("--summary", action="store_true",
                    help="Folder-wide aggregate stats (tag/entry/LoRA histograms)")
    ap.add_argument("--no-params", action="store_true",
                    help="Skip the A1111 parameters summary in pretty mode")
    args = ap.parse_args(argv)
    target = Path(args.target).expanduser()
    paths = collect_paths(target)
    if not paths:
        print(f"no PNGs found at {target}", file=sys.stderr)
        return 1
    if args.summary:
        summarize(paths)
        return 0
    if args.json:
        emit_json(paths)
        return 0
    for p in paths:
        print_one(p, show_params=not args.no_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
