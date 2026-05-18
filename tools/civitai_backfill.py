#!/usr/bin/env python3
"""Backfill Civitai-compatible PNG metadata on a folder of existing
generations.

Use case: you have a backlog of PNGs from before you wired the
GrimmRibbity Civitai Save node into your workflow (or before it learned
to compute SHA hashes). The PNGs already carry the workflow JSON in
their `prompt` chunk (ComfyUI writes that automatically). This tool:

  1. Walks the target directory for *.png
  2. For each file, reads the embedded workflow trace
  3. Pulls model_label + loras + positive/negative/seed via
     civitai_save.extract_workflow_metadata
  4. Computes SHA256 hashes for the model + each LoRA (cached on disk
     in data/hash_cache.json — same cache the live save node uses)
  5. Rebuilds the A1111 `parameters` block with proper Hashes: JSON
  6. Rewrites the PNG with the updated parameters chunk

Files that already carry a parameters chunk with a Hashes: block are
skipped — re-running the tool over the same folder is idempotent.

Usage:
    python3 tools/civitai_backfill.py /path/to/output [--dry-run]
                                                       [--recursive]
                                                       [--limit N]
                                                       [--verbose]

Designed to run independently of ComfyUI (folder_paths is consulted
when available for resolving model paths, but not required — pass
--models-root and/or --loras-root to override).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

# Run from the repo root or via `python3 -m tools.civitai_backfill`. We
# add the parent (the package root) to sys.path so civitai_save imports
# work whichever the user picked.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from PIL import Image, PngImagePlugin
except ImportError:
    print("error: Pillow is required (pip install Pillow)", file=sys.stderr)
    sys.exit(2)

from civitai_save import (  # noqa: E402
    _to_latin1_safe,
    build_a1111_parameters,
    extract_workflow_metadata,
    get_cached_sha256,
    resolve_lora_path,
    resolve_model_path,
)


def _iter_pngs(root: Path, recursive: bool) -> Iterator[Path]:
    pattern = "**/*.png" if recursive else "*.png"
    for p in sorted(root.glob(pattern)):
        if p.is_file():
            yield p


def _load_workflow(png: Path) -> dict | None:
    """Read the `prompt` chunk that ComfyUI writes into PNGs."""
    try:
        with Image.open(png) as img:
            text = (img.info.get("prompt") or "").strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _existing_has_hashes(png: Path) -> bool:
    """True if the PNG already carries a parameters chunk with a Hashes:
    JSON block — used to skip already-processed files."""
    try:
        with Image.open(png) as img:
            params = (img.info.get("parameters") or "")
    except Exception:
        return False
    return "Hashes:" in params and "model" in params


def _resolve_size(workflow: dict) -> tuple[int, int]:
    """Best-effort width/height extraction from an EmptyLatentImage or
    similar latent-init node. Falls back to (0, 0) — the resulting
    parameters block will still be valid."""
    for node in workflow.values():
        ct = (node or {}).get("class_type", "")
        ins = (node or {}).get("inputs") or {}
        if ct in ("EmptyLatentImage", "EmptySD3LatentImage", "EmptyLatentImageFromPresetsSDXL"):
            w = ins.get("width") or ins.get("w") or 0
            h = ins.get("height") or ins.get("h") or 0
            try:
                w, h = int(w), int(h)
                if w and h:
                    return w, h
            except (TypeError, ValueError):
                pass
    return 0, 0


def _backfill_one(png: Path, *, dry_run: bool, verbose: bool) -> str:
    """Process a single PNG. Returns one of: 'updated', 'skipped' (already
    has hashes), 'no-workflow' (PNG has no Comfy prompt chunk), 'no-model'
    (workflow trace had no resolvable checkpoint), 'error'."""
    if _existing_has_hashes(png):
        if verbose:
            print(f"  skip {png.name} (already has Hashes:)")
        return "skipped"

    workflow = _load_workflow(png)
    if workflow is None:
        if verbose:
            print(f"  no-workflow {png.name}")
        return "no-workflow"

    meta = extract_workflow_metadata(workflow)
    model_label = meta.get("model_label")
    if not model_label:
        if verbose:
            print(f"  no-model {png.name}")
        return "no-model"

    model_resolved = resolve_model_path(model_label)
    if not model_resolved:
        if verbose:
            print(f"  unresolvable-model {png.name} ({model_label!r})")
        return "no-model"
    model_name, model_full_path = model_resolved
    model_sha = get_cached_sha256(model_label, model_full_path)

    loras: list[tuple[str, str | None, float]] = []
    for lname, strength in meta.get("loras", []):
        full = resolve_lora_path(lname)
        if not full:
            continue
        loras.append((lname, get_cached_sha256(lname, full), float(strength)))

    width, height = _resolve_size(workflow)
    if not (width and height):
        # Fall back to the actual image dimensions — accurate for any
        # ComfyUI output since ComfyUI writes the final image at the
        # generated resolution.
        try:
            with Image.open(png) as img:
                width, height = img.width, img.height
        except Exception:
            return "error"

    params = build_a1111_parameters(
        positive=meta.get("positive", ""),
        negative=meta.get("negative", ""),
        width=width, height=height,
        steps=meta.get("steps"),
        sampler_name=meta.get("sampler_name"),
        scheduler=meta.get("scheduler"),
        cfg=meta.get("cfg"),
        seed=meta.get("seed"),
        model_name=model_name,
        model_sha256=model_sha,
        loras=loras,
    )

    if dry_run:
        if verbose:
            print(f"  would update {png.name} ({len(loras)} LoRA hashes)")
        return "updated"

    # Rewrite: load the PNG, copy every existing tEXt chunk, replace
    # parameters with the rebuilt block, save under a temp name then
    # rename atomically. Workflow JSON ('prompt' / 'workflow' chunks)
    # must be preserved for re-loading the gen back into ComfyUI.
    try:
        with Image.open(png) as img:
            preserved = dict(img.info)
            png_info = PngImagePlugin.PngInfo()
            for key, value in preserved.items():
                if key == "parameters":
                    continue
                if isinstance(value, str):
                    png_info.add_text(key, value)
            png_info.add_text("parameters", _to_latin1_safe(params))
            tmp = png.with_suffix(".png.tmp")
            img.save(tmp, format="PNG", pnginfo=png_info, compress_level=4)
        tmp.replace(png)
    except Exception as e:
        print(f"  error {png.name}: {e}")
        return "error"
    if verbose:
        print(f"  updated {png.name} ({len(loras)} LoRA hashes)")
    return "updated"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="directory of PNGs to process")
    p.add_argument("--recursive", action="store_true",
                    help="walk subdirectories")
    p.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing files")
    p.add_argument("--limit", type=int, default=0,
                    help="stop after processing N files (0 = no limit)")
    p.add_argument("--verbose", action="store_true",
                    help="log each file's outcome")
    args = p.parse_args()

    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 2

    counts = {"updated": 0, "skipped": 0, "no-workflow": 0,
              "no-model": 0, "error": 0}
    started = time.monotonic()
    processed = 0

    for png in _iter_pngs(args.root, args.recursive):
        outcome = _backfill_one(png, dry_run=args.dry_run, verbose=args.verbose)
        counts[outcome] = counts.get(outcome, 0) + 1
        processed += 1
        if args.limit and processed >= args.limit:
            break

    elapsed = time.monotonic() - started
    print(f"\nProcessed {processed} file(s) in {elapsed:.1f}s")
    for k in ("updated", "skipped", "no-workflow", "no-model", "error"):
        if counts[k]:
            print(f"  {k:<12} {counts[k]}")
    if args.dry_run:
        print("\n(dry run — no files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
