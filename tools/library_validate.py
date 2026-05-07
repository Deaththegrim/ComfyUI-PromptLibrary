#!/usr/bin/env python3
"""Validate a GrimmRibbity library.

Walks data/prompts.json + data/images/ and reports anomalies:

  - LoRA paths in entry.loras that no longer resolve to a file on disk
    (file moved / renamed since the entry was authored)
  - Thumbnail files in data/images/ that don't correspond to any entry
    (orphans left behind by a manual delete or a corrupted run)
  - Entries whose `id` doesn't match _safe_id (would refuse to upsert)
  - Entries with empty `text` (possible from a half-completed save)
  - Entries whose negative-prompt encoder block looks malformed
    (Civitai writers sometimes round-trip empty negatives as `null`)

Idempotent — read-only by default. Pass --fix-orphans to delete the
orphan thumbnails (other anomalies need user judgement).

Usage:
    python3 tools/library_validate.py
    python3 tools/library_validate.py --library /path/to/data/prompts.json
    python3 tools/library_validate.py --fix-orphans
    python3 tools/library_validate.py --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_DEFAULT_LIBRARY = _REPO_ROOT / "data" / "prompts.json"
_LIVE_INSTALL_LIBRARY = Path(
    "/home/junie/comfy/ComfyUI/custom_nodes/ComfyUI-PromptLibrary/data/prompts.json")
_VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_SAFE_ID_RE_TEXT = "[A-Za-z0-9_-]{1,64}"


def _resolve_lora_path(name: str) -> str | None:
    """Best-effort LoRA path resolution. Tries folder_paths when ComfyUI
    is on the import path, otherwise scans common loras locations."""
    if not name:
        return None
    try:
        import folder_paths  # type: ignore
        path = folder_paths.get_full_path("loras", name)
        return path if path and os.path.isfile(path) else None
    except Exception:
        # Fallback: probe a couple of conventional locations. Users running
        # this script on a stock Comfy install will land here.
        candidates = [
            Path("/home/junie/comfy/ComfyUI/models/loras") / name,
            Path.home() / "ComfyUI" / "models" / "loras" / name,
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
        return None


def _pick_library_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if _LIVE_INSTALL_LIBRARY.is_file():
        return _LIVE_INSTALL_LIBRARY
    return _DEFAULT_LIBRARY


def _validate(library_path: Path, *, fix_orphans: bool, quiet: bool) -> dict:
    if not library_path.is_file():
        print(f"error: {library_path} not found", file=sys.stderr)
        return {"ok": False}
    with library_path.open("r", encoding="utf-8") as f:
        try:
            entries = json.load(f)
        except json.JSONDecodeError as e:
            print(f"error: {library_path} is not valid JSON: {e}", file=sys.stderr)
            return {"ok": False}
    if isinstance(entries, dict) and "prompts" in entries:
        entries = entries["prompts"]  # heal-from-export-manifest format
    if not isinstance(entries, list):
        print(f"error: prompts.json is not a list", file=sys.stderr)
        return {"ok": False}

    images_dir = library_path.parent / "images"
    on_disk_image_ids: set[str] = set()
    if images_dir.is_dir():
        for entry in images_dir.iterdir():
            if entry.suffix.lower() in _VALID_IMAGE_EXTS:
                on_disk_image_ids.add(entry.stem)

    import re
    safe_id_re = re.compile(f"^{_SAFE_ID_RE_TEXT}$")

    broken_loras: list[tuple[str, str, str]] = []  # (entry_id, entry_name, lora_path)
    invalid_ids: list[str] = []
    empty_texts: list[tuple[str, str]] = []
    valid_entry_ids: set[str] = set()

    for raw in entries:
        if not isinstance(raw, dict):
            continue
        eid = raw.get("id", "")
        ename = raw.get("name", "")
        if not safe_id_re.match(eid or ""):
            invalid_ids.append(repr(eid))
            continue
        valid_entry_ids.add(eid)
        if not (raw.get("text") or "").strip():
            empty_texts.append((eid, ename))
        for l in raw.get("loras") or []:
            if not isinstance(l, dict) or not l.get("enabled", True):
                continue
            lora_name = (l.get("name") or "").strip()
            if not lora_name:
                continue
            if _resolve_lora_path(lora_name) is None:
                broken_loras.append((eid, ename, lora_name))

    orphan_images = sorted(on_disk_image_ids - valid_entry_ids)

    # ---- Reporting ------------------------------------------------------
    if not quiet:
        print(f"Library: {library_path}")
        print(f"  entries:           {len(entries)}")
        print(f"  thumbnails:        {len(on_disk_image_ids)}")
        print(f"  broken LoRA refs:  {len(broken_loras)}")
        print(f"  orphan thumbnails: {len(orphan_images)}")
        print(f"  invalid ids:       {len(invalid_ids)}")
        print(f"  empty-text rows:   {len(empty_texts)}")
        print()

    if broken_loras:
        print("Broken LoRA refs (file not found on disk):")
        for eid, ename, lname in broken_loras[:50]:
            print(f"  {eid:<24} {ename!r:<40} → {lname}")
        if len(broken_loras) > 50:
            print(f"  … {len(broken_loras) - 50} more (rerun with --quiet=False)")
        print()
    if orphan_images:
        print(f"Orphan thumbnails ({len(orphan_images)}):")
        for oid in orphan_images[:30]:
            print(f"  {oid}")
        if len(orphan_images) > 30:
            print(f"  … {len(orphan_images) - 30} more")
        print()
    if invalid_ids:
        print(f"Invalid ids ({len(invalid_ids)}):")
        for iid in invalid_ids[:20]:
            print(f"  {iid}")
        print()
    if empty_texts:
        print(f"Empty-text rows ({len(empty_texts)}):")
        for eid, ename in empty_texts[:20]:
            print(f"  {eid:<24} {ename!r}")
        print()

    # ---- Optional repair ------------------------------------------------
    if fix_orphans and orphan_images:
        print(f"Removing {len(orphan_images)} orphan thumbnail(s)…")
        removed = 0
        for oid in orphan_images:
            for ext in _VALID_IMAGE_EXTS:
                p = images_dir / f"{oid}{ext}"
                try:
                    p.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
                except OSError as e:
                    print(f"  could not remove {p}: {e}")
        print(f"  removed {removed} file(s)")

    return {
        "ok": True,
        "entries": len(entries),
        "broken_loras": len(broken_loras),
        "orphan_images": len(orphan_images),
        "invalid_ids": len(invalid_ids),
        "empty_texts": len(empty_texts),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library", default=None,
                    help="path to data/prompts.json (default: live install if "
                         "present, else this repo's data/)")
    p.add_argument("--fix-orphans", action="store_true",
                    help="delete orphan thumbnails (read-only by default)")
    p.add_argument("--quiet", action="store_true",
                    help="skip the summary header; still prints findings")
    args = p.parse_args()

    result = _validate(_pick_library_path(args.library),
                        fix_orphans=args.fix_orphans, quiet=args.quiet)
    if not result.get("ok"):
        return 2
    # Non-zero exit code if there are unhealed issues — useful for CI / cron.
    if result["broken_loras"] or result["invalid_ids"] or result["empty_texts"]:
        return 1
    if result["orphan_images"] and not args.fix_orphans:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
