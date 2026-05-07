#!/usr/bin/env python3
"""Repair Smart Detailer widget-value shifts in saved workflow JSONs.

When new required widgets are added to a node, ComfyUI loads saved
workflows by POSITION — values shift off-by-N and validation fails. This
tool walks any GrimmRibbitySmartDetailer nodes in a workflow JSON,
detects stale widget_values lengths, and rewrites them to the current
widget order using sentinel-aware defaults.

  python3 tools/fix_workflow_widgets.py <workflow.json> [-o <out.json>]

By default, prints the patched JSON to stdout. With -o, writes in-place
to the given file (creating a `.bak` of the original first).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


# Current widget order for GrimmRibbitySmartDetailer (v0.52.1+). Order
# matches detailer_node.GrimmRibbitySmartDetailer.INPUT_TYPES return:
# required first, in declaration order; then optional, same. Keep in sync
# whenever you touch INPUT_TYPES order.
_CURRENT_WIDGETS = [
    # required
    ("enable_face",         "bool",   True),
    ("enable_eyes",         "bool",   True),
    ("enable_hands",        "bool",   False),
    ("enable_skin",         "bool",   False),
    ("bbox_face",           "str",    "(none)"),
    ("bbox_eyes",           "str",    "(none)"),
    ("bbox_hands",          "str",    "(none)"),
    ("sam_model",           "str",    "(none)"),
    ("seed",                "int",    0),
    ("steps",               "int",    25),
    ("cfg",                 "float",  6.0),
    ("sampler_name",        "str",    "dpmpp_3m_sde_gpu"),
    ("scheduler",           "str",    "karras"),
    ("denoise",             "float",  0.45),
    ("guide_size",          "int",    1024),
    ("max_size",            "int",    1536),
    ("bbox_threshold",      "float",  0.45),
    ("max_per_target",      "int",    0),
    ("tiled_decode",        "bool",   True),
    ("tiled_encode",        "bool",   False),
    ("mask_strength",       "float",  1.0),
    ("same_seed_per_target","bool",   False),
    ("bypass",              "bool",   False),
    # optional
    ("wildcard_prefix",     "str",    ""),
    ("face_threshold",      "float",  -1.0),
    ("face_denoise",        "float",  -1.0),
    ("face_max",            "int",    0),
    ("face_steps",          "int",    0),
    ("eyes_threshold",      "float",  -1.0),
    ("eyes_denoise",        "float",  -1.0),
    ("eyes_max",            "int",    0),
    ("eyes_steps",          "int",    0),
    ("hands_threshold",     "float",  -1.0),
    ("hands_denoise",       "float",  -1.0),
    ("hands_max",           "int",    0),
    ("hands_steps",         "int",    0),
    ("skin_threshold",      "float",  -1.0),
    ("skin_denoise",        "float",  -1.0),
    ("skin_max",            "int",    0),
    ("skin_steps",          "int",    0),
    ("force_inpaint",       "bool",   True),
    ("drop_size",           "int",    10),
    ("nms_iou",             "float",  0.5),
    ("yolo_imgsz",          "int",    960),
    ("max_bbox_area_pct",   "float",  0.95),
    ("draw_preview",        "bool",   True),
]


def _coerce(val, kind: str, default):
    """Best-effort cast a saved value to the expected type. Falls back to
    the default when the cast fails or the value clearly belongs to a
    different widget (e.g. a sampler name landing in the scheduler slot)."""
    if val is None:
        return default
    try:
        if kind == "bool":
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            if isinstance(val, str):
                return val.strip().lower() in ("true", "1", "yes", "on")
            return default
        if kind == "int":
            if isinstance(val, bool):
                return int(val)
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, str) and val.strip():
                return int(float(val))
            return default
        if kind == "float":
            if isinstance(val, bool):
                return float(val)
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str) and val.strip():
                return float(val)
            return default
        # str — accept anything stringifiable
        return str(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _patch_node(node: dict) -> tuple[bool, list]:
    """If `node` is a GrimmRibbitySmartDetailer, rebuild its widgets_values
    list in current widget order, coercing any salvageable value and
    falling back to defaults otherwise. Returns (changed, new_values)."""
    cls = node.get("type") or node.get("class_type")
    if cls != "GrimmRibbitySmartDetailer":
        return False, []
    old_values = node.get("widgets_values") or []
    new_values = []
    for i, (_name, kind, default) in enumerate(_CURRENT_WIDGETS):
        old = old_values[i] if i < len(old_values) else None
        new_values.append(_coerce(old, kind, default))
    return (new_values != old_values, new_values)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow", help="Path to a workflow JSON file")
    ap.add_argument("-o", "--in-place", action="store_true",
                     help="Write back to the same file (with .bak backup)")
    args = ap.parse_args()

    src = Path(args.workflow)
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 1
    data = json.loads(src.read_text("utf-8"))

    nodes = data.get("nodes", [])
    if not nodes and isinstance(data, dict):
        # API/exec format: prompt JSON has nodes keyed by id under top-level
        nodes = list(data.values())

    patched_count = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        changed, new_values = _patch_node(node)
        if changed:
            node["widgets_values"] = new_values
            patched_count += 1
            nid = node.get("id", "?")
            print(f"[fix] node #{nid} widgets_values rewritten "
                   f"({len(new_values)} entries)", file=sys.stderr)

    if patched_count == 0:
        print("[fix] no GrimmRibbitySmartDetailer nodes needed patching",
               file=sys.stderr)

    out_text = json.dumps(data, indent=2, ensure_ascii=False)
    if args.in_place:
        bak = src.with_suffix(src.suffix + ".bak")
        shutil.copy2(src, bak)
        src.write_text(out_text, encoding="utf-8")
        print(f"[fix] patched {patched_count} node(s); backup at {bak}",
               file=sys.stderr)
    else:
        sys.stdout.write(out_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
