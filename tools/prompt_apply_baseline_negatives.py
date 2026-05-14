"""Bulk-apply the standard quality-gate negative to all human-rendering
prompts that currently have an empty negative field.

Reads the user's most-used negative (the same one wired into 354 prompts
already) and writes it to every other human prompt whose `negative` is
empty. Style/location prompts that don't render humans are left alone.

Always backs up first.

    cd ~/ComfyUI-PromptLibrary
    python3 tools/prompt_apply_baseline_negatives.py --dry-run
    python3 tools/prompt_apply_baseline_negatives.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
from pathlib import Path

LIBRARY = Path(__file__).resolve().parent.parent / "data" / "prompts.json"

# The most-used negative in the existing library (354 entries). Same
# quality-gate set as the cult-of-cards DEFAULT_NEGATIVE. Rejects build
# artifacts, anatomy errors, common SDXL/Pony failure modes.
BASELINE_NEGATIVE = (
    "low_quality, worst_quality, lowres, blurry, jpeg_artifacts, "
    "bad_quality, compression_artifacts, bad_anatomy, deformed, mutated, "
    "extra_limbs, extra_arms, extra_legs, malformed_limbs, "
    "disproportionate_body, broken_pose, twisted_torso, broken_spine, "
    "bad_hands, mutated_hands, deformed_hands, extra_fingers, "
    "missing_fingers, fused_fingers, six_fingers, claw_hands, "
    "twisted_fingers, deformed_feet, extra_toes, fused_toes, missing_toes, "
    "fused_teeth, extra_teeth, malformed_teeth, oversized_teeth, "
    "weird_smile, distorted_mouth, deformed_face, asymmetric_eyes, "
    "cross-eyed, distorted_face, lazy_eye, plastic_skin, doll-like, "
    "uncanny_valley, waxy_skin, watermark, signature, text, logo, "
    "username, multiple_views"
)


# Re-use the human-indicator regex from prompt_audit.py.
import re
HUMAN_INDICATORS = re.compile(
    r"\b(1girl|2girls|3girls|1boy|2boys|3boys|woman|girl|man|boy|"
    r"female|male|figure|portrait|character)\b", re.I,
)


def is_human(prompt: dict) -> bool:
    text = prompt.get("text", "") or ""
    name = (prompt.get("name") or prompt.get("id") or "").lower()
    return bool(HUMAN_INDICATORS.search(text) or HUMAN_INDICATORS.search(name))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                     help="Print what would change without writing")
    ap.add_argument("--all", action="store_true",
                     help="Apply to ALL empty-negative entries, not just humans "
                          "(style/location prompts will also get the negative)")
    ap.add_argument("--library", type=Path, default=LIBRARY)
    args = ap.parse_args()

    raw = json.loads(args.library.read_text())
    if not isinstance(raw, list):
        print("Unexpected library shape — expected top-level list.",
              file=sys.stderr)
        sys.exit(1)

    changed: list[str] = []
    for p in raw:
        if (p.get("negative") or "").strip():
            continue
        if not args.all and not is_human(p):
            continue
        if args.dry_run:
            changed.append(p.get("id", "?"))
            continue
        p["negative"] = BASELINE_NEGATIVE
        p["updated_at"] = _dt.datetime.now(_dt.UTC).isoformat()
        changed.append(p.get("id", "?"))

    if args.dry_run:
        print(f"Would add baseline negative to {len(changed)} entries:")
        for cid in changed:
            print(f"  {cid}")
        return

    if not changed:
        print("No empty-negative human prompts found. Nothing to do.")
        return

    # Backup
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = args.library.with_name(
        f"{args.library.name}.bak-baseline-neg-{stamp}")
    shutil.copy2(args.library, bak)
    args.library.write_text(json.dumps(raw, indent=2))
    print(f"Updated {len(changed)} entries.")
    print(f"Backup: {bak}")


if __name__ == "__main__":
    main()
