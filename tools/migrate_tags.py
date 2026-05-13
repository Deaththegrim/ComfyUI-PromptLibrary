"""Migrate flat library tags to the prefix:label convention.

The Library node's gallery groups chips by `:` prefix into one row per
group. Two-flavor families like "sdxl female" / "sdxl male" or "duo
character" / "trio character" land in separate flat rows under the old
scheme — under the new scheme they collapse into one grouped row with
shared sub-chips, matching the Cards taxonomy we just shipped.

Skips entries whose id starts with `cult_card_` — those are managed by
~/cult-of-cards/tools/build_card_prompts.py and a re-run of that script
would overwrite anything we change here.

Run:
    python3 tools/migrate_tags.py            # apply + write backup
    python3 tools/migrate_tags.py --dry-run  # preview, no write

The mapping is idempotent: re-running produces 0 changes.
"""

from __future__ import annotations
import argparse
import datetime as _dt
import json
import shutil
from pathlib import Path

LIBRARY_JSON = Path.home() / "ComfyUI-PromptLibrary" / "data" / "prompts.json"

# old_tag -> tuple of new tags. Each old tag expands to its full
# replacement set (umbrella + namespaced, or just the renamed standalone).
# A tag that's already in the new form maps to itself (no-op) — listed
# explicitly so re-running drops nothing.
TAG_RENAMES: dict[str, tuple[str, ...]] = {
    # Multi-flavor families: introduce the umbrella + namespaced sub-tag.
    "sdxl female":          ("SDXL", "SDXL:female"),
    "sdxl male":            ("SDXL", "SDXL:male"),
    "sdxl genderbend":      ("SDXL", "SDXL:genderbend"),
    "sdxl male as female":  ("SDXL", "SDXL:male_as_female"),
    "cartoon milfs":        ("Cartoon", "Cartoon:milfs"),
    "duo character":        ("Character", "Character:duo"),
    "trio character":       ("Character", "Character:trio"),
    "lora training":        ("LoRA", "LoRA:training"),
    "lora":                 ("LoRA", "LoRA:basic"),
    # Singleton categories — just capitalize.
    "style":        ("Style",),
    "location":     ("Location",),
    "clothing":     ("Clothing",),
    "creature":     ("Creature",),
    "fusion":       ("Fusion",),
    "expression":   ("Expression",),
    "gremmy":       ("Gremmy",),
    "researcher":   ("Researcher",),
    "oc":           ("OC",),
    "poses":        ("Poses",),
    "pony":         ("Pony",),
    "proportions":  ("Proportions",),
}


def _migrate_tag_list(old_tags: list[str]) -> list[str]:
    """Apply renames + dedupe while preserving the user's insertion order
    (first-seen wins). Tags not in the map pass through untouched —
    that's how cult_card_*'s `Cards`, `Cards:fire`, etc. stay stable."""
    out: list[str] = []
    seen: set[str] = set()
    for t in old_tags:
        replacements = TAG_RENAMES.get(t, (t,))
        for r in replacements:
            if r not in seen:
                seen.add(r)
                out.append(r)
    return out


def _backup(path: Path) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = path.with_name(f"{path.name}.bak-tag-migrate-{ts}")
    shutil.copy2(path, dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Show counts without writing")
    ap.add_argument("--library", default=str(LIBRARY_JSON))
    args = ap.parse_args()

    lib = Path(args.library)
    data = json.loads(lib.read_text())

    changed = 0
    skipped_card = 0
    for entry in data:
        if entry.get("id", "").startswith("cult_card_"):
            skipped_card += 1
            continue
        old = entry.get("tags") or []
        new = _migrate_tag_list(old)
        if new != old:
            entry["tags"] = new
            changed += 1

    from collections import Counter
    after = Counter(t for e in data for t in (e.get("tags") or []))
    print(f"entries scanned    : {len(data)}")
    print(f"cult_card_ skipped : {skipped_card}")
    print(f"entries changed    : {changed}")
    print()
    print("--- top tags after migration ---")
    for t, c in after.most_common(25):
        print(f"  {c:4d}  {t}")

    if args.dry_run:
        print("\ndry-run: no write")
        return

    backup = _backup(lib)
    print(f"\nbackup: {backup}")
    lib.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"wrote : {lib}")


if __name__ == "__main__":
    main()
