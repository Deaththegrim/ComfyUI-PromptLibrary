"""Universal prompt-quality auditor for the ComfyUI PromptLibrary.

Surfaces every entry that fails one of the universal prompt-craft criteria
codified in the `comfy-prompts` skill: age-floor lint violations, empty
negatives, single-word weighted focus terms, multi-entity subjects without
ensemble composition cues, "unseen X" anti-patterns, etc.

Runs READ-ONLY. Outputs a report; you decide which entries to retune.

    cd ~/ComfyUI-PromptLibrary
    python3 tools/prompt_audit.py
    python3 tools/prompt_audit.py --json
    python3 tools/prompt_audit.py --tag SDXL:female
    python3 tools/prompt_audit.py --severity high
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LIBRARY = Path(__file__).resolve().parent.parent / "data" / "prompts.json"


# Universal: hard age-floor gate. Same regex set the cult-of-cards
# prompt builder uses, applied to every prompt in the library.
AGE_FLOOR_PATTERNS = [
    (re.compile(r"\bchild(?:ren)?\b", re.I), "child/children"),
    (re.compile(r"\bkid(?:s|do)?\b", re.I), "kid/kids"),
    (re.compile(r"\byoung\b(?!\s+adult\b)", re.I),
     "young (use 'young adult' or absolute age)"),
    (re.compile(r"\byouth(?:s|ful)?\b", re.I), "youth/youthful"),
    (re.compile(r"\bteen(?:s|age|ager|aged)?\b", re.I), "teen/teenage/teenager"),
    (re.compile(r"\badolescen[ct]\w*\b", re.I), "adolescent/adolescence"),
    (re.compile(r"\b(?:toddler|infant|baby|babies|preteen|preteens|underage)\b", re.I),
     "toddler/infant/baby/preteen/underage"),
    (re.compile(r"\bschool(?:girl|boy|kid|aged?)\b", re.I), "schoolgirl/schoolboy/school-aged"),
    (re.compile(r"\b(?:loli|shota)\b", re.I), "loli/shota"),
]

# Tags that suggest the prompt renders a human figure (and so the age-floor
# lint should fire). Booru-style prompts use "1girl"/"1boy" tokens; others
# use the category tag.
HUMAN_INDICATORS = re.compile(
    r"\b(1girl|2girls|3girls|1boy|2boys|3boys|woman|girl|man|boy|"
    r"female|male|figure|portrait|character)\b", re.I,
)

# "Unseen" anti-pattern from lesson 7.
UNSEEN_PATTERNS = [
    re.compile(r"\bunseen\b", re.I),
    re.compile(r"\binvisible\b", re.I),
    re.compile(r"\bimplied\b", re.I),
    re.compile(r"\boff[-\s]screen\b", re.I),
]

# Cute-leak vocab from lesson 2. Fires only if the prompt's tags suggest
# horror/creature/cult/monster AND the token isn't part of an established
# style anchor (e.g. "naturalist plate meets folk-horror engraving" is the
# beast-faction's intentional Dürer/Goya register, not a leak).
CUTE_LEAK_TOKENS = re.compile(
    r"\b(?:cute|kawaii|adorable|fluffy|fuzzy|googly eyes|"
    r"naturalist plate composition|iridescent shimmer|pastel colors|"
    r"bright daylight|rainbow colors)\b", re.I,
)
HORROR_TAG_HINTS = re.compile(
    r"(?:horror|monster|creature|cult|undead|ghoul|demon|rot|blood|"
    r"corpse|carrion|nuke|wasteland|insect)", re.I,
)
# Tokens that, when present in the body, signal the cute term is
# intentional and the prompt has anti-cute negatives applied.
CUTE_ANTIDOTES = re.compile(
    r"\b(?:body horror|grotesque|nightmarish|monstrous|predatory|"
    r"decomposing|rotted|putrefied|partly-putrefied|skeletal|"
    r"folk-horror)\b", re.I,
)
HORROR_TAG_HINTS = re.compile(
    r"(?:horror|monster|creature|cult|undead|ghoul|demon|rot|blood|"
    r"corpse|carrion|nuke|wasteland|beast|insect)", re.I,
)

# Plural-subject keywords that hint a card might need ensemble composition.
MULTI_ENTITY_HINTS = re.compile(
    r"\b(?:flock|swarm|swarms|swarming|mass of|pile of|circle of|"
    r"ring of|pack|crowd|cluster|huddle|brood of|clutch of|cascade|"
    r"choir|congregation|legion)\b", re.I,
)

# Tags that mark a prompt as a modifier/building-block (intentionally terse).
# Skip stub-body lint for these — they're meant to be combined with other
# prompts, not used standalone.
MODIFIER_TAGS = {
    "Clothing", "Creature", "Style", "Location", "Expression",
    "Proportions", "LoRA", "LoRA:basic", "LoRA:training",
    "Fusion",  # partial-trait modifier prompts
}

# Tags whose prompts often describe ensemble scenery (crowds, trees, etc.)
# as intentional background. Skip MULTI_ENTITY_HINT for these.
ENSEMBLE_BACKGROUND_TAGS = {"Location", "Style"}

# Cult-of-cards card IDs that already have the multi-entity composition
# directive swapped at build time. Auditor shouldn't double-flag them.
# Mirrors `MULTI_ENTITY` in ~/cult-of-cards/tools/build_card_prompts.py.
CULT_MULTI_ENTITY = {
    "cult_card_match_basket", "cult_card_cinder_choir", "cult_card_cinder_pyre",
    "cult_card_barnacle_swarm",
    "cult_card_ash_crow_flock", "cult_card_ambush_pack",
    "cult_card_mutant_brood", "cult_card_chain_reaction",
    "cult_card_thorn_circle",
    "cult_card_maggot_mass", "cult_card_maggot_choir",
    "cult_card_crawling_pyre",
}


def find_age_violations(text: str) -> list[str]:
    """Return human-readable list of age-floor violations in `text`."""
    return [reason for pat, reason in AGE_FLOOR_PATTERNS if pat.search(text)]


def is_human_prompt(prompt: dict) -> bool:
    """Heuristic: does this prompt render a human figure?"""
    if HUMAN_INDICATORS.search(prompt.get("text", "")):
        return True
    name = (prompt.get("name") or prompt.get("id") or "").lower()
    if HUMAN_INDICATORS.search(name):
        return True
    return False


def is_horror_prompt(prompt: dict) -> bool:
    """Heuristic: is this prompt in a horror/creature/cult register?"""
    blob = " ".join([
        " ".join(prompt.get("tags", [])),
        prompt.get("id", ""),
        prompt.get("name", ""),
    ])
    return bool(HORROR_TAG_HINTS.search(blob))


def audit_prompt(prompt: dict) -> dict:
    """Score a prompt against the universal criteria.

    Returns a dict with `severity` (high/med/low/clean) and a list of
    `issues` (each a {code, msg} dict). High = safety or build-breaking;
    Med = quality miss; Low = optional polish.

    Skip rules: modifier-category tags (Clothing/Creature/Style/Location/
    Expression/Proportions/LoRA) are intentionally terse and are used as
    building blocks combined with other prompts. Cards already in the
    cult-of-cards MULTI_ENTITY build-time set are intentionally rendered
    as ensembles — the build script swaps the composition directive."""
    issues: list[dict] = []
    text = prompt.get("text", "") or ""
    negative = prompt.get("negative", "") or ""
    tags = prompt.get("tags", [])

    # Modifier-category prompts are short by design — they're outfit kits,
    # creature type tags, style modifiers, etc. mixed into composed prompts.
    is_modifier = any(t in MODIFIER_TAGS for t in tags)

    # HIGH: age-floor violations on human-rendering prompts
    if is_human_prompt(prompt):
        for reason in find_age_violations(text):
            issues.append({"code": "AGE_FLOOR", "sev": "high",
                              "msg": f"age-floor token in text: {reason}"})

    # HIGH: empty negative on prompts that render humans (quality gate)
    if is_human_prompt(prompt) and not negative.strip():
        issues.append({"code": "EMPTY_NEGATIVE", "sev": "high",
                          "msg": "no negative prompt — missing quality gates"})

    # MED: "unseen X" anti-pattern — skip Style modifiers (they often
    # describe what the STYLE implies, not what to render absent).
    if "Style" not in tags:
        for pat in UNSEEN_PATTERNS:
            m = pat.search(text)
            if m:
                issues.append({"code": "UNSEEN_PATTERN", "sev": "med",
                                  "msg": f"'{m.group(0)}' — model can't render absence"})
                break

    # MED: cute-leak tokens in a horror-context prompt — only fire when
    # the prompt LACKS the anti-cute / body-horror antidote vocabulary.
    if is_horror_prompt(prompt) and not CUTE_ANTIDOTES.search(text):
        m = CUTE_LEAK_TOKENS.search(text)
        if m:
            issues.append({"code": "CUTE_LEAK", "sev": "med",
                              "msg": f"cute-leak token in horror prompt: '{m.group(0)}'"})

    # LOW: tiny text body — skip for modifier-category prompts
    if not is_modifier and len(text.strip()) < 50:
        issues.append({"code": "STUB_BODY", "sev": "low",
                          "msg": f"text body very short ({len(text.strip())} chars)"})

    # LOW: multi-entity hint without ensemble cue — skip for cards that
    # the build-time MULTI_ENTITY set already handles via composition swap,
    # and skip Location/Style prompts (ensemble scenery is intentional).
    is_ensemble_bg = any(t in ENSEMBLE_BACKGROUND_TAGS for t in tags)
    if (not is_ensemble_bg
            and prompt.get("id") not in CULT_MULTI_ENTITY
            and MULTI_ENTITY_HINTS.search(text)
            and not re.search(
                r"multiple subjects|ensemble composition|dense composition",
                text, re.I)):
        issues.append({"code": "MULTI_ENTITY_HINT", "sev": "low",
                          "msg": "plural subject without ensemble composition directive"})

    # Severity rollup
    sevs = [i["sev"] for i in issues]
    if "high" in sevs:
        severity = "high"
    elif "med" in sevs:
        severity = "med"
    elif "low" in sevs:
        severity = "low"
    else:
        severity = "clean"

    return {
        "id": prompt.get("id"),
        "name": prompt.get("name"),
        "tags": tags,
        "severity": severity,
        "issues": issues,
        "is_human": is_human_prompt(prompt),
        "is_horror": is_horror_prompt(prompt),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                     help="Emit machine-readable JSON instead of text report")
    ap.add_argument("--tag", help="Only audit prompts with this tag")
    ap.add_argument("--severity", choices=["high", "med", "low", "clean"],
                     help="Filter to entries at this severity")
    ap.add_argument("--code",
                     help="Filter to entries flagged with this issue code "
                          "(e.g. AGE_FLOOR, EMPTY_NEGATIVE, CUTE_LEAK)")
    ap.add_argument("--library", type=Path, default=LIBRARY,
                     help=f"Path to prompts.json (default: {LIBRARY})")
    args = ap.parse_args()

    raw = json.loads(args.library.read_text())
    prompts = raw if isinstance(raw, list) else raw.get("prompts", [])
    if args.tag:
        prompts = [p for p in prompts if args.tag in p.get("tags", [])]

    reports = [audit_prompt(p) for p in prompts]
    if args.severity:
        reports = [r for r in reports if r["severity"] == args.severity]
    if args.code:
        reports = [r for r in reports
                    if any(i["code"] == args.code for i in r["issues"])]

    if args.json:
        print(json.dumps(reports, indent=2))
        return

    # Text report
    print(f"Audited {len(prompts)} prompts from {args.library}")
    print()

    sev_counts = Counter(r["severity"] for r in reports)
    print(f"Severity rollup:")
    for sev in ("high", "med", "low", "clean"):
        print(f"  {sev:6s} {sev_counts.get(sev, 0)}")
    print()

    code_counts = Counter()
    for r in reports:
        for i in r["issues"]:
            code_counts[i["code"]] += 1
    if code_counts:
        print(f"Issue codes (most common first):")
        for code, n in code_counts.most_common():
            print(f"  {code:25s} {n}")
        print()

    flagged = [r for r in reports if r["severity"] != "clean"]
    flagged.sort(key=lambda r: (-({"high": 3, "med": 2, "low": 1}[r["severity"]]),
                                 r["id"] or ""))

    if not flagged:
        print("No issues found.")
        return

    print(f"Flagged entries ({len(flagged)}):")
    print("=" * 78)
    for r in flagged:
        tag_str = ",".join(r["tags"][:3])
        print(f"[{r['severity'].upper():4s}] {r['id']:35s} ({tag_str})")
        for i in r["issues"]:
            print(f"        - {i['code']}: {i['msg']}")


if __name__ == "__main__":
    main()
