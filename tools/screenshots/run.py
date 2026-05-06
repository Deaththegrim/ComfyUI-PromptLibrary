#!/usr/bin/env python3
"""Screenshot runner for ComfyUI-PromptLibrary.

Drives a Playwright-controlled browser against a running ComfyUI instance,
loads workflow JSON snippets from docs/workflows/, scripts the UI through a
list of scenarios, and saves PNGs into docs/screenshots/.

Usage:
    python3 tools/screenshots/run.py                  # all scenarios, headless
    python3 tools/screenshots/run.py --headed         # show the browser window
    python3 tools/screenshots/run.py --only modal-loras gallery-list-view
    python3 tools/screenshots/run.py --comfy http://127.0.0.1:8188

Scenarios that need a saved workflow JSON look it up under docs/workflows/.
Add new ones by saving a workflow from ComfyUI (drag the relevant nodes,
File → Save) into that folder, then declaring a Scenario below.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from playwright.sync_api import Page, sync_playwright


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / "docs" / "workflows"
SCREENSHOTS = REPO_ROOT / "docs" / "screenshots"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def load_workflow(page: Page, name: str) -> None:
    """Load a workflow JSON via the ComfyUI app object — bypasses the file
    picker entirely. Equivalent to drag-dropping the JSON onto the canvas."""
    path = WORKFLOWS / name
    if not path.exists():
        raise FileNotFoundError(f"workflow JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    # Wait for the global app object to be ready.
    page.wait_for_function("() => !!(window.app && window.app.loadGraphData)",
                            timeout=20_000)
    page.evaluate("(wf) => window.app.loadGraphData(wf)", data)
    # Give the canvas time to render the loaded graph + the gallery DOM widget.
    page.wait_for_timeout(800)


def reset_gallery_state(page: Page) -> None:
    """Strip every gallery-related localStorage key + force grid view + a
    known tile size + sort, so each scenario starts from the same UI state.
    Called before load_workflow, since the JS reads localStorage on the
    onNodeCreated hook that runs at workflow-load time."""
    page.evaluate("""() => {
        for (const k of Object.keys(localStorage)) {
            if (/promptLibrary/i.test(k) || /^pl-/i.test(k)) {
                localStorage.removeItem(k);
            }
        }
        // Pre-seed the keys the JS actually reads so the gallery starts
        // from a known view + sort + tile size on every scenario.
        localStorage.setItem('comfy.PromptLibrary.view', 'grid');
        localStorage.setItem('comfy.PromptLibrary.tileSize', '110');
        localStorage.setItem('comfy.PromptLibrary.sort', 'name_asc');
    }""")


def wait_for_gallery(page: Page) -> None:
    """Block until the Library node's embedded gallery has rendered tiles
    (or shown the empty state)."""
    page.wait_for_selector(".pl-gallery", timeout=15_000)
    # Either tiles or the empty state — both mean the gallery loaded.
    page.wait_for_function(
        """() => {
            const g = document.querySelector('.pl-gallery');
            if (!g) return false;
            return g.querySelector('.pl-tile, .pl-empty-state') !== null;
        }""",
        timeout=15_000,
    )


def screenshot_modal(page: Page, out: Path, *, scroll_to: str | None = None) -> None:
    """Snapshot the entire modal, including content scrolled below the fold.
    The modal has max-height: 80vh + overflow-y: auto, so its bounding-box
    height is the visible (clipped) height, not the full content. To capture
    the full thing, we temporarily disable max-height so the modal expands,
    grow the viewport to match, then snap. Optional `scroll_to` selector
    scrolls a child into view first (only relevant before the height swap)."""
    el = page.wait_for_selector(".pl-modal", timeout=10_000)
    if scroll_to:
        page.evaluate(f"() => document.querySelector({scroll_to!r})?.scrollIntoView({{block: 'center'}})")
        page.wait_for_timeout(150)
    # Drop the height clamp + scroll so the modal renders at full content
    # size, then resize the viewport so the screenshot can fit it. Restore
    # both afterwards so subsequent scenarios see a clean state.
    original_viewport = page.viewport_size
    page.evaluate("""() => {
        const m = document.querySelector('.pl-modal');
        if (!m) return;
        m.style.maxHeight = 'none';
        m.style.overflowY = 'visible';
    }""")
    page.wait_for_timeout(100)
    box = el.bounding_box()
    if not box:
        el.screenshot(path=str(out))
        return
    pad = 24
    needed_h = int(box["y"] + box["height"] + pad)
    needed_w = int(box["x"] + box["width"] + pad)
    if original_viewport and (needed_h > original_viewport["height"]
                                or needed_w > original_viewport["width"]):
        page.set_viewport_size({"width": max(original_viewport["width"], needed_w),
                                  "height": max(original_viewport["height"], needed_h)})
        page.wait_for_timeout(150)
        # Re-measure after resize — reflow may have shifted the modal.
        box = el.bounding_box() or box
    page.screenshot(path=str(out), clip={
        "x": max(0, box["x"] - pad),
        "y": max(0, box["y"] - pad),
        "width": box["width"] + pad * 2,
        "height": box["height"] + pad * 2,
    })
    if original_viewport:
        page.set_viewport_size(original_viewport)


def screenshot_gallery(page: Page, out: Path) -> None:
    """Snapshot just the Library node (its DOM widget container)."""
    el = page.wait_for_selector(".pl-gallery", timeout=10_000)
    el.screenshot(path=str(out))


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    workflow: str | None
    action: Callable[[Page, Path], None]
    description: str


def _scn_library_gallery(page: Page, out: Path) -> None:
    wait_for_gallery(page)
    screenshot_gallery(page, out)


def _scn_gallery_list_view(page: Page, out: Path) -> None:
    wait_for_gallery(page)
    # Toggle the list-view button. The button's textContent is "≡".
    page.evaluate("""() => {
        const btns = document.querySelectorAll('.pl-view-toggle .pl-btn');
        for (const b of btns) if (b.textContent.trim() === '≡') return b.click();
    }""")
    page.wait_for_timeout(300)
    screenshot_gallery(page, out)


def _scn_gallery_search(page: Page, out: Path) -> None:
    wait_for_gallery(page)
    inp = page.wait_for_selector(".pl-search-wrap input", timeout=5_000)
    inp.fill("a")
    page.wait_for_timeout(300)
    screenshot_gallery(page, out)


def _scn_gallery_tag_filter(page: Page, out: Path) -> None:
    wait_for_gallery(page)
    # Click the first non-"all" tag chip if there is one.
    clicked = page.evaluate("""() => {
        const chips = document.querySelectorAll('.pl-tag-chip');
        for (const c of chips) {
            if (c.classList.contains('all') || c.classList.contains('pl-tag-mode')) continue;
            c.click(); return true;
        }
        return false;
    }""")
    if not clicked:
        print(f"   note: no tag chips visible; capturing default state")
    page.wait_for_timeout(300)
    screenshot_gallery(page, out)


def _scn_gallery_bulk_select(page: Page, out: Path) -> None:
    wait_for_gallery(page)
    # Toggle the per-tile checkbox on the first 2 tiles to enter bulk mode.
    page.evaluate("""() => {
        const tiles = document.querySelectorAll('.pl-tile');
        for (let i = 0; i < Math.min(2, tiles.length); i++) {
            const check = tiles[i].querySelector('.pl-tile-check');
            if (check) check.click();
        }
    }""")
    page.wait_for_timeout(300)
    screenshot_gallery(page, out)


def _scn_gallery_context_menu(page: Page, out: Path) -> None:
    wait_for_gallery(page)
    tile = page.wait_for_selector(".pl-tile", timeout=5_000)
    tile.click(button="right")
    page.wait_for_selector(".pl-context-menu", timeout=3_000)
    # Compute the union bounding box of the gallery + the floating menu so
    # the screenshot is tight around the relevant UI rather than the whole
    # ComfyUI chrome.
    rect = page.evaluate("""() => {
        const g = document.querySelector('.pl-gallery');
        const m = document.querySelector('.pl-context-menu');
        if (!g || !m) return null;
        const gr = g.getBoundingClientRect();
        const mr = m.getBoundingClientRect();
        return {
            x: Math.min(gr.left, mr.left),
            y: Math.min(gr.top, mr.top),
            right: Math.max(gr.right, mr.right),
            bottom: Math.max(gr.bottom, mr.bottom),
        };
    }""")
    if rect:
        pad = 12
        page.screenshot(path=str(out), clip={
            "x": max(0, rect["x"] - pad),
            "y": max(0, rect["y"] - pad),
            "width": rect["right"] - rect["x"] + pad * 2,
            "height": rect["bottom"] - rect["y"] + pad * 2,
        })
    else:
        page.screenshot(path=str(out))


def _scn_edit_prompt_modal_loras(page: Page, out: Path) -> None:
    wait_for_gallery(page)
    # Right-click first tile → Edit.
    tile = page.wait_for_selector(".pl-tile", timeout=5_000)
    tile.click(button="right")
    page.wait_for_selector(".pl-context-menu", timeout=3_000)
    page.evaluate("""() => {
        const items = document.querySelectorAll('.pl-context-menu .item');
        for (const i of items) if (i.textContent.trim().startsWith('Edit')) return i.click();
    }""")
    # Wait for modal, then add a LoRA row so the new section is in shot.
    page.wait_for_selector(".pl-modal .pl-loras-add", timeout=5_000)
    page.evaluate("() => document.querySelector('.pl-modal .pl-loras-add')?.click()")
    page.wait_for_timeout(400)
    screenshot_modal(page, out, scroll_to=".pl-loras-section")


def _scn_modal_history(page: Page, out: Path) -> None:
    wait_for_gallery(page)
    tile = page.wait_for_selector(".pl-tile", timeout=5_000)
    tile.click(button="right")
    page.wait_for_selector(".pl-context-menu", timeout=3_000)
    page.evaluate("""() => {
        const items = document.querySelectorAll('.pl-context-menu .item');
        for (const i of items) if (i.textContent.trim().startsWith('Edit')) return i.click();
    }""")
    page.wait_for_selector(".pl-modal", timeout=5_000)
    # Open the History details.
    page.evaluate("""() => {
        const dets = document.querySelectorAll('.pl-modal details');
        for (const d of dets) {
            const sum = d.querySelector('summary');
            if (sum && /history/i.test(sum.textContent)) {
                d.open = true;
                d.dispatchEvent(new Event('toggle'));
                return;
            }
        }
    }""")
    page.wait_for_timeout(800)
    screenshot_modal(page, out, scroll_to=".pl-modal details")


SCENARIOS: list[Scenario] = [
    Scenario("library-gallery", "library-minimal.json",
              _scn_library_gallery,
              "Full Library gallery, default state."),
    Scenario("gallery-list-view", "library-minimal.json",
              _scn_gallery_list_view,
              "Library gallery in single-column list mode."),
    Scenario("gallery-search-clear", "library-minimal.json",
              _scn_gallery_search,
              "Library gallery with a search query active (× clear visible)."),
    Scenario("gallery-tag-filter", "library-minimal.json",
              _scn_gallery_tag_filter,
              "Library gallery with one tag chip active."),
    Scenario("gallery-bulk-select", "library-minimal.json",
              _scn_gallery_bulk_select,
              "Library gallery with two tiles checkbox-selected."),
    Scenario("gallery-context-menu", "library-minimal.json",
              _scn_gallery_context_menu,
              "Right-click context menu on a tile."),
    Scenario("edit-prompt-modal-loras", "library-minimal.json",
              _scn_edit_prompt_modal_loras,
              "Edit Prompt modal open, LoRA section expanded with one row."),
    Scenario("modal-history", "library-minimal.json",
              _scn_modal_history,
              "Edit Prompt modal with the History disclosure expanded."),
]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run(comfy_url: str, headed: bool, only: list[str] | None) -> int:
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    selected = [s for s in SCENARIOS if not only or s.name in only]
    if not selected:
        print(f"No scenarios matched: {only!r}")
        print("Available:")
        for s in SCENARIOS:
            print(f"  - {s.name}")
        return 1

    failures: list[tuple[str, str]] = []
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=not headed)
        context = browser.new_context(viewport={"width": 1280, "height": 900},
                                       device_scale_factor=1)
        page = context.new_page()
        page.set_default_timeout(15_000)

        for scn in selected:
            print(f"→ {scn.name}: {scn.description}")
            try:
                page.goto(comfy_url, wait_until="domcontentloaded")
                page.wait_for_function(
                    "() => !!(window.app && window.app.graph)",
                    timeout=30_000)
                reset_gallery_state(page)
                if scn.workflow:
                    load_workflow(page, scn.workflow)
                out = SCREENSHOTS / f"{scn.name}.png"
                scn.action(page, out)
                if out.exists():
                    print(f"   ✓ {out.relative_to(REPO_ROOT)} ({out.stat().st_size // 1024} KB)")
                else:
                    failures.append((scn.name, "screenshot file missing"))
                # Reset between scenarios — clear the graph + close any open
                # modals so leftover state from a previous scenario doesn't
                # bleed into the next.
                page.evaluate("""() => {
                    document.querySelectorAll('.pl-modal, .pl-context-menu')
                        .forEach(el => el.remove());
                    if (window.app && window.app.graph) {
                        try { window.app.graph.clear(); } catch (e) {}
                    }
                }""")
            except Exception as e:
                failures.append((scn.name, str(e)))
                print(f"   ✗ failed: {e}")

        context.close()
        browser.close()

    print()
    if failures:
        print(f"{len(failures)} scenario(s) failed:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        return 2
    print(f"All {len(selected)} scenario(s) captured into {SCREENSHOTS.relative_to(REPO_ROOT)}/")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy", default="http://127.0.0.1:8188",
                     help="ComfyUI base URL (default: %(default)s)")
    ap.add_argument("--headed", action="store_true",
                     help="Show the browser window (useful for debugging)")
    ap.add_argument("--only", nargs="*",
                     help="Run only these scenarios (by name)")
    ap.add_argument("--list", action="store_true",
                     help="List scenarios and exit")
    args = ap.parse_args()

    if args.list:
        for s in SCENARIOS:
            wf = f" [{s.workflow}]" if s.workflow else ""
            print(f"  {s.name}{wf}\n      {s.description}")
        return 0

    return run(args.comfy, args.headed, args.only)


if __name__ == "__main__":
    sys.exit(main())
