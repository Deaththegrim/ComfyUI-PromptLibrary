# Screenshot runner

Scripts the ComfyUI frontend via Playwright to capture deterministic
screenshots of the Library gallery + Edit Prompt modal. Replaces the
manual capture-and-crop loop for the UI-only shots (modal, gallery
states, etc.).

## Setup

Once per machine:

```sh
pip install --user playwright
python3 -m playwright install firefox
```

Already installed if you've run the AID export tooling.

## Usage

```sh
# all scenarios, headless
python3 tools/screenshots/run.py

# show the browser window (good for debugging a flaky scenario)
python3 tools/screenshots/run.py --headed

# one or more scenarios
python3 tools/screenshots/run.py --only gallery-list-view modal-history

# list scenarios
python3 tools/screenshots/run.py --list

# point at a different ComfyUI instance
python3 tools/screenshots/run.py --comfy http://192.168.1.50:8188
```

ComfyUI must be running at the URL (default `http://127.0.0.1:8188`).
Output PNGs land in `docs/screenshots/<scenario>.png`.

## Adding a scenario

1. If the scenario needs a graph view of more than just the Library node,
   build the graph in ComfyUI, **File → Save** the workflow JSON into
   `docs/workflows/<name>.json`.
2. In `run.py`, add an `_scn_<your_name>(page, out)` function that does
   the UI dance (clicks, waits, etc.) and finishes with one of:
   - `screenshot_gallery(page, out)` — clipped to the Library node's
     embedded gallery.
   - `screenshot_modal(page, out, scroll_to=...)` — full-modal capture,
     temporarily lifts the `max-height` clamp so content below the fold
     is included.
   - Manual `page.screenshot(...)` with a custom clip rect.
3. Append a `Scenario(...)` entry to the `SCENARIOS` list.

## How state is reset

Before each scenario, `reset_gallery_state()` strips every `pl-*` /
`comfy.PromptLibrary.*` localStorage key and pre-seeds:

- `comfy.PromptLibrary.view` = `grid`
- `comfy.PromptLibrary.tileSize` = `110`
- `comfy.PromptLibrary.sort` = `name_asc`

After each scenario, any open modal / context menu is removed and the
graph is cleared. This keeps the runs deterministic — no scenario
inherits leftover UI state from the previous one.

## What's NOT in here

- **Real generations**: capturing a node mid-sample needs a real GPU run,
  the user's models, and time. Out of scope for the headless runner.
- **Civitai PNG metadata**: needs a generated image file. Capture by hand.
- **Per-node graph shots beyond the Library node**: add a workflow JSON
  + a scenario; the canvas screenshot itself is one `page.screenshot()`
  with a clip rect calculated from `app.canvas.ds.scale` and the node's
  pos/size. Not wired up yet — file an issue or ask if you want it.
