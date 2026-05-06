# Screenshot Coverage TODO

Saved into `docs/screenshots/` with kebab-case names so the README can
reference them directly. Already done at top, missing below.

Aim for 1280–1600 px wide, lossless PNG, dark theme, real data (anything
already in your library is fine — no need to seed fake entries).

## Done

Captured by the screenshot runner — re-run any time the UI changes:

```sh
python3 tools/screenshots/run.py
```

- [x] `library-gallery.png` — full thumbnail gallery, default state
- [x] `library-node-with-sockets.png` — full Library node showing v0.32 MODEL/CLIP optional sockets
- [x] `style-node-with-sockets.png` — full Style node showing v0.29 positive/negative split I/O + bypass widget
- [x] `gallery-list-view.png` — single-column list mode
- [x] `gallery-search-clear.png` — search active with × clear button
- [x] `gallery-tag-filter.png` — `anima character` chip active, grid filtered
- [x] `gallery-bulk-select.png` — two tiles checkbox-selected, bulk action bar
- [x] `gallery-context-menu.png` — right-click menu on a tile
- [x] `edit-prompt-modal-loras.png` — Edit Prompt modal with the LoRA section open + one row
- [x] `modal-history.png` — Edit Prompt modal with the History disclosure expanded
- [x] `comic-frame-overview.png` — Comic Frame node combining character + scene + background (manual)
- [x] `sdxl-pipeline.png` — sampler chain end-to-end (manual)

## v0.29.0 / v0.32.0 — Style node + Library MODEL+CLIP follow-ups

- [ ] `style-node-workflow.png` — Style node wired between CheckpointLoader and KSampler (or the SDXL Sampler), with `positive` and `negative` flowing into the sampler. Show the gallery embedded in the node body with an entry selected. (Manual capture — needs a complete graph.)
- [ ] `style-node-bypass.png` — same node with `bypass=true` so the user sees the toggle in context. (Manual.)
- [ ] `library-node-loras-applied.png` — Library node with MODEL+CLIP wired and a tile selected showing patched outputs flowing downstream. (Manual.)

## v0.30.0 — Prompt log

- [ ] `prompt-log-toggle.png` — SDXL or Anima sampler with `save_prompt_log` toggled on, `prompt_log_path` filled (or empty to show the default-path placeholder).
- [ ] `prompt-log-jsonl-snippet.png` — terminal/editor showing 2–3 JSONL lines of a real `prompts.jsonl` (redact anything personal). Just a snippet, no need for the whole file.

## Other library nodes

- [ ] `multi-library-3-panels.png` — Multi Library node showing three panels with different filters (e.g. Character / Style / Clothing).
- [ ] `random-by-tag-overnight.png` — three Random by Tag nodes feeding a concat → CLIPTextEncode → KSampler chain. The classic overnight pattern.
- [ ] `wildcard-expand-text.png` — Wildcard Expand node with a `{a|b|c}` and `__char__` reference, output text shown.
- [ ] `save-node.png` — Save node mid-run capturing a freshly-generated image to the library.

## Specialty nodes

- [ ] `character-anchor.png` — Character Anchor in a comic-page workflow with IPAdapter Plus wired upstream.
- [ ] `comic-page-regional.png` — Comic Page (Regional) with the panel-layout mask preview visible + per-panel prompts.
- [ ] `lora-picker.png` — LoRA Picker emitting an inline `<lora:...>` token into a prompt chain.
- [ ] `civitai-save.png` — Save Image (Civitai) writing a PNG and the resulting Civitai-style metadata row in an image viewer's properties.

## Gallery features (zoom-ins)

Most of these are now automated — see the **Done** list above. Two left:

- [ ] `gallery-drag-reorder.png` — mid-drag of a tile in Manual sort mode. (Hard to script — drag gesture timing is brittle.)
- [ ] `gallery-empty-state.png` — empty library with the empty-state message. (Needs a temporary library reset; better captured manually on a fresh install.)

## Modal disclosures

- [ ] `modal-image-paste.png` — modal mid-paste with the toast confirming clipboard image. (Needs simulated clipboard paste — manual.)

## After capture

1. Drop into `docs/screenshots/` with the filename above.
2. Reference in README under the matching node section.
3. `git add docs/screenshots/<file>.png && git commit && git push`.

To wire a screenshot into the README quickly, the existing pattern is:

```markdown
![Caption](docs/screenshots/<file>.png)
```

Place it directly under the node bullet that introduces the feature.
