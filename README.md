# Ribbity — ComfyUI Prompt Library

A visual prompt manager for ComfyUI. Browse a thumbnail grid of saved prompts, tag them, drop them into your workflow with one click. Filesystem-backed — your library survives browser clears and syncs cleanly across machines.

Four nodes, one shared library, no third-party dependencies.

## Install

### Comfy Portable (Windows)

1. Download the latest [release zip](https://github.com/Deaththegrim/ComfyUI-PromptLibrary/releases/latest).
2. Extract into `ComfyUI_windows_portable\ComfyUI\custom_nodes\` so the path becomes `…\custom_nodes\ComfyUI-PromptLibrary\__init__.py`.
3. Restart ComfyUI.

### Linux / Mac (git clone)

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/Deaththegrim/ComfyUI-PromptLibrary
# restart ComfyUI
```

To update later:
```sh
cd ComfyUI/custom_nodes/ComfyUI-PromptLibrary
git pull
# restart ComfyUI
```

Your `data/` folder is gitignored, so `git pull` won't touch your library. On first launch after a version bump, the package automatically copies `data/` to `data-backup-<old_version>-<timestamp>/` next to it as a safety net.

## Nodes

After install, find them in the node menu under **utils/**:

- **Ribbity — Library** — visual gallery picker, outputs `STRING` (the selected prompt's text)
- **Ribbity — Save** — write prompts to the library during workflow runs (inputs: name, text, optional `IMAGE` thumbnail, optional tags)
- **Ribbity — Random by Tag** — pick a random library entry by tag filter, for overnight gen loops
- **Ribbity — Wildcard Expand** — expand `{a|b|c}` alternatives and `__name__` library refs in any string

## Gallery

The Library node embeds a full gallery in the node body:

- Click a thumbnail to select it as the prompt output. Click again to deselect.
- Right-click for **Edit / Duplicate / Export this / Delete**.
- Hover a tile and click the corner checkbox to multi-select. Shift-click extends the range; Ctrl/Cmd-click toggles individuals. With one or more selected, the bulk bar appears with **Tag / Export / Delete**.
- Drag-and-drop tiles to reorder when sort mode is **Manual**.
- Keyboard: arrow keys move focus, **Enter** selects, **Delete** removes, **/** focuses search, **Esc** clears the bulk selection.

### Toolbar

| Control | What it does |
|---|---|
| Search | Matches name, text, tags, and id |
| Model dropdown | Lifts `model:*` tags into a top-level filter (Anima, SDXL, Pony, …) |
| Sort | Manual (drag-reorder) / Name A-Z / Z-A / Newest / Oldest / Recent edit |
| Tile size slider | 60–200 px, persisted per browser |
| Import | Auto-detects `.csv` (columns: `name, text, tags, id` — tags use `;` inside cell) or Ribbity `.zip` |
| Export | Packs currently visible prompts + thumbnails into a zip download |
| Refresh | Reload from disk |

### Tag chips

Tags use `category:value` syntax (e.g. `style:cyberpunk`, `character:elf`). The chip row groups them by category. Chips OR-filter the grid; combine with the search box for AND.

`model:*` tags are special — they're lifted out of the chip row into the dedicated dropdown.

### Editing

Right-click → Edit, or click the `+` tile to add. The modal is non-blocking — drag it around, the canvas stays interactive. Fields:

- **Name** — human-readable label
- **ID** — auto-derived from the name (`Cyberpunk Style` → `cyberpunk_style`); override only if you want a specific filesystem name. Collisions auto-append `_2`, `_3`, …
- **Tags** — comma-separated; autocompletes from existing tags via a `<datalist>`
- **Prompt text** — the actual prompt string
- **Reference image** — optional thumbnail; PNG/JPG/WebP/GIF/BMP, capped at 16 MB
- **History** — disclosure showing every prior version of this entry (max 20). Click any row to revert.

## Wildcards & overnight loops

Inside any prompt text:

- `{red|green|blue}` — picks one at random per generation
- `__character__` — resolves to a random library entry whose **id** matches `character`, or whose **tags** contain `character`. Recurse-safe (max depth 8); cycles bottom out as literals.

For a fully randomized overnight loop, chain three **Random by Tag** nodes:

```
[Random by Tag: character]  ─┐
[Random by Tag: background] ─┼─ concat ─→ CLIP Text Encode ─→ KSampler
[Random by Tag: action]      ─┘
```

Each node has a `seed` input — set `control_after_generate=randomize` to draw a fresh combination every queue. Wildcards in the picked prompts (`old {grumpy|wise} wizard`) re-roll per generation.

The Wildcard Expand node has independent toggles for `expand_choices` and `expand_named_refs` if you want one without the other.

## Storage

Everything lives in `ComfyUI-PromptLibrary/data/`:

```
data/
├── prompts.json           # entries: id, name, text, tags, history, timestamps
├── images/<id>.<ext>      # thumbnails
└── .last_version          # auto-backup marker
```

Both `prompts.json` and `images/` are gitignored. Back up the whole `data/` folder to keep your library safe across reinstalls.

External edits to `prompts.json` are picked up automatically — the gallery polls the file every 2 s and refreshes on change. So you can version-control your library in git or edit it in your favourite text editor.

## Sharing libraries

- **Export some prompts**: filter the gallery (search, tags, model, anything), click **Export**. You get a zip with `prompts.json` + thumbnails.
- **Send the zip** to a friend.
- **They click Import** in their gallery and pick the zip. Existing entries with the same id get history-bumped before being overwritten; new ones are appended.

Same flow works for backups: export the whole library to a zip and stash it.

## CSV import

For seeding from a spreadsheet, save as CSV with these columns (header row required):

| name | text | tags | id |
|---|---|---|---|
| Cyberpunk Style | "vibrant neon, rain" | style;cyberpunk | (blank) |
| Wizard | "old man with staff" | character;fantasy | wizard |

- `name` is required
- `text` is the prompt body
- `tags` use `;` as the in-cell separator (because `,` is the CSV delimiter)
- `id` is optional; blank means auto-derive from name

## Compatibility & deps

- ComfyUI any reasonably recent version (tested with 0.19.x, frontend 1.42.x)
- Python 3.10+
- No third-party deps beyond what ComfyUI already ships (`aiohttp`, `numpy`, `Pillow`)

## Development

```sh
git clone https://github.com/Deaththegrim/ComfyUI-PromptLibrary
cd ComfyUI-PromptLibrary
python3 -m venv .testenv
.testenv/bin/pip install aiohttp pillow
.testenv/bin/python -m unittest discover tests
```

100+ tests, runs in ~0.13 s.

## Credits

Co-created by **[Deaththegrim](https://github.com/Deaththegrim)** and
**RibbityRabbit** (whose feedback shaped most of the feature set — the
branding is theirs too). Pair-programmed with **[Claude Code](https://claude.com/claude-code)**
(Anthropic's CLI for Claude). Design decisions, scope calls, UX direction,
and final review are the maintainers'; Claude wrote most of the code under
that direction.

If you fork, remix, or publish a derivative, a credit line back to
[Deaththegrim/ComfyUI-PromptLibrary](https://github.com/Deaththegrim/ComfyUI-PromptLibrary)
is appreciated.

## License

MIT — see [LICENSE](LICENSE). You're free to use, modify, redistribute, and
include this in commercial work. The one ask: keep the copyright notice
("Copyright (c) 2026 Deaththegrim and RibbityRabbit") intact in copies and
substantial portions.
