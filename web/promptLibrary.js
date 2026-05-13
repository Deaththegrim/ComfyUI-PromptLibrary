import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { lsSelKey, lsReadSel, lsWriteSel, lsClearSel } from "./lsSelection.js";

const NODE_NAME = "PromptLibrary";
const STYLE_NODE_NAME = "PromptLibraryStyle";
const GALLERY_NODE_NAMES = new Set([NODE_NAME, STYLE_NODE_NAME]);
const MULTI_NODE_NAME = "PromptLibraryMulti";
const MULTI_PANELS = 3;
const COMIC_FRAME_NODE_NAME = "PromptLibraryComicFrame";
const BACKGROUND_NODE_NAME = "PromptLibraryBackground";
const STYLE_ID = "prompt-library-style";

const CSS = `
.pl-gallery, .pl-modal, .pl-context-menu, .pl-det-grid {
  --pl-fg: var(--fg-color, #ddd);
  --pl-fg-muted: var(--descrip-text, #888);
  --pl-fg-placeholder: #666;
  --pl-fg-strong: #fff;
  --pl-bg-input: var(--input-bg, var(--comfy-input-bg, #1c1c1c));
  --pl-bg-elevated: var(--button-surface, var(--comfy-menu-bg, #2a2a2a));
  --pl-bg-deep: var(--bg-color, #1a1a1a);
  --pl-bg-hover: var(--button-hover-surface, #383838);
  --pl-bg-selected: #1f3550;
  --pl-bg-selected-strong: #2d5070;
  --pl-bg-empty: var(--tr-even-bg-color, #232323);
  --pl-bg-modal-header: #1f1f1f;
  --pl-border: var(--border-color, var(--border-default, #444));
  --pl-border-strong: #555;
  --pl-border-soft: #111;
  --pl-accent: var(--accent-primary, #6cf);
  --pl-accent-fg: #111;
  --pl-danger: var(--error-text, #f88);
  --pl-focus-outline: #f9a;
  /* State colors — distinct from accent so the eye reads them as state, not as competing primaries. */
  --pl-success: #6c9b46;
  --pl-success-strong: #7eaf52;
  --pl-success-edge: #4f7a35;
  --pl-rating: #f5b94a;
  /* Spacing scale (4px base). Use tokens, not raw px. */
  --pl-sp-2xs: 2px;
  --pl-sp-xs: 4px;
  --pl-sp-sm: 8px;
  --pl-sp-md: 12px;
  --pl-sp-lg: 16px;
  --pl-sp-xl: 24px;
  /* Radius scale */
  --pl-r-sm: 4px;
  --pl-r-md: 6px;
}
.pl-gallery { display: flex; flex-direction: column; gap: var(--pl-sp-sm); padding: var(--pl-sp-xs); box-sizing: border-box;
  width: 100%; height: 100%; min-height: 0; color: var(--pl-fg); font-family: sans-serif; font-size: 12px;
  position: relative; }
.pl-gallery.pl-drop-target { outline: 2px dashed var(--pl-accent); outline-offset: -4px; background: var(--pl-bg-selected); }
.pl-gallery.pl-drop-target::before { content: "Drop CSV or ZIP to import";
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  background: rgba(28, 36, 48, 0.85); color: var(--pl-accent); font-size: 14px; font-weight: 600;
  padding: var(--pl-sp-md); text-align: center; word-break: break-word;
  pointer-events: none; z-index: 10; border-radius: var(--pl-r-sm); }
.pl-panel { display: flex; flex-direction: column; gap: var(--pl-sp-xs); box-sizing: border-box;
  width: 100%; height: 100%; min-height: 0; }
.pl-panel-header { background: var(--pl-bg-elevated); color: var(--pl-fg); padding: var(--pl-sp-xs) var(--pl-sp-sm); border-radius: var(--pl-r-sm);
  font-weight: 600; font-size: 12px; outline: none; cursor: text;
  border: 1px solid transparent; flex: 0 0 auto; }
.pl-panel-header:hover { border-color: var(--pl-border); }
.pl-panel-header:focus { background: var(--pl-bg-input); border-color: var(--pl-accent); }
.pl-panel-body { flex: 1 1 0; min-height: 0; display: flex; }
.pl-panel-body .pl-gallery { padding: 0; }
.pl-toolbar { display: flex; gap: var(--pl-sp-sm); align-items: center; }
.pl-toolbar input, .pl-toolbar select { flex: 1; min-width: 0; background: var(--pl-bg-input); color: var(--pl-fg);
  border: 1px solid var(--pl-border); padding: var(--pl-sp-xs) var(--pl-sp-sm); border-radius: var(--pl-r-sm); font-size: 12px; }
.pl-toolbar select { flex: 0 0 auto; max-width: 130px; }
.pl-btn { background: var(--pl-bg-elevated); color: var(--pl-fg); border: 1px solid var(--pl-border);
  padding: var(--pl-sp-xs) var(--pl-sp-sm); cursor: pointer;
  border-radius: var(--pl-r-sm); font-size: 12px; }
.pl-btn:hover { background: var(--pl-bg-hover); }
.pl-btn[disabled], .pl-btn.pl-busy { opacity: 0.5; cursor: progress; }
.pl-btn.pl-confirm-armed { background: var(--pl-danger); color: var(--pl-accent-fg); border-color: var(--pl-danger); }
.pl-grid { flex: 1 1 0; min-height: 0; overflow-y: auto; display: grid; gap: var(--pl-sp-sm); align-content: start;
  grid-template-columns: repeat(auto-fill, minmax(var(--pl-tile-size, 110px), 1fr));
  grid-auto-rows: max-content;
  padding-right: var(--pl-sp-2xs); }
.pl-tile { position: relative; display: flex; flex-direction: column;
  background: var(--pl-bg-elevated); border: 2px solid transparent;
  border-radius: var(--pl-r-sm); cursor: pointer; overflow: hidden;
  transition: border-color 80ms ease, transform 80ms ease; }
.pl-tile:hover { border-color: var(--pl-border-strong); transform: scale(1.02); }
.pl-tile.selected, .pl-tile.selected:hover { border-color: var(--pl-accent); }
.pl-tile.focused { box-shadow: 0 0 0 2px var(--pl-focus-outline) inset; }
.pl-tile.dragging { opacity: 0.4; }
/* Drop-target indicator — a 3px accent-coloured line on the LEFT edge
   of the target tile, conveying "the dragged tile will land before this
   one". Cleaner than outlining the entire tile (the previous behaviour),
   which made it ambiguous whether the drop replaces or sits next to. */
.pl-tile.drag-over { position: relative; }
.pl-tile.drag-over::before { content: ""; position: absolute;
  left: -4px; top: 0; bottom: 0; width: 3px; background: var(--pl-accent);
  border-radius: 2px; pointer-events: none; }
/* List view stacks tiles vertically, so the indicator becomes a top-edge
   horizontal line instead. */
.pl-grid.list-view .pl-tile.drag-over::before { left: 0; right: 0; bottom: auto;
  top: -2px; width: auto; height: 3px; }
.pl-tile-img { position: relative; width: 100%; height: 0; padding-bottom: 100%;
  overflow: hidden; background: var(--pl-bg-deep); flex: 0 0 auto; }
.pl-tile-img > img, .pl-tile-img > .pl-placeholder {
  position: absolute; inset: 0; }
.pl-tile-check { position: absolute; top: var(--pl-sp-xs); left: var(--pl-sp-xs); width: 16px; height: 16px;
  background: rgba(0,0,0,0.7); color: var(--pl-fg-strong); border: 1px solid var(--pl-fg-muted); border-radius: var(--pl-r-sm);
  display: none; align-items: center; justify-content: center; font-size: 11px;
  z-index: 1; cursor: pointer; user-select: none; }
.pl-tile:hover .pl-tile-check, .pl-tile.selected .pl-tile-check { display: flex; }
.pl-tile.selected .pl-tile-check { background: var(--pl-accent); color: var(--pl-accent-fg); border-color: var(--pl-accent); }
.pl-context-menu { position: fixed; z-index: 10001; background: var(--pl-bg-elevated); color: var(--pl-fg);
  border: 1px solid var(--pl-border); border-radius: var(--pl-r-sm); box-shadow: 0 4px 16px rgba(0,0,0,0.6);
  padding: var(--pl-sp-xs) 0; min-width: 140px; font-size: 12px; user-select: none; }
.pl-context-menu .item { padding: var(--pl-sp-sm) var(--pl-sp-md); cursor: pointer; }
.pl-context-menu .item:hover { background: var(--pl-bg-hover); }
.pl-context-menu .item.danger { color: var(--pl-danger); }
.pl-context-menu .sep { height: 1px; background: var(--pl-border); margin: 4px 0; }
.pl-context-menu .pl-ctx-stars { display: flex; align-items: center; gap: 2px; cursor: default; }
.pl-context-menu .pl-ctx-stars:hover { background: transparent; }
.pl-ctx-star { color: var(--pl-fg-muted); font-size: 14px; cursor: pointer; padding: 0 1px; }
.pl-ctx-star.on { color: var(--pl-rating); }
.pl-ctx-star:hover { color: var(--pl-rating); }
.pl-bulk-bar { display: flex; align-items: center; gap: var(--pl-sp-sm); padding: var(--pl-sp-sm);
  background: var(--pl-bg-selected); color: var(--pl-fg); border-radius: var(--pl-r-sm); font-size: 12px; }
.pl-bulk-bar .count { font-weight: bold; flex: 1; }
.pl-empty-state { grid-column: 1 / -1; padding: var(--pl-sp-xl) var(--pl-sp-md); text-align: center;
  color: var(--pl-fg-muted); font-size: 12px; line-height: 1.5; background: var(--pl-bg-empty);
  border: 1px dashed var(--pl-border); border-radius: var(--pl-r-sm); }
.pl-empty-state strong { color: var(--pl-fg); display: block; margin-bottom: 4px; font-size: 13px; }
.pl-search-wrap { position: relative; flex: 1; min-width: 0; display: flex; }
.pl-search-wrap input { width: 100%; padding-right: 22px; }
.pl-search-clear { position: absolute; right: 4px; top: 50%; transform: translateY(-50%);
  background: transparent; border: none; color: var(--pl-fg-muted); font-size: 14px;
  cursor: pointer; padding: 0 4px; line-height: 1; }
.pl-search-clear:hover { color: var(--pl-fg-strong); }
.pl-tile-size { display: flex; align-items: center; gap: 4px; }
.pl-tile-size input { width: 70px; }
.pl-tags-row { display: flex; flex-direction: column; gap: 3px; padding: 0 2px 2px; }
.pl-tag-group { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.pl-tag-group-label { color: var(--pl-fg-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
  margin-right: 2px; min-width: 60px; }
.pl-tag-chip { background: var(--pl-bg-elevated); color: var(--pl-fg); border: 1px solid var(--pl-border); padding: 2px 8px;
  border-radius: 10px; font-size: 11px; cursor: pointer; user-select: none; }
.pl-tag-chip:hover { background: var(--pl-bg-hover); }
.pl-tag-chip.active { background: var(--pl-bg-selected-strong); color: var(--pl-fg-strong); border-color: var(--pl-accent); }
.pl-tag-chip.all { font-weight: bold; }
.pl-tag-chip.pl-tag-mode { font-family: monospace; font-weight: bold; min-width: 36px; text-align: center; }
.pl-tile img { width: 100%; height: 100%; object-fit: cover; display: block; }
.pl-tile .pl-placeholder { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
  font-size: 22px; color: var(--pl-fg-placeholder); }
.pl-tile .pl-name { background: var(--pl-bg-input); color: var(--pl-fg); padding: var(--pl-sp-xs) var(--pl-sp-sm); font-size: 11px;
  line-height: 1.3; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  border-top: 1px solid var(--pl-border-soft); }
.pl-tile.selected .pl-name { background: var(--pl-bg-selected); color: var(--pl-fg-strong); }
.pl-add { aspect-ratio: 1 / 1; align-items: center; justify-content: center; font-size: 28px; color: var(--pl-fg-muted);
  background: var(--pl-bg-empty); border: 2px dashed var(--pl-border-strong); }
.pl-add:hover { color: var(--pl-fg); border-color: var(--pl-fg-muted); }
.pl-grid.list-view { grid-template-columns: 1fr; gap: 4px; grid-auto-rows: max-content; }
.pl-grid.list-view .pl-tile { flex-direction: row; align-items: stretch; min-height: 56px; }
.pl-grid.list-view .pl-tile-img { width: 56px !important; min-width: 56px; height: 56px !important;
  padding-bottom: 0 !important; flex: 0 0 56px !important; }
.pl-grid.list-view .pl-tile .pl-name { flex: 1; display: flex; align-items: center;
  padding: var(--pl-sp-sm) var(--pl-sp-md); font-size: 13px; border-top: none; border-left: 1px solid var(--pl-border-soft); }
.pl-view-toggle { display: flex; gap: 2px; }
.pl-view-toggle .pl-btn { padding: var(--pl-sp-xs) var(--pl-sp-sm); font-size: 13px; line-height: 1; }
.pl-view-toggle .pl-btn.active { background: var(--pl-bg-selected-strong); border-color: var(--pl-accent); color: var(--pl-fg-strong); }
.pl-modal { position: fixed; z-index: 10000; background: var(--pl-bg-elevated); color: var(--pl-fg);
  padding: 0 var(--pl-sp-lg) var(--pl-sp-lg);
  border-radius: var(--pl-r-md); width: min(540px, calc(100vw - var(--pl-sp-xl))); max-height: 80vh; overflow-y: auto;
  box-shadow: 0 8px 32px rgba(0,0,0,0.6); border: 1px solid var(--pl-border);
  display: flex; flex-direction: column; gap: var(--pl-sp-md); font-family: sans-serif; font-size: 13px; }
.pl-modal-header { position: sticky; top: 0; z-index: 1; }
.pl-modal-header { display: flex; align-items: center; gap: var(--pl-sp-sm); cursor: move;
  user-select: none; padding: var(--pl-sp-sm) var(--pl-sp-md); margin: 0 calc(var(--pl-sp-lg) * -1) var(--pl-sp-xs);
  background: var(--pl-bg-modal-header);
  border-radius: var(--pl-r-md) var(--pl-r-md) 0 0; border-bottom: 1px solid var(--pl-border); }
.pl-modal-header h3 { flex: 1; margin: 0; font-size: 13px; }
.pl-modal-close { background: transparent; border: none; color: var(--pl-fg-muted); font-size: 18px;
  line-height: 1; cursor: pointer; padding: 0 4px; }
.pl-modal-close:hover { color: var(--pl-fg-strong); }
.pl-modal label { display: flex; flex-direction: column; gap: var(--pl-sp-xs); font-size: 11px; color: var(--pl-fg-muted); }
.pl-modal input[type=text], .pl-modal textarea { background: var(--pl-bg-input); color: var(--pl-fg); border: 1px solid var(--pl-border);
  padding: var(--pl-sp-sm); border-radius: var(--pl-r-sm); font-size: 12px; font-family: inherit; }
.pl-modal textarea { resize: vertical; min-height: 100px; }
.pl-modal-actions { display: flex; justify-content: flex-end; gap: var(--pl-sp-sm); margin-top: var(--pl-sp-xs); }
.pl-modal-actions .danger { color: var(--pl-danger); border-color: #844; }
.pl-thumb-preview { max-width: 120px; max-height: 120px; object-fit: contain;
  background: var(--pl-bg-input); border: 1px solid var(--pl-border); border-radius: var(--pl-r-sm); display: block; }
.pl-status { font-size: 11px; color: var(--pl-fg-muted); min-height: 14px; }
.pl-status.error { color: var(--pl-danger); }
.pl-history { display: flex; flex-direction: column; gap: var(--pl-sp-sm); max-height: 240px;
  overflow-y: auto; padding: var(--pl-sp-xs); background: var(--pl-bg-input); border-radius: var(--pl-r-sm);
  margin-top: var(--pl-sp-xs); }
.pl-history-empty { color: var(--pl-fg-placeholder); font-size: 11px; padding: var(--pl-sp-sm); text-align: center; }
.pl-loras-section { display: flex; flex-direction: column; gap: var(--pl-sp-sm); padding: var(--pl-sp-md);
  background: var(--pl-bg-input); border: 1px solid var(--pl-border); border-radius: var(--pl-r-sm); }
.pl-loras-add-wrap { display: flex; flex-direction: column; gap: 2px; align-items: flex-start; }
.pl-loras-add { background: var(--pl-success); color: #0e1809; border: none;
  padding: var(--pl-sp-sm) var(--pl-sp-md);
  font-size: 13px; font-weight: 600; border-radius: var(--pl-r-sm); cursor: pointer; }
.pl-loras-add:hover { background: var(--pl-success-strong); }
.pl-loras-add[disabled] { opacity: 0.45; cursor: not-allowed; }
.pl-loras-add-help { font-size: 11px; color: var(--pl-fg-muted); }
.pl-loras-list { display: flex; flex-direction: column; gap: var(--pl-sp-sm); }
.pl-lora-row { display: grid;
  grid-template-columns: 60px minmax(120px, 2fr) minmax(110px, 1fr) minmax(100px, 1.2fr) auto;
  gap: var(--pl-sp-sm); align-items: end;
  padding: var(--pl-sp-sm); background: var(--pl-bg-elevated); border: 1px solid var(--pl-border-soft); border-radius: var(--pl-r-sm); }
.pl-lora-row label { font-size: 11px; color: var(--pl-fg-muted); text-transform: uppercase;
  letter-spacing: 0.4px; }
/* The number column has no label above its input, so its baseline sits 1
   row-of-label below everything else. Push it down to line up with the
   input row instead of bottom-aligning to the row edge. */
.pl-lora-num { font-size: 12px; color: var(--pl-fg); align-self: end; padding-bottom: 6px;
  font-weight: 600; }
.pl-lora-row select, .pl-lora-row input[type=text], .pl-lora-row input[type=number] {
  background: var(--pl-bg-input); color: var(--pl-fg); border: 1px solid var(--pl-border);
  padding: var(--pl-sp-xs) var(--pl-sp-sm); border-radius: var(--pl-r-sm); font-size: 11px; font-family: inherit; min-width: 0; width: 100%; box-sizing: border-box; }
.pl-lora-row select:disabled { opacity: 0.6; }
.pl-lora-strength { display: flex; flex-direction: column; gap: var(--pl-sp-xs); }
.pl-lora-strength-bar { display: flex; align-items: center; gap: var(--pl-sp-xs); }
.pl-lora-strength-tag { display: inline-block; width: 14px; min-width: 14px;
  font-size: 10px; font-weight: 600; color: var(--pl-fg-muted); text-align: center;
  font-family: monospace; }
.pl-lora-strength-header { display: flex; align-items: center; justify-content: space-between; }
.pl-lora-strength-link { background: transparent; color: var(--pl-fg-muted);
  border: 1px solid var(--pl-border); border-radius: var(--pl-r-sm); padding: 1px 5px;
  font-size: 11px; cursor: pointer; line-height: 1; }
.pl-lora-strength-link:hover { color: var(--pl-fg-strong); border-color: var(--pl-fg-muted); }
.pl-lora-strength-link.linked { color: var(--pl-accent); border-color: var(--pl-accent); }
.pl-lora-missing-warn { font-size: 10px; color: var(--pl-danger); padding: 2px 4px;
  background: rgba(255, 100, 100, 0.08); border-left: 2px solid var(--pl-danger);
  border-radius: 0 2px 2px 0; line-height: 1.3; }
/* Library validator: warning badge + modal section listing the findings. */
.pl-health-btn { color: var(--pl-danger); border-color: var(--pl-danger);
  background: rgba(255, 100, 100, 0.06); font-weight: 600; padding: var(--pl-sp-xs) var(--pl-sp-sm); }
.pl-health-btn:hover { background: rgba(255, 100, 100, 0.14);
  color: var(--pl-fg-strong); }
.pl-validator-section { display: flex; flex-direction: column; gap: var(--pl-sp-xs);
  border: 1px solid var(--pl-border-soft); border-radius: var(--pl-r-sm); padding: var(--pl-sp-sm);
  background: var(--pl-bg-input); }
.pl-validator-section-head { font-weight: 600; color: var(--pl-fg-strong);
  font-size: 12px; }
.pl-validator-list { display: flex; flex-direction: column; gap: var(--pl-sp-xs);
  max-height: 220px; overflow-y: auto; }
.pl-validator-row { display: flex; align-items: center; gap: var(--pl-sp-sm); padding: var(--pl-sp-xs);
  font-size: 11px; color: var(--pl-fg); border-radius: var(--pl-r-sm);
  background: var(--pl-bg-elevated); }
.pl-validator-row > span { flex: 1 1 auto; word-break: break-all; }
.pl-validator-row .pl-btn { padding: 2px 8px; font-size: 11px; flex: 0 0 auto; }
.pl-validator-more { font-size: 10px; color: var(--pl-fg-muted);
  padding: var(--pl-sp-xs); text-align: center; }
/* Custom-styled range input — the browser default is a near-invisible thin
   line. Track is a 4px green-on-grey bar; thumb is a 14px green disc. */
.pl-lora-strength-bar input[type=range] { flex: 1 1 0; min-width: 0; -webkit-appearance: none;
  appearance: none; height: 4px; background: var(--pl-border); border-radius: 2px; outline: none;
  padding: 0; cursor: pointer; }
.pl-lora-strength-bar input[type=range]::-webkit-slider-thumb { -webkit-appearance: none;
  appearance: none; width: 14px; height: 14px; border-radius: 50%; background: var(--pl-success);
  border: 1px solid var(--pl-success-edge); cursor: grab; }
.pl-lora-strength-bar input[type=range]::-moz-range-thumb { width: 14px; height: 14px;
  border-radius: 50%; background: var(--pl-success); border: 1px solid var(--pl-success-edge); cursor: grab; }
.pl-lora-strength-bar input[type=range]::-moz-range-track { background: var(--pl-border);
  height: 4px; border-radius: 2px; }
.pl-lora-strength-bar input[type=number] { width: 60px; flex: 0 0 auto; }
.pl-lora-row .pl-lora-delete { background: transparent; color: var(--pl-fg-muted);
  border: 1px solid var(--pl-border); padding: var(--pl-sp-xs) var(--pl-sp-md); font-size: 11px;
  border-radius: var(--pl-r-sm); cursor: pointer; align-self: end; }
.pl-lora-row .pl-lora-delete:hover { color: var(--pl-danger); border-color: var(--pl-danger); }
/* Soft-delete state: row stays visible but greyed + struck through until
   Save commits or Restore reverses. */
.pl-lora-row-deleted { opacity: 0.4; }
.pl-lora-row-deleted .pl-lora-num,
.pl-lora-row-deleted select, .pl-lora-row-deleted input[type=text],
.pl-lora-row-deleted input[type=number] { text-decoration: line-through; }
.pl-lora-row-deleted .pl-lora-delete { color: var(--pl-accent);
  border-color: var(--pl-accent); }
.pl-lora-disabled-toggle { display: flex; align-items: center; gap: 4px; font-size: 11px;
  color: var(--pl-fg-muted); }
.pl-lora-disabled-toggle input { accent-color: var(--pl-success); }
.pl-toast-stack { position: fixed; right: max(var(--pl-sp-lg), 1vw); top: var(--pl-sp-lg); z-index: 10002;
  display: flex; flex-direction: column; gap: var(--pl-sp-sm); max-width: min(360px, calc(100vw - var(--pl-sp-xl))); pointer-events: none; }
.pl-toast { background: var(--pl-bg-elevated, #2a2a2a); color: var(--pl-fg, #ddd);
  border: 1px solid var(--pl-border, #444); border-radius: var(--pl-r-sm); padding: var(--pl-sp-sm) var(--pl-sp-md);
  box-shadow: 0 4px 16px rgba(0,0,0,0.6); font-size: 12px; line-height: 1.4;
  pointer-events: auto; max-width: 360px; word-wrap: break-word;
  animation: pl-toast-in 140ms ease-out; }
.pl-toast.error { border-left: 4px solid var(--pl-danger, #f88); }
.pl-toast.success { border-left: 4px solid var(--pl-accent, #6cf); }
@keyframes pl-toast-in { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: none; } }
.pl-rating-input { display: flex; gap: 2px; }
.pl-star { background: transparent; border: none; color: var(--pl-fg-muted); padding: 0 2px;
  font-size: 18px; line-height: 1; cursor: pointer; }
.pl-star.on { color: var(--pl-rating); }
.pl-star:hover { color: var(--pl-rating); }
.pl-rating-badge { position: absolute; bottom: var(--pl-sp-xs); right: var(--pl-sp-xs);
  background: rgba(0, 0, 0, 0.7); color: var(--pl-rating); font-size: 10px;
  padding: 1px var(--pl-sp-xs); border-radius: var(--pl-r-sm); letter-spacing: 1px;
  pointer-events: none; z-index: 1; }
.pl-grid.list-view .pl-rating-badge { position: static; flex: 0 0 auto; align-self: center;
  margin-right: var(--pl-sp-sm); }
.pl-notes { min-height: 40px !important; max-height: 100px; }
.pl-count-badge { font-size: 11px; color: var(--pl-fg-muted); padding: 0 4px; white-space: nowrap; }
.pl-fav-btn { font-size: 14px; line-height: 1; padding: var(--pl-sp-xs) var(--pl-sp-sm); }
.pl-fav-btn.active { color: var(--pl-rating); border-color: var(--pl-rating); background: var(--pl-bg-input); }
.pl-comic { display: flex; flex-direction: column; gap: var(--pl-sp-sm); padding: var(--pl-sp-sm); box-sizing: border-box;
  width: 100%; height: 100%; min-height: 0; color: var(--pl-fg); font-family: sans-serif; font-size: 12px; }
.pl-comic-header { display: flex; align-items: center; gap: var(--pl-sp-sm); padding: var(--pl-sp-2xs) 0;
  border-bottom: 1px solid var(--pl-border); margin-bottom: var(--pl-sp-xs); }
.pl-comic-header strong { flex: 1; color: var(--pl-fg); font-size: 12px; }
.pl-comic-header .pl-count-badge { color: var(--pl-fg-muted); }
.pl-comic-frames { flex: 1 1 0; min-height: 0; overflow-y: auto; display: flex;
  flex-direction: column; gap: var(--pl-sp-xs); padding-right: var(--pl-sp-2xs); }
.pl-frame-row { display: flex; gap: var(--pl-sp-xs); align-items: stretch;
  background: var(--pl-bg-elevated); border: 1px solid var(--pl-border);
  border-radius: var(--pl-r-sm); padding: var(--pl-sp-xs); }
.pl-frame-row.current { border-color: var(--pl-accent); background: var(--pl-bg-selected); }
.pl-frame-num { flex: 0 0 22px; display: flex; align-items: center; justify-content: center;
  font-weight: bold; color: var(--pl-fg-muted); font-size: 11px; cursor: grab; user-select: none; }
.pl-frame-row.current .pl-frame-num { color: var(--pl-accent); }
.pl-frame-text { flex: 1; min-width: 0; background: var(--pl-bg-input); color: var(--pl-fg);
  border: 1px solid var(--pl-border); border-radius: var(--pl-r-sm); padding: var(--pl-sp-xs) var(--pl-sp-sm);
  font-family: inherit; font-size: 12px; resize: vertical; min-height: 32px; }
.pl-frame-actions { display: flex; flex-direction: column; gap: 2px; flex: 0 0 auto; }
.pl-frame-actions .pl-btn { padding: 1px 6px; font-size: 11px; line-height: 1; }
.pl-comic-toolbar { display: flex; gap: var(--pl-sp-sm); align-items: center; flex: 0 0 auto; }
.pl-bg-locked-wrap { display: flex; align-items: center; gap: var(--pl-sp-sm); padding: var(--pl-sp-xs) var(--pl-sp-sm);
  background: var(--pl-bg-selected); color: var(--pl-fg); border-radius: var(--pl-r-sm);
  border-left: 4px solid var(--pl-accent); font-size: 11px; }
.pl-bg-locked-wrap .lock { font-size: 14px; }
.pl-history-row { display: grid; grid-template-columns: auto 1fr auto; gap: var(--pl-sp-sm);
  align-items: start; padding: var(--pl-sp-sm); background: var(--pl-bg-elevated); border-radius: var(--pl-r-sm);
  font-size: 11px; }
.pl-history-ts { color: var(--pl-fg-muted); white-space: nowrap; }
.pl-history-body { color: var(--pl-fg); word-break: break-word; min-width: 0; }
.pl-history-body strong { color: var(--pl-fg-strong); display: block; margin-bottom: 2px; }
.pl-history-tags { color: var(--pl-accent); font-size: 10px; margin-top: 2px; }
.pl-history-row button { font-size: 10px; padding: var(--pl-sp-2xs) var(--pl-sp-sm); }

/* Smart Detailer per-target grid — 7 columns (label + 6 categories) × 6
   rows (enable + 5 numeric stats). Implemented as a row-container pattern
   per modern data-grid conventions (Linear / Airtable / Tailwind UI):
   each row is its own grid track; rows separate via subtle 1px borders
   (not heavy lines); hover lifts row bg. No bg tiles on labels — they
   sit on the row bg with type alone differentiating them. */
.pl-det-grid { display: flex; flex-direction: column;
  background: #1c1c1c; border: 1px solid #3a3a3a;
  border-radius: 6px; overflow: hidden;
  font-size: 11px; box-sizing: border-box; }
.pl-det-row { display: grid; grid-template-columns: 70px repeat(6, 1fr);
  gap: var(--pl-sp-sm); align-items: center; flex-shrink: 0;
  min-height: 38px; padding: var(--pl-sp-sm) var(--pl-sp-md); }
.pl-det-row + .pl-det-row { border-top: 1px solid #3a3a3a; }
.pl-det-row-header { background: #141414; min-height: 32px;
  padding: var(--pl-sp-xs) var(--pl-sp-md); }
.pl-det-row-header + .pl-det-row { border-top: 1px solid #555; }
.pl-det-row-data { transition: background 80ms ease;
  gap: 0; padding-left: 0; padding-right: 0; }
.pl-det-row-data:hover { background: rgba(255, 255, 255, 0.03); }
.pl-det-row-data > .pl-det-l {
  padding-left: var(--pl-sp-md); padding-right: var(--pl-sp-sm); }
.pl-det-cell { display: flex; align-items: center; justify-content: center;
  min-width: 0; padding: 0 var(--pl-sp-sm); }
.pl-det-row-data > .pl-det-cell:last-child { padding-right: var(--pl-sp-md); }
.pl-det-row-data > * + * { border-left: 1px solid #2a2a2a; }
.pl-det-corner { background: transparent; }
/* Category headers — medium-saturation tinted bg + dark text.
   HSL ~45% saturation, 50% lightness. */
.pl-det-h { font-weight: 700; text-align: center; padding: var(--pl-sp-xs) 0;
  border-radius: var(--pl-r-sm); color: #0c0c0c; letter-spacing: 0.5px;
  font-size: 11px; text-transform: uppercase; }
.pl-det-h.face  { background: #479f5b; }
.pl-det-h.eyes  { background: #479ebb; }
.pl-det-h.feet  { background: #7b47bb; color: #fff; }
.pl-det-h.hands { background: #bb479e; }
.pl-det-l { color: var(--pl-fg); cursor: help;
  text-align: right; font-variant-numeric: tabular-nums;
  font-weight: 500; padding-right: var(--pl-sp-xs); }
.pl-det-i { width: 100%; padding: var(--pl-sp-xs) var(--pl-sp-sm); box-sizing: border-box;
  background: var(--pl-bg-elevated); color: var(--pl-fg-strong); border: 1px solid var(--pl-border);
  border-radius: var(--pl-r-sm); text-align: center; font-size: 11px;
  font-family: inherit; }
.pl-det-i:hover { border-color: var(--pl-border-strong); }
.pl-det-i:focus { border-color: var(--pl-accent); outline: none;
  background: var(--pl-bg-input); }
.pl-det-i.dim { color: var(--pl-fg-muted); font-style: italic; }
.pl-det-toggle { width: 16px; height: 16px; accent-color: var(--pl-accent);
  margin: 0 auto; display: block; cursor: pointer; }
/* Preset toolbar — sits above the per-target grid header. Picker + small
   buttons live on one row; the inline save form drops down below when
   active (display:flex toggled in JS). */
.pl-det-presets { display: flex; flex-wrap: wrap; align-items: center;
  gap: var(--pl-sp-sm); padding: var(--pl-sp-sm) var(--pl-sp-md);
  background: #161616; border-bottom: 1px solid #3a3a3a;
  font-size: 11px; color: var(--pl-fg); }
.pl-det-presets-label { font-weight: 600; opacity: 0.85; }
.pl-det-presets select { flex: 1 1 140px; min-width: 0;
  background: var(--pl-bg-elevated); color: var(--pl-fg-strong);
  border: 1px solid var(--pl-border); border-radius: var(--pl-r-sm);
  padding: 3px 6px; font-size: 11px; font-family: inherit; }
.pl-det-presets button { padding: 3px 8px; font-size: 11px;
  background: var(--pl-bg-elevated); color: var(--pl-fg-strong);
  border: 1px solid var(--pl-border); border-radius: var(--pl-r-sm);
  cursor: pointer; font-family: inherit; }
.pl-det-presets button:hover { border-color: var(--pl-accent); }
.pl-det-presets button:disabled { opacity: 0.45; cursor: not-allowed; }
.pl-det-presets button.danger:hover { border-color: #cc4747; color: #ff8a8a; }
.pl-det-preset-status { font-size: 10px; opacity: 0.7;
  flex-basis: 100%; padding-left: 2px; }
.pl-det-preset-save-form { display: none; flex-basis: 100%; gap: var(--pl-sp-sm);
  align-items: center; padding-top: var(--pl-sp-xs); }
.pl-det-preset-save-form.open { display: flex; flex-wrap: wrap; }
.pl-det-preset-save-form input[type="text"] {
  flex: 1 1 140px; min-width: 0;
  background: var(--pl-bg-input); color: var(--pl-fg-strong);
  border: 1px solid var(--pl-border); border-radius: var(--pl-r-sm);
  padding: 3px 6px; font-size: 11px; font-family: inherit; }
/* Custom tooltip — attached to body so it escapes Comfy's widget layer
   z-index/clipping. Native HTML title attributes are unreliable inside
   Comfy's Vue-wrapped DOM widgets (the canvas captures pointer events
   for graph interaction, breaking native tooltip detection), so we
   drive show/hide ourselves via pointerenter/leave. */
.pl-tooltip { position: fixed; z-index: 100000;
  background: #1a1a1a; color: #e5e7eb;
  border: 1px solid #3a3a3a; border-radius: 4px;
  padding: 6px 10px; font-size: 11px; line-height: 1.45;
  max-width: 320px; pointer-events: none;
  opacity: 0; transition: opacity 100ms ease;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6);
  font-family: sans-serif; white-space: normal; }
.pl-tooltip.show { opacity: 1; }
/* Civitai save folder picker — inline bar in the node + modal browser */
.pl-civ-pathbar { display: flex; flex-wrap: wrap; align-items: center;
  gap: var(--pl-sp-sm); padding: var(--pl-sp-xs) var(--pl-sp-sm);
  background: #161616; border: 1px solid #3a3a3a; border-radius: var(--pl-r-sm);
  font-size: 11px; color: var(--pl-fg); margin-top: var(--pl-sp-xs);
  box-sizing: border-box; }
.pl-civ-pathbar .pl-civ-current { flex: 1 1 auto; min-width: 0;
  font-family: monospace; color: var(--pl-fg-strong);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pl-civ-pathbar button { padding: 3px 8px; font-size: 11px;
  background: var(--pl-bg-elevated); color: var(--pl-fg-strong);
  border: 1px solid var(--pl-border); border-radius: var(--pl-r-sm);
  cursor: pointer; font-family: inherit; }
.pl-civ-pathbar button:hover { border-color: var(--pl-accent); }
.pl-browser-overlay { position: fixed; inset: 0; z-index: 11000;
  background: rgba(0, 0, 0, 0.55); display: flex;
  align-items: center; justify-content: center; }
.pl-browser-modal { width: min(640px, 92vw); max-height: 80vh;
  background: var(--pl-bg-deep, #1a1a1a); color: var(--pl-fg);
  border: 1px solid var(--pl-border, #3a3a3a); border-radius: 8px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.7); display: flex; flex-direction: column;
  font-family: sans-serif; font-size: 12px; overflow: hidden; }
.pl-browser-modal header { padding: var(--pl-sp-md);
  border-bottom: 1px solid var(--pl-border); display: flex;
  flex-direction: column; gap: var(--pl-sp-sm); }
.pl-browser-modal header h3 { margin: 0; font-size: 13px; font-weight: 600; }
.pl-browser-roots { display: flex; flex-wrap: wrap; gap: var(--pl-sp-xs); }
.pl-browser-roots button { padding: 2px 8px; font-size: 11px;
  background: var(--pl-bg-elevated); color: var(--pl-fg);
  border: 1px solid var(--pl-border); border-radius: 12px;
  cursor: pointer; font-family: inherit; }
.pl-browser-roots button:hover { border-color: var(--pl-accent); }
.pl-browser-pathrow { display: flex; align-items: stretch;
  gap: var(--pl-sp-xs); }
.pl-browser-pathrow input { flex: 1 1 auto; min-width: 0;
  background: var(--pl-bg-input); color: var(--pl-fg-strong);
  border: 1px solid var(--pl-border); border-radius: var(--pl-r-sm);
  padding: 4px 8px; font-size: 11px; font-family: monospace; }
.pl-browser-pathrow button { padding: 3px 10px; font-size: 11px;
  background: var(--pl-bg-elevated); color: var(--pl-fg-strong);
  border: 1px solid var(--pl-border); border-radius: var(--pl-r-sm);
  cursor: pointer; font-family: inherit; }
.pl-browser-list { flex: 1 1 auto; min-height: 200px; overflow-y: auto;
  padding: var(--pl-sp-sm); display: flex; flex-direction: column;
  gap: 2px; }
.pl-browser-row { padding: 4px 8px; border-radius: var(--pl-r-sm);
  cursor: pointer; display: flex; align-items: center; gap: var(--pl-sp-sm);
  user-select: none; }
.pl-browser-row:hover { background: rgba(255, 255, 255, 0.05); }
.pl-browser-row.parent { color: var(--pl-fg-muted); font-style: italic; }
.pl-browser-row .icon { width: 16px; text-align: center; opacity: 0.8; }
.pl-browser-status { padding: 4px var(--pl-sp-md); font-size: 11px;
  color: var(--pl-fg-muted); border-top: 1px solid var(--pl-border); }
.pl-browser-status.error { color: #ff8a8a; }
.pl-browser-modal footer { padding: var(--pl-sp-md);
  border-top: 1px solid var(--pl-border); display: flex; gap: var(--pl-sp-sm);
  align-items: center; justify-content: flex-end; }
.pl-browser-modal footer button { padding: 5px 14px; font-size: 12px;
  background: var(--pl-bg-elevated); color: var(--pl-fg-strong);
  border: 1px solid var(--pl-border); border-radius: var(--pl-r-sm);
  cursor: pointer; font-family: inherit; }
.pl-browser-modal footer button.primary { background: var(--pl-accent);
  color: var(--pl-accent-fg, #0a0a0a); border-color: var(--pl-accent); }
.pl-browser-modal footer button.primary:hover { filter: brightness(1.1); }
.pl-browser-mkdir { display: flex; gap: var(--pl-sp-xs); align-items: center;
  margin-right: auto; }
.pl-browser-mkdir input { flex: 0 0 160px;
  background: var(--pl-bg-input); color: var(--pl-fg-strong);
  border: 1px solid var(--pl-border); border-radius: var(--pl-r-sm);
  padding: 4px 8px; font-size: 11px; font-family: inherit; }
`;

function injectStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const el = document.createElement("style");
  el.id = STYLE_ID;
  el.textContent = CSS;
  document.head.appendChild(el);
}

const TOAST_STACK_ID = "pl-toast-stack";
function toast(message, kind = "info", durationMs = 4000) {
  let stack = document.getElementById(TOAST_STACK_ID);
  if (!stack) {
    stack = document.createElement("div");
    stack.id = TOAST_STACK_ID;
    stack.className = "pl-toast-stack";
    document.body.appendChild(stack);
  }
  const t = document.createElement("div");
  t.className = `pl-toast ${kind}`;
  t.setAttribute("role", kind === "error" ? "alert" : "status");
  t.textContent = message;
  t.addEventListener("click", () => t.remove());
  stack.appendChild(t);
  setTimeout(() => t.remove(), durationMs);
  return t;
}

// Wrap an async button handler so the button is disabled and shows a busy
// label while the work runs. Returns a function suitable for assignment to
// btn.onclick. Restores the original label even if the handler throws.
function withBusy(btn, busyLabel, fn) {
  return async (...args) => {
    if (btn.disabled) return;
    const originalLabel = btn.textContent;
    btn.disabled = true;
    btn.classList.add("pl-busy");
    if (busyLabel) btn.textContent = busyLabel;
    try {
      return await fn(...args);
    } finally {
      btn.disabled = false;
      btn.classList.remove("pl-busy");
      btn.textContent = originalLabel;
    }
  };
}

function confirmDestructive(message, { confirmLabel = "Delete", timeoutMs = 8000 } = {}) {
  return new Promise((resolve) => {
    let stack = document.getElementById(TOAST_STACK_ID);
    if (!stack) {
      stack = document.createElement("div");
      stack.id = TOAST_STACK_ID;
      stack.className = "pl-toast-stack";
      document.body.appendChild(stack);
    }
    const t = document.createElement("div");
    t.className = "pl-toast error";
    t.setAttribute("role", "alertdialog");
    const msg = document.createElement("div");
    msg.textContent = message;
    msg.style.marginBottom = "8px";
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex; gap:6px; justify-content:flex-end;";
    const no = document.createElement("button");
    no.className = "pl-btn";
    no.textContent = "Cancel";
    const yes = document.createElement("button");
    yes.className = "pl-btn pl-confirm-armed";
    yes.textContent = confirmLabel;
    let done = false;
    const finish = (v) => { if (done) return; done = true; t.remove(); resolve(v); };
    no.onclick = () => finish(false);
    yes.onclick = () => finish(true);
    actions.append(no, yes);
    t.append(msg, actions);
    stack.appendChild(t);
    yes.focus();
    setTimeout(() => finish(false), timeoutMs);
  });
}

async function fetchValidate() {
  const res = await api.fetchApi("/prompt_library/validate");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function fixOrphans() {
  const res = await api.fetchApi("/prompt_library/fix_orphans",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function fetchList() {
  const res = await api.fetchApi("/prompt_library/list");
  const data = await res.json();
  return data.prompts || [];
}

function slugify(name) {
  return (name || "").trim().toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
}

async function upsert({ id, name, text, negative, tags, rating, notes, loras, imageFile, clearImage }) {
  const body = new FormData();
  if (id) body.append("id", id);
  body.append("name", name);
  body.append("text", text);
  if (negative !== undefined) body.append("negative", negative);
  if (tags !== undefined) body.append("tags", tags);
  if (rating !== undefined && rating !== null) body.append("rating", String(rating));
  if (notes !== undefined) body.append("notes", notes);
  // The loras array is JSON-encoded into a single form field — multipart can't
  // easily express a list of objects natively, and the backend already
  // distinguishes "key omitted" (loras=undefined) from "explicit empty"
  // (loras=[]) so we only attach it when the modal actually rendered the section.
  if (loras !== undefined) body.append("loras", JSON.stringify(loras));
  if (clearImage) body.append("clear_image", "1");
  if (imageFile) body.append("image", imageFile, imageFile.name);
  const res = await api.fetchApi("/prompt_library/upsert", { method: "POST", body });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

// Cache the LoRA list across modal opens — fetched once per session unless
// invalidated by a 'lora_library.updated' websocket event (none currently
// emitted; the cache lives only as long as the page does, which is fine).
let _loraListPromise = null;
function loadLoraList() {
  if (_loraListPromise === null) {
    _loraListPromise = api.fetchApi("/prompt_library/loras")
      .then(r => r.ok ? r.json() : { loras: [] })
      .then(d => Array.isArray(d.loras) ? d.loras : [])
      .catch(() => []);
  }
  return _loraListPromise;
}

async function deletePrompt(id) {
  const res = await api.fetchApi("/prompt_library/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function importCsv(file, mode = "add_only") {
  const body = new FormData();
  body.append("file", file, file.name);
  body.append("mode", mode);
  const res = await api.fetchApi("/prompt_library/import_csv", { method: "POST", body });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

async function importZip(file, mode = "add_only") {
  const body = new FormData();
  body.append("file", file, file.name);
  body.append("mode", mode);
  const res = await api.fetchApi("/prompt_library/import_zip", { method: "POST", body });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

async function restoreLastSnapshot() {
  const res = await api.fetchApi("/prompt_library/restore_snapshot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

async function exportZip(ids) {
  const res = await api.fetchApi("/prompt_library/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids: ids || [] }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const count = parseInt(res.headers.get("X-GrimmRibbity-Count") || res.headers.get("X-Ribbity-Count") || "0", 10);
  const blob = await res.blob();
  return { blob, count };
}

async function bulkDelete(ids) {
  const res = await api.fetchApi("/prompt_library/bulk_delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function duplicatePrompt(id, name) {
  const res = await api.fetchApi("/prompt_library/duplicate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, name }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

async function reorderPrompts(ids) {
  const res = await api.fetchApi("/prompt_library/reorder", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

let _activeContextMenu = null;
function openContextMenu(x, y, items) {
  if (_activeContextMenu) _activeContextMenu.remove();
  const menu = document.createElement("div");
  menu.className = "pl-context-menu";
  const closeMenu = () => { menu.remove(); _activeContextMenu = null; };
  for (const entry of items) {
    if (entry === "sep") {
      const sep = document.createElement("div");
      sep.className = "sep";
      menu.appendChild(sep);
      continue;
    }
    // Special "stars" entry: 5 inline clickable stars for quick rating.
    // {kind:"stars", label, current, action(n)}
    if (entry?.kind === "stars") {
      const row = document.createElement("div");
      row.className = "item pl-ctx-stars";
      const lbl = document.createElement("span");
      lbl.textContent = entry.label;
      lbl.style.flex = "1";
      row.appendChild(lbl);
      for (let i = 1; i <= 5; i++) {
        const star = document.createElement("span");
        star.className = "pl-ctx-star" + (i <= (entry.current || 0) ? " on" : "");
        star.textContent = i <= (entry.current || 0) ? "★" : "☆";
        star.title = `${i} star${i === 1 ? "" : "s"}`;
        star.onclick = (ev) => {
          ev.stopPropagation();
          closeMenu();
          // Click same rating to clear.
          entry.action(i === (entry.current || 0) ? 0 : i);
        };
        row.appendChild(star);
      }
      menu.appendChild(row);
      continue;
    }
    const el = document.createElement("div");
    el.className = "item" + (entry.danger ? " danger" : "");
    el.textContent = entry.label;
    el.onclick = () => {
      closeMenu();
      entry.action();
    };
    menu.appendChild(el);
  }
  // Position; clamp to viewport.
  document.body.appendChild(menu);
  const rect = menu.getBoundingClientRect();
  const left = Math.min(x, window.innerWidth - rect.width - 4);
  const top = Math.min(y, window.innerHeight - rect.height - 4);
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
  _activeContextMenu = menu;
  const dismiss = (e) => {
    if (!menu.contains(e.target)) {
      menu.remove();
      _activeContextMenu = null;
      document.removeEventListener("mousedown", dismiss, true);
    }
  };
  setTimeout(() => document.addEventListener("mousedown", dismiss, true), 0);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

async function fetchHistory(id) {
  const res = await api.fetchApi(`/prompt_library/history/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function revertPrompt(id, ts) {
  const res = await api.fetchApi("/prompt_library/revert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, ts }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

function relativeTime(ts) {
  const sec = Math.max(0, Date.now() / 1000 - ts);
  if (sec < 60) return `${Math.round(sec)}s ago`;
  if (sec < 3600) return `${Math.round(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.round(sec / 3600)}h ago`;
  return `${Math.round(sec / 86400)}d ago`;
}

const SORT_MODES = {
  manual:      { label: "Manual",      cmp: (a, b) => (a.order||0) - (b.order||0) },
  name_asc:    { label: "Name A-Z",    cmp: (a, b) => a.name.localeCompare(b.name) },
  name_desc:   { label: "Name Z-A",    cmp: (a, b) => b.name.localeCompare(a.name) },
  newest:      { label: "Newest",      cmp: (a, b) => (b.created_at||0) - (a.created_at||0) },
  oldest:      { label: "Oldest",      cmp: (a, b) => (a.created_at||0) - (b.created_at||0) },
  recent_edit: { label: "Recent edit", cmp: (a, b) => (b.updated_at||0) - (a.updated_at||0) },
  rating_desc: { label: "Top rated",   cmp: (a, b) => (b.rating||0) - (a.rating||0) || a.name.localeCompare(b.name) },
};
const SORT_KEY = "comfy.PromptLibrary.sort";

// Cache-buster shared across the page. Bumped on websocket update events;
// keeps the same value across re-renders so the browser can cache thumbnails
// instead of re-downloading them every time the gallery refreshes.
let _imageCacheKey = Date.now();
function imageUrl(id) {
  const path = `/prompt_library/image/${id}?v=${_imageCacheKey}`;
  return api.apiURL ? api.apiURL(path) : path;
}

let _modalStack = 0;

const LORAS_PER_ENTRY_CAP = 10;

// Chunked-render thresholds for the gallery grid. Below the threshold
// every tile is built in one frame (the v0.36-39 path). Above it, the
// first _TILES_PER_CHUNK tiles render synchronously (above-fold paint
// stays instant) and the rest stream in via requestIdleCallback so a
// 1000+ tile library doesn't block the main thread for 100+ ms.
const _CHUNKED_RENDER_THRESHOLD = 200;
const _TILES_PER_CHUNK = 60;

function _scheduleIdle(fn) {
  if (typeof requestIdleCallback === "function") {
    requestIdleCallback(fn, { timeout: 100 });
  } else {
    setTimeout(fn, 0);
  }
}

// Custom tooltip — required because native HTML `title` is unreliable
// inside Comfy's Vue-wrapped DOM-widget overlay. The canvas underneath
// the overlay captures pointer events, so the browser rarely registers
// the sustained hover the native tooltip needs. We drive show/hide
// ourselves on pointerenter/leave; one shared element lives on <body>.
let _plTooltipEl = null;
let _plTooltipTimer = null;
function _plEnsureTooltip() {
  if (_plTooltipEl) return _plTooltipEl;
  _plTooltipEl = document.createElement("div");
  _plTooltipEl.className = "pl-tooltip";
  document.body.appendChild(_plTooltipEl);
  return _plTooltipEl;
}
function _plPositionTooltip(target) {
  const tip = _plTooltipEl;
  if (!tip) return;
  const r = target.getBoundingClientRect();
  // Render off-screen to measure, then place under the target, centered.
  tip.style.left = "-9999px";
  tip.style.top = "0";
  tip.classList.add("show");
  const tw = tip.offsetWidth;
  const th = tip.offsetHeight;
  let left = r.left + r.width / 2 - tw / 2;
  let top = r.bottom + 6;
  // Clamp to viewport with 8px padding; flip above target if no room below.
  if (left < 8) left = 8;
  if (left + tw > window.innerWidth - 8) left = window.innerWidth - tw - 8;
  if (top + th > window.innerHeight - 8) top = r.top - th - 6;
  tip.style.left = left + "px";
  tip.style.top = top + "px";
}
function _plAttachTip(target, text) {
  if (!text || !target) return;
  target.addEventListener("pointerenter", () => {
    clearTimeout(_plTooltipTimer);
    _plTooltipTimer = setTimeout(() => {
      const tip = _plEnsureTooltip();
      tip.textContent = text;
      _plPositionTooltip(target);
    }, 350);
  });
  target.addEventListener("pointerleave", () => {
    clearTimeout(_plTooltipTimer);
    if (_plTooltipEl) _plTooltipEl.classList.remove("show");
  });
}

function buildLoraSection(initialLoras) {
  // The whole section: green "+ Add LoRA" CTA + helper line + list of rows.
  // Rows are hidden until "+ Add LoRA" is pressed. getValue() reads the
  // current rows back out as a clean JSON-friendly list.
  const wrap = document.createElement("div");
  wrap.className = "pl-loras-section";

  const heading = document.createElement("div");
  heading.style.fontSize = "12px";
  heading.style.color = "var(--pl-fg)";
  heading.style.fontWeight = "600";
  heading.textContent = "LoRAs";
  // Tiny disclosure under the heading so users know which node consumes
  // these — the STRING-output Library node ignores them on purpose.
  const headingHint = document.createElement("div");
  headingHint.style.fontSize = "10px";
  headingHint.style.color = "var(--pl-fg-muted)";
  headingHint.style.marginTop = "-4px";
  headingHint.textContent = "Applied by the Style node when this entry is selected.";

  const addWrap = document.createElement("div");
  addWrap.className = "pl-loras-add-wrap";
  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "pl-loras-add";
  addBtn.textContent = "+ Add LoRA";
  const addHelp = document.createElement("div");
  addHelp.className = "pl-loras-add-help";
  addHelp.textContent = "Adds another line to load a lora";
  addWrap.append(addBtn, addHelp);

  const list = document.createElement("div");
  list.className = "pl-loras-list";

  // The dropdown options are filled in lazily — we keep one shared <datalist>
  // populated once the fetch resolves, and rebuild every row's <select> from
  // it so the user sees a populated picker without waiting on the network.
  const rows = [];
  let loraNames = [];
  const loraNamesPromise = loadLoraList().then(names => {
    loraNames = names;
    for (const row of rows) row.refreshOptions(loraNames);
  });

  const renumber = () => {
    for (let i = 0; i < rows.length; i++) {
      rows[i].setIndex(i + 1);
    }
    addBtn.disabled = rows.length >= LORAS_PER_ENTRY_CAP;
    list.style.display = rows.length ? "" : "none";
    heading.style.display = rows.length ? "" : "none";
    headingHint.style.display = rows.length ? "" : "none";
  };

  const buildRow = (initial) => {
    const row = document.createElement("div");
    row.className = "pl-lora-row";

    const numCell = document.createElement("div");
    numCell.className = "pl-lora-num";

    const modelCell = document.createElement("div");
    modelCell.style.display = "flex";
    modelCell.style.flexDirection = "column";
    modelCell.style.gap = "3px";
    const modelLabel = document.createElement("label");
    modelLabel.textContent = "Model";
    const modelSelect = document.createElement("select");
    // Visible warning row that appears above the dropdown when the saved
    // value isn't in the loaded LoRA list. Without this, the user only
    // notices the (missing) marker when they click the dropdown — a row
    // wired to a renamed/moved file looks fine at a glance and silently
    // contributes nothing at run time.
    const missingWarn = document.createElement("div");
    missingWarn.className = "pl-lora-missing-warn";
    missingWarn.style.display = "none";
    const refreshOptions = (names) => {
      const current = modelSelect.value || initial?.name || "";
      modelSelect.replaceChildren();
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = names.length ? "(pick a LoRA)" : "(no LoRAs found)";
      modelSelect.appendChild(placeholder);
      for (const n of names) {
        const opt = document.createElement("option");
        opt.value = n;
        opt.textContent = n;
        modelSelect.appendChild(opt);
      }
      // Preserve a value that's no longer in the list (LoRA removed from
      // disk) so the user can see + delete it rather than silently losing
      // it. Surface a visible warning above the dropdown too, so the row
      // doesn't quietly contribute nothing at run time.
      const isMissing = current && !names.includes(current);
      if (isMissing) {
        const opt = document.createElement("option");
        opt.value = current;
        opt.textContent = `${current}  (missing)`;
        modelSelect.appendChild(opt);
        missingWarn.textContent = `⚠ '${current}' not found in models/loras/. ` +
          "Re-pick a LoRA above, or delete this row.";
        missingWarn.style.display = "";
      } else {
        missingWarn.style.display = "none";
        missingWarn.textContent = "";
      }
      modelSelect.value = current;
    };
    refreshOptions(loraNames);
    modelCell.append(modelLabel, modelSelect, missingWarn);

    const strengthCell = document.createElement("div");
    strengthCell.className = "pl-lora-strength";
    const strengthLabel = document.createElement("label");
    strengthLabel.textContent = "Strength (M / C)";
    strengthLabel.title = "Top slider = MODEL strength, bottom = CLIP strength. "
      + "Click the 🔗 / 🔓 button to link/unlink them — when linked, dragging "
      + "either slider drags the other to match. The Style node passes each "
      + "independently to comfy.sd.load_lora_for_models.";

    // Round to 2dp on the way in too — float math would otherwise show e.g.
    // 0.8500000000000001 in the number input and look broken.
    const roundStrength = (v) => Math.round(v * 100) / 100;

    // Build one slider+number pair for either model or clip strength. Same
    // shape repeated twice so model and clip get independent controls; the
    // backend has always tracked them as separate fields, the UI was just
    // mirroring them silently.
    const buildStrengthPair = (initialValue, prefix) => {
      const bar = document.createElement("div");
      bar.className = "pl-lora-strength-bar";
      const tag = document.createElement("span");
      tag.className = "pl-lora-strength-tag";
      tag.textContent = prefix;
      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = "-2";
      slider.max = "2";
      slider.step = "0.05";
      slider.value = String(roundStrength(initialValue ?? 1.0));
      const num = document.createElement("input");
      num.type = "number";
      num.min = "-2";
      num.max = "2";
      num.step = "0.05";
      num.value = slider.value;
      slider.addEventListener("input", () => {
        num.value = String(roundStrength(Number(slider.value)));
      });
      num.addEventListener("input", () => {
        const v = Math.max(-2, Math.min(2, Number(num.value) || 0));
        slider.value = String(v);
      });
      bar.append(tag, slider, num);
      const setValue = (v) => {
        const clamped = Math.max(-2, Math.min(2, roundStrength(Number(v) || 0)));
        slider.value = String(clamped);
        num.value = slider.value;
      };
      return { bar, slider, num, setValue,
        read: () => Math.max(-2, Math.min(2, Number(num.value) || 0)) };
    };

    const modelStrength = buildStrengthPair(
      initial?.strength_model ?? 1.0, "M");
    // strength_clip falls back to strength_model when an entry was saved
    // before the dual-slider UI existed (everything pre-v0.40.3 mirrored
    // the two values).
    const clipStrength = buildStrengthPair(
      initial?.strength_clip ?? initial?.strength_model ?? 1.0, "C");

    // Linked-by-default toggle. Initial state: linked when the entry was
    // saved with model == clip (or pre-v0.41 mirroring), unlinked when
    // they differ. Clicking the button flips state; entering linked mode
    // immediately syncs CLIP to the current MODEL value.
    const linkBtn = document.createElement("button");
    linkBtn.type = "button";
    linkBtn.className = "pl-lora-strength-link";
    let strengthLinked = (initial?.strength_clip === undefined)
      || (initial?.strength_model === initial?.strength_clip);
    const renderLink = () => {
      linkBtn.textContent = strengthLinked ? "🔗" : "🔓";
      linkBtn.title = strengthLinked
        ? "Linked — CLIP follows MODEL. Click to unlink."
        : "Unlinked — set CLIP independently. Click to relink (CLIP snaps "
          + "back to MODEL value).";
      linkBtn.classList.toggle("linked", strengthLinked);
    };
    renderLink();
    linkBtn.onclick = () => {
      strengthLinked = !strengthLinked;
      if (strengthLinked) clipStrength.setValue(modelStrength.read());
      renderLink();
    };

    // While linked, dragging either slider updates the other so the user
    // sees the "they're moving together" affordance. The mutex flag
    // breaks the otherwise-infinite recursion (slider.input → setValue →
    // implicit input event).
    let _strengthMutex = false;
    const linkSync = (src, dst) => () => {
      if (!strengthLinked || _strengthMutex) return;
      _strengthMutex = true;
      try { dst.setValue(src.read()); } finally { _strengthMutex = false; }
    };
    modelStrength.slider.addEventListener("input", linkSync(modelStrength, clipStrength));
    modelStrength.num.addEventListener("input", linkSync(modelStrength, clipStrength));
    clipStrength.slider.addEventListener("input", linkSync(clipStrength, modelStrength));
    clipStrength.num.addEventListener("input", linkSync(clipStrength, modelStrength));

    const strengthHeader = document.createElement("div");
    strengthHeader.className = "pl-lora-strength-header";
    strengthHeader.append(strengthLabel, linkBtn);
    strengthCell.append(strengthHeader, modelStrength.bar, clipStrength.bar);

    const triggersCell = document.createElement("div");
    triggersCell.style.display = "flex";
    triggersCell.style.flexDirection = "column";
    triggersCell.style.gap = "3px";
    const triggersLabel = document.createElement("label");
    triggersLabel.textContent = "Trigger words";
    const triggersInput = document.createElement("input");
    triggersInput.type = "text";
    triggersInput.placeholder = "optional";
    triggersInput.value = initial?.triggers || "";
    triggersCell.append(triggersLabel, triggersInput);

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "pl-lora-delete";
    deleteBtn.textContent = "Delete";
    deleteBtn.title = "Mark this LoRA as removed. Click Restore to put it back, "
      + "or Save the modal to commit the deletion.";
    // Soft-delete pattern: the row stays visible (greyed out + struck through)
    // until Save commits, and the delete button toggles to Restore. A misclick
    // is reversible without abandoning all the other edits in the modal —
    // matches the soft-delete UX in the gallery's bulk actions.
    let pendingDelete = false;
    const setPendingDelete = (flag) => {
      pendingDelete = flag;
      row.classList.toggle("pl-lora-row-deleted", flag);
      deleteBtn.textContent = flag ? "Restore" : "Delete";
      deleteBtn.title = flag
        ? "Restore this LoRA — undo the pending deletion."
        : "Mark this LoRA as removed. Click Restore to put it back, "
          + "or Save the modal to commit the deletion.";
    };
    deleteBtn.onclick = () => setPendingDelete(!pendingDelete);

    row.append(numCell, modelCell, strengthCell, triggersCell, deleteBtn);

    const rowApi = {
      el: row,
      setIndex(n) { numCell.textContent = `LoRA ${n}`; },
      refreshOptions,
      getValue() {
        if (pendingDelete) return null;
        const name = (modelSelect.value || "").trim();
        if (!name) return null;
        return {
          name,
          strength_model: modelStrength.read(),
          strength_clip: clipStrength.read(),
          triggers: triggersInput.value.trim(),
          enabled: true,
        };
      },
    };
    return rowApi;
  };

  addBtn.onclick = () => {
    if (rows.length >= LORAS_PER_ENTRY_CAP) return;
    const row = buildRow(null);
    rows.push(row);
    list.appendChild(row.el);
    renumber();
  };

  // Preload existing LoRA rows for entries that already have a stack saved.
  for (const l of (initialLoras || []).slice(0, LORAS_PER_ENTRY_CAP)) {
    const row = buildRow(l);
    rows.push(row);
    list.appendChild(row.el);
  }

  wrap.append(heading, headingHint, addWrap, list);
  renumber();

  return {
    el: wrap,
    getLoras() {
      return rows.map(r => r.getValue()).filter(v => v !== null);
    },
    waitForOptions: () => loraNamesPromise,
  };
}

function openPromptModal({ existing, onSave, onDelete, nameExists }) {
  try {
    return _openPromptModalInner({ existing, onSave, onDelete, nameExists });
  } catch (e) {
    // The modal builder constructs hundreds of DOM nodes synchronously and
    // attaches a half-dozen event listeners. A throw mid-construction would
    // leave a half-built DOM in document.body and the gallery in a
    // half-disabled state (modal listeners on document still firing). Catch,
    // surface as a toast, and let the caller continue.
    console.error("[PromptLibrary] openPromptModal failed:", e);
    document.querySelectorAll(".pl-modal").forEach(el => el.remove());
    toast(`Could not open Edit Prompt modal: ${e?.message || e}`, "error", 6000);
    return null;
  }
}

function _openPromptModalInner({ existing, onSave, onDelete, nameExists }) {
  const modal = document.createElement("div");
  modal.className = "pl-modal";

  const header = document.createElement("div");
  header.className = "pl-modal-header";
  const title = document.createElement("h3");
  title.textContent = existing ? "Edit prompt" : "Add prompt";
  const closeX = document.createElement("button");
  closeX.className = "pl-modal-close";
  closeX.type = "button";
  closeX.textContent = "✕";
  closeX.title = "Close";
  header.append(title, closeX);

  const nameLabel = document.createElement("label");
  nameLabel.textContent = "Name";
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.value = existing?.name || "";
  nameLabel.appendChild(nameInput);

  const tagsLabel = document.createElement("label");
  tagsLabel.textContent = "Tags (comma-separated; supports category:value)";
  const tagsWrap = document.createElement("div");
  tagsWrap.style.position = "relative";
  const tagsInput = document.createElement("input");
  tagsInput.type = "text";
  tagsInput.value = (existing?.tags || []).join(", ");
  tagsInput.placeholder = "character, style:cyberpunk, model:anima";
  tagsInput.autocomplete = "off";
  tagsInput.style.width = "100%";
  tagsWrap.appendChild(tagsInput);
  // Custom per-token autocomplete — datalist replaces the whole input
  // value when you click a suggestion, which breaks comma-separated lists.
  // This drop-down completes only the trailing token (text after the last
  // comma), preserving the preceding tags.
  const tagsDropdown = document.createElement("div");
  tagsDropdown.style.cssText = "position:absolute;left:0;right:0;top:100%;"
      + "background:#2a2a2a;border:1px solid #444;border-radius:3px;"
      + "max-height:200px;overflow-y:auto;z-index:1000;display:none;"
      + "box-shadow:0 4px 12px rgba(0,0,0,0.5);font-size:12px;";
  tagsWrap.appendChild(tagsDropdown);
  tagsLabel.appendChild(tagsWrap);
  let _allTags = [];
  api.fetchApi("/prompt_library/tags").then(r => r.json()).then(d => {
    _allTags = d.tags || [];
  }).catch(() => {});

  const _activeToken = () => {
    // The trailing comma-separated token (substring after last comma).
    // What the user is currently typing.
    const v = tagsInput.value;
    const cursor = tagsInput.selectionStart ?? v.length;
    const before = v.slice(0, cursor);
    const lastComma = before.lastIndexOf(",");
    return {
      token: before.slice(lastComma + 1).trimStart().toLowerCase(),
      tokenStart: lastComma + 1,
      cursor,
    };
  };
  let _tagsHighlight = -1;
  const _renderTagSuggestions = () => {
    const { token } = _activeToken();
    tagsDropdown.replaceChildren();
    if (!token) {
      tagsDropdown.style.display = "none";
      return;
    }
    const existing = new Set(
      tagsInput.value.split(",").map(s => s.trim().toLowerCase()).filter(Boolean)
    );
    const matches = _allTags
      .filter(t => t.toLowerCase().includes(token) && !existing.has(t.toLowerCase()))
      .slice(0, 8);
    if (matches.length === 0) {
      tagsDropdown.style.display = "none";
      _tagsHighlight = -1;
      return;
    }
    matches.forEach((t, i) => {
      const item = document.createElement("div");
      item.textContent = t;
      item.style.cssText = "padding:5px 8px;cursor:pointer;color:#ddd;"
          + (i === _tagsHighlight ? "background:#3a3a3a;color:#fff;" : "");
      item.onmousedown = (e) => {
        e.preventDefault();  // Prevent input-blur before we read selection
        _insertTagSuggestion(t);
      };
      item.onmouseenter = () => {
        _tagsHighlight = i;
        for (const sib of tagsDropdown.children) sib.style.background = "";
        item.style.background = "#3a3a3a";
        item.style.color = "#fff";
      };
      tagsDropdown.appendChild(item);
    });
    tagsDropdown.style.display = "block";
  };
  const _insertTagSuggestion = (suggestion) => {
    const { tokenStart, cursor } = _activeToken();
    const before = tagsInput.value.slice(0, tokenStart);
    const after = tagsInput.value.slice(cursor);
    // Add a leading space if the prior char isn't whitespace, and a trailing
    // ", " so the user can keep typing the next tag without manual punctuation.
    const sep = (before && !before.endsWith(", ") && !before.endsWith(" "))
        ? (before.endsWith(",") ? " " : ", ") : "";
    tagsInput.value = before + sep + suggestion + ", " + after.replace(/^[\s,]+/, "");
    tagsDropdown.style.display = "none";
    _tagsHighlight = -1;
    tagsInput.focus();
    const newCursor = (before + sep + suggestion + ", ").length;
    tagsInput.setSelectionRange(newCursor, newCursor);
  };
  tagsInput.addEventListener("input", _renderTagSuggestions);
  tagsInput.addEventListener("focus", _renderTagSuggestions);
  tagsInput.addEventListener("blur", () => {
    // Delay so an mousedown on a suggestion can fire first.
    setTimeout(() => { tagsDropdown.style.display = "none"; }, 100);
  });
  tagsInput.addEventListener("keydown", (e) => {
    if (tagsDropdown.style.display === "none") return;
    const items = [...tagsDropdown.children];
    if (!items.length) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      _tagsHighlight = (_tagsHighlight + 1) % items.length;
      _renderTagSuggestions();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      _tagsHighlight = (_tagsHighlight - 1 + items.length) % items.length;
      _renderTagSuggestions();
    } else if (e.key === "Enter" || e.key === "Tab") {
      if (_tagsHighlight >= 0 && _tagsHighlight < items.length) {
        e.preventDefault();
        _insertTagSuggestion(items[_tagsHighlight].textContent);
      }
    } else if (e.key === "Escape") {
      tagsDropdown.style.display = "none";
      _tagsHighlight = -1;
    }
  });

  const idLabel = document.createElement("label");
  idLabel.textContent = existing ? "ID (read-only)" : "ID (optional — auto from name)";
  const idInput = document.createElement("input");
  idInput.type = "text";
  idInput.value = existing?.id || "";
  if (existing) idInput.readOnly = true;
  else nameInput.addEventListener("input", () => {
    if (!idInput.dataset.userEdited) idInput.placeholder = slugify(nameInput.value);
  });
  idInput.addEventListener("input", () => { idInput.dataset.userEdited = "1"; });
  idLabel.appendChild(idInput);

  const textLabel = document.createElement("label");
  textLabel.textContent = "Prompt text";
  const textArea = document.createElement("textarea");
  textArea.value = existing?.text || "";
  textLabel.appendChild(textArea);

  // Optional negative prompt — paired with the positive on the Library/Save/
  // Random nodes' second STRING output. Empty stays empty (no field written).
  const negLabel = document.createElement("label");
  negLabel.textContent = "Negative prompt (optional)";
  const negArea = document.createElement("textarea");
  negArea.value = existing?.negative || "";
  negArea.placeholder = "lowres, bad_anatomy, watermark…  (leave blank to omit)";
  negLabel.appendChild(negArea);

  // Rating: 5 toggle stars. Clicking the active rating clears it (rating = 0).
  const ratingLabel = document.createElement("label");
  ratingLabel.textContent = "Rating";
  const ratingWrap = document.createElement("div");
  ratingWrap.className = "pl-rating-input";
  ratingWrap.setAttribute("role", "radiogroup");
  ratingWrap.setAttribute("aria-label", "Rating, 0 to 5 stars");
  let ratingValue = Math.max(0, Math.min(5, Number(existing?.rating || 0)));
  const stars = [];
  const paintStars = () => {
    for (let i = 1; i <= 5; i++) {
      stars[i - 1].textContent = i <= ratingValue ? "★" : "☆";
      stars[i - 1].classList.toggle("on", i <= ratingValue);
      stars[i - 1].setAttribute("aria-checked", i === ratingValue ? "true" : "false");
    }
  };
  for (let i = 1; i <= 5; i++) {
    const star = document.createElement("button");
    star.type = "button";
    star.className = "pl-star";
    star.setAttribute("role", "radio");
    star.setAttribute("aria-label", `${i} star${i === 1 ? "" : "s"}`);
    star.title = `${i} star${i === 1 ? "" : "s"}`;
    star.onclick = () => { ratingValue = ratingValue === i ? 0 : i; paintStars(); };
    stars.push(star);
    ratingWrap.appendChild(star);
  }
  paintStars();
  ratingLabel.appendChild(ratingWrap);

  const notesLabel = document.createElement("label");
  notesLabel.textContent = "Notes (private — not used in generation)";
  const notesArea = document.createElement("textarea");
  notesArea.className = "pl-notes";
  notesArea.value = existing?.notes || "";
  notesArea.placeholder = "context, intended use, what works well...";
  notesLabel.appendChild(notesArea);

  // LoRA stack — only consumed by the Style node, but the section is shown
  // for every entry so the same library can serve both the STRING and the
  // Style nodes without separate edit flows.
  const loraSection = buildLoraSection(existing?.loras || []);

  const imgLabel = document.createElement("label");
  imgLabel.textContent = "Reference image (optional)";
  const imgInput = document.createElement("input");
  imgInput.type = "file";
  imgInput.accept = "image/png,image/jpeg,image/webp,image/gif,image/bmp";
  imgLabel.appendChild(imgInput);

  let clearImage = false;
  const preview = document.createElement("img");
  preview.className = "pl-thumb-preview";
  preview.style.display = "none";
  if (existing?.has_image) {
    preview.src = imageUrl(existing.id);
    preview.style.display = "block";
  }
  imgLabel.appendChild(preview);

  if (existing?.has_image) {
    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "pl-btn";
    clearBtn.textContent = "Remove image";
    clearBtn.onclick = () => {
      clearImage = true;
      preview.style.display = "none";
      imgInput.value = "";
      clearBtn.disabled = true;
    };
    imgLabel.appendChild(clearBtn);
  }

  // History disclosure (only for existing entries).
  let historyDetails = null;
  if (existing) {
    historyDetails = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "History";
    summary.style.cursor = "pointer";
    summary.style.fontSize = "12px";
    historyDetails.appendChild(summary);

    const list = document.createElement("div");
    list.className = "pl-history";
    list.replaceChildren(Object.assign(document.createElement("div"),
      { className: "pl-history-empty", textContent: "loading..." }));
    historyDetails.appendChild(list);

    const renderHistory = (snapshots) => {
      list.replaceChildren();
      if (!snapshots.length) {
        list.appendChild(Object.assign(document.createElement("div"),
          { className: "pl-history-empty", textContent: "No prior versions yet." }));
        return;
      }
      for (const snap of snapshots) {
        const row = document.createElement("div");
        row.className = "pl-history-row";

        const ts = document.createElement("span");
        ts.className = "pl-history-ts";
        ts.textContent = relativeTime(snap.ts);
        ts.title = new Date(snap.ts * 1000).toLocaleString();

        const body = document.createElement("div");
        body.className = "pl-history-body";
        const nameEl = document.createElement("strong");
        nameEl.textContent = snap.name || "(no name)";
        body.appendChild(nameEl);
        const textEl = document.createElement("div");
        textEl.textContent = (snap.text || "").slice(0, 140) + ((snap.text || "").length > 140 ? "..." : "");
        body.appendChild(textEl);
        if (snap.tags?.length) {
          const tagsEl = document.createElement("div");
          tagsEl.className = "pl-history-tags";
          tagsEl.textContent = snap.tags.join(", ");
          body.appendChild(tagsEl);
        }

        const revertBtn = document.createElement("button");
        revertBtn.type = "button";
        revertBtn.className = "pl-btn";
        revertBtn.textContent = "Revert";
        revertBtn.onclick = async () => {
          if (!await confirmDestructive(
            `Revert "${existing.name}" to this version? Current values will be saved to history first.`,
            { confirmLabel: "Revert" }
          )) return;
          revertBtn.disabled = true;
          try {
            await revertPrompt(existing.id, snap.ts);
            close();  // refresh-via-websocket will update the gallery
          } catch (e) {
            revertBtn.disabled = false;
            status.classList.add("error");
            status.textContent = e.message;
          }
        };

        row.append(ts, body, revertBtn);
        list.appendChild(row);
      }
    };

    historyDetails.addEventListener("toggle", async () => {
      if (!historyDetails.open) return;
      try {
        const data = await fetchHistory(existing.id);
        renderHistory(data.history || []);
      } catch (e) {
        list.replaceChildren(Object.assign(document.createElement("div"),
          { className: "pl-history-empty", textContent: `Failed: ${e.message}` }));
      }
    });
  }

  imgInput.onchange = () => {
    const f = imgInput.files[0];
    // Reject non-image MIME types so we don't hand a blob URL of arbitrary
    // user-supplied data to <img>. Belt-and-suspenders for CodeQL's
    // js/xss-through-dom alert — img elements can't execute scripts from a
    // blob URL anyway, but the type check is good defensive practice.
    if (f && typeof f.type === "string" && f.type.startsWith("image/")) {
      preview.src = URL.createObjectURL(f);
      preview.style.display = "block";
      clearImage = false;
    }
  };

  // Paste-from-clipboard: when the modal is focused and the user pastes an
  // image (Ctrl+V from a screenshot, browser, etc.), drop it into the file
  // input via DataTransfer so the existing change handler picks it up.
  const onPaste = (e) => {
    if (!modal.contains(document.activeElement) && !modal.contains(e.target)) return;
    const items = e.clipboardData?.items || [];
    for (const item of items) {
      if (item.kind === "file" && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (!file) continue;
        const ext = (file.type.split("/")[1] || "png").replace("jpeg", "jpg");
        const named = new File([file], `pasted-${Date.now()}.${ext}`, { type: file.type });
        const dt = new DataTransfer();
        dt.items.add(named);
        imgInput.files = dt.files;
        imgInput.dispatchEvent(new Event("change"));
        e.preventDefault();
        toast("Pasted image from clipboard.", "success", 2000);
        return;
      }
    }
  };
  document.addEventListener("paste", onPaste);

  const status = document.createElement("div");
  status.className = "pl-status";

  const actions = document.createElement("div");
  actions.className = "pl-modal-actions";

  const saveBtn = document.createElement("button");
  saveBtn.className = "pl-btn";
  saveBtn.textContent = "Save";
  const cancelBtn = document.createElement("button");
  cancelBtn.className = "pl-btn";
  cancelBtn.textContent = "Cancel";

  // Stop key events from bubbling to LiteGraph (typing in inputs would otherwise
  // trigger canvas shortcuts like delete-node).
  for (const ev of ["keydown", "keyup", "keypress"]) {
    modal.addEventListener(ev, (e) => e.stopPropagation());
  }

  let dragCleanup = null;
  const onKey = (e) => {
    if (e.key === "Escape" && modal.contains(document.activeElement)) {
      e.stopPropagation();
      close();
    }
  };
  const close = () => {
    document.removeEventListener("keydown", onKey, true);
    document.removeEventListener("paste", onPaste);
    dragCleanup?.();
    modal.remove();
    _modalStack = Math.max(0, _modalStack - 1);
  };
  document.addEventListener("keydown", onKey, true);
  closeX.onclick = close;
  cancelBtn.onclick = close;

  saveBtn.onclick = async () => {
    const name = nameInput.value.trim();
    if (!name) { status.textContent = "name required"; status.classList.add("error"); return; }
    const customId = idInput.value.trim();
    if (customId && !/^[A-Za-z0-9_-]{1,64}$/.test(customId)) {
      status.textContent = "id must be A-Z, 0-9, _ or - (max 64)";
      status.classList.add("error");
      return;
    }
    // Duplicate-name guard for the +Add path. The backend's _unique_id
    // silently appends '_2', '_3', ... when the slug collides — that's
    // how users end up with multiple entries sharing a name without
    // realising it. When opening this modal in Add-new mode (no
    // existing entry passed in), check the gallery for a name match
    // first and ask before creating the duplicate. Edit mode (existing
    // ≠ null) bypasses the check — those saves target a known id.
    if (!existing && typeof nameExists === "function" && nameExists(name)) {
      const ok = await confirmDestructive(
        `An entry named "${name}" already exists. Save anyway as a new ` +
        `(auto-numbered) entry, or cancel and edit the existing one?`,
        { confirmLabel: "Save as new" });
      if (!ok) return;
    }
    saveBtn.disabled = true;
    status.classList.remove("error");
    status.textContent = "saving...";
    try {
      await onSave({
        id: existing?.id || customId,
        name,
        text: textArea.value,
        negative: negArea.value,
        tags: tagsInput.value,
        rating: ratingValue,
        notes: notesArea.value,
        loras: loraSection.getLoras(),
        imageFile: imgInput.files[0] || null,
        clearImage,
      });
      close();
    } catch (e) {
      saveBtn.disabled = false;
      status.classList.add("error");
      status.textContent = e.message;
    }
  };

  if (existing && onDelete) {
    const delBtn = document.createElement("button");
    delBtn.className = "pl-btn danger";
    delBtn.textContent = "Delete";
    delBtn.onclick = async () => {
      if (!await confirmDestructive(`Delete "${existing.name}"?`)) return;
      delBtn.disabled = true;
      try { await onDelete(existing.id); close(); }
      catch (e) { delBtn.disabled = false; status.classList.add("error"); status.textContent = e.message; }
    };
    actions.appendChild(delBtn);
    const spacer = document.createElement("div");
    spacer.style.flex = "1";
    actions.appendChild(spacer);
  }
  actions.appendChild(cancelBtn);
  actions.appendChild(saveBtn);

  const children = [header, nameLabel, idLabel, tagsLabel, textLabel, negLabel,
    ratingLabel, notesLabel, loraSection.el, imgLabel];
  if (historyDetails) children.push(historyDetails);
  children.push(status, actions);
  modal.append(...children);

  // Initial position: cascade modals so stacked windows don't overlap exactly.
  const offset = (_modalStack++ % 6) * 24;
  modal.style.left = `calc(50% - 270px + ${offset}px)`;
  modal.style.top = `calc(15% + ${offset}px)`;
  document.body.appendChild(modal);

  dragCleanup = makeDraggable(modal, header);
  nameInput.focus();
}

function makeDraggable(panel, handle) {
  let dragging = false;
  let startX = 0, startY = 0, panelX = 0, panelY = 0;

  const onDown = (e) => {
    if (e.button !== 0 || e.target.closest("button, input, textarea, select")) return;
    const rect = panel.getBoundingClientRect();
    panel.style.left = rect.left + "px";
    panel.style.top = rect.top + "px";
    panelX = rect.left;
    panelY = rect.top;
    startX = e.clientX;
    startY = e.clientY;
    dragging = true;
    e.preventDefault();
  };
  const onMove = (e) => {
    if (!dragging) return;
    const x = panelX + (e.clientX - startX);
    const y = panelY + (e.clientY - startY);
    const maxX = window.innerWidth - panel.offsetWidth;
    const maxY = window.innerHeight - 40;
    panel.style.left = Math.max(0, Math.min(maxX, x)) + "px";
    panel.style.top = Math.max(0, Math.min(maxY, y)) + "px";
  };
  const onUp = () => { dragging = false; };

  handle.addEventListener("mousedown", onDown);
  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);

  return () => {
    handle.removeEventListener("mousedown", onDown);
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
  };
}

function buildGallery(node, idWidget, propsKey = "pl_state") {
  const container = document.createElement("div");
  container.className = "pl-gallery";
  container.setAttribute("role", "region");
  container.setAttribute("aria-label", "Prompt library gallery");
  container.setAttribute("tabindex", "0");

  // Per-node UI state (filter / tags / sort / tile size / view) persists via
  // node.properties so it round-trips with the workflow JSON. localStorage
  // still seeds defaults for fresh nodes that have nothing saved yet.
  const readState = () => {
    node.properties = node.properties || {};
    return node.properties[propsKey] || {};
  };
  // localStorage key — survives across browser tab close/reopen, ComfyUI
  // restart, and different workflow tabs in the same browser. Per-node-id
  // AND per-propsKey so Multi-Library's three panels don't stomp each
  // other. Helpers extracted to lsSelection.js for unit testability.
  const _lsKey = () => lsSelKey(node?.id, propsKey);
  const readLocalSelection = () => lsReadSel(localStorage, node?.id, propsKey);
  const writeLocalSelection = () => lsWriteSel(localStorage, node?.id, propsKey, [...checkedIds]);
  const writeState = () => {
    node.properties = node.properties || {};
    node.properties[propsKey] = {
      filter: filter.value || "",
      activeTags: [...activeTags],
      tagFilterMode,
      favoritesOnly,
      sort: sortSelect.value,
      tileSize: sizeInput.value,
      view: viewMode,
      // Three-tier selection persistence:
      //   1. idWidget.value — comma-joined ids, ComfyUI auto-serialises
      //      this into widgets_values for the workflow JSON
      //   2. node.properties[propsKey].selectedIds — same payload, but in
      //      properties (also workflow-JSON-serialised, but tolerates
      //      future widget-value handling changes)
      //   3. localStorage[`pl_sel_<node.id>`] — survives browser close,
      //      ComfyUI restart, different workflow tabs
      // syncFromWidget falls back through the chain if earlier tiers are
      // empty, so selection sticks until the user explicitly clears it.
      selectedIds: [...checkedIds],
    };
    writeLocalSelection();
  };
  const initialState = readState();

  const toolbar = document.createElement("div");
  toolbar.className = "pl-toolbar";

  const searchWrap = document.createElement("div");
  searchWrap.className = "pl-search-wrap";
  const filter = document.createElement("input");
  filter.type = "text";
  filter.placeholder = "search name, text, tags...";
  for (const ev of ["keydown", "keyup", "keypress"]) {
    filter.addEventListener(ev, (e) => e.stopPropagation());
  }
  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.className = "pl-search-clear";
  clearBtn.textContent = "×";
  clearBtn.title = "Clear search";
  clearBtn.style.display = "none";
  clearBtn.onclick = () => { filter.value = ""; clearBtn.style.display = "none"; render(); };
  filter.addEventListener("input", () => {
    clearBtn.style.display = filter.value ? "block" : "none";
  });
  searchWrap.append(filter, clearBtn);
  const modelSelect = document.createElement("select");
  modelSelect.title = "Filter by model (model:* tags)";
  // populated in render() once we know the data

  const sortSelect = document.createElement("select");
  for (const [key, mode] of Object.entries(SORT_MODES)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = mode.label;
    sortSelect.appendChild(opt);
  }
  sortSelect.value = initialState.sort || localStorage.getItem(SORT_KEY) || "name_asc";
  sortSelect.title = "Sort";

  const importBtn = document.createElement("button");
  importBtn.className = "pl-btn";
  importBtn.textContent = "Import";
  importBtn.title = "Import a CSV (name,text,tags,id) or a GrimmRibbity .zip. Default skips "
    + "entries whose id already exists (preserves your local edits and thumbnails). "
    + "Shift-click to overwrite existing entries with the file's values.";
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".csv,text/csv,.zip,application/zip";
  fileInput.style.display = "none";
  // Stash the mode on the button so the change handler reads the right value.
  importBtn.dataset.importMode = "add_only";
  importBtn.onclick = (e) => {
    importBtn.dataset.importMode = e.shiftKey ? "update" : "add_only";
    fileInput.click();
  };

  const undoBtn = document.createElement("button");
  undoBtn.className = "pl-btn";
  undoBtn.textContent = "Undo Import";
  undoBtn.title = "Restore the library from the most recent pre-import snapshot. "
    + "Useful if a CSV / ZIP / Scan LoRAs / Import BG run clobbered something. "
    + "The undo itself is also snapshotted, so you can redo by undoing again.";

  const exportBtn = document.createElement("button");
  exportBtn.className = "pl-btn";
  exportBtn.textContent = "Export";
  exportBtn.title = "Export the currently visible prompts (with thumbnails) as a .zip";

  const scanLorasBtn = document.createElement("button");
  scanLorasBtn.className = "pl-btn";
  scanLorasBtn.textContent = "Scan LoRAs";
  scanLorasBtn.title = "Walk models/loras/ and add a library entry per LoRA, with "
    + "auto-detected preview thumbnails and trigger words from the safetensors metadata. "
    + "Existing entries are skipped — re-running won't clobber edits.";

  const importBgBtn = document.createElement("button");
  importBgBtn.className = "pl-btn";
  importBgBtn.textContent = "Import BG";
  importBgBtn.title = "Bulk-import the GrimmRibbity Background node's preset locations as "
    + "library entries (tagged 'location'). Filter by the 'location' chip after import. "
    + "Existing entries are skipped; Shift-click to refresh them.";

  const queueAllBtn = document.createElement("button");
  queueAllBtn.className = "pl-btn";
  queueAllBtn.textContent = "Queue ▶▶";
  queueAllBtn.title = "Queue the current workflow once per selected entry (or per visible "
    + "entry if no selection). Sets THIS gallery's prompt_id to that entry's id before "
    + "each queue and restores it after. Pair with a PromptLibrarySave or ThumbnailSaver "
    + "node wired to your output to auto-fill thumbnails across many entries.";

  const refreshBtn = document.createElement("button");
  refreshBtn.className = "pl-btn";
  refreshBtn.textContent = "Refresh";

  const SIZE_KEY = "comfy.PromptLibrary.tileSize";
  const sizeWrap = document.createElement("div");
  sizeWrap.className = "pl-tile-size";
  const sizeInput = document.createElement("input");
  sizeInput.type = "range";
  sizeInput.min = "60";
  sizeInput.max = "400";
  sizeInput.step = "8";
  sizeInput.value = initialState.tileSize || localStorage.getItem(SIZE_KEY) || "110";
  sizeInput.title = "Tile size";
  const applySize = () => {
    container.style.setProperty("--pl-tile-size", `${sizeInput.value}px`);
  };
  sizeInput.oninput = () => {
    applySize();
    localStorage.setItem(SIZE_KEY, sizeInput.value);
    writeState();
  };
  sizeWrap.appendChild(sizeInput);

  const VIEW_KEY = "comfy.PromptLibrary.view";
  const viewWrap = document.createElement("div");
  viewWrap.className = "pl-view-toggle";
  const gridViewBtn = document.createElement("button");
  gridViewBtn.className = "pl-btn";
  gridViewBtn.textContent = "▦";
  gridViewBtn.title = "Grid view";
  const listViewBtn = document.createElement("button");
  listViewBtn.className = "pl-btn";
  listViewBtn.textContent = "≡";
  listViewBtn.title = "List view";
  viewWrap.append(gridViewBtn, listViewBtn);
  let viewMode = initialState.view || localStorage.getItem(VIEW_KEY) || "grid";
  if (viewMode !== "list" && viewMode !== "grid") viewMode = "grid";
  const applyView = () => {
    grid.classList.toggle("list-view", viewMode === "list");
    gridViewBtn.classList.toggle("active", viewMode === "grid");
    listViewBtn.classList.toggle("active", viewMode === "list");
    sizeWrap.style.display = viewMode === "list" ? "none" : "";
  };
  gridViewBtn.onclick = () => { viewMode = "grid"; localStorage.setItem(VIEW_KEY, viewMode); applyView(); writeState(); };
  listViewBtn.onclick = () => { viewMode = "list"; localStorage.setItem(VIEW_KEY, viewMode); applyView(); writeState(); };

  const favBtn = document.createElement("button");
  favBtn.className = "pl-btn pl-fav-btn";
  favBtn.textContent = "★";
  favBtn.title = "Show only favorites (rating ≥ 4)";
  let favoritesOnly = !!initialState.favoritesOnly;
  const applyFavBtn = () => favBtn.classList.toggle("active", favoritesOnly);
  applyFavBtn();
  favBtn.onclick = () => { favoritesOnly = !favoritesOnly; applyFavBtn(); render(); };

  const countBadge = document.createElement("span");
  countBadge.className = "pl-count-badge";
  countBadge.title = "Visible / total prompts";

  // Library health badge — fetches /prompt_library/validate on demand
  // (or after a refresh) and shows a count of unhealed issues. Clicking
  // opens a modal with the findings + a "Remove orphan thumbnails" action.
  // Hidden when the count is zero so a clean library has no UI clutter.
  const healthBtn = document.createElement("button");
  healthBtn.className = "pl-btn pl-health-btn";
  healthBtn.style.display = "none";
  healthBtn.type = "button";
  healthBtn.title = "Library health — click for the findings list.";
  healthBtn.onclick = () => openValidatorModal();

  toolbar.append(searchWrap, modelSelect, sortSelect, sizeWrap, viewWrap, favBtn, countBadge, healthBtn, importBtn, undoBtn, exportBtn, scanLorasBtn, importBgBtn, queueAllBtn, refreshBtn, fileInput);

  const tagsRow = document.createElement("div");
  tagsRow.className = "pl-tags-row";

  const grid = document.createElement("div");
  grid.className = "pl-grid";
  grid.setAttribute("role", "listbox");
  grid.setAttribute("aria-multiselectable", "true");
  grid.setAttribute("aria-label", "Prompts");

  let prompts = [];
  let lastVisible = [];
  let focusedIndex = -1;        // for keyboard nav
  // Bumped on every render() call. The chunked-render path captures the
  // current value and bails out before each chunk if it sees a newer
  // token — prevents a fresh filter-change from interleaving with stale
  // tiles from the previous render's still-pending chunks.
  let _renderToken = 0;
  const activeTags = new Set(Array.isArray(initialState.activeTags) ? initialState.activeTags : []);
  let tagFilterMode = initialState.tagFilterMode === "all" ? "all" : "any";
  if (initialState.filter) filter.value = initialState.filter;
  // Unified selection: drives both the prompt output (joined into idWidget.value)
  // and bulk actions (Tag/Export/Delete bar).
  const checkedIds = new Set();
  // Diagnostic: ghost-select bug investigation. Every selection mutation logs
  // origin so a multi-Library workflow shows which node fired. Cheap when the
  // user isn't watching the console; turn off via `window._plDebugSelect=false`.
  if (typeof window._plDebugSelect === "undefined") window._plDebugSelect = true;
  const _logSel = (origin, before, after) => {
    if (!window._plDebugSelect) return;
    const a = [...before].sort().join(",");
    const b = [...after].sort().join(",");
    if (a === b) return;
    const nodeId = (node && (node.id ?? "?"));
    const nodeType = (node && (node.type || node.comfyClass || "Lib")) || "Lib";
    console.log(`[PL select] node#${nodeId} (${nodeType}) ${origin}: [${a}] -> [${b}]`);
  };
  const syncWidget = () => {
    idWidget.value = [...checkedIds].join(",");
    node.setDirtyCanvas(true, true);
  };
  // Repopulate checkedIds from the (comma-separated) widget value. Used on
  // workflow load and on Nodes 2.0 setValue, so the gallery highlights match
  // whatever was saved. Tolerates legacy single-id values.
  //
  // If idWidget.value is empty (e.g. browser tab reopened, ComfyUI restored
  // the workflow but the widget value didn't round-trip), fall back to the
  // selectedIds mirror saved in node.properties — that's the belt-and-
  // suspenders backup for selection persistence across page refreshes.
  const syncFromWidget = (origin = "syncFromWidget") => {
    const before = new Set(checkedIds);
    checkedIds.clear();
    const widgetVal = (idWidget.value || "").trim();
    let source = "widget";
    if (widgetVal) {
      for (const raw of widgetVal.split(",")) {
        const id = raw.trim();
        if (id) checkedIds.add(id);
      }
    } else {
      // Widget empty — fall back through properties → localStorage. The
      // first tier with a non-empty selection wins; subsequent tiers stay
      // untouched so we don't clobber the canonical store.
      const stash = (node.properties && node.properties[propsKey]) || {};
      const propIds = Array.isArray(stash.selectedIds) ? stash.selectedIds : [];
      for (const id of propIds) {
        if (id) checkedIds.add(id);
      }
      if (checkedIds.size) {
        source = "properties";
      } else {
        for (const id of readLocalSelection()) {
          if (id) checkedIds.add(id);
        }
        if (checkedIds.size) source = "localStorage";
      }
      // Mirror restored selection back into idWidget so downstream nodes
      // see the canonical comma-joined string on the very next eval.
      if (checkedIds.size) {
        idWidget.value = [...checkedIds].join(",");
      }
    }
    _logSel(`${origin}(${source})`, before, checkedIds);
  };

  const bulkBar = document.createElement("div");
  bulkBar.className = "pl-bulk-bar";
  bulkBar.style.display = "none";
  const bulkCount = document.createElement("span");
  bulkCount.className = "count";
  const bulkClearBtn = document.createElement("button");
  bulkClearBtn.className = "pl-btn";
  bulkClearBtn.textContent = "Clear";
  bulkClearBtn.onclick = () => {
    const wereSelected = [...checkedIds];
    checkedIds.clear();
    applySelectionChange(wereSelected);
  };
  const bulkExportBtn = document.createElement("button");
  bulkExportBtn.className = "pl-btn";
  bulkExportBtn.textContent = "Export";
  const bulkTagBtn = document.createElement("button");
  bulkTagBtn.className = "pl-btn";
  bulkTagBtn.textContent = "Tag";
  const bulkDuplicateBtn = document.createElement("button");
  bulkDuplicateBtn.className = "pl-btn";
  bulkDuplicateBtn.textContent = "Duplicate";
  bulkDuplicateBtn.title = "Duplicate every selected entry (a copy with '-copy' suffix is created for each)";
  const bulkDeleteBtn = document.createElement("button");
  bulkDeleteBtn.className = "pl-btn";
  bulkDeleteBtn.style.color = "#f88";
  bulkDeleteBtn.textContent = "Delete";
  bulkBar.append(bulkCount, bulkClearBtn, bulkTagBtn, bulkDuplicateBtn, bulkExportBtn, bulkDeleteBtn);

  container.append(toolbar, tagsRow, bulkBar, grid);

  const updateBulkBar = () => {
    if (checkedIds.size === 0) {
      bulkBar.style.display = "none";
    } else {
      bulkBar.style.display = "flex";
      bulkCount.textContent = `${checkedIds.size} selected`;
    }
  };

  // In-place selection updates — replace the previous "rebuild every tile on
  // every click" pattern. For a 700-entry library, toggling selection used
  // to rebuild ~21000 DOM nodes per click; now it just flips classes on the
  // affected tile(s).
  //
  // Trade-off: the "selected tiles bubble to the top" behaviour from
  // render()'s sort step doesn't fire on a selection-only change, so a
  // selected tile stays visually in place until the next sort/filter
  // change. Most workflows click tiles they can already see, so this is
  // a net win for huge libraries; if it ever feels wrong we can call
  // render() instead.
  const _refreshTileSelection = (id) => {
    const tile = grid.querySelector(`[data-prompt-id="${id}"]`);
    if (!tile) return;
    const sel = checkedIds.has(id);
    tile.classList.toggle("selected", sel);
    tile.setAttribute("aria-selected", sel ? "true" : "false");
    const checkbox = tile.querySelector(".pl-tile-check");
    if (checkbox) checkbox.textContent = sel ? "✓" : "";
  };

  const _refreshFocusedTile = () => {
    // Move the .focused class from the previous tile (if any) to the one
    // at lastVisible[focusedIndex]. Cheap: at most two DOM mutations.
    for (const t of grid.querySelectorAll(".pl-tile.focused")) {
      t.classList.remove("focused");
    }
    if (focusedIndex >= 0 && focusedIndex < lastVisible.length) {
      const id = lastVisible[focusedIndex].id;
      const tile = grid.querySelector(`[data-prompt-id="${id}"]`);
      tile?.classList.add("focused");
    }
  };

  const applySelectionChange = (idsToRefresh) => {
    syncWidget();
    // Mirror the live selection into node.properties so a page-refresh
    // restore from the workflow JSON has both the widget value and the
    // properties stash to choose from. writeState is cheap (a small dict
    // assign); we already call it on every filter/sort/etc. change.
    writeState();
    if (idsToRefresh && idsToRefresh.length) {
      for (const id of idsToRefresh) _refreshTileSelection(id);
    }
    _refreshFocusedTile();
    updateBulkBar();
  };

  // -------------------------------------------------------------------------
  // Library validator — toolbar badge + modal listing findings.
  // -------------------------------------------------------------------------

  let _lastValidation = null;

  const refreshHealthBadge = async () => {
    try {
      _lastValidation = await fetchValidate();
    } catch (e) {
      // Validator failures shouldn't be loud; the gallery itself already
      // works. Hide the badge and log to console for the curious.
      console.warn("[PromptLibrary] validate failed:", e);
      healthBtn.style.display = "none";
      return;
    }
    const issues =
      (_lastValidation.broken_loras?.length || 0) +
      (_lastValidation.orphan_images?.length || 0) +
      (_lastValidation.invalid_ids?.length || 0) +
      (_lastValidation.empty_texts?.length || 0);
    if (issues === 0) {
      healthBtn.style.display = "none";
      return;
    }
    healthBtn.style.display = "";
    healthBtn.textContent = `⚠ ${issues}`;
    healthBtn.title = `Library has ${issues} issue${issues === 1 ? "" : "s"} — click for details.`;
  };

  const openValidatorModal = async () => {
    // Fetch fresh on open so the user always sees current state, even
    // if entries have changed since the last refreshHealthBadge call.
    let v;
    try {
      v = await fetchValidate();
    } catch (e) {
      toast(`Validator failed: ${e.message}`, "error");
      return;
    }
    _lastValidation = v;

    const modal = document.createElement("div");
    modal.className = "pl-modal";
    modal.style.width = "min(560px, calc(100vw - 24px))";

    const header = document.createElement("div");
    header.className = "pl-modal-header";
    const title = document.createElement("h3");
    title.textContent = "Library health";
    const closeX = document.createElement("button");
    closeX.className = "pl-modal-close";
    closeX.type = "button";
    closeX.textContent = "✕";
    header.append(title, closeX);

    const summary = document.createElement("div");
    summary.style.fontSize = "12px";
    summary.style.color = "var(--pl-fg-muted)";
    summary.textContent = `${v.entries_count} entries · ${v.thumbnails_count} thumbnails`
      + (v.lora_index_available ? "" : " · LoRA index unavailable (running outside Comfy?)");

    const body = document.createElement("div");
    body.style.display = "flex";
    body.style.flexDirection = "column";
    body.style.gap = "12px";
    body.style.fontSize = "12px";

    const buildSection = (heading, items, renderItem, action) => {
      if (!items || items.length === 0) return null;
      const sec = document.createElement("div");
      sec.className = "pl-validator-section";
      const h = document.createElement("div");
      h.className = "pl-validator-section-head";
      h.textContent = `${heading} (${items.length})`;
      sec.appendChild(h);
      const list = document.createElement("div");
      list.className = "pl-validator-list";
      for (const item of items.slice(0, 50)) {
        list.appendChild(renderItem(item));
      }
      if (items.length > 50) {
        const more = document.createElement("div");
        more.className = "pl-validator-more";
        more.textContent = `… ${items.length - 50} more not shown`;
        list.appendChild(more);
      }
      sec.appendChild(list);
      if (action) sec.appendChild(action);
      return sec;
    };

    // Broken LoRAs
    const brokenSec = buildSection("Broken LoRA references",
      v.broken_loras,
      (item) => {
        const row = document.createElement("div");
        row.className = "pl-validator-row";
        const name = document.createElement("span");
        name.textContent = `${item.name || item.id}: ${item.lora}`;
        row.appendChild(name);
        const editBtn = document.createElement("button");
        editBtn.className = "pl-btn";
        editBtn.textContent = "Edit";
        editBtn.onclick = async () => {
          // Re-fetch the entry then open it in the modal so the user
          // can pick a replacement LoRA or remove the row.
          const list = await fetchList();
          const entry = list.find(p => p.id === item.id);
          if (entry) openPromptModal({
            existing: entry,
            onSave: async (payload) => { await upsert(payload); await refresh(); refreshHealthBadge(); },
            onDelete: async (id) => { await deletePrompt(id); await refresh(); refreshHealthBadge(); },
          });
        };
        row.appendChild(editBtn);
        return row;
      });

    // Orphan thumbnails — with a single bulk-cleanup action.
    let orphanCleanupBtn = null;
    if (v.orphan_images && v.orphan_images.length > 0) {
      orphanCleanupBtn = document.createElement("button");
      orphanCleanupBtn.className = "pl-btn";
      orphanCleanupBtn.style.alignSelf = "flex-start";
      orphanCleanupBtn.style.marginTop = "4px";
      orphanCleanupBtn.textContent = "Remove orphan thumbnails";
      orphanCleanupBtn.onclick = async () => {
        if (!await confirmDestructive(
          `Delete ${v.orphan_images.length} orphan thumbnail file(s)?`,
          { confirmLabel: "Remove" })) return;
        try {
          const result = await fixOrphans();
          toast(`Removed ${result.removed} orphan thumbnail(s).`, "success");
          await refreshHealthBadge();
          modal.remove();
          openValidatorModal();  // reopen with fresh data
        } catch (e) { toast(`Cleanup failed: ${e.message}`, "error"); }
      };
    }
    const orphanSec = buildSection("Orphan thumbnails",
      (v.orphan_images || []).map(id => ({ id })),
      (item) => {
        const row = document.createElement("div");
        row.className = "pl-validator-row";
        row.textContent = item.id;
        return row;
      },
      orphanCleanupBtn);

    // Invalid IDs
    const invalidSec = buildSection("Invalid entry IDs",
      v.invalid_ids,
      (id) => {
        const row = document.createElement("div");
        row.className = "pl-validator-row";
        row.textContent = String(id || "(empty)");
        return row;
      });

    // Empty-text entries
    const emptySec = buildSection("Entries with empty prompt text",
      v.empty_texts,
      (item) => {
        const row = document.createElement("div");
        row.className = "pl-validator-row";
        const name = document.createElement("span");
        name.textContent = `${item.name || "(unnamed)"} (${item.id})`;
        row.appendChild(name);
        const editBtn = document.createElement("button");
        editBtn.className = "pl-btn";
        editBtn.textContent = "Edit";
        editBtn.onclick = async () => {
          const list = await fetchList();
          const entry = list.find(p => p.id === item.id);
          if (entry) openPromptModal({
            existing: entry,
            onSave: async (payload) => { await upsert(payload); await refresh(); refreshHealthBadge(); },
            onDelete: async (id) => { await deletePrompt(id); await refresh(); refreshHealthBadge(); },
          });
        };
        row.appendChild(editBtn);
        return row;
      });

    const sections = [brokenSec, orphanSec, invalidSec, emptySec].filter(Boolean);
    if (sections.length === 0) {
      const allClean = document.createElement("div");
      allClean.style.color = "var(--pl-fg-muted)";
      allClean.style.padding = "8px";
      allClean.textContent = "✓ Library is clean — no issues found.";
      body.appendChild(allClean);
    } else {
      for (const s of sections) body.appendChild(s);
    }

    const actions = document.createElement("div");
    actions.className = "pl-modal-actions";
    const closeBtn = document.createElement("button");
    closeBtn.className = "pl-btn";
    closeBtn.textContent = "Close";
    actions.appendChild(closeBtn);

    const close = () => modal.remove();
    closeX.onclick = close;
    closeBtn.onclick = close;

    modal.append(header, summary, body, actions);
    modal.style.left = "calc(50% - 280px)";
    modal.style.top = "12%";
    document.body.appendChild(modal);
    makeDraggable(modal, header);

    for (const ev of ["keydown", "keyup", "keypress"]) {
      modal.addEventListener(ev, (e) => e.stopPropagation());
    }
  };

  // -------------------------------------------------------------------------
  // Event delegation — one set of handlers on the grid instead of per-tile
  // closures. For a 700-tile library in Manual sort, the previous per-tile
  // approach attached ~7 handler closures per tile (~4900 closures total),
  // each capturing checkedIds / syncWidget / render / refresh / etc. via
  // the closure scope. Delegated handlers below resolve the affected tile +
  // entry on demand via target.closest + dataset.promptId, so render only
  // creates the tile DOM (no per-tile closure construction) and selection /
  // context / drag actions stay routed correctly.
  // -------------------------------------------------------------------------

  const _entryFromTile = (tile) =>
    tile ? prompts.find(p => p.id === tile.dataset.promptId) : null;

  const _indexFromTile = (tile) =>
    tile ? lastVisible.findIndex(p => p.id === tile.dataset.promptId) : -1;

  const _toggleSelectionAt = (id, idx, e) => {
    if (e.shiftKey && lastVisible.length) {
      const anchor = focusedIndex >= 0 ? focusedIndex : idx;
      const i0 = Math.min(anchor, idx);
      const i1 = Math.max(anchor, idx);
      const affected = [];
      for (let i = i0; i <= i1; i++) {
        const rid = lastVisible[i].id;
        checkedIds.add(rid);
        affected.push(rid);
      }
      focusedIndex = idx;
      applySelectionChange(affected);
      return;
    }
    if (checkedIds.has(id)) checkedIds.delete(id);
    else checkedIds.add(id);
    focusedIndex = idx;
    applySelectionChange([id]);
  };

  const _openAddPrompt = () => openPromptModal({
    onSave: async (payload) => {
      const created = await upsert(payload);
      checkedIds.add(created.id);
      syncWidget();
      await refresh();
    },
    // Tells the modal to warn before saving a name that collides with
    // an existing entry. Case-insensitive match — 'Foo' and 'foo' get
    // the same auto-id so they collide regardless of case.
    nameExists: (n) => prompts.some(
      p => (p.name || "").toLowerCase() === n.toLowerCase()),
  });

  const _openTileContextMenu = (e, p) => {
    e.preventDefault();
    openContextMenu(e.clientX, e.clientY, [
      { kind: "stars", label: "Rate", current: p.rating || 0,
        action: async (n) => {
          try {
            await upsert({
              id: p.id, name: p.name, text: p.text || "",
              tags: (p.tags || []).join(", "),
              rating: n, notes: p.notes || "",
            });
            await refresh();
          } catch (err) { toast(`Rating failed: ${err.message}`, "error"); }
        } },
      "sep",
      { label: "Edit...", action: () => openPromptModal({
          existing: p,
          onSave: async (payload) => { await upsert(payload); await refresh(); },
          onDelete: async (id) => {
            await deletePrompt(id);
            if (checkedIds.delete(id)) syncWidget();
            await refresh();
          },
        }) },
      { label: "Duplicate", action: async () => {
          try { await duplicatePrompt(p.id); await refresh(); }
          catch (err) { toast(`Duplicate failed: ${err.message}`, "error"); }
        } },
      { label: "Export this", action: async () => {
          try {
            const { blob } = await exportZip([p.id]);
            const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
            downloadBlob(blob, `ribbity-${p.id}-${stamp}.zip`);
          } catch (err) { toast(`Export failed: ${err.message}`, "error"); }
        } },
      "sep",
      { label: "Delete", danger: true, action: async () => {
          if (!await confirmDestructive(`Delete "${p.name}"?`)) return;
          try {
            await deletePrompt(p.id);
            if (checkedIds.delete(p.id)) syncWidget();
            await refresh();
          } catch (err) { toast(`Delete failed: ${err.message}`, "error"); }
        } },
    ]);
  };

  // Single click handler routes: + tile (add), checkbox (toggle), tile (toggle/range).
  grid.addEventListener("click", (e) => {
    const tile = e.target.closest(".pl-tile");
    if (!tile) return;
    if (tile.classList.contains("pl-add")) {
      _openAddPrompt();
      return;
    }
    const id = tile.dataset.promptId;
    if (!id) return;
    const idx = _indexFromTile(tile);
    if (e.target.closest(".pl-tile-check")) {
      e.stopPropagation();
      if (checkedIds.has(id)) checkedIds.delete(id);
      else checkedIds.add(id);
      focusedIndex = idx;
      applySelectionChange([id]);
      return;
    }
    _toggleSelectionAt(id, idx, e);
  });

  grid.addEventListener("contextmenu", (e) => {
    const tile = e.target.closest(".pl-tile");
    if (!tile || tile.classList.contains("pl-add")) return;
    const p = _entryFromTile(tile);
    if (!p) return;
    _openTileContextMenu(e, p);
  });

  // Drag handlers — fire only in Manual sort mode. dataTransfer carries the
  // dragged id; the drop handler reorders against lastVisible's id list.
  grid.addEventListener("dragstart", (e) => {
    if (sortSelect.value !== "manual") return;
    const tile = e.target.closest(".pl-tile");
    if (!tile || tile.classList.contains("pl-add")) return;
    tile.classList.add("dragging");
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", tile.dataset.promptId || "");
  });
  grid.addEventListener("dragend", (e) => {
    e.target.closest(".pl-tile")?.classList.remove("dragging");
  });
  grid.addEventListener("dragover", (e) => {
    if (sortSelect.value !== "manual") return;
    const tile = e.target.closest(".pl-tile");
    if (!tile || tile.classList.contains("pl-add")) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    tile.classList.add("drag-over");
  });
  grid.addEventListener("dragleave", (e) => {
    e.target.closest(".pl-tile")?.classList.remove("drag-over");
  });
  grid.addEventListener("drop", async (e) => {
    if (sortSelect.value !== "manual") return;
    const tile = e.target.closest(".pl-tile");
    if (!tile) return;
    e.preventDefault();
    tile.classList.remove("drag-over");
    const draggedId = e.dataTransfer.getData("text/plain");
    const targetId = tile.dataset.promptId;
    if (!draggedId || draggedId === targetId) return;
    const ids = lastVisible.map(v => v.id);
    const from = ids.indexOf(draggedId);
    const to = ids.indexOf(targetId);
    if (from < 0 || to < 0) return;
    ids.splice(to, 0, ids.splice(from, 1)[0]);
    try {
      await reorderPrompts(ids);
      await refresh();
    } catch (err) { toast(`Reorder failed: ${err.message}`, "error"); }
  });
  bulkExportBtn.onclick = withBusy(bulkExportBtn, "Exporting…", async () => {
    const ids = [...checkedIds];
    if (!ids.length) return;
    try {
      const { blob, count } = await exportZip(ids);
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      downloadBlob(blob, `ribbity-export-${stamp}-${count}prompts.zip`);
      toast(`Exported ${count} prompt${count === 1 ? "" : "s"}.`, "success");
    } catch (e) { toast(`Export failed: ${e.message}`, "error"); }
  });
  bulkDeleteBtn.onclick = withBusy(bulkDeleteBtn, "Deleting…", async () => {
    const ids = [...checkedIds];
    if (!ids.length) return;
    if (!await confirmDestructive(`Delete ${ids.length} prompts?`)) return;
    try {
      await bulkDelete(ids);
      checkedIds.clear();
      syncWidget();
      await refresh();
      toast(`Deleted ${ids.length} prompt${ids.length === 1 ? "" : "s"}.`, "success");
    } catch (e) { toast(`Delete failed: ${e.message}`, "error"); }
  });
  bulkDuplicateBtn.onclick = withBusy(bulkDuplicateBtn, "Duplicating…", async () => {
    const ids = [...checkedIds];
    if (!ids.length) return;
    let dupes = 0;
    let fails = 0;
    for (const id of ids) {
      try {
        await duplicatePrompt(id);
        dupes++;
      } catch (e) { fails++; }
    }
    await refresh();
    if (fails === 0) {
      toast(`Duplicated ${dupes} prompt${dupes === 1 ? "" : "s"}.`, "success");
    } else {
      toast(`Duplicated ${dupes}; ${fails} failed (see console).`, fails === ids.length ? "error" : "warning");
    }
  });
  bulkTagBtn.onclick = withBusy(bulkTagBtn, "Tagging…", async () => {
    const ids = [...checkedIds];
    if (!ids.length) return;
    const input = prompt(`Add tags to ${ids.length} prompts (comma-separated). Prefix - to remove (e.g. "-old, new"):`, "");
    if (input === null) return;
    const adds = [], removes = [];
    for (const raw of input.split(",")) {
      const t = raw.trim().toLowerCase();
      if (!t) continue;
      if (t.startsWith("-")) removes.push(t.slice(1).trim());
      else adds.push(t);
    }
    if (!adds.length && !removes.length) return;
    let ok = 0;
    const fails = [];
    for (const id of ids) {
      const p = prompts.find(x => x.id === id);
      if (!p) { fails.push({ id, reason: "not found" }); continue; }
      const newTags = new Set([...(p.tags || []), ...adds]);
      for (const r of removes) newTags.delete(r);
      const fd = new FormData();
      fd.append("id", id);
      fd.append("name", p.name);
      fd.append("text", p.text || "");
      fd.append("tags", [...newTags].join(", "));
      try {
        await api.fetchApi("/prompt_library/upsert", { method: "POST", body: fd });
        ok++;
      } catch (e) {
        fails.push({ id, reason: e.message });
      }
    }
    await refresh();
    if (!fails.length) {
      toast(`Tagged ${ok} prompt${ok === 1 ? "" : "s"}.`, "success");
    } else {
      console.warn("[PromptLibrary] bulk tag failures:", fails);
      toast(`${ok}/${ids.length} tagged; ${fails.length} failed (see console).`, "error");
    }
  });

  const updateModelSelect = () => {
    const models = new Set();
    for (const p of prompts) {
      for (const t of p.tags || []) {
        if (t.startsWith("model:")) models.add(t.slice(6));
      }
    }
    const sorted = [...models].sort();
    const previous = modelSelect.value || "";
    modelSelect.replaceChildren();
    const allOpt = document.createElement("option");
    allOpt.value = "";
    allOpt.textContent = sorted.length ? "All models" : "(no model tags)";
    modelSelect.appendChild(allOpt);
    for (const m of sorted) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      modelSelect.appendChild(opt);
    }
    modelSelect.value = sorted.includes(previous) ? previous : "";
    modelSelect.disabled = sorted.length === 0;
  };

  const renderTags = () => {
    // model:* tags are lifted into the modelSelect dropdown — exclude them here.
    // Other tags group by the prefix before ':' (e.g. "style:cyberpunk" -> group "style").
    const groups = new Map();
    for (const p of prompts) {
      for (const t of p.tags || []) {
        if (t.startsWith("model:")) continue;
        const idx = t.indexOf(":");
        const [g, label] = idx > 0 ? [t.slice(0, idx), t.slice(idx + 1)] : ["general", t];
        if (!groups.has(g)) groups.set(g, new Map());
        groups.get(g).set(t, label);
      }
    }

    tagsRow.replaceChildren();

    const topGroup = document.createElement("div");
    topGroup.className = "pl-tag-group";
    const allChip = document.createElement("div");
    allChip.className = "pl-tag-chip all" + (activeTags.size === 0 ? " active" : "");
    allChip.textContent = "All";
    allChip.onclick = () => { activeTags.clear(); render(); };
    topGroup.appendChild(allChip);
    // ANY / ALL toggle. Only meaningful when 2+ tags are active, but we always
    // show it so the user can pre-set the mode before clicking chips.
    const modeChip = document.createElement("div");
    modeChip.className = "pl-tag-chip pl-tag-mode";
    modeChip.title = tagFilterMode === "all"
      ? "Match prompts that have ALL active tags. Click to switch to ANY."
      : "Match prompts that have ANY active tag. Click to switch to ALL.";
    modeChip.textContent = tagFilterMode === "all" ? "ALL" : "ANY";
    modeChip.onclick = () => {
      tagFilterMode = tagFilterMode === "all" ? "any" : "all";
      render();
    };
    topGroup.appendChild(modeChip);
    if (groups.size === 0) {
      const hint = document.createElement("span");
      hint.className = "pl-tag-group-label";
      hint.textContent = "no tags yet";
      topGroup.appendChild(hint);
    }
    tagsRow.appendChild(topGroup);

    const sortedGroups = [...groups.keys()].sort((a, b) =>
      a === "general" ? 1 : b === "general" ? -1 : a.localeCompare(b)
    );
    for (const g of sortedGroups) {
      const row = document.createElement("div");
      row.className = "pl-tag-group";
      const lbl = document.createElement("span");
      lbl.className = "pl-tag-group-label";
      lbl.textContent = g === "general" ? "" : g;
      row.appendChild(lbl);
      const tagsInGroup = [...groups.get(g).entries()].sort((a, b) => a[1].localeCompare(b[1]));
      for (const [fullTag, label] of tagsInGroup) {
        const chip = document.createElement("div");
        chip.className = "pl-tag-chip" + (activeTags.has(fullTag) ? " active" : "");
        chip.textContent = label;
        chip.title = fullTag;
        chip.onclick = () => {
          if (activeTags.has(fullTag)) activeTags.delete(fullTag);
          else activeTags.add(fullTag);
          render();
        };
        row.appendChild(chip);
      }
      tagsRow.appendChild(row);
    }
  };

  const render = () => {
    writeState();
    updateModelSelect();
    renderTags();
    grid.replaceChildren();
    // Multi-term search: each whitespace-separated term must match somewhere
    // in name/text/tags/id (AND across terms, OR within sources). Quotes are
    // not parsed — search is plain substring per term.
    const terms = filter.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    let visible = prompts;
    const model = modelSelect.value;
    if (model) {
      const want = `model:${model}`;
      visible = visible.filter(p => (p.tags || []).includes(want));
    }
    if (activeTags.size) {
      const tagPredicate = tagFilterMode === "all"
        ? (p) => [...activeTags].every(t => (p.tags || []).includes(t))
        : (p) => (p.tags || []).some(t => activeTags.has(t));
      visible = visible.filter(tagPredicate);
    }
    if (favoritesOnly) {
      visible = visible.filter(p => (p.rating || 0) >= 4);
    }
    if (terms.length) {
      visible = visible.filter(p => {
        // Search reaches into LoRA fields too — file paths and trigger
        // words are part of an entry's identity once you've curated a
        // stack. Without this, a query like 'turbo' or 'green eyes' that
        // only appears in the loras list misses entries that legitimately
        // match. Disabled rows still match (the user might re-enable them
        // later); their text is still authored on the entry.
        const loraText = (p.loras || []).map(l =>
          `${l.name || ""} ${l.triggers || ""}`).join(" ");
        const haystacks = [
          p.name.toLowerCase(),
          (p.text || "").toLowerCase(),
          (p.negative || "").toLowerCase(),
          (p.tags || []).join(" ").toLowerCase(),
          (p.id || "").toLowerCase(),
          (p.notes || "").toLowerCase(),
          loraText.toLowerCase(),
        ];
        return terms.every(term => haystacks.some(h => h.includes(term)));
      });
    }
    const mode = SORT_MODES[sortSelect.value] || SORT_MODES.name_asc;
    visible = [...visible].sort(mode.cmp);
    // Lift selected tiles to the top so the user can see what's currently
    // checked without scrolling. Stable within each group so the sort order
    // is preserved; deselecting drops the tile back to its natural spot.
    // Skipped in Manual mode because drag-reorder relies on the natural order.
    if (sortSelect.value !== "manual" && checkedIds.size) {
      visible.sort((a, b) => Number(checkedIds.has(b.id)) - Number(checkedIds.has(a.id)));
    }
    lastVisible = visible;
    // Drop checked ids that no longer exist in the library; keep ones that
    // are merely filtered out so multi-select persists across filter changes.
    // Skip when prompts is empty: that means fetchList hasn't resolved yet
    // (we got here from onConfigure's render call running before the initial
    // refresh awaits), and pruning would wipe a freshly-restored selection.
    if (prompts.length > 0) {
      for (const id of [...checkedIds]) {
        if (!prompts.some(p => p.id === id)) checkedIds.delete(id);
      }
    }
    updateBulkBar();
    countBadge.textContent = visible.length === prompts.length
      ? `${prompts.length}`
      : `${visible.length}/${prompts.length}`;
    if (focusedIndex >= visible.length) focusedIndex = visible.length - 1;

    if (prompts.length === 0) {
      const empty = document.createElement("div");
      empty.className = "pl-empty-state";
      const head = document.createElement("strong");
      head.textContent = "No prompts yet";
      empty.appendChild(head);
      empty.appendChild(document.createTextNode(
        "Click the + tile to add one, or use Import to seed the library from a CSV/ZIP."
      ));
      grid.appendChild(empty);
    } else if (visible.length === 0) {
      const empty = document.createElement("div");
      empty.className = "pl-empty-state";
      const head = document.createElement("strong");
      head.textContent = "No matches";
      empty.appendChild(head);
      empty.appendChild(document.createTextNode(
        "Clear the search/filter or pick a different model."
      ));
      grid.appendChild(empty);
    }
    const isManual = sortSelect.value === "manual";

    // Build a single tile element. Extracted from the inline forEach so
    // the chunked-render scheduler below can call it from a deferred
    // callback without duplicating the construction code.
    const buildTile = (p, idx) => {
      const tile = document.createElement("div");
      tile.className = "pl-tile"
        + (checkedIds.has(p.id) ? " selected" : "")
        + (idx === focusedIndex ? " focused" : "");
      const tipParts = [p.name];
      if (p.rating) tipParts.push("★".repeat(p.rating) + "☆".repeat(5 - p.rating));
      if (p.notes) {
        const excerpt = p.notes.length > 140 ? p.notes.slice(0, 140) + "…" : p.notes;
        tipParts.push(excerpt);
      }
      tile.title = tipParts.join("\n");
      tile.dataset.promptId = p.id;
      tile.tabIndex = -1;
      tile.setAttribute("role", "option");
      tile.setAttribute("aria-label", p.name);
      tile.setAttribute("aria-selected", checkedIds.has(p.id) ? "true" : "false");
      tile.draggable = isManual;

      const tileImg = document.createElement("div");
      tileImg.className = "pl-tile-img";
      if (p.has_image) {
        const img = document.createElement("img");
        img.src = imageUrl(p.id);
        img.loading = "lazy";
        img.alt = p.name;
        img.onerror = () => {
          img.replaceWith(Object.assign(document.createElement("div"),
            { className: "pl-placeholder", textContent: "?" }));
        };
        tileImg.appendChild(img);
      } else {
        const ph = document.createElement("div");
        ph.className = "pl-placeholder";
        ph.textContent = "T";
        tileImg.appendChild(ph);
      }

      const checkbox = document.createElement("div");
      checkbox.className = "pl-tile-check";
      checkbox.textContent = checkedIds.has(p.id) ? "✓" : "";
      checkbox.title = "Toggle selection";
      tileImg.appendChild(checkbox);
      if (p.rating) {
        const ratingBadge = document.createElement("div");
        ratingBadge.className = "pl-rating-badge";
        ratingBadge.textContent = `${"★".repeat(p.rating)}`;
        ratingBadge.title = `${p.rating}/5`;
        ratingBadge.setAttribute("aria-label", `${p.rating} of 5 stars`);
        tileImg.appendChild(ratingBadge);
      }
      tile.appendChild(tileImg);

      const nm = document.createElement("div");
      nm.className = "pl-name";
      nm.textContent = p.name;
      tile.appendChild(nm);
      return tile;
    };

    const buildAddTile = () => {
      const t = document.createElement("div");
      t.className = "pl-tile pl-add";
      t.textContent = "+";
      t.title = "Add prompt";
      return t;
    };

    // Bump the render token. Any in-flight chunked render from a previous
    // render() call sees a token mismatch and exits before touching the
    // grid — prevents stale tiles being appended on top of a fresh render.
    _renderToken++;
    const myToken = _renderToken;

    // Synchronous render path: small libraries (< chunk threshold) build
    // every tile in one frame as before. Above the threshold, render the
    // first chunk synchronously (so the above-fold tiles paint instantly)
    // and stream the rest via requestIdleCallback so the browser stays
    // responsive during the long render. For a 1000-tile library this
    // brings the first-paint hitch from ~100 ms to ~10 ms.
    if (visible.length < _CHUNKED_RENDER_THRESHOLD) {
      const fragment = document.createDocumentFragment();
      visible.forEach((p, idx) => fragment.appendChild(buildTile(p, idx)));
      fragment.appendChild(buildAddTile());
      grid.appendChild(fragment);
      return;
    }

    // Above-threshold path. First chunk synchronous, rest deferred.
    const firstChunk = document.createDocumentFragment();
    const initial = Math.min(_TILES_PER_CHUNK, visible.length);
    for (let i = 0; i < initial; i++) firstChunk.appendChild(buildTile(visible[i], i));
    grid.appendChild(firstChunk);

    let nextIdx = initial;
    const renderChunk = () => {
      if (myToken !== _renderToken) return;  // a fresher render is in flight
      const stop = Math.min(nextIdx + _TILES_PER_CHUNK, visible.length);
      const chunk = document.createDocumentFragment();
      for (let i = nextIdx; i < stop; i++) chunk.appendChild(buildTile(visible[i], i));
      grid.appendChild(chunk);
      nextIdx = stop;
      if (nextIdx < visible.length) {
        _scheduleIdle(renderChunk);
      } else {
        // All real tiles in. The + add-tile sits at the very end so it's
        // not interleaved between chunks during the streaming render.
        if (myToken === _renderToken) grid.appendChild(buildAddTile());
      }
    };
    _scheduleIdle(renderChunk);
  };

  const refresh = async () => {
    try {
      prompts = await fetchList();
      render();
    } catch (e) {
      grid.replaceChildren();
      const err = document.createElement("div");
      err.className = "pl-status error";
      err.textContent = `failed to load: ${e.message}`;
      grid.appendChild(err);
    }
    // Health-badge refresh runs in the background — the gallery doesn't
    // wait on it. Network failures hide the badge silently (the gallery
    // itself works regardless of validator availability).
    refreshHealthBadge();
  };

  // Debounce search input — every keystroke would otherwise trigger a full
  // grid rebuild (700+ tile DOM nodes for big libraries). 80 ms feels
  // instant when you stop typing but coalesces a burst of keystrokes into
  // one render. The clear-search button still calls render() directly so
  // clicking × is immediate.
  let _filterDebounce = null;
  filter.addEventListener("input", () => {
    if (_filterDebounce) clearTimeout(_filterDebounce);
    _filterDebounce = setTimeout(() => {
      _filterDebounce = null;
      render();
    }, 80);
  });
  refreshBtn.onclick = withBusy(refreshBtn, "…", refresh);
  applySize();
  applyView();
  sortSelect.onchange = () => {
    localStorage.setItem(SORT_KEY, sortSelect.value);
    render();
  };
  modelSelect.onchange = render;
  async function handleImportFile(file, mode) {
    if (!file) return;
    const isZip = file.name.toLowerCase().endsWith(".zip") || file.type === "application/zip";
    const isCsv = file.name.toLowerCase().endsWith(".csv") || file.type === "text/csv";
    if (!isZip && !isCsv) {
      toast(`Unsupported file: ${file.name} (need .csv or .zip)`, "error");
      return;
    }
    const originalLabel = importBtn.textContent;
    importBtn.disabled = true;
    importBtn.classList.add("pl-busy");
    importBtn.textContent = "Importing…";
    try {
      const result = isZip ? await importZip(file, mode) : await importCsv(file, mode);
      const kind = isZip ? "ZIP" : "CSV";
      const errs = result.errors?.length || 0;
      const skipped = result.skipped || 0;
      const summary = `${kind} import (${mode}): ${result.added} added, `
        + `${result.updated} updated, ${skipped} skipped`
        + (errs ? ` (${errs} errors — see console)` : "");
      if (errs) console.warn(`[PromptLibrary] ${kind} import errors:`, result.errors);
      toast(summary, errs ? "error" : "success", 6000);
      await refresh();
    } catch (e) {
      toast(`Import failed: ${e.message}`, "error");
    } finally {
      importBtn.disabled = false;
      importBtn.classList.remove("pl-busy");
      importBtn.textContent = originalLabel;
    }
  }
  fileInput.onchange = async () => {
    const file = fileInput.files[0];
    const mode = importBtn.dataset.importMode || "add_only";
    try { await handleImportFile(file, mode); }
    finally {
      fileInput.value = "";
      importBtn.dataset.importMode = "add_only"; // reset for next time
    }
  };

  undoBtn.onclick = withBusy(undoBtn, "Restoring…", async () => {
    if (!await confirmDestructive(
      "Restore the library from the most recent pre-import snapshot? "
      + "This will replace the current library state. The current state is "
      + "snapshotted first so you can re-undo.",
      { confirmLabel: "Undo import" }
    )) return;
    try {
      const result = await restoreLastSnapshot();
      toast(`Restored from ${result.restored_from} (${result.entries} entries).`, "success", 6000);
      await refresh();
    } catch (e) {
      toast(`Undo failed: ${e.message}`, "error");
    }
  });

  // Drag-and-drop file import on the container. We only react to drops that
  // carry actual files (dataTransfer.types includes "Files"); workflow-JSON
  // drops, internal tile reorder drags, and chrome-internal drags pass through.
  const onDragEnter = (e) => {
    if (!e.dataTransfer?.types?.includes("Files")) return;
    e.preventDefault();
    container.classList.add("pl-drop-target");
  };
  const onDragOver = (e) => {
    if (!e.dataTransfer?.types?.includes("Files")) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  };
  const onDragLeave = (e) => {
    if (e.target === container) container.classList.remove("pl-drop-target");
  };
  const onDrop = async (e) => {
    if (!e.dataTransfer?.types?.includes("Files")) return;
    e.preventDefault();
    container.classList.remove("pl-drop-target");
    const file = e.dataTransfer.files?.[0];
    await handleImportFile(file);
  };
  container.addEventListener("dragenter", onDragEnter);
  container.addEventListener("dragover", onDragOver);
  container.addEventListener("dragleave", onDragLeave);
  container.addEventListener("drop", onDrop);

  exportBtn.onclick = withBusy(exportBtn, "Exporting…", async () => {
    if (!lastVisible.length) {
      toast("Nothing to export (the current filter shows no prompts).", "info");
      return;
    }
    try {
      const exportingAll = lastVisible.length === prompts.length;
      const ids = exportingAll ? [] : lastVisible.map(p => p.id);
      const { blob, count } = await exportZip(ids);
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      downloadBlob(blob, `ribbity-export-${stamp}-${count}prompts.zip`);
      toast(`Exported ${count} prompt${count === 1 ? "" : "s"}.`, "success");
    } catch (e) {
      toast(`Export failed: ${e.message}`, "error");
    }
  });

  scanLorasBtn.onclick = withBusy(scanLorasBtn, "Scanning…", async () => {
    // Default-on: include triggers from safetensors metadata, skip existing
    // entries so re-runs don't clobber user edits. Hold shift while clicking
    // to switch into refresh-mode (re-reads metadata + thumbnails for
    // already imported LoRAs).
    const refreshExisting = !!(window.event && window.event.shiftKey);
    try {
      const res = await api.fetchApi("/prompt_library/scan_loras", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          default_weight: 1.0,
          include_triggers: true,
          refresh_existing: refreshExisting,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const errs = data.errors?.length || 0;
      const summary = `LoRA scan: ${data.added} added, ${data.updated} refreshed, `
        + `${data.skipped} skipped${errs ? ` (${errs} errors — see console)` : ""}`;
      if (errs) console.warn("[PromptLibrary] LoRA scan errors:", data.errors);
      toast(summary, errs ? "error" : "success", 6000);
      await refresh();
    } catch (e) {
      toast(`LoRA scan failed: ${e.message}`, "error");
    }
  });

  importBgBtn.onclick = withBusy(importBgBtn, "Importing…", async () => {
    const refreshExisting = !!(window.event && window.event.shiftKey);
    try {
      const res = await api.fetchApi("/prompt_library/import_backgrounds", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_existing: refreshExisting }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const errs = data.errors?.length || 0;
      const summary = `Background import: ${data.added} added, ${data.updated} refreshed, `
        + `${data.skipped} skipped${errs ? ` (${errs} errors — see console)` : ""}`;
      if (errs) console.warn("[PromptLibrary] Background import errors:", data.errors);
      toast(summary, errs ? "error" : "success", 6000);
      await refresh();
    } catch (e) {
      toast(`Background import failed: ${e.message}`, "error");
    }
  });

  // Queue-many: drives the workflow's first PromptLibrary node through every
  // checked-or-visible entry, queueing one run per id. Pairs with a
  // PromptLibrarySave node wired to your output to auto-fill thumbnails.
  queueAllBtn.onclick = withBusy(queueAllBtn, "Queueing…", async () => {
    const ids = checkedIds.size > 0
      ? [...checkedIds]
      : lastVisible.map(p => p.id);
    if (!ids.length) {
      toast("Nothing to queue (no selection, no visible entries).", "info");
      return;
    }
    // Drive THIS gallery's parent node, not the first PromptLibrary in the
    // graph. That makes Queue ▶▶ work from a Multi panel (cycles its own
    // panel, leaving the others fixed) or a Style node (cycles styles)
    // without forcing a separate Library node into the workflow.
    const libraryNode = node;
    if (!idWidget) {
      toast("Gallery has no prompt_id widget — workflow may be from an older "
        + "version. Re-add the node.", "error");
      return;
    }
    // If the workflow includes a Thumbnail Saver, drive its prompt_id in
    // lock-step so each run thumbnails the entry it loaded.
    const saverNode = app.graph?._nodes?.find(n => n.type === "PromptLibraryThumbnailSaver");
    const saverIdWidget = saverNode?.widgets?.find(w => w.name === "prompt_id");
    if (ids.length > 10) {
      const ok = await confirmDestructive(
        `Queue ${ids.length} workflow runs? Each will run with a different library entry.`,
        { confirmLabel: "Queue all" });
      if (!ok) return;
    }
    // Snapshot the canonical widget values so we can restore them after the
    // loop. Without this, idWidget.value ends up as just the last queued id
    // — desyncing from checkedIds (which the user still sees highlighted)
    // and shrinking the persisted multi-select to a single entry on the
    // next page refresh.
    const savedIdWidgetValue = idWidget.value;
    const savedSaverIdValue = saverIdWidget?.value;
    let queued = 0;
    const fails = [];
    try {
      for (const id of ids) {
        idWidget.value = id;
        if (saverIdWidget) saverIdWidget.value = id;
        libraryNode.setDirtyCanvas?.(true, true);
        if (saverNode) saverNode.setDirtyCanvas?.(true, true);
        try {
          await app.queuePrompt(0, 1);
          queued++;
        } catch (e) {
          fails.push({ id, error: e?.message || String(e) });
        }
      }
    } finally {
      idWidget.value = savedIdWidgetValue;
      if (saverIdWidget) saverIdWidget.value = savedSaverIdValue;
      libraryNode.setDirtyCanvas?.(true, true);
      if (saverNode) saverNode.setDirtyCanvas?.(true, true);
    }
    if (!fails.length) {
      toast(`Queued ${queued} workflow run${queued === 1 ? "" : "s"}.`, "success", 6000);
    } else {
      console.warn("[PromptLibrary] queue failures:", fails);
      toast(`${queued}/${ids.length} queued; ${fails.length} failed (see console).`,
        "error", 8000);
    }
  });

  // Refresh whenever any save/delete fires server-side (incl. the Save node).
  const onExternal = () => refresh();
  window.addEventListener("prompt-library-updated", onExternal);

  // Make the grid focusable so keyboard nav has somewhere to land.
  grid.tabIndex = 0;
  grid.style.outline = "none";

  const tilesPerRow = () => {
    const tile = grid.querySelector(".pl-tile:not(.pl-add)");
    if (!tile) return 1;
    return Math.max(1, Math.floor(grid.clientWidth / tile.offsetWidth));
  };

  const moveFocus = (delta) => {
    if (!lastVisible.length) return;
    if (focusedIndex < 0) focusedIndex = 0;
    else focusedIndex = Math.max(0, Math.min(lastVisible.length - 1, focusedIndex + delta));
    // Just shift the .focused class to the new tile in place — the previous
    // path called the full render(), which on a 700-tile library meant
    // arrow-key navigation rebuilt 700 DOM nodes per keystroke.
    _refreshFocusedTile();
    const target = grid.querySelectorAll(".pl-tile")[focusedIndex];
    target?.scrollIntoView({ block: "nearest" });
  };

  const onGridKey = (e) => {
    // Slash focuses search from anywhere within the gallery.
    if (e.key === "/" && document.activeElement !== filter) {
      e.preventDefault();
      filter.focus();
      filter.select();
      return;
    }
    // Other keys only fire when grid is focused (not search/etc).
    if (document.activeElement !== grid) return;
    if (e.key === "ArrowRight") { e.preventDefault(); moveFocus(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); moveFocus(-1); }
    else if (e.key === "ArrowDown") { e.preventDefault(); moveFocus(tilesPerRow()); }
    else if (e.key === "ArrowUp") { e.preventDefault(); moveFocus(-tilesPerRow()); }
    else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      const p = lastVisible[focusedIndex];
      if (!p) return;
      if (checkedIds.has(p.id)) checkedIds.delete(p.id);
      else checkedIds.add(p.id);
      applySelectionChange([p.id]);
    }
    else if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      const p = lastVisible[focusedIndex];
      if (!p) return;
      (async () => {
        if (!await confirmDestructive(`Delete "${p.name}"?`)) return;
        try {
          await deletePrompt(p.id);
          if (checkedIds.delete(p.id)) syncWidget();
          await refresh();
        } catch (err) { toast(`Delete failed: ${err.message}`, "error"); }
      })();
    }
    else if (e.key === "Escape") {
      e.preventDefault();
      checkedIds.clear();
      syncWidget();
      focusedIndex = -1;
      render();
    }
  };
  container.addEventListener("keydown", onGridKey);
  // Stop key events from propagating to LiteGraph when interacting with the gallery.
  container.addEventListener("keydown", (e) => e.stopPropagation());
  // Let the wheel scroll the grid (and other inner scrollers) instead of
  // zooming the LiteGraph canvas. Capture phase + manual scroll so we beat
  // LiteGraph's wheel handler, which otherwise eats the event for canvas
  // zoom even though the cursor is over our DOM widget.
  container.addEventListener("wheel", (e) => {
    let dy = e.deltaY;
    if (e.deltaMode === 1) dy *= 16;          // lines → px
    else if (e.deltaMode === 2) dy *= e.target?.clientHeight || 400;  // pages → px
    for (let el = e.target; el && el !== container.parentNode; el = el.parentNode) {
      if (!(el instanceof HTMLElement)) continue;
      const style = getComputedStyle(el);
      if (!/(auto|scroll)/.test(style.overflowY)) continue;
      if (el.scrollHeight <= el.clientHeight) continue;
      const atTop = el.scrollTop <= 0;
      const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 1;
      if ((dy < 0 && atTop) || (dy > 0 && atBottom)) return;  // let canvas zoom at edges
      el.scrollTop += dy;
      e.preventDefault();
      e.stopPropagation();
      return;
    }
  }, { passive: false, capture: true });

  // Thorough teardown — runs from the node's onRemoved hook so deleting a
  // Library node leaves no residual state. Without this, deleted nodes
  // briefly showed stale 'selected' tiles in the DOM until GC, and the
  // hidden idWidget retained its prior comma-separated id list, which
  // Comfy's undo/restore path could re-read into a fresh instance.
  const cleanup = () => {
    window.removeEventListener("prompt-library-updated", onExternal);
    if (_filterDebounce) {
      clearTimeout(_filterDebounce);
      _filterDebounce = null;
    }
    checkedIds.clear();
    activeTags.clear();
    // Clear the rendered grid + tag chips + bulk bar so the DOM is empty
    // before GC. Anything still holding a reference to the container sees
    // a clean slate instead of last-known-selection.
    try {
      grid.replaceChildren();
      tagsRow?.replaceChildren?.();
      bulkBar.style.display = "none";
    } catch (_e) {}
    // Reset the underlying widget value too — a future undo/redo or
    // workflow re-paste should NOT inherit the deleted node's selection.
    if (idWidget) idWidget.value = "";
    // Drop the localStorage tier so a future node with the same id (or a
    // different workflow loaded into this tab that reuses the id) starts
    // clean instead of inheriting this node's selection as a ghost.
    lsClearSel(localStorage, node?.id, propsKey);
  };
  container._promptLibraryCleanup = cleanup;

  grid.replaceChildren(Object.assign(document.createElement("div"), {
    className: "pl-status", textContent: "loading...",
  }));
  syncFromWidget("buildGallery.init");
  refresh();

  return { container, refresh, render, syncFromWidget, cleanup };
}

function buildMultiPanel(node, panelIndex) {
  const labelWidget = node.widgets.find(w => w.name === `label_${panelIndex}`);
  const idWidget = node.widgets.find(w => w.name === `prompt_id_${panelIndex}`);
  const sepWidget = node.widgets.find(w => w.name === `separator_${panelIndex}`);

  // Hide all three underlying string widgets — gallery + header drive them.
  for (const w of [labelWidget, idWidget, sepWidget]) {
    if (!w) continue;
    w.hidden = true;
    w.computeSize = () => [0, -4];
    w.draw = () => {};
  }

  const panel = document.createElement("div");
  panel.className = "pl-panel";

  const header = document.createElement("div");
  header.className = "pl-panel-header";
  header.contentEditable = "true";
  header.spellcheck = false;
  header.textContent = labelWidget?.value || `Panel ${panelIndex}`;
  header.title = "Click to rename this panel";
  header.addEventListener("input", () => {
    if (labelWidget) labelWidget.value = header.textContent;
  });
  // Stop typed keys from reaching LiteGraph (which would delete the node, etc.)
  for (const ev of ["keydown", "keyup", "keypress"]) {
    header.addEventListener(ev, (e) => e.stopPropagation());
  }
  // Enter commits without inserting a newline.
  header.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); header.blur(); }
  });

  const body = document.createElement("div");
  body.className = "pl-panel-body";

  const { container, render, syncFromWidget, cleanup } = buildGallery(node, idWidget, `pl_state_${panelIndex}`);
  body.appendChild(container);

  panel.appendChild(header);
  panel.appendChild(body);
  panel.style.minHeight = "260px";
  panel.style.width = "100%";

  return { panel, render, syncFromWidget, cleanup, header, labelWidget };
}

function registerMultiNode(nodeType) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated?.apply(this, arguments);

    this._promptLibraryPanels = [];
    for (let i = 1; i <= MULTI_PANELS; i++) {
      const built = buildMultiPanel(this, i);
      this.addDOMWidget(`panel_${i}`, "PromptLibraryGallery", built.panel, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 260,
      });
      this._promptLibraryPanels.push(built);
    }

    this.size = [380, 820];
    if (typeof this.setSize === "function") this.setSize([380, 820]);
    return r;
  };

  const onConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const r = onConfigure?.apply(this, arguments);
    // Re-sync each panel with its (now-restored-from-workflow) widget values.
    for (const p of this._promptLibraryPanels || []) {
      if (p.labelWidget) p.header.textContent = p.labelWidget.value || p.header.textContent;
      p.syncFromWidget?.("multi.onConfigure");
      p.render?.();
    }
    return r;
  };

  const onRemoved = nodeType.prototype.onRemoved;
  nodeType.prototype.onRemoved = function () {
    for (const p of this._promptLibraryPanels || []) p.cleanup?.();
    this._promptLibraryPanels = [];
    return onRemoved?.apply(this, arguments);
  };
}

// =============================================================================
// Background node — adds a small "🔒 LOCKED" header above the textarea so the
// node's role as a continuity anchor is obvious in the workflow. Pure visual
// — the value still flows through the underlying STRING widget unchanged.
// =============================================================================

function registerBackgroundNode(nodeType) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated?.apply(this, arguments);
    const banner = document.createElement("div");
    banner.className = "pl-bg-locked-wrap";
    const lockIcon = document.createElement("span");
    lockIcon.className = "lock";
    lockIcon.textContent = "🔒";
    const lockText = document.createElement("span");
    lockText.textContent = "LOCKED — every frame uses this background";
    banner.append(lockIcon, lockText);
    this.addDOMWidget("locked_banner", "PromptLibraryBackgroundBanner", banner, {
      serialize: false, hideOnZoom: false, getMinHeight: () => 28,
    });
    return r;
  };
}

// =============================================================================
// Comic Frame node — DOM widget that edits an ordered list of per-frame
// action strings. The list is serialized into the hidden `frames_json` STRING
// widget so it round-trips with the workflow. The frame_index INT widget gets
// its `max` clamped to the current frame count so the user can't pick out of
// range.
// =============================================================================

function buildComicEditor(node, framesWidget, indexWidget) {
  const root = document.createElement("div");
  root.className = "pl-comic";

  const parseFrames = () => {
    try {
      const v = JSON.parse(framesWidget?.value || "[]");
      return Array.isArray(v) ? v.map(s => String(s)) : [];
    } catch { return []; }
  };
  const writeFrames = (frames) => {
    if (framesWidget) framesWidget.value = JSON.stringify(frames);
    if (indexWidget) {
      indexWidget.options.max = Math.max(1, frames.length);
      if (indexWidget.value > frames.length) indexWidget.value = Math.max(1, frames.length);
    }
    node.setDirtyCanvas?.(true, true);
  };

  let frames = parseFrames();

  const header = document.createElement("div");
  header.className = "pl-comic-header";
  const title = document.createElement("strong");
  title.textContent = "Frames";
  const count = document.createElement("span");
  count.className = "pl-count-badge";
  header.append(title, count);

  const list = document.createElement("div");
  list.className = "pl-comic-frames";

  const toolbar = document.createElement("div");
  toolbar.className = "pl-comic-toolbar";
  const addBtn = document.createElement("button");
  addBtn.className = "pl-btn";
  addBtn.textContent = "+ Add frame";
  const clearBtn = document.createElement("button");
  clearBtn.className = "pl-btn";
  clearBtn.textContent = "Clear all";
  toolbar.append(addBtn, clearBtn);

  const render = () => {
    list.replaceChildren();
    count.textContent = `${frames.length} frame${frames.length === 1 ? "" : "s"}`;
    const currentIdx = (indexWidget?.value || 1) - 1;
    frames.forEach((text, i) => {
      const row = document.createElement("div");
      row.className = "pl-frame-row" + (i === currentIdx ? " current" : "");
      const num = document.createElement("div");
      num.className = "pl-frame-num";
      num.textContent = i + 1;
      num.title = "Click to make this the current frame";
      num.onclick = () => {
        if (indexWidget) {
          indexWidget.value = i + 1;
          render();
          node.setDirtyCanvas?.(true, true);
        }
      };
      const ta = document.createElement("textarea");
      ta.className = "pl-frame-text";
      ta.value = text;
      ta.placeholder = `frame ${i + 1} action — e.g. "kicks the door open"`;
      ta.rows = 2;
      ta.oninput = () => { frames[i] = ta.value; writeFrames(frames); };
      for (const ev of ["keydown", "keyup", "keypress"]) {
        ta.addEventListener(ev, (e) => e.stopPropagation());
      }
      const actions = document.createElement("div");
      actions.className = "pl-frame-actions";
      const upBtn = document.createElement("button");
      upBtn.className = "pl-btn"; upBtn.textContent = "↑"; upBtn.title = "Move up";
      upBtn.onclick = () => {
        if (i === 0) return;
        [frames[i - 1], frames[i]] = [frames[i], frames[i - 1]];
        writeFrames(frames); render();
      };
      const downBtn = document.createElement("button");
      downBtn.className = "pl-btn"; downBtn.textContent = "↓"; downBtn.title = "Move down";
      downBtn.onclick = () => {
        if (i >= frames.length - 1) return;
        [frames[i + 1], frames[i]] = [frames[i], frames[i + 1]];
        writeFrames(frames); render();
      };
      const delBtn = document.createElement("button");
      delBtn.className = "pl-btn"; delBtn.textContent = "✕"; delBtn.title = "Delete frame";
      delBtn.style.color = "var(--pl-danger)";
      delBtn.onclick = () => {
        frames.splice(i, 1);
        writeFrames(frames); render();
      };
      actions.append(upBtn, downBtn, delBtn);
      row.append(num, ta, actions);
      list.appendChild(row);
    });
    if (frames.length === 0) {
      const hint = document.createElement("div");
      hint.className = "pl-empty-state";
      hint.textContent = "No frames yet — click \"+ Add frame\" to start your comic.";
      list.appendChild(hint);
    }
  };

  addBtn.onclick = () => {
    frames.push("");
    writeFrames(frames);
    if (indexWidget) indexWidget.value = frames.length;
    render();
  };
  clearBtn.onclick = async () => {
    if (!frames.length) return;
    if (!await confirmDestructive(`Clear all ${frames.length} frames?`, { confirmLabel: "Clear" })) return;
    frames = [];
    writeFrames(frames);
    render();
  };

  // External index changes (user edits the INT widget) → re-highlight the row.
  if (indexWidget) {
    const origCallback = indexWidget.callback;
    indexWidget.callback = function (...args) {
      const r = origCallback?.apply(this, args);
      render();
      return r;
    };
  }

  root.append(header, list, toolbar);
  // Kick the index widget's max into shape on first paint.
  writeFrames(frames);
  render();
  return { root, refresh: () => { frames = parseFrames(); render(); } };
}

function registerComicFrameNode(nodeType) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated?.apply(this, arguments);
    const framesWidget = this.widgets.find(w => w.name === "frames_json");
    const indexWidget = this.widgets.find(w => w.name === "frame_index");
    if (framesWidget) {
      framesWidget.hidden = true;
      framesWidget.computeSize = () => [0, -4];
      framesWidget.draw = () => {};
    }
    const { root, refresh } = buildComicEditor(this, framesWidget, indexWidget);
    root.style.minHeight = "240px";
    root.style.width = "100%";
    this.addDOMWidget("frames", "PromptLibraryComicFrames", root, {
      serialize: false, hideOnZoom: false, getMinHeight: () => 240,
    });
    this._comicRefresh = refresh;
    this.size = [380, 360];
    if (typeof this.setSize === "function") this.setSize([380, 360]);
    return r;
  };

  const onConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const r = onConfigure?.apply(this, arguments);
    this._comicRefresh?.();
    return r;
  };
}

// -----------------------------------------------------------------------------
// Smart Detailer per-target grid — replaces the long stack of hidden widgets
// (enable/threshold/denoise/max/steps/crop/cycles × face/eyes/hands/feet) with
// a compact 5-column grid: 1 row-label column + 4 target columns. The
// underlying widgets stay in `node.widgets` (so workflow JSON serialises
// correctly) but get hidden behind the DOM grid; reads + writes go through
// each cell input -> the underlying widget's .value setter.
// -----------------------------------------------------------------------------
const SMART_DETAILER_NAME = "GrimmRibbitySmartDetailer";

// Names of widgets the preset toolbar reads from / writes to. Anything not
// in this list (image/model/clip/vae/conditioning sockets, the per_target_grid
// DOM widget) is ignored so wiring isn't disturbed.
const SMART_DETAILER_PRESET_KEYS = [
  "enable_face", "enable_eyes", "enable_hands", "enable_feet",
  "bbox_face", "bbox_eyes", "bbox_hands", "bbox_feet", "sam_model",
  "seed", "steps", "cfg", "sampler_name", "scheduler",
  "denoise", "guide_size", "max_size", "bbox_threshold", "max_per_target",
  "tiled_decode", "tiled_encode", "mask_strength", "same_seed_per_target",
  "bypass", "wildcard_prefix",
  "face_threshold", "face_denoise", "face_max", "face_steps",
  "eyes_threshold", "eyes_denoise", "eyes_max", "eyes_steps",
  "hands_threshold", "hands_denoise", "hands_max", "hands_steps",
  "feet_threshold", "feet_denoise", "feet_max", "feet_steps",
  "face_crop_factor", "eyes_crop_factor", "hands_crop_factor", "feet_crop_factor",
  "face_cycles", "eyes_cycles", "hands_cycles", "feet_cycles",
  "force_inpaint", "drop_size", "min_detail_size", "nms_iou",
  "yolo_imgsz", "max_bbox_area_pct", "draw_preview",
];

function _snapshotDetailerSettings(node) {
  // Read the current value of every preset-managed widget on the node.
  // Widgets not present in this node version (older saves loading newer
  // workflows, or vice versa) are skipped — the resulting preset is a
  // sparse map.
  const out = {};
  const byName = {};
  for (const w of node.widgets || []) byName[w.name] = w;
  for (const key of SMART_DETAILER_PRESET_KEYS) {
    const w = byName[key];
    if (!w) continue;
    out[key] = w.value;
  }
  return out;
}

function _applyDetailerSettings(node, settings) {
  // Set every preset-managed widget value from the preset's settings dict.
  // Unknown keys (a future-version preset loaded on an older build) are
  // silently dropped. Missing keys (an older preset missing widgets added
  // later) leave the current value in place.
  if (!settings || typeof settings !== "object") return 0;
  let applied = 0;
  const byName = {};
  for (const w of node.widgets || []) byName[w.name] = w;
  for (const key of Object.keys(settings)) {
    if (!SMART_DETAILER_PRESET_KEYS.includes(key)) continue;
    const w = byName[key];
    if (!w) continue;
    const v = settings[key];
    if (typeof w.callback === "function") {
      try { w.callback(v); } catch (_e) { /* fall through to direct assignment */ }
    }
    w.value = v;
    applied += 1;
  }
  node.setDirtyCanvas?.(true, true);
  return applied;
}

async function _fetchDetailerPresets() {
  try {
    const res = await api.fetchApi("/grimmribbity/detailer_presets");
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data.presets) ? data.presets : [];
  } catch (_e) {
    return [];
  }
}

async function _fetchDetailerPreset(id) {
  if (!id) return null;
  try {
    const res = await api.fetchApi(
      `/grimmribbity/detailer_presets/${encodeURIComponent(id)}`);
    if (!res.ok) return null;
    const data = await res.json();
    return data.preset || null;
  } catch (_e) {
    return null;
  }
}

async function _saveDetailerPreset(name, settings, { id = "", description = "" } = {}) {
  const body = JSON.stringify({ name, settings, description, id });
  const res = await api.fetchApi("/grimmribbity/detailer_presets",
    { method: "POST", body, headers: { "Content-Type": "application/json" } });
  if (!res.ok) {
    let msg = `save failed: HTTP ${res.status}`;
    try { msg = (await res.json()).error || msg; } catch (_e) {}
    throw new Error(msg);
  }
  return (await res.json()).preset;
}

async function _deleteDetailerPreset(id) {
  if (!id) return false;
  const res = await api.fetchApi(
    `/grimmribbity/detailer_presets/${encodeURIComponent(id)}/delete`,
    { method: "POST" });
  return res.ok;
}

function _buildDetailerPresetToolbar(node, refreshGridCb) {
  // Returns the toolbar DOM. Lives at the top of the SmartDetailer grid
  // widget so the picker is visible alongside the per-target overrides.
  const bar = document.createElement("div");
  bar.className = "pl-det-presets";

  const label = document.createElement("span");
  label.className = "pl-det-presets-label";
  label.textContent = "Preset:";
  bar.appendChild(label);

  const select = document.createElement("select");
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "(none)";
  select.appendChild(placeholder);
  bar.appendChild(select);

  const applyBtn = document.createElement("button");
  applyBtn.type = "button";
  applyBtn.textContent = "📋 Apply";
  applyBtn.title = "Overwrite every detailer widget value with the selected preset.";
  bar.appendChild(applyBtn);

  const saveBtn = document.createElement("button");
  saveBtn.type = "button";
  saveBtn.textContent = "💾 Save";
  saveBtn.title = "Save current detailer settings as a new preset (or update the selected one).";
  bar.appendChild(saveBtn);

  const delBtn = document.createElement("button");
  delBtn.type = "button";
  delBtn.textContent = "🗑";
  delBtn.className = "danger";
  delBtn.title = "Delete the selected preset.";
  bar.appendChild(delBtn);

  const status = document.createElement("div");
  status.className = "pl-det-preset-status";
  status.textContent = "";
  bar.appendChild(status);

  // Inline save form — hidden until user clicks Save and we need a name.
  const saveForm = document.createElement("div");
  saveForm.className = "pl-det-preset-save-form";
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.placeholder = "preset name…";
  nameInput.maxLength = 120;
  const confirmBtn = document.createElement("button");
  confirmBtn.type = "button";
  confirmBtn.textContent = "Save";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.textContent = "Cancel";
  saveForm.appendChild(nameInput);
  saveForm.appendChild(confirmBtn);
  saveForm.appendChild(cancelBtn);
  bar.appendChild(saveForm);

  // Index of preset metadata by id so the picker can describe + delete.
  let cache = [];

  const setStatus = (text, isError = false) => {
    status.textContent = text || "";
    status.style.color = isError ? "#ff8a8a" : "";
  };

  const repopulateSelect = (selectedId = "") => {
    // Wipe and rebuild option list, then restore selection if id still present.
    while (select.options.length > 1) select.remove(1);
    for (const p of cache) {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.builtin ? `★ ${p.name}` : p.name;
      if (p.description) opt.title = p.description;
      select.appendChild(opt);
    }
    const keep = selectedId && cache.some(p => p.id === selectedId)
      ? selectedId : "";
    select.value = keep;
    node.properties = node.properties || {};
    node.properties.detailerPresetId = keep;
    updateButtonState();
  };

  const updateButtonState = () => {
    const id = select.value;
    applyBtn.disabled = !id;
    delBtn.disabled = !id;
    const picked = cache.find(p => p.id === id);
    if (picked) {
      const tag = picked.builtin ? "builtin" : "user";
      setStatus(`Selected: ${picked.name} (${tag})`);
    } else {
      setStatus("");
    }
  };

  const openSaveForm = (preName = "") => {
    saveForm.classList.add("open");
    nameInput.value = preName;
    nameInput.focus();
    nameInput.select();
  };
  const closeSaveForm = () => {
    saveForm.classList.remove("open");
    nameInput.value = "";
  };

  const reload = async (selectedId = "") => {
    cache = await _fetchDetailerPresets();
    const wanted = selectedId || (node.properties && node.properties.detailerPresetId) || "";
    repopulateSelect(wanted);
  };

  select.addEventListener("change", updateButtonState);

  applyBtn.addEventListener("click", async () => {
    const id = select.value;
    if (!id) return;
    applyBtn.disabled = true;
    setStatus("Loading preset…");
    const preset = await _fetchDetailerPreset(id);
    applyBtn.disabled = false;
    if (!preset) { setStatus("Preset missing — reload?", true); return; }
    const n = _applyDetailerSettings(node, preset.settings || {});
    refreshGridCb?.();
    node.properties = node.properties || {};
    node.properties.detailerPresetId = preset.id;
    setStatus(`Applied "${preset.name}" (${n} fields)`);
  });

  saveBtn.addEventListener("click", () => {
    const id = select.value;
    const picked = cache.find(p => p.id === id);
    // If a non-builtin user preset is selected, default to "update in place"
    // by prefilling the name; otherwise prompt for a new name.
    const defaultName = picked && !picked.builtin ? picked.name : "";
    openSaveForm(defaultName);
  });

  confirmBtn.addEventListener("click", async () => {
    const rawName = (nameInput.value || "").trim();
    if (!rawName) { nameInput.focus(); return; }
    const id = select.value;
    const picked = cache.find(p => p.id === id);
    // Update-in-place only when the user kept the SAME non-builtin preset name.
    // Builtin or renamed → create new entry (backend dedupes by name too).
    const sameName = picked && !picked.builtin
      && rawName.toLowerCase() === picked.name.toLowerCase();
    const targetId = sameName ? picked.id : "";
    confirmBtn.disabled = true;
    setStatus("Saving…");
    try {
      const settings = _snapshotDetailerSettings(node);
      const saved = await _saveDetailerPreset(rawName, settings, { id: targetId });
      closeSaveForm();
      await reload(saved.id);
      setStatus(`Saved "${saved.name}"`);
    } catch (e) {
      setStatus(String(e.message || e), true);
    } finally {
      confirmBtn.disabled = false;
    }
  });
  cancelBtn.addEventListener("click", closeSaveForm);

  nameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); confirmBtn.click(); }
    else if (e.key === "Escape") { e.preventDefault(); closeSaveForm(); }
  });

  delBtn.addEventListener("click", async () => {
    const id = select.value;
    if (!id) return;
    const picked = cache.find(p => p.id === id);
    const label = picked ? picked.name : "this preset";
    // Single-click confirm: button text doubles as the prompt for 3s.
    if (delBtn.dataset.armed !== "yes") {
      delBtn.dataset.armed = "yes";
      const old = delBtn.textContent;
      delBtn.textContent = `Delete "${label}"?`;
      const timeout = setTimeout(() => {
        delBtn.dataset.armed = "";
        delBtn.textContent = old;
      }, 3000);
      delBtn._disarmTimer = timeout;
      return;
    }
    clearTimeout(delBtn._disarmTimer);
    delBtn.dataset.armed = "";
    delBtn.textContent = "🗑";
    setStatus("Deleting…");
    const ok = await _deleteDetailerPreset(id);
    if (!ok) { setStatus("Delete failed", true); return; }
    await reload("");
    setStatus(`Deleted "${label}"`);
  });

  // Initial population happens async; expose a reload hook so onConfigure
  // (workflow reload) can re-sync the picker against the saved properties.
  reload();

  return { bar, reload };
}

function _buildSmartDetailerGrid(node) {
  const TARGETS = ["face", "eyes", "hands", "feet"];
  // sentinel: the "use global / preset value" placeholder. Cells holding
  // the sentinel render dimmed-italic so override status is visible at a
  // glance. -1 for the global-fallback floats; 0 for max/steps; 0 for
  // crop_factor (uses preset); 1 for cycles (default is one pass).
  // Booleans don't dim.
  const ROWS = [
    { key: "enable",    label: "enable",    pat: "enable_X",      kind: "bool" },
    { key: "threshold", label: "threshold", pat: "X_threshold",   kind: "float", step: 0.01, min: -1, max: 1, sentinel: -1 },
    { key: "denoise",   label: "denoise",   pat: "X_denoise",     kind: "float", step: 0.01, min: -1, max: 1, sentinel: -1 },
    { key: "max",       label: "max N",     pat: "X_max",         kind: "int",   step: 1, min: 0, max: 64,    sentinel: 0  },
    { key: "steps",     label: "steps",     pat: "X_steps",       kind: "int",   step: 1, min: 0, max: 200,   sentinel: 0  },
    { key: "crop",      label: "crop",      pat: "X_crop_factor", kind: "float", step: 0.05, min: 0, max: 10, sentinel: 0  },
    { key: "cycles",    label: "cycles",    pat: "X_cycles",      kind: "int",   step: 1, min: 1, max: 8,     sentinel: 1  },
  ];
  const ROW_TOOLTIPS = {
    enable:    "Run the detailer pass on this category. Unchecked = column is skipped entirely.",
    threshold: "YOLO detection confidence (0–1). Lower = catch more candidates, higher = stricter. Set to −1 to inherit the global default.",
    denoise:   "Inpaint denoising strength (0–1). Higher = more aggressive redrawing of detected regions. Set to −1 to inherit the global default.",
    max:       "Max detections to process per image, ranked top-down by confidence. 0 = use the preset default.",
    steps:     "Sampler steps for the inpaint pass on this category. 0 = use the preset default.",
    crop:      "Crop factor around each detection before sampling. 1.0 = tight, 2.0 = generous context. 0 = use the preset default.",
    cycles:    "How many times to re-run this pass on the same detections. 2-3 cycles compound detail when one pass barely moves the result. Linear time cost.",
  };

  const widgetsByName = {};
  for (const w of node.widgets || []) widgetsByName[w.name] = w;

  const grid = document.createElement("div");
  grid.className = "pl-det-grid";

  // Preset toolbar at the top: dropdown + Apply/Save/Delete. refreshGridCb
  // is wired below once `cells` is defined; the closure threads through the
  // value returned from this function.
  const presetToolbarHandle = { reload: null };
  const presetBar = _buildDetailerPresetToolbar(node, () => {
    // After applying a preset, every cell's input needs to reflect the new
    // widget value. Forward to the per-cell refresh hook collected below.
    presetToolbarHandle.refresh?.();
  });
  grid.appendChild(presetBar.bar);
  presetToolbarHandle.reload = presetBar.reload;

  // Header row: empty corner + 6 category labels in their own row container.
  const headerRow = document.createElement("div");
  headerRow.className = "pl-det-row pl-det-row-header";
  const corner = document.createElement("div");
  corner.className = "pl-det-corner";
  headerRow.appendChild(corner);
  for (const t of TARGETS) {
    const h = document.createElement("div");
    h.className = `pl-det-h ${t}`;
    h.textContent = t;
    headerRow.appendChild(h);
  }
  grid.appendChild(headerRow);

  // Track every input for onConfigure resync — workflow loads call setValue
  // on the underlying widgets; we need to refresh the grid from those values.
  const cells = [];

  for (let rowIdx = 0; rowIdx < ROWS.length; rowIdx++) {
    const row = ROWS[rowIdx];
    const tooltip = ROW_TOOLTIPS[row.key] || "";

    const dataRow = document.createElement("div");
    dataRow.className = "pl-det-row pl-det-row-data";

    const lbl = document.createElement("div");
    lbl.className = "pl-det-l";
    lbl.textContent = row.label;
    _plAttachTip(lbl, tooltip);
    dataRow.appendChild(lbl);

    for (const t of TARGETS) {
      const wname = row.pat.replace("X", t);
      const w = widgetsByName[wname];
      const cell = document.createElement("div");
      cell.className = "pl-det-cell";
      _plAttachTip(cell, tooltip);
      if (!w) {
        // Missing widget — empty cell wrapper keeps the column aligned.
        dataRow.appendChild(cell);
        continue;
      }

      // Hide the underlying widget but keep its value live for serialisation.
      w.hidden = true;
      w.computeSize = () => [0, -4];
      w.draw = () => {};

      const input = document.createElement("input");
      if (row.kind === "bool") {
        input.type = "checkbox";
        input.className = "pl-det-toggle";
        input.checked = !!w.value;
        input.addEventListener("change", () => {
          w.value = input.checked;
          node.setDirtyCanvas?.(true, true);
        });
      } else {
        input.type = "number";
        input.className = "pl-det-i";
        input.step = String(row.step);
        if (typeof row.min === "number") input.min = String(row.min);
        if (typeof row.max === "number") input.max = String(row.max);
        const isSentinel = (v) => {
          if (typeof v !== "number") return false;
          // Floats with sentinel=-1 dim when negative (covers users who
          // type values < 0 even though they shouldn't); ints / floats
          // with sentinel=0 dim only when exactly 0.
          if (row.sentinel < 0) return v < 0;
          return v === 0;
        };
        const writeVal = () => {
          const raw = input.value;
          if (raw === "" || raw === null) return;
          const v = row.kind === "int" ? parseInt(raw, 10) : parseFloat(raw);
          if (!isNaN(v)) {
            w.value = v;
            node.setDirtyCanvas?.(true, true);
            input.classList.toggle("dim", isSentinel(v));
          }
        };
        const refresh = () => {
          const v = w.value;
          input.value = (v === undefined || v === null) ? "" : String(v);
          input.classList.toggle("dim", isSentinel(v));
        };
        refresh();
        input.addEventListener("change", writeVal);
        input.addEventListener("blur", writeVal);
        cells.push({ input, refresh });
      }
      cell.appendChild(input);
      dataRow.appendChild(cell);
    }
    grid.appendChild(dataRow);
  }

  // Boolean toggles also need a refresh hook for onConfigure.
  for (const w of node.widgets || []) {
    if (!w.name) continue;
    if (!w.name.startsWith("enable_")) continue;
    const target = w.name.slice("enable_".length);
    if (!TARGETS.includes(target)) continue;
    // Find the matching checkbox in the grid and add a refresher.
    const idx = TARGETS.indexOf(target);
    const checkbox = grid.querySelectorAll(".pl-det-toggle")[idx];
    if (checkbox) cells.push({
      input: checkbox,
      refresh: () => { checkbox.checked = !!w.value; },
    });
  }

  const refresh = () => cells.forEach(c => c.refresh());
  presetToolbarHandle.refresh = refresh;
  return { grid, refresh, reloadPresets: presetToolbarHandle.reload };
}

const CIVITAI_SAVE_NAME = "GrimmRibbityCivitaiSave";

async function _civitaiBrowseDirs(path) {
  const url = `/grimmribbity/browse_dirs?path=${encodeURIComponent(path || "")}`;
  try {
    const res = await api.fetchApi(url);
    if (!res.ok) return { error: `HTTP ${res.status}`, dirs: [], roots: [] };
    return await res.json();
  } catch (e) {
    return { error: String(e.message || e), dirs: [], roots: [] };
  }
}

async function _civitaiMkdir(parent, name) {
  const body = JSON.stringify({ parent, name });
  const res = await api.fetchApi("/grimmribbity/mkdir", {
    method: "POST", body,
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try { msg = (await res.json()).error || msg; } catch (_e) {}
    throw new Error(msg);
  }
  return (await res.json()).path;
}

function _openCivitaiFolderBrowser(initialPath, onPick) {
  // Build modal once, populate from the backend, return chosen path through
  // onPick callback. Modal closes on Esc, Cancel, or successful pick.
  const overlay = document.createElement("div");
  overlay.className = "pl-browser-overlay";

  const modal = document.createElement("div");
  modal.className = "pl-browser-modal";
  overlay.appendChild(modal);

  const header = document.createElement("header");
  const title = document.createElement("h3");
  title.textContent = "Pick a save folder";
  header.appendChild(title);

  const roots = document.createElement("div");
  roots.className = "pl-browser-roots";
  header.appendChild(roots);

  const pathRow = document.createElement("div");
  pathRow.className = "pl-browser-pathrow";
  const pathInput = document.createElement("input");
  pathInput.type = "text";
  pathInput.spellcheck = false;
  pathInput.value = initialPath || "";
  pathInput.placeholder = "type or paste a path…";
  const goBtn = document.createElement("button");
  goBtn.type = "button";
  goBtn.textContent = "Go";
  pathRow.appendChild(pathInput);
  pathRow.appendChild(goBtn);
  header.appendChild(pathRow);

  modal.appendChild(header);

  const list = document.createElement("div");
  list.className = "pl-browser-list";
  modal.appendChild(list);

  const status = document.createElement("div");
  status.className = "pl-browser-status";
  modal.appendChild(status);

  const footer = document.createElement("footer");
  const mkdirGroup = document.createElement("div");
  mkdirGroup.className = "pl-browser-mkdir";
  const mkdirInput = document.createElement("input");
  mkdirInput.type = "text";
  mkdirInput.placeholder = "new folder name…";
  mkdirInput.maxLength = 80;
  const mkdirBtn = document.createElement("button");
  mkdirBtn.type = "button";
  mkdirBtn.textContent = "📂 Create";
  mkdirGroup.appendChild(mkdirInput);
  mkdirGroup.appendChild(mkdirBtn);
  footer.appendChild(mkdirGroup);

  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.textContent = "Cancel";
  const pickBtn = document.createElement("button");
  pickBtn.type = "button";
  pickBtn.className = "primary";
  pickBtn.textContent = "✅ Use this folder";
  footer.appendChild(cancelBtn);
  footer.appendChild(pickBtn);
  modal.appendChild(footer);

  let currentPath = initialPath || "";

  const setStatus = (msg, isError = false) => {
    status.textContent = msg || "";
    status.classList.toggle("error", !!isError);
  };

  const navigate = async (target) => {
    setStatus("Loading…");
    const data = await _civitaiBrowseDirs(target);
    currentPath = data.path || target || "";
    pathInput.value = currentPath;
    // Repopulate roots quick-jumps every time so they're always available.
    // Using replaceChildren() instead of innerHTML="" keeps the security
    // hook quiet — we never assign untrusted HTML, just clear children.
    roots.replaceChildren();
    for (const r of (data.roots || [])) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = r.label;
      btn.addEventListener("click", () => navigate(r.path));
      roots.appendChild(btn);
    }
    // Render dir entries. Parent first (when present), then alpha subdirs.
    list.replaceChildren();
    if (data.parent && data.parent !== currentPath) {
      const parentRow = document.createElement("div");
      parentRow.className = "pl-browser-row parent";
      const ic = document.createElement("span");
      ic.className = "icon"; ic.textContent = "🆙";
      const lbl = document.createElement("span");
      lbl.textContent = ".. (parent)";
      parentRow.appendChild(ic);
      parentRow.appendChild(lbl);
      parentRow.addEventListener("click", () => navigate(data.parent));
      list.appendChild(parentRow);
    }
    for (const d of (data.dirs || [])) {
      const row = document.createElement("div");
      row.className = "pl-browser-row";
      const ic = document.createElement("span");
      ic.className = "icon"; ic.textContent = "📁";
      const lbl = document.createElement("span");
      lbl.textContent = d.name;
      row.appendChild(ic);
      row.appendChild(lbl);
      row.addEventListener("dblclick", () => {
        // Double-click descends AND picks — common power-user shortcut.
        navigate(d.path);
      });
      row.addEventListener("click", () => navigate(d.path));
      list.appendChild(row);
    }
    if (data.error) {
      setStatus(data.error, true);
    } else {
      setStatus(`${(data.dirs || []).length} subfolder(s)`);
    }
  };

  const close = () => {
    document.removeEventListener("keydown", onKey);
    overlay.remove();
  };

  const onKey = (e) => {
    if (e.key === "Escape") { e.preventDefault(); close(); }
    else if (e.key === "Enter" && document.activeElement === pathInput) {
      e.preventDefault(); navigate(pathInput.value);
    }
  };

  goBtn.addEventListener("click", () => navigate(pathInput.value));
  cancelBtn.addEventListener("click", close);
  pickBtn.addEventListener("click", () => {
    const chosen = pathInput.value || currentPath;
    close();
    onPick?.(chosen);
  });
  overlay.addEventListener("click", (e) => {
    // Click on the backdrop (not the modal) closes.
    if (e.target === overlay) close();
  });

  mkdirBtn.addEventListener("click", async () => {
    const name = (mkdirInput.value || "").trim();
    if (!name) { mkdirInput.focus(); return; }
    mkdirBtn.disabled = true;
    setStatus(`Creating "${name}"…`);
    try {
      const fresh = await _civitaiMkdir(currentPath, name);
      mkdirInput.value = "";
      await navigate(fresh);
      setStatus(`Created ${fresh}`);
    } catch (e) {
      setStatus(String(e.message || e), true);
    } finally {
      mkdirBtn.disabled = false;
    }
  });
  mkdirInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); mkdirBtn.click(); }
  });

  document.body.appendChild(overlay);
  document.addEventListener("keydown", onKey);
  navigate(initialPath);
}

function registerCivitaiSaveNode(nodeType) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated?.apply(this, arguments);
    const node = this;
    const pathWidget = (node.widgets || []).find(w => w.name === "output_path");
    if (!pathWidget) return r;

    // Inline bar under (after, in widget order) the existing string widget.
    // We keep the underlying STRING widget editable so power users can paste
    // / type / use date-substitution tokens directly. The DOM widget just
    // adds a Browse button and a clearer "current path" view.
    const bar = document.createElement("div");
    bar.className = "pl-civ-pathbar";

    const label = document.createElement("span");
    label.textContent = "📁";
    bar.appendChild(label);

    const current = document.createElement("span");
    current.className = "pl-civ-current";
    bar.appendChild(current);

    const browseBtn = document.createElement("button");
    browseBtn.type = "button";
    browseBtn.textContent = "Browse…";
    browseBtn.title = "Open the folder picker to choose a save directory.";
    bar.appendChild(browseBtn);

    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.textContent = "↺";
    clearBtn.title = "Reset to ComfyUI/output (default).";
    bar.appendChild(clearBtn);

    const refreshCurrent = () => {
      const v = (pathWidget.value || "").trim();
      current.textContent = v ? v : "(default: ComfyUI/output)";
      current.title = v || "ComfyUI/output";
    };
    refreshCurrent();

    browseBtn.addEventListener("click", () => {
      _openCivitaiFolderBrowser(pathWidget.value || "", (chosen) => {
        if (!chosen) return;
        pathWidget.value = chosen;
        refreshCurrent();
        node.setDirtyCanvas?.(true, true);
      });
    });
    clearBtn.addEventListener("click", () => {
      pathWidget.value = "";
      refreshCurrent();
      node.setDirtyCanvas?.(true, true);
    });

    node.addDOMWidget("output_path_browser", "GrimmRibbityCivitaiFolderBar",
                      bar, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => 32,
      getValue: () => "",
      setValue: () => {},
    });

    node._civitaiRefreshPath = refreshCurrent;
    return r;
  };

  const onConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const ret = onConfigure?.apply(this, arguments);
    // Workflow load just rehydrated output_path on the underlying STRING
    // widget; reflect that in the bar's path display.
    this._civitaiRefreshPath?.();
    return ret;
  };
}


function registerSmartDetailerNode(nodeType) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated?.apply(this, arguments);
    const built = _buildSmartDetailerGrid(this);
    this.addDOMWidget("per_target_grid", "GrimmRibbitySmartDetailerGrid",
                       built.grid, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => 290,
      getValue: () => "",
      setValue: () => {},
    });
    this._smartDetailerRefresh = built.refresh;
    this._smartDetailerReloadPresets = built.reloadPresets;
    // Bump default node width so the 7 grid columns (label + 6 targets) fit.
    if (Array.isArray(this.size) && this.size[0] < 480) {
      this.size = [480, this.size[1] || 720];
      this.setSize?.(this.size);
    }
    return r;
  };

  const onConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const ret = onConfigure?.apply(this, arguments);
    // Workflow load: underlying widgets just got their saved values; refresh
    // every cell input so the grid reflects them.
    this._smartDetailerRefresh?.();
    // The picker re-selects the saved preset id from node.properties so the
    // "currently active" indicator survives reload.
    this._smartDetailerReloadPresets?.();
    return ret;
  };
}


let _wsListenerInstalled = false;
function installWebsocketBridge() {
  if (_wsListenerInstalled) return;
  _wsListenerInstalled = true;
  api.addEventListener("prompt_library.updated", () => {
    _imageCacheKey = Date.now();  // bust thumbnail cache on any change
    window.dispatchEvent(new CustomEvent("prompt-library-updated"));
  });
}

// Per-node-group color theming: deep saturated title bar with bold black
// title text + white drop shadow + uniform dark-grey body across the suite.
// Wraps onNodeCreated AND onConfigure — the latter is needed because
// LGraphNode.configure() restores `color`/`bgcolor` from the saved workflow
// JSON after onNodeCreated runs, undoing our theme.
const NODE_BODY_COLOR = "#1e1e1e";
const NODE_TITLE_TEXT_COLOR = "#0a0a0a";
const NODE_COLORS = {
  // Library / data nodes — saturated purple
  "PromptLibrary":              "#a060e0",
  "PromptLibraryMulti":         "#a060e0",
  "PromptLibrarySave":          "#a060e0",
  "PromptLibraryThumbnailSaver":"#a060e0",
  "PromptLibraryRandom":        "#a060e0",
  "PromptLibraryWildcard":      "#a060e0",
  // Comic authoring — burnt orange
  "PromptLibraryScene":      "#e8852f",
  "PromptLibraryBackground": "#e8852f",
  "PromptLibraryComicFrame": "#e8852f",
  "PromptLibraryComicFrameEncode": "#e8852f",
  // Sampling — deep teal
  "GrimmRibbitySamplerSDXL":    "#34a4c8",
  "GrimmRibbityHiResFixScript": "#34a4c8",
  "GrimmRibbityPackSDXLTuple":  "#34a4c8",
  // Output / save — saturated emerald
  "GrimmRibbityCivitaiSave": "#3eba6c",
  // LoRA picker — magenta / hot pink so it's distinct from the data nodes
  "GrimmRibbityLoraPicker": "#d84ba8",
  // Anima sampler / HiResFix — rose / mauve to set apart from teal SDXL
  "GrimmRibbityAnimaSampler": "#c84a7a",
  "GrimmRibbityAnimaHiResFixScript": "#c84a7a",
  // Detailer — burnt orange, distinct from Scene/Background's brighter orange
  "GrimmRibbitySmartDetailer": "#d8754a",
  // Character anchor — olive / muted gold for the IPAdapter wrap
  "GrimmRibbityCharacterAnchor": "#8eaa3e",
  // Upscaler — darker teal sibling of the SDXL sampler family
  "GrimmRibbityUpscaleSDXL": "#2c8aa4",
};
// Colors used by previous theme revisions. When a saved workflow loads with
// one of these stuck on a node, we treat it as stale and replace with the
// current theme. Manual user colours (anything not in this list) are
// preserved on reload.
const STALE_THEME_COLORS = new Set([
  // v0.22.1 (dark theme)
  "#6b4a8c", "#3d2752",
  "#b07a3a", "#5d3e1c",
  "#3a7c8c", "#1f3d52",
  "#3a8c5b", "#1f4d34",
  // v0.22.2 (pastel bars)
  "#c8a8e8", "#e8b878", "#8cc8d8", "#a8d8b8",
  // current bars (auto-refresh on reload if values changed)
  "#a060e0", "#e8852f", "#34a4c8", "#3eba6c", "#d84ba8", "#c84a7a",
  // shared body
  "#1e1e1e",
]);
function _isReplaceable(value, defaultValue) {
  if (!value) return true;
  if (value === defaultValue) return true;
  if (typeof value === "string" && STALE_THEME_COLORS.has(value.toLowerCase())) return true;
  return false;
}
// Earlier theme revisions hooked onDrawTitleText to paint a bold title
// with a white drop-shadow halo. Some ComfyUI builds render the title
// via Vue/HTML while ALSO firing the canvas hook, which produced a
// doubled / ghosted title (visible on screenshots from a friend's
// install). The styling is a nice-to-have; visible breakage isn't.
// We keep `title_text_color` (which the default renderer respects) and
// drop the canvas override entirely.

function applyNodeColors(nodeType, nodeData) {
  const titleColor = NODE_COLORS[nodeData.name];
  if (!titleColor) return;

  const setColors = (node) => {
    if (_isReplaceable(node.color, LiteGraph?.NODE_DEFAULT_COLOR)) {
      node.color = titleColor;
    }
    if (_isReplaceable(node.bgcolor, LiteGraph?.NODE_DEFAULT_BGCOLOR)) {
      node.bgcolor = NODE_BODY_COLOR;
    }
    node.title_text_color = NODE_TITLE_TEXT_COLOR;
  };

  const origCreate = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const r = origCreate?.apply(this, arguments);
    setColors(this);
    return r;
  };

  // LGraphNode.configure() restores saved `color`/`bgcolor` AFTER
  // onNodeCreated has run — re-apply here so the theme survives reload.
  const origConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const r = origConfigure?.apply(this, arguments);
    setColors(this);
    return r;
  };

  // Class-level fallback for the title text color. We deliberately do NOT
  // install onDrawTitleText — see comment above _drawTitleText removal.
  nodeType.title_text_color = NODE_TITLE_TEXT_COLOR;
}

app.registerExtension({
  name: "comfy.PromptLibrary",
  async setup() {
    installWebsocketBridge();
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    // Apply colors to every node we own, regardless of which branch below
    // handles its widget setup. Safe to call before the dispatch — the
    // wrapped onNodeCreated chains correctly with the JS-side widget builders.
    applyNodeColors(nodeType, nodeData);

    if (nodeData.name === MULTI_NODE_NAME) {
      injectStyle();
      registerMultiNode(nodeType);
      return;
    }
    if (nodeData.name === COMIC_FRAME_NODE_NAME) {
      injectStyle();
      registerComicFrameNode(nodeType);
      return;
    }
    if (nodeData.name === BACKGROUND_NODE_NAME) {
      injectStyle();
      registerBackgroundNode(nodeType);
      return;
    }
    if (nodeData.name === SMART_DETAILER_NAME) {
      injectStyle();
      registerSmartDetailerNode(nodeType);
      return;
    }
    if (nodeData.name === CIVITAI_SAVE_NAME) {
      injectStyle();
      registerCivitaiSaveNode(nodeType);
      return;
    }
    if (!GALLERY_NODE_NAMES.has(nodeData.name)) return;
    injectStyle();

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);

      const idWidget = this.widgets.find(w => w.name === "prompt_id");
      if (idWidget) {
        // Hide the underlying string widget; gallery clicks write to its value,
        // and ComfyUI auto-serializes it into the workflow JSON. We hide via
        // three mechanisms to cover both the legacy LiteGraph renderer and
        // the new Nodes 2.0 renderer:
        //   - widget.hidden = true       → Nodes 2.0 (Vue-rendered) skips it
        //   - computeSize → [0, -4]      → legacy renderer collapses the row
        //   - draw = noop                → legacy renderer extra safety
        // Do NOT mutate widget.type to a custom string — Nodes 2.0's typed
        // renderer treats unknown types as broken and bails on the whole node.
        idWidget.hidden = true;
        idWidget.computeSize = () => [0, -4];
        idWidget.draw = () => {};
      }

      const { container, render, syncFromWidget, cleanup } = buildGallery(this, idWidget);
      // Defensive: container needs explicit dimensions because Nodes 2.0
      // doesn't always give DOM widgets a sized wrapper before first paint.
      container.style.minHeight = "240px";
      container.style.width = "100%";

      const galleryWidget = this.addDOMWidget("gallery", "PromptLibraryGallery", container, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 240,
        // Nodes 2.0 may call getValue/setValue during reactivity sync; without
        // these the widget can be treated as malformed and skipped.
        getValue: () => idWidget?.value || "",
        setValue: (v) => {
          if (window._plDebugSelect) {
            console.log(`[PL select] node#${this.id ?? "?"} galleryWidget.setValue(${JSON.stringify(v)})`);
          }
          if (idWidget) idWidget.value = v;
          syncFromWidget?.("galleryWidget.setValue");
          render?.();
        },
      });
      this._promptLibraryRender = render;
      this._promptLibrarySyncFromWidget = syncFromWidget;
      this._promptLibraryCleanup = cleanup;
      this._promptLibraryGalleryWidget = galleryWidget;

      // The Style node has more sockets (MODEL/CLIP in+out, CONDITIONING in+out,
      // STRING out) and an extra_text textarea widget — give it more vertical
      // room so the gallery doesn't end up squashed.
      const initialSize = nodeData.name === STYLE_NODE_NAME ? [340, 480] : [320, 320];
      this.size = initialSize;
      if (typeof this.setSize === "function") this.setSize(initialSize);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      // Re-render so the tile matching the workflow's saved prompt_id gets highlighted.
      this._promptLibrarySyncFromWidget?.("onConfigure");
      this._promptLibraryRender?.();
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this._promptLibraryCleanup?.();
      // Null out every callback ref so the deleted node object doesn't keep
      // the gallery DOM / closure alive. Comfy's undo stack can resurrect
      // a removed node, but onConfigure rebuilds the gallery from scratch
      // (onNodeCreated runs again on resurrect), so dropping the old refs
      // here is safe and prevents stale state from leaking into the new
      // gallery instance.
      this._promptLibraryCleanup = null;
      this._promptLibraryRender = null;
      this._promptLibrarySyncFromWidget = null;
      this._promptLibraryGalleryWidget = null;
      return onRemoved?.apply(this, arguments);
    };
  },
});
