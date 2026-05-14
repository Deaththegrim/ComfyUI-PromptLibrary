"""Shared runtime helpers for the GrimmRibbity custom-node pack.

Two concerns live here:

1. HIP-only synchronize helper for the KSampler↔VAE boundaries that
   previously wedged Impact Pack's enhance_detail (2026-05-05 MIOpen+
   allocator fingerprint). CUDA proper takes the no-op path.

2. Cross-module cache registry + lazy install of a wrapper around
   comfy.model_management.unload_all_models, so each module can register
   its own module-level caches and have them all dropped on the same
   trigger ("Unload Models" button, /free route, auto-eviction).

The hook installer is intentionally lazy: at custom-node load time
comfy.model_management may not yet be importable, so each consumer calls
ensure_unload_hook() from its first hot-path entry.
"""
from __future__ import annotations

import logging
from typing import Callable

import torch


_IS_HIP = bool(getattr(torch.version, "hip", None))


def hip_sync(where: str) -> None:
    """torch.cuda.synchronize() on HIP only; no-op on CUDA proper.
    `where` is a short tag for the debug log so we can correlate sync
    fires with pipeline boundaries when troubleshooting wedges."""
    if not _IS_HIP:
        return
    try:
        torch.cuda.synchronize()
        logging.debug("[GrimmRibbity] sync (%s)", where)
    except Exception:
        pass


_CACHE_CLEARERS: list[Callable[[], None]] = []


def register_cache(clearer: Callable[[], None]) -> None:
    """Register a zero-arg callable that empties some module-level cache.
    Called at module-import time; the actual eviction fires when Comfy
    runs unload_all_models(). Use a lambda when the cache identifier
    isn't yet bound at registration time."""
    _CACHE_CLEARERS.append(clearer)


def _evict_all() -> None:
    for clearer in _CACHE_CLEARERS:
        try:
            clearer()
        except Exception as exc:
            logging.warning("[GrimmRibbity] cache clearer failed (%s)", exc)


_UNLOAD_HOOK_INSTALLED = False


def ensure_unload_hook() -> None:
    """Idempotently wrap comfy.model_management.unload_all_models so every
    registered cache clearer fires before the original. Silent when
    comfy isn't importable (test environments)."""
    global _UNLOAD_HOOK_INSTALLED
    if _UNLOAD_HOOK_INSTALLED:
        return
    try:
        import comfy.model_management as _mm
    except Exception:
        return
    original = getattr(_mm, "unload_all_models", None)
    if original is None:
        return
    if getattr(original, "_grimmribbity_wrapped", False):
        _UNLOAD_HOOK_INSTALLED = True
        return

    def wrapped(*args, **kwargs):
        _evict_all()
        return original(*args, **kwargs)

    wrapped._grimmribbity_wrapped = True  # type: ignore[attr-defined]
    _mm.unload_all_models = wrapped
    _UNLOAD_HOOK_INSTALLED = True


# Auto-tile threshold in *latent* pixels. Above this, decode/encode falls
# back to the tiled path even when the user picked "true". A 192-latent
# image is 1536px decoded — non-tiled decode of larger latents reliably
# OOMs on 16GB AMD VRAM under TTM placement pressure.
AUTO_TILE_LATENT_THRESHOLD = 192


def _soft_empty_cache() -> None:
    try:
        import comfy.model_management as _mm
        _mm.soft_empty_cache()
    except (ImportError, ModuleNotFoundError):
        pass


def _raise_non_oom(exc: BaseException) -> None:
    """Re-raise `exc` unless it's an OOM. Lets the caller's shrink-loop
    only catch genuine OOMs and let other exceptions propagate."""
    try:
        import comfy.model_management as _mm
        _mm.raise_non_oom(exc)
    except (ImportError, ModuleNotFoundError):
        raise exc


def oom_safe_vae_decode(vae, latent, *, mode: str = "true",
                         tile: int = 256, overlap: int = 64):
    """Decode a latent with consistent tile policy + OOM shrink loop.

    `mode`:
      - "false"        → returns None (caller emits placeholder).
      - "true"         → auto-tile when latent's longest side exceeds
                         AUTO_TILE_LATENT_THRESHOLD, else non-tiled.
      - "true (tiled)" → force tiled decode.

    On tiled OOM, walks tile 256→128→64 then falls through to vae.decode().
    """
    if mode == "false" or vae is None or latent is None:
        return None
    samples = latent["samples"] if isinstance(latent, dict) else latent
    latent_max = max(samples.shape[-1], samples.shape[-2])
    needs_tile = (mode == "true (tiled)" or
                  (mode == "true" and latent_max > AUTO_TILE_LATENT_THRESHOLD))
    if not (needs_tile and hasattr(vae, "decode_tiled")):
        return vae.decode(samples)
    while True:
        # comfy's decode_tiled_ internally uses tile_x // 2 for the steps calc,
        # so overlap must stay strictly < tile // 2 or we hit div-by-zero.
        eff_overlap = min(overlap, max(8, tile // 4))
        try:
            try:
                return vae.decode_tiled(samples, tile_x=tile, tile_y=tile, overlap=eff_overlap)
            except TypeError:
                return vae.decode_tiled(samples, tile_x=tile, tile_y=tile)
        except Exception as exc:
            _raise_non_oom(exc)
            if tile <= 64:
                logging.info("[GrimmRibbity] vae decode OOM at tile=%d, "
                              "falling through to vae.decode()", tile)
                _soft_empty_cache()
                return vae.decode(samples)
            tile //= 2
            logging.info("[GrimmRibbity] vae decode OOM, retrying at tile=%d", tile)
            _soft_empty_cache()


def oom_safe_vae_encode(vae, image, *, mode: str = "true",
                         tile: int = 256, overlap: int = 64) -> dict:
    """Encode an HWC image with consistent tile policy + OOM shrink loop.
    Returns {"samples": tensor}.

    `mode`:
      - "true"         → auto-tile when image's longest pixel side exceeds
                         AUTO_TILE_LATENT_THRESHOLD * 8, else non-tiled.
      - "true (tiled)" → force tiled encode.

    On tiled OOM, walks tile 256→128→64 then falls through to vae.encode().
    """
    side_max = max(image.shape[-2], image.shape[-3])
    needs_tile = (mode == "true (tiled)" or
                  (mode == "true" and side_max > AUTO_TILE_LATENT_THRESHOLD * 8))
    if not (needs_tile and hasattr(vae, "encode_tiled")):
        return {"samples": vae.encode(image)}
    while True:
        # comfy's encode_tiled_ internally uses tile_x // 2 for the steps calc,
        # so overlap must stay strictly < tile // 2 or we hit div-by-zero.
        eff_overlap = min(overlap, max(8, tile // 4))
        try:
            try:
                return {"samples": vae.encode_tiled(
                    image, tile_x=tile, tile_y=tile, overlap=eff_overlap)}
            except TypeError:
                return {"samples": vae.encode_tiled(
                    image, tile_x=tile, tile_y=tile)}
        except Exception as exc:
            _raise_non_oom(exc)
            if tile <= 64:
                logging.info("[GrimmRibbity] vae encode OOM at tile=%d, "
                              "falling through to vae.encode()", tile)
                _soft_empty_cache()
                return {"samples": vae.encode(image)}
            tile //= 2
            logging.info("[GrimmRibbity] vae encode OOM, retrying at tile=%d", tile)
            _soft_empty_cache()
