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
