"""Pytest bootstrap for GrimmRibbity.

The package's __init__.py does `from server import PromptServer` at import
time. When pytest walks the package tree (rootdir is ComfyUI's, with
pythonpath=.), it tries to import this package's __init__.py before any test
runs and crashes on ComfyUI's real server module. Stub `server` here so the
package init succeeds during collection.
"""
import sys
import types


def _install_server_stub():
    if "server" in sys.modules:
        return
    fake = types.ModuleType("server")

    class _Routes:
        def get(self, *_a, **_k):
            return lambda f: f

        def post(self, *_a, **_k):
            return lambda f: f

    fake.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=_Routes())
    )
    sys.modules["server"] = fake


_install_server_stub()
