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


def _install_folder_paths_stub():
    if "folder_paths" in sys.modules:
        return
    fp = types.ModuleType("folder_paths")
    fp.get_filename_list = lambda _key: []
    fp.get_full_path = lambda *_a, **_k: None
    fp.folder_names_and_paths = {}
    fp.models_dir = "/tmp"
    fp.supported_pt_extensions = {".pt", ".pth", ".safetensors"}
    sys.modules["folder_paths"] = fp


_install_server_stub()
_install_folder_paths_stub()
