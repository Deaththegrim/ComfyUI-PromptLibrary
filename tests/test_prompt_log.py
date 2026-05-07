"""Unit tests for prompt_log.py — JSONL append, default path, build_record."""

import importlib.util
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
NODE_DIR = HERE.parent


def _stub_minimal_deps() -> None:
    """Stub the heavy ComfyUI deps so prompt_log.py + its civitai_save import
    can load in .testenv without torch / a live ComfyUI."""
    if "server" not in sys.modules:
        fake = types.ModuleType("server")
        class _Routes:
            def get(self, *_a, **_k): return lambda f: f
            def post(self, *_a, **_k): return lambda f: f
        fake.PromptServer = types.SimpleNamespace(
            instance=types.SimpleNamespace(routes=_Routes()))
        sys.modules["server"] = fake
    # folder_paths is consulted by prompt_log for the default output dir;
    # provide a bare module so it imports — tests that exercise the default
    # path inject their own get_output_directory().
    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        fp.get_output_directory = lambda: str(Path(tempfile.gettempdir()) / "comfy_out")
        fp.get_filename_list = lambda *_a, **_k: []
        fp.get_full_path = lambda *_a, **_k: None
        sys.modules["folder_paths"] = fp


def _load_pkg(tmp_root: Path) -> types.ModuleType:
    """Load the package under a unique name + register it so submodules can
    do `from .civitai_save import ...`. Returns the prompt_log submodule."""
    _stub_minimal_deps()
    pkg_name = f"plib_pl_{tmp_root.name}"
    spec = importlib.util.spec_from_file_location(
        pkg_name, NODE_DIR / "__init__.py",
        submodule_search_locations=[str(NODE_DIR)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)

    sub_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.prompt_log", NODE_DIR / "prompt_log.py")
    pl = importlib.util.module_from_spec(sub_spec)
    sys.modules[f"{pkg_name}.prompt_log"] = pl
    sub_spec.loader.exec_module(pl)
    return pl


class PromptLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="plog_"))
        self.pl = _load_pkg(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        # Leave sys.modules tidy — repeated imports under the same package
        # name would otherwise stack up as the test class iterates.
        for k in list(sys.modules):
            if k.startswith("plib_pl_"):
                del sys.modules[k]

    def test_resolve_relative_under_output_dir(self):
        # folder_paths.get_output_directory returns a tempdir; relative path
        # should sit under it.
        p = self.pl.resolve_log_path("custom.jsonl")
        self.assertTrue(str(p).endswith("custom.jsonl"))
        # Absolute path is honoured verbatim.
        abs_path = self.tmp / "abs.jsonl"
        self.assertEqual(self.pl.resolve_log_path(str(abs_path)), abs_path)

    def test_resolve_empty_uses_default(self):
        p = self.pl.resolve_log_path("")
        self.assertTrue(str(p).endswith("prompt_logs/prompts.jsonl"))

    def test_append_creates_dirs_and_writes_jsonl(self):
        log = self.tmp / "a" / "b" / "log.jsonl"
        self.pl.append_prompt_log({"hello": "world"}, log)
        self.pl.append_prompt_log({"second": 2}, log)
        lines = log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0]), {"hello": "world"})
        self.assertEqual(json.loads(lines[1]), {"second": 2})

    def test_append_swallows_io_error(self):
        # Path that can't possibly be created (parent is a regular file).
        blocker = self.tmp / "blocker"
        blocker.write_text("x")
        log = blocker / "child.jsonl"
        # Must not raise — sampling should never crash because logging failed.
        self.pl.append_prompt_log({"a": 1}, log)

    def test_build_record_runtime_overrides_trace(self):
        prompt_trace = {
            "1": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42, "steps": 99, "cfg": 9.0,
                    "sampler_name": "trace_sampler",
                    "scheduler": "trace_sched",
                },
            },
        }
        rec = self.pl.build_record(
            sampler_node="GrimmRibbityAnimaSampler",
            prompt_trace=prompt_trace,
            runtime={"seed": 7, "steps": 25, "cfg": 4.0,
                      "sampler_name": "rt_sampler", "scheduler": "rt_sched",
                      "batch_size": 2, "denoise": 0.8},
        )
        self.assertEqual(rec["sampler"], "GrimmRibbityAnimaSampler")
        self.assertEqual(rec["seed"], 7)
        self.assertEqual(rec["steps"], 25)
        self.assertEqual(rec["sampler_name"], "rt_sampler")
        self.assertEqual(rec["batch_size"], 2)
        self.assertEqual(rec["denoise"], 0.8)
        # Timestamp present, ISO 8601-ish.
        self.assertIn("T", rec["ts"])
        self.assertTrue(rec["ts"].endswith("Z"))

    def test_build_record_handles_no_trace(self):
        rec = self.pl.build_record(
            sampler_node="GrimmRibbitySamplerSDXL",
            prompt_trace=None,
            runtime={"seed": 1, "steps": 10, "cfg": 5.0,
                      "sampler_name": "euler", "scheduler": "simple",
                      "batch_size": 1},
        )
        self.assertNotIn("model", rec)
        self.assertNotIn("loras", rec)
        self.assertEqual(rec["seed"], 1)

    def test_build_record_skips_none_runtime_fields(self):
        rec = self.pl.build_record(
            sampler_node="x",
            prompt_trace=None,
            runtime={"seed": 1, "denoise": None, "batch_size": 1},
        )
        self.assertNotIn("denoise", rec)
        self.assertEqual(rec["batch_size"], 1)

    def test_build_record_extracts_loras_and_model_from_trace(self):
        """When the workflow trace contains a CheckpointLoader + LoraLoader
        chain feeding the sampler, build_record should pull both into the
        log line (model_label + loras list)."""
        prompt_trace = {
            "1": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 5, "steps": 20, "cfg": 7.0,
                    "sampler_name": "euler", "scheduler": "simple",
                    "model": ["10", 0], "positive": ["20", 0], "negative": ["20", 0],
                },
            },
            "10": {
                "class_type": "LoraLoader",
                "inputs": {"model": ["11", 0], "lora_name": "Anima.safetensors",
                            "strength_model": 0.85, "strength_clip": 0.85},
            },
            "11": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "Anima/anima_v3.safetensors"},
            },
            "20": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "a girl"},
            },
        }
        rec = self.pl.build_record(
            sampler_node="GrimmRibbityAnimaSampler",
            prompt_trace=prompt_trace,
            runtime={"seed": 5, "steps": 20, "cfg": 7.0,
                      "sampler_name": "euler", "scheduler": "simple"},
        )
        self.assertIn("model", rec)
        self.assertIn("anima_v3", rec["model"])
        self.assertEqual(len(rec["loras"]), 1)
        self.assertEqual(rec["loras"][0]["name"], "Anima.safetensors")
        self.assertAlmostEqual(rec["loras"][0]["strength"], 0.85)
        self.assertEqual(rec["positive"], "a girl")


if __name__ == "__main__":
    unittest.main(verbosity=2)
