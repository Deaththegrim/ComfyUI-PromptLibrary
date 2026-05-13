"""Tests for the tools/inspect_png.py CLI inspector.

Most logic is in pure parse helpers that don't need a real PNG; the one
exception is the PIL round trip which is covered in the existing
LibrarySnapshotPngRoundTripTests under test_civitai_save.py. Here we
test the chunk → structured-dict transforms + the folder-walking
aggregator on hand-built dicts."""

import io
import json
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

import sys
TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import inspect_png as ins


def _make_png(tmp_path: Path, name: str, *,
              library: dict | None = None,
              parameters: str | None = None) -> Path:
    """Build a tiny PNG on disk with our chunks for the parser to read.
    Used by the end-to-end inspector tests below."""
    info = PngImagePlugin.PngInfo()
    if library is not None:
        info.add_text("grimmribbity_library", json.dumps(library))
    if parameters is not None:
        info.add_text("parameters", parameters)
    path = tmp_path / name
    Image.new("RGB", (4, 4), (0, 0, 0)).save(
        path, format="PNG", pnginfo=info)
    return path


class ParseLibrarySnapshotTests(unittest.TestCase):
    def test_returns_none_when_missing(self):
        self.assertIsNone(ins.parse_library_snapshot({}))

    def test_returns_none_when_unparseable(self):
        # The chunk is present but its content isn't valid JSON. Skip
        # silently rather than crash — the inspector is read-only and
        # should never bring down a folder walk over one bad file.
        self.assertIsNone(ins.parse_library_snapshot(
            {"grimmribbity_library": "{not json"}))

    def test_decodes_valid_json(self):
        snap = {"schema": 1, "entries": [{"id": "x"}]}
        out = ins.parse_library_snapshot(
            {"grimmribbity_library": json.dumps(snap)})
        self.assertEqual(out, snap)


class ParseA1111ParametersTests(unittest.TestCase):
    def test_empty_chunk_returns_empty(self):
        self.assertEqual(ins.parse_a1111_parameters({}), {})

    def test_extracts_steps_sampler_seed(self):
        raw = (
            "a cat\nNegative prompt: lowres\n"
            "Steps: 30, Sampler: dpmpp_3m_sde_gpu karras, "
            "CFG scale: 5, Seed: 666, Size: 1024x1024, "
            "Model hash: c73e78822c, Model: steinillustrious_V16"
        )
        out = ins.parse_a1111_parameters({"parameters": raw})
        self.assertEqual(out["steps"], "30")
        self.assertEqual(out["sampler"], "dpmpp_3m_sde_gpu karras")
        self.assertEqual(out["seed"], "666")
        self.assertEqual(out["cfg_scale"], "5")
        self.assertEqual(out["model"], "steinillustrious_V16")
        self.assertEqual(out["negative"], "lowres")
        # _raw preserves the original for callers that want it.
        self.assertEqual(out["_raw"], raw)

    def test_non_string_chunk_returns_empty(self):
        # Defensive — chunks should be strings but a bad encoder could
        # land non-strings.
        self.assertEqual(
            ins.parse_a1111_parameters({"parameters": 123}),
            {},
        )


class RenderEntryTests(unittest.TestCase):
    def test_missing_entry_marked(self):
        lines = ins.render_entry({"id": "ghost", "missing": True})
        self.assertEqual(len(lines), 1)
        self.assertIn("MISSING", lines[0])

    def test_basic_entry(self):
        lines = ins.render_entry({
            "id": "cult_card_frog",
            "name": "Bog Frog -- Cult Card",
            "tags": ["Cards", "Cards:water"],
            "loras": [],
            "negative": "",
        })
        joined = "\n".join(lines)
        self.assertIn("cult_card_frog", joined)
        self.assertIn("Bog Frog", joined)
        self.assertIn("Cards, Cards:water", joined)

    def test_lora_symmetric_strength(self):
        # When model + clip strength match, show a single value (cleaner).
        lines = ins.render_entry({
            "id": "x",
            "loras": [{"name": "a.safetensors",
                       "strength_model": 0.5,
                       "strength_clip": 0.5,
                       "enabled": True}],
        })
        joined = "\n".join(lines)
        self.assertIn("a.safetensors @ 0.50", joined)
        # No split-strength notation when they agree.
        self.assertNotIn("model=", joined)

    def test_lora_asymmetric_strength_split(self):
        lines = ins.render_entry({
            "id": "x",
            "loras": [{"name": "b.safetensors",
                       "strength_model": 0.7,
                       "strength_clip": 0.3,
                       "enabled": True}],
        })
        joined = "\n".join(lines)
        self.assertIn("model=0.70/clip=0.30", joined)

    def test_negative_truncated_in_pretty(self):
        lines = ins.render_entry({
            "id": "x",
            "loras": [],
            "negative": "z" * 200,
        })
        joined = "\n".join(lines)
        # Pretty output snips at 80 chars + ellipsis.
        self.assertIn("…", joined)
        self.assertLess(len(joined), 250)


class ReadChunksRoundTripTests(unittest.TestCase):
    """Confirm read_chunks survives the actual PIL encode/decode."""

    def test_round_trip_chunks(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_png(
                Path(tmp), "a.png",
                library={"schema": 1, "entries": [{"id": "alpha"}]},
                parameters="Steps: 20, Seed: 1",
            )
            chunks = ins.read_chunks(p)
            self.assertIn("grimmribbity_library", chunks)
            self.assertIn("parameters", chunks)

    def test_read_chunks_on_garbage_returns_empty(self):
        # File exists but isn't a PNG — should return {} silently.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "not_a_png.png"
            bad.write_bytes(b"definitely not a png")
            self.assertEqual(ins.read_chunks(bad), {})

    def test_read_chunks_on_missing_file(self):
        self.assertEqual(ins.read_chunks(Path("/no/such/file.png")), {})


class CollectPathsTests(unittest.TestCase):
    def test_single_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_png(Path(tmp), "single.png", library={"entries": []})
            out = ins.collect_paths(p)
            self.assertEqual(out, [p])

    def test_folder_recursive(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            a = _make_png(root, "a.png", library={"entries": []})
            b = _make_png(root / "sub", "b.png", library={"entries": []})
            # Non-PNG should be ignored
            (root / "ignore.txt").write_text("not png")
            out = ins.collect_paths(root)
            self.assertEqual(set(out), {a, b})

    def test_nonexistent_returns_empty(self):
        self.assertEqual(ins.collect_paths(Path("/no/such/place")), [])


class SummarizeTests(unittest.TestCase):
    """Aggregator math. Captures stdout via contextlib so we can assert
    the histogram lines without depending on terminal width / colour."""

    def _run(self, paths):
        import io as _io
        import contextlib
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            ins.summarize(paths)
        return buf.getvalue()

    def test_counts_pngs_with_and_without_snapshot(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_png(root, "with_snap.png",
                      library={"entries": [{"id": "alpha",
                                             "tags": ["Cards"],
                                             "loras": []}]})
            _make_png(root, "no_snap.png", parameters="Steps: 10")
            out = self._run(sorted(root.glob("*.png")))
            self.assertIn("PNGs scanned         : 2", out)
            self.assertIn("with library chunk : 1", out)

    def test_tag_and_entry_histograms(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Three renders of the same card → entry shows up 3x.
            for i in range(3):
                _make_png(root, f"r{i}.png", library={"entries": [{
                    "id": "cult_card_frog",
                    "tags": ["Cards", "Cards:water"],
                    "loras": [{"name": "lora_a.safetensors",
                               "strength_model": 1.0, "strength_clip": 1.0,
                               "enabled": True}],
                }]})
            out = self._run(sorted(root.glob("*.png")))
            self.assertIn("   3  cult_card_frog", out)
            self.assertIn("   3  Cards:water", out)
            self.assertIn("   3  lora_a.safetensors", out)

    def test_missing_entries_counted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_png(root, "x.png", library={"entries": [
                {"id": "ghost", "missing": True},
            ]})
            out = self._run(sorted(root.glob("*.png")))
            self.assertIn("missing entries    : 1", out)


if __name__ == "__main__":
    unittest.main()
