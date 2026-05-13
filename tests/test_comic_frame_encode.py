"""Unit tests for PromptLibraryComicFrameEncode's pure helpers.

The encoding itself needs a live CLIP + torch — not exercised here. These
tests pin the frame-index resolution, action-text selection, and active-
section filtering, which is where regressions are most likely to land
without anyone noticing during a refactor."""

import json
import sys
import types
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
NODE_DIR = HERE.parent


def _stub_server() -> None:
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


class ResolveActionTests(unittest.TestCase):
    """frame_index 1-indexed, clamped to [1, len(frames)]; empty frames
    list returns ('', 0). Exercises the same JSON parse the existing
    ComicFrame uses so adding the encoder doesn't introduce a different
    interpretation of the workflow's frames widget."""

    @classmethod
    def setUpClass(cls):
        _stub_server()
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "plib_cfe", str(NODE_DIR / "__init__.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.cls = mod.PromptLibraryComicFrameEncode

    def test_empty_json_returns_empty(self):
        self.assertEqual(self.cls._resolve_action("[]", 1), ("", 0))
        self.assertEqual(self.cls._resolve_action("", 1), ("", 0))
        self.assertEqual(self.cls._resolve_action("null", 5), ("", 0))

    def test_malformed_json_returns_empty(self):
        self.assertEqual(self.cls._resolve_action("not json", 1), ("", 0))
        self.assertEqual(self.cls._resolve_action("{}", 1), ("", 0))

    def test_picks_correct_index(self):
        frames = json.dumps(["a", "b", "c"])
        self.assertEqual(self.cls._resolve_action(frames, 1), ("a", 3))
        self.assertEqual(self.cls._resolve_action(frames, 2), ("b", 3))
        self.assertEqual(self.cls._resolve_action(frames, 3), ("c", 3))

    def test_index_clamps_at_both_ends(self):
        frames = json.dumps(["a", "b"])
        # Below 1 → 1
        self.assertEqual(self.cls._resolve_action(frames, 0), ("a", 2))
        self.assertEqual(self.cls._resolve_action(frames, -5), ("a", 2))
        # Above len → len
        self.assertEqual(self.cls._resolve_action(frames, 99), ("b", 2))

    def test_strips_whitespace_in_frames(self):
        frames = json.dumps(["  spaced  ", "\ttabs\t"])
        self.assertEqual(self.cls._resolve_action(frames, 1), ("spaced", 2))
        self.assertEqual(self.cls._resolve_action(frames, 2), ("tabs", 2))

    def test_non_list_payload_treated_as_empty(self):
        # User pastes a malformed but valid-JSON value into frames_json —
        # treat as no frames rather than crashing.
        self.assertEqual(self.cls._resolve_action('"a string"', 1), ("", 0))
        self.assertEqual(self.cls._resolve_action("42", 1), ("", 0))


class ActiveSectionsTests(unittest.TestCase):
    """Filtering empties without reordering — preserves the
    character → scene → background → action → extra encoding order,
    which affects the final conditioning."""

    @classmethod
    def setUpClass(cls):
        _stub_server()
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "plib_cfe2", str(NODE_DIR / "__init__.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.cls = mod.PromptLibraryComicFrameEncode

    def test_all_filled_passes_through(self):
        out = self.cls._active_sections([
            ("character", "a"), ("scene", "b"), ("background", "c"),
        ])
        self.assertEqual(out, [("character", "a"), ("scene", "b"),
                                ("background", "c")])

    def test_empty_strings_filtered(self):
        out = self.cls._active_sections([
            ("character", "a"), ("scene", ""), ("background", "c"),
        ])
        self.assertEqual(out, [("character", "a"), ("background", "c")])

    def test_whitespace_only_filtered(self):
        out = self.cls._active_sections([
            ("character", "a"), ("scene", "   \n\t"), ("background", "c"),
        ])
        self.assertEqual(out, [("character", "a"), ("background", "c")])

    def test_none_values_filtered(self):
        # Defensive: if a wire emits "" or the user clears a multiline,
        # we should not propagate.
        out = self.cls._active_sections([
            ("character", ""), ("scene", ""),
        ])
        self.assertEqual(out, [])

    def test_strips_whitespace_in_output(self):
        # The encoder hands the stripped string to the tokenizer so
        # leading/trailing whitespace doesn't waste a token slot.
        out = self.cls._active_sections([
            ("character", "  hello "), ("scene", "world\n"),
        ])
        self.assertEqual(out, [("character", "hello"), ("scene", "world")])


class SectionTokenStatsTests(unittest.TestCase):
    """The token-counting diagnostic must never raise from a path that
    also returns CONDITIONING — verified by feeding broken/missing
    tokenizer return shapes through a fake CLIP."""

    @classmethod
    def setUpClass(cls):
        _stub_server()
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "plib_cfe3", str(NODE_DIR / "__init__.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # staticmethod wrapper sidesteps Python's descriptor protocol —
        # without it, attaching a module function to a class turns it into
        # an unbound method that expects `self` as the first positional arg.
        cls.stats = staticmethod(mod._comicframe_section_token_stats)

    def _fake_clip(self, tokens_return):
        class FakeClip:
            def tokenize(self, text):
                return tokens_return
        return FakeClip()

    def test_sdxl_shape_counts_real_tokens(self):
        # Two chunks of 77, each with 1 BOS + 1 EOS + 75 real, plus the
        # second chunk padded with zeros after some real tokens.
        chunk_a = [(49406, 1.0)] + [(100 + i, 1.0) for i in range(75)] + [(49407, 1.0)]
        chunk_b = [(49406, 1.0)] + [(200 + i, 1.0) for i in range(30)] + [(0, 1.0)] * 45 + [(49407, 1.0)]
        clip = self._fake_clip({"g": [chunk_a, chunk_b], "l": [chunk_a, chunk_b]})
        out = self.stats(clip, "irrelevant")
        # 75 real in chunk_a + 30 real in chunk_b = 105
        self.assertEqual(out, {"real": 105, "chunks": 2})

    def test_sd15_shape_handled(self):
        chunk = [(49406, 1.0)] + [(100 + i, 1.0) for i in range(40)] + [(49407, 1.0)]
        clip = self._fake_clip({"tokens": [chunk]})
        out = self.stats(clip, "x")
        self.assertEqual(out, {"real": 40, "chunks": 1})

    def test_tokenize_raising_returns_zeros(self):
        class BrokenClip:
            def tokenize(self, text):
                raise RuntimeError("tokenizer asplode")
        out = self.stats(BrokenClip(), "x")
        self.assertEqual(out, {"real": 0, "chunks": 0})

    def test_empty_tokenize_return_handled(self):
        out = self.stats(self._fake_clip({}), "x")
        self.assertEqual(out, {"real": 0, "chunks": 0})
        out = self.stats(self._fake_clip(None), "x")
        self.assertEqual(out, {"real": 0, "chunks": 0})


if __name__ == "__main__":
    unittest.main()
