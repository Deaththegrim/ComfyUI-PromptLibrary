"""Unit tests for the consolidated comic-pipeline nodes:
``PromptLibrarySceneFull`` and ``PromptLibraryComicComposer``.

Encoding itself still requires a live CLIP + torch, so the focus here
is on the pure helpers — Scene Full's join behaviour, Composer's
``_weight_wrap`` weighting, and the section-ordering contract the
encoder commits to. These are the parts most likely to drift during
a refactor and not get caught visually."""

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


def _load_module():
    """Import the package against the stubbed server module so the route
    decorators don't try to talk to a real PromptServer instance."""
    _stub_server()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "plib_sfc", str(NODE_DIR / "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SceneFullBuildTests(unittest.TestCase):
    """Scene Full preserves the old Scene's atmosphere joining and adds
    three independent output channels (background / ambient / negative).
    The contract is: each output is independent — strip whitespace and
    pass through, dropdown `(none)` is treated as empty everywhere."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()
        cls.cls = cls.mod.PromptLibrarySceneFull
        cls.NONE = cls.mod._SCENE_NONE

    def test_joins_atmosphere_dropdowns_in_order(self):
        scene, bg, ambient, neg = self.cls().build(
            time_of_day="night", weather="rain", lighting="moonbeam",
            camera_angle="from above", mood="ominous", framing="centered",
            extra="anamorphic",
            background="", ambient_action="", scene_negative="",
        )
        self.assertEqual(
            scene,
            "night, rain, moonbeam, from above, ominous, centered, anamorphic",
        )
        self.assertEqual(bg, "")
        self.assertEqual(ambient, "")
        self.assertEqual(neg, "")

    def test_none_dropdown_skipped_but_keeps_order(self):
        scene, _b, _a, _n = self.cls().build(
            time_of_day=self.NONE, weather="rain", lighting=self.NONE,
            camera_angle="from above", mood=self.NONE, framing="centered",
            extra="",
            background="", ambient_action="", scene_negative="",
        )
        self.assertEqual(scene, "rain, from above, centered")

    def test_background_and_ambient_and_negative_emit_independently(self):
        # The four outputs are independent — none of them mixes into another.
        scene, bg, ambient, neg = self.cls().build(
            time_of_day=self.NONE, weather=self.NONE, lighting=self.NONE,
            camera_angle=self.NONE, mood=self.NONE, framing=self.NONE,
            extra="",
            background="  dark moonlit moor, distant tombstones  ",
            ambient_action=" smoking on the fire escape ",
            scene_negative=" noisy background, weird artifacts ",
        )
        self.assertEqual(scene, "")
        self.assertEqual(bg, "dark moonlit moor, distant tombstones")
        self.assertEqual(ambient, "smoking on the fire escape")
        self.assertEqual(neg, "noisy background, weird artifacts")

    def test_separator_only_affects_atmosphere_join(self):
        # The custom separator applies to the dropdowns-and-extra join,
        # NOT to background / ambient / negative (each is its own STRING).
        scene, bg, ambient, neg = self.cls().build(
            time_of_day="night", weather="rain", lighting="",
            camera_angle="", mood="", framing="",
            extra="",
            background="brick wall", ambient_action="smoking",
            scene_negative="bad anatomy",
            separator=" | ",
        )
        self.assertEqual(scene, "night | rain")
        self.assertEqual(bg, "brick wall")
        self.assertEqual(ambient, "smoking")
        self.assertEqual(neg, "bad anatomy")

    def test_everything_empty_returns_empty_tuple(self):
        out = self.cls().build(
            time_of_day=self.NONE, weather=self.NONE, lighting=self.NONE,
            camera_angle=self.NONE, mood=self.NONE, framing=self.NONE,
            extra="", background="", ambient_action="", scene_negative="",
        )
        self.assertEqual(out, ("", "", "", ""))


class ComicComposerWeightTests(unittest.TestCase):
    """``_weight_wrap`` is the only behaviour distinct from
    ``PromptLibraryComicFrameEncode``'s helpers; the rest is inherited.
    Pin the no-op-at-1.0 semantics and the formatting."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()
        cls.cls = cls.mod.PromptLibraryComicComposer

    def test_weight_one_returns_text_unchanged(self):
        self.assertEqual(self.cls._weight_wrap("hello", 1.0), "hello")

    def test_weight_one_with_float_drift_treated_as_one(self):
        # Comfy stores INT/FLOAT widgets as floats; 0.999999 from a slider
        # should not produce `(hello:1.00)` garbage in the prompt.
        self.assertEqual(self.cls._weight_wrap("hello", 1.0 + 1e-9), "hello")
        self.assertEqual(self.cls._weight_wrap("hello", 1.0 - 1e-9), "hello")

    def test_above_one_wraps_with_two_decimal_format(self):
        self.assertEqual(self.cls._weight_wrap("hello", 1.25), "(hello:1.25)")
        self.assertEqual(self.cls._weight_wrap("hello", 1.5), "(hello:1.50)")

    def test_below_one_still_wraps_for_de_emphasis(self):
        # The user may genuinely want to de-emphasise the action; preserve
        # that. The widget's min=0.5 floor lives in INPUT_TYPES; the helper
        # itself doesn't clamp, it just renders.
        self.assertEqual(self.cls._weight_wrap("hello", 0.7), "(hello:0.70)")


class ComicComposerSectionOrderTests(unittest.TestCase):
    """The conditioning the Composer emits is the sequence-axis concat
    of its positive sections in a fixed order. Document and pin that
    order — the section list is the contract. (Encoding itself isn't
    exercised here; we drive it with a FakeClip that records the calls.)"""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()
        cls.cls = cls.mod.PromptLibraryComicComposer

    def _drive(self, **overrides):
        """Build a fake CLIP whose tokenize records the texts it was
        called with, and whose encode_from_tokens_scheduled returns a
        unique sentinel per call so _concat sees real (different) values.
        Returns the list of section texts in the order encode walked
        them, plus the final tuple."""
        calls = []
        # Inheriting _concat does torch.cat on real tensors — to avoid a
        # torch dependency in tests we monkey-patch _concat on the class
        # instance to a list-append instead, then assert against the
        # recorded list. The order of the sections IS the contract.
        class FakeClip:
            def tokenize(self, text):
                calls.append(text)
                return {"g": [[(49406, 1.0), (1, 1.0), (49407, 1.0)]],
                         "l": [[(49406, 1.0), (1, 1.0), (49407, 1.0)]]}
            def encode_from_tokens_scheduled(self, tokens):
                return [[f"COND_{len(calls)}", {"pooled_output": None}]]

        # Patch _concat to a no-op accumulator that keeps the first
        # conditioning — we're not testing the concat, only the order
        # the encoder walked sections.
        node = self.cls()
        original_concat = self.cls._concat
        self.cls._concat = staticmethod(lambda to, frm: to)
        try:
            out = node.encode(
                clip=FakeClip(),
                frames_json='["frameaction"]',
                frame_index=1,
                character="CHAR",
                scene="SCENE",
                background="BG",
                ambient_action="AMB",
                extra_positive="EXTRA",
                character_negative="CN",
                scene_negative="SN",
                global_negative="GN",
                extra_negative="EN",
                frame_action_weight=1.0,
                base_seed=0,
                diagnostic_print=False,
            )
        finally:
            self.cls._concat = original_concat
        return calls, out

    def test_positive_then_negative_then_section_order(self):
        calls, _out = self._drive()
        # encode() walks positive sections, then negative sections.
        # Within each, order is documented in the encode() method's
        # `positive_sections` and `negative_sections` lists.
        # Diagnostic-print runs another tokenize per section for stats;
        # but with diagnostic_print=False the stats call is bypassed and
        # we only see the real encoding tokenize() calls.
        expected = [
            "CHAR",      # positive[0]
            "SCENE",     # positive[1]
            "AMB",       # positive[2] — ambient_action
            "BG",        # positive[3] — background
            "frameaction",  # positive[4] — per-frame action (weight=1.0 = no wrap)
            "EXTRA",     # positive[5]
            "CN",        # negative[0]
            "SN",        # negative[1]
            "GN",        # negative[2]
            "EN",        # negative[3]
        ]
        self.assertEqual(calls, expected)

    def test_empty_sections_skipped_from_encoding(self):
        # Inheriting _active_sections from the parent: empty/whitespace
        # entries are filtered out BEFORE encoding. Confirm via the
        # recorded tokenize calls.
        calls = []
        class FakeClip:
            def tokenize(self, text):
                calls.append(text)
                return {"g": [[(49406, 1.0), (1, 1.0), (49407, 1.0)]],
                         "l": [[(49406, 1.0), (1, 1.0), (49407, 1.0)]]}
            def encode_from_tokens_scheduled(self, tokens):
                return [[f"COND_{len(calls)}", {"pooled_output": None}]]
        original_concat = self.cls._concat
        self.cls._concat = staticmethod(lambda to, frm: to)
        try:
            self.cls().encode(
                clip=FakeClip(),
                frames_json='["theaction"]',
                frame_index=1,
                character="C", scene="", background="", ambient_action="",
                extra_positive="",
                character_negative="", scene_negative="",
                global_negative="GLOBAL", extra_negative="",
                frame_action_weight=1.0,
                base_seed=0,
                diagnostic_print=False,
            )
        finally:
            self.cls._concat = original_concat
        # Only the non-empty sections should have been tokenized.
        # Positive: just character + action. Negative: just global_negative.
        self.assertEqual(calls, ["C", "theaction", "GLOBAL"])

    def test_action_weight_applied_to_action_section_only(self):
        # frame_action_weight wraps the per-frame action only — not the
        # ambient_action, not the character, etc.
        calls = []
        class FakeClip:
            def tokenize(self, text):
                calls.append(text)
                return {"g": [[(49406, 1.0), (1, 1.0), (49407, 1.0)]],
                         "l": [[(49406, 1.0), (1, 1.0), (49407, 1.0)]]}
            def encode_from_tokens_scheduled(self, tokens):
                return [[f"C_{len(calls)}", {"pooled_output": None}]]
        original_concat = self.cls._concat
        self.cls._concat = staticmethod(lambda to, frm: to)
        try:
            self.cls().encode(
                clip=FakeClip(),
                frames_json='["panel_verb"]',
                frame_index=1,
                character="char", scene="scn", background="",
                ambient_action="amb", extra_positive="",
                character_negative="", scene_negative="",
                global_negative="", extra_negative="",
                frame_action_weight=1.3,
                base_seed=0,
                diagnostic_print=False,
            )
        finally:
            self.cls._concat = original_concat
        # Order: character, scene, ambient_action, action(weight-wrapped).
        self.assertIn("(panel_verb:1.30)", calls)
        self.assertNotIn("(amb:1.30)", calls)
        self.assertNotIn("(char:1.30)", calls)


if __name__ == "__main__":
    unittest.main()
