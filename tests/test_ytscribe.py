import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ytscribe as y  # noqa: E402


class FormatTests(unittest.TestCase):
    def test_srt_and_timestamps(self):
        segs = [{"start": 0, "end": 1.5, "text": " a"}, {"start": 3661.25, "end": 3662, "text": " b"}]
        self.assertIn("01:01:01,250 --> 01:01:02,000", y.to_srt(segs))
        self.assertEqual(y.to_timestamped(segs), "[00:00:00] a\n[01:01:01] b")

    def test_split_and_lang(self):
        self.assertEqual(len(y.split_text("x" * 10 + "\n" + "y" * 10 + "\n", 12)), 2)
        self.assertEqual(y.guess_language("Привет, как дела?"), "ru")
        self.assertEqual(y.guess_language("Hello there"), "en")
        self.assertEqual(y.safe_name('a/b:c?'), "a b c")


class EngineTests(unittest.TestCase):
    """Spawns the engine the same way the real script does, against a fake vibe-server."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bin = self.tmp / "vibe-server"
        self.bin.write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT / 'tests' / 'fake_vibe_server.py'} \"$@\"\n")
        self.bin.chmod(self.bin.stat().st_mode | stat.S_IEXEC)
        self.model = self.tmp / "ggml-base.bin"
        self.model.write_bytes(b"x")
        self.audio = self.tmp / "a.wav"
        self.audio.write_bytes(b"RIFF....")

    def test_spawn_load_transcribe(self):
        eng = y.VibeEngine(None, str(self.bin), self.model, False, None, False)
        eng.start()
        try:
            self.assertTrue(y.http_ok(f"{eng.url}/health"))
            _, ready = y.http_json("GET", f"{eng.url}/ready")
            self.assertEqual(ready["model"], str(self.model))
            res = eng.transcribe(self.audio, "en", None)
            self.assertEqual(res["text"], "Hello world. Second line.")
            res = eng.transcribe(self.audio, "ru", None)
            self.assertEqual(y.guess_language(res["text"]), "ru")
            # A second client attaches to the running engine and sees the model is already loaded.
            y.VibeEngine(eng.url, None, self.model, False, None, False).start()
        finally:
            eng.stop()


if __name__ == "__main__":
    unittest.main()
