import io
import stat
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class MacTests(unittest.TestCase):
    """macOS code paths, exercised on any OS by patching the platform checks."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_rosetta_python_gets_native_arm64_build(self):
        sysctl = mock.Mock(stdout="1\n")
        with mock.patch.object(y, "IS_MAC", True), mock.patch.object(y, "IS_WIN", False), \
                mock.patch("platform.machine", return_value="x86_64"), \
                mock.patch.object(y.subprocess, "run", return_value=sysctl):
            self.assertEqual(y.vibe_asset_name(), "vibe-server-darwin-arm64")
        with mock.patch.object(y, "IS_MAC", True), mock.patch("platform.machine", return_value="arm64"):
            self.assertEqual(y.vibe_asset_name(), "vibe-server-darwin-arm64")

    def test_bundle_with_ffmpeg_is_unpacked(self):
        tgz = self.tmp / "bundle.tar.gz"
        with tarfile.open(tgz, "w:gz") as tf:
            for name in ("vibe-server", "ffmpeg"):
                data = b"#!/bin/sh\necho " + name.encode() + b"\n"
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o755
                tf.addfile(info, io.BytesIO(data))
        bin_dir = self.tmp / "bin"
        with mock.patch.object(y, "BIN_DIR", bin_dir):
            y._install_archive(tgz.as_uri(), bin_dir / "vibe-server")
            self.assertTrue((bin_dir / "vibe-server").is_file())
            self.assertTrue((bin_dir / "ffmpeg").stat().st_mode & stat.S_IEXEC)
            with mock.patch.object(y.shutil, "which", return_value=None), \
                    mock.patch.object(y, "VIBE_APP_DIRS", []):
                self.assertEqual(y.find_ffmpeg(), str(bin_dir / "ffmpeg"))

    def test_finds_vibe_app_binaries_and_models(self):
        app = self.tmp / "vibe.app/Contents/MacOS"
        app.mkdir(parents=True)
        for name in ("vibe-server", "ffmpeg"):
            (app / name).write_text("#!/bin/sh\n")
            (app / name).chmod(0o755)
        models = self.tmp / "Application Support/github.com.thewh1teagle.vibe"
        models.mkdir(parents=True)
        (models / "ggml-large-v3-turbo.bin").write_bytes(b"x")
        with mock.patch.object(y, "VIBE_APP_DIRS", [app]), mock.patch.object(y, "VIBE_MODEL_DIRS", [models]), \
                mock.patch.object(y, "BIN_DIR", self.tmp / "none"), mock.patch.object(y, "MODEL_DIR", self.tmp / "none"), \
                mock.patch.object(y.shutil, "which", return_value=None), mock.patch.dict("os.environ", {}, clear=False):
            y.os.environ.pop("VIBE_SERVER_BIN", None)
            y.os.environ.pop("VIBE_SERVER_FFMPEG_PATH", None)
            self.assertEqual(y.find_vibe_server(None), str(app / "vibe-server"))
            self.assertEqual(y.find_ffmpeg(), str(app / "ffmpeg"))
            self.assertEqual(y.ensure_model("large-v3-turbo", False), models / "ggml-large-v3-turbo.bin")

    def test_data_dir_on_mac(self):
        with mock.patch.object(y, "IS_MAC", True):
            self.assertEqual(y._default_data_dir(), Path.home() / "Library/Application Support/ytscribe")


if __name__ == "__main__":
    unittest.main()
