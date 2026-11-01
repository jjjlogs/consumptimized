#!/usr/bin/env python3
"""
ytscribe — download YouTube videos with yt-dlp, transcribe them locally with the
Vibe engine (vibe-server, whisper.cpp), save the transcripts and summarize them.

  * one URL, many URLs, playlists, channels or a batch file of URLs
  * English or Russian audio (or auto-detect)
  * checks that yt-dlp, ffmpeg, vibe-server and a Whisper model are present,
    installs what is missing, starts the engine if it is not running and loads
    the model
  * summaries in a preset style or your own prompt, through Claude, any
    OpenAI-compatible API (OpenAI, LM Studio, llama.cpp ...) or a local Ollama

Only the Python standard library is needed. Run `ytscribe.py --help`.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import ssl
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

APP = "ytscribe"
IS_MAC = platform.system() == "Darwin"
IS_WIN = platform.system() == "Windows"


def _default_data_dir() -> Path:
    if IS_MAC:
        return Path.home() / "Library" / "Application Support" / APP
    if IS_WIN:
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP


DATA_DIR = Path(os.environ.get("YTSCRIBE_HOME") or _default_data_dir())
BIN_DIR = DATA_DIR / "bin"
MODEL_DIR = DATA_DIR / "models"

VIBE_REPO = "thewh1teagle/vibe"
VIBE_APP_ID = "github.com.thewh1teagle.vibe"
# Vibe desktop app installs (macOS .app bundles keep sidecars in Contents/MacOS or Contents/Resources).
VIBE_APP_DIRS = [
    Path("/Applications/vibe.app/Contents/MacOS"),
    Path("/Applications/vibe.app/Contents/Resources"),
    Path.home() / "Applications/vibe.app/Contents/MacOS",
    Path.home() / "Applications/vibe.app/Contents/Resources",
    Path("/usr/bin"),
    Path("/usr/lib/vibe"),
    Path("/usr/lib/vibe/binaries"),
    Path(os.environ.get("LOCALAPPDATA", "/nonexistent")) / "vibe",
    Path(os.environ.get("PROGRAMFILES", "/nonexistent")) / "vibe",
]
# Where the Vibe desktop app stores models it downloaded (its app_local_data_dir).
VIBE_MODEL_DIRS = [
    Path.home() / "Library/Application Support" / VIBE_APP_ID,
    Path.home() / ".local/share" / VIBE_APP_ID,
    Path(os.environ.get("LOCALAPPDATA", "/nonexistent")) / VIBE_APP_ID,
]
EXE = ".exe" if IS_WIN else ""


def _ssl_context() -> ssl.SSLContext:
    """python.org builds on macOS ship without CA certificates until 'Install Certificates.command'
    is run; fall back to certifi or the system bundle so HTTPS downloads still verify."""
    ctx = ssl.create_default_context()
    if ctx.cert_store_stats().get("x509_ca", 0) == 0 and not os.environ.get("SSL_CERT_FILE"):
        try:
            import certifi  # type: ignore
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            for bundle in ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt"):
                if os.path.exists(bundle):
                    ctx = ssl.create_default_context(cafile=bundle)
                    break
    return ctx


SSL_CTX = _ssl_context()
urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=SSL_CTX)))
WHISPER_MODELS = {
    # name -> file in huggingface.co/ggerganov/whisper.cpp
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large-v3": "ggml-large-v3.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
    "large-v3-turbo-q5_0": "ggml-large-v3-turbo-q5_0.bin",
}
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}"

# --------------------------------------------------------------------------- #
# Summary styles. {language} is replaced with the summary language name.
# --------------------------------------------------------------------------- #
STYLES = {
    "brief": "Summarize the transcript in one short paragraph (3-5 sentences) "
             "capturing the main point.",
    "bullets": "Summarize the transcript as 5-12 concise bullet points covering "
               "the key ideas, facts and conclusions, in the order they appear.",
    "detailed": "Write a detailed, well-structured summary of the transcript with "
                "Markdown headings for each major topic. Keep important numbers, "
                "names, arguments and examples.",
    "chapters": "Split the transcript into chapters. For each chapter give the "
                "start timestamp [HH:MM:SS], a short title and 1-3 sentences "
                "describing it. Use the timestamps present in the transcript.",
    "takeaways": "List the key takeaways and any actionable advice, steps, "
                 "recommendations or tools mentioned, as a Markdown checklist.",
    "tldr": "Give a one-line TL;DR, then 3 bullet points with the most important details.",
    "study": "Write study notes: key concepts with short definitions, the main "
             "explanation in bullet form, and 5 review questions with answers.",
}
LANG_NAMES = {"en": "English", "ru": "Russian"}

SYSTEM_PROMPT = (
    "You summarize transcripts of YouTube videos. The transcript was produced "
    "by automatic speech recognition and may contain recognition errors; infer "
    "the intended words where obvious. Do not invent facts that are not in the "
    "transcript. Write the answer in {language}. Output Markdown only."
)


def log(msg: str) -> None:
    print(f"[{APP}] {msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> None:
    log(f"ERROR: {msg}")
    sys.exit(code)


# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib only)
# --------------------------------------------------------------------------- #
def http_json(method: str, url: str, body=None, headers=None, timeout=60):
    data = None
    hdrs = {"Accept": "application/json", "User-Agent": APP}
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return resp.status, (json.loads(raw) if raw else None)


def http_ok(url: str, timeout=3) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    log(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": APP})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if sys.stderr.isatty() and time.time() - last > 0.5:
                pct = f"{done * 100 / total:5.1f}%" if total else ""
                print(f"\r  {done / 1e6:8.1f} MB {pct}", end="", file=sys.stderr)
                last = time.time()
    if sys.stderr.isatty():
        print(file=sys.stderr)
    tmp.replace(dst)


def multipart(fields: dict, file_field: str, file_path: Path):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        if v is None:
            continue
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()
    body = b"".join(parts) + head + file_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


# --------------------------------------------------------------------------- #
# Dependencies: yt-dlp, ffmpeg
# --------------------------------------------------------------------------- #
def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    if IS_MAC and shutil.which("xattr"):
        # Files fetched by a browser carry a quarantine flag that makes Gatekeeper block them.
        subprocess.run(["xattr", "-d", "com.apple.quarantine", str(path)], capture_output=True)


def ensure_ytdlp(auto_install: bool) -> list[str]:
    for exe in (shutil.which("yt-dlp"), str(BIN_DIR / f"yt-dlp{EXE}")):
        if exe and Path(exe).is_file() and os.access(exe, os.X_OK):
            return [exe]
    probe = subprocess.run([sys.executable, "-m", "yt_dlp", "--version"], capture_output=True)
    if probe.returncode == 0:
        return [sys.executable, "-m", "yt_dlp"]
    if not auto_install:
        die("yt-dlp not found. Install it (brew install yt-dlp / pip install -U yt-dlp) "
            "or run without --no-install.")
    if IS_MAC and shutil.which("brew"):
        log("yt-dlp not found, installing with Homebrew ...")
        if subprocess.run(["brew", "install", "yt-dlp"]).returncode == 0 and shutil.which("yt-dlp"):
            return [shutil.which("yt-dlp")]
    log("installing yt-dlp with pip ...")
    cmd = [sys.executable, "-m", "pip", "install", "--user", "-U", "yt-dlp"]
    if subprocess.run(cmd).returncode == 0:
        return [sys.executable, "-m", "yt_dlp"]
    # PEP 668 managed Pythons (Homebrew, Debian ...) refuse --user installs: use the standalone binary.
    name = "yt-dlp.exe" if IS_WIN else "yt-dlp_macos" if IS_MAC else "yt-dlp"
    target = BIN_DIR / f"yt-dlp{EXE}"
    download(f"https://github.com/yt-dlp/yt-dlp/releases/latest/download/{name}", target)
    _make_executable(target)
    return [str(target)]


def find_ffmpeg() -> str | None:
    candidates = [shutil.which("ffmpeg"), os.environ.get("VIBE_SERVER_FFMPEG_PATH"), str(BIN_DIR / f"ffmpeg{EXE}")]
    # Homebrew prefixes are not on PATH when the script runs from cron/launchd/Shortcuts.
    candidates += ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"] if IS_MAC else []
    candidates += [str(d / f"ffmpeg{EXE}") for d in VIBE_APP_DIRS]  # bundled with the Vibe app
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return None


def ensure_ffmpeg(auto_install: bool) -> str:
    if ff := find_ffmpeg():
        return ff
    if auto_install and IS_MAC:
        if shutil.which("brew"):
            log("ffmpeg not found, installing with Homebrew ...")
            subprocess.run(["brew", "install", "ffmpeg"])
        else:
            # The macOS vibe-server release archive ships an ffmpeg binary next to the server.
            log("ffmpeg not found, fetching the Vibe macOS bundle that includes ffmpeg ...")
            install_vibe_server(with_ffmpeg=True)
        if ff := find_ffmpeg():
            return ff
    hint = ("brew install ffmpeg" if IS_MAC else "winget install ffmpeg" if IS_WIN
            else "sudo apt install ffmpeg  (or your distro's package manager)")
    die(f"ffmpeg is required by yt-dlp and Vibe for audio conversion. Install it: {hint}")
    return ""


# --------------------------------------------------------------------------- #
# Vibe engine (vibe-server)
# --------------------------------------------------------------------------- #
def vibe_asset_name() -> str:
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    if IS_MAC:
        # Apple Silicon Macs report x86_64 when Python runs under Rosetta; prefer the native build.
        if arch == "amd64" and subprocess.run(["sysctl", "-n", "hw.optional.arm64"],
                                              capture_output=True, text=True).stdout.strip() == "1":
            arch = "arm64"
        return f"vibe-server-darwin-{arch}"
    if IS_WIN:
        return "vibe-server-windows-amd64.exe"
    if platform.system() == "Linux":
        return f"vibe-server-linux-{arch}"
    die(f"unsupported platform {platform.system()}/{machine} for vibe-server")
    return ""


def find_vibe_server(explicit: str | None) -> str | None:
    candidates = [explicit, os.environ.get("VIBE_SERVER_BIN"), shutil.which("vibe-server"),
                  str(BIN_DIR / f"vibe-server{EXE}")]
    candidates += [str(d / f"vibe-server{EXE}") for d in VIBE_APP_DIRS]  # bundled with the Vibe app
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    if explicit:
        die(f"--vibe-bin {explicit} is not an executable file")
    return None


def install_vibe_server(with_ffmpeg: bool = False) -> str:
    asset = vibe_asset_name()
    # On macOS the release also comes as a .tar.gz that bundles ffmpeg.
    archive = None
    if with_ffmpeg and IS_MAC:
        archive = f"{asset}-with-ffmpeg.tar.gz"
    wanted = archive or asset
    tags = [os.environ["VIBE_SERVER_VERSION"]] if os.environ.get("VIBE_SERVER_VERSION") else vibe_server_tags(wanted)
    target = BIN_DIR / f"vibe-server{EXE}"
    for tag in tags[:5]:
        url = f"https://github.com/{VIBE_REPO}/releases/download/{tag}/{wanted}"
        try:
            if archive:
                _install_archive(url, target)
            else:
                download(url, target)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # tag without a binary for this platform (yet)
                continue
            raise
        _make_executable(target)
        log(f"installed vibe-server {tag} -> {target}")
        return str(target)
    die(f"could not download {wanted}. Get it from https://github.com/{VIBE_REPO}/releases?q=server-v "
        f"and pass --vibe-bin, or set VIBE_SERVER_VERSION=server-vX.Y.Z")
    return ""


def _install_archive(url: str, target: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "bundle.tar.gz"
        download(url, tgz)
        out = Path(tmp) / "x"
        with tarfile.open(tgz) as tf:
            for member in tf.getmembers():  # refuse absolute paths / .. entries
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise RuntimeError(f"unsafe path in archive: {member.name}")
            if hasattr(tarfile, "data_filter"):  # Python 3.12+: also reject links/devices
                tf.extractall(out, filter="data")
            else:
                tf.extractall(out)
        files = [p for p in out.rglob("*") if p.is_file()]
        server = next((p for p in files if p.name.startswith("vibe-server")), None)
        if not server:
            raise RuntimeError("archive does not contain vibe-server")
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(server, target)
        for p in files:  # ffmpeg / ffprobe and any dylibs it needs
            if p != server:
                dst = BIN_DIR / p.name
                shutil.copy2(p, dst)
                if p.name.startswith("ff"):
                    _make_executable(dst)


def vibe_server_tags(asset: str) -> list[str]:
    """Newest-first vibe-server release tags: GitHub API, falling back to `git ls-remote`."""
    log("looking up the latest vibe-server release ...")
    try:
        _, releases = http_json("GET", f"https://api.github.com/repos/{VIBE_REPO}/releases?per_page=60")
        tags = [r["tag_name"] for r in releases if r.get("tag_name", "").startswith("server-v")
                and any(a["name"] == asset for a in r.get("assets", []))]
        if tags:
            return tags
    except Exception as exc:
        log(f"GitHub API unavailable ({exc}), trying git ls-remote")
    # On a Mac without Xcode tools /usr/bin/git is only a stub that pops up an installer.
    if not shutil.which("git") or (IS_MAC and subprocess.run(["xcode-select", "-p"], capture_output=True).returncode):
        return []
    out = subprocess.run(["git", "ls-remote", "--tags", f"https://github.com/{VIBE_REPO}", "server-v*"],
                         capture_output=True, text=True)
    tags = {line.split("refs/tags/")[1].removesuffix("^{}") for line in out.stdout.splitlines() if "refs/tags/" in line}

    def version(tag: str):
        return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.\-]", tag.removeprefix("server-v")))
    return sorted(tags, key=version, reverse=True)


def ensure_model(model: str, auto_install: bool) -> Path:
    path = Path(model).expanduser()
    if path.is_file():
        return path.resolve()
    if model not in WHISPER_MODELS:
        die(f"model '{model}' is neither a file nor one of: {', '.join(WHISPER_MODELS)}")
    filename = WHISPER_MODELS[model]
    # Reuse a model the Vibe desktop app already downloaded instead of fetching it again.
    for folder in [MODEL_DIR] + VIBE_MODEL_DIRS:
        if (folder / filename).is_file():
            return folder / filename
    path = MODEL_DIR / filename
    if not auto_install:
        die(f"model {path} is missing")
    download(MODEL_URL.format(filename), path)
    return path


class VibeEngine:
    """Makes sure a vibe-server is reachable and has the requested model loaded."""

    def __init__(self, url: str | None, binary: str | None, model: Path, auto_install: bool,
                 gpu_device: int | None, no_gpu: bool, ffmpeg: str | None = None):
        self.ffmpeg = ffmpeg
        self.url = url.rstrip("/") if url else None
        self.binary = binary
        self.model = model
        self.auto_install = auto_install
        self.gpu_device = gpu_device
        self.no_gpu = no_gpu
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        if self.url and http_ok(f"{self.url}/health"):
            log(f"using running Vibe engine at {self.url}")
        elif self.url:
            die(f"no Vibe engine responding at {self.url}/health")
        else:
            self._spawn()
        self._load_model()

    def _spawn(self) -> None:
        exe = find_vibe_server(self.binary)
        if not exe:
            if not self.auto_install:
                die("vibe-server not found. Install Vibe or pass --vibe-bin.")
            log("vibe-server not found, installing ...")
            exe = install_vibe_server()
        log(f"starting Vibe engine: {exe}")
        env = dict(os.environ)
        if self.ffmpeg:
            env.setdefault("VIBE_SERVER_FFMPEG_PATH", self.ffmpeg)
            # vibe-server looks for ffmpeg on PATH first, so put ours there too.
            env["PATH"] = os.pathsep.join([str(Path(self.ffmpeg).parent), env.get("PATH", "")])
        self.proc = subprocess.Popen(
            [exe, "serve", "--host", "127.0.0.1", "--port", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        threading.Thread(target=self._drain, args=(self.proc.stderr,), daemon=True).start()
        deadline = time.time() + 120
        port = None
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    die(f"vibe-server exited with code {self.proc.returncode}")
                continue
            try:
                msg = json.loads(line)
                if msg.get("status") == "ready":
                    port = msg["port"]
                    break
            except (json.JSONDecodeError, AttributeError):
                pass
        if not port:
            die("vibe-server did not become ready in time")
        threading.Thread(target=self._drain, args=(self.proc.stdout,), daemon=True).start()
        self.url = f"http://127.0.0.1:{port}"
        log(f"Vibe engine is running at {self.url}")

    @staticmethod
    def _drain(stream) -> None:
        for line in stream:
            if os.environ.get("YTSCRIBE_DEBUG"):
                log(f"vibe: {line.rstrip()}")

    def _load_model(self) -> None:
        try:
            status, info = http_json("GET", f"{self.url}/ready")
        except urllib.error.HTTPError:
            status, info = 503, None
        if status == 200 and info and Path(str(info.get("model", ""))).name == self.model.name:
            log(f"model already loaded: {info.get('model')}")
            return
        log(f"loading model {self.model} ...")
        body = {"path": str(self.model)}
        if self.gpu_device is not None:
            body["gpu_device"] = self.gpu_device
        if self.no_gpu:
            body["no_gpu"] = True
        try:
            http_json("POST", f"{self.url}/v1/models/load", body, timeout=600)
        except urllib.error.HTTPError as exc:
            die(f"model load failed: HTTP {exc.code} {exc.read().decode(errors='replace')}")

    def transcribe(self, audio: Path, language: str, prompt: str | None) -> dict:
        fields = {"response_format": "verbose_json", "prompt": prompt}
        if language == "auto":
            fields["detect_language"] = "true"
        else:
            fields["language"] = language
        body, ctype = multipart(fields, "file", audio)
        for attempt in range(60):
            req = urllib.request.Request(f"{self.url}/v1/audio/transcriptions", data=body,
                                         headers={"Content-Type": ctype}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=6 * 3600) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if exc.code == 429:  # engine busy with another job
                    time.sleep(10)
                    continue
                raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
        raise RuntimeError("Vibe engine stayed busy for 10 minutes")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.proc:
            for stream in (self.proc.stdout, self.proc.stderr):
                if stream:
                    stream.close()


# --------------------------------------------------------------------------- #
# YouTube
# --------------------------------------------------------------------------- #
def expand_inputs(ytdlp: list[str], inputs: list[str], cookies: str | None) -> list[dict]:
    """Turns URLs / playlists / channels into a flat list of {id, url, title}."""
    videos, seen = [], set()
    for src in inputs:
        cmd = ytdlp + ["--flat-playlist", "--ignore-errors", "--no-warnings",
                       "--print", "%(id)s\t%(url)s\t%(webpage_url)s\t%(title)s", src]
        if cookies:
            cmd[len(ytdlp):len(ytdlp)] = ["--cookies-from-browser", cookies]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0 and not out.stdout.strip():
            log(f"could not resolve {src}: {out.stderr.strip()[-300:]}")
            continue
        for line in out.stdout.splitlines():
            vid, url, page, title = (line.split("\t") + ["", "", "", ""])[:4]
            if not vid or vid in seen:
                continue
            seen.add(vid)
            link = page if page.startswith("http") else url if url.startswith("http") \
                else f"https://www.youtube.com/watch?v={vid}"
            videos.append({"id": vid, "url": link, "title": title})
    return videos


def download_audio(ytdlp: list[str], url: str, workdir: Path, cookies: str | None,
                   ffmpeg: str | None = None) -> tuple[Path, dict]:
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = ytdlp + ["--no-playlist", "--no-warnings", "--quiet", "--no-progress",
                   "-f", "bestaudio/best", "-x", "--audio-format", "wav",
                   "--postprocessor-args", "ExtractAudio:-ar 16000 -ac 1",
                   "--write-info-json", "-o", str(workdir / "%(id)s.%(ext)s"),
                   "--print", "after_move:filepath", url]
    if cookies:
        cmd[len(ytdlp):len(ytdlp)] = ["--cookies-from-browser", cookies]
    if ffmpeg:
        cmd[len(ytdlp):len(ytdlp)] = ["--ffmpeg-location", ffmpeg]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {out.stderr.strip()[-500:]}")
    path = Path(out.stdout.strip().splitlines()[-1])
    info = {}
    info_file = path.with_suffix(".info.json")
    if info_file.is_file():
        full = json.loads(info_file.read_text(encoding="utf-8"))
        info = {k: full.get(k) for k in ("id", "title", "channel", "uploader", "upload_date",
                                         "duration", "webpage_url", "language", "description")}
        info_file.unlink()
    return path, info


# --------------------------------------------------------------------------- #
# Transcript formatting
# --------------------------------------------------------------------------- #
def ts(sec: float, srt: bool = False) -> str:
    ms = int(round(sec * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}" if srt else f"{h:02}:{m:02}:{s:02}"


def to_srt(segments: list[dict]) -> str:
    out = []
    for i, seg in enumerate((s for s in segments if s["text"].strip()), 1):
        out.append(f"{i}\n{ts(seg['start'], True)} --> {ts(seg['end'], True)}\n{seg['text'].strip()}\n")
    return "\n".join(out)


def to_timestamped(segments: list[dict]) -> str:
    return "\n".join(f"[{ts(s['start'])}] {s['text'].strip()}" for s in segments if s["text"].strip())


def safe_name(text: str, limit: int = 80) -> str:
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", text or "").strip()
    return re.sub(r"\s+", " ", text)[:limit].rstrip(" .") or "untitled"


def guess_language(text: str) -> str:
    cyr = len(re.findall(r"[Ѐ-ӿ]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    return "ru" if cyr > lat else "en"


# --------------------------------------------------------------------------- #
# Summarizers
# --------------------------------------------------------------------------- #
class Summarizer:
    def __init__(self, backend: str, model: str | None, base_url: str | None,
                 max_chars: int, temperature: float):
        self.backend = backend
        self.model = model
        self.base_url = base_url
        self.max_chars = max_chars
        self.temperature = temperature

    @classmethod
    def auto(cls, args) -> "Summarizer | None":
        backend = args.summarizer
        if backend == "auto":
            if os.environ.get("ANTHROPIC_API_KEY"):
                backend = "anthropic"
            elif os.environ.get("OPENAI_API_KEY") or args.llm_url:
                backend = "openai"
            elif http_ok((args.llm_url or "http://localhost:11434") + "/api/tags"):
                backend = "ollama"
            else:
                log("no summarizer available (set ANTHROPIC_API_KEY / OPENAI_API_KEY or run Ollama); "
                    "transcripts will be saved without summaries")
                return None
        if backend == "none":
            return None
        defaults = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini", "ollama": "qwen2.5:7b"}
        urls = {"anthropic": "https://api.anthropic.com", "openai": "https://api.openai.com/v1",
                "ollama": "http://localhost:11434"}
        s = cls(backend, args.llm_model or defaults[backend], (args.llm_url or urls[backend]).rstrip("/"),
                args.chunk_chars, args.temperature)
        log(f"summarizer: {backend} ({s.model})")
        return s

    def _chat(self, system: str, user: str) -> str:
        if self.backend == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY") or die("ANTHROPIC_API_KEY is not set")
            _, r = http_json("POST", f"{self.base_url}/v1/messages", {
                "model": self.model, "max_tokens": 8000, "temperature": self.temperature,
                "system": system, "messages": [{"role": "user", "content": user}],
            }, headers={"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=600)
            return "".join(b.get("text", "") for b in r["content"] if b.get("type") == "text").strip()
        if self.backend == "openai":
            headers = {}
            if key := os.environ.get("OPENAI_API_KEY"):
                headers["Authorization"] = f"Bearer {key}"
            _, r = http_json("POST", f"{self.base_url}/chat/completions", {
                "model": self.model, "temperature": self.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }, headers=headers, timeout=600)
            return r["choices"][0]["message"]["content"].strip()
        # ollama
        _, r = http_json("POST", f"{self.base_url}/api/chat", {
            "model": self.model, "stream": False, "options": {"temperature": self.temperature},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }, timeout=1800)
        return r["message"]["content"].strip()

    def summarize(self, transcript: str, instruction: str, language: str, title: str) -> str:
        system = SYSTEM_PROMPT.format(language=LANG_NAMES.get(language, language))
        chunks = split_text(transcript, self.max_chars)
        if len(chunks) == 1:
            return self._chat(system, f"{instruction}\n\nVideo title: {title}\n\n<transcript>\n{transcript}\n</transcript>")
        # Map-reduce for transcripts longer than the model's comfortable context.
        notes = []
        for i, chunk in enumerate(chunks, 1):
            log(f"  summarizing part {i}/{len(chunks)}")
            notes.append(self._chat(system, (
                f"This is part {i} of {len(chunks)} of the transcript of '{title}'. Write thorough "
                f"notes of everything important in this part, keeping timestamps.\n\n"
                f"<transcript>\n{chunk}\n</transcript>")))
        joined = "\n\n".join(f"## Part {i}\n{n}" for i, n in enumerate(notes, 1))
        return self._chat(system, f"{instruction}\n\nVideo title: {title}\n\nThe transcript was too long, "
                                  f"so here are notes made from its consecutive parts:\n\n<notes>\n{joined}\n</notes>")


def split_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks, cur, size = [], [], 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > limit and cur:
            chunks.append("".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line)
    if cur:
        chunks.append("".join(cur))
    return chunks


def build_instruction(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    return STYLES[args.style]


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog=APP, formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Download YouTube videos, transcribe them locally with Vibe and summarize them.",
        epilog="examples:\n"
               "  ytscribe.py https://youtu.be/VIDEO_ID\n"
               "  ytscribe.py -l ru --style detailed URL1 URL2\n"
               "  ytscribe.py -b urls.txt --style chapters -o ~/notes\n"
               "  ytscribe.py 'https://www.youtube.com/playlist?list=...' --limit 10\n"
               "  ytscribe.py URL --prompt 'List every book recommended in the video'\n"
               "  ytscribe.py URL --summarizer ollama --llm-model llama3.1:8b\n"
               f"\nstyles: {', '.join(STYLES)}")
    p.add_argument("urls", nargs="*", help="video, playlist or channel URLs (or video IDs)")
    p.add_argument("-b", "--batch-file", help="file with one URL per line (# comments allowed)")
    p.add_argument("-o", "--output", default="transcripts", help="output directory (default: ./transcripts)")
    p.add_argument("-l", "--language", default="auto", choices=["auto", "en", "ru"],
                   help="spoken language of the videos (default: auto-detect)")
    p.add_argument("--limit", type=int, help="process at most N videos")
    p.add_argument("--force", action="store_true", help="redo videos that already have a transcript")
    p.add_argument("--keep-audio", action="store_true", help="keep the downloaded WAV files")
    p.add_argument("--cookies-from-browser", metavar="BROWSER",
                   help="pass browser cookies to yt-dlp (e.g. chrome, firefox) for age/region locked videos")

    g = p.add_argument_group("Vibe engine")
    g.add_argument("-m", "--model", default="large-v3-turbo",
                   help=f"whisper model name ({', '.join(WHISPER_MODELS)}) or path to a ggml .bin "
                        "(default: large-v3-turbo)")
    g.add_argument("--vibe-url", help="use an already running vibe-server, e.g. http://127.0.0.1:3022")
    g.add_argument("--vibe-bin", help="path to the vibe-server binary")
    g.add_argument("--gpu-device", type=int, help="GPU index (see `vibe-server devices`)")
    g.add_argument("--no-gpu", action="store_true", help="run whisper on the CPU")
    g.add_argument("--initial-prompt", help="whisper prompt: names/terms to improve recognition")
    g.add_argument("--no-install", action="store_true",
                   help="never download/install yt-dlp, vibe-server or models; fail instead")

    s = p.add_argument_group("summary")
    s.add_argument("-s", "--style", default="bullets", choices=list(STYLES), help="summary preset (default: bullets)")
    s.add_argument("-p", "--prompt", help="your own summary instruction (overrides --style)")
    s.add_argument("--prompt-file", help="read the summary instruction from a file")
    s.add_argument("--summary-language", choices=["same", "en", "ru"], default="same",
                   help="language of the summary (default: same as the video)")
    s.add_argument("--summarizer", default="auto", choices=["auto", "anthropic", "openai", "ollama", "none"],
                   help="auto: Claude if ANTHROPIC_API_KEY, OpenAI-compatible if OPENAI_API_KEY/--llm-url, "
                        "else local Ollama, else none")
    s.add_argument("--llm-model", help="model for the summarizer")
    s.add_argument("--llm-url", help="base URL for the summarizer (e.g. http://localhost:1234/v1 for LM Studio)")
    s.add_argument("--chunk-chars", type=int, default=120_000,
                   help="split transcripts longer than this into parts (lower it for small local models)")
    s.add_argument("--temperature", type=float, default=0.3)
    s.add_argument("--combined", action="store_true",
                   help="also write one combined summary across all processed videos")
    s.add_argument("--resummarize", action="store_true",
                   help="only (re)write summaries for existing transcripts in the output dir")
    return p.parse_args(argv)


def collect_inputs(args) -> list[str]:
    items = list(args.urls)
    if args.batch_file:
        for line in Path(args.batch_file).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                items.append(line)
    return [u if re.match(r"https?://", u) else f"https://www.youtube.com/watch?v={u}" for u in items]


def write_summary(summarizer: Summarizer, instruction: str, args, folder: Path, meta: dict) -> None:
    transcript = (folder / "transcript_timestamped.txt").read_text(encoding="utf-8")
    lang = meta.get("language") or guess_language(transcript)
    out_lang = lang if args.summary_language == "same" else args.summary_language
    title = meta.get("title") or folder.name
    summary = summarizer.summarize(transcript, instruction, out_lang, title)
    header = (f"# {title}\n\n"
              f"- URL: {meta.get('url', '')}\n- Channel: {meta.get('channel') or ''}\n"
              f"- Language: {lang}\n- Summary style: {args.prompt and 'custom' or args.style}\n\n---\n\n")
    (folder / "summary.md").write_text(header + summary + "\n", encoding="utf-8")
    meta["summary_language"] = out_lang


def main(argv=None) -> int:
    args = parse_args(argv)
    out_root = Path(args.output).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    instruction = build_instruction(args)
    summarizer = Summarizer.auto(args)

    if args.resummarize:
        if not summarizer:
            die("--resummarize needs a summarizer")
        done = 0
        for meta_file in sorted(out_root.glob("*/meta.json")):
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            log(f"summarizing {meta.get('title')}")
            write_summary(summarizer, instruction, args, meta_file.parent, meta)
            meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            done += 1
        log(f"rewrote {done} summaries")
        return 0

    inputs = collect_inputs(args)
    if not inputs:
        die("give at least one URL or --batch-file (see --help)")

    auto_install = not args.no_install
    ffmpeg = ensure_ffmpeg(auto_install)
    ytdlp = ensure_ytdlp(auto_install)
    model = ensure_model(args.model, auto_install)
    engine = VibeEngine(args.vibe_url, args.vibe_bin, model, auto_install, args.gpu_device, args.no_gpu, ffmpeg)
    keep_awake()

    videos = expand_inputs(ytdlp, inputs, args.cookies_from_browser)
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        die("no videos found")
    log(f"{len(videos)} video(s) to process")

    engine.start()
    audio_dir = out_root / ".audio"
    results, failures = [], []
    try:
        for n, video in enumerate(videos, 1):
            existing = next(out_root.glob(f"*[[]{video['id']}[]]/meta.json"), None)
            if existing and not args.force:
                log(f"[{n}/{len(videos)}] skip (already done): {existing.parent.name}")
                results.append(existing.parent)
                continue
            try:
                log(f"[{n}/{len(videos)}] downloading {video['url']}")
                audio, info = download_audio(ytdlp, video["url"], audio_dir, args.cookies_from_browser, ffmpeg)
                title = info.get("title") or video["title"] or video["id"]
                folder = out_root / f"{safe_name(title)} [{video['id']}]"
                folder.mkdir(parents=True, exist_ok=True)

                log(f"[{n}/{len(videos)}] transcribing '{title}' ({args.language})")
                t0 = time.time()
                result = engine.transcribe(audio, args.language, args.initial_prompt)
                segments = result.get("segments") or []
                text = (result.get("text") or "").strip()
                lang = args.language if args.language != "auto" else guess_language(text)
                (folder / "transcript.txt").write_text(text + "\n", encoding="utf-8")
                (folder / "transcript_timestamped.txt").write_text(to_timestamped(segments) or text, encoding="utf-8")
                (folder / "transcript.srt").write_text(to_srt(segments), encoding="utf-8")
                (folder / "transcript.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                log(f"  transcribed in {time.time() - t0:.0f}s, {len(text)} chars")

                meta = {**info, "id": video["id"], "url": info.get("webpage_url") or video["url"],
                        "title": title, "language": lang, "model": model.name,
                        "transcribed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                meta.pop("webpage_url", None)
                if summarizer:
                    log(f"[{n}/{len(videos)}] summarizing")
                    try:
                        write_summary(summarizer, instruction, args, folder, meta)
                    except Exception as exc:  # keep the transcript even if the LLM fails
                        log(f"  summary failed: {exc}")
                (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                if not args.keep_audio:
                    audio.unlink(missing_ok=True)
                results.append(folder)
                log(f"  saved -> {folder}")
            except Exception as exc:
                log(f"[{n}/{len(videos)}] FAILED {video['url']}: {exc}")
                failures.append((video["url"], str(exc)))
    finally:
        engine.stop()
        if not args.keep_audio and audio_dir.exists() and not any(audio_dir.iterdir()):
            audio_dir.rmdir()

    write_index(out_root)
    if args.combined and summarizer and results:
        combine(summarizer, instruction, args, out_root, results)
    log(f"done: {len(results)} ok, {len(failures)} failed. Output: {out_root.resolve()}")
    for url, err in failures:
        log(f"  failed: {url} — {err[:200]}")
    return 1 if failures else 0


def keep_awake() -> None:
    """On macOS, stop the Mac from idle-sleeping while long downloads/transcriptions run."""
    if IS_MAC and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write_index(out_root: Path) -> None:
    rows = ["# Transcripts\n", "| Title | Language | Duration | Summary | Transcript |", "|---|---|---|---|---|"]
    for meta_file in sorted(out_root.glob("*/meta.json")):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        d = meta_file.parent.name
        link = d.replace(" ", "%20").replace("[", "%5B").replace("]", "%5D")
        dur = ts(meta["duration"]) if meta.get("duration") else ""
        summ = f"[summary]({link}/summary.md)" if (meta_file.parent / "summary.md").exists() else ""
        title = (meta.get("title") or d).replace("|", "\\|")
        rows.append(f"| [{title}]({meta.get('url', '')}) | {meta.get('language', '')} | {dur} | {summ} | "
                    f"[txt]({link}/transcript.txt) |")
    (out_root / "index.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def combine(summarizer: Summarizer, instruction: str, args, out_root: Path, folders: list[Path]) -> None:
    parts = []
    for f in folders:
        s = f / "summary.md"
        if s.exists():
            parts.append(s.read_text(encoding="utf-8"))
    if not parts:
        return
    lang = args.summary_language if args.summary_language != "same" else guess_language("".join(parts))
    log("writing combined summary")
    system = SYSTEM_PROMPT.format(language=LANG_NAMES[lang])
    text = summarizer._chat(system, (
        f"Below are summaries of {len(parts)} videos. {instruction}\nCombine them into one overview: "
        "common themes first, then what is specific to each video (cite the video title).\n\n"
        + "\n\n=====\n\n".join(parts)))
    (out_root / "combined_summary.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("interrupted")
        sys.exit(130)
