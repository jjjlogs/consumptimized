# ytscribe

Download YouTube videos (one, many, playlists or channels) with **yt-dlp**, transcribe them
**locally** with the [Vibe](https://github.com/thewh1teagle/vibe) engine (`vibe-server`, whisper.cpp)
from English or Russian, save the transcripts and summarize them the way you ask.

Single file, Python 3.9+ standard library only. Works on **macOS** (Apple Silicon and Intel),
Linux and Windows.

## macOS

```bash
xcode-select --install          # gives you python3 + git, if you don't have them yet
brew install ffmpeg yt-dlp      # optional: the script installs them itself if missing
chmod +x ytscribe.py
./ytscribe.py https://youtu.be/VIDEO_ID
```

- Runs on the GPU through Metal on Apple Silicon (native `arm64` build, also when Python runs under Rosetta).
- If you have the **Vibe app** in `/Applications` or `~/Applications`, its bundled `vibe-server` and
  `ffmpeg` are used, and so are the models it already downloaded.
- Without Homebrew, a missing ffmpeg comes from Vibe's macOS bundle (`vibe-server-darwin-*-with-ffmpeg`),
  and a missing yt-dlp from pip or the standalone `yt-dlp_macos` binary.
- Files go to `~/Library/Application Support/ytscribe` (override with `YTSCRIBE_HOME`).
- A python.org Python without CA certificates still works: certifi or `/etc/ssl/cert.pem` is used.
- The Mac is kept awake (`caffeinate`) while a batch runs.

## What it checks and sets up for you

| Step | Behaviour |
|---|---|
| ffmpeg | must be installed (the script tells you how if it isn't) |
| yt-dlp | uses `yt-dlp` on PATH / `python -m yt_dlp`; otherwise installs it with pip, or downloads the standalone binary |
| Vibe engine | looks for `vibe-server` (`--vibe-bin`, `$VIBE_SERVER_BIN`, PATH, the app data `bin` folder, the Vibe desktop app's bundled one); downloads the latest `server-v*` release from GitHub if missing |
| running | starts `vibe-server serve` on a free port and waits for its `ready` line, or attaches to one you already run (`--vibe-url`) after checking `/health` |
| model | downloads the Whisper model (default `large-v3-turbo`, good for EN + RU) into the app data folder (`~/Library/Application Support/ytscribe/models` on macOS, `~/.local/share/ytscribe/models` on Linux), reusing the Vibe app's models if present, and loads it via `/v1/models/load`, unless it is already loaded |

Use `--no-install` to forbid any download/installation.

## Usage

```bash
# one video, language auto-detected, bullet summary
./ytscribe.py https://youtu.be/VIDEO_ID

# several Russian videos, detailed summary written in English
./ytscribe.py -l ru --style detailed --summary-language en URL1 URL2

# bulk: file with one URL per line, a playlist, or a channel
./ytscribe.py -b urls.txt --style chapters -o ~/notes
./ytscribe.py "https://www.youtube.com/playlist?list=..." --limit 20 --combined

# your own summary instructions
./ytscribe.py URL --prompt "List every book and tool recommended, with timestamps"
./ytscribe.py URL --prompt-file my_summary_prompt.txt

# re-summarize everything already transcribed, in a different way
./ytscribe.py --resummarize --style study -o ~/notes
```

Summary presets (`--style`): `brief`, `bullets` (default), `detailed`, `chapters`, `takeaways`, `tldr`, `study`.

### Summarizer backends

`--summarizer auto` (default) picks, in order:

1. **Claude** if `ANTHROPIC_API_KEY` is set (default model `claude-sonnet-5-5`)
2. **OpenAI-compatible** if `OPENAI_API_KEY` or `--llm-url` is set (OpenAI, LM Studio `http://localhost:1234/v1`, llama.cpp server, ...)
3. **Ollama** if it is running on `localhost:11434` (default model `qwen2.5:7b`, which handles Russian well)
4. otherwise only transcripts are saved

Override with `--summarizer anthropic|openai|ollama|none` and `--llm-model`. Long transcripts are
summarized in parts and then merged (`--chunk-chars`, lower it for small local models).

## Output

```
transcripts/
  index.md                       table of all videos with links
  combined_summary.md            with --combined
  <Title> [<video id>]/
    transcript.txt               plain text
    transcript_timestamped.txt   [HH:MM:SS] per segment
    transcript.srt               subtitles
    transcript.json              raw Vibe verbose_json
    summary.md
    meta.json                    title, channel, url, duration, language, model ...
```

Videos that already have a `meta.json` are skipped on the next run (use `--force` to redo), so
interrupted bulk runs can just be restarted.

## Useful options

- `-m small|medium|large-v3|large-v3-turbo|/path/to/ggml-model.bin` choose the Whisper model
- `--initial-prompt "names, jargon"` help recognition of specific terms
- `--gpu-device N` / `--no-gpu` (list GPUs with `vibe-server devices`)
- `--cookies-from-browser firefox` for age-restricted or members-only videos
- `--keep-audio` keep the 16 kHz WAV files
- `YTSCRIBE_DEBUG=1` print the engine's log output

## Tests

```bash
python3 -m unittest discover tests
```

The tests run the engine startup, model loading and transcription against a fake `vibe-server`
(`tests/fake_vibe_server.py`) that has the same CLI handshake and HTTP routes.
