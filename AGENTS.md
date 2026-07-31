# Kokoro TTS Server — Agent Context

## Project Identity

- **Name:** Kokoro TTS Server
- **Path:** `~/dev/kokoro/`
- **Purpose:** Lightweight local TTS server via Docker, exposing a FastAPI REST endpoint backed by Kokoro-82M.
- **Stack:** Python 3.11, FastAPI, Uvicorn, Kokoro-82M, Docker Compose.
- **GPU:** Pinned to the host's NVIDIA GTX 1060 6GB (`GPU-ebd852dc-b885-2874-feb2-1b37c939588b`) via Docker device reservation. Torch is pinned to `2.7.1+cu126` because newer builds dropped Pascal (sm_61) kernels. The Tesla P100 on the host is left untouched. Automatic CPU fallback when VRAM is insufficient (see Constraints & Notes).

## Key Files

| File | Role |
|---|---|
| `docker-compose.yml` | Single service `tts`, maps host `8001` → container `8000`. Mounts `~/.cache/huggingface` (model cache) and `./output` (audio files). |
| `Dockerfile` | `python:3.11-slim` base. Pins `torch==2.7.1+cu126` (Pascal-compatible, from the PyTorch cu126 index) before installing `kokoro>=0.9.2`, `fastapi>=0.115.0`, `uvicorn[standard]>=0.34.0`, `soundfile`, `numpy`. |
| `server.py` | FastAPI app. Serves web UI, audio file listing, and TTS endpoint. Lazy-init `KPipeline`. |
| `static/index.html` | Single-page web UI with TTS form, inline player, file browser with play/download/share. Mobile-responsive. |
| `.gitignore` | Ignores `output/` and `*.wav` |

## Architecture

```
Browser ──GET /────────────────────> FastAPI ──> index.html (web UI)
Browser ──GET /media/list──────────> FastAPI ──> JSON file list (audio + video)
Browser ──GET /media/{file}────────> FastAPI ──> media file (correct MIME)
Browser ──POST /v1/audio/speech────> FastAPI ──> KPipeline ──> WAV response
             {"text","voice","speed"}           │
                                              Downloads model
                                              on first call
                                              (~300 MB from HF)
```

- `KPipeline` is initialised once (singleton) and reused across requests.
- Pipeline uses `lang_code='a' (American English).
- Audio output: 24 kHz, 16-bit mono WAV.
- Generated files written to `/app/output/` (mounted to `./output/` on host).
- `/media/list` detects `.wav`, `.mp3`, `.ogg`, `.m4a`, `.flac`, `.mp4`, `.webm`, `.mov`, `.mkv`.
- `/audio/list` is kept for backward compat (`.wav` only).

## API Contract

### `GET /`

Serves the web UI (`static/index.html`).

### `GET /health`

```
Response: 200 {"status": "ok"}
```

### `POST /v1/audio/speech`

```
Request:
{
  "text": string       (required)
  "voice": string      (default "af_heart")
  "speed": float       (default 1.0)
}

Headers (response):
  X-Audio-Filename: <uuid>.wav

Response: 200 audio/wav (binary) | 500 {"detail": "..."}
```

### `GET /media/list`

```
Response: 200
[
  {
    "id": "a1b2c3d4e5f6",
    "filename": "a1b2c3d4e5f6.wav",
    "size": 345000,
    "created": 1745000000.0,
    "type": "audio",
    "mime": "audio/wav"
  }
]
```

Supported: `wav`, `mp3`, `ogg`, `m4a`, `flac`, `mp4`, `webm`, `mov`, `mkv`. Sorted newest-first.

### `GET /audio/list`

Backward-compatible — same format, `.wav` only.

### `GET /media/{filename}`

### `GET /audio/{filename}`

Serves any recognised media file with correct MIME type. Rejects path traversal.

### `GET /`

Serves the web UI (`static/index.html`). The UI renders audio files with `<audio>` and video files with `<video>`.

## Dependency Graph (pip)

```
torch == 2.7.1+cu126   # from https://download.pytorch.org/whl/cu126 (Pascal sm_61 support)
fastapi >= 0.115.0
uvicorn[standard] >= 0.34.0
kokoro >= 0.9.2
soundfile >= 0.13.0
numpy >= 1.26.0
```

Kokoro pulls in transformers, spaCy (en_core_web_sm), and misaki. Torch bundles its own CUDA 12.6 runtime, so no additional system deps are needed — but the host must have the NVIDIA driver + container toolkit installed.

## Common Tasks

### Rebuild after code changes

```bash
cd ~/dev/kokoro && docker compose up -d --build
```

### View logs

```bash
cd ~/dev/kokoro && docker compose logs -f
```

### Shell into container

```bash
docker exec -it kokoro-tts bash
```

### Test from host (API)

```bash
curl -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from agent.", "voice": "af_bella"}' \
  --output /tmp/test.wav
```

### Test from host (web UI)

Open `http://localhost:8001/` in a browser. Also works from any device on the same LAN — replace `localhost` with the host's LAN IP.

## Voice Reference

Kokoro-82M has ~150 voices. Common ones:

| ID | Character | Style |
|---|---|---|
| `af_heart` | Female | Warm, friendly |
| `af_bella` | Female | Bright, clear |
| `af_nicole` | Female | Professional |
| `af_sarah` | Female | Soft |
| `af_sky` | Female | Calm |
| `am_adam` | Male | Neutral |
| `am_michael` | Male | Deep, resonant |
| `am_fenrir` | Male | Low, gruff |
| `am_puck` | Male | Light, jovial |

Voice IDs follow the pattern `{a}{m/f}_{name}` where `a` = American English, `m`/`f` = male/female.

## Constraints & Notes

- **Lazy init:** First request takes ~10–30s (model download + spaCy model install). Subsequent requests are < 1s.
- **No auth:** Server has no authentication. Use a reverse proxy (nginx/Caddy) if exposing outside localhost or the internet.
- **GPU wired:** The container requests the host's GTX 1060 (Pascal sm_61) via `deploy.resources.reservations.devices` pinned by UUID. Torch is pinned to `2.7.1+cu126` — newer PyPI torch builds (cu13x) dropped sm_61 kernels. First request still takes ~10–30s for model load; subsequent requests are faster on GPU.
- **CPU fallback (automatic):** `server.py` guards against missing/low VRAM in three ways — (1) `_pick_device()` checks `torch.cuda.mem_get_info()` and uses CPU if free VRAM < `MIN_FREE_VRAM` (1.5 GiB); (2) `_build_pipeline()` retries on CPU if CUDA init raises; (3) `_generate_audio()` catches mid-generation CUDA OOM, rebuilds the pipeline on CPU, and retries the request once (subsequent requests stay on CPU to avoid thrashing).
- **Single worker:** Uvicorn runs without `--workers` to keep things simple.
- **Host port 8001** is used because something else is on 8000.
- **Model cache persisted** via `~/.cache/huggingface` volume mount.
- **Audio output persisted** via `./output` volume mount — files survive container rebuilds and are accessible from the host filesystem.
- **Audio files are gitignored** (`output/` and `*.wav`).

## Future Improvements (if needed)

- [x] Mount HuggingFace cache volume to avoid re-downloading weights
- [ ] Add support for SSML or phoneme-level control
- [ ] Expose available voices via a `GET /voices` endpoint
- [ ] Add streaming (chunked transfer) for long-form TTS
- [x] Add a simple web UI (Gradio or custom HTML)
