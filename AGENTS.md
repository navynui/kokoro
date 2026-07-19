# Kokoro TTS Server — Agent Context

## Project Identity

- **Name:** Kokoro TTS Server
- **Path:** `~/dev/kokoro/`
- **Purpose:** Lightweight local TTS server via Docker, exposing a FastAPI REST endpoint backed by Kokoro-82M.
- **Stack:** Python 3.11, FastAPI, Uvicorn, Kokoro-82M, Docker Compose.
- **GPU:** Tesla P100 available on host but **not used by this container** (VRAM saturated at ~95%). Kokoro runs on CPU comfortably.

## Key Files

| File | Role |
|---|---|
| `docker-compose.yml` | Single service `tts`, maps host `8001` → container `8000`. Mounts `~/.cache/huggingface` for model persistence. No GPU passthrough. |
| `Dockerfile` | `python:3.11-slim` base. Pins `kokoro>=0.9.2`, `fastapi>=0.115.0`, `uvicorn[standard]>=0.34.0`, `soundfile`, `numpy`. |
| `server.py` | FastAPI app. Lazy-init `KPipeline` (model downloads on first request). |

## Architecture

```
Client ──POST /v1/audio/speech──> FastAPI ──> KPipeline ──> WAV response
              {"text", "voice", "speed"}           │
                                              Downloads model
                                              on first call
                                              (~300 MB from HF)
```

- `KPipeline` is initialised once (singleton) and reused across requests.
- Pipeline uses `lang_code='a'` (American English).
- Audio output: 24 kHz, 16-bit mono WAV.
- Generated files written to `/tmp/tts_output/` and cleaned up by OS.

## API Contract

### `POST /v1/audio/speech`

```
Request:
{
  "text": string       (required)
  "voice": string      (default "af_heart")
  "speed": float       (default 1.0)
}

Response: 200 audio/wav (binary) | 500 {"detail": "..."}
```

### `GET /health`

```
Response: 200 {"status": "ok"}
```

## Dependency Graph (pip)

```
fastapi >= 0.115.0
uvicorn[standard] >= 0.34.0
kokoro >= 0.9.2
soundfile >= 0.13.0
numpy >= 1.26.0
```

Kokoro pulls in torch, transformers, spaCy (en_core_web_sm), and misaki. No additional system deps needed.

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

### Test from host

```bash
curl -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from agent.", "voice": "af_bella"}' \
  --output /tmp/test.wav
```

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
- **No auth:** Server has no authentication. Use a reverse proxy (nginx/Caddy) if exposing outside localhost.
- **GPU not wired:** Container intentionally avoids `--gpus` to keep the image small and avoid CUDA dependency. Kokoro is fast enough on CPU.
- **Single worker:** Uvicorn runs without `--workers` to keep things simple. For high throughput, add `--workers 4` to the Dockerfile CMD and ensure `KPipeline` re-init doesn't become a bottleneck.
- **Host port 8001** is used because something else is on 8000.
- **Model cache persisted** via `~/.cache/huggingface` volume mount — weights survive container rebuilds. First request still downloads (~300 MB), subsequent rebuilds reuse instantly.

## Future Improvements (if needed)

- [x] Mount HuggingFace cache volume to avoid re-downloading weights
- [ ] Add support for SSML or phoneme-level control
- [ ] Expose available voices via a `GET /voices` endpoint
- [ ] Add streaming (chunked transfer) for long-form TTS
- [ ] Add a simple web UI (Gradio or custom HTML)
