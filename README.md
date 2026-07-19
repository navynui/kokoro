# Kokoro TTS Server

A lightweight, production-ready local TTS server using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — a fast, high-quality text-to-speech model that runs comfortably on CPU.

Serves a REST API compatible with typical `/v1/audio/speech` patterns. Containerised with Docker for easy deployment.

## Quick Start

```bash
cd ~/dev/kokoro
docker compose up -d
```

Server is live at **`http://localhost:8001`** — open it in a browser on any device on your LAN for the web UI.

- **Web UI** — `http://localhost:8001/` — generate speech, play inline, browse/download/share files
- **Swagger UI** — `http://localhost:8001/docs` — interactive API docs

On first request the container downloads the ~300 MB model weights automatically.

## API

### `POST /v1/audio/speech`

Generate speech from text.

**Request body:**

```json
{
  "text": "Hello world!",
  "voice": "af_bella",
  "speed": 1.0
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | (required) | Text to synthesise |
| `voice` | string | `"af_heart"` | Voice ID (see below) |
| `speed` | number | `1.0` | Speaking speed |

**Response:** WAV audio (`audio/wav`), 24 kHz, 16-bit mono.

### `GET /health`

Health check — returns `{"status": "ok"}`.

### `GET /`

Web UI — a mobile-friendly page for generating speech, playing audio inline, browsing the file library, and downloading or sharing files.

### `GET /audio/list`

Returns a JSON array of generated WAV files (newest first):

```json
[
  {
    "id": "a1b2c3d4e5f6",
    "filename": "a1b2c3d4e5f6.wav",
    "size": 345000,
    "created": 1745000000.0
  }
]
```

### `GET /audio/{filename}`

Serve a generated WAV file for playback or download.

## Voices

| ID | Description |
|---|---|
| `af_heart` | Warm female |
| `af_bella` | Bright female |
| `af_nicole` | Clear female |
| `af_sarah` | Soft female |
| `af_sky` | Calm female |
| `am_adam` | Neutral male |
| `am_michael` | Deep male |
| `am_fenrir` | Low male |
| `am_puck` | Light male |

Many more are available — Kokoro ships with ~150 built-in voices. Browse the full list via the [Kokoro voice catalog](https://huggingface.co/hexgrad/Kokoro-82M).

## Testing

### Via the web UI

Open `http://localhost:8001/` in any browser (desktop or phone). Type text, pick a voice, and click **Generate**. The audio plays inline and appears in the file list for download or sharing.

### Via curl

```bash
curl -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"text": "The quick brown fox jumps over the lazy dog.", "voice": "af_bella"}' \
  --output speech.wav

# Play it
aplay speech.wav          # Linux
afplay speech.wav         # macOS
start speech.wav          # Windows
```

### From your phone

1. Find your server's LAN IP: `ip addr show | grep 'inet '` (look for `192.168.x.x` or similar)
2. Open `http://<LAN_IP>:8001/` on your phone
3. Generate, play, and share audio directly

## Configuration

Edit `docker-compose.yml` to change the host port:

```yaml
ports:
  - "8001:8000"   # change 8001 to any available port
```

### Persistent Storage

| Mount | Host path | Container path | Purpose |
|---|---|---|
| HuggingFace cache | `~/.cache/huggingface` | `/root/.cache/huggingface` | Model weights (~300 MB) survive rebuilds |
| Audio output | `./output` | `/app/output` | Generated WAV files survive rebuilds, accessible from host |

To clear the model cache and force a fresh download:

```bash
rm -rf ~/.cache/huggingface/hub/hexgrad*
docker compose down && docker compose up -d
```

To clear generated audio:

```bash
rm -f output/*.wav
```

### GPU Acceleration

If you free up GPU headroom, uncomment the `deploy` block in `docker-compose.yml` to use the NVIDIA GPU. Kokoro will automatically pick it up — no code changes needed.

## Project Structure

```
~/dev/kokoro/
├── docker-compose.yml   # Service definition, port mapping, volume mounts
├── Dockerfile           # Container build (python:3.11-slim)
├── server.py            # FastAPI application
├── static/
│   └── index.html       # Web UI — TTS form, player, file browser
├── output/              # Generated WAV files (gitignored)
├── .gitignore
├── README.md            # This file
└── AGENTS.md            # AI assistant context
```

## Stop the Server

```bash
docker compose -f ~/dev/kokoro/docker-compose.yml down
```

## Why Kokoro?

- **82M parameters** (~300 MB) — tiny compared to Bark's ~1.2B
- **Runs on CPU** — real-time inference with no GPU required
- **Near-commercial quality** — competitive with cloud TTS
- **~50–100× faster than real-time** on modern CPUs
- **Simple Docker deployment** — no CUDA runtime needed
