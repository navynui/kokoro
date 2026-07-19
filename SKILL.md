# Kokoro TTS Server — Agent Skill

Generate high-quality speech from text using Kokoro-82M via a local Dockerized FastAPI server.

## Quick Reference

| Action | Command |
|---|---|
| **Start server** | `docker compose -f ~/dev/kokoro/docker-compose.yml up -d` |
| **Rebuild & start** | `docker compose -f ~/dev/kokoro/docker-compose.yml up -d --build` |
| **Stop server** | `docker compose -f ~/dev/kokoro/docker-compose.yml down` |
| **View logs** | `docker compose -f ~/dev/kokoro/docker-compose.yml logs -f` |
| **Web UI** | `http://localhost:8001/` |
| **Swagger UI** | `http://localhost:8001/docs` |

## Background

Kokoro-82M is an 82M-parameter TTS model (~300 MB) that runs on CPU and delivers near-commercial quality speech. This project wraps it in a FastAPI server inside Docker. The server also serves a mobile-responsive web UI for generating, playing, browsing, downloading, and sharing audio files.

- **Server URL:** `http://localhost:8001` (or host LAN IP from other devices)
- **Audio format:** 24 kHz, 16-bit mono WAV
- **Model cache:** Persisted at `~/.cache/huggingface` (volume mount)
- **Output dir:** `~/dev/kokoro/output/` (volume mount, gitignored)

## API Endpoints

### `POST /v1/audio/speech`

Generate speech from text.

**Request:**

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
| `voice` | string | `"af_heart"` | Voice ID |
| `speed` | number | `1.0` | Speaking speed (0.5–2.0) |

**Response:** `200 audio/wav` (binary) with header `X-Audio-Filename: <uuid>.wav`.

**Error:** `500 {"detail": "..."}`

### `POST /v1/audio/speech` (curl)

```bash
curl -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice": "af_bella"}' \
  --output speech.wav
```

### `GET /health`

Returns `{"status": "ok"}`.

### `GET /audio/list`

Returns a JSON array of generated WAV files, newest first:

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

Serves a WAV file for playback or download.

### `GET /`

Serves the web UI (`static/index.html`).

## Voice Reference

Common voices (150+ available):

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

Voice IDs follow the pattern `{a}{m/f}_{name}` where `a` = American English, `m`/`f` = male/female.

## Agent Workflows

### 1. Generate a WAV file

```bash
curl -s -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg t "$TEXT" '{text: $t, voice: "af_bella", speed: 1.0}')" \
  --output /tmp/output.wav
```

After generation, the file also appears in `~/dev/kokoro/output/` with a UUID filename. The response header `X-Audio-Filename` contains the filename.

### 2. List all generated files

```bash
curl -s http://localhost:8001/audio/list | python3 -m json.tool
```

### 3. Use generated audio in a HyperFrames video

Generated WAV files are accessible at `http://localhost:8001/audio/{filename}`. Use this URL as the `src` in HyperFrames audio blocks:

```html
<div class="clip" data-duration="5">
  <audio src="http://kokoro-tts:8001/audio/a1b2c3d4e5f6.wav" data-media-play></audio>
</div>
```

> **Note:** If HyperFrames runs in a separate Docker container on the same Docker network, use the service name `kokoro-tts` as the hostname instead of `localhost`.

### 4. Generate speech for a script / voiceover

```python
import requests, json

def tts(text, voice="af_bella", speed=1.0, output_path="speech.wav"):
    resp = requests.post(
        "http://localhost:8001/v1/audio/speech",
        json={"text": text, "voice": voice, "speed": speed},
    )
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)
    filename = resp.headers.get("X-Audio-Filename", "unknown")
    return output_path, filename
```

### 5. Get the public URL of a generated file

```python
import requests

files = requests.get("http://localhost:8001/audio/list").json()
if files:
    newest = files[0]["filename"]
    url = f"http://localhost:8001/audio/{newest}"
    # Use this URL in a HyperFrames composition or share link
```

## Integration with HyperFrames

When building a HyperFrames video composition that needs a voiceover:

1. **Generate** the narration via the TTS endpoint
2. **Get the URL** from `/audio/list` — the newest file is the one just created
3. **Reference it** in the composition HTML using `<audio src="http://kokoro-tts:8001/audio/{filename}" data-media-play>`
4. **Render** the HyperFrames composition — the audio will play in sync

If the Kokoro server and HyperFrames renderer are on the same Docker network, use the container name `kokoro-tts` as the hostname. If HyperFrames runs on the host, use `localhost:8001`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| First request takes 10–30s | Model downloading (~300 MB) | Normal — subsequent requests are < 1s |
| `index.html not found` | Static files not in image | Rebuild: `docker compose up -d --build` |
| Port conflict | Port 8001 already in use | Check with `ss -tlnp \| grep 8001` |
| Permission denied on `output/` | Volume mount ownership | `sudo chown -R $USER:$USER ~/dev/kokoro/output` |
| Web UI not accessible from phone | Firewall | Allow port 8001 on host firewall |

## Project State

- **Path:** `~/dev/kokoro/`
- **Key files:** `server.py`, `static/index.html`, `docker-compose.yml`, `Dockerfile`, `.gitignore`, `README.md`, `AGENTS.md`, `output/` (gitignored)
- **Default branch:** `main`
- **Remote:** `git@github.com:navynui/kokoro.git`
