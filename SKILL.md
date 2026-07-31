# Kokoro TTS Server — Agent Skill

Generate speech from text using three local TTS engines — Kokoro-82M (GPU), Thai Vachana (PyThaiTTS), and Piper — via a local Dockerized FastAPI server.

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

The server wraps three local engines in a FastAPI server inside Docker: **Kokoro-82M** (GPU-accelerated on a GTX 1060, English, ~150 voices), **Thai Vachana** (PyThaiTTS VachanaTTS2, VITS-ONNX on CPU, 4 Thai voices), and **Piper** (ONNX on CPU, 15+ languages). All inference runs locally. The server also serves a mobile-responsive web UI with an engine/voice picker for generating, playing, browsing, downloading, and sharing audio files.

- **Server URL:** `http://localhost:8001` (or host LAN IP from other devices)
- **Audio format:** 16-bit mono WAV — 24 kHz (Kokoro), 22.05 kHz (Thai/Piper)
- **Model cache:** Persisted at `~/.cache/huggingface` (volume mount)
- **Voice models:** `~/dev/kokoro/piper_models/` and `~/dev/kokoro/pythai_voices/` (volume mounts, gitignored)
- **Output dir:** `~/dev/kokoro/output/` (volume mount, gitignored)

## API Endpoints

### `POST /v1/audio/speech`

Generate speech from text.

**Request:**

```json
{
  "text": "Hello world!",
  "voice": "af_bella",
  "speed": 1.0,
  "engine": "kokoro"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | (required) | Text to synthesise |
| `voice` | string | `"af_heart"` | Voice ID (see `/voices` per engine) |
| `speed` | number | `1.0` | Speaking speed (0.5–2.0) |
| `engine` | string | `"kokoro"` | `kokoro` \| `thai` \| `piper` |

**Response:** `200 audio/wav` (binary) with header `X-Audio-Filename: <uuid>.wav`. Sample rate is engine-dependent (24 kHz Kokoro, 22.05 kHz Thai/Piper).

**Error:** `500 {"detail": "..."}` (or `422` for an unknown engine)

**Engine quick reference:**

| Engine | Thai? | Offline? | Voices |
|---|---|---|---|
| `kokoro` (default) | ❌ | ✅ | ~150 (curated in `/voices`), e.g. `af_bella` |
| `thai` | ✅ | ✅ | `th_f_1`, `th_f_2`, `th_m_1`, `th_m_2` |
| `piper` | ❌ | ✅ | 15 curated, e.g. `en_US-lessac-medium` |

First use of each engine downloads its model (~60–300 MB); subsequent requests are fast and fully offline.

### `POST /v1/audio/speech` (curl)

```bash
curl -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice": "af_bella"}' \
  --output speech.wav
```

### `GET /health`

Returns `{"status": "ok"}`.

### `GET /voices`

Lists voices for an engine: `GET /voices?engine=kokoro|thai|piper`. Returns `[{"id", "name", "language"}]`; unknown engine → `422`.

### `GET /`

Serves the web UI (`static/index.html`). The UI shows **all** media (audio + video) in one unified file browser with inline play/download/share.

### `GET /media/list`

Returns all recognised media files (audio + video), newest first:

```json
[
  {
    "id": "a1b2c3d4e5f6",
    "filename": "a1b2c3d4e5f6.wav",
    "size": 345000,
    "created": 1745000000.0,
    "type": "audio",
    "mime": "audio/wav"
  },
  {
    "id": "video123",
    "filename": "output.mp4",
    "size": 2048000,
    "created": 1745000100.0,
    "type": "video",
    "mime": "video/mp4"
  }
]
```

Supported: `wav`, `mp3`, `ogg`, `m4a`, `flac`, `mp4`, `webm`, `mov`, `mkv`.

### `GET /audio/list`

Backward-compatible — returns only `.wav` files.

### `GET /media/{filename}`

### `GET /audio/{filename}`

Serve any media file with correct MIME type for playback or download.

## Voice Reference

### Kokoro (engine `kokoro`)

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

Voice IDs follow the pattern `{a}{m/f}_{name}` where `a` = American English, `m`/`f` = male/female. The full list is served by `GET /voices?engine=kokoro`.

### Thai (engine `thai`)

`th_f_1`, `th_f_2` (female), `th_m_1`, `th_m_2` (male). Text is preprocessed automatically (digits → Thai words, `ๆ` expansion).

### Piper (engine `piper`)

Curated multilingual voices, e.g. `en_US-lessac-medium`, `en_GB-alan-medium`, `de_DE-thorsten-medium`, `fr_FR-siwis-medium`, `es_ES-davefx-medium`, `it_IT-riccardo-x_low`, `pt_BR-faber-medium`, `ru_RU-irina-medium`, `uk_UA-ukrainian_ts-medium`, `vi_VN-vais1000-medium`, `ar_AR-omarsalim-medium`, `zh_CN-huayan-medium`. Full list: `GET /voices?engine=piper`.

## Agent Workflows

### 1. Generate a WAV file

```bash
curl -s -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg t "$TEXT" '{text: $t, voice: "af_bella", speed: 1.0, engine: "kokoro"}')" \
  --output /tmp/output.wav
```

After generation, the file also appears in `~/dev/kokoro/output/` with a UUID filename. The response header `X-Audio-Filename` contains the filename.

### 1b. Generate Thai speech (engine `thai`)

```bash
curl -s -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"text": "สวัสดีครับ", "voice": "th_f_1", "engine": "thai"}' \
  --output /tmp/thai.wav
```

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

def tts(text, voice="af_bella", speed=1.0, engine="kokoro", output_path="speech.wav"):
    resp = requests.post(
        "http://localhost:8001/v1/audio/speech",
        json={"text": text, "voice": voice, "speed": speed, "engine": engine},
    )
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)
    filename = resp.headers.get("X-Audio-Filename", "unknown")
    return output_path, filename
```

### 5. Get the public URL of the newest file

```python
import requests

files = requests.get("http://localhost:8001/media/list").json()
if files:
    newest = files[0]["filename"]
    url = f"http://localhost:8001/media/{newest}"
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
| First use of an engine takes 10–60s | Model downloading (Kokoro ~300 MB; Piper/Thai ~60 MB per voice) | Normal — subsequent requests are < 1s |
| `index.html not found` | Static files not in image | Rebuild: `docker compose up -d --build` |
| Port conflict | Port 8001 already in use | Check with `ss -tlnp \| grep 8001` |
| Permission denied on `output/` | Volume mount ownership | `sudo chown -R $USER:$USER ~/dev/kokoro/output` |
| Web UI not accessible from phone | Firewall | Allow port 8001 on host firewall |

## Project State

- **Path:** `~/dev/kokoro/`
- **Key files:** `server.py`, `static/index.html`, `docker-compose.yml`, `Dockerfile`, `.gitignore`, `README.md`, `AGENTS.md`, `SKILL.md`, `output/`, `piper_models/`, `pythai_voices/` (last three gitignored)
- **Default branch:** `main`
- **Remote:** `git@github.com:navynui/kokoro.git`
