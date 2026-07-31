# Kokoro TTS Server

A lightweight, production-ready local TTS server with **five local engines**: Kokoro-82M (GPU-accelerated on an NVIDIA GTX 1060 with automatic CPU fallback), Thai Vachana (PyThaiTTS, CPU/ONNX), Thai MMS (Meta, GPU), Thai Thonburian (F5-TTS Mega, GPU), and Piper (CPU/ONNX, 15+ languages). All inference runs locally — no cloud APIs.

Serves a REST API compatible with typical `/v1/audio/speech` patterns. Containerised with Docker for easy deployment.

## Quick Start

```bash
cd ~/dev/kokoro
docker compose up -d
```

Server is live at **`http://localhost:8001`** — open it in a browser on any device on your LAN for the web UI.

- **Web UI** — `http://localhost:8001/` — generate speech, play inline, browse/download/share files
- **Swagger UI** — `http://localhost:8001/docs` — interactive API docs

On first use each engine downloads its model automatically: Kokoro (~300 MB once), Piper (~60 MB per voice), Thai Vachana (~60 MB per voice). Subsequent requests are fast and fully offline.

## API

### `POST /v1/audio/speech`

Generate speech from text.

**Request body:**

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
| `speed` | number | `1.0` | Speaking speed |
| `engine` | string | `"kokoro"` | `kokoro` \| `thai` \| `piper` \| `mms` \| `f5` |

**Response:** WAV audio (`audio/wav`), 16-bit mono. Sample rate is engine-dependent (Kokoro 24 kHz, Thai/Piper 22.05 kHz).

### `GET /health`

Health check — returns `{"status": "ok"}`.

### `GET /voices`

List voices for an engine:

```
GET /voices?engine=kokoro | thai | piper | mms | f5
```

Returns a JSON array of `{"id", "name", "language"}` objects (language omitted for Kokoro).

### `GET /`

Web UI — a mobile-friendly page for generating speech, playing audio/video inline, browsing the media library, and downloading or sharing files.

### `GET /media/list`

Returns a JSON array of all generated media files (audio + video), newest first:

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

Supported formats: `wav`, `mp3`, `ogg`, `m4a`, `flac` (audio) and `mp4`, `webm`, `mov`, `mkv` (video).

### `GET /audio/list`

Backward-compatible — returns only `.wav` files (same format as above, without `type`/`mime` fields).

### `GET /media/{filename}`

### `GET /audio/{filename}`

Serve any media file with the correct MIME type for playback or download.

## Engines

| Engine | `engine` value | Runs on | Languages | Voices |
|---|---|---|---|---|
| **Kokoro-82M** | `kokoro` | GPU (GTX 1060), CPU fallback | English | ~150, curated list below |
| **Thai Vachana** (PyThaiTTS) | `thai` | CPU (ONNX) | Thai | `th_f_1`, `th_f_2`, `th_m_1`, `th_m_2` |
| **Thai MMS** (Meta) | `mms` | GPU | Thai | `facebook/mms-tts-tha` (16 kHz) |
| **Thai Thonburian** (F5-TTS Mega) | `f5` | GPU | Thai | `default` (auto-built ref voice, 24 kHz) |
| **Piper** | `piper` | CPU (ONNX) | 15+ (EN, DE, FR, ES, IT, PT, RU, UK, VI, AR, ZH, NL, PL, TR, …) | curated list in `/voices` |

The web UI exposes an **Engine** dropdown next to the voice picker; voice lists load from `/voices?engine=…`. Thai engines, best quality first: **Thonburian `f5`** (natural prosody, diffusion) → **MMS `mms`** (stable VITS) → **Vachana `thai`** (fast CPU).

**First-use downloads** (persisted, see Persistent Storage):
- Kokoro: model weights (~300 MB) via the HuggingFace cache
- Piper: `piper_models/<voice>.onnx` (~60 MB each) from `rhasspy/piper-voices`
- Thai: `pythai_voices/<voice>.onnx` (~60 MB each) from `VIZINTZOR/VachanaTTS`
- MMS: `facebook/mms-tts-tha` (~145 MB) via the HuggingFace cache
- Thonburian: `f5_models/mega_f5_last.safetensors` (**1.35 GB**) + auto-built ref voice

### curl examples

```bash
# Thai — best quality (Thonburian F5)
curl -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"text": "สวัสดีครับ", "engine": "f5", "voice": "default"}' \
  --output thai_best.wav

# Thai — fast (Meta MMS)
curl -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"text": "สวัสดีครับ", "engine": "mms", "voice": "facebook/mms-tts-tha"}' \
  --output thai_mms.wav

```bash
# Thai
curl -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"text": "สวัสดีครับ", "engine": "thai", "voice": "th_f_1"}' \
  --output thai.wav

# Piper
curl -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Piper.", "engine": "piper", "voice": "en_US-lessac-medium"}' \
  --output piper.wav
```

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
| HuggingFace cache | `~/.cache/huggingface` | `/root/.cache/huggingface` | Kokoro model weights (~300 MB) survive rebuilds |
| Audio output | `./output` | `/app/output` | Generated WAV files survive rebuilds, accessible from host |
| Piper voices | `./piper_models` | `/app/piper_models` | Piper ONNX voices (~60 MB each) |
| Thai voices | `./pythai_voices` | `/app/voices` | Thai Vachana ONNX voices (~60 MB each) |
| Thonburian models | `./f5_models` | `/app/f5_models` | F5-TTS Mega checkpoint (1.35 GB) + ref voice |

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

GPU engines (Kokoro, Thai MMS, Thai Thonburian/F5) run on the **fastest available GPU**: the host's **Tesla P100 (16 GB)** first, falling back to the **GTX 1060 (6 GB)**, then CPU if neither is usable. Verified: cached F5 synthesis ~2.2s on P100 vs ~4.4s on 1060.

**How it's configured:**

- `docker-compose.yml` exposes both GPUs via a Docker device reservation (P100 listed first so it maps to cuda:0):

  ```yaml
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids:
              - GPU-37d0153e-32ad-2825-9300-de8903297fd1  # Tesla P100 (preferred)
              - GPU-ebd852dc-b885-2874-feb2-1b37c939588b  # GTX 1060
            capabilities: [gpu]
  ```

  To target different GPUs, replace the UUIDs (find yours with `nvidia-smi --query-gpu=index,name,uuid --format=csv`). Only the listed GPUs are exposed to the container.

- `Dockerfile` pins `torch==2.7.1+cu126` from the [PyTorch index](https://download.pytorch.org/whl/cu126). Newer PyPI torch builds (cu13x) dropped kernels for Pascal (sm_60/61), which both the P100 and GTX 1060 require.

**Host requirements:** NVIDIA driver plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (`nvidia-smi` must work on the host).

**Automatic fallback:** `server.py`'s `_pick_device()` selects the best device at engine init:

1. **Prefer Tesla P100** (fastest), then GTX 1060, then CPU.
2. **Skip GPUs with < 1.5 GiB free VRAM** (`MIN_FREE_VRAM` in `server.py`).
3. **Init failure** — if CUDA model loading fails (e.g., OOM while loading weights), it retries on CPU.
4. **Mid-generation OOM (Kokoro)** — the server rebuilds the pipeline on CPU and retries that request once; subsequent requests stay on CPU to avoid thrashing.

Kokoro and MMS need ~1 GB VRAM; F5 needs ~1.7 GB, so OOM is unlikely unless other processes are using the cards.

## Project Structure

```
~/dev/kokoro/
├── docker-compose.yml   # Service definition, port mapping, volume mounts
├── Dockerfile           # Container build (python:3.11-slim)
├── server.py            # FastAPI application
├── static/
│   └── index.html       # Web UI — TTS form, player, file browser
├── output/              # Generated WAV files (gitignored)
├── piper_models/        # Downloaded Piper ONNX voices (gitignored)
├── pythai_voices/       # Downloaded Thai Vachana ONNX voices (gitignored)
├── f5_models/           # Downloaded Thonburian F5 checkpoint (gitignored)
├── .gitignore
├── README.md            # This file
├── AGENTS.md            # AI assistant context
└── SKILL.md             # Agent skill document — how to use this server
```

## Stop the Server

```bash
docker compose -f ~/dev/kokoro/docker-compose.yml down
```

## Why Kokoro?

- **82M parameters** (~300 MB) — tiny compared to Bark's ~1.2B
- **GPU-accelerated** — runs on the GTX 1060 when available, falls back to CPU automatically
- **Near-commercial quality** — competitive with cloud TTS
- **~50–100× faster than real-time** on modern CPUs
- **Simple Docker deployment** — single container; torch bundles the CUDA runtime, host needs the NVIDIA container toolkit
