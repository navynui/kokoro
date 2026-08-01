# Kokoro TTS Server — Agent Context

## Project Identity

- **Name:** Kokoro TTS Server
- **Path:** `~/dev/kokoro/`
- **Purpose:** Lightweight local TTS server via Docker, exposing a FastAPI REST endpoint backed by Kokoro-82M.
- **Stack:** Python 3.11, FastAPI, Uvicorn, Kokoro-82M (GPU) + Thai Vachana/MMS/Thonburian-F5/JaiTTS-F5/OmniVoice (see Engines) + Piper (CPU), Docker Compose.
- **GPU:** Both host GPUs are exposed to the container — Tesla P100 (cuda:0, preferred, fastest) and GTX 1060 (cuda:1). Device selection prefers P100 → 1060 → CPU (`_pick_device()` in server.py). Torch is pinned to `2.7.1+cu126` because newer builds dropped Pascal (sm_60/61) kernels. Automatic CPU fallback when VRAM is insufficient (see Constraints & Notes). Kokoro, MMS, F5, JaiTTS, and OmniVoice use the GPU; Thai Vachana and Piper run on CPU (ONNX).

## Key Files

| File | Role |
|---|---|
| `docker-compose.yml` | Single service `tts`, maps host `8001` → container `8000`. Mounts `~/.cache/huggingface` (model cache), `./output` (audio files), `./piper_models`, `./pythai_voices`, `./f5_models`, `./jaitts_models`, `./omnivoice_models`, `./ref_voices` (voice models). |
| `Dockerfile` | `python:3.11-slim` base. Pins `torch==2.7.1+cu126` (Pascal-compatible, from the PyTorch cu126 index) before installing `kokoro>=0.9.2`, `piper-tts>=1.6.0`, `pythaitts>=0.4.2`, `omnivoice>=0.2.1`, `fastapi>=0.115.0`, `uvicorn[standard]>=0.34.0`, `soundfile`, `numpy`. |
| `server.py` | FastAPI app. Serves web UI, audio file listing, and TTS endpoint. Lazy-init `KPipeline`. |
| `static/index.html` | Single-page web UI with TTS form, inline player, file browser with play/download/share. Mobile-responsive. |
| `.gitignore` | Ignores `output/` and `*.wav` |

## Architecture

```
Browser ──GET /────────────────────> FastAPI ──> index.html (web UI)
Browser ──GET /media/list──────────> FastAPI ──> JSON file list (audio + video)
Browser ──GET /media/{file}────────> FastAPI ──> media file (correct MIME)
Browser ──POST /v1/audio/speech────> FastAPI ──> engine ──> WAV response
             {"text","voice","speed","engine"}   │
             + optional {"ref","ref_text"}          ├─ kokoro: KPipeline (GPU)
             (voice cloning, f5/jaitts/omnivoice)     ├─ thai:    PyThaiTTS/Vachana (CPU/ONNX)
                                                    └─ piper:   PiperVoice (CPU/ONNX)
  MMS (`mms`): transformers VitsModel (facebook/mms-tts-tha), 16 kHz, GPU.
  F5 (`f5`): ThonburianTTS FlowTTSPipeline (F5-TTS Mega), 24 kHz, GPU; auto-builds a default ref voice via MMS on first use.
  JaiTTS (`jaitts`): JTS-AI JaiTTS-F5TTS checkpoint (same FlowTTSPipeline), 24 kHz, GPU; research prototype from the JaiTTS paper (better Thai CER than ThonburianTTS); shares the same ref-voice mechanism.
  OmniVoice (`omnivoice`): hotdogs/omnivoice-thai (Qwen3-0.6B MaskGIT diffusion), 24 kHz, GPU; zero-shot cloning via ref_audio+ref_text, plus an `instruct` voice-design mode and auto voice.
  Voice clones: POST/GET/DELETE /api/ref-voices — register a ref clip + transcript; f5/jaitts/omnivoice then
  condition on it instead of the default (zero-shot cloning, no fine-tuning).
                                              Downloads model
                                              on first use
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
  "engine": string     (default "kokoro"; "kokoro" | "thai" | "piper" | "mms" | "f5" | "jaitts" | "omnivoice")
  "ref": string        (optional, f5/jaitts/omnivoice) filename in ./output or ./ref_voices to clone
  "ref_text": string   (required when "ref" is set) exact transcript of the ref clip
  "instruct": string   (optional, omnivoice only) voice-design instruction, e.g. "male, low pitch"; overrides voice/ref
}

Headers (response):
  X-Audio-Filename: <uuid>.wav

Response: 200 audio/wav (binary) | 400/404/500 {"detail": "..."}
```

- Voice cloning (f5/jaitts/omnivoice): `voice` may be a registered ref-voice name (see below); `ref` + `ref_text` clone from an arbitrary existing media file in `./output` or `./ref_voices`. F5/JaiTTS fall back to the auto-built MMS default ref voice; OmniVoice falls back to auto voice (no reference — the model picks a voice itself).
- OmniVoice voice design: `voice` may also be a design preset id (listed in `/voices?engine=omnivoice`, e.g. `male-low`, `female-bright`, `whisper`) mapped to an `instruct` string; or pass `instruct` directly. Resolution: `instruct` → `ref` → design preset → registered ref voice → auto.

- Kokoro: 24 kHz WAV, GPU with CPU fallback.
- Thai (`thai`): 22.05 kHz WAV, voices `th_f_1|th_f_2|th_m_1|th_m_2`, models in `/app/voices` (mounted `./pythai_voices`).
- Piper (`piper`): 22.05 kHz WAV, curated voices (e.g. `en_US-lessac-medium`), models in `/app/piper_models` (mounted `./piper_models`).
- OmniVoice (`omnivoice`): 24 kHz WAV, GPU. Model + higgs audio tokenizer downloaded on first use into `/app/omnivoice_models` (mounted `./omnivoice_models`) — only the files needed for inference (the repo also ships a 4.9 GB optimizer.bin that a plain snapshot_download would pull; `_download_hf_file` fetches just `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`).

### `GET /voices`

```
GET /voices?engine=kokoro|thai|piper|mms|f5|jaitts|omnivoice
Response: 200 [{"id", "name", "language"}, ...] | 422 unknown engine
```

For `f5`/`jaitts`/`omnivoice`, the list appends one entry per registered reference voice (`Voice clone: <name>`); `omnivoice` also appends curated voice-design presets (`Design: …`).

### `POST /api/ref-voices`

Register a reference voice for cloning. Multipart form fields: `name` (slug, <= 64 chars), `text` (exact transcript, required), `pace` (float, default 1.0; > 1 = slower/more natural speech, applied as `speed / pace`), `audio` (any format pydub/ffmpeg decodes; 1–60s; stored as trimmed 24 kHz mono WAV, silence-clipped to <= 12s so inference conditioning stays deterministic). If the clip has multiple parts (e.g. English + Thai), the longest silence-delimited <= 12s segment is kept — the transcript must match that kept part, otherwise the byte-ratio duration estimate drifts.

```
Response: 200 {"id", "text", "pace", "source", "created", "duration"} | 400/413
```

### `GET /api/ref-voices`

```
Response: 200 [{"id", "name", "language", "text", "pace", "source", "created", "duration"}, ...]
```

### `GET /api/ref-voices/{name}`

Serves the registered clip as `audio/wav`.

### `PATCH /api/ref-voices/{name}`

Update a voice's `pace` and/or `text` without re-uploading audio (JSON body, both optional). Useful to tune a clone that sounds rushed — e.g. `{"pace": 1.8}`.

```
Response: 200 {"id", ...meta} | 400/404
```

### `DELETE /api/ref-voices/{name}`

```
Response: 200 {"ok": true} | 404
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

### `DELETE /media/{filename}`

Deletes a single media file from `./output` (used by the 🗑 button in the web UI).

```
Response: 200 {"ok": true, "filename": "..."} | 400/404
```

### `POST /media/cleanup`

Deletes all `.wav` files older than `days` (default 14) from `./output` (used by the 🧹 Cleanup button in the web UI).

```
Response: 200 {"ok": true, "removed": ["..."], "count": N}
```

### `GET /`

Serves the web UI (`static/index.html`). The UI renders audio files with `<audio>` and video files with `<video>`.

## Dependency Graph (pip)

```
torch == 2.7.1+cu126   # from https://download.pytorch.org/whl/cu126 (Pascal sm_61 support)
fastapi >= 0.115.0
uvicorn[standard] >= 0.34.0
kokoro >= 0.9.2
piper-tts >= 1.6.0     # pulls onnxruntime (CPU)
pythaitts >= 0.4.2     # pulls onnxruntime, vachanatts, pythainlp, ssg
omnivoice >= 0.2.1     # pulls transformers>=5.3.0, gradio, webdataset, accelerate, librosa; HiggsAudioV2TokenizerModel is transformers-5.x-native
thonburian-tts (git)   # pulls f5-tts, vocos, librosa, torchaudio/torchvision; cloned+pinned in Dockerfile (repo lacks __init__.py)
soundfile >= 0.13.0
numpy >= 1.26.0
```

Kokoro pulls in transformers, spaCy (en_core_web_sm), and misaki. OmniVoice requires `transformers>=5.3.0` (kokoro pins no transformers version — regression-test kokoro after any transformers upgrade). Torch bundles its own CUDA 12.6 runtime, so no additional system deps are needed — but the host must have the NVIDIA driver + container toolkit installed.

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
- **GPU wired:** The container exposes both host GPUs via `deploy.resources.reservations.devices` — Tesla P100 (`GPU-37d0153e-32ad-2825-9300-de8903297fd1`, cuda:0) and GTX 1060 (`GPU-ebd852dc-b885-2874-feb2-1b37c939588b`, cuda:1). `_pick_device()` prefers the P100 (fastest) by name, then the 1060, then CPU, skipping devices with < `MIN_FREE_VRAM` free. Torch is pinned to `2.7.1+cu126` — newer PyPI torch builds (cu13x) dropped sm_60/61 kernels. F5 cached synthesis: ~2.2s on P100 (vs ~4.4s on 1060).
- **CPU fallback (automatic):** `server.py` guards against missing/low VRAM in three ways — (1) `_pick_device()` checks `torch.cuda.mem_get_info()` and uses CPU if free VRAM < `MIN_FREE_VRAM` (1.5 GiB); (2) `_build_pipeline()` retries on CPU if CUDA init raises; (3) `_generate_kokoro()` catches mid-generation CUDA OOM, rebuilds the pipeline on CPU, and retries the request once (subsequent requests stay on CPU to avoid thrashing). Applies to the Kokoro engine only.
- **Engines:** `engine` field on the API selects `kokoro` (GPU) / `thai` (PyThaiTTS VachanaTTS2, CPU/ONNX) / `piper` (Piper, CPU/ONNX) / `mms` (Meta MMS-TTS Thai, GPU, 16 kHz) / `f5` (ThonburianTTS F5-TTS Mega, GPU, 24 kHz) / `jaitts` (JTS-AI JaiTTS-F5TTS, GPU, 24 kHz; Thai voice-cloning research prototype with better Thai CER than `f5`) / `omnivoice` (hotdogs/omnivoice-thai, Qwen3-0.6B MaskGIT diffusion, GPU, 24 kHz; zero-shot cloning + `instruct` voice design + auto voice). All are lazy-loaded singletons. Piper voices download on first use from `rhasspy/piper-voices` (URL derived from voice id: `{lang}/{lang}_{dialect}/{name}/{quality}/{voice_id}.onnx`); Thai voices from `VIZINTZOR/VachanaTTS` (written by `vachanatts` to `/app/voices`). Thai text is preprocessed (digits → Thai words, ๆ expansion) before synthesis.
- **F5 specifics:** `./f5_models` holds the 1.35 GB `mega_f5_last.safetensors` + vocab + `ref_voice.wav`. The ref voice is synthesized once via MMS (`F5_REF_TEXT`) so the pipeline never triggers its whisper-based `transcribe()` (which would need a ~3 GB ASR model). F5 runs fp32 on the GTX 1060 (~1.6 GB VRAM, ~4 s/sentence cached). Requires `ffmpeg` (pydub).
- **JaiTTS specifics:** `./jaitts_models` holds the 1.35 GB `model.pt` + vocab from `JTS-AI/JaiTTS-F5TTS` (Apache 2.0). Same `flowtts` pipeline as `f5`; vocab is byte-identical to F5's, so it reuses the same char set and the same MMS-built ref voice (`F5_REF_TEXT`). Uses `AudioConfig(cfg_strength=2.5)` per the JaiTTS quickstart. The XLM-R duration-predictor variant from the paper is not released — only the base F5 checkpoint. Warm synthesis ~2.3s on P100.
- **Voice cloning (f5/jaitts):** zero-shot — the flowtts pipeline conditions on any (ref_wav, ref_text) pair. Registered voices live in `./ref_voices` (`{name}.wav` + `{name}.json`), volume-mounted and gitignored, and are NOT exposed via `/media/list`. Upload requires the exact transcript (phase 1; auto-transcribe with a local ASR is a possible phase 2 — the pipeline's built-in `transcribe()` would pull a ~3 GB whisper model). `ref`/`ref_text` on the TTS request clones from any existing file in `./output` or `./ref_voices`. Cloning quality depends on the clip: 5–20s, single speaker, clean audio, accurate transcript.
- **OmniVoice specifics:** `./omnivoice_models/omnivoice-thai` holds the 2.45 GB fp32 `model.safetensors` (Qwen3-0.6B) + tokenizer files; the higgs audio tokenizer (0.8 GB) downloads into the HF cache. Loaded fp16 on GPU (~1.2 GB VRAM) via `_pick_device()`. `load_asr=False` because ref voices always carry transcripts — avoids a ~3 GB Whisper download. `generate(..., language="Thai", speed=request.speed/pace)`; no duration-scale patch needed (MaskGIT duration estimator, unlike flowtts). Voice-design via `instruct` is exposed as curated presets (`/voices?engine=omnivoice`, `Design: …`) plus a free-form `instruct` request field. Thai numerals are preprocessed before synthesis (`_preprocess_omnivoice_text`): digits → Thai words, `%` → เปอร์เซ็นต์, thousands separators stripped — the EN/ZH-heavy base otherwise misreads digits in Chinese. Voice-design voices are **frozen on first use** (`_get_frozen_ref`): the designed voice is baked into `omnivoice_models/frozen/<key>.wav` (key = preset id, or md5 of a free-form instruct) and all later requests clone from it, so separate API calls (e.g. per-line narration clips) share one consistent voice. All omnivoice generation uses a fixed seed → identical requests produce identical audio. Delete a frozen file to reset that voice.
- **GPU idle unload:** GPU models (Kokoro, MMS, F5, JaiTTS, OmniVoice) are dropped from VRAM after `IDLE_UNLOAD_MINUTES` (default 10, set in docker-compose `environment`; `0` disables) with no API/web activity. A FastAPI middleware (`activity_middleware`) records activity + in-flight requests; a daemon thread polls every 30s and calls `_unload_gpu_models()` (drops singletons, `gc.collect()`, `torch.cuda.empty_cache()`). Re-init after unload is fast (weights on disk): F5 ~2s, MMS ~1s, Kokoro ~3s, OmniVoice ~2–4s (2.45 GB from disk). CPU engines (Thai/Piper) stay loaded.
- **Thai duration scale (Dockerfile patch):** the flowtts UTF-8 byte-ratio duration formula underestimates for Thai, producing rushed speech at `speed=1.0` (JaiTTS paper calls this out). The Dockerfile patches the installed `flowtts/infer/utils_infer.py` to multiply the predicted duration by `F5_DURATION_SCALE` (env, default `1.8`, set in docker-compose). Also patches `flowtts/inference.py` to use unique per-request temp ref filenames (was fixed `temp_short_ref.wav`/`ref_converted.wav`). F5-family pipelines (f5, jaitts, and all clones) now sound natural at `speed=1.0`; the old workaround of `speed=0.5` is no longer needed.
- **Voice clone tuning:** each registered voice has a `pace` (default 1.0) applied as `speed / pace` server-side — raise it if a clone sounds rushed, lower it if too slow. Adjustable after registration via `PATCH /api/ref-voices/{name}` or the ✎ button in the web UI. F5-family `ModelConfig` uses a fixed `seed=0` so cloned-voice outputs are reproducible (the default `-1` randomized sampling every call, which made clone fidelity vary run-to-run).
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
- [x] Expose OmniVoice `instruct` voice-design mode — curated presets in `/voices?engine=omnivoice` (`Design: …`) + optional `instruct` request field + UI input (shown for the omnivoice engine)
- [ ] Pre-download OmniVoice model files in the Dockerfile so the first request doesn't wait on 2.45 GB
- [ ] Add OmniVoice mid-generation CUDA OOM retry-on-CPU (currently only Kokoro has it)
