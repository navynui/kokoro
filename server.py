from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Literal, Optional
from kokoro import KPipeline
import torch
import threading
import soundfile as sf
import numpy as np
import uuid
import shutil
import urllib.request
import os
import time
import gc
import json
import io
import re
from pathlib import Path

app = FastAPI(title="Kokoro TTS Server", version="1.5.0")

# ── GPU idle unload ─────────────────────────────────────────────────────
# GPU models (Kokoro, MMS, F5) are dropped from VRAM after this many
# minutes without any API/web activity. Set to 0 to disable.
IDLE_UNLOAD_MINUTES = float(os.environ.get("IDLE_UNLOAD_MINUTES", "10"))
IDLE_UNLOAD_SECONDS = IDLE_UNLOAD_MINUTES * 60
_last_activity = time.time()
_active_requests = 0
_activity_lock = threading.Lock()
_unload_lock = threading.Lock()


def _touch_activity() -> None:
    global _last_activity
    with _activity_lock:
        _last_activity = time.time()


def _unload_gpu_models() -> None:
    """Drop GPU singletons so VRAM is released; next request re-initializes."""
    global pipeline, _mms, _f5, _jaitts, _omnivoice
    with _unload_lock:
        if pipeline is None and _mms is None and _f5 is None and _jaitts is None and _omnivoice is None:
            return
        print(f"[server] Idle > {IDLE_UNLOAD_MINUTES:.0f} min — unloading GPU models from VRAM")
        pipeline = None
        _mms = None
        _f5 = None
        _jaitts = None
        _omnivoice = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[server] GPU models unloaded; next request will re-initialize")


def _idle_unload_loop() -> None:
    while True:
        time.sleep(30)
        try:
            if IDLE_UNLOAD_SECONDS <= 0:
                continue
            with _activity_lock:
                idle_secs = time.time() - _last_activity
                busy = _active_requests > 0
            if not busy and idle_secs >= IDLE_UNLOAD_SECONDS:
                _unload_gpu_models()
        except Exception as e:
            print(f"[server] idle-unload error: {e}")


@app.middleware("http")
async def activity_middleware(request: Request, call_next):
    """Any request (API or web) counts as activity and resets the idle timer."""
    global _active_requests
    _touch_activity()
    with _activity_lock:
        _active_requests += 1
    try:
        return await call_next(request)
    finally:
        with _activity_lock:
            _active_requests -= 1

# Lazy init — model downloads on first request, not at import time
pipeline: KPipeline | None = None
_pipeline_lock = threading.Lock()

# Kokoro-82M needs roughly 1 GiB VRAM. If less than this is free, prefer
# CPU instead of risking a CUDA out-of-memory error.
MIN_FREE_VRAM = 1.5 * 1024 ** 3  # 1.5 GiB


def _pick_device() -> str:
    """Pick the fastest usable GPU (Tesla P100 → GTX 1060), else CPU."""
    if not torch.cuda.is_available():
        return 'cpu'
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    # Prefer Tesla P100 (fastest), then any other NVIDIA GPU, then CPU
    order = sorted(range(len(names)), key=lambda i: (0 if 'P100' in names[i] else 1, i))
    for i in order:
        free, _total = torch.cuda.mem_get_info(i)
        if free >= MIN_FREE_VRAM:
            print(f"[server] Using {names[i]} (cuda:{i}) — {free / 1024**3:.2f} GiB free")
            return f'cuda:{i}'
    print("[server] No GPU with enough free VRAM; using CPU")
    return 'cpu'


def _build_pipeline(device: str) -> KPipeline:
    print(f"[server] Initializing KPipeline on device: {device}")
    try:
        return KPipeline(lang_code='a', device=device)
    except RuntimeError:
        if device != 'cuda':
            raise
        print("[server] CUDA init failed (likely not enough VRAM); falling back to CPU")
        return KPipeline(lang_code='a', device='cpu')


def get_pipeline() -> KPipeline:
    global pipeline
    if pipeline is None:
        with _pipeline_lock:
            if pipeline is None:
                pipeline = _build_pipeline(_pick_device())
    return pipeline


BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
REF_VOICES_DIR = BASE_DIR / "ref_voices"
REF_VOICES_DIR.mkdir(exist_ok=True)

# ── Engines ──────────────────────────────────────────────────────────────
# kokoro (GPU, default) | thai (PyThaiTTS/VachanaTTS2, CPU) | piper (CPU)

PIPER_DIR = BASE_DIR / "piper_models"
F5_DIR = BASE_DIR / "f5_models"
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
F5_CHECKPOINT = "https://huggingface.co/biodatlab/ThonburianTTS/resolve/main/megaF5/mega_f5_last.safetensors"
F5_VOCAB = "https://huggingface.co/biodatlab/ThonburianTTS/resolve/main/megaF5/mega_vocab.txt"
# Text spoken by the auto-built default reference voice (F5 clones this)
F5_REF_TEXT = "สวัสดีครับ ยินดีที่ได้รู้จัก"

JAITTS_DIR = BASE_DIR / "jaitts_models"
JAITTS_CHECKPOINT = "https://huggingface.co/JTS-AI/JaiTTS-F5TTS/resolve/main/model.pt"
JAITTS_VOCAB = "https://huggingface.co/JTS-AI/JaiTTS-F5TTS/resolve/main/vocab.txt"

OMNIVOICE_DIR = BASE_DIR / "omnivoice_models"
OMNIVOICE_REPO = "hotdogs/omnivoice-thai"
OMNIVOICE_MODEL_DIR = OMNIVOICE_DIR / "omnivoice-thai"
# Files needed for inference. The repo also ships training artifacts
# (optimizer.bin 4.9 GB, scheduler.bin, random_states, scaler.pt) that a
# plain snapshot_download would pull — fetch only what from_pretrained needs.
OMNIVOICE_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]

KOKORO_VOICES = [
    {"id": "af_heart", "name": "Heart (warm female)"},
    {"id": "af_bella", "name": "Bella (bright female)"},
    {"id": "af_nicole", "name": "Nicole (professional female)"},
    {"id": "af_sarah", "name": "Sarah (soft female)"},
    {"id": "af_sky", "name": "Sky (calm female)"},
    {"id": "am_adam", "name": "Adam (neutral male)"},
    {"id": "am_michael", "name": "Michael (deep male)"},
    {"id": "am_fenrir", "name": "Fenrir (low male)"},
    {"id": "am_puck", "name": "Puck (light male)"},
]

THAI_VOICES = [
    {"id": "th_f_1", "name": "Female 1", "language": "Thai"},
    {"id": "th_f_2", "name": "Female 2", "language": "Thai"},
    {"id": "th_m_1", "name": "Male 1", "language": "Thai"},
    {"id": "th_m_2", "name": "Male 2", "language": "Thai"},
]

MMS_VOICES = [
    {"id": "facebook/mms-tts-tha", "name": "Meta MMS Thai", "language": "Thai"},
]

F5_VOICES = [
    {"id": "default", "name": "Default (auto-built ref voice)", "language": "Thai"},
]

JAITTS_VOICES = [
    {"id": "default", "name": "Default (auto-built ref voice)", "language": "Thai"},
]

OMNIVOICE_VOICES = [
    {"id": "default", "name": "Auto voice (model picks; cloning via registered ref voices)", "language": "Thai"},
]

PIPER_VOICES = [
    {"id": "en_US-lessac-medium", "name": "Lessac", "language": "English (US)"},
    {"id": "en_GB-alan-medium", "name": "Alan", "language": "English (UK)"},
    {"id": "de_DE-thorsten-medium", "name": "Thorsten", "language": "German"},
    {"id": "fr_FR-siwis-medium", "name": "Siwis", "language": "French"},
    {"id": "es_ES-davefx-medium", "name": "David", "language": "Spanish"},
    {"id": "it_IT-riccardo-x_low", "name": "Riccardo", "language": "Italian"},
    {"id": "pt_BR-faber-medium", "name": "Faber", "language": "Portuguese (BR)"},
    {"id": "ru_RU-irina-medium", "name": "Irina", "language": "Russian"},
    {"id": "uk_UA-ukrainian_ts-medium", "name": "Ukrainian", "language": "Ukrainian"},
    {"id": "vi_VN-vais1000-medium", "name": "VAIS 1000", "language": "Vietnamese"},
    {"id": "ar_AR-omarsalim-medium", "name": "Omar Salim", "language": "Arabic"},
    {"id": "zh_CN-huayan-medium", "name": "Huayan", "language": "Chinese (Mandarin)"},
    {"id": "nl_NL-mls-medium", "name": "MLS", "language": "Dutch"},
    {"id": "pl_PL-darkman-medium", "name": "Darkman", "language": "Polish"},
    {"id": "tr_TR-fahrettin-medium", "name": "Fahrettin", "language": "Turkish"},
]


def _piper_paths(voice_id: str) -> tuple[str, str]:
    """Map 'en_US-lessac-medium' → (onnx_url, config_url) on HF."""
    parts = voice_id.split("-")
    if len(parts) < 3:
        raise ValueError(f"Invalid piper voice id: {voice_id}")
    quality = parts[-1]
    lang_dialect = parts[0]            # en_US
    name = "-".join(parts[1:-1])       # lessac
    lang = lang_dialect.split("_")[0]  # en
    base = f"{PIPER_BASE}/{lang}/{lang_dialect}/{name}/{quality}/{voice_id}"
    return base + ".onnx", base + ".onnx.json"


def _download(url: str, dest: Path) -> None:
    """Download url to dest (atomic), skipping if already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"[server] Downloading {url} → {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)


def _download_hf_file(repo_id: str, filename: str, dest_dir: Path) -> Path:
    """Download a single HuggingFace file into dest_dir, skipping if present."""
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    from huggingface_hub import hf_hub_download
    print(f"[server] Downloading {repo_id}/{filename} → {dest_dir.name}/")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest_dir))
    return dest


_piper_lock = threading.Lock()
_piper_voices: dict[str, "PiperVoice"] = {}


def get_piper_voice(voice_id: str) -> "PiperVoice":
    """Lazy-load a Piper ONNX voice, downloading it on first use."""
    if voice_id in _piper_voices:
        return _piper_voices[voice_id]
    with _piper_lock:
        if voice_id in _piper_voices:
            return _piper_voices[voice_id]
        from piper import PiperVoice
        onnx_url, json_url = _piper_paths(voice_id)
        onnx_path = PIPER_DIR / f"{voice_id}.onnx"
        json_path = PIPER_DIR / f"{voice_id}.onnx.json"
        _download(onnx_url, onnx_path)
        _download(json_url, json_path)
        print(f"[server] Loading Piper voice {voice_id}…")
        voice = PiperVoice.load(str(onnx_path), config_path=str(json_path))
        _piper_voices[voice_id] = voice
        return voice


_thai_lock = threading.Lock()
_pythai = None


def get_pythai():
    """Lazy singleton for PyThaiTTS (VachanaTTS2, CPU/ONNX)."""
    global _pythai
    if _pythai is None:
        with _thai_lock:
            if _pythai is None:
                from pythaitts import TTS as PyThaiTTS
                (BASE_DIR / "voices").mkdir(exist_ok=True)
                print("[server] Initializing PyThaiTTS (vachana)…")
                _pythai = PyThaiTTS(pretrained="vachana")
    return _pythai


_mms_lock = threading.Lock()
_mms = None


def get_mms():
    """Lazy singleton for Meta MMS-TTS Thai (transformers VITS, GPU if available)."""
    global _mms
    if _mms is None:
        with _mms_lock:
            if _mms is None:
                from transformers import VitsModel, AutoTokenizer
                device = _pick_device()
                print(f"[server] Initializing MMS-TTS (mms-tts-tha) on {device}…")
                _mms = {
                    "model": VitsModel.from_pretrained("facebook/mms-tts-tha").to(device),
                    "tok": AutoTokenizer.from_pretrained("facebook/mms-tts-tha"),
                    "device": device,
                }
    return _mms


_f5_lock = threading.Lock()
_f5 = None


def _make_f5_ref_voice(ref: Path) -> None:
    """Synthesize a short Thai reference clip with MMS so F5 can clone it."""
    print("[server] Building default F5 reference voice (via MMS)…")
    mms = get_mms()
    model, tok, device = mms["model"], mms["tok"], mms["device"]
    inputs = tok(F5_REF_TEXT, return_tensors="pt").to(device)
    with torch.no_grad():
        wave = model(**inputs).waveform[0].cpu().numpy()
    sf.write(str(ref), wave, model.config.sampling_rate)


def get_f5():
    """Lazy singleton for ThonburianTTS (F5-TTS Mega, GPU)."""
    global _f5
    if _f5 is None:
        with _f5_lock:
            if _f5 is None:
                from flowtts.inference import FlowTTSPipeline, ModelConfig, AudioConfig
                device = _pick_device()
                F5_DIR.mkdir(parents=True, exist_ok=True)
                ckpt = F5_DIR / "mega_f5_last.safetensors"
                vocab = F5_DIR / "mega_vocab.txt"
                ref = F5_DIR / "ref_voice.wav"
                _download(F5_CHECKPOINT, ckpt)
                _download(F5_VOCAB, vocab)
                if not ref.exists():
                    _make_f5_ref_voice(ref)
                print(f"[server] Initializing ThonburianTTS (F5 Mega) on {device}…")
                model_config = ModelConfig(
                    language="th",
                    model_type="F5",
                    checkpoint=str(ckpt),
                    vocab_file=str(vocab),
                    device=device,
                    seed=0,  # fixed seed -> reproducible voice cloning (was random)
                )
                pipeline = FlowTTSPipeline(
                    model_config=model_config,
                    audio_config=AudioConfig(),
                    temp_dir=str(F5_DIR / "temp"),
                )
                _f5 = {"pipeline": pipeline, "ref": str(ref), "ref_text": F5_REF_TEXT}
    return _f5


_jaitts_lock = threading.Lock()
_jaitts = None


def get_jaitts():
    """Lazy singleton for JaiTTS-F5TTS (Thai voice cloning, F5-TTS base, GPU)."""
    global _jaitts
    if _jaitts is None:
        with _jaitts_lock:
            if _jaitts is None:
                from flowtts.inference import FlowTTSPipeline, ModelConfig, AudioConfig
                device = _pick_device()
                JAITTS_DIR.mkdir(parents=True, exist_ok=True)
                ckpt = JAITTS_DIR / "model.pt"
                vocab = JAITTS_DIR / "vocab.txt"
                ref = JAITTS_DIR / "ref_voice.wav"
                _download(JAITTS_CHECKPOINT, ckpt)
                _download(JAITTS_VOCAB, vocab)
                if not ref.exists():
                    _make_f5_ref_voice(ref)
                print(f"[server] Initializing JaiTTS-F5TTS on {device}…")
                model_config = ModelConfig(
                    language="th",
                    model_type="F5",
                    checkpoint=str(ckpt),
                    vocab_file=str(vocab),
                    device=device,
                    seed=0,  # fixed seed -> reproducible voice cloning (was random)
                )
                pipeline = FlowTTSPipeline(
                    model_config=model_config,
                    # JaiTTS paper/quickstart recommends cfg_strength=2.5
                    audio_config=AudioConfig(cfg_strength=2.5),
                    temp_dir=str(JAITTS_DIR / "temp"),
                )
                _jaitts = {"pipeline": pipeline, "ref": str(ref), "ref_text": F5_REF_TEXT}
    return _jaitts


_omnivoice_lock = threading.Lock()
_omnivoice = None


def get_omnivoice():
    """Lazy singleton for OmniVoice Thai (Qwen3-0.6B MaskGIT diffusion, GPU).

    Voice cloning always supplies ref_text (our registered ref voices carry
    transcripts), so load_asr=False — avoids a ~3 GB Whisper download that
    would otherwise be used to auto-transcribe reference clips.
    """
    global _omnivoice
    if _omnivoice is None:
        with _omnivoice_lock:
            if _omnivoice is None:
                from omnivoice import OmniVoice
                device = _pick_device()
                OMNIVOICE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                for f in OMNIVOICE_FILES:
                    _download_hf_file(OMNIVOICE_REPO, f, OMNIVOICE_MODEL_DIR)
                print(f"[server] Initializing OmniVoice Thai on {device}…")
                model = OmniVoice.from_pretrained(
                    str(OMNIVOICE_MODEL_DIR),
                    device_map=device,
                    dtype=torch.float16 if device != "cpu" else torch.float32,
                    load_asr=False,
                )
                _omnivoice = {"model": model, "device": device, "ref": None, "ref_text": None}
    return _omnivoice


# ── Voice cloning (reference voices) ────────────────────────────────────
# F5-family engines (f5, jaitts) clone a voice from a reference clip: the
# pipeline is conditioned on (ref_wav, ref_text) instead of the built-in
# MMS-built default. Registered voices live in REF_VOICES_DIR as
# {name}.wav + {name}.json (transcript + provenance).

MAX_REF_AUDIO_MB = 50
_REF_NAME_RE = re.compile(r"[^0-9A-Za-z_\-\u0E00-\u0E7F]")


def _safe_ref_name(name: str) -> str:
    """Normalize a user-supplied voice name to a safe filesystem slug."""
    name = _REF_NAME_RE.sub("", name.strip().replace(" ", "-"))
    if not name or name in (".", "..") or name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid voice name")
    if len(name) > 64:
        raise HTTPException(status_code=400, detail="Voice name too long (max 64 chars)")
    return name


def _ref_wav_path(name: str) -> Path:
    return REF_VOICES_DIR / f"{name}.wav"


def _ref_meta_path(name: str) -> Path:
    return REF_VOICES_DIR / f"{name}.json"


def _list_ref_voices() -> list[dict]:
    """Return metadata for all registered reference voices (sorted by name)."""
    out = []
    for meta in sorted(REF_VOICES_DIR.glob("*.json")):
        name = meta.stem
        if not _ref_wav_path(name).exists():
            continue
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({
            "id": name,
            "name": name,
            "language": "Thai (voice clone)",
            "text": m.get("text", ""),
            "pace": m.get("pace", 1.0),
            "source": m.get("source", ""),
            "created": m.get("created", 0),
            "duration": m.get("duration", 0),
        })
    return out


def _get_ref_voice(name: str) -> Optional[tuple[str, str, float]]:
    """Return (wav_path, ref_text, pace) for a registered voice, or None."""
    wav = _ref_wav_path(name)
    meta = _ref_meta_path(name)
    if not wav.exists() or not meta.exists():
        return None
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
        text = m.get("text", "")
        pace = float(m.get("pace", 1.0))
    except Exception:
        return None
    return str(wav), text, pace


def _clone_voices() -> list[dict]:
    """Voice entries appended to the /voices list for f5/jaitts engines."""
    return [
        {"id": v["id"], "name": f"Voice clone: {v['id']}", "language": "Thai (voice clone)"}
        for v in _list_ref_voices()
    ]


def _resolve_ref_voice(engine: dict, request: "TTSRequest") -> tuple[str, str, float]:
    """Resolve a reference voice for F5-family engines.

    Returns (ref_wav_path, ref_text, pace). pace > 1 slows the voice
    (duration stretch); the engine applies it as speed/pace.

    Priority:
      1. request.ref (Option C) — a file in ./output or ./ref_voices
      2. request.voice (Option A) — a registered reference voice
      3. the engine's built-in default (auto-built MMS ref voice)
    """
    if request.ref:
        if not request.ref_text or not request.ref_text.strip():
            raise HTTPException(status_code=400, detail="ref_text is required when using ref")
        name = request.ref
        if "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="Invalid ref filename")
        for base in (OUTPUT_DIR, REF_VOICES_DIR):
            p = base / name
            if p.exists() and p.is_file():
                return str(p), request.ref_text.strip(), 1.0
        raise HTTPException(status_code=404, detail=f"ref file not found: {name}")
    if request.voice and request.voice != "default":
        found = _get_ref_voice(request.voice)
        if found is None:
            raise HTTPException(status_code=404, detail=f"Unknown voice: {request.voice}")
        return found
    return engine["ref"], engine["ref_text"], 1.0

# Media types we recognise
MEDIA_EXTENSIONS = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".m4a", ".flac"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv"}
ALL_MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def _is_media(f: Path) -> bool:
    return f.suffix.lower() in ALL_MEDIA_EXTENSIONS and f.is_file()


def _mediatype(suffix: str) -> str:
    return MEDIA_EXTENSIONS.get(suffix.lower(), "application/octet-stream")


def _list_files(extensions: set[str] | None = None):
    """Return sorted list of media files, newest first."""
    files = []
    for f in sorted(OUTPUT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if extensions is None and _is_media(f):
            pass  # include all media
        elif extensions and f.suffix.lower() in extensions and f.is_file():
            pass  # include only matching extensions
        else:
            continue
        stat = f.stat()
        ext = f.suffix.lower()
        files.append({
            "id": f.stem,
            "filename": f.name,
            "size": stat.st_size,
            "created": stat.st_mtime,
            "type": "video" if ext in VIDEO_EXTENSIONS else "audio",
            "mime": _mediatype(ext),
        })
    return files


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0
    engine: Literal["kokoro", "thai", "piper", "mms", "f5", "jaitts", "omnivoice"] = "kokoro"
    # Voice cloning (f5/jaitts engines): reference an existing media file directly
    ref: Optional[str] = None
    ref_text: Optional[str] = None


# ── Web UI ──────────────────────────────────────────────────────────────


@app.get("/")
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Kokoro TTS</h1><p>index.html not found.</p>")


@app.get("/media/list")
async def list_media():
    """Return all audio & video files, newest first."""
    return _list_files()


@app.get("/audio/list")
async def list_audio():
    """Return WAV files only (backward compat)."""
    return _list_files({".wav"})


@app.get("/media/{filename}")
@app.get("/audio/{filename}")
async def get_file(filename: str):
    """Serve any media file with correct MIME type."""
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    filepath = OUTPUT_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    mime = _mediatype(filepath.suffix)
    return FileResponse(str(filepath), media_type=mime)


# ── TTS API ─────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/voices")
async def list_voices(engine: str = "kokoro"):
    """List available voices for an engine."""
    if engine == "kokoro":
        return KOKORO_VOICES
    if engine == "thai":
        return THAI_VOICES
    if engine == "mms":
        return MMS_VOICES
    if engine == "f5":
        return F5_VOICES + _clone_voices()
    if engine == "jaitts":
        return JAITTS_VOICES + _clone_voices()
    if engine == "omnivoice":
        return OMNIVOICE_VOICES + _clone_voices()
    if engine == "piper":
        return PIPER_VOICES
    raise HTTPException(status_code=422, detail=f"Unknown engine: {engine}")


# ── Voice cloning API ────────────────────────────────────────────────────


@app.post("/api/ref-voices")
async def add_ref_voice(
    name: str = Form(""),
    text: str = Form(""),
    pace: float = Form(1.0),
    audio: UploadFile = File(...),
):
    """Register a reference voice for cloning (f5/jaitts engines).

    Multipart form: name (slug), text (required transcript of the clip),
    pace (speech-speed compensation; > 1 = slower/more natural, default 1.0),
    audio (any format pydub/ffmpeg can decode). Stored as a 24 kHz mono WAV,
    silence-trimmed and clipped to <= 12s so inference conditioning is stable.
    """
    from pydub import AudioSegment, silence

    safe = _safe_ref_name(name)
    transcript = text.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="ref_text (text) is required")
    if len(transcript) > 1000:
        raise HTTPException(status_code=400, detail="ref_text too long (max 1000 chars)")
    if not 0.2 <= pace <= 5.0:
        raise HTTPException(status_code=400, detail="pace must be between 0.2 and 5.0")

    data = await audio.read()
    if len(data) > MAX_REF_AUDIO_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Audio too large (max {MAX_REF_AUDIO_MB} MB)")

    try:
        seg = AudioSegment.from_file(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode audio (unsupported format?)")
    if len(seg) < 1000:
        raise HTTPException(status_code=400, detail="Reference audio too short (< 1s)")
    if len(seg) > 60_000:
        raise HTTPException(status_code=400, detail="Reference audio too long (> 60s) — keep it 5–20s")

    seg = seg.set_channels(1).set_frame_rate(24000)
    try:
        from flowtts.inference import remove_silence_edges
        seg = remove_silence_edges(seg)
    except Exception:
        pass

    # Clip to <=12s: flowtts clips refs to 12s at inference anyway; doing it here
    # (longest silence-delimited segment) keeps conditioning deterministic instead
    # of cutting mid-word on a long, multi-part clip.
    if len(seg) > 12_000:
        chunks = silence.split_on_silence(
            seg, min_silence_len=700, silence_thresh=-45, keep_silence=300, seek_step=10
        )
        candidates = [c for c in chunks if 1000 <= len(c) <= 12_000]
        seg = max(candidates, key=len) if candidates else seg[:12_000]

    REF_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    seg.export(str(_ref_wav_path(safe)), format="wav")
    meta = {
        "text": transcript,
        "pace": round(float(pace), 3),
        "source": audio.filename or "",
        "created": time.time(),
        "duration": round(len(seg) / 1000.0, 3),
    }
    _ref_meta_path(safe).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[server] Registered reference voice '{safe}' ({meta['duration']}s, pace {meta['pace']})")
    return {"id": safe, **meta}


@app.get("/api/ref-voices")
async def list_ref_voices():
    """List registered reference voices (voice clones)."""
    return _list_ref_voices()


@app.get("/api/ref-voices/{name}")
async def get_ref_voice(name: str):
    """Serve a registered reference voice's audio."""
    safe = _safe_ref_name(name)
    wav = _ref_wav_path(safe)
    if not wav.exists():
        raise HTTPException(status_code=404, detail="Voice not found")
    return FileResponse(str(wav), media_type="audio/wav")


@app.delete("/api/ref-voices/{name}")
async def delete_ref_voice(name: str):
    """Remove a registered reference voice."""
    safe = _safe_ref_name(name)
    removed = False
    for p in (_ref_wav_path(safe), _ref_meta_path(safe)):
        if p.exists():
            p.unlink()
            removed = True
    if not removed:
        raise HTTPException(status_code=404, detail="Voice not found")
    print(f"[server] Removed reference voice '{safe}'")
    return {"ok": True, "id": safe}


class RefVoiceUpdate(BaseModel):
    pace: Optional[float] = None
    text: Optional[str] = None


@app.patch("/api/ref-voices/{name}")
async def update_ref_voice(name: str, update: RefVoiceUpdate):
    """Update a registered voice's pace and/or transcript without re-uploading audio."""
    safe = _safe_ref_name(name)
    meta_path = _ref_meta_path(safe)
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Voice not found")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="Voice metadata corrupted")
    if update.pace is not None:
        if not 0.2 <= update.pace <= 5.0:
            raise HTTPException(status_code=400, detail="pace must be between 0.2 and 5.0")
        meta["pace"] = round(float(update.pace), 3)
    if update.text is not None:
        t = update.text.strip()
        if not t:
            raise HTTPException(status_code=400, detail="text must not be empty")
        meta["text"] = t
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[server] Updated reference voice '{safe}'")
    return {"id": safe, **meta}


def _generate_kokoro(pipe: KPipeline, request: TTSRequest) -> np.ndarray:
    generator = pipe(
        request.text,
        voice=request.voice,
        speed=request.speed,
        split_pattern=r'\n+',
    )

    all_audio = []
    for _, _, audio in generator:
        if audio is not None and len(audio) > 0:
            all_audio.append(audio)

    if not all_audio:
        raise HTTPException(status_code=500, detail="No audio generated")

    return np.concatenate(all_audio)


def _generate_thai(request: TTSRequest, out_path: Path) -> None:
    tts = get_pythai()
    # Numbers → Thai words, ๆ expansion (built-in preprocessor)
    from pythaitts import preprocess_text
    text = preprocess_text(request.text)
    tts.model(
        text=text,
        speaker_idx=request.voice,
        speed=request.speed,
        return_type="file",
        filename=str(out_path),
    )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise HTTPException(status_code=500, detail="Thai TTS produced no audio")


def _generate_piper(request: TTSRequest, out_path: Path) -> None:
    from piper import SynthesisConfig
    voice = get_piper_voice(request.voice)
    chunks = list(voice.synthesize(
        request.text,
        syn_config=SynthesisConfig(length_scale=1.0 / max(request.speed, 0.1)),
    ))
    if not chunks:
        raise HTTPException(status_code=500, detail="Piper produced no audio")
    audio = np.concatenate([c.audio_float_array for c in chunks])
    sf.write(str(out_path), audio, chunks[0].sample_rate)


def _generate_mms(request: TTSRequest, out_path: Path) -> None:
    mms = get_mms()
    model, tok, device = mms["model"], mms["tok"], mms["device"]
    inputs = tok(request.text, return_tensors="pt").to(device)
    with torch.no_grad():
        waveform = model(**inputs).waveform[0].cpu().numpy()
    if waveform.size == 0:
        raise HTTPException(status_code=500, detail="MMS produced no audio")
    sf.write(str(out_path), waveform, model.config.sampling_rate)


def _generate_f5(request: TTSRequest, out_path: Path) -> None:
    f5 = get_f5()
    ref_voice, ref_text, pace = _resolve_ref_voice(f5, request)
    f5["pipeline"](
        text=request.text,
        ref_voice=ref_voice,
        ref_text=ref_text,
        output_file=str(out_path),
        speed=request.speed / pace,
    )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise HTTPException(status_code=500, detail="F5 produced no audio")


def _generate_jaitts(request: TTSRequest, out_path: Path) -> None:
    j = get_jaitts()
    ref_voice, ref_text, pace = _resolve_ref_voice(j, request)
    j["pipeline"](
        text=request.text,
        ref_voice=ref_voice,
        ref_text=ref_text,
        output_file=str(out_path),
        speed=request.speed / pace,
    )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise HTTPException(status_code=500, detail="JaiTTS produced no audio")


def _generate_omnivoice(request: TTSRequest, out_path: Path) -> None:
    om = get_omnivoice()
    # Same ref resolution as F5/JaiTTS: request.ref file → registered voice →
    # auto mode (no reference; the model picks a voice itself).
    ref_audio, ref_text, pace = _resolve_ref_voice(om, request)
    kwargs = {
        "text": request.text,
        "language": "Thai",
        "speed": request.speed / pace,
    }
    if ref_audio:
        kwargs["ref_audio"] = ref_audio
        kwargs["ref_text"] = ref_text
    audios = om["model"].generate(**kwargs)
    if not audios or len(audios[0]) == 0:
        raise HTTPException(status_code=500, detail="OmniVoice produced no audio")
    sf.write(str(out_path), audios[0], om["model"].sampling_rate)


def _is_oom(e: Exception) -> bool:
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    out_id = uuid.uuid4().hex[:12]
    out_path = OUTPUT_DIR / f"{out_id}.wav"

    try:
        if request.engine == "kokoro":
            combined = _generate_kokoro(get_pipeline(), request)
            sf.write(str(out_path), combined, 24000)
        elif request.engine == "thai":
            _generate_thai(request, out_path)
        elif request.engine == "mms":
            _generate_mms(request, out_path)
        elif request.engine == "f5":
            _generate_f5(request, out_path)
        elif request.engine == "jaitts":
            _generate_jaitts(request, out_path)
        elif request.engine == "omnivoice":
            _generate_omnivoice(request, out_path)
        else:  # piper
            _generate_piper(request, out_path)
    except HTTPException:
        raise
    except Exception as e:
        if request.engine != "kokoro" or not _is_oom(e):
            raise HTTPException(status_code=500, detail=str(e))
        # Kokoro: VRAM exhausted mid-generation — rebuild on CPU and retry once.
        global pipeline
        print("[server] CUDA out of memory during generation; falling back to CPU")
        with _pipeline_lock:
            pipeline = _build_pipeline('cpu')
        try:
            combined = _generate_kokoro(get_pipeline(), request)
            sf.write(str(out_path), combined, 24000)
        except Exception as e2:
            raise HTTPException(status_code=500, detail=str(e2))

    return FileResponse(
        str(out_path),
        media_type="audio/wav",
        filename="speech.wav",
        headers={"X-Audio-Filename": f"{out_id}.wav"},
    )


if IDLE_UNLOAD_MINUTES > 0:
    threading.Thread(target=_idle_unload_loop, daemon=True).start()
    print(f"[server] GPU idle-unload enabled ({IDLE_UNLOAD_MINUTES:.0f} min)")
