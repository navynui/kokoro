from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from kokoro import KPipeline
import torch
import threading
import soundfile as sf
import numpy as np
import uuid
from pathlib import Path

app = FastAPI(title="Kokoro TTS Server", version="1.2.0")

# Lazy init — model downloads on first request, not at import time
pipeline: KPipeline | None = None
_pipeline_lock = threading.Lock()

# Kokoro-82M needs roughly 1 GiB VRAM. If less than this is free, prefer
# CPU instead of risking a CUDA out-of-memory error.
MIN_FREE_VRAM = 1.5 * 1024 ** 3  # 1.5 GiB


def _pick_device() -> str:
    """Pick cuda only when a GPU is present with enough free VRAM."""
    if not torch.cuda.is_available():
        return 'cpu'
    free, _total = torch.cuda.mem_get_info()
    if free < MIN_FREE_VRAM:
        print(
            f"[server] Only {free / 1024**3:.2f} GiB VRAM free on "
            f"{torch.cuda.get_device_name(0)}; using CPU instead"
        )
        return 'cpu'
    return 'cuda'


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


def _generate_audio(pipe: KPipeline, request: TTSRequest) -> np.ndarray:
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


def _is_oom(e: Exception) -> bool:
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    try:
        combined = _generate_audio(get_pipeline(), request)
    except Exception as e:
        if not _is_oom(e):
            raise HTTPException(status_code=500, detail=str(e))
        # VRAM exhausted mid-generation — rebuild on CPU and retry once.
        global pipeline
        print("[server] CUDA out of memory during generation; falling back to CPU")
        with _pipeline_lock:
            pipeline = _build_pipeline('cpu')
        try:
            combined = _generate_audio(get_pipeline(), request)
        except Exception as e2:
            raise HTTPException(status_code=500, detail=str(e2))

    out_id = uuid.uuid4().hex[:12]
    out_path = OUTPUT_DIR / f"{out_id}.wav"
    sf.write(str(out_path), combined, 24000)

    return FileResponse(
        str(out_path),
        media_type="audio/wav",
        filename="speech.wav",
        headers={"X-Audio-Filename": f"{out_id}.wav"},
    )
