from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import uuid
from pathlib import Path

app = FastAPI(title="Kokoro TTS Server", version="1.2.0")

# Lazy init — model downloads on first request, not at import time
pipeline: KPipeline | None = None


def get_pipeline() -> KPipeline:
    global pipeline
    if pipeline is None:
        pipeline = KPipeline(lang_code='a')
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


@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    try:
        pipe = get_pipeline()
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

        combined = np.concatenate(all_audio)

        out_id = uuid.uuid4().hex[:12]
        out_path = OUTPUT_DIR / f"{out_id}.wav"
        sf.write(str(out_path), combined, 24000)

        return FileResponse(
            str(out_path),
            media_type="audio/wav",
            filename="speech.wav",
            headers={"X-Audio-Filename": f"{out_id}.wav"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
