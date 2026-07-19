from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import os
import uuid
from pathlib import Path

app = FastAPI(title="Kokoro TTS Server", version="1.1.0")

# Lazy init — model downloads on first request, not at import time
pipeline: KPipeline | None = None


def get_pipeline() -> KPipeline:
    global pipeline
    if pipeline is None:
        pipeline = KPipeline(lang_code='a')
    return pipeline


BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "output"
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)


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


@app.get("/audio/list")
async def list_audio():
    """Return a JSON list of generated WAV files, newest first."""
    files = []
    for f in sorted(AUDIO_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.suffix.lower() == ".wav":
            stat = f.stat()
            files.append({
                "id": f.stem,
                "filename": f.name,
                "size": stat.st_size,
                "created": stat.st_mtime,
            })
    return files


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve a generated WAV file."""
    # Basic safety — prevent directory traversal
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    filepath = AUDIO_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(filepath), media_type="audio/wav")


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
        out_path = AUDIO_DIR / f"{out_id}.wav"
        sf.write(str(out_path), combined, 24000)

        return FileResponse(
            str(out_path),
            media_type="audio/wav",
            filename="speech.wav",
            headers={"X-Audio-Filename": f"{out_id}.wav"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
