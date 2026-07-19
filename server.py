from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import os
import uuid
import io

app = FastAPI(title="Kokoro TTS Server", version="1.0.0")

# Lazy init — model downloads on first request, not at import time
pipeline: KPipeline | None = None


def get_pipeline() -> KPipeline:
    global pipeline
    if pipeline is None:
        pipeline = KPipeline(lang_code='a')
    return pipeline

AUDIO_DIR = "/tmp/tts_output"
os.makedirs(AUDIO_DIR, exist_ok=True)


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0


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
        out_path = os.path.join(AUDIO_DIR, f"{out_id}.wav")
        sf.write(out_path, combined, 24000)

        return FileResponse(out_path, media_type="audio/wav", filename="speech.wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
