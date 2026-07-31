FROM python:3.11-slim

WORKDIR /app

# ffmpeg required by pydub (ThonburianTTS/F5 ref-audio); git for pip install git+…
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    # GTX 1060 is Pascal (sm_61). Newer PyPI torch builds (cu13x) dropped
    # kernels for sm_61, so pin an older CUDA 12.6 build that supports it.
    pip install --no-cache-dir torch==2.7.1+cu126 \
        --index-url https://download.pytorch.org/whl/cu126 && \
    # Match torchvision/torchaudio to the pinned torch
    pip install --no-cache-dir torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 \
        --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir \
        fastapi>=0.115.0 \
        uvicorn[standard]>=0.34.0 \
        kokoro>=0.9.2 \
        piper-tts>=1.6.0 \
        pythaitts>=0.4.2 \
        soundfile>=0.13.0 \
        numpy>=1.26.0 && \
    # ThonburianTTS (F5-TTS Thai) — pulls f5-tts + vocos + librosa etc.
    # NB: plain `pip install git+…` installs deps but NO code, because the
    # repo's flowtts/ package lacks __init__.py (find_packages() skips it).
    # Clone, patch the missing __init__ files, install from the local dir.
    git clone --depth 1 https://github.com/biodatlab/thonburian-tts.git /opt/thonburian-tts && \
    touch /opt/thonburian-tts/flowtts/__init__.py /opt/thonburian-tts/flowtts/infer/__init__.py && \
    pip install --no-cache-dir /opt/thonburian-tts && \
    rm -rf /opt/thonburian-tts && \
    # Re-pin torch trio: f5-tts's dep chain may try to upgrade torch to cu13x,
    # which drops Pascal sm_61 support.
    pip install --no-cache-dir torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 \
        --index-url https://download.pytorch.org/whl/cu126

COPY server.py .
COPY static static/

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
