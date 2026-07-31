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

# Patch the installed flowtts (thonburian-tts fork):
#  1) Thai-aware duration scale — the UTF-8 byte-ratio duration estimate is
#     brittle for Thai (rushed/compressed speech). F5_DURATION_SCALE (env,
#     default 1.8) stretches the predicted duration so speed=1.0 sounds natural.
#  2) Unique temp ref filenames — the pipeline used fixed temp_short_ref.wav /
#     ref_converted.wav, a shared-state hazard between overlapping requests.
RUN python - <<'EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.11/site-packages/flowtts/infer/utils_infer.py")
s = p.read_text()
old = "duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)"
new = ("duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed "
       "* float(os.environ.get('F5_DURATION_SCALE', '1.8')))")
assert old in s, "duration line not found in utils_infer.py"
p.write_text(s.replace(old, new))

p2 = pathlib.Path("/usr/local/lib/python3.11/site-packages/flowtts/inference.py")
s2 = p2.read_text()
s2 = s2.replace('ref_wav_path = self.temp_dir / "ref_converted.wav"',
                'import uuid as _uuid; ref_wav_path = self.temp_dir / f"ref_converted_{_uuid.uuid4().hex}.wav"')
s2 = s2.replace('temp_short_ref = self.temp_dir / "temp_short_ref.wav"',
                'import uuid as _uuid2; temp_short_ref = self.temp_dir / f"temp_short_ref_{_uuid2.uuid4().hex}.wav"')
assert "ref_converted_" in s2 and "temp_short_ref_" in s2, "temp-file patch failed"
p2.write_text(s2)
print("flowtts patched (duration scale + unique temp refs)")
EOF

COPY server.py .
COPY static static/

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
