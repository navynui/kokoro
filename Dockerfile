FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    # GTX 1060 is Pascal (sm_61). Newer PyPI torch builds (cu13x) dropped
    # kernels for sm_61, so pin an older CUDA 12.6 build that supports it.
    pip install --no-cache-dir torch==2.7.1+cu126 \
        --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir \
        fastapi>=0.115.0 \
        uvicorn[standard]>=0.34.0 \
        kokoro>=0.9.2 \
        soundfile>=0.13.0 \
        numpy>=1.26.0

COPY server.py .
COPY static static/

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
