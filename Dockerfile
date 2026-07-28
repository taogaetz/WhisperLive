# syntax=docker/dockerfile:1.7
FROM python:3.10-slim-bookworm@sha256:9643927a6fc74bd81b0f1bbb5cce3cb4a491f46b4c5dbee770f28e575f180015 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_XET=1 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib \
    PASCALSCRIBE_HISTORY_PATH=/data/transcripts.sqlite3 \
    PASCALSCRIBE_SPEAKER_MODEL=/opt/pascalscribe/models/wespeaker-voxceleb-resnet34-LM.onnx \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/pascal.lock /tmp/pascal.lock
COPY requirements/pascal-cuda.lock /tmp/pascal-cuda.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/pascal.lock \
    && python -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/pascal-cuda.lock \
    # These are model-authoring helpers, not inference dependencies.
    && python -m pip uninstall --yes hf-xet mpmath sympy \
    && rm /tmp/pascal.lock /tmp/pascal-cuda.lock

RUN useradd --create-home --uid 10001 scribe \
    && mkdir -p /data /opt/pascalscribe/models \
    && chown -R scribe:scribe /data

FROM base AS test

COPY whisper_live /app/whisper_live
COPY run_server.py /app/run_server.py
USER root
RUN python -m pip install --no-cache-dir httpx==0.28.1 pytest==8.4.2
COPY tests /app/tests
RUN chmod -R a+rX /app
USER scribe
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]

FROM base AS runtime

ARG ASR_MODEL_REPOSITORY=dropbox-dash/faster-whisper-large-v3-turbo
ARG ASR_MODEL_REVISION=0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf

LABEL org.opencontainers.image.title="PascalScribe" \
      org.opencontainers.image.description="Realtime large-v3-turbo transcription for NVIDIA Pascal" \
      org.opencontainers.image.source="https://github.com/taogaetz/PascalScribe" \
      org.opencontainers.image.licenses="MIT"

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${ASR_MODEL_REPOSITORY}', revision='${ASR_MODEL_REVISION}', local_dir='/opt/pascalscribe/models/faster-whisper-large-v3-turbo')" \
    && chmod -R a+rX /opt/pascalscribe/models/faster-whisper-large-v3-turbo

ADD --chmod=644 --checksum=sha256:7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068 \
    https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx \
    /opt/pascalscribe/models/wespeaker-voxceleb-resnet34-LM.onnx

COPY --chmod=644 third_party/wespeaker-model.NOTICE /opt/pascalscribe/models/NOTICE
COPY --chmod=644 third_party/large-v3-turbo.NOTICE /opt/pascalscribe/models/large-v3-turbo.NOTICE
COPY whisper_live /app/whisper_live
COPY scripts/smoke_stream.py /app/scripts/smoke_stream.py
COPY run_server.py /app/run_server.py
RUN chmod -R a+rX /app

EXPOSE 8000 9090
VOLUME ["/data"]

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=4 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/openapi.json', timeout=2).close()"]

USER scribe
CMD ["python", "run_server.py", \
     "--model_path", "/opt/pascalscribe/models/faster-whisper-large-v3-turbo", \
     "--enable_rest"]
