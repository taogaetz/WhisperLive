# syntax=docker/dockerfile:1.7
FROM python:3.10-slim-bookworm@sha256:9643927a6fc74bd81b0f1bbb5cce3cb4a491f46b4c5dbee770f28e575f180015 AS app

ARG ASR_MODEL_REPOSITORY=Systran/faster-whisper-small.en
ARG ASR_MODEL_REVISION=d1d751a5f8271d482d14ca55d9e2deeebbae577f

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_XET=1 \
    HF_HOME=/models/huggingface \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WHISPERLIVE_SPEAKER_MODEL=/opt/whisperlive/models/wespeaker-voxceleb-resnet34-LM.onnx

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/pascal.lock /tmp/pascal.lock
COPY requirements/pascal-cuda.lock /tmp/pascal-cuda.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/pascal.lock \
    && python -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/pascal-cuda.lock \
    && rm /tmp/pascal.lock /tmp/pascal-cuda.lock

RUN useradd --create-home --uid 10001 whisperlive \
    && mkdir -p /models /opt/whisperlive/models \
    && chown -R whisperlive:whisperlive /models

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${ASR_MODEL_REPOSITORY}', revision='${ASR_MODEL_REVISION}', local_dir='/opt/whisperlive/models/faster-whisper-small.en')" \
    && chmod -R a+rX /opt/whisperlive/models/faster-whisper-small.en

ADD --chmod=644 --checksum=sha256:7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068 \
    https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx \
    /opt/whisperlive/models/wespeaker-voxceleb-resnet34-LM.onnx

COPY --chmod=644 third_party/wespeaker-model.NOTICE /opt/whisperlive/models/NOTICE
COPY whisper_live /app/whisper_live
COPY scripts/smoke_stream.py /app/scripts/smoke_stream.py
COPY run_server.py /app/run_server.py
RUN chmod -R a+rX /app

EXPOSE 8000 9090
VOLUME ["/models"]

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=4 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/openapi.json', timeout=2).close()"]

FROM app AS test

USER root
RUN python -m pip install --no-cache-dir httpx==0.28.1 pytest==8.4.2
COPY tests /app/tests
RUN chmod -R a+rX /app/tests
USER whisperlive
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_runtime.py", "tests/test_diarization.py", "tests/test_base_backend.py", "tests/test_server_extended.py", "tests/test_telemetry.py", "tests/test_metrics.py"]

FROM app AS runtime

USER whisperlive
CMD ["python", "run_server.py", \
     "--backend", "faster_whisper", \
     "--faster_whisper_custom_model_path", "/opt/whisperlive/models/faster-whisper-small.en", \
     "--cache_path", "/models/whisper-live", \
     "--enable_rest"]
