image := "whisperlive:1080ti"
test_image := "whisperlive:1080ti-test"

default:
    @just --list

lock:
    uv pip compile requirements/pascal.in \
      --python-version 3.10 \
      --python-platform x86_64-unknown-linux-gnu \
      --generate-hashes \
      --output-file requirements/pascal.lock

format:
    ruff format whisper_live/runtime.py whisper_live/diarization.py tests/test_runtime.py tests/test_diarization.py scripts/smoke_stream.py
    nix fmt

lint:
    ruff check whisper_live/runtime.py whisper_live/diarization.py tests/test_runtime.py tests/test_diarization.py scripts/smoke_stream.py

test:
    docker build --target test -f docker/Dockerfile.pascal -t {{test_image}} .
    docker run --rm {{test_image}}

build:
    docker build --target runtime -f docker/Dockerfile.pascal -t {{image}} .

run: build
    docker run --rm --init \
      --name whisperlive-1080ti \
      --device=nvidia.com/gpu=all \
      --publish 127.0.0.1:8000:8000 \
      --publish 127.0.0.1:9090:9090 \
      --volume whisperlive-model-cache:/models \
      {{image}}

smoke: build
    docker run --rm --user 0 --network host \
      --entrypoint python \
      --volume "{{justfile_directory()}}/assets:/samples:ro" \
      {{image}} \
      /app/scripts/smoke_stream.py --audio /samples/jfk.flac
