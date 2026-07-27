image := "whisperlive:1080ti"
test_image := "whisperlive:1080ti-test"
dev_name := "whisperlive-1080ti-dev"
docker := env_var_or_default("DOCKER", "docker")
dev_uid := `id -u`
dev_gid := `id -g`

default:
    @just --list

lock:
    uv pip compile requirements/pascal.in \
      --python-version 3.10 \
      --python-platform x86_64-unknown-linux-gnu \
      --generate-hashes \
      --output-file requirements/pascal.lock

format:
    ruff format whisper_live/runtime.py whisper_live/diarization.py whisper_live/telemetry.py tests/test_runtime.py tests/test_diarization.py tests/test_telemetry.py scripts/smoke_stream.py
    nix fmt

lint:
    ruff check whisper_live/runtime.py whisper_live/diarization.py whisper_live/telemetry.py tests/test_runtime.py tests/test_diarization.py tests/test_telemetry.py scripts/smoke_stream.py

test:
    {{docker}} build --target test -f docker/Dockerfile.pascal -t {{test_image}} .
    {{docker}} run --rm {{test_image}}

build:
    {{docker}} build --target runtime -f docker/Dockerfile.pascal -t {{image}} .

run: build
    {{docker}} run --rm --init \
      --name whisperlive-1080ti \
      --device=nvidia.com/gpu=all \
      --publish 127.0.0.1:8000:8000 \
      --publish 127.0.0.1:9090:9090 \
      --volume whisperlive-model-cache:/models \
      {{image}}

# UI files are bind-mounted read-only and trigger an automatic browser refresh.
# Python/backend changes need only a container restart; host Python stays untouched.
dev: build
    {{docker}} run --rm --init \
      --name {{dev_name}} \
      --device=nvidia.com/gpu=all \
      --user {{dev_uid}}:{{dev_gid}} \
      --env WHISPERLIVE_DEV_MODE=1 \
      --env WHISPERLIVE_WEBSOCKET_PUBLIC_PORT=19090 \
      --publish 127.0.0.1:18000:8000 \
      --publish 127.0.0.1:19090:9090 \
      --volume whisperlive-dev-model-cache:/models \
      --mount type=bind,source="{{justfile_directory()}}/whisper_live/web",target=/app/whisper_live/web,readonly \
      {{image}}

dev-up: build
    {{docker}} run --detach --rm --init \
      --name {{dev_name}} \
      --device=nvidia.com/gpu=all \
      --user {{dev_uid}}:{{dev_gid}} \
      --env WHISPERLIVE_DEV_MODE=1 \
      --env WHISPERLIVE_WEBSOCKET_PUBLIC_PORT=19090 \
      --publish 127.0.0.1:18000:8000 \
      --publish 127.0.0.1:19090:9090 \
      --volume whisperlive-dev-model-cache:/models \
      --mount type=bind,source="{{justfile_directory()}}/whisper_live/web",target=/app/whisper_live/web,readonly \
      {{image}}

dev-logs:
    {{docker}} logs --follow {{dev_name}}

dev-stop:
    {{docker}} stop {{dev_name}}

smoke: build
    {{docker}} run --rm --user 0 --network host \
      --entrypoint python \
      --volume "{{justfile_directory()}}/assets:/samples:ro" \
      {{image}} \
      /app/scripts/smoke_stream.py --audio /samples/jfk.flac
