image := "pascalscribe:1080ti"
test_image := "pascalscribe:test"
dev_name := "pascalscribe-dev"
docker := env_var_or_default("DOCKER", "docker")
dev_uid := `id -u`
dev_gid := `id -g`
dev_model := env_var_or_default("PASCALSCRIBE_DEV_MODEL", "/opt/pascalscribe/models/faster-whisper-large-v3-turbo")

default:
    @just --list

lock:
    uv pip compile requirements/pascal.in \
      --python-version 3.10 \
      --python-platform x86_64-unknown-linux-gnu \
      --generate-hashes \
      --output-file requirements/pascal.lock

format:
    ruff format run_server.py whisper_live tests scripts/smoke_stream.py
    nix fmt

lint:
    ruff check run_server.py whisper_live tests scripts/smoke_stream.py

test:
    {{docker}} build --target test -t {{test_image}} .
    {{docker}} run --rm {{test_image}}

build:
    {{docker}} build --target runtime -t {{image}} .

run: build
    {{docker}} run --rm --init \
      --name pascalscribe \
      --device=nvidia.com/gpu=all \
      --publish 127.0.0.1:8000:8000 \
      --publish 127.0.0.1:9090:9090 \
      --volume pascalscribe-history:/data \
      {{image}}

# UI files are bind-mounted read-only and trigger an automatic browser refresh.
# Python/backend changes need only a container restart; host Python stays untouched.
dev-volume-init: build
    {{docker}} run --rm --user 0 \
      --volume pascalscribe-dev-history:/data \
      --entrypoint sh \
      {{image}} -c 'touch /data/.initialized && chown -R "{{dev_uid}}:{{dev_gid}}" /data'

dev: dev-volume-init
    {{docker}} run --rm --init \
      --name {{dev_name}} \
      --device=nvidia.com/gpu=all \
      --user {{dev_uid}}:{{dev_gid}} \
      --env PASCALSCRIBE_DEV_MODE=1 \
      --env PASCALSCRIBE_WEBSOCKET_PUBLIC_PORT=19090 \
      --publish 127.0.0.1:18000:8000 \
      --publish 127.0.0.1:19090:9090 \
      --volume pascalscribe-dev-history:/data \
      --mount type=bind,source="{{justfile_directory()}}/whisper_live/web",target=/app/whisper_live/web,readonly \
      {{image}} \
      python run_server.py --model_path "{{dev_model}}" --enable_rest

dev-up: dev-volume-init
    {{docker}} run --detach --rm --init \
      --name {{dev_name}} \
      --device=nvidia.com/gpu=all \
      --user {{dev_uid}}:{{dev_gid}} \
      --env PASCALSCRIBE_DEV_MODE=1 \
      --env PASCALSCRIBE_WEBSOCKET_PUBLIC_PORT=19090 \
      --publish 127.0.0.1:18000:8000 \
      --publish 127.0.0.1:19090:9090 \
      --volume pascalscribe-dev-history:/data \
      --mount type=bind,source="{{justfile_directory()}}/whisper_live/web",target=/app/whisper_live/web,readonly \
      {{image}} \
      python run_server.py --model_path "{{dev_model}}" --enable_rest

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
