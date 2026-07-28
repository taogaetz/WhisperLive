# PascalScribe

PascalScribe is a self-hosted streaming transcription appliance for one target:
the NVIDIA GTX 1080 Ti (`sm_61`).

It is not a general Whisper runtime. The container has one ASR backend, one
pinned model, one CUDA precision, and one web interface:

- `large-v3-turbo` through CTranslate2 on CUDA with `int8_float32`
- live 16 kHz PCM audio over WebSocket
- CPU speaker labeling with WeSpeaker ONNX embeddings
- a browser microphone client with GPU power and memory readouts
- completed transcript history in SQLite
- an OpenAI-compatible file-transcription endpoint

The Turbo model and speaker model are built into the image. Nothing is
downloaded when the service starts, and nothing is installed into the host
Python environment.

## Data flow

```text
browser microphone
  -> 16 kHz float32 PCM over WebSocket :9090
  -> faster-whisper / large-v3-turbo on the GTX 1080 Ti
  -> optional CPU speaker label
  -> live partials and completed segments in the browser
  -> completed segments only in /data/transcripts.sqlite3
```

Audio, rolling partials, and raw WebSocket messages are never persisted.

## Hardware and runtime

Required:

- NVIDIA GTX 1080 Ti
- a working NVIDIA driver
- Docker with NVIDIA CDI support
- Linux on `amd64`

The image is deliberately not intended for CPU inference, AMD GPUs, newer
NVIDIA architectures, alternative Whisper models, translation, overlapping
speaker separation, or high-concurrency serving. One live stream is the
recommended operating point.

## Run

```console
docker run --rm --init \
  --device=nvidia.com/gpu=all \
  -p 127.0.0.1:8000:8000 \
  -p 127.0.0.1:9090:9090 \
  -v pascalscribe-history:/data \
  ghcr.io/taogaetz/pascalscribe:1080ti
```

Open <http://127.0.0.1:8000>.

| Port | Interface |
|---|---|
| `8000` | web UI, history API, telemetry, file transcription |
| `9090` | live PCM WebSocket transcription |

`/data` is the only persistent volume.

## Live WebSocket API

Connect to port `9090`, send one JSON options message, then send mono 16 kHz
float32 PCM frames:

```json
{
  "uid": "session-uuid",
  "language": "en",
  "task": "transcribe",
  "use_vad": true,
  "enable_diarization": true,
  "same_output_threshold": 2
}
```

The server replies with chronological segment snapshots. A completed segment
has `"completed": true` and may include `"speaker": "SPEAKER_00"`.

## HTTP API

Transcribe a file:

```console
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=whisper-1
```

Browse stored transcripts:

```text
GET /api/history
GET /api/history/{session_id}
```

The browser uses `GET /api/status` and `GET /api/telemetry`.

## Development

Nix supplies the development tools; Python dependencies stay in Docker:

```console
nix develop
just test
just build
just run
```

For the bind-mounted, auto-reloading web UI:

```console
DOCKER="sudo docker" just dev-up
```

Development ports are `18000` for HTTP and `19090` for WebSocket. UI files
reload automatically; Python changes require a container restart.

The flake exports `nixosModules.default` with the service option
`services.pascalScribe`.

## Security

Transcripts may contain private material. The example binds both interfaces to
localhost. Keep that boundary or publish through an authenticated private
network or reverse proxy.

The image runs as UID `10001`. Give that UID write access to any host directory
mounted at `/data`.

## Source and license

PascalScribe began as a specialization of
[Collabora WhisperLive](https://github.com/collabora/WhisperLive). The
historical upstream README is preserved in [SOURCE.md](SOURCE.md).

The code remains MIT licensed. Original copyright is retained in
[LICENSE](LICENSE), and redistributed model notices are in
[third_party](third_party).
