# GTX 1080 Ti image

This branch provides one focused image for NVIDIA Pascal (`sm_61`):

- CTranslate2/faster-whisper on the GPU
- `small.en` pinned and baked into the image
- the one required CUDA 12 runtime library: cuBLAS
- CPU ONNX WeSpeaker embeddings for online speaker labels
- WebSocket streaming on port 9090
- OpenAI-compatible file/SSE REST API on port 8000
- dependency-free live microphone console on port 8000

It intentionally omits PyTorch, CUDA 13, cuDNN, NVRTC, Transformers,
OpenVINO, TensorRT, pyannote, SciPy, librosa, and the training/evaluation
packages. Whisper's built-in audio-to-English `translate` task remains
available; WhisperLive's separate Small100 post-translation path does not.

## Build and run

```console
nix develop
just test
just run
```

The run recipe uses NixOS's NVIDIA CDI device:

```console
docker run --rm --init \
  --device=nvidia.com/gpu=all \
  -p 127.0.0.1:9090:9090 \
  -p 127.0.0.1:8000:8000 \
  -v whisperlive-model-cache:/models \
  ghcr.io/taogaetz/whisperlive:1080ti
```

No Python packages are installed on the host. The Nix development shell
contains only Docker, Git, Just, uv, Ruff, and ShellCheck.

The image auto-selects `int8_float32` on the GTX 1080 Ti. Set
`WHISPERLIVE_COMPUTE_TYPE=float32` to compare accuracy or performance.

## Browser console

Open <http://127.0.0.1:8000/> in a browser on the machine running the
container. The console captures the microphone, streams mono float32 PCM to
the WebSocket endpoint, and shows:

- the latest revisable partial phrase
- completed transcript lines with online speaker labels
- input level, elapsed time, estimated stream lag, and runtime details
- live GPU load, VRAM, temperature, power, clock, fan, and session capacity
- the raw server event stream for debugging

The console is plain HTML, CSS, and JavaScript served by the existing FastAPI
process. It adds no Node runtime, web framework, or extra Python package to
the image. Microphone capture requires a secure browser context; loopback
addresses such as `127.0.0.1` and `localhost` qualify.

## Contained dashboard development

The development server uses the same runtime image and GPU stack as
production, but publishes separate loopback ports:

```console
nix develop
just dev
```

Open <http://127.0.0.1:18000/>. The WebSocket endpoint is on port 19090.
`whisper_live/web` is bind-mounted read-only into the container, and the
browser automatically reloads when its HTML, CSS, JavaScript, or worklet
changes. There is no Node process and no Python package installed on the
host.

Use `just dev-up`, `just dev-logs`, and `just dev-stop` for a detached
container. Backend Python changes require restarting the development
container, but Docker reuses the dependency and model layers.

On a host where Docker requires privilege escalation, prefix recipes with
`DOCKER="sudo docker"`, for example:

```console
DOCKER="sudo docker" just dev-up
```

## Concurrent sessions

Opening the page does not allocate a transcription session. Each press of
**Start microphone** opens a new WebSocket connection with an independent
UUID, audio buffer, transcript, and diarization speaker state. The shared
Whisper model is loaded once; inference calls are serialized unless batch
inference is explicitly enabled.

The default limit is four active sessions and five minutes per connection.
An additional client receives `WAIT` and is disconnected; it is not currently
placed in a durable queue. Reloading or closing the page ends that session,
and dashboard transcripts are not persisted on the server.

## Streaming protocol

Live raw audio uses WebSockets, not a chunked HTTP request. Connect to
`ws://127.0.0.1:9090`, send a JSON options message, then send mono 16 kHz
float32 PCM frames:

```json
{
  "uid": "session-id",
  "language": "en",
  "task": "transcribe",
  "model": "small.en",
  "use_vad": true,
  "enable_diarization": true
}
```

The server returns partial and completed `segments`. Completed segments may
include `"speaker": "SPEAKER_00"`.

Run the small protocol client against an already-running container:

```console
just smoke
```

The smoke recipe runs the client from the same image, so it does not add audio
or WebSocket packages to the host development shell.

An upstream source that only speaks HTTP still needs a small adapter that
holds the WebSocket open and forwards its audio chunks.

## REST API

The REST endpoint accepts complete files and can emit normal JSON or SSE:

```console
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@assets/jfk.flac \
  -F model=whisper-1 \
  -F response_format=json
```

## Speaker labeling scope

This is low-latency online speaker labeling: each completed ASR segment gets
one embedding and one speaker label. It supports stable `SPEAKER_XX` labels
and known-speaker enrollment through the REST fields already implemented by
WhisperLive. Turns shorter than one second are assigned to the nearest
established speaker rather than creating a new identity, which limits cluster
fragmentation from unreliable short-window embeddings.

It is not overlap-aware diarization. If two people talk over each other inside
one ASR segment, it does not separate both voices. That feature would require a
segmentation model and would substantially enlarge the image.

## NixOS module

The flake exports `nixosModules.default`. The host must already have the
NVIDIA driver and container toolkit/CDI configured.

```nix
{
  inputs.whisperlive.url = "github:taogaetz/WhisperLive/codex/pascal-sm61";
  imports = [ inputs.whisperlive.nixosModules.default ];

  services.whisperlivePascal = {
    enable = true;
    image = "ghcr.io/taogaetz/whisperlive@sha256:7f986191e4bfe4b96d7a177520fb264e60010a3941462f192ab5abc6e080ac20";
  };
}
```

The module default uses this same immutable digest. The `:1080ti` tag remains
available for manual testing.
