# GTX 1080 Ti image

This branch provides one focused image for NVIDIA Pascal (`sm_61`):

- CTranslate2/faster-whisper on the GPU
- `small.en` pinned and baked into the image
- the one required CUDA 12 runtime library: cuBLAS
- CPU ONNX WeSpeaker embeddings for online speaker labels
- WebSocket streaming on port 9090
- OpenAI-compatible file/SSE REST API on port 8000

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
    image = "ghcr.io/taogaetz/whisperlive:1080ti"; # replace with @sha256:...
  };
}
```

Pin the published image digest before treating the service as production.
