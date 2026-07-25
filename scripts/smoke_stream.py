#!/usr/bin/env python3
"""Minimal real-time WhisperLive WebSocket smoke client."""

import argparse
import json
import time
import uuid

import av
import numpy as np
from websockets.sync.client import connect


def decode_audio(paths, gap_seconds):
    chunks = []
    silence = np.zeros(int(16000 * gap_seconds), dtype=np.float32)

    for index, path in enumerate(paths):
        container = av.open(path)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        decoded = []
        try:
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    decoded.append(
                        resampled.to_ndarray().reshape(-1).astype(np.float32)
                    )
            for resampled in resampler.resample(None):
                decoded.append(resampled.to_ndarray().reshape(-1).astype(np.float32))
        finally:
            container.close()

        if decoded:
            if index:
                chunks.append(silence)
            chunks.append(np.concatenate(decoded))

    if not chunks:
        raise ValueError("No audio samples decoded")
    return np.concatenate(chunks)


def receive_available(websocket, messages, timeout, verbose=True):
    while True:
        try:
            payload = json.loads(websocket.recv(timeout=timeout))
        except TimeoutError:
            return
        messages.append(payload)
        if verbose:
            print(json.dumps(payload, ensure_ascii=False))
        timeout = 0.01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument("--gap-seconds", type=float, default=0.8)
    parser.add_argument("--chunk-ms", type=int, default=250)
    parser.add_argument("--diarization", action="store_true")
    parser.add_argument("--expect-speakers", type=int)
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    audio = decode_audio(args.audio, args.gap_seconds)
    chunk_samples = int(16000 * args.chunk_ms / 1000)
    messages = []

    with connect(f"ws://{args.host}:{args.port}", open_timeout=10) as websocket:
        websocket.send(
            json.dumps(
                {
                    "uid": str(uuid.uuid4()),
                    "language": "en",
                    "task": "transcribe",
                    "model": "small.en",
                    "use_vad": True,
                    "enable_diarization": args.diarization,
                }
            )
        )
        receive_available(websocket, messages, timeout=30, verbose=not args.quiet)
        if not any(message.get("message") == "SERVER_READY" for message in messages):
            raise RuntimeError("Server did not become ready")

        for offset in range(0, len(audio), chunk_samples):
            chunk = audio[offset : offset + chunk_samples]
            websocket.send(chunk.astype(np.float32, copy=False).tobytes())
            receive_available(
                websocket,
                messages,
                timeout=0.01,
                verbose=not args.quiet,
            )
            if not args.no_realtime:
                time.sleep(len(chunk) / 16000)

        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            receive_available(
                websocket,
                messages,
                timeout=0.25,
                verbose=not args.quiet,
            )
        websocket.send(b"END_OF_AUDIO")

    segments = {
        (segment["start"], segment["end"], segment["text"]): segment
        for message in messages
        for segment in message.get("segments", [])
    }
    completed = [segment for segment in segments.values() if segment["completed"]]
    if not segments:
        raise RuntimeError("Server returned no transcription segments")
    if args.diarization and not any("speaker" in segment for segment in completed):
        raise RuntimeError("Diarization was requested but no speaker labels arrived")

    completed_speakers = sorted(
        {segment["speaker"] for segment in completed if "speaker" in segment}
    )
    if (
        args.expect_speakers is not None
        and len(completed_speakers) != args.expect_speakers
    ):
        raise RuntimeError(
            f"Expected {args.expect_speakers} speakers, got {completed_speakers}"
        )

    print(
        json.dumps(
            {
                "audio_seconds": round(len(audio) / 16000, 2),
                "segments": completed or list(segments.values()),
                "completed_speakers": completed_speakers,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
