import os
import time
import threading
import collections
import json
import functools
import logging
import shutil
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, Form, Request, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import PlainTextResponse, StreamingResponse
import uvicorn
from faster_whisper import WhisperModel

import numpy as np
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed
from whisper_live.backend.base import ServeClientBase
from whisper_live.history import TranscriptStore
from whisper_live.runtime import resolve_runtime
from whisper_live.telemetry import NvidiaTelemetry

logging.basicConfig(level=logging.INFO)


class ClientManager:
    def __init__(self, max_clients=4, max_connection_time=600):
        """
        Initializes the ClientManager with specified limits on client connections and connection durations.

        Args:
            max_clients (int, optional): The maximum number of simultaneous client connections allowed. Defaults to 4.
            max_connection_time (int, optional): The maximum duration (in seconds) a client can stay connected. Defaults
                                                 to 600 seconds (10 minutes).
        """
        self.clients = {}
        self.start_times = {}
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time
        self.lock = threading.Lock()

    def add_client(self, websocket, client):
        """
        Adds a client and their connection start time to the tracking dictionaries.

        Args:
            websocket: The websocket associated with the client to add.
            client: The client object to be added and tracked.
        """
        with self.lock:
            self.clients[websocket] = client
            self.start_times[websocket] = time.time()

    def get_client(self, websocket):
        """
        Retrieves a client associated with the given websocket.

        Args:
            websocket: The websocket associated with the client to retrieve.

        Returns:
            The client object if found, False otherwise.
        """
        with self.lock:
            if websocket in self.clients:
                return self.clients[websocket]
            return False

    def remove_client(self, websocket):
        """
        Removes a client and their connection start time from the tracking dictionaries. Performs cleanup on the
        client if necessary.

        Args:
            websocket: The websocket associated with the client to be removed.
        """
        with self.lock:
            client = self.clients.pop(websocket, None)
            self.start_times.pop(websocket, None)
        if client:
            client.cleanup()

    def get_wait_time(self):
        """
        Calculates the estimated wait time for new clients based on the remaining connection times of current clients.

        Returns:
            The estimated wait time in minutes for new clients to connect. Returns 0 if there are available slots.
        """
        with self.lock:
            wait_time = None
            for start_time in self.start_times.values():
                current_client_time_remaining = self.max_connection_time - (
                    time.time() - start_time
                )
                if wait_time is None or current_client_time_remaining < wait_time:
                    wait_time = current_client_time_remaining
        return wait_time / 60 if wait_time is not None else 0

    def snapshot(self):
        """Return aggregate session capacity without exposing client IDs."""
        with self.lock:
            active = len(self.clients)
            oldest = (
                max(0.0, time.time() - min(self.start_times.values()))
                if self.start_times
                else 0.0
            )
            return {
                "active": active,
                "capacity": self.max_clients,
                "max_connection_seconds": self.max_connection_time,
                "oldest_session_seconds": round(oldest, 1),
            }

    def is_server_full(self, websocket, options):
        """
        Checks if the server is at its maximum client capacity and sends a wait message to the client if necessary.

        Args:
            websocket: The websocket of the client attempting to connect.
            options: A dictionary of options that may include the client's unique identifier.

        Returns:
            True if the server is full, False otherwise.
        """
        with self.lock:
            if len(self.clients) >= self.max_clients:
                wait_time = None
                for start_time in self.start_times.values():
                    remaining = self.max_connection_time - (time.time() - start_time)
                    if wait_time is None or remaining < wait_time:
                        wait_time = remaining
                wait_time_minutes = wait_time / 60 if wait_time is not None else 0
                response = {
                    "uid": options["uid"],
                    "status": "WAIT",
                    "message": wait_time_minutes,
                }
                websocket.send(json.dumps(response))
                return True
            return False

    def is_client_timeout(self, websocket):
        """
        Checks if a client has exceeded the maximum allowed connection time and disconnects them if so, issuing a warning.

        Args:
            websocket: The websocket associated with the client to check.

        Returns:
            True if the client's connection time has exceeded the maximum limit, False otherwise.
        """
        with self.lock:
            elapsed_time = time.time() - self.start_times[websocket]
            client = self.clients.get(websocket)
        if elapsed_time >= self.max_connection_time and client:
            client.disconnect()
            logging.warning(
                f"Client with uid '{client.client_uid}' disconnected due to overtime."
            )
            return True
        return False


class TranscriptionServer:
    RATE = 16000
    WEB_ROOT = Path(__file__).with_name("web")

    def __init__(self):
        self.client_manager = None
        self.no_voice_activity_chunks = 0
        self.use_vad = True
        self.raw_pcm_input = False
        self.audio_formats = {}
        self.segment_post_processor = None
        self.gpu_telemetry = NvidiaTelemetry()
        self.transcript_store = None

    def register_web_dashboard(
        self,
        app: FastAPI,
        websocket_port: int,
        model_path: Optional[str],
    ) -> None:
        """Serve the dependency-free browser client alongside the REST API."""
        dev_mode = os.environ.get("PASCALSCRIBE_DEV_MODE") == "1"
        public_websocket_port = int(
            os.environ.get("PASCALSCRIBE_WEBSOCKET_PUBLIC_PORT", websocket_port)
        )

        app.mount(
            "/ui",
            StaticFiles(directory=str(self.WEB_ROOT)),
            name="pascalscribe-ui",
        )

        @app.get("/", include_in_schema=False)
        async def dashboard():
            return FileResponse(
                self.WEB_ROOT / "index.html",
                media_type="text/html",
                headers={"Cache-Control": "no-cache"},
            )

        @app.get("/api/status", include_in_schema=False)
        async def dashboard_status():
            device, compute_type = resolve_runtime()
            speaker_model = os.environ.get("PASCALSCRIBE_SPEAKER_MODEL")
            return {
                "status": "ready",
                "backend": "faster_whisper",
                "device": device,
                "compute_type": compute_type,
                "model": Path(model_path).name,
                "websocket_port": public_websocket_port,
                "sample_rate": self.RATE,
                "dev_mode": dev_mode,
                "history_enabled": self.transcript_store is not None,
                "diarization_available": bool(
                    speaker_model and Path(speaker_model).is_file()
                ),
            }

        @app.get("/api/telemetry", include_in_schema=False)
        async def dashboard_telemetry():
            sessions = (
                self.client_manager.snapshot()
                if self.client_manager is not None
                else {
                    "active": 0,
                    "capacity": 0,
                    "max_connection_seconds": 0,
                    "oldest_session_seconds": 0.0,
                }
            )
            return {
                "gpu": await asyncio.to_thread(self.gpu_telemetry.snapshot),
                "sessions": sessions,
            }

        @app.get("/api/history", include_in_schema=False)
        async def dashboard_history(limit: int = 100):
            if self.transcript_store is None:
                return {"enabled": False, "sessions": []}
            sessions = await asyncio.to_thread(
                self.transcript_store.list_sessions,
                limit,
            )
            return {"enabled": True, "sessions": sessions}

        @app.get("/api/history/{session_id}", include_in_schema=False)
        async def dashboard_history_session(session_id: str):
            if self.transcript_store is None:
                return JSONResponse(
                    {"error": "Transcript history is disabled"},
                    status_code=404,
                )
            session = await asyncio.to_thread(
                self.transcript_store.get_session,
                session_id,
            )
            if session is None:
                return JSONResponse(
                    {"error": "Transcript session not found"},
                    status_code=404,
                )
            return session

        if dev_mode:

            @app.get("/api/dev/events", include_in_schema=False)
            async def dashboard_dev_events(request: Request):
                async def changes():
                    known = self._web_assets_mtime()
                    while not await request.is_disconnected():
                        await asyncio.sleep(0.4)
                        current = self._web_assets_mtime()
                        if current != known:
                            known = current
                            yield "event: reload\ndata: changed\n\n"

                return StreamingResponse(
                    changes(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )

    def _web_assets_mtime(self):
        """Return a change token for dependency-free dashboard live reload."""
        assets = []
        for path in self.WEB_ROOT.rglob("*"):
            if path.is_file():
                try:
                    stat = path.stat()
                    assets.append((str(path), stat.st_mtime_ns, stat.st_size))
                except OSError:
                    # Editors may briefly replace a file atomically.
                    continue
        return hash(tuple(sorted(assets)))

    def initialize_client(self, websocket, options, model_path):
        from whisper_live.backend.faster_whisper_backend import (
            ServeClientFasterWhisper,
        )

        client: Optional[ServeClientBase] = ServeClientFasterWhisper(
            websocket,
            language=options["language"],
            task=options["task"],
            client_uid=options["uid"],
            model=model_path,
            initial_prompt=options.get("initial_prompt"),
            vad_parameters=options.get("vad_parameters"),
            use_vad=self.use_vad,
            send_last_n_segments=options.get("send_last_n_segments", 10),
            no_speech_thresh=options.get("no_speech_thresh", 0.45),
            clip_audio=options.get("clip_audio", False),
            same_output_threshold=options.get("same_output_threshold", 10),
            hotwords=options.get("hotwords"),
            diarization=self._create_diarizer(options),
            word_timestamps=options.get("word_timestamps", False),
        )

        # Attach segment post-processor if configured
        if self.segment_post_processor is not None:
            client.segment_post_processor = self.segment_post_processor

        self.client_manager.add_client(websocket, client)
        logging.info("Streaming client ready with bundled large-v3-turbo model.")

    def _create_diarizer(self, options):
        """Create a SpeakerDiarizer if the client requested diarization.

        Returns:
            SpeakerDiarizer or None
        """
        if not options.get("enable_diarization", False):
            return None
        try:
            from whisper_live.diarization import SpeakerDiarizer

            return SpeakerDiarizer(
                similarity_threshold=options.get("diarization_threshold", 0.45),
                max_speakers=options.get("max_speakers", 10),
                hf_token=options.get("hf_token"),
            )
        except (ImportError, FileNotFoundError) as exc:
            logging.warning("Speaker diarization disabled: %s", exc)
            return None

    def get_audio_from_websocket(self, websocket):
        """
        Receives audio buffer from websocket and creates a numpy array out of it.

        Args:
            websocket: The websocket to receive audio from.

        Returns:
            A numpy array containing the audio.
        """
        frame_data = websocket.recv()
        if frame_data == b"END_OF_AUDIO":
            return False
        audio_format = self.audio_formats.get(websocket)
        if audio_format == "uint8":
            audio_np = np.frombuffer(frame_data, dtype=np.uint8)
            return (audio_np.astype(np.float32) - 128.0) / 128.0
        if self.raw_pcm_input or audio_format == "int16":
            audio_np = np.frombuffer(frame_data, dtype=np.int16)
            return audio_np.astype(np.float32) / 32768.0
        return np.frombuffer(frame_data, dtype=np.float32)

    def handle_new_connection(self, websocket, model_path):
        try:
            logging.info("New client connected")
            options = websocket.recv()
            options = json.loads(options)

            self.use_vad = options.get("use_vad")
            if self.client_manager.is_server_full(websocket, options):
                websocket.close()
                return False  # Indicates that the connection should not continue
            audio_format = options.get("audio_format", "float32")
            if audio_format not in {"float32", "int16", "uint8"}:
                raise ValueError(f"Unsupported audio_format: {audio_format}")
            self.audio_formats[websocket] = audio_format

            self.initialize_client(websocket, options, model_path)
            client = self.client_manager.get_client(websocket)
            if self.transcript_store is not None and client:
                try:
                    history_id = self.transcript_store.start_session(
                        language=options.get("language"),
                        model=Path(model_path).name,
                        diarization=options.get("enable_diarization", False),
                    )
                    client.transcript_history_id = history_id
                    client.transcript_sink = functools.partial(
                        self.transcript_store.update_session,
                        history_id,
                    )
                except Exception as exc:
                    logging.error("Could not start transcript history: %s", exc)
            return True
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON from client")
            return False
        except ConnectionClosed:
            logging.info("Connection closed by client")
            return False
        except Exception as e:
            logging.error(f"Error during new connection initialization: {str(e)}")
            return False

    def process_audio_frames(self, websocket):
        frame_np = self.get_audio_from_websocket(websocket)
        client = self.client_manager.get_client(websocket)
        if frame_np is False:
            return False

        client.add_frames(frame_np)
        return True

    def recv_audio(self, websocket, model_path):
        """
        Receive audio chunks from a client in an infinite loop.

        Continuously receives audio frames from a connected client
        over a WebSocket connection. It processes the audio frames using a
        voice activity detection (VAD) model to determine if they contain speech
        or not. If the audio frame contains speech, it is added to the client's
        audio data for ASR.
        If the maximum number of clients is reached, the method sends a
        "WAIT" status to the client, indicating that they should wait
        until a slot is available.
        If a client's connection exceeds the maximum allowed time, it will
        be disconnected, and the client's resources will be cleaned up.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            model_path (str): Path to the bundled CTranslate2 Turbo model.

        Raises:
            Exception: If there is an error during the audio frame processing.
        """
        if not self.handle_new_connection(websocket, model_path):
            return

        try:
            while not self.client_manager.is_client_timeout(websocket):
                if not self.process_audio_frames(websocket):
                    break
        except ConnectionClosed:
            logging.info("Connection closed by client")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
        finally:
            if self.client_manager.get_client(websocket):
                self.cleanup(websocket)
                websocket.close()
            del websocket

    def _stream_transcription(
        self, file, language, prompt, temperature, timestamp_granularities, model_path
    ):
        """Return a StreamingResponse that yields SSE events per segment."""

        async def _sse_generator():
            tmp_path = None
            try:
                suffix = os.path.splitext(file.filename)[1] or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copyfileobj(file.file, tmp)
                    tmp_path = tmp.name

                device, compute_type = resolve_runtime()
                transcriber = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=True,
                )
                segments, info = transcriber.transcribe(
                    tmp_path,
                    language=language,
                    initial_prompt=prompt,
                    temperature=temperature,
                    vad_filter=False,
                    word_timestamps=(
                        timestamp_granularities and "word" in timestamp_granularities
                    ),
                )

                for seg in segments:
                    seg_dict = {
                        "id": seg.id,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                    }
                    if timestamp_granularities and "word" in timestamp_granularities:
                        seg_dict["words"] = [
                            {
                                "word": w.word,
                                "start": w.start,
                                "end": w.end,
                                "probability": w.probability,
                            }
                            for w in seg.words
                        ]
                    yield f"data: {json.dumps(seg_dict)}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        return StreamingResponse(_sse_generator(), media_type="text/event-stream")

    @staticmethod
    def _normalize_form_list(values):
        """Normalize repeated or comma-separated multipart form fields."""
        if not values:
            return []
        normalized = []
        for value in values:
            if isinstance(value, str):
                normalized.extend(
                    item.strip() for item in value.split(",") if item.strip()
                )
        return normalized

    async def _create_rest_diarizer(
        self, known_speaker_names, known_speaker_references
    ):
        """Create a diarizer from OpenAI-compatible known speaker fields."""
        speaker_names = self._normalize_form_list(known_speaker_names)
        speaker_references = known_speaker_references or []

        if speaker_references and not speaker_names:
            raise ValueError(
                "known_speaker_references requires matching known_speaker_names"
            )
        if (
            speaker_names
            and speaker_references
            and len(speaker_names) != len(speaker_references)
        ):
            raise ValueError(
                "known_speaker_names and known_speaker_references must have the same length"
            )
        if not speaker_names and not speaker_references:
            return None

        from whisper_live.diarization import SpeakerDiarizer, load_audio

        diarizer = SpeakerDiarizer(
            max_speakers=max(10, len(speaker_names)),
            speaker_names=speaker_names,
        )

        for speaker_name, reference in zip(speaker_names, speaker_references):
            suffix = os.path.splitext(reference.filename or "")[1] or ".wav"
            reference_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await reference.read())
                    reference_path = tmp.name
                audio_np = load_audio(reference_path)
                if not diarizer.enroll_speaker(speaker_name, audio_np):
                    raise ValueError(
                        f"known_speaker_references for '{speaker_name}' is too short"
                    )
            finally:
                if reference_path and os.path.exists(reference_path):
                    os.unlink(reference_path)

        return diarizer

    @staticmethod
    def _speaker_labels_for_segments(segments, audio_np, diarizer, sample_rate=16000):
        if diarizer is None or audio_np is None:
            return {}
        labels = {}
        for index, segment in enumerate(segments):
            start = max(0, int(segment.start * sample_rate))
            end = min(len(audio_np), int(segment.end * sample_rate))
            if end <= start:
                continue
            speaker = diarizer.identify_speaker(audio_np[start:end], sample_rate)
            if speaker:
                labels[index] = speaker
        return labels

    def run(
        self,
        host,
        port=9090,
        model_path=None,
        max_clients=4,
        max_connection_time=600,
        rest_port=8000,
        enable_rest=False,
        cors_origins: Optional[str] = None,
        raw_pcm_input=False,
        api_key: Optional[str] = None,
        rate_limit_rpm: int = 0,
        history_path: Optional[str] = None,
        segment_post_processor=None,
    ):
        """
        Run the transcription server.

        Args:
            host (str): The host address to bind the server.
            port (int): The port number to bind the server.
            model_path (str): Path to the bundled CTranslate2 Turbo model.
            segment_post_processor (callable, optional): A callable that receives
                a transcription segment dict and returns a modified segment dict.
                Applied to every segment before sending to the client. Useful for
                plugging in custom post-processing (e.g. formatting, redaction).
                Defaults to None.
            history_path (str, optional): SQLite path for completed streaming
                transcript sessions. Source audio is not retained.
        """
        self.raw_pcm_input = raw_pcm_input
        self.transcript_store = (
            TranscriptStore(history_path) if history_path is not None else None
        )

        if max_clients < 1:
            raise ValueError(f"max_clients must be >= 1, got {max_clients}")
        if max_connection_time <= 0:
            raise ValueError(
                f"max_connection_time must be > 0, got {max_connection_time}"
            )

        self.segment_post_processor = segment_post_processor
        self.client_manager = ClientManager(max_clients, max_connection_time)
        if not model_path or not os.path.isdir(model_path):
            raise ValueError(f"Turbo model path is not a directory: {model_path}")

        # New OpenAI-compatible REST API (toggleable via enable_rest boolean)
        if enable_rest:
            app = FastAPI(title="PascalScribe API")
            self.register_web_dashboard(
                app,
                websocket_port=port,
                model_path=model_path,
            )
            origins = (
                [o.strip() for o in cors_origins.split(",")] if cors_origins else []
            )
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
                allow_headers=["*"],  # Allows all headers
            )

            # Optional API key authentication
            if api_key:

                @app.middleware("http")
                async def _check_api_key(request: Request, call_next):
                    auth = request.headers.get("Authorization", "")
                    if auth != f"Bearer {api_key}":
                        return JSONResponse(
                            {"error": "Invalid or missing API key"}, status_code=401
                        )
                    return await call_next(request)

            # Optional rate limiting (requests per minute per client IP)
            if rate_limit_rpm > 0:
                _rate_lock = threading.Lock()
                _rate_buckets: dict = {}  # ip -> deque of timestamps

                @app.middleware("http")
                async def _rate_limit(request: Request, call_next):
                    client_ip = request.client.host if request.client else "unknown"
                    now = time.time()
                    with _rate_lock:
                        bucket = _rate_buckets.setdefault(
                            client_ip, collections.deque()
                        )
                        # Discard entries older than 60s
                        while bucket and bucket[0] < now - 60:
                            bucket.popleft()
                        if len(bucket) >= rate_limit_rpm:
                            return JSONResponse(
                                {"error": "Rate limit exceeded"}, status_code=429
                            )
                        bucket.append(now)
                    return await call_next(request)

            @app.post("/v1/audio/transcriptions")
            async def transcribe(
                file: UploadFile,
                model: str = Form(default="whisper-1"),
                language: Optional[str] = Form(default=None),
                prompt: Optional[str] = Form(default=None),
                response_format: str = Form(default="json"),
                temperature: float = Form(default=0.0),
                timestamp_granularities: Optional[List[str]] = Form(default=None),
                # Stubs for unsupported OpenAI params
                chunking_strategy: Optional[str] = Form(default=None),
                include: Optional[List[str]] = Form(default=None),
                known_speaker_names: Optional[List[str]] = Form(default=None),
                known_speaker_references: Optional[List[UploadFile]] = File(
                    default=None
                ),
                stream: bool = Form(default=False),
                hotwords: Optional[str] = Form(default=None),
            ):
                if stream:
                    return self._stream_transcription(
                        file,
                        language,
                        prompt,
                        temperature,
                        timestamp_granularities,
                        model_path,
                    )

                ignored_params = []
                if chunking_strategy:
                    ignored_params.append(f"chunking_strategy='{chunking_strategy}'")
                if include:
                    ignored_params.append(f"include={include}")
                if ignored_params:
                    logging.warning(
                        f"Unsupported OpenAI params ignored: {', '.join(ignored_params)}"
                    )

                supported_formats = ["json", "text", "srt", "verbose_json", "vtt"]
                if response_format not in supported_formats:
                    return JSONResponse(
                        {
                            "error": f"Unsupported response_format. Supported: {supported_formats}"
                        },
                        status_code=400,
                    )

                if model != "whisper-1":
                    logging.warning(
                        "Ignoring requested model '%s'; using Turbo.", model
                    )
                model_name = model_path

                tmp_path = None
                try:
                    suffix = os.path.splitext(file.filename)[1] or ".wav"
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        shutil.copyfileobj(file.file, tmp)
                        tmp_path = tmp.name

                    device, compute_type = resolve_runtime()

                    transcriber = WhisperModel(
                        model_name, device=device, compute_type=compute_type
                    )
                    segments, info = transcriber.transcribe(
                        tmp_path,
                        language=language,
                        initial_prompt=prompt,
                        temperature=temperature,
                        vad_filter=False,
                        word_timestamps=(
                            timestamp_granularities
                            and "word" in timestamp_granularities
                        ),
                        hotwords=hotwords,
                    )
                    segments = list(segments)

                    text = " ".join([s.text.strip() for s in segments])

                    if response_format == "text":
                        return PlainTextResponse(text)
                    elif response_format == "json":
                        return {"text": text}
                    elif response_format == "verbose_json":
                        verbose = {
                            "task": "transcribe",
                            "language": info.language,
                            "duration": info.duration,
                            "text": text,
                            "segments": [],
                        }
                        speaker_labels = {}
                        try:
                            rest_diarizer = await self._create_rest_diarizer(
                                known_speaker_names, known_speaker_references
                            )
                        except ValueError as e:
                            return JSONResponse({"error": str(e)}, status_code=400)
                        if rest_diarizer is not None:
                            from whisper_live.diarization import load_audio

                            audio_np = load_audio(tmp_path)
                            speaker_labels = self._speaker_labels_for_segments(
                                segments, audio_np, rest_diarizer
                            )
                        for index, seg in enumerate(segments):
                            seg_dict = {
                                "id": seg.id,
                                "seek": seg.seek,
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text.strip(),
                                "tokens": seg.tokens,
                                "temperature": seg.temperature,
                                "avg_logprob": seg.avg_logprob,
                                "compression_ratio": seg.compression_ratio,
                                "no_speech_prob": seg.no_speech_prob,
                            }
                            if index in speaker_labels:
                                seg_dict["speaker"] = speaker_labels[index]
                            if (
                                timestamp_granularities
                                and "word" in timestamp_granularities
                            ):
                                seg_dict["words"] = [
                                    {
                                        "word": w.word,
                                        "start": w.start,
                                        "end": w.end,
                                        "probability": w.probability,
                                    }
                                    for w in seg.words
                                ]
                            verbose["segments"].append(seg_dict)
                        return verbose
                    elif response_format in ["srt", "vtt"]:
                        output = []
                        for i, seg in enumerate(segments, 1):
                            start = f"{int(seg.start // 3600):02}:{int((seg.start % 3600) // 60):02}:{seg.start % 60:06.3f}"
                            end = f"{int(seg.end // 3600):02}:{int((seg.end % 3600) // 60):02}:{seg.end % 60:06.3f}"
                            if response_format == "srt":
                                output.append(
                                    f"{i}\n{start.replace('.', ',')} --> {end.replace('.', ',')}\n{seg.text.strip()}\n"
                                )
                            else:  # vtt
                                output.append(
                                    f"{start} --> {end}\n{seg.text.strip()}\n"
                                )
                        return PlainTextResponse("\n".join(output))
                except Exception as e:
                    return JSONResponse({"error": str(e)}, status_code=500)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            threading.Thread(
                target=uvicorn.run,
                args=(app,),
                kwargs={"host": "0.0.0.0", "port": rest_port, "log_level": "info"},
                daemon=True,
            ).start()
            logging.info("HTTP API started on http://0.0.0.0:%s", rest_port)

        # Original WebSocket server (always supported)
        extra_ws_kwargs = {}
        if api_key:

            def _ws_auth(path, request_headers):
                auth = request_headers.get("Authorization", "")
                token_param = None
                # Check query string for token parameter
                if "?" in path:
                    from urllib.parse import urlparse, parse_qs

                    parsed = urlparse(path)
                    token_param = parse_qs(parsed.query).get("token", [None])[0]
                if auth == f"Bearer {api_key}" or token_param == api_key:
                    return None  # Allow connection
                return (401, [("Content-Type", "text/plain")], b"Unauthorized\n")

            extra_ws_kwargs["process_request"] = _ws_auth

        with serve(
            functools.partial(
                self.recv_audio,
                model_path=model_path,
            ),
            host,
            port,
            **extra_ws_kwargs,
        ) as server:
            server.serve_forever()

    def cleanup(self, websocket):
        """
        Cleans up resources associated with a given client's websocket.

        Args:
            websocket: The websocket associated with the client to be cleaned up.
        """
        client = self.client_manager.get_client(websocket)
        if client:
            history_id = getattr(client, "transcript_history_id", None)
            self.client_manager.remove_client(websocket)
            if self.transcript_store is not None and history_id is not None:
                try:
                    self.transcript_store.finish_session(history_id)
                except Exception as exc:
                    logging.error("Could not finish transcript history: %s", exc)
        self.audio_formats.pop(websocket, None)
