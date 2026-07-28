import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=9090,
        help="Websocket port to run the server on.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the bundled CTranslate2 Turbo model.",
    )
    parser.add_argument(
        "--omp_num_threads",
        "-omp",
        type=int,
        default=1,
        help="Number of threads to use for OpenMP",
    )
    parser.add_argument(
        "--max_clients",
        type=int,
        default=4,
        help="Maximum clients supported by the server.",
    )
    parser.add_argument(
        "--max_connection_time",
        type=int,
        default=300,
        help="The maximum duration (in seconds) a client can stay connected. Defaults to 300 seconds (5 minutes)",
    )
    parser.add_argument(
        "--rest_port", type=int, default=8000, help="Port for the REST API server."
    )
    parser.add_argument(
        "--enable_rest",
        action="store_true",
        help="Enable the OpenAI-compatible REST API endpoint.",
    )
    parser.add_argument(
        "--cors-origins",
        type=str,
        default=None,
        help="Comma-separated list of allowed CORS origins (e.g., 'http://localhost:3000,http://example.com'). Defaults to localhost/127.0.0.1 on the WebSocket port.",
    )
    parser.add_argument(
        "--raw_pcm_input",
        action="store_true",
        help="Expect raw PCM int16 audio from clients instead of float32. "
        "Audio will be normalized to float32 range [-1.0, 1.0].",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Optional API key for authenticating REST API and WebSocket connections. "
        'Clients must send "Authorization: Bearer <key>" header or "?token=<key>" query parameter.',
    )
    parser.add_argument(
        "--rate_limit_rpm",
        type=int,
        default=0,
        help="Maximum REST API requests per minute per client IP. 0 = unlimited (default).",
    )
    parser.add_argument(
        "--history_path",
        type=str,
        default=os.environ.get("PASCALSCRIBE_HISTORY_PATH"),
        help=(
            "SQLite path for completed streaming transcripts. Disabled when omitted."
        ),
    )
    args = parser.parse_args()

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)

    from whisper_live.server import TranscriptionServer

    server = TranscriptionServer()
    server.run(
        "0.0.0.0",
        port=args.port,
        model_path=args.model_path,
        max_clients=args.max_clients,
        max_connection_time=args.max_connection_time,
        rest_port=args.rest_port,
        enable_rest=args.enable_rest,
        cors_origins=args.cors_origins,
        raw_pcm_input=args.raw_pcm_input,
        api_key=args.api_key,
        rate_limit_rpm=args.rate_limit_rpm,
        history_path=args.history_path,
    )
