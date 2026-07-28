import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from whisper_live.history import TranscriptStore
from whisper_live.server import ClientManager, TranscriptionServer


class TestClientManager(unittest.TestCase):
    def setUp(self):
        self.manager = ClientManager(max_clients=1, max_connection_time=60)

    def test_add_snapshot_and_remove(self):
        websocket = MagicMock()
        client = MagicMock()
        self.manager.add_client(websocket, client)

        self.assertIs(self.manager.get_client(websocket), client)
        self.assertEqual(
            self.manager.snapshot(),
            {
                "active": 1,
                "capacity": 1,
                "max_connection_seconds": 60,
                "oldest_session_seconds": 0.0,
            },
        )

        self.manager.remove_client(websocket)
        client.cleanup.assert_called_once()
        self.assertFalse(self.manager.get_client(websocket))

    def test_full_client_receives_wait_message(self):
        self.manager.add_client(MagicMock(), MagicMock())
        websocket = MagicMock()

        self.assertTrue(self.manager.is_server_full(websocket, {"uid": "new-client"}))
        message = json.loads(websocket.send.call_args.args[0])
        self.assertEqual(message["status"], "WAIT")
        self.assertEqual(message["uid"], "new-client")

    def test_timeout_disconnects_client(self):
        websocket = MagicMock()
        client = MagicMock()
        self.manager.add_client(websocket, client)
        self.manager.start_times[websocket] = time.time() - 61

        self.assertTrue(self.manager.is_client_timeout(websocket))
        client.disconnect.assert_called_once()


class TestWebDashboard(unittest.TestCase):
    def setUp(self):
        self.server = TranscriptionServer()
        self.app = FastAPI()
        self.model_path = "/opt/models/faster-whisper-large-v3-turbo"
        self.server.register_web_dashboard(
            self.app,
            websocket_port=9090,
            model_path=self.model_path,
        )
        self.client = TestClient(self.app)

    def test_dashboard_and_assets_are_served(self):
        dashboard = self.client.get("/")

        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("<title>PascalScribe</title>", dashboard.text)
        self.assertIn('aria-label="GPU telemetry"', dashboard.text)
        self.assertEqual(self.client.get("/ui/app.js").status_code, 200)
        self.assertEqual(self.client.get("/ui/audio-worklet.js").status_code, 200)
        self.assertEqual(self.client.get("/ui/utterances.mjs").status_code, 200)

    @patch("whisper_live.server.resolve_runtime", return_value=("cuda", "int8_float32"))
    def test_status_describes_fixed_runtime(self, _resolve_runtime):
        with patch.dict(
            os.environ,
            {"PASCALSCRIBE_SPEAKER_MODEL": "/does/not/exist.onnx"},
        ):
            payload = self.client.get("/api/status").json()

        self.assertEqual(payload["backend"], "faster_whisper")
        self.assertEqual(payload["device"], "cuda")
        self.assertEqual(payload["compute_type"], "int8_float32")
        self.assertEqual(payload["model"], "faster-whisper-large-v3-turbo")
        self.assertFalse(payload["diarization_available"])

    def test_telemetry_contains_gpu_and_session_counts(self):
        self.server.client_manager = ClientManager(1, 300)
        self.server.gpu_telemetry.snapshot = MagicMock(
            return_value={"available": True, "power_watts": 100}
        )

        payload = self.client.get("/api/telemetry").json()

        self.assertEqual(payload["gpu"]["power_watts"], 100)
        self.assertEqual(payload["sessions"]["active"], 0)
        self.assertEqual(payload["sessions"]["capacity"], 1)


class TestWebHistory(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.server = TranscriptionServer()
        self.server.transcript_store = TranscriptStore(
            Path(self.tempdir.name) / "transcripts.sqlite3"
        )
        app = FastAPI()
        self.server.register_web_dashboard(
            app,
            websocket_port=9090,
            model_path="/models/faster-whisper-large-v3-turbo",
        )
        self.client = TestClient(app)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_lists_and_reads_saved_session(self):
        session_id = self.server.transcript_store.start_session(
            language="en",
            model="large-v3-turbo",
            diarization=True,
        )
        self.server.transcript_store.update_session(
            session_id,
            [
                {
                    "start": "0.000",
                    "end": "2.500",
                    "text": "Stored sentence.",
                    "completed": True,
                    "speaker": "SPEAKER_00",
                }
            ],
        )
        self.server.transcript_store.finish_session(session_id)

        listing = self.client.get("/api/history").json()
        detail = self.client.get(f"/api/history/{session_id}").json()

        self.assertEqual(listing["sessions"][0]["id"], session_id)
        self.assertEqual(detail["segments"][0]["text"], "Stored sentence.")

    def test_missing_session_is_404(self):
        self.assertEqual(
            self.client.get("/api/history/not-a-session").status_code,
            404,
        )


class TestAudioInput(unittest.TestCase):
    def setUp(self):
        self.server = TranscriptionServer()

    def test_end_marker_stops_stream(self):
        websocket = MagicMock()
        websocket.recv.return_value = b"END_OF_AUDIO"
        self.assertFalse(self.server.get_audio_from_websocket(websocket))

    def test_float32_input(self):
        websocket = MagicMock()
        samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        websocket.recv.return_value = samples.tobytes()
        np.testing.assert_array_equal(
            self.server.get_audio_from_websocket(websocket),
            samples,
        )

    def test_int16_input_is_normalized(self):
        websocket = MagicMock()
        samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        self.server.audio_formats[websocket] = "int16"
        websocket.recv.return_value = samples.tobytes()

        result = self.server.get_audio_from_websocket(websocket)

        np.testing.assert_array_almost_equal(
            result,
            samples.astype(np.float32) / 32768.0,
        )


class TestServerValidation(unittest.TestCase):
    def test_rejects_missing_model(self):
        with self.assertRaisesRegex(ValueError, "Turbo model path"):
            TranscriptionServer().run("127.0.0.1", model_path="/missing")

    def test_rejects_invalid_capacity_before_model_check(self):
        with self.assertRaisesRegex(ValueError, "max_clients"):
            TranscriptionServer().run(
                "127.0.0.1",
                model_path="/missing",
                max_clients=0,
            )
