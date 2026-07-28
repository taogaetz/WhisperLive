import tempfile
import threading
import unittest
from pathlib import Path

from whisper_live.history import TranscriptStore


class TestTranscriptStore(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = TranscriptStore(Path(self.tempdir.name) / "transcripts.sqlite3")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_persists_only_completed_segments(self):
        session_id = self.store.start_session(
            language="en",
            model="large-v3-turbo",
            diarization=True,
        )
        self.store.update_session(
            session_id,
            [
                {
                    "start": "0.000",
                    "end": "1.250",
                    "text": "still changing",
                    "completed": False,
                },
                {
                    "start": "0.000",
                    "end": "2.500",
                    "text": "Final sentence.",
                    "completed": True,
                    "speaker": "SPEAKER_00",
                },
            ],
        )
        self.store.finish_session(session_id)

        session = self.store.get_session(session_id)

        self.assertEqual(session["status"], "complete")
        self.assertEqual(session["segment_count"], 1)
        self.assertEqual(session["duration_seconds"], 2.5)
        self.assertEqual(
            session["segments"],
            [
                {
                    "start": "0.000",
                    "end": "2.500",
                    "text": "Final sentence.",
                    "completed": True,
                    "speaker": "SPEAKER_00",
                }
            ],
        )

    def test_updates_are_idempotent_and_ordered(self):
        session_id = self.store.start_session()
        later = {
            "start": "4.000",
            "end": "5.000",
            "text": "Second.",
            "completed": True,
        }
        earlier = {
            "start": "0.000",
            "end": "2.000",
            "text": "First.",
            "completed": True,
        }

        self.store.update_session(session_id, [later, earlier])
        self.store.update_session(session_id, [later, earlier])

        session = self.store.get_session(session_id)
        self.assertEqual(session["segment_count"], 2)
        self.assertEqual(
            [segment["text"] for segment in session["segments"]],
            ["First.", "Second."],
        )

    def test_concurrent_updates_share_one_database(self):
        session_id = self.store.start_session()

        def write_segment(index):
            self.store.update_session(
                session_id,
                [
                    {
                        "start": f"{index:.3f}",
                        "end": f"{index + 0.5:.3f}",
                        "text": f"Segment {index}",
                        "completed": True,
                    }
                ],
            )

        threads = [
            threading.Thread(target=write_segment, args=(index,)) for index in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(self.store.get_session(session_id)["segment_count"], 8)

    def test_lists_newest_sessions_first(self):
        first = self.store.start_session()
        second = self.store.start_session()
        sessions = self.store.list_sessions()

        self.assertEqual([session["id"] for session in sessions], [second, first])
        self.assertEqual(sessions[0]["preview"], "")
