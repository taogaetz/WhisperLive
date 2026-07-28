"""Persistent transcript history backed by SQLite."""

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _utc_now():
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


class TranscriptStore:
    """Store completed streaming segments without retaining source audio."""

    def __init__(self, path):
        self.path = str(Path(path).expanduser())
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self):
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    language TEXT,
                    model TEXT,
                    diarization INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS segments (
                    session_id TEXT NOT NULL,
                    segment_key TEXT NOT NULL,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    text TEXT NOT NULL,
                    speaker TEXT,
                    PRIMARY KEY (session_id, segment_key),
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_started_at
                    ON sessions(started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_segments_session_start
                    ON segments(session_id, start);
                """
            )

    def start_session(self, *, language=None, model=None, diarization=False):
        session_id = str(uuid.uuid4())
        now = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    id, started_at, updated_at, status, language, model,
                    diarization
                ) VALUES (?, ?, ?, 'active', ?, ?, ?)
                """,
                (
                    session_id,
                    now,
                    now,
                    language,
                    model,
                    int(bool(diarization)),
                ),
            )
        return session_id

    def update_session(self, session_id, segments):
        completed = [
            segment
            for segment in segments
            if segment.get("completed") and str(segment.get("text", "")).strip()
        ]
        now = _utc_now()
        with self._lock, self._connect() as connection:
            for segment in completed:
                try:
                    start = float(segment["start"])
                    end = float(segment["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                if start < 0 or end <= start:
                    continue
                segment_key = f"{start:.3f}:{end:.3f}"
                connection.execute(
                    """
                    INSERT INTO segments (
                        session_id, segment_key, start, end, text, speaker
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, segment_key) DO UPDATE SET
                        text = excluded.text,
                        speaker = excluded.speaker
                    """,
                    (
                        session_id,
                        segment_key,
                        start,
                        end,
                        str(segment["text"]).strip(),
                        segment.get("speaker") or segment.get("speaker_id"),
                    ),
                )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )

    def finish_session(self, session_id):
        now = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET ended_at = ?, updated_at = ?, status = 'complete'
                WHERE id = ?
                """,
                (now, now, session_id),
            )

    def list_sessions(self, limit=100):
        limit = max(1, min(int(limit), 200))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    sessions.*,
                    COUNT(segments.segment_key) AS segment_count,
                    COALESCE(MAX(segments.end), 0) AS duration_seconds,
                    COALESCE((
                        SELECT text
                        FROM segments AS first_segment
                        WHERE first_segment.session_id = sessions.id
                        ORDER BY first_segment.start
                        LIMIT 1
                    ), '') AS preview
                FROM sessions
                LEFT JOIN segments ON segments.session_id = sessions.id
                GROUP BY sessions.id
                ORDER BY sessions.started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._session_summary(row) for row in rows]

    def get_session(self, session_id):
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    sessions.*,
                    COUNT(segments.segment_key) AS segment_count,
                    COALESCE(MAX(segments.end), 0) AS duration_seconds,
                    '' AS preview
                FROM sessions
                LEFT JOIN segments ON segments.session_id = sessions.id
                WHERE sessions.id = ?
                GROUP BY sessions.id
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            segment_rows = connection.execute(
                """
                SELECT start, end, text, speaker
                FROM segments
                WHERE session_id = ?
                ORDER BY start, end
                """,
                (session_id,),
            ).fetchall()

        session = self._session_summary(row)
        session["segments"] = [
            {
                "start": f"{segment['start']:.3f}",
                "end": f"{segment['end']:.3f}",
                "text": segment["text"],
                "completed": True,
                **(
                    {"speaker": segment["speaker"]}
                    if segment["speaker"] is not None
                    else {}
                ),
            }
            for segment in segment_rows
        ]
        return session

    @staticmethod
    def _session_summary(row):
        preview = row["preview"] if "preview" in row.keys() else ""
        return {
            "id": row["id"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "language": row["language"],
            "model": row["model"],
            "diarization": bool(row["diarization"]),
            "segment_count": int(row["segment_count"]),
            "duration_seconds": round(float(row["duration_seconds"]), 3),
            "preview": preview,
        }
