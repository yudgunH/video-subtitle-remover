"""Incremental, crash-safe checkpoints for long OCR scans."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time


SCHEMA_VERSION = 1


@dataclass
class OcrCheckpointState:
    last_frame: int = 0
    complete: bool = False
    sampled_results: dict[int, list] = field(default_factory=dict)
    chinese_records: dict[int, list] = field(default_factory=dict)


def default_checkpoint_directory() -> Path:
    if os.name == "nt" and os.getenv("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"])
    elif os.getenv("XDG_CACHE_HOME"):
        root = Path(os.environ["XDG_CACHE_HOME"])
    else:
        root = Path.home() / ".cache"
    return root / "VideoSubtitleRemover" / "ocr_checkpoints"


class OcrCheckpointStore:
    """Store sampled OCR rows incrementally in one SQLite file per video."""

    def __init__(self, video_path, fingerprint, checkpoint_directory=None):
        canonical_path = os.path.normcase(os.path.abspath(str(video_path)))
        path_digest = hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()[:24]
        directory = Path(checkpoint_directory or default_checkpoint_directory())
        self.path = directory / f"{path_digest}.sqlite3"
        self.fingerprint_json = json.dumps(
            fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=20)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    schema_version INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    last_frame INTEGER NOT NULL,
                    complete INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    frame_no INTEGER PRIMARY KEY,
                    boxes TEXT NOT NULL,
                    chinese_records TEXT NOT NULL
                )
                """
            )
            return connection
        except BaseException:
            # sqlite3.Connection's context manager does not close the handle.
            # Close it explicitly so Windows can replace a corrupt cache file.
            connection.close()
            raise

    def _reset(self, connection):
        connection.execute("DELETE FROM detections")
        connection.execute("DELETE FROM metadata")
        connection.execute(
            """
            INSERT INTO metadata
                (id, schema_version, fingerprint, last_frame, complete, updated_at)
            VALUES (1, ?, ?, 0, 0, ?)
            """,
            (SCHEMA_VERSION, self.fingerprint_json, time.time()),
        )

    def _ensure_fingerprint(self, connection):
        row = connection.execute(
            "SELECT schema_version, fingerprint FROM metadata WHERE id = 1"
        ).fetchone()
        if row != (SCHEMA_VERSION, self.fingerprint_json):
            self._reset(connection)

    def _recover_corrupt_database(self):
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{self.path}{suffix}")
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass

    def load(self) -> OcrCheckpointState:
        if not self.path.exists():
            return OcrCheckpointState()
        try:
            with closing(self._connect()) as connection, connection:
                self._ensure_fingerprint(connection)
                row = connection.execute(
                    "SELECT last_frame, complete FROM metadata WHERE id = 1"
                ).fetchone()
                sampled_results = {}
                chinese_records = {}
                for frame_no, boxes_json, records_json in connection.execute(
                    "SELECT frame_no, boxes, chinese_records FROM detections ORDER BY frame_no"
                ):
                    boxes = json.loads(boxes_json)
                    records = json.loads(records_json)
                    if boxes:
                        sampled_results[int(frame_no)] = [tuple(box) for box in boxes]
                    if records:
                        for record in records:
                            record["box"] = tuple(record["box"])
                        chinese_records[int(frame_no)] = records
                return OcrCheckpointState(
                    last_frame=int(row[0]),
                    complete=bool(row[1]),
                    sampled_results=sampled_results,
                    chinese_records=chinese_records,
                )
        except (sqlite3.DatabaseError, json.JSONDecodeError, OSError, ValueError):
            self._recover_corrupt_database()
            return OcrCheckpointState()

    def save(
        self,
        last_frame,
        sampled_results=None,
        chinese_records=None,
        complete=False,
        _retried=False,
    ):
        sampled_results = sampled_results or {}
        chinese_records = chinese_records or {}
        try:
            with closing(self._connect()) as connection, connection:
                self._ensure_fingerprint(connection)
                frame_numbers = sorted(
                    set(sampled_results).union(chinese_records)
                )
                for frame_no in frame_numbers:
                    boxes = sampled_results.get(frame_no, [])
                    records = chinese_records.get(frame_no, [])
                    connection.execute(
                        """
                        INSERT INTO detections (frame_no, boxes, chinese_records)
                        VALUES (?, ?, ?)
                        ON CONFLICT(frame_no) DO UPDATE SET
                            boxes = excluded.boxes,
                            chinese_records = excluded.chinese_records
                        """,
                        (
                            int(frame_no),
                            json.dumps(boxes, ensure_ascii=False, separators=(",", ":")),
                            json.dumps(records, ensure_ascii=False, separators=(",", ":")),
                        ),
                    )
                connection.execute(
                    """
                    UPDATE metadata
                    SET last_frame = MAX(last_frame, ?),
                        complete = MAX(complete, ?),
                        updated_at = ?
                    WHERE id = 1
                    """,
                    (int(last_frame), int(bool(complete)), time.time()),
                )
        except sqlite3.DatabaseError:
            if _retried:
                raise
            self._recover_corrupt_database()
            with closing(self._connect()) as connection, connection:
                self._reset(connection)
            # Retry once after rebuilding the exact app-owned cache file.
            self.save(
                last_frame,
                sampled_results=sampled_results,
                chinese_records=chinese_records,
                complete=complete,
                _retried=True,
            )
