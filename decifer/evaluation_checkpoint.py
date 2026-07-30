"""Transactional checkpoints for staged minicif evaluation."""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1


class EvaluationCheckpoint:
    def __init__(self, path, signature, *, reset=False):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if reset:
            for suffix in ("", "-wal", "-shm"):
                Path(self.path + suffix).unlink(missing_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        try:
            self._initialize(signature)
        except Exception:
            self.connection.close()
            raise

    def _initialize(self, signature):
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS completed_units (
                sample_index INTEGER NOT NULL,
                prompt_mode TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                PRIMARY KEY (sample_index, prompt_mode)
            );
            CREATE TABLE IF NOT EXISTS candidate_rows (
                sample_index INTEGER NOT NULL,
                prompt_mode TEXT NOT NULL,
                rep INTEGER NOT NULL,
                row_json TEXT NOT NULL,
                PRIMARY KEY (sample_index, prompt_mode, rep)
            );
            CREATE TABLE IF NOT EXISTS refinement_records (
                sample_index INTEGER NOT NULL,
                prompt_mode TEXT NOT NULL,
                rep INTEGER NOT NULL,
                record_zlib BLOB NOT NULL,
                PRIMARY KEY (sample_index, prompt_mode, rep)
            );
            """
        )
        normalized = _json_dumps(signature)
        stored = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'signature'"
        ).fetchone()
        if stored is None:
            self.connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                [
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("signature", normalized),
                ],
            )
            self.connection.commit()
        elif stored[0] != normalized:
            raise ValueError(
                f"evaluation checkpoint configuration differs: {self.path}. "
                "Use a different --out-dir or pass --restart-splits."
            )

    def completed_units(self):
        return {
            (int(sample_index), prompt_mode)
            for sample_index, prompt_mode in self.connection.execute(
                "SELECT sample_index, prompt_mode FROM completed_units"
            )
        }

    def save_unit(self, sample_index, prompt_mode, rows, refinement_records):
        sample_index = int(sample_index)
        prompt_mode = str(prompt_mode)
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                "DELETE FROM candidate_rows "
                "WHERE sample_index = ? AND prompt_mode = ?",
                (sample_index, prompt_mode),
            )
            self.connection.execute(
                "DELETE FROM refinement_records "
                "WHERE sample_index = ? AND prompt_mode = ?",
                (sample_index, prompt_mode),
            )
            self.connection.executemany(
                """
                INSERT INTO candidate_rows
                    (sample_index, prompt_mode, rep, row_json)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        sample_index,
                        prompt_mode,
                        int(row["rep"]),
                        _json_dumps(row),
                    )
                    for row in rows
                ],
            )
            self.connection.executemany(
                """
                INSERT INTO refinement_records
                    (sample_index, prompt_mode, rep, record_zlib)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        sample_index,
                        prompt_mode,
                        int(record["rep"]),
                        sqlite3.Binary(
                            zlib.compress(
                                _json_dumps(record).encode("utf-8"),
                                level=6,
                            )
                        ),
                    )
                    for record in refinement_records
                ],
            )
            self.connection.execute(
                """
                INSERT OR REPLACE INTO completed_units
                    (sample_index, prompt_mode, completed_at)
                VALUES (?, ?, ?)
                """,
                (sample_index, prompt_mode, timestamp),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def close(self):
        self.connection.close()


def checkpoint_paths(out_dir):
    root = os.path.join(out_dir, "evaluation_checkpoints")
    if not os.path.isdir(root):
        return []
    return sorted(
        os.path.join(root, filename)
        for filename in os.listdir(root)
        if filename.endswith(".sqlite3")
    )


def load_checkpoint_frame(path):
    connection = sqlite3.connect(path)
    try:
        rows = [
            json.loads(row_json)
            for (row_json,) in connection.execute(
                """
                SELECT row_json
                FROM candidate_rows
                ORDER BY sample_index, prompt_mode, rep
                """
            )
        ]
    finally:
        connection.close()
    return pd.DataFrame(rows)


def load_combined_checkpoint_frame(paths):
    frames = [load_checkpoint_frame(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def export_split_metrics(paths, out_dir):
    split_dir = os.path.join(out_dir, "split_metrics")
    os.makedirs(split_dir, exist_ok=True)
    for path in paths:
        split = Path(path).stem
        load_checkpoint_frame(path).to_csv(
            os.path.join(split_dir, f"{split}.csv"),
            index=False,
        )


def export_refinement_records(paths, output_path):
    count = 0
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for path in paths:
            connection = sqlite3.connect(path)
            try:
                records = connection.execute(
                    """
                    SELECT record_zlib
                    FROM refinement_records
                    ORDER BY sample_index, prompt_mode, rep
                    """
                )
                for (compressed,) in records:
                    handle.write(zlib.decompress(compressed).decode("utf-8"))
                    handle.write("\n")
                    count += 1
            finally:
                connection.close()
    return count


def _json_dumps(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
        default=_json_default,
    )


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")
