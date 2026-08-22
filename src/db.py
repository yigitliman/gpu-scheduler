import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.config import settings
from src.models import Job, JobState

_COLUMNS = "id, user, gpus, est_seconds, state, submitted_at, started_at, finished_at"


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """A connection that commits on success, rolls back on failure, and always closes.

    `with sqlite3.connect(...)` alone only manages the transaction: its __exit__
    commits or rolls back and then leaves the connection open, so every helper
    below used to leak one file handle per call.
    """
    Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user          TEXT    NOT NULL,
                gpus          INTEGER NOT NULL,
                est_seconds   INTEGER NOT NULL,
                state         TEXT    NOT NULL,
                submitted_at  REAL    NOT NULL,
                started_at    REAL,
                finished_at   REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)")


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        user=row["user"],
        gpus=row["gpus"],
        est_seconds=row["est_seconds"],
        state=JobState(row["state"]),
        submitted_at=row["submitted_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def submit_job(user: str, gpus: int, est_seconds: int, now: float) -> Job:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (user, gpus, est_seconds, state, submitted_at) VALUES (?,?,?,?,?)",
            (user, gpus, est_seconds, JobState.QUEUED.value, now),
        )
        job_id = cur.lastrowid
    return get_job(job_id)


def get_job(job_id: int) -> Job | None:
    with connect() as conn:
        row = conn.execute(f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def list_jobs(state: JobState | None = None) -> list[Job]:
    query = f"SELECT {_COLUMNS} FROM jobs"
    params: tuple = ()
    if state is not None:
        query += " WHERE state = ?"
        params = (state.value,)
    query += " ORDER BY id"
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_job(r) for r in rows]


def list_schedulable_jobs() -> list[Job]:
    """Every job the scheduler reasons about: active ones, plus finished ones
    because their decayed usage still shapes fair-share priority."""
    return [j for j in list_jobs() if j.state != JobState.CANCELLED]


def mark_started(job_id: int, now: float) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET state = ?, started_at = ? WHERE id = ? AND state = ?",
            (JobState.RUNNING.value, now, job_id, JobState.QUEUED.value),
        )


def mark_finished(job_id: int, now: float) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET state = ?, finished_at = ? WHERE id = ? AND state = ?",
            (JobState.COMPLETED.value, now, job_id, JobState.RUNNING.value),
        )


def cancel_job(job_id: int, now: float) -> bool:
    """Cancel a queued or running job. Returns False if it had already ended."""
    with connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET state = ?, finished_at = ? WHERE id = ? AND state IN (?, ?)",
            (
                JobState.CANCELLED.value,
                now,
                job_id,
                JobState.QUEUED.value,
                JobState.RUNNING.value,
            ),
        )
        return cur.rowcount > 0
