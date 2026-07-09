from dataclasses import dataclass
from enum import StrEnum


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: int
    user: str
    gpus: int
    est_seconds: int
    state: JobState
    submitted_at: float
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def is_active(self) -> bool:
        return self.state in (JobState.QUEUED, JobState.RUNNING)

    def expected_finish(self) -> float:
        """Wall-clock time this running job is expected to release its GPUs."""
        if self.started_at is None:
            raise ValueError("expected_finish() is only defined for started jobs.")
        return self.started_at + self.est_seconds

    def gpu_seconds_consumed(self, now: float) -> float:
        """GPU-seconds this job has burned so far. Running jobs count time elapsed."""
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else now
        return self.gpus * max(0.0, end - self.started_at)
