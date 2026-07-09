"""
Prometheus metrics for the scheduler.

A cluster operator cares about two things a job queue can hide: are the GPUs
actually busy, and is anyone waiting far longer than everyone else. Utilisation
and a queue-wait histogram answer both. Per-user decayed usage is exported as a
gauge so the fairness decisions the scheduler makes are auditable from a
dashboard rather than only from the code.
"""

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

GPUS_TOTAL = Gauge("cluster_gpus_total", "GPUs in the cluster.")
GPUS_ALLOCATED = Gauge("cluster_gpus_allocated", "GPUs currently allocated to running jobs.")
QUEUE_DEPTH = Gauge("scheduler_queue_depth", "Jobs waiting in the queue.")
RUNNING_JOBS = Gauge("scheduler_running_jobs", "Jobs currently running.")

USER_DECAYED_USAGE = Gauge(
    "scheduler_user_decayed_gpu_hours",
    "Decayed GPU-hours attributed to each user, the input to fair-share.",
    ["user"],
)

JOBS_STARTED = Counter("scheduler_jobs_started_total", "Jobs moved from queued to running.")
JOBS_FINISHED = Counter("scheduler_jobs_finished_total", "Jobs that ran to completion.")

QUEUE_WAIT = Histogram(
    "scheduler_queue_wait_seconds",
    "Time a job spent queued before it started.",
    buckets=[1, 5, 15, 30, 60, 300, 900, 3600, 14400],
)


def observe_start(wait_seconds: float) -> None:
    JOBS_STARTED.inc()
    QUEUE_WAIT.observe(max(0.0, wait_seconds))


def observe_finish() -> None:
    JOBS_FINISHED.inc()


def set_cluster_state(total: int, allocated: int, queued: int, running: int) -> None:
    GPUS_TOTAL.set(total)
    GPUS_ALLOCATED.set(allocated)
    QUEUE_DEPTH.set(queued)
    RUNNING_JOBS.set(running)


def set_user_usage(usage: dict[str, float]) -> None:
    # Clearing first keeps a user's series from freezing at its last value after
    # their usage has decayed away entirely.
    USER_DECAYED_USAGE.clear()
    for user, gpu_hours in usage.items():
        USER_DECAYED_USAGE.labels(user=user).set(gpu_hours)


def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
