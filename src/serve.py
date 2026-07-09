"""
Fair-share GPU scheduler API.

Jobs are submitted, queued, and started by a background loop that ticks every
TICK_SECONDS. Job execution is simulated: a job "runs" for its estimated
duration and then releases its GPUs. Simulating lets the scheduling policy be
demonstrated and tested without a GPU cluster attached, which is the part of the
system worth showing.
"""

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src import db, metrics
from src.config import settings
from src.fairshare import decayed_usage_by_user, fairshare_factor, order_queue, priority
from src.models import JobState
from src.scheduler import plan_tick


def run_tick(now: float) -> None:
    jobs = db.list_schedulable_jobs()
    plan = plan_tick(jobs, now, settings.TOTAL_GPUS)

    for job in plan.finish:
        db.mark_finished(job.id, now)
        metrics.observe_finish()
        logger.info(f"job {job.id} ({job.user}) finished")

    for job in plan.start:
        db.mark_started(job.id, now)
        metrics.observe_start(now - job.submitted_at)
        logger.info(f"job {job.id} ({job.user}) started on {job.gpus} GPU(s)")

    _refresh_metrics(now)


def _refresh_metrics(now: float) -> None:
    jobs = db.list_schedulable_jobs()
    running = [j for j in jobs if j.state == JobState.RUNNING]
    queued = [j for j in jobs if j.state == JobState.QUEUED]
    metrics.set_cluster_state(
        total=settings.TOTAL_GPUS,
        allocated=sum(j.gpus for j in running),
        queued=len(queued),
        running=len(running),
    )
    metrics.set_user_usage(decayed_usage_by_user(jobs, now))


async def _scheduler_loop() -> None:
    while True:
        try:
            run_tick(time.time())
        except Exception as exc:  # keep the loop alive; a bad tick must not kill the cluster
            logger.error(f"scheduler tick failed: {exc}")
        await asyncio.sleep(settings.TICK_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    task = asyncio.create_task(_scheduler_loop())
    logger.info(f"scheduler started with {settings.TOTAL_GPUS} GPUs")
    yield
    task.cancel()


app = FastAPI(
    title="Fair-Share GPU Scheduler",
    description="Queues GPU jobs across users with fair-share priority and EASY backfill.",
    version="1.0.0",
    lifespan=lifespan,
)

app.get("/metrics", include_in_schema=False)(metrics.metrics_endpoint)


class JobRequest(BaseModel):
    user: str = Field(..., min_length=1, max_length=64)
    gpus: int = Field(..., ge=1)
    est_seconds: int = Field(..., ge=1)


class JobResponse(BaseModel):
    id: int
    user: str
    gpus: int
    est_seconds: int
    state: JobState
    submitted_at: float
    started_at: float | None
    finished_at: float | None


@app.get("/health")
def health():
    return {"status": "ok", "total_gpus": settings.TOTAL_GPUS}


@app.post("/jobs", response_model=JobResponse, status_code=201)
def submit(request: JobRequest):
    # A job asking for more GPUs than exist could never be scheduled, and would
    # hold a reservation forever while smaller jobs backfill past it.
    if request.gpus > settings.TOTAL_GPUS:
        raise HTTPException(
            status_code=400,
            detail=f"Cluster has {settings.TOTAL_GPUS} GPUs; job requested {request.gpus}.",
        )
    if request.gpus > settings.MAX_GPUS_PER_JOB:
        raise HTTPException(status_code=400, detail="Job exceeds the per-job GPU limit.")
    if request.est_seconds > settings.MAX_EST_SECONDS:
        raise HTTPException(status_code=400, detail="Job exceeds the maximum runtime estimate.")

    job = db.submit_job(request.user, request.gpus, request.est_seconds, time.time())
    return job


@app.get("/jobs", response_model=list[JobResponse])
def list_jobs(state: JobState | None = None):
    return db.list_jobs(state)


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: int):
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.delete("/jobs/{job_id}")
def cancel(job_id: int):
    if db.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not db.cancel_job(job_id, time.time()):
        raise HTTPException(status_code=409, detail="Job has already finished.")
    return {"id": job_id, "state": JobState.CANCELLED}


@app.get("/queue")
def queue():
    """The queue in the order the scheduler would start jobs, with the priority
    components that put them there."""
    now = time.time()
    jobs = db.list_schedulable_jobs()
    usage = decayed_usage_by_user(jobs, now)
    known_users = {j.user for j in jobs}
    return [
        {
            "id": j.id,
            "user": j.user,
            "gpus": j.gpus,
            "est_seconds": j.est_seconds,
            "waited_seconds": round(now - j.submitted_at, 1),
            "fairshare_factor": round(fairshare_factor(j.user, usage, known_users), 4),
            "priority": round(priority(j, usage, known_users, now), 4),
        }
        for j in order_queue(jobs, now)
    ]


@app.get("/cluster")
def cluster():
    jobs = db.list_schedulable_jobs()
    running = [j for j in jobs if j.state == JobState.RUNNING]
    allocated = sum(j.gpus for j in running)
    return {
        "total_gpus": settings.TOTAL_GPUS,
        "allocated_gpus": allocated,
        "free_gpus": settings.TOTAL_GPUS - allocated,
        "utilization": round(allocated / settings.TOTAL_GPUS, 4),
        "running_jobs": len(running),
        "queued_jobs": sum(1 for j in jobs if j.state == JobState.QUEUED),
    }


@app.get("/usage")
def usage():
    """Decayed GPU-hours per user: the accounting behind fair-share."""
    now = time.time()
    jobs = db.list_schedulable_jobs()
    decayed = decayed_usage_by_user(jobs, now)
    known_users = {j.user for j in jobs}
    return {
        "half_life_hours": settings.HALF_LIFE_HOURS,
        "users": sorted(
            (
                {
                    "user": user,
                    "decayed_gpu_hours": round(gpu_hours, 4),
                    "fairshare_factor": round(fairshare_factor(user, decayed, known_users), 4),
                }
                for user, gpu_hours in decayed.items()
            ),
            key=lambda u: -u["decayed_gpu_hours"],
        ),
    }
