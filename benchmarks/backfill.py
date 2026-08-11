"""
What is backfill actually worth?

Runs one synthetic workload through the scheduler twice (once in strict
priority order, once with EASY backfill) and reports the difference. The
cluster is simulated, not real: jobs run for exactly their estimate, so this
measures the scheduling policy in isolation rather than a production cluster.

    python benchmarks/backfill.py

The workload is a fixed seed, so the numbers in the README are reproducible.
"""

import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import Job, JobState  # noqa: E402
from src.scheduler import plan_tick  # noqa: E402

TOTAL_GPUS = 8
NUM_JOBS = 400
USERS = ["ada", "bela", "cem", "dora", "emre"]
SEED = 20260809

# A mix that makes head-of-line blocking bite: most jobs are small and short,
# but a minority ask for most of the cluster, and those are what the queue
# stalls behind.
SHAPES = [
    # (gpus, seconds, weight)
    (1, 600, 40),
    (1, 3600, 20),
    (2, 1800, 15),
    (4, 7200, 15),
    (8, 10800, 10),
]


def make_workload() -> list[Job]:
    rng = random.Random(SEED)
    population = [(g, s) for g, s, w in SHAPES for _ in range(w)]

    jobs = []
    submitted = 0.0
    for job_id in range(1, NUM_JOBS + 1):
        gpus, est = rng.choice(population)
        # Arrivals are bursty: usually seconds apart, occasionally a long gap.
        submitted += rng.expovariate(1 / 400.0)
        jobs.append(
            Job(
                id=job_id,
                user=rng.choice(USERS),
                gpus=gpus,
                est_seconds=est,
                state=JobState.QUEUED,
                submitted_at=submitted,
            )
        )
    return jobs


def simulate(jobs: list[Job], backfill: bool) -> dict[str, float]:
    """Advance an event-driven clock until every job has finished."""
    state = {j.id: replace(j) for j in jobs}
    arrivals = sorted({j.submitted_at for j in jobs})
    now = arrivals[0]
    gpu_seconds_used = 0.0
    waits: list[float] = []

    while True:
        # Only jobs that have actually been submitted are visible to the planner.
        visible = [j for j in state.values() if j.submitted_at <= now]
        plan = plan_tick(visible, now, TOTAL_GPUS, backfill=backfill)

        for job in plan.finish:
            state[job.id] = replace(job, state=JobState.COMPLETED, finished_at=now)
        for job in plan.start:
            state[job.id] = replace(job, state=JobState.RUNNING, started_at=now)
            waits.append(now - job.submitted_at)

        running = [j for j in state.values() if j.state == JobState.RUNNING]
        pending = [j for j in state.values() if j.state == JobState.QUEUED]
        if not running and not pending:
            break

        # Jump straight to the next moment anything can change: a running job
        # finishing, or a queued job arriving.
        candidates = [j.expected_finish() for j in running]
        candidates += [j.submitted_at for j in state.values() if j.submitted_at > now]
        if not candidates:
            break
        nxt = min(candidates)

        gpu_seconds_used += sum(j.gpus for j in running) * (nxt - now)
        now = nxt

    makespan = now - arrivals[0]
    return {
        "makespan_hours": makespan / 3600.0,
        "utilisation_pct": 100.0 * gpu_seconds_used / (TOTAL_GPUS * makespan),
        "mean_wait_hours": statistics.mean(waits) / 3600.0,
        "median_wait_hours": statistics.median(waits) / 3600.0,
        "p95_wait_hours": sorted(waits)[int(0.95 * len(waits))] / 3600.0,
    }


def main() -> None:
    jobs = make_workload()
    strict = simulate(jobs, backfill=False)
    easy = simulate(jobs, backfill=True)

    print(f"{NUM_JOBS} jobs, {TOTAL_GPUS} GPUs, {len(USERS)} users, seed {SEED}\n")
    header = f"{'metric':<22}{'strict priority':>18}{'EASY backfill':>16}{'change':>12}"
    print(header)
    print("-" * len(header))
    for key, label, unit in [
        ("utilisation_pct", "GPU utilisation", "%"),
        ("makespan_hours", "Makespan", " h"),
        ("mean_wait_hours", "Mean queue wait", " h"),
        ("median_wait_hours", "Median queue wait", " h"),
        ("p95_wait_hours", "p95 queue wait", " h"),
    ]:
        a, b = strict[key], easy[key]
        delta = (b - a) / a * 100.0 if a else 0.0
        print(f"{label:<22}{a:>17.1f}{unit}{b:>15.1f}{unit}{delta:>11.1f}%")


if __name__ == "__main__":
    main()
