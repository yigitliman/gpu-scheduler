from src.models import Job, JobState
from src.scheduler import plan_tick


def running(job_id: int, user: str, gpus: int, started_at: float, est_seconds: int) -> Job:
    return Job(
        id=job_id,
        user=user,
        gpus=gpus,
        est_seconds=est_seconds,
        state=JobState.RUNNING,
        submitted_at=started_at,
        started_at=started_at,
    )


def queued(job_id: int, user: str, gpus: int, est_seconds: int, submitted_at: float = 0.0) -> Job:
    return Job(
        id=job_id,
        user=user,
        gpus=gpus,
        est_seconds=est_seconds,
        state=JobState.QUEUED,
        submitted_at=submitted_at,
    )


def test_a_job_that_fits_starts_immediately():
    plan = plan_tick([queued(1, "a", gpus=2, est_seconds=60)], now=0.0, total_gpus=4)
    assert [j.id for j in plan.start] == [1]


def test_running_job_finishes_once_its_estimate_elapses():
    job = running(1, "a", gpus=2, started_at=0.0, est_seconds=100)
    assert plan_tick([job], now=99.0, total_gpus=4).finish == []
    assert [j.id for j in plan_tick([job], now=100.0, total_gpus=4).finish] == [1]


def test_scheduler_never_allocates_more_gpus_than_the_cluster_has():
    jobs = [queued(i, f"u{i}", gpus=3, est_seconds=60) for i in range(1, 6)]
    plan = plan_tick(jobs, now=0.0, total_gpus=8)
    assert sum(j.gpus for j in plan.start) <= 8


def test_freed_gpus_are_reused_within_the_same_tick():
    """A job finishing on this tick must release its GPUs to a job starting on it,
    otherwise the cluster idles for a full tick after every completion."""
    done = running(1, "a", gpus=4, started_at=0.0, est_seconds=100)
    waiting = queued(2, "b", gpus=4, est_seconds=60, submitted_at=10.0)

    plan = plan_tick([done, waiting], now=100.0, total_gpus=4)
    assert [j.id for j in plan.finish] == [1]
    assert [j.id for j in plan.start] == [2]


def test_short_job_backfills_into_the_gap_ahead_of_a_blocked_big_job():
    # 2 of 4 GPUs are busy until t=100. The head of the queue wants all 4, so it
    # is reserved for t=100. A 1-GPU job finishing at t=50 fits in the gap.
    blocker = running(1, "r", gpus=2, started_at=0.0, est_seconds=100)
    big = queued(2, "a", gpus=4, est_seconds=60)
    short = queued(3, "b", gpus=1, est_seconds=50)

    plan = plan_tick([blocker, big, short], now=0.0, total_gpus=4)
    assert [j.id for j in plan.start] == [3]


def test_backfill_refuses_a_job_that_would_overrun_the_reservation():
    """The whole point of the reservation: a backfilled job must not push back the
    higher-priority job it jumped ahead of."""
    blocker = running(1, "r", gpus=2, started_at=0.0, est_seconds=100)
    big = queued(2, "a", gpus=4, est_seconds=60)
    long_job = queued(3, "b", gpus=1, est_seconds=200)

    plan = plan_tick([blocker, big, long_job], now=0.0, total_gpus=4)
    assert plan.start == []


def test_backfill_takes_the_short_job_and_leaves_the_long_one():
    blocker = running(1, "r", gpus=2, started_at=0.0, est_seconds=100)
    big = queued(2, "a", gpus=4, est_seconds=60)
    short = queued(3, "b", gpus=1, est_seconds=50)
    long_job = queued(4, "c", gpus=1, est_seconds=200)

    plan = plan_tick([blocker, big, short, long_job], now=0.0, total_gpus=4)
    assert [j.id for j in plan.start] == [3]


def test_job_larger_than_the_cluster_does_not_block_the_queue_forever():
    """It cannot be scheduled, so it must not hold a reservation that stops
    everything else. The API rejects such jobs; the scheduler stays safe anyway."""
    impossible = queued(1, "a", gpus=16, est_seconds=60)
    normal = queued(2, "b", gpus=1, est_seconds=60)

    plan = plan_tick([impossible, normal], now=0.0, total_gpus=8)
    assert [j.id for j in plan.start] == [2]


def test_cancelled_jobs_are_ignored():
    cancelled = Job(
        id=1,
        user="a",
        gpus=8,
        est_seconds=60,
        state=JobState.CANCELLED,
        submitted_at=0.0,
    )
    plan = plan_tick([cancelled, queued(2, "b", gpus=8, est_seconds=60)], now=0.0, total_gpus=8)
    assert [j.id for j in plan.start] == [2]
