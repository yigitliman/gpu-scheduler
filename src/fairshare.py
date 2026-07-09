"""
Fair-share priority.

Two forces decide which queued job runs next:

1. Fair-share factor. A user who has recently consumed more than their equal
   slice of the cluster is pushed down; a user who has consumed little is pulled
   up. Consumption is decayed with a half-life, so last week's heavy training run
   matters less than yesterday's, and eventually stops mattering at all. Without
   the decay, early heavy users would be penalised forever.

2. Age factor. Fair-share on its own starves a persistently heavy user: their
   factor decays towards zero, so a steady trickle of jobs from lighter users
   keeps out-ranking them forever. Priority therefore also grows with time spent
   queued, without an upper bound, so that any job eventually reaches the front.
   That guarantee is what makes the queue trustworthy to the people using it.

The fair-share formula is the classic one Slurm uses: F = 2 ** (-U / S), where U
is the user's share of total decayed usage and S is the share they are entitled
to (an equal slice). Using their exact entitlement gives F = 0.5, using nothing
gives F = 1.0, using double gives F = 0.25.
"""

from collections.abc import Iterable

from src.config import settings
from src.models import Job, JobState


def decay_weight(age_seconds: float, half_life_hours: float) -> float:
    """Weight in (0, 1] applied to usage that finished `age_seconds` ago."""
    if age_seconds <= 0:
        return 1.0
    half_lives = age_seconds / (half_life_hours * 3600.0)
    return 0.5**half_lives


def decayed_usage_by_user(jobs: Iterable[Job], now: float) -> dict[str, float]:
    """Decayed GPU-hours consumed per user.

    A finished job's contribution is decayed by how long ago it finished. A
    running job's contribution is counted at full weight, since it is happening
    right now and should immediately push the user down the queue.
    """
    usage: dict[str, float] = {}
    for job in jobs:
        if job.started_at is None:
            continue
        gpu_hours = job.gpu_seconds_consumed(now) / 3600.0
        if gpu_hours <= 0:
            continue
        age = (now - job.finished_at) if job.finished_at is not None else 0.0
        usage[job.user] = usage.get(job.user, 0.0) + gpu_hours * decay_weight(
            age, settings.HALF_LIFE_HOURS
        )
    return usage


def fairshare_factor(user: str, usage: dict[str, float], known_users: set[str]) -> float:
    """Fair-share factor in (0, 1]. Higher means more entitled to run next."""
    total = sum(usage.values())
    num_users = max(len(known_users), 1)
    if total <= 0:
        return 1.0

    share = 1.0 / num_users
    used_fraction = usage.get(user, 0.0) / total
    return 2.0 ** (-used_fraction / share)


def age_factor(job: Job, now: float) -> float:
    """Waiting-time bonus, deliberately unbounded.

    One unit is earned per AGE_UNIT_HOURS spent queued. Letting this grow without
    limit is what turns "a starved job eventually runs" from a hope into a
    guarantee: whatever a user's fair-share factor has decayed to, waiting long
    enough will out-weigh any fresh job. Slurm caps the equivalent term
    (PriorityMaxAge), which bounds how much age can distort fairness but gives up
    the guarantee. The trade here is the other way round.
    """
    waited_hours = max(0.0, now - job.submitted_at) / 3600.0
    return waited_hours / settings.AGE_UNIT_HOURS


def priority(job: Job, usage: dict[str, float], known_users: set[str], now: float) -> float:
    """Scheduling priority of a queued job. Higher runs first."""
    fair = fairshare_factor(job.user, usage, known_users)
    age = age_factor(job, now)
    return settings.W_FAIR * fair + settings.W_AGE * age


def order_queue(jobs: list[Job], now: float) -> list[Job]:
    """Queued jobs, highest priority first. Ties break on submission order (FIFO)."""
    queued = [j for j in jobs if j.state == JobState.QUEUED]
    usage = decayed_usage_by_user(jobs, now)
    known_users = {j.user for j in jobs}
    return sorted(
        queued,
        key=lambda j: (-priority(j, usage, known_users, now), j.submitted_at, j.id),
    )
