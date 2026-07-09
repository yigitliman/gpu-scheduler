from src.config import settings
from src.fairshare import (
    decay_weight,
    decayed_usage_by_user,
    fairshare_factor,
    order_queue,
)
from src.models import Job, JobState

HOUR = 3600.0
NOW = 1_000_000.0


def finished_job(
    job_id: int, user: str, gpus: int, ran_hours: float, ended_ago_hours: float
) -> Job:
    finished_at = NOW - ended_ago_hours * HOUR
    started_at = finished_at - ran_hours * HOUR
    return Job(
        id=job_id,
        user=user,
        gpus=gpus,
        est_seconds=int(ran_hours * HOUR),
        state=JobState.COMPLETED,
        submitted_at=started_at,
        started_at=started_at,
        finished_at=finished_at,
    )


def queued_job(job_id: int, user: str, gpus: int, waited_hours: float) -> Job:
    return Job(
        id=job_id,
        user=user,
        gpus=gpus,
        est_seconds=3600,
        state=JobState.QUEUED,
        submitted_at=NOW - waited_hours * HOUR,
    )


def test_decay_halves_usage_after_one_half_life():
    assert decay_weight(0, 168) == 1.0
    assert decay_weight(168 * HOUR, 168) == 0.5
    assert decay_weight(336 * HOUR, 168) == 0.25


def test_usage_from_old_job_decays_below_recent_job():
    old = finished_job(1, "old", gpus=1, ran_hours=1, ended_ago_hours=settings.HALF_LIFE_HOURS)
    recent = finished_job(2, "recent", gpus=1, ran_hours=1, ended_ago_hours=0)

    usage = decayed_usage_by_user([old, recent], NOW)
    assert usage["recent"] > usage["old"]
    assert usage["old"] == usage["recent"] * 0.5


def test_fairshare_factor_is_one_when_nothing_has_been_used():
    assert fairshare_factor("a", {}, {"a", "b"}) == 1.0


def test_fairshare_factor_is_half_at_exactly_the_entitled_share():
    # Two users, each entitled to 1/2. "a" used exactly half of all usage.
    assert fairshare_factor("a", {"a": 10.0, "b": 10.0}, {"a", "b"}) == 0.5


def test_fairshare_factor_punishes_over_use_and_rewards_under_use():
    usage = {"heavy": 90.0, "light": 10.0}
    users = {"heavy", "light"}
    assert fairshare_factor("heavy", usage, users) < 0.5
    assert fairshare_factor("light", usage, users) > 0.5


def test_lighter_user_is_queued_ahead_of_heavier_user():
    history = finished_job(1, "heavy", gpus=8, ran_hours=10, ended_ago_hours=0.1)
    heavy_job = queued_job(2, "heavy", gpus=1, waited_hours=0)
    light_job = queued_job(3, "light", gpus=1, waited_hours=0)

    assert [j.id for j in order_queue([history, heavy_job, light_job], NOW)] == [3, 2]


def test_aging_eventually_rescues_a_starved_heavy_user():
    """A heavy user's fair-share sinks, so only unbounded aging can put their job
    ahead of an endless stream of fresh jobs from lighter users."""
    history = finished_job(1, "heavy", gpus=8, ran_hours=500, ended_ago_hours=0.01)
    fresh_light = queued_job(2, "light", gpus=1, waited_hours=0)

    starved = queued_job(3, "heavy", gpus=1, waited_hours=1)
    assert order_queue([history, fresh_light, starved], NOW)[0].id == 2

    patient = queued_job(3, "heavy", gpus=1, waited_hours=500)
    assert order_queue([history, fresh_light, patient], NOW)[0].id == 3


def test_identical_jobs_are_ordered_deterministically():
    """Same user, same wait, so the same priority. The order must still be stable
    rather than dependent on however the rows came back from the database."""
    first = queued_job(1, "a", gpus=1, waited_hours=0)
    second = queued_job(2, "a", gpus=1, waited_hours=0)
    assert [j.id for j in order_queue([second, first], NOW)] == [1, 2]
