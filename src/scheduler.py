"""
The scheduling decision, as a pure function.

`plan_tick` takes the current jobs and the wall clock and returns what should
happen: which running jobs are done, and which queued jobs should start. It
touches no database and no global clock, which is what makes the fairness and
backfill behaviour straightforward to test.

Backfill is the part that separates a scheduler from a queue. When the
highest-priority job cannot fit in the free GPUs, we do not simply idle the
cluster until it can. We compute the moment enough GPUs will have been released
for it (its reservation), then let lower-priority jobs run in the gap, but only
those expected to finish before that moment. A backfilled job therefore cannot
delay the job it jumped ahead of. This is the conservative variant, commonly
called EASY backfill.

The reservation relies on user-supplied runtime estimates, which is a real and
deliberate weakness: it is also how Slurm behaves. A job that overruns its
estimate does delay the reserved job. See the README for how that is mitigated.
"""

from dataclasses import dataclass, replace

from src.fairshare import decayed_usage_by_user, order_queue
from src.models import Job, JobState


@dataclass
class TickPlan:
    finish: list[Job]
    start: list[Job]


def _earliest_time_with_gpus(running: list[Job], needed: int, free_now: int, now: float) -> float:
    """When will `needed` GPUs be free, assuming running jobs finish on estimate?

    Returns infinity if the request can never be satisfied by the running set,
    which happens only when the job asks for more GPUs than the cluster has.
    """
    if free_now >= needed:
        return now

    free = free_now
    for job in sorted(running, key=lambda j: j.expected_finish()):
        free += job.gpus
        if free >= needed:
            return job.expected_finish()
    return float("inf")


def plan_tick(
    jobs: list[Job], now: float, total_gpus: int, backfill: bool = True
) -> TickPlan:
    """Decide what happens this tick.

    `backfill=False` reduces the policy to strict priority order, where the head
    of the queue blocks everything behind it. Nothing in the service turns it
    off; it exists so the benchmark can measure what backfill is worth.
    """
    finish = [j for j in jobs if j.state == JobState.RUNNING and now >= j.expected_finish()]
    finished_ids = {j.id for j in finish}

    # A simulated view of the cluster: finished jobs have released their GPUs,
    # and each job we decide to start is immediately marked running so that the
    # next iteration sees the reduced capacity and the refreshed priorities.
    sim: dict[int, Job] = {j.id: j for j in jobs if j.id not in finished_ids}
    started_ids: list[int] = []

    def running() -> list[Job]:
        return [j for j in sim.values() if j.state == JobState.RUNNING]

    # Tracked incrementally rather than recounted: the loop below asks for the
    # free capacity once per queued job it considers.
    allocated = sum(j.gpus for j in running())

    def free_gpus() -> int:
        return total_gpus - allocated

    # Priorities are fixed for the whole tick: a job started now has consumed no
    # GPU-seconds yet, so starting one changes neither the usage map nor anyone's
    # fair-share factor. The queue is therefore ordered once and then walked,
    # rather than re-derived (a full rescan of every job's history plus a re-sort)
    # after each job that starts.
    visible = list(sim.values())
    usage = decayed_usage_by_user(visible, now)
    known_users = {j.user for j in visible}
    queue = order_queue(visible, now, usage, known_users)

    index = 0
    while index < len(queue):
        head = queue[index]
        if head.gpus <= free_gpus():
            sim[head.id] = replace(head, state=JobState.RUNNING, started_at=now)
            started_ids.append(head.id)
            allocated += head.gpus
            index += 1
            continue

        if not backfill:
            break

        # The head cannot run yet. Hold a reservation for it, then let smaller
        # jobs use the idle GPUs, but only if they finish before the reservation.
        reservation = _earliest_time_with_gpus(running(), head.gpus, free_gpus(), now)
        for job in queue[index + 1 :]:
            if job.gpus <= free_gpus() and now + job.est_seconds <= reservation:
                sim[job.id] = replace(job, state=JobState.RUNNING, started_at=now)
                started_ids.append(job.id)
                allocated += job.gpus
        break

    by_id = {j.id: j for j in jobs}
    return TickPlan(finish=finish, start=[by_id[i] for i in started_ids])
