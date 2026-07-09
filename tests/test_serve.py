import time

import pytest
from fastapi.testclient import TestClient

from src import db
from src.config import settings
from src.models import JobState
from src.serve import app, run_tick


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "scheduler.db"))
    monkeypatch.setattr(settings, "TOTAL_GPUS", 4)
    # Park the background loop so it cannot start jobs mid-assertion.
    monkeypatch.setattr(settings, "TICK_SECONDS", 3600.0)

    with TestClient(app) as c:
        yield c


def submit(client, user: str, gpus: int, est_seconds: int = 60):
    return client.post("/jobs", json={"user": user, "gpus": gpus, "est_seconds": est_seconds})


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["total_gpus"] == 4


def test_submitted_job_is_queued(client):
    response = submit(client, "alice", gpus=1)
    assert response.status_code == 201
    assert response.json()["state"] == JobState.QUEUED


def test_job_asking_for_more_gpus_than_the_cluster_has_is_rejected(client):
    response = submit(client, "alice", gpus=99)
    assert response.status_code == 400


def test_unknown_job_returns_404(client):
    assert client.get("/jobs/424242").status_code == 404
    assert client.delete("/jobs/424242").status_code == 404


def test_queued_job_can_be_cancelled_once(client):
    job_id = submit(client, "alice", gpus=1).json()["id"]
    assert client.delete(f"/jobs/{job_id}").status_code == 200
    assert client.delete(f"/jobs/{job_id}").status_code == 409


def test_queue_exposes_priority_components(client):
    submit(client, "alice", gpus=1)
    entries = client.get("/queue").json()
    assert len(entries) == 1
    assert {"fairshare_factor", "priority", "waited_seconds"} <= set(entries[0])


def test_cluster_reports_free_capacity(client):
    body = client.get("/cluster").json()
    assert body["total_gpus"] == 4
    assert body["free_gpus"] == 4
    assert body["utilization"] == 0.0


def test_metrics_endpoint_serves_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "scheduler_queue_depth" in response.text


def test_tick_starts_queued_jobs_and_updates_the_cluster(client):

    submit(client, "alice", gpus=2, est_seconds=100)
    submit(client, "bob", gpus=2, est_seconds=100)

    run_tick(now=1000.0)

    assert {j.state for j in db.list_jobs()} == {JobState.RUNNING}
    assert client.get("/cluster").json()["free_gpus"] == 0


def test_tick_finishes_jobs_and_releases_their_gpus(client):

    submit(client, "alice", gpus=4, est_seconds=100)
    run_tick(now=1000.0)
    assert client.get("/cluster").json()["free_gpus"] == 0

    run_tick(now=1000.0 + 100)
    assert client.get("/cluster").json()["free_gpus"] == 4
    assert db.list_jobs()[0].state == JobState.COMPLETED


def test_usage_accounting_reflects_a_finished_job(client):
    """The job must sit on the real timeline: /usage decays usage against the wall
    clock, so a job that "finished" at t=0 would have decayed away to nothing."""

    started = time.time() - 3600
    submit(client, "alice", gpus=2, est_seconds=3600)
    run_tick(now=started)
    run_tick(now=started + 3600)

    users = client.get("/usage").json()["users"]
    alice = next(u for u in users if u["user"] == "alice")
    assert alice["decayed_gpu_hours"] == pytest.approx(2.0, rel=1e-3)
