from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ModelConfig, Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.providers.fake import FakeProvider, FakeScenario  # noqa: E402

FAKE_MODELS = [
    ModelConfig(id="fake-a", label="Modell A", provider="fake", model="fake-a-model"),
    ModelConfig(id="fake-b", label="Modell B", provider="fake", model="fake-b-model"),
]


def make_settings(tmp_path: Path, **overrides) -> Settings:
    settings = Settings(
        env="test",
        db_path=tmp_path / "test.sqlite3",
        allow_fake_providers=True,
        models=list(FAKE_MODELS),
        request_timeout_s=5,
        **overrides,
    )
    settings.validate()
    return settings


@pytest.fixture(autouse=True)
def reset_fakes():
    FakeProvider.reset()
    yield
    FakeProvider.reset()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest_asyncio.fixture
async def client(settings: Settings):
    import httpx

    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", timeout=30
        ) as c:
            c.app = app
            yield c


@pytest_asyncio.fixture
async def session_id(client) -> str:
    resp = await client.post("/api/sessions", json={"title": "Testsitzung"})
    assert resp.status_code == 201
    return resp.json()["id"]


def scenario(model: str, **kwargs) -> FakeScenario:
    return FakeProvider.configure(model, FakeScenario(**kwargs))


async def wait_for_jobs(client, session_id: str, timeout: float = 10.0) -> dict:
    """Wartet, bis alle Aufträge der Sitzung einen Endzustand haben."""
    import asyncio
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        bundle = (await client.get(f"/api/sessions/{session_id}")).json()
        jobs = [j for entry in bundle["sparks"] for j in entry["jobs"]]
        if jobs and all(j["status"] not in ("queued", "running") for j in jobs):
            # Der Abschlusslauf (Einschätzung) folgt unmittelbar.
            if all(entry.get("summary") for entry in bundle["sparks"]):
                return bundle
        await asyncio.sleep(0.05)
    raise AssertionError("Aufträge wurden nicht rechtzeitig fertig")


@pytest_asyncio.fixture
async def live_server(settings):
    """Echter uvicorn-Server.

    Nötig für SSE: httpx' ASGITransport puffert Antworten vollständig und
    eignet sich deshalb nicht für einen laufenden Ereignisstrom.
    """
    import asyncio

    import uvicorn

    app = create_app(settings)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    else:  # pragma: no cover
        raise AssertionError("Server ist nicht gestartet")
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=10)
