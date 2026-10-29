"""Ensure the deployed FastAPI app runs without importing DQN dependencies."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_dependencies_exclude_ml_and_ui_only_packages():
    runtime = (ROOT / "requirements.txt").read_text().lower()
    development = (ROOT / "requirements-dev.txt").read_text().lower()
    dockerfile = (ROOT / "Dockerfile").read_text().lower()
    assert "torch" not in runtime
    assert "pygame" not in runtime
    assert "matplotlib" not in runtime
    assert "torch" in development
    assert "pip install --no-cache-dir -r requirements.txt" in dockerfile
    assert "torch" not in dockerfile
    assert "copy . ." not in dockerfile


def _request(base_url, path, payload=None, method=None):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url + path,
        data=body,
        method=method or ("POST" if body is not None else "GET"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def test_web_app_starts_and_supports_gameplay_without_torch():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    guarded_server = r"""
import builtins
import os
import uvicorn

real_import = builtins.__import__
blocked = {"torch", "agent", "legacy_agent", "arena_agents"}
def guarded_import(name, *args, **kwargs):
    if any(name == item or name.startswith(item + ".") for item in blocked):
        raise AssertionError("production web import attempted DQN module: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
uvicorn.run("web_app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", guarded_server],
        cwd=ROOT,
        env={**os.environ, "PORT": str(port)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise AssertionError("Uvicorn exited before the guarded web app started")
            try:
                health = _request(base_url, "/healthz")
                if health == {"status": "ok"}:
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        else:
            raise AssertionError("the guarded web app did not start")

        state = _request(base_url, "/api/games", {})
        assert [option["name"] for option in state["agents"]] == ["Greedy Heuristic", "Expectimax"]

        # Human moves work before and after handing the unchanged board to AI.
        state = _request(base_url, f"/api/games/{state['id']}/moves", {
            "action": state["valid_actions"][0], "expected_version": state["version"],
        })
        assert state["move_count"] == 1

        for agent_name, depth in (("Greedy Heuristic", 2), ("Expectimax", 1)):
            state = _request(base_url, f"/api/games/{state['id']}/agent", {
                "name": agent_name, "depth": depth,
            }, method="PATCH")
            state = _request(base_url, f"/api/games/{state['id']}/control", {"control": "ai"})
            state = _request(base_url, f"/api/games/{state['id']}/ai-move", {
                "expected_version": state["version"],
            })
            assert state["move_count"] >= 2
            assert state["last_decision"]["agent"] == agent_name
            if agent_name == "Expectimax":
                assert state["last_decision"]["diagnostics"]["search_depth"] == 1

            board_after_ai = state["board"]
            state = _request(base_url, f"/api/games/{state['id']}/control", {"control": "human"})
            assert state["board"] == board_after_ai
            state = _request(base_url, f"/api/games/{state['id']}/moves", {
                "action": state["valid_actions"][0], "expected_version": state["version"],
            })
            assert state["control"] == "human"

        # Unsupported legacy/DQN options are rejected, not advertised as unavailable.
        try:
            _request(base_url, f"/api/games/{state['id']}/agent", {
                "name": "Raw-Score Dueling DQN", "depth": 2,
            }, method="PATCH")
        except urllib.error.HTTPError as error:
            assert error.code == 422
        else:
            raise AssertionError("a DQN agent unexpectedly remained available")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
