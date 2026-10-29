"""Optional browser regression coverage for the theme and live-game behavior."""

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest


playwright_api = pytest.importorskip("playwright.sync_api")


def test_theme_default_persistence_and_live_game_are_independent():
    port_probe = socket.socket()
    port_probe.bind(("127.0.0.1", 0))
    port = port_probe.getsockname()[1]
    port_probe.close()
    base_url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "web_app:app", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        for _ in range(100):
            if server.poll() is not None:
                pytest.fail("Uvicorn exited before its health endpoint became ready")
            try:
                urllib.request.urlopen(f"{base_url}/healthz", timeout=0.2).close()
                break
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        else:
            pytest.fail("Uvicorn did not become ready")

        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except Exception as exc:
                pytest.skip(f"Chromium is not installed for Playwright: {exc}")

            context = browser.new_context(color_scheme="light")
            page = context.new_page()
            page.goto(base_url, wait_until="networkidle")
            page.wait_for_selector("#board .tile")

            assert page.locator("html").get_attribute("data-theme") == "dark"
            assert page.locator("body").evaluate("element => getComputedStyle(element).backgroundColor") == "rgb(23, 28, 32)"
            toggle = page.locator("#theme-toggle")
            assert toggle.get_attribute("aria-label") == "Switch to light mode"
            assert toggle.locator(".icon-sun").evaluate("element => getComputedStyle(element).display") != "none"

            toggle.click()
            assert page.locator("html").get_attribute("data-theme") == "light"
            page.wait_for_function(
                "getComputedStyle(document.body).backgroundColor === 'rgb(244, 242, 236)'"
            )
            assert toggle.get_attribute("aria-label") == "Switch to dark mode"
            assert toggle.locator(".icon-moon").evaluate("element => getComputedStyle(element).display") != "none"
            assert page.locator("#ai-toggle").evaluate("element => getComputedStyle(element).color") == "rgb(255, 254, 250)"
            assert page.evaluate("localStorage.getItem('2048-ai-theme')") == "light"
            page.reload(wait_until="networkidle")
            assert page.locator("html").get_attribute("data-theme") == "light"
            assert page.locator("#theme-toggle").get_attribute("aria-label") == "Switch to dark mode"

            page.locator("#theme-toggle").click()
            assert page.locator("html").get_attribute("data-theme") == "dark"
            page.keyboard.press("ArrowLeft")
            page.wait_for_function("Number(document.querySelector('#moves').textContent.replaceAll(',', '')) > 0")

            before_toggle = {
                "board": page.locator("#board").inner_text(),
                "moves": page.locator("#moves").inner_text(),
                "score": page.locator("#score").inner_text(),
                "agent": page.locator("#agent-select").input_value(),
                "depth": page.locator("#depth-value").inner_text(),
                "speed": page.locator("#speed-select").input_value(),
            }
            api_requests = []
            page.on("request", lambda request: api_requests.append(request.url) if "/api/games" in request.url else None)
            page.locator("#theme-toggle").click()
            assert page.locator("#board").inner_text() == before_toggle["board"]
            assert page.locator("#moves").inner_text() == before_toggle["moves"]
            assert page.locator("#score").inner_text() == before_toggle["score"]
            assert page.locator("#agent-select").input_value() == before_toggle["agent"]
            assert page.locator("#depth-value").inner_text() == before_toggle["depth"]
            assert page.locator("#speed-select").input_value() == before_toggle["speed"]
            assert not api_requests

            page.select_option("#speed-select", "170")
            before_ai = int(before_toggle["moves"].replace(",", ""))
            page.locator("#ai-toggle").click()
            page.wait_for_function(
                "count => Number(document.querySelector('#moves').textContent.replaceAll(',', '')) > count",
                arg=before_ai,
                timeout=10000,
            )
            assert page.locator("#status-pill").get_attribute("data-status") == "playing"
            page.locator("#theme-toggle").click()
            page.wait_for_function(
                "count => Number(document.querySelector('#moves').textContent.replaceAll(',', '')) > count",
                arg=before_ai + 1,
                timeout=10000,
            )
            page.locator("#ai-toggle").click()
            page.wait_for_function(
                "document.querySelector('#status-pill').dataset.status === 'stopped'",
                timeout=5000,
            )
            stopped_moves = int(page.locator("#moves").inner_text().replace(",", ""))
            page.keyboard.press("ArrowUp")
            page.wait_for_function(
                "count => Number(document.querySelector('#moves').textContent.replaceAll(',', '')) > count",
                arg=stopped_moves,
                timeout=5000,
            )

            page.set_viewport_size({"width": 375, "height": 812})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert page.locator("#theme-toggle").is_visible()

            blocked_context = browser.new_context(color_scheme="light")
            blocked_context.add_init_script(
                "Object.defineProperty(window, 'localStorage', {get() { throw new DOMException('blocked', 'SecurityError'); }});"
            )
            blocked_page = blocked_context.new_page()
            blocked_page.goto(base_url, wait_until="networkidle")
            assert blocked_page.locator("html").get_attribute("data-theme") == "dark"
            blocked_page.locator("#theme-toggle").click()
            assert blocked_page.locator("html").get_attribute("data-theme") == "light"

            blocked_context.close()
            context.close()
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
