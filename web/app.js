const $ = (selector) => document.querySelector(selector);

const els = {
  themeToggle: $("#theme-toggle"),
  board: $("#board"), score: $("#score"), best: $("#best-score"), moves: $("#moves"), maxTile: $("#max-tile"),
  newGame: $("#new-game"), undo: $("#undo"), agent: $("#agent-select"), description: $("#agent-description"),
  depthRow: $("#depth-row"), depth: $("#depth-value"), depthMinus: $("#depth-minus"), depthPlus: $("#depth-plus"),
  aiToggle: $("#ai-toggle"), aiLabel: $("#ai-toggle-label"), status: $("#status-pill"), pilot: $("#pilot-label"),
  pilotSubtitle: $("#pilot-subtitle"), hint: $("#control-hint"), speed: $("#speed-select"), diagnostics: $("#diagnostic-content"),
  overlay: $("#overlay"), overlayTitle: $("#overlay-title"), overlayCopy: $("#overlay-copy"), overlayNew: $("#overlay-new"),
  benchmarkRows: $("#benchmark-rows"), benchmarkNote: $("#benchmark-note"),
};

let game = null;
let agentOptions = [];
const supportedAgentNames = new Set(["Greedy Heuristic", "Expectimax"]);
let aiRunning = false;
let aiLoopToken = 0;
let busy = false;
let pointerStart = null;

function setTheme(theme, persist = false) {
  const selected = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = selected;
  const nextAction = selected === "dark" ? "light" : "dark";
  const label = `Switch to ${nextAction} mode`;
  els.themeToggle.setAttribute("aria-label", label);
  els.themeToggle.title = label;
  const themeColor = getComputedStyle(document.documentElement).getPropertyValue("--meta-theme").trim();
  document.querySelector('meta[name="theme-color"]').content = themeColor;
  if (persist) {
    try { window.localStorage.setItem("2048-ai-theme", selected); }
    catch (_) { /* The current page can still toggle when storage is disabled. */ }
  }
}

function toggleTheme() {
  const current = document.documentElement.dataset.theme === "light" ? "light" : "dark";
  setTheme(current === "dark" ? "light" : "dark", true);
}

setTheme(document.documentElement.dataset.theme, false);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = response.status === 204 ? null : await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data?.detail || `Request failed (${response.status})`);
  return data;
}

function gamePath(suffix = "") { return `/api/games/${game.id}${suffix}`; }

function update(next) {
  game = next;
  render();
}

function tileClass(value) {
  if (!value) return "tile tile-v0";
  if (value <= 2048) return `tile tile-v${value}`;
  return "tile tile-large";
}

function renderBoard() {
  const fragment = document.createDocumentFragment();
  game.board.flat().forEach((value, index) => {
    const tile = document.createElement("div");
    tile.className = tileClass(value);
    tile.setAttribute("role", "gridcell");
    tile.setAttribute("aria-label", value ? `Row ${Math.floor(index / 4) + 1}, column ${index % 4 + 1}: ${value}` : "Empty");
    tile.textContent = value ? String(value) : "";
    fragment.append(tile);
  });
  els.board.replaceChildren(fragment);
}

function renderAgents() {
  const previous = game.agent;
  els.agent.replaceChildren();
  agentOptions.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = item.available ? item.name : `${item.name} · unavailable`;
    option.disabled = !item.available;
    els.agent.append(option);
  });
  els.agent.value = previous;
  const chosen = agentOptions.find((item) => item.name === game.agent);
  els.description.textContent = chosen?.description || "Choose an available agent.";
  if (chosen && !chosen.available && chosen.reason) els.description.textContent += ` ${chosen.reason}`;
  els.depthRow.hidden = game.agent !== "Expectimax";
  els.depth.textContent = String(game.depth);
}

function renderStatus() {
  const labels = {
    ready: "HUMAN READY", human: "HUMAN PLAYING", stopped: "AI STOPPED", thinking: "AI THINKING",
    playing: "AI PLAYING", finished: "GAME FINISHED", error: "AGENT ERROR",
  };
  const status = game.status || "ready";
  els.status.dataset.status = status;
  els.status.querySelector("span").textContent = labels[status] || "HUMAN READY";
  els.pilot.textContent = game.control === "ai" ? game.agent : "You";
  els.pilotSubtitle.textContent = game.control === "ai" ? `Autoplay · ${game.agent}` : "Keyboard · touch · your call";
  const selected = agentOptions.find((item) => item.name === game.agent);
  const canAI = selected?.available && !game.game_over;
  els.aiToggle.disabled = busy || (!aiRunning && !canAI);
  els.aiLabel.textContent = aiRunning || game.control === "ai" ? "Take Control" : "Let AI Play";
  els.aiToggle.classList.toggle("taking-control", aiRunning || game.control === "ai");
  els.hint.textContent = game.error || (game.won && !game.game_over
    ? "2048 reached. Keep playing to build a higher tile, or take control from the AI."
    : game.game_over
    ? "The board is full. Start a new game to play again."
    : game.control === "ai"
      ? "AI is moving on this board. Take control to stop it; no pending search can alter your next move."
      : "The AI takes over from this exact board. Your game and random sequence stay intact.");
  if (game.last_decision) {
    const d = game.last_decision;
    const detail = [
      `${d.source === "ai" ? d.agent : "Human"} → ${d.action}`,
      d.latency_ms != null ? `decision ${d.latency_ms} ms` : null,
      ...Object.entries(d.diagnostics || {}).slice(0, 8).map(([key, value]) => `${key}: ${typeof value === "number" ? Number(value.toFixed(4)) : value}`),
    ].filter(Boolean);
    els.diagnostics.textContent = detail.join("\n");
  } else {
    els.diagnostics.textContent = "Make a move to see decision details.";
  }
}

function renderOverlay() {
  if (game.game_over) {
    els.overlay.hidden = false;
    els.overlayTitle.textContent = "Board complete";
    els.overlayCopy.textContent = `Final score ${game.score.toLocaleString()} after ${game.move_count} moves.`;
  } else {
    els.overlay.hidden = true;
  }
}

function render() {
  agentOptions = (game.agents || agentOptions).filter((item) => supportedAgentNames.has(item.name));
  renderBoard();
  renderAgents();
  els.score.textContent = Number(game.score).toLocaleString();
  els.best.textContent = Number(game.best_score).toLocaleString();
  els.moves.textContent = Number(game.move_count).toLocaleString();
  els.maxTile.textContent = game.max_tile ? Number(game.max_tile).toLocaleString() : "—";
  els.undo.disabled = busy || aiRunning || !game.can_undo;
  els.newGame.disabled = busy;
  els.agent.disabled = busy;
  els.depthMinus.disabled = busy || game.depth <= 1;
  els.depthPlus.disabled = busy || game.depth >= 3;
  renderStatus();
  renderOverlay();
}

function setBusy(value) { busy = value; if (game) render(); }

async function startSession() {
  setBusy(true);
  try {
    update(await api("/api/games", { method: "POST", body: JSON.stringify({}) }));
    await loadBenchmarks();
  } catch (error) {
    showError(error);
  } finally { setBusy(false); }
}

function showError(error) {
  if (game) {
    game = { ...game, error: error.message, status: "error" };
    render();
  } else {
    els.hint.textContent = error.message;
  }
}

async function sendHumanMove(action) {
  if (!game || busy || aiRunning || game.control !== "human" || game.game_over) return;
  setBusy(true);
  try {
    update(await api(gamePath("/moves"), {
      method: "POST", body: JSON.stringify({ action, expected_version: game.version }),
    }));
  } catch (error) { showError(error); }
  finally { setBusy(false); }
}

const wait = (duration) => new Promise((resolve) => window.setTimeout(resolve, duration));

async function aiLoop(token) {
  while (aiRunning && token === aiLoopToken && !game.game_over) {
    await wait(Number(els.speed.value));
    if (!aiRunning || token !== aiLoopToken) return;
    try {
      const next = await api(gamePath("/ai-move"), {
        method: "POST", body: JSON.stringify({ expected_version: game.version }),
      });
      if (token !== aiLoopToken || !aiRunning) return;
      update(next);
    } catch (error) {
      if (token !== aiLoopToken) return;
      aiRunning = false;
      aiLoopToken += 1;
      try { update(await api(gamePath())); } catch (_) { /* retain the last browser snapshot */ }
      showError(error);
      return;
    }
  }
  if (token === aiLoopToken && game?.game_over) {
    aiRunning = false;
    render();
  }
}

async function takeControl() {
  // Invalidate the browser loop immediately; the server version also cancels
  // any in-flight search before it can commit an outdated decision.
  aiRunning = false;
  aiLoopToken += 1;
  render();
  try { update(await api(gamePath("/control"), { method: "POST", body: JSON.stringify({ control: "human" }) })); }
  catch (error) { showError(error); }
}

async function letAIPlay() {
  if (busy || !game || game.game_over) return;
  const selected = agentOptions.find((item) => item.name === game.agent);
  if (!selected?.available) return;
  setBusy(true);
  try {
    update(await api(gamePath("/control"), { method: "POST", body: JSON.stringify({ control: "ai" }) }));
    aiRunning = true;
    aiLoopToken += 1;
    render();
    void aiLoop(aiLoopToken);
  } catch (error) { showError(error); }
  finally { setBusy(false); }
}

async function resetGame() {
  aiRunning = false;
  aiLoopToken += 1;
  setBusy(true);
  try { update(await api(gamePath("/new"), { method: "POST", body: JSON.stringify({}) })); }
  catch (error) { showError(error); }
  finally { setBusy(false); }
}

async function undoMove() {
  aiRunning = false;
  aiLoopToken += 1;
  setBusy(true);
  try { update(await api(gamePath("/undo"), { method: "POST" })); }
  catch (error) { showError(error); }
  finally { setBusy(false); }
}

async function configureAgent() {
  const selected = els.agent.value;
  const depth = Number(els.depth.value);
  aiRunning = false;
  aiLoopToken += 1;
  setBusy(true);
  try {
    update(await api(gamePath("/agent"), {
      method: "PATCH", body: JSON.stringify({ name: selected, depth }),
    }));
  } catch (error) { showError(error); }
  finally { setBusy(false); }
}

async function changeDepth(delta) {
  const next = Math.max(1, Math.min(3, game.depth + delta));
  if (next === game.depth) return;
  els.depth.value = String(next);
  aiRunning = false;
  aiLoopToken += 1;
  // The endpoint applies depth and agent selection atomically.
  setBusy(true);
  try {
    update(await api(gamePath("/agent"), {
      method: "PATCH", body: JSON.stringify({ name: game.agent, depth: next }),
    }));
  } catch (error) { showError(error); }
  finally { setBusy(false); }
}

async function loadBenchmarks() {
  try {
    const data = await api("/api/benchmarks");
    els.benchmarkRows.replaceChildren();
    if (!data.rows.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 5;
      cell.className = "table-loading";
      cell.textContent = "No verified canonical summaries are available in this deployment.";
      row.append(cell);
      els.benchmarkRows.append(row);
      return;
    }
    for (const item of data.rows) {
      const row = document.createElement("tr");
      const values = [item.agent, Number(item.mean_score).toLocaleString(undefined, { maximumFractionDigits: 2 }), Number(item.mean_moves).toLocaleString(undefined, { maximumFractionDigits: 2 }), Number(item.max_tile_observed).toLocaleString(), `${Number(item.reached_2048_pct).toLocaleString()}%`];
      values.forEach((value) => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); });
      els.benchmarkRows.append(row);
    }
    els.benchmarkNote.textContent = `${data.conditions} ${data.rng_note}`;
  } catch (error) {
    els.benchmarkRows.textContent = error.message;
  }
}

els.newGame.addEventListener("click", resetGame);
els.themeToggle.addEventListener("click", toggleTheme);
els.overlayNew.addEventListener("click", resetGame);
els.undo.addEventListener("click", undoMove);
els.aiToggle.addEventListener("click", () => (aiRunning || game?.control === "ai") ? takeControl() : letAIPlay());
els.agent.addEventListener("change", configureAgent);
els.depthMinus.addEventListener("click", () => changeDepth(-1));
els.depthPlus.addEventListener("click", () => changeDepth(1));

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement || event.target instanceof HTMLTextAreaElement) return;
  const key = event.key.toLowerCase();
  const keys = { arrowup: "up", w: "up", arrowright: "right", d: "right", arrowdown: "down", s: "down", arrowleft: "left", a: "left" };
  if (keys[key]) { event.preventDefault(); void sendHumanMove(keys[key]); }
});

els.board.addEventListener("pointerdown", (event) => {
  pointerStart = { x: event.clientX, y: event.clientY };
  els.board.setPointerCapture?.(event.pointerId);
});
els.board.addEventListener("pointerup", (event) => {
  if (!pointerStart) return;
  const dx = event.clientX - pointerStart.x;
  const dy = event.clientY - pointerStart.y;
  pointerStart = null;
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 28) return;
  void sendHumanMove(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? "right" : "left") : (dy > 0 ? "down" : "up"));
});
els.board.addEventListener("pointercancel", () => { pointerStart = null; });

void startSession();
