"""FastAPI entry point for the 2048 browser demo."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web_game import (
    AgentFailure,
    GameService,
    GameServiceError,
)


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("2048-web")
WEB_DIR = Path(__file__).resolve().parent / "web"
service = GameService()
AI_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="2048-ai")
app = FastAPI(title="2048 AI Lab", version="1.0.0")


class NewGameRequest(BaseModel):
    seed: Optional[int] = None


class MoveRequest(BaseModel):
    action: Literal["up", "right", "down", "left"]
    expected_version: int = Field(ge=0)


class AIMoveRequest(BaseModel):
    expected_version: int = Field(ge=0)


class ControlRequest(BaseModel):
    control: Literal["human", "ai"]


class AgentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    depth: int = Field(default=2, ge=1, le=3)


def service_error_response(exc: GameServiceError) -> JSONResponse:
    if isinstance(exc, AgentFailure):
        logger.warning("Agent move failed: %s", exc)
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.exception_handler(GameServiceError)
async def handle_service_error(_request, exc: GameServiceError):
    return service_error_response(exc)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request", "errors": exc.errors()},
    )


@app.get("/healthz")
def health_check():
    return {"status": "ok"}


@app.get("/")
def home():
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/games")
def create_game(payload: NewGameRequest = NewGameRequest()):
    return service.new_game(seed=payload.seed)


@app.get("/api/games/{session_id}")
def get_game(session_id: str):
    return service.get_game(session_id)


@app.post("/api/games/{session_id}/moves")
def move(session_id: str, payload: MoveRequest):
    return service.move(session_id, payload.action, payload.expected_version)


@app.post("/api/games/{session_id}/control")
def set_control(session_id: str, payload: ControlRequest):
    return service.set_control(session_id, payload.control)


@app.post("/api/games/{session_id}/ai-move")
async def ai_move(session_id: str, payload: AIMoveRequest):
    # Search runs off the event loop so the browser remains able to stop.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        AI_EXECUTOR,
        lambda: service.ai_move(session_id, payload.expected_version),
    )


@app.post("/api/games/{session_id}/new")
def reset_game(session_id: str, payload: NewGameRequest = NewGameRequest()):
    return service.reset(session_id, seed=payload.seed)


@app.post("/api/games/{session_id}/undo")
def undo_move(session_id: str):
    return service.undo_move(session_id)


@app.patch("/api/games/{session_id}/agent")
def configure_agent(session_id: str, payload: AgentRequest):
    return service.configure_agent(session_id, payload.name, payload.depth)


@app.get("/api/benchmarks")
def benchmarks():
    return service.benchmarks()


# Keep the explicit page and API routes above this catch-all static mount.
# The HTML requests /styles.css and /app.js from the web/ directory.
app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
