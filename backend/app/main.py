"""Localhost FastAPI + WebSocket entrypoint for the LocalVox Phase 5 Desktop Companion."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from backend.app.history import ConversationSessionDetail, ConversationSessionSummary
from backend.app.runtime import CompanionRuntime
from backend.app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationContinueResponse,
    ConversationListResponse,
    EnrollmentSampleResult,
    HealthResponse,
    MemoryBulkDeleteRequest,
    MemoryCreateRequest,
    MemoryListResponse,
    MemoryMutationResponse,
    MemoryUpdateRequest,
    SettingsPayload,
    SettingsResponse,
    StateSnapshotResponse,
    SystemMonitorResponse,
    VoiceControlResponse,
    VoiceProfileSummary,
)
from backend.llm.ollama_client import OllamaError
from backend.memory.models import (
    MemoryCategory,
    MemoryNotFoundError,
    MemoryPrivacyError,
    utc_now_iso,
)
from backend.pipeline.audio_io import AudioDeviceError


def create_app(
    runtime: CompanionRuntime | None = None,
    *,
    config_path: Path | str | None = None,
    auto_start_wakeword: bool | None = None,
) -> FastAPI:
    """Create configured FastAPI companion application bound strictly to local origins."""
    rt = runtime if runtime is not None else CompanionRuntime(config_path=config_path)
    should_autostart = (runtime is None) if auto_start_wakeword is None else auto_start_wakeword

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        rt.broadcaster.attach_loop(asyncio.get_running_loop())
        if (
            should_autostart
            and rt.config.wakeword.enabled
            and rt._pipeline.wakeword_detector.info().models_present
        ):
            try:
                rt.start_voice(mode="wakeword")
            except Exception as exc:
                rt._logger.warning("Auto-start hands-free wakeword skipped: %s", exc)
        try:
            yield
        finally:
            rt.broadcaster.attach_loop(None)
            rt.shutdown()

    api = FastAPI(
        title="LocalVox Desktop Companion API",
        version="0.5.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    api.state.runtime = rt

    api.add_middleware(
        CORSMiddleware,
        allow_origins=list(rt.config.server.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    @api.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @api.get("/api/health", response_model=HealthResponse)
    def get_health() -> HealthResponse:
        return rt.get_health()

    @api.get("/api/state", response_model=StateSnapshotResponse)
    def get_state() -> StateSnapshotResponse:
        return rt.get_state_snapshot()

    @api.post("/api/chat", response_model=ChatResponse)
    async def post_chat(payload: ChatRequest) -> ChatResponse:
        try:
            return await asyncio.to_thread(
                rt.send_text,
                payload.text,
                speak=payload.speak,
                language=payload.language,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (OllamaError, AudioDeviceError, FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @api.post("/api/voice/start", response_model=VoiceControlResponse)
    def post_voice_start(
        mode: Annotated[Literal["auto", "direct", "wakeword"], Query()] = "auto",
    ) -> VoiceControlResponse:
        return rt.start_voice(mode=mode)

    @api.post("/api/voice/stop", response_model=VoiceControlResponse)
    def post_voice_stop() -> VoiceControlResponse:
        return rt.stop_voice()

    @api.post("/api/voice/stop-speaking", response_model=VoiceControlResponse)
    def post_stop_speaking() -> VoiceControlResponse:
        return rt.stop_speaking()

    @api.post("/api/wakeword/enable", response_model=VoiceControlResponse)
    async def post_wakeword_enable() -> VoiceControlResponse:
        return await asyncio.to_thread(rt.set_wakeword_monitoring, True)

    @api.post("/api/wakeword/disable", response_model=VoiceControlResponse)
    async def post_wakeword_disable() -> VoiceControlResponse:
        return await asyncio.to_thread(rt.set_wakeword_monitoring, False)

    @api.get("/api/wakeword/profile", response_model=VoiceProfileSummary)
    def get_wakeword_profile() -> VoiceProfileSummary:
        return rt.get_voice_profile()

    @api.post("/api/wakeword/enroll/start", response_model=VoiceProfileSummary)
    def post_wakeword_enroll_start() -> VoiceProfileSummary:
        return rt.start_voice_enrollment()

    @api.post("/api/wakeword/enroll/record", response_model=EnrollmentSampleResult)
    async def post_wakeword_enroll_record(
        timeout_s: Annotated[float, Query(ge=2.0, le=15.0)] = 6.5,
    ) -> EnrollmentSampleResult:
        try:
            return await asyncio.to_thread(
                rt.record_voice_enrollment_sample,
                timeout_s=timeout_s,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (AudioDeviceError, RuntimeError, FileNotFoundError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @api.delete("/api/wakeword/profile", response_model=VoiceProfileSummary)
    def delete_wakeword_profile() -> VoiceProfileSummary:
        return rt.delete_voice_profile()

    @api.get("/api/memories", response_model=MemoryListResponse)
    def get_memories(
        query: Annotated[str, Query(max_length=256)] = "",
        category: Annotated[MemoryCategory | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> MemoryListResponse:
        return rt.list_memories(query=query, category=category, limit=limit)

    @api.post("/api/memories", response_model=MemoryMutationResponse, status_code=201)
    def post_memory(payload: MemoryCreateRequest) -> MemoryMutationResponse:
        try:
            return rt.create_memory(payload)
        except (MemoryPrivacyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.patch("/api/memories/{memory_id}", response_model=MemoryMutationResponse)
    def patch_memory(memory_id: str, payload: MemoryUpdateRequest) -> MemoryMutationResponse:
        try:
            return rt.update_memory(memory_id, payload)
        except MemoryNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (MemoryPrivacyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.delete("/api/memories/{memory_id}", response_model=MemoryMutationResponse)
    def delete_memory(memory_id: str) -> MemoryMutationResponse:
        if not rt.delete_memory(memory_id):
            raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")
        return MemoryMutationResponse(status="ok", action="deleted", deleted_count=1)

    @api.post("/api/memories/clear", response_model=MemoryMutationResponse)
    def clear_memories(payload: MemoryBulkDeleteRequest) -> MemoryMutationResponse:
        try:
            return rt.clear_all_memories(payload.confirm_phrase)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.get("/api/conversations", response_model=ConversationListResponse)
    def get_conversations(
        query: Annotated[str, Query(max_length=256)] = "",
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> ConversationListResponse:
        return rt.list_conversations(query=query, limit=limit)

    @api.post("/api/conversations/new", response_model=ConversationSessionSummary)
    def post_new_conversation() -> ConversationSessionSummary:
        return rt.new_conversation()

    @api.get("/api/conversations/{session_id}", response_model=ConversationSessionDetail)
    def get_conversation_detail(session_id: str) -> ConversationSessionDetail:
        detail = rt.get_conversation(session_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        return detail

    @api.post(
        "/api/conversations/{session_id}/continue",
        response_model=ConversationContinueResponse,
    )
    def post_continue_conversation(session_id: str) -> ConversationContinueResponse:
        result = rt.continue_conversation(session_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        return result

    @api.delete("/api/conversations/{session_id}")
    def delete_conversation(session_id: str) -> dict[str, object]:
        if not rt.delete_conversation(session_id):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        return {"status": "ok", "deleted": True}

    @api.get("/api/system", response_model=SystemMonitorResponse)
    async def get_system_monitor() -> SystemMonitorResponse:
        return await asyncio.to_thread(rt.get_system_monitor)

    @api.get("/api/settings", response_model=SettingsResponse)
    async def get_settings() -> SettingsResponse:
        return await asyncio.to_thread(rt.get_settings)

    @api.put("/api/settings", response_model=SettingsResponse)
    async def put_settings(payload: SettingsPayload) -> SettingsResponse:
        try:
            return await asyncio.to_thread(rt.update_settings, payload)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if origin is not None and origin not in rt.config.server.allowed_origins:
            await websocket.close(code=1008, reason="Origin not allowed")
            return

        await websocket.accept()
        queue = rt.broadcaster.subscribe()
        try:
            snapshot = rt.get_state_snapshot().model_dump()
            await websocket.send_json(
                {
                    "type": "snapshot",
                    "timestamp": utc_now_iso(),
                    **snapshot,
                }
            )

            async def sender() -> None:
                with contextlib.suppress(WebSocketDisconnect):
                    while True:
                        event = await queue.get()
                        await websocket.send_json(event)

            async def receiver() -> None:
                with contextlib.suppress(WebSocketDisconnect):
                    while True:
                        text = await websocket.receive_text()
                        if text.strip() == "ping":
                            await websocket.send_json(
                                {"type": "pong", "timestamp": utc_now_iso()}
                            )

            send_task = asyncio.create_task(sender())
            recv_task = asyncio.create_task(receiver())
            try:
                _, pending = await asyncio.wait(
                    {send_task, recv_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            finally:
                for task in (send_task, recv_task):
                    if not task.done():
                        task.cancel()
        except WebSocketDisconnect:
            pass
        finally:
            rt.broadcaster.unsubscribe(queue)

    return api


app = create_app()


def main() -> None:
    """Launch LocalVox FastAPI companion bound strictly to localhost."""
    rt: CompanionRuntime = app.state.runtime
    uvicorn.run(
        app,
        host=rt.config.server.host,
        port=rt.config.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
