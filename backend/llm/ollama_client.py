"""Local Ollama LLM client and streaming benchmark metrics."""

import json
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from backend.config import AppConfig, get_config
from backend.logger import get_logger, stage_timer


class OllamaError(RuntimeError):
    """Base error for Ollama interactions."""


class OllamaUnavailableError(OllamaError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the configured model is not available locally in Ollama."""


class OllamaInferenceError(OllamaError):
    """Raised when Ollama returns an error during generation."""


class LlmResult(BaseModel):
    """Response text and timing statistics from local LLM inference."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    response_text: str
    thinking_text: str
    ttft_ms: float
    total_time_ms: float
    load_duration_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    tokens_per_sec: float


def _match_model_name(target: str, available_names: list[str]) -> str | None:
    """Match exact or :latest-qualified model name against Ollama tags."""
    if target in available_names:
        return target
    if ":" not in target and f"{target}:latest" in available_names:
        return f"{target}:latest"
    return None


class OllamaClient:
    """Local HTTP streaming client for Ollama inference and inspection."""

    def __init__(
        self,
        config: AppConfig | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._config = config if config is not None else get_config()
        self._llm_cfg = self._config.llm
        self._logger = get_logger("localvox.llm", self._config)
        self._external_client = http_client

    def _build_client(self) -> httpx.Client:
        if self._external_client is not None:
            return self._external_client
        return httpx.Client(base_url=self._llm_cfg.base_url, timeout=self._llm_cfg.timeout_s)

    def list_models(self) -> list[dict[str, Any]]:
        """Query local Ollama /api/tags and return installed model descriptors."""
        client = self._build_client()
        should_close = self._external_client is None
        try:
            response = client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
            return list(payload.get("models", []))
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(
                f"Ollama service unreachable at {self._llm_cfg.base_url}: {exc}"
            ) from exc
        finally:
            if should_close:
                client.close()

    def verify_model_exists(self, model_name: str | None = None) -> dict[str, Any]:
        """Verify the configured model is pulled locally; never pull silently."""
        target = model_name if model_name is not None else self._llm_cfg.model_name
        models = self.list_models()
        names = [str(m.get("name", "")) for m in models]
        matched = _match_model_name(target, names)
        if matched is None:
            raise OllamaModelNotFoundError(
                f"Model '{target}' is not installed locally in Ollama. "
                f"Run `ollama pull {target}` first. Available: {names}"
            )
        for model in models:
            if model.get("name") == matched:
                return model
        return {"name": matched}

    def get_running_memory_mb(self, model_name: str | None = None) -> float | None:
        """Return currently loaded model memory footprint in MiB from /api/ps."""
        target = model_name if model_name is not None else self._llm_cfg.model_name
        client = self._build_client()
        should_close = self._external_client is None
        try:
            response = client.get("/api/ps")
            response.raise_for_status()
            models = response.json().get("models", [])
            matched = _match_model_name(target, [str(m.get("name", "")) for m in models])
            if matched is None:
                return None
            for model in models:
                if model.get("name") == matched:
                    size_bytes = model.get("size") or model.get("size_vram")
                    if size_bytes:
                        return round(float(size_bytes) / (1024.0 * 1024.0), 2)
            return None
        except httpx.HTTPError:
            return None
        finally:
            if should_close:
                client.close()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model_name: str | None = None,
        think: bool | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LlmResult:
        """Send a streaming local chat request and measure TTFT and token throughput."""
        target_model = model_name if model_name is not None else self._llm_cfg.model_name
        self.verify_model_exists(target_model)

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": True,
            "think": self._llm_cfg.think if think is None else think,
            "keep_alive": self._llm_cfg.keep_alive,
            "options": {
                "temperature": self._llm_cfg.temperature if temperature is None else temperature,
                "top_p": self._llm_cfg.top_p,
                "num_predict": self._llm_cfg.max_tokens if max_tokens is None else max_tokens,
                "num_ctx": self._llm_cfg.num_ctx,
            },
        }

        client = self._build_client()
        should_close = self._external_client is None
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        ttft_ms: float | None = None
        final_chunk: dict[str, Any] = {}

        try:
            with stage_timer("llm", log_path=self._config.logging.latency_log_file) as rec:
                start = time.perf_counter()
                with client.stream("POST", "/api/chat", json=payload) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")
                        raise OllamaInferenceError(
                            f"Ollama returned HTTP {response.status_code}: {body}"
                        )
                    for raw_line in response.iter_lines():
                        if not raw_line:
                            continue
                        chunk = json.loads(raw_line)
                        if "error" in chunk:
                            raise OllamaInferenceError(str(chunk["error"]))
                        msg = chunk.get("message", {})
                        content_delta = msg.get("content", "")
                        thinking_delta = msg.get("thinking", "")
                        if ttft_ms is None and (content_delta or thinking_delta):
                            ttft_ms = round((time.perf_counter() - start) * 1000.0, 2)
                        if content_delta:
                            content_parts.append(content_delta)
                        if thinking_delta:
                            thinking_parts.append(thinking_delta)
                        if chunk.get("done"):
                            final_chunk = chunk
                rec["model"] = target_model

            total_ms = float(rec["latency_ms"])
        except httpx.HTTPError as exc:
            raise OllamaInferenceError(f"Ollama streaming request failed: {exc}") from exc
        finally:
            if should_close:
                client.close()

        eval_count = int(final_chunk.get("eval_count") or 0)
        prompt_count = int(final_chunk.get("prompt_eval_count") or 0)
        eval_duration_ns = int(final_chunk.get("eval_duration") or 0)
        load_duration_ns = final_chunk.get("load_duration")
        load_ms = (
            round(float(load_duration_ns) / 1_000_000.0, 2)
            if load_duration_ns is not None
            else None
        )

        if eval_duration_ns > 0 and eval_count > 0:
            tps = round(eval_count / (eval_duration_ns / 1_000_000_000.0), 2)
        elif total_ms > 0 and eval_count > 0:
            tps = round(eval_count / (total_ms / 1000.0), 2)
        else:
            tps = 0.0

        result = LlmResult(
            model_name=target_model,
            response_text="".join(content_parts).strip(),
            thinking_text="".join(thinking_parts).strip(),
            ttft_ms=ttft_ms if ttft_ms is not None else round(total_ms, 2),
            total_time_ms=round(total_ms, 2),
            load_duration_ms=load_ms,
            prompt_tokens=prompt_count,
            completion_tokens=eval_count,
            tokens_per_sec=tps,
        )
        self._logger.info(
            "LLM generation completed (model=%s, ttft_ms=%.1f, total_ms=%.1f, tok_s=%.1f)",
            target_model,
            result.ttft_ms,
            result.total_time_ms,
            result.tokens_per_sec,
        )
        return result

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model_name: str | None = None,
        think: bool | None = None,
        max_tokens: int | None = None,
    ) -> LlmResult:
        """Run single-turn inference using the configured or overridden system prompt."""
        sys_msg = system_prompt if system_prompt is not None else self._llm_cfg.system_prompt
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]
        return self.chat(
            messages,
            model_name=model_name,
            think=think,
            max_tokens=max_tokens,
        )
