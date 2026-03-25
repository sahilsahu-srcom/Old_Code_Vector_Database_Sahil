"""
File Summary:
Unified LLM client abstraction layer for RAG system.
Provides a common interface for multiple LLM backends (Ollama, OpenAI, Anthropic)
with retry logic, streaming support, logging, and observability.

====================================================================
SYSTEM PIPELINE FLOW (Architecture + Object Interaction)
====================================================================

build_llm_client(cfg)
||
├── Select backend from config ------------------------> "ollama" | "openai" | "anthropic"
│
├── OllamaClient()     [Class → Object] ---------------> Local LLM (default, zero cost)
├── OpenAIClient()     [Class → Object] ---------------> Cloud LLM (fallback)
└── AnthropicClient()  [Class → Object] ---------------> Claude LLM (fallback)

====================================================================

BaseLLMClient (Abstract Layer)
||
├── chat()  [Function] --------------------------------> Main entry point
│       │
│       ├── _chat_with_retry() ------------------------> Non-streaming execution
│       │       │
│       │       ├── _chat_raw() (backend impl) --------> Actual API call
│       │       ├── Retry loop ------------------------> Exponential backoff
│       │       └── log_llm_call() --------------------> Observability logging
│       │
│       └── _stream_with_logging() --------------------> Streaming execution
│               │
│               ├── _stream_raw() ---------------------> Token streaming
│               ├── Token estimation ------------------> Approx token count
│               └── log_llm_call() --------------------> Final logging
│
├── complete()  [Function] ----------------------------> Wrapper over chat()
│
├── health_check()  [Abstract] ------------------------> Backend availability check
│
└── _chat_raw()  [Abstract] ---------------------------> Must be implemented by backend

====================================================================

OllamaClient (Local Backend)
||
├── health_check() -----------------------------------> Check /api/tags endpoint
│
├── _chat_raw() --------------------------------------> POST /api/chat (non-stream)
│       │
│       ├── Build JSON payload
│       ├── Send HTTP request
│       └── Parse response → content + token counts
│
└── _stream_raw() ------------------------------------> Streaming response
        │
        ├── Iterate line-by-line response
        ├── Extract token chunks
        └── Yield tokens

====================================================================

OpenAIClient (Cloud Backend)
||
├── health_check() -----------------------------------> models.list()
│
├── _chat_raw() --------------------------------------> chat.completions.create()
│       │
│       ├── Send request
│       ├── Extract response text
│       └── Extract token usage
│
└── _stream_raw() ------------------------------------> Streaming chunks
        │
        └── Yield delta tokens

====================================================================

AnthropicClient (Claude Backend)
||
├── health_check() -----------------------------------> models.list()
│
└── _chat_raw() --------------------------------------> messages.create()
        │
        ├── Separate system prompt
        ├── Build message structure
        ├── Execute request
        └── Extract content + token usage

====================================================================

KEY DESIGN FEATURES
====================================================================

• Unified interface across all LLM providers
• Retry with exponential backoff (robustness)
• Streaming + non-streaming support
• Token usage tracking (observability)
• Backend-agnostic architecture
• Factory pattern for clean initialization

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

from rag_core import logger as log
from rag_core.config import RAGConfig


# ─── Message schema ───────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str   # "system" | "user" | "assistant"
    content: str



# ─── Abstract base ────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    """
    All concrete clients inherit from here.
    They must implement _chat_raw and optionally _stream_raw.
    """

    _MAX_RETRIES = 3
    _RETRY_BASE  = 1.5   # seconds, doubles per retry

    def __init__(self, model: str):
        self.model = model
        self._log = log.get("llm")

    # ── Public helpers ─────────────────────────────────────────────────────


    def chat(
        self,
        messages: List[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stream: bool = False,
        session_id: str = "",

    ) -> Union[str, Iterator[str]]:
        if stream:
            return self._stream_with_logging(messages, max_tokens, temperature, session_id)
        return self._chat_with_retry(messages, max_tokens, temperature, session_id)

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.3,

    ) -> str:
        return self._chat_with_retry(
            [Message(role="user", content=prompt)],
            max_tokens,
            temperature,
            session_id="",
        )

    @abstractmethod
    def health_check(self) -> bool: ...

    # ── Retry wrapper ──────────────────────────────────────────────────────


    def _chat_with_retry(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        session_id: str,

    ) -> str:
        t0 = time.perf_counter()
        last_exc: Optional[Exception] = None

        for attempt in range(self._MAX_RETRIES):
            try:
                content, pt, ct = self._chat_raw(messages, max_tokens, temperature)
                latency_ms = (time.perf_counter() - t0) * 1000
                log.log_llm_call(session_id, self.__class__.__name__, self.model,
                                 latency_ms, pt, ct)
                return content
            except Exception as exc:
                last_exc = exc
                wait = self._RETRY_BASE * (2 ** attempt)
                self._log.warning("LLM attempt %d failed: %s — retry in %.1fs", attempt + 1, exc, wait)
                time.sleep(wait)

        raise RuntimeError(f"LLM call failed after {self._MAX_RETRIES} retries: {last_exc}") from last_exc

    def _stream_with_logging(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        session_id: str,

    ) -> Iterator[str]:
        t0 = time.perf_counter()
        tokens = 0
        try:
            for chunk in self._stream_raw(messages, max_tokens, temperature):
                tokens += len(chunk) // 4
                yield chunk
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            prompt_tokens = sum(len(m.content) for m in messages) // 4
            log.log_llm_call(session_id, self.__class__.__name__, self.model,
                             latency_ms, prompt_tokens, tokens)

    # ── To be implemented ──────────────────────────────────────────────────

    @abstractmethod
    def _chat_raw(
        self, messages: List[Message], max_tokens: int, temperature: float

    ) -> tuple[str, int, int]:
        """Returns (content, prompt_tokens, completion_tokens)."""
        ...

    def _stream_raw(
        self, messages: List[Message], max_tokens: int, temperature: float

    ) -> Iterator[str]:
        """Default: fall back to non-streaming."""
        content, _, _ = self._chat_raw(messages, max_tokens, temperature)
        yield content



# ─── Ollama backend ───────────────────────────────────────────────────────────

class OllamaClient(BaseLLMClient):
    """
    Uses Ollama's /api/chat endpoint (OpenAI-compatible format available too,
    but native gives keep_alive and better error messages).
    """

    def __init__(self, base_url: str, model: str, timeout: int, keep_alive: str):
        super().__init__(model)
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._keep_alive = keep_alive


    def health_check(self) -> bool:
        try:
            import urllib.request
            urllib.request.urlopen(f"{self._base}/api/tags", timeout=5)
            return True
        except Exception:
            return False


    def _chat_raw(
        self, messages: List[Message], max_tokens: int, temperature: float

    ) -> tuple[str, int, int]:
        import json as _json, urllib.request as _req

        payload = _json.dumps({
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }).encode()

        req = _req.Request(
            f"{self._base}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _req.urlopen(req, timeout=self._timeout) as resp:
            data = _json.loads(resp.read())

        content = data["message"]["content"]
        pt = data.get("prompt_eval_count", len(payload) // 4)
        ct = data.get("eval_count", len(content) // 4)
        return content, pt, ct

    def _stream_raw(
        self, messages: List[Message], max_tokens: int, temperature: float

    ) -> Iterator[str]:
        import json as _json, urllib.request as _req

        payload = _json.dumps({
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "keep_alive": self._keep_alive,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }).encode()

        req = _req.Request(
            f"{self._base}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _req.urlopen(req, timeout=self._timeout) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = _json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break



# ─── OpenAI backend ───────────────────────────────────────────────────────────

class OpenAIClient(BaseLLMClient):

    def __init__(self, api_key: str, model: str, base_url: Optional[str], timeout: int):
        super().__init__(model)
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(
                api_key=api_key or None,
                base_url=base_url,
                timeout=timeout,
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")


    def health_check(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False


    def _chat_raw(
        self, messages: List[Message], max_tokens: int, temperature: float

    ) -> tuple[str, int, int]:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = resp.choices[0].message.content or ""
        pt = resp.usage.prompt_tokens if resp.usage else 0
        ct = resp.usage.completion_tokens if resp.usage else 0
        return content, pt, ct

    def _stream_raw(
        self, messages: List[Message], max_tokens: int, temperature: float

    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta



# ─── Anthropic backend ────────────────────────────────────────────────────────

class AnthropicClient(BaseLLMClient):

    def __init__(self, api_key: str, model: str, timeout: int):
        super().__init__(model)
        try:
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic(api_key=api_key or None, timeout=timeout)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")


    def health_check(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False


    def _chat_raw(
        self, messages: List[Message], max_tokens: int, temperature: float

    ) -> tuple[str, int, int]:
        system_text = ""
        chat_msgs = []
        for m in messages:
            if m.role == "system":
                system_text = m.content
            else:
                chat_msgs.append({"role": m.role, "content": m.content})

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=chat_msgs,
        )
        if system_text:
            kwargs["system"] = system_text

        resp = self._client.messages.create(**kwargs)
        content = resp.content[0].text if resp.content else ""
        pt = resp.usage.input_tokens if resp.usage else 0
        ct = resp.usage.output_tokens if resp.usage else 0
        return content, pt, ct



# ─── Factory ──────────────────────────────────────────────────────────────────

def build_llm_client(cfg: RAGConfig) -> BaseLLMClient:
    """Instantiate the correct backend from RAGConfig."""
    backend = cfg.llm_backend
    if backend == "ollama":
        return OllamaClient(
            base_url=cfg.ollama.base_url,
            model=cfg.ollama.model,
            timeout=cfg.ollama.timeout,
            keep_alive=cfg.ollama.keep_alive,
        )
    elif backend == "openai":
        return OpenAIClient(
            api_key=cfg.openai.api_key,
            model=cfg.openai.model,
            base_url=cfg.openai.base_url,
            timeout=cfg.openai.timeout,
        )
    elif backend == "anthropic":
        return AnthropicClient(
            api_key=cfg.anthropic.api_key,
            model=cfg.anthropic.model,
            timeout=cfg.anthropic.timeout,
        )
    raise ValueError(f"Unknown LLM backend: {backend!r}")

