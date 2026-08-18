"""
OpenAI-compatible FastAPI server for DeepSeek.

Point any OpenAI client at http://localhost:8000/v1 :

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions   (stream=true supported)
    GET  /healthz

Requests under /v1 are rate limited per client IP (default 30/min, set via
RATE_LIMIT_PER_MINUTE); /healthz is exempt.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from deepseek.auth import LoginRequired
from deepseek.client import DeepSeekClient, UpstreamError

from .config import (
    MODEL_MAP,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    is_known_model,
    resolve_model_type,
)
from .openai_format import (
    completion_response,
    extract_tool_calls,
    trim_at_role_boundary,
    messages_to_prompt,
    stream_chunks,
    stream_chunks_with_tools,
)
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest

load_dotenv()

# uvicorn configures logging with disable_existing_loggers, so hang our
# debug output off its own logger to be sure it reaches the console.
log = logging.getLogger("uvicorn.error")
# Set DEBUG_REQUESTS=1 to log each request's shape (roles + tool names). Useful
# when a client's tool calls aren't arriving: it shows whether `tools` was sent
# at all, which is the difference between "bridge ignored them" and "client
# never offered them".
DEBUG_REQUESTS = os.getenv("DEBUG_REQUESTS", "").lower() in ("1", "true", "yes", "on")
# Seconds to advertise in Retry-After when DeepSeek rate-limits us.
UPSTREAM_RETRY_AFTER = os.getenv("UPSTREAM_RETRY_AFTER", "30")

app = FastAPI(title="DeepSeek OpenAI-compatible API", version="0.1.0")
install_rate_limit(app, RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0))

# One shared client (and its signed-in session) built lazily on first use.
_client: DeepSeekClient | None = None
_client_lock = threading.Lock()


def get_client() -> DeepSeekClient:
    """Build (once) the shared client and its signed-in session.

    Session resolution: cached file → headless capture off the persistent
    profile. If neither works and SERVER_INTERACTIVE_LOGIN is on (the default),
    it opens a visible browser window so you can sign in — the triggering
    request blocks until you finish. If interactive login is off, it raises
    `LoginRequired`, which the endpoint turns into an actionable 503.

    This touches Playwright's sync API, so callers must invoke it OFF the event
    loop (via run_in_threadpool); calling it inside the asyncio loop raises
    "Playwright Sync API inside the asyncio loop"."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = DeepSeekClient(allow_interactive=SERVER_INTERACTIVE_LOGIN)
    return _client


def _error(message: str, status: int = 500, err_type: str = "server_error",
           headers: dict = None):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
        headers=headers or None,
    )


def _upstream_error(exc: UpstreamError):
    """Map a DeepSeek stream error onto the status a client can act on.

    A rate limit must be a 429 with Retry-After: OpenAI-compatible clients back
    off and retry on that, whereas a 200 with empty content just looks like the
    model had nothing to say.
    """
    if exc.is_rate_limit:
        return _error(str(exc), status=429, err_type="rate_limit_exceeded",
                      headers={"Retry-After": UPSTREAM_RETRY_AFTER})
    return _error(f"DeepSeek reported: {exc}", status=502, err_type="upstream_error")


class _Prefetched:
    """A stream with its first delta already pulled.

    The endpoint consumes one delta before returning a response, so an error
    frame -- which DeepSeek sends immediately -- becomes a real status code
    instead of a 200 whose body turns out to be an error mid-flight.
    """

    def __init__(self, stream, first):
        self._stream, self._first = stream, first

    def __iter__(self):
        if self._first is not None:
            yield self._first
        yield from self._stream

    @property
    def conversation_id(self):
        return getattr(self._stream, "conversation_id", None)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": created, "owned_by": "deepseek"}
            for name in MODEL_MAP
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    if not is_known_model(req.model):
        return _error(
            f"The model `{req.model}` does not exist. Available models: "
            f"{', '.join(MODEL_MAP)}",
            status=404, err_type="model_not_found",
        )

    # A thread's model is fixed when it's created, so on resume we ignore `model`
    # (the OpenAI SDK always sends one) and let the existing thread's model stand.
    model_type = None if req.conversation_id else resolve_model_type(req.model)
    if DEBUG_REQUESTS:
        log.warning("request: model=%s stream=%s roles=%s tools=%s tool_choice=%s",
                    req.model, req.stream, [m.role for m in req.messages],
                    [ (t.get("function") or t).get("name") for t in (req.tools or []) ],
                    req.tool_choice)

    prompt = messages_to_prompt(req.messages, req.tools, req.tool_choice)
    # DeepSeek has no native tool channel; `tools` are emulated in the prompt
    # and parsed back out of the reply. `tool_choice: none` means don't offer.
    tools = None if req.tool_choice == "none" else req.tools

    try:
        # Off the event loop: get_client() uses Playwright's sync API, which
        # errors if run inside the asyncio loop.
        client = await run_in_threadpool(get_client)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:  # session/login failure
        return _error(f"Failed to initialise DeepSeek session: {e}")

    if req.stream:
        raw = client.stream(
            prompt, conversation_id=req.conversation_id,
            model=model_type, thinking=req.thinking, search=req.search,
        )
        it = iter(raw)
        try:
            first = await run_in_threadpool(lambda: next(it, None))
        except UpstreamError as e:
            return _upstream_error(e)
        except Exception as e:
            return _error(f"DeepSeek request failed: {e}")

        def gen():
            stream = _Prefetched(raw, first)
            if tools:
                yield from stream_chunks_with_tools(req.model, stream, tools)
            else:
                yield from stream_chunks(req.model, stream)

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        reply = await run_in_threadpool(
            client.chat, prompt, req.conversation_id,
            model_type, req.thinking, req.search,
        )
    except UpstreamError as e:
        return _upstream_error(e)
    except Exception as e:
        return _error(f"DeepSeek request failed: {e}")

    calls, text = extract_tool_calls(reply.text, tools)
    if not tools:
        # No tools this turn, so extract_tool_calls returned early — the reply
        # can still run on past its turn, so trim it here too.
        text = trim_at_role_boundary(text)
    if DEBUG_REQUESTS:
        log.warning("reply: calls=%s text=%r",
                    [(c["function"]["name"], c["function"]["arguments"]) for c in calls or []],
                    (text or "")[:200])
    return completion_response(req.model, text, prompt, reply.conversation_id,
                               tool_calls=calls)
