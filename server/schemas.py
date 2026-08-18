"""Pydantic models for the OpenAI-compatible request/response shapes we support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from .config import DEFAULT_MODEL


class ChatMessage(BaseModel):
    role: str
    # content is a plain string, or a list of parts (OpenAI vision-style). We only
    # read text parts; non-text parts are ignored.
    content: Union[str, List[dict], None] = None
    # Set on assistant turns that called tools, and on the `tool` turns that
    # answer them. We replay both back into the prompt so a multi-step tool
    # conversation keeps its shape (see openai_format.messages_to_prompt).
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = False
    # Pass a conversation_id from a previous response to resume that thread.
    conversation_id: Optional[str] = None
    # Tools to enable for this request, independent of the model. OpenAI clients
    # pass these via extra_body: `thinking` (DeepThink), `search` (web).
    thinking: bool = False
    search: bool = False
    # OpenAI function calling. DeepSeek's web chat has no native tool channel, so
    # these are emulated: the schemas go into the prompt and the model's JSON
    # reply is parsed back into `tool_calls`.
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # Accepted for compatibility but not all are forwarded to DeepSeek.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
