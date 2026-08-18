"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.

Tool calling is EMULATED. The web chat has no native function-calling channel,
so when a request carries `tools` we (1) describe them in the prompt with a
strict output contract, and (2) parse the model's JSON reply back into OpenAI
`tool_calls`. Without this the model narrates its intent in prose ("Action:
some_tool ...") and the caller sees text where it expected a tool call.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Iterable, List, Optional, Tuple

from .schemas import ChatMessage

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}

# The contract we ask the model to follow when tools are available. Kept blunt
# and example-led: the failure mode we're fixing is the model *describing* a
# call instead of emitting one.
_TOOL_PROTOCOL = """\
# Tool calling

You have access to the tools listed below. They are the ONLY tools that exist.

To call one or more tools, reply with a single JSON object and NOTHING else —
no prose before it, no explanation after it, in exactly this shape:

```json
{"tool_calls": [{"name": "<tool name>", "arguments": {"<arg>": "<value>"}}]}
```

Rules, all of them mandatory:
- Use ONLY tool names from the list. Never invent a tool that is not listed.
- `arguments` must be a JSON object matching that tool's parameter schema.
- NEVER write a call as prose. Lines like "Action: some_tool" or
  "Action Input: {...}" are not tool calls and do nothing.
- NEVER write out, guess, or imagine a tool's result. Stop after the JSON; the
  real result comes back to you in the next turn.
- When you are done using tools and want to answer the user, reply with normal
  text and no JSON block.

## Available tools

"""


def _text_of(content) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def _function_of(tool: dict) -> dict:
    """The function spec of a tool entry, tolerating the flat (pre-`tools`) shape."""
    if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
        return tool["function"]
    return tool if isinstance(tool, dict) else {}


def tool_names(tools) -> List[str]:
    names = []
    for t in tools or []:
        n = _function_of(t).get("name")
        if n:
            names.append(n)
    return names


def tools_preamble(tools, tool_choice=None) -> str:
    """Render the tool schemas plus the output contract as prompt text."""
    blocks = []
    for t in tools or []:
        fn = _function_of(t)
        name = fn.get("name")
        if not name:
            continue
        spec = {
            "name": name,
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }
        blocks.append(json.dumps(spec, ensure_ascii=False))
    if not blocks:
        return ""

    text = _TOOL_PROTOCOL + "\n\n".join(blocks)

    # tool_choice steers whether a call is optional, mandatory, or pinned.
    if isinstance(tool_choice, dict):
        forced = _function_of(tool_choice).get("name")
        if forced:
            text += (f"\n\nFor this turn you MUST call `{forced}` and no other tool. "
                     "Reply with the JSON object only.")
    elif tool_choice == "required":
        text += ("\n\nFor this turn you MUST call one of the tools above. "
                 "Reply with the JSON object only.")
    elif tool_choice == "none":
        text += ("\n\nFor this turn do NOT call any tool. Answer the user in "
                 "plain text.")
    return text


def _render_assistant_tool_calls(calls) -> str:
    """Replay a past assistant tool call in the same format we ask the model for."""
    rendered = []
    for c in calls or []:
        fn = c.get("function", {}) if isinstance(c, dict) else {}
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except (ValueError, TypeError):
                args = {"_raw": args}
        rendered.append({"name": fn.get("name") or c.get("name"), "arguments": args})
    body = json.dumps({"tool_calls": rendered}, ensure_ascii=False)
    return f"```json\n{body}\n```"


def messages_to_prompt(messages: List[ChatMessage], tools=None,
                       tool_choice=None) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer.

    A lone user message with no tools is sent verbatim. Anything else — system
    prompts, multi-turn, tool results — is serialised with role labels and a
    trailing 'Assistant:' cue so the model continues in the right voice.
    """
    preamble = tools_preamble(tools, tool_choice)

    if len(messages) == 1 and messages[0].role == "user" and not preamble:
        return _text_of(messages[0].content)

    lines = []
    if preamble:
        lines.append(preamble)

    for m in messages:
        if m.role == "tool":
            # The answer to a call we made. Label it with the tool's name when
            # the caller gave us one, so the model can match it to its request.
            who = m.name or m.tool_call_id or "tool"
            lines.append(f"Tool result ({who}): {_text_of(m.content)}")
            continue

        label = _ROLE_LABELS.get(m.role, m.role.capitalize())
        body = _text_of(m.content)
        if m.role == "assistant" and m.tool_calls:
            calls = _render_assistant_tool_calls(m.tool_calls)
            body = f"{body}\n{calls}".strip() if body else calls
        lines.append(f"{label}: {body}")

    lines.append("Assistant:")
    return "\n\n".join(lines)


# --- parsing the model's reply back into tool calls -------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)

# Salvage path: some replies narrate the call in ReAct prose instead of emitting
# JSON ("Action: some_tool  Action Input: {...}"). The preamble tells the model
# not to, but a prompt is not a decoder constraint, so we parse it anyway rather
# than hand the caller a wall of text where a tool call belonged.
_REACT_RE = re.compile(
    r"Action\s*:\s*[`\"']?([\w.\-]+)[`\"']?\s*[\r\n]*"
    r"Action\s*Input\s*:\s*(\{.*?\})\s*(?:$|[\r\n])",
    re.DOTALL | re.IGNORECASE,
)


def _resolve_name(name: str, allowed: List[str]) -> Optional[str]:
    """Map a model-written tool name onto an offered one, or None.

    Exact match first, then a unique suffix match — models routinely drop a
    namespace prefix (`mcp__server__do_thing` -> `do_thing`) or keep it when the
    caller offered the short form.
    """
    if name in allowed:
        return name
    hits = [a for a in allowed if a.endswith(name) or name.endswith(a)]
    return hits[0] if len(hits) == 1 else None


def _extract_react_calls(text: str, allowed: List[str]) -> Tuple[Optional[List[dict]], str]:
    """Parse ReAct-style `Action:` / `Action Input:` prose into tool calls."""
    calls, spans = [], []
    for m in _REACT_RE.finditer(text):
        name = _resolve_name(m.group(1), allowed)
        if not name:
            continue
        try:
            args = json.loads(m.group(2))
        except (ValueError, TypeError):
            continue
        if not isinstance(args, dict):
            continue
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
        spans.append(m.span())
    if not calls:
        return None, text
    leftover = text
    for start, end in reversed(spans):
        leftover = leftover[:start] + leftover[end:]
    return calls, leftover.strip()


def _iter_json_candidates(text: str) -> Iterable[str]:
    """Yield substrings of `text` that might be the tool-call JSON object.

    Fenced blocks first (what we asked for), then any brace-balanced object in
    the raw text (what we sometimes get).
    """
    for m in _FENCE_RE.finditer(text):
        yield m.group(1)
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]


def _normalise_calls(obj, allowed: List[str]) -> Optional[List[dict]]:
    """Pull a list of {name, arguments} out of a parsed JSON object, or None.

    Lenient about the wrapper (`tool_calls`, `tool_call`, or a bare call) because
    the model is not a schema-constrained decoder. Strict about the names: a call
    to a tool that wasn't offered is a hallucination, not a call.
    """
    if isinstance(obj, dict):
        raw = obj.get("tool_calls") or obj.get("tool_call") or obj
    else:
        raw = obj
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return None

    calls = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if not isinstance(name, str):
            return None
        name = _resolve_name(name, allowed) if allowed else name
        if not name:
            return None
        args = fn.get("arguments", fn.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except (ValueError, TypeError):
                args = {}
        if not isinstance(args, dict):
            return None
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    return calls or None


def extract_tool_calls(text: str, tools) -> Tuple[Optional[List[dict]], str]:
    """Split a reply into (tool_calls, leftover_text).

    Returns (None, text) when the model answered in prose — the normal case for
    a final answer.
    """
    allowed = tool_names(tools)
    if not allowed or not text:
        return None, text
    for candidate in _iter_json_candidates(text):
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        calls = _normalise_calls(obj, allowed)
        if calls:
            leftover = text.replace(candidate, "", 1)
            leftover = _FENCE_RE.sub("", leftover).strip()
            return calls, leftover
    return _extract_react_calls(text, allowed)


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — DeepSeek's web API gives us no count."""
    return max(1, len(text) // 4)


def completion_response(model: str, content: str, prompt: str,
                        conversation_id: str = None,
                        tool_calls: Optional[List[dict]] = None) -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    `conversation_id` is an extra top-level field (outside OpenAI's schema) you
    send back to resume the conversation.
    """
    pt, ct = _est_tokens(prompt), _est_tokens(content)
    message = {"role": "assistant", "content": content or None}
    finish = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish = "tool_calls"
    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def _frame(cid: str, created: int, model: str, delta: dict,
           finish=None, extra: dict = None) -> str:
    obj = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if extra:
        obj.update(extra)
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def stream_chunks(model: str, stream: Iterable[str]) -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}\\n\\n`) for a streamed completion.

    `stream` is the client's stream object; after it's consumed we read its
    `.conversation_id` and attach it to the final chunk.
    """
    cid, created = _id(), _now()

    # First frame announces the assistant role.
    yield _frame(cid, created, model, {"role": "assistant", "content": ""})
    for d in stream:
        if d:
            yield _frame(cid, created, model, {"content": d})
    conversation_id = getattr(stream, "conversation_id", None)
    yield _frame(cid, created, model, {}, finish="stop",
                 extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"


def stream_chunks_with_tools(model: str, stream: Iterable[str], tools) -> Iterable[str]:
    """Streaming variant for tool-enabled requests.

    A tool call is only recognisable once its JSON object is complete, so the
    upstream text is buffered and emitted as one delta: either the prose answer
    or a `tool_calls` delta. Token-by-token output is lost for these requests —
    correctness over cosmetics, since a half-parsed call is worse than useless.
    """
    cid, created = _id(), _now()
    yield _frame(cid, created, model, {"role": "assistant", "content": ""})

    buf = "".join(d for d in stream if d)
    conversation_id = getattr(stream, "conversation_id", None)
    calls, leftover = extract_tool_calls(buf, tools)

    if calls:
        delta = {"tool_calls": [
            {
                "index": i,
                "id": c["id"],
                "type": "function",
                "function": c["function"],
            }
            for i, c in enumerate(calls)
        ]}
        if leftover:
            delta["content"] = leftover
        yield _frame(cid, created, model, delta)
        finish = "tool_calls"
    else:
        if buf:
            yield _frame(cid, created, model, {"content": buf})
        finish = "stop"

    yield _frame(cid, created, model, {}, finish=finish,
                 extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"
