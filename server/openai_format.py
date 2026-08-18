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
- NEVER write a call as prose or as code. Lines like "Action: some_tool",
  "Action Input: {...}", `some_tool({"arg": "value"})`, or
  `<some_tool arg="value">` are not tool calls and do nothing. Only the JSON
  object above is a tool call.
- NEVER write out, guess, or imagine a tool's result. Stop after the JSON; the
  real result comes back to you in the next turn.
- NEVER announce a call in words. "I'll read the file now", "Let me check
  that", "First I need to open it" — these do nothing, and the turn ends there.
  If you intend to use a tool, the JSON object IS your entire reply, right now.
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

    # The contract goes LAST, immediately before the generation point. Leading
    # with it loses to the conversation that follows: the expert model in
    # particular would answer "I'll read the file now" and stop, announcing a
    # call instead of emitting one.
    if preamble:
        lines.append(preamble)

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


# Second salvage path: `tool_name({"arg": "value"})` call syntax, which the
# expert model reaches for in preference to a JSON object. The name is only
# accepted if it resolves to a tool that was actually offered, so this can't
# turn ordinary prose containing parentheses into a call.
_CALL_SYNTAX_RE = re.compile(r"([A-Za-z_][\w.\-]*)\s*\(\s*(?=[{)])")


def _extract_call_syntax(text: str, allowed: List[str]) -> Tuple[Optional[List[dict]], str]:
    """Parse `tool_name({...})` / `tool_name()` invocations into tool calls."""
    calls, spans = [], []
    for m in _CALL_SYNTAX_RE.finditer(text):
        name = _resolve_name(m.group(1), allowed)
        if not name:
            continue
        end = m.end()
        if text[end:end + 1] == ")":          # no-argument call
            args = {}
            end += 1
        else:
            obj, partial = _scan_object(text, end)
            if not obj:
                obj = _repair_truncated(partial or "") or ""
            try:
                args = json.loads(obj)
            except (ValueError, TypeError):
                continue
            if not isinstance(args, dict):
                continue
            end += len(obj)
            if text[end:end + 1] == ")":
                end += 1
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
        spans.append((m.start(), end))
    if not calls:
        return None, text
    leftover = text
    for start, stop in reversed(spans):
        leftover = leftover[:start] + leftover[stop:]
    return calls, leftover.strip()


# Third salvage path: XML-ish dialects. `<tool_call>` wrappers appear when a
# client's own prompt format leaks into the reply, and attribute form
# (`<read_file path="a.txt">`) is what several agent frameworks train models to
# emit. Both are only honoured for tools that were actually offered.
_XML_TAG_RE = re.compile(r"<\s*([A-Za-z_][\w.\-]*)\s*((?:[\w.\-]+\s*=\s*\"[^\"]*\"\s*)*)/?>")
_XML_ATTR_RE = re.compile(r"([\w.\-]+)\s*=\s*\"([^\"]*)\"")


def _coerce(value: str):
    """Turn an XML attribute string into a JSON scalar where it clearly is one."""
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value
    return parsed if isinstance(parsed, (int, float, bool, list, dict)) else value


def _extract_xml_calls(text: str, allowed: List[str]) -> Tuple[Optional[List[dict]], str]:
    """Parse `<tool_name attr="value">` and bare `<tool_call>name</tool_call>`."""
    calls, spans = [], []
    for m in _XML_TAG_RE.finditer(text):
        tag, attrs = m.group(1), m.group(2)
        name = _resolve_name(tag, allowed)
        end = m.end()
        if not name:
            # A generic wrapper: the tool name is the tag's content instead.
            if tag.lower() not in ("tool_call", "tool", "function_call", "invoke"):
                continue
            rest = text[m.end():]
            inner = re.match(r"\s*([\w.\-]+)", rest)
            if not inner:
                continue
            name = _resolve_name(inner.group(1), allowed)
            if not name:
                continue
            end = m.end() + inner.end()
            attrs = ""
        args = {k: _coerce(v) for k, v in _XML_ATTR_RE.findall(attrs)}
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
        spans.append((m.start(), end))
    if not calls:
        return None, text
    leftover = text
    for start, stop in reversed(spans):
        leftover = leftover[:start] + leftover[stop:]
    # Tidy the now-orphaned closing tags.
    leftover = re.sub(r"</\s*[\w.\-]+\s*>", "", leftover).strip()
    return calls, leftover


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


def _scan_object(text: str, start: int) -> Tuple[Optional[str], Optional[str]]:
    """Scan one JSON object starting at `text[start]` == '{'.

    Returns (complete, partial): `complete` is the balanced object if one closes,
    otherwise `partial` is the truncated remainder (for the repair path below).

    Brace counting MUST ignore braces inside string literals. A quote-blind
    scanner mis-balances on ordinary payloads — `{"text": "hi {name}"}` — and
    the tool call leaks to the caller as raw JSON text.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1], None
    return None, text[start:]


def _repair_truncated(fragment: str) -> Optional[str]:
    """Close a cut-off JSON object so it can be parsed, or None.

    A reply can end mid-object (upstream cut, length cap). The fragment is still
    a real tool call the caller should get, so we balance the open brackets.

    We never complete a truncated VALUE. `"tabId": 15` cut from `1514652929`
    parses fine and points at a different tab — a wrong argument is worse than a
    missing one, because the caller acts on it instead of erroring. So a
    half-written value is dropped along with its key, and a tool left missing a
    required argument fails loudly where it belongs.
    """
    if '"tool_calls"' not in fragment and '"name"' not in fragment:
        return None

    stack = []
    in_string = False
    escaped = False
    for ch in fragment:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if not stack:
        return None

    repaired = fragment
    if in_string:
        # Cut inside a string: drop the whole "key": "half-written pair.
        opening = repaired.rfind('"')
        repaired = repaired[:opening]
        repaired = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", repaired)
    else:
        # Cut inside a bare literal (number/true/null) or right after a key.
        repaired = re.sub(r',?\s*"[^"]*"\s*:\s*(?:-?\d+(?:\.\d*)?(?:[eE][-+]?\d*)?'
                          r'|t(?:r(?:u(?:e)?)?)?|f(?:a(?:l(?:s(?:e)?)?)?)?'
                          r'|n(?:u(?:l(?:l)?)?)?)?$', "", repaired)
    repaired = re.sub(r",\s*$", "", repaired)
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def _iter_json_candidates(text: str) -> Iterable[str]:
    """Yield substrings of `text` that might be the tool-call JSON object.

    Fenced blocks first (what we asked for), then any string-aware balanced
    object in the raw text, then a repaired tail if the reply was cut off.
    """
    for m in _FENCE_RE.finditer(text):
        yield m.group(1)
    partials = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            complete, partial = _scan_object(text, i)
            if complete:
                yield complete
                i += len(complete)
                continue
            if partial:
                partials.append(partial)
            break
        i += 1
    for partial in partials:
        repaired = _repair_truncated(partial)
        if repaired:
            yield repaired


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
        # Entries are validated INDIVIDUALLY and bad ones skipped. Failing the
        # whole batch on one bad entry is how a valid `browser_tabs` call ends
        # up rendered as raw JSON text next to an invented tool name.
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        name = _resolve_name(name, allowed) if allowed else name
        if not name:
            continue
        args = fn.get("arguments", fn.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except (ValueError, TypeError):
                args = {}
        if not isinstance(args, dict):
            continue
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
    calls, leftover = _extract_react_calls(text, allowed)
    if calls:
        return calls, leftover
    calls, leftover = _extract_call_syntax(text, allowed)
    if calls:
        return calls, leftover
    return _extract_xml_calls(text, allowed)


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
