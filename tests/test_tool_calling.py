"""Emulated tool calling: prompt contract in, OpenAI tool_calls out.

Runs without a DeepSeek session — the upstream client is faked.

    ./venv/bin/python -m pytest tests/test_tool_calling.py
"""

import json

from fastapi.testclient import TestClient

import server.api as api
from server.openai_format import extract_tool_calls, messages_to_prompt
from server.schemas import ChatMessage

TOOLS = [{"type": "function", "function": {
    "name": "browser_navigate",
    "description": "Navigate to a URL",
    "parameters": {"type": "object", "properties": {"url": {"type": "string"}},
                   "required": ["url"]}}}]

MCP_TOOLS = [{"type": "function", "function": {
    "name": "mcp__openbrowser__browser_navigate",
    "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}}]


def _names(calls):
    return [c["function"]["name"] for c in calls or []]


def _args(calls, i=0):
    return json.loads(calls[i]["function"]["arguments"])


def test_fenced_json_is_a_tool_call():
    calls, left = extract_tool_calls(
        '```json\n{"tool_calls":[{"name":"browser_navigate",'
        '"arguments":{"url":"https://youtube.com"}}]}\n```', TOOLS)
    assert _names(calls) == ["browser_navigate"]
    assert _args(calls)["url"] == "https://youtube.com"
    assert left == ""


def test_bare_json_and_prose_wrapper():
    for text in (
        '{"tool_calls":[{"name":"browser_navigate","arguments":{"url":"https://a.com"}}]}',
        'Sure.\n```json\n{"tool_calls":[{"name":"browser_navigate",'
        '"arguments":{"url":"https://a.com"}}]}\n```',
    ):
        calls, _ = extract_tool_calls(text, TOOLS)
        assert _args(calls)["url"] == "https://a.com"


def test_singular_and_openai_shaped_wrappers():
    calls, _ = extract_tool_calls(
        '{"tool_call":{"name":"browser_navigate","arguments":{"url":"https://c.com"}}}', TOOLS)
    assert _names(calls) == ["browser_navigate"]
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"type":"function","function":{"name":"browser_navigate",'
        '"arguments":"{\\"url\\":\\"https://d.com\\"}"}}]}', TOOLS)
    assert _args(calls)["url"] == "https://d.com"


def test_plain_answer_is_not_a_tool_call():
    calls, left = extract_tool_calls("YouTube is a video site.", TOOLS)
    assert calls is None and left == "YouTube is a video site."


def test_unoffered_tool_is_rejected():
    # The regression that started this: models invent tools that don't exist.
    calls, _ = extract_tool_calls(
        '{"tool_calls":[{"name":"browser_get_all_pages","arguments":{}}]}', TOOLS)
    assert calls is None


def test_react_prose_is_salvaged():
    calls, left = extract_tool_calls(
        "I'll navigate to YouTube.com using OpenBrowser.\n\n"
        'Action: mcp__openbrowser__browser_navigate '
        'Action Input: {"url": "https://www.youtube.com"}', MCP_TOOLS)
    assert _names(calls) == ["mcp__openbrowser__browser_navigate"]
    assert _args(calls)["url"] == "https://www.youtube.com"
    assert "Action:" not in left


def test_react_prose_resolves_a_dropped_namespace_prefix():
    calls, _ = extract_tool_calls(
        'Action: browser_navigate\nAction Input: {"url": "https://x.com"}', MCP_TOOLS)
    assert _names(calls) == ["mcp__openbrowser__browser_navigate"]


def test_react_prose_for_an_unoffered_tool_stays_text():
    calls, _ = extract_tool_calls("Action: browser_get_all_pages Action Input: {}", MCP_TOOLS)
    assert calls is None


def test_prompt_carries_schemas_and_forbids_prose_calls():
    prompt = messages_to_prompt(
        [ChatMessage(role="user", content="go to youtube")], TOOLS)
    assert "browser_navigate" in prompt
    assert "tool_calls" in prompt
    assert "Action:" in prompt  # the explicit "don't do this" rule


def test_prompt_replays_past_calls_and_results():
    prompt = messages_to_prompt([
        ChatMessage(role="user", content="go to youtube"),
        ChatMessage(role="assistant", tool_calls=[{"id": "call_1", "type": "function",
            "function": {"name": "browser_navigate",
                         "arguments": '{"url":"https://youtube.com"}'}}]),
        ChatMessage(role="tool", tool_call_id="call_1", name="browser_navigate",
                    content="navigated ok"),
    ], TOOLS)
    assert "Tool result (browser_navigate): navigated ok" in prompt
    assert "https://youtube.com" in prompt


# --- endpoint level, with a faked upstream ---------------------------------

class _FakeReply:
    def __init__(self, text):
        self.text, self.conversation_id = text, "conv-1"


class _FakeStream:
    def __init__(self, text):
        self.parts = [text[i:i + 7] for i in range(0, len(text), 7)]
        self.conversation_id = "conv-1"

    def __iter__(self):
        return iter(self.parts)


class _FakeClient:
    canned = ""

    def chat(self, *a, **k):
        return _FakeReply(self.canned)

    def stream(self, *a, **k):
        return _FakeStream(self.canned)


def _client(canned):
    fake = _FakeClient()
    fake.canned = canned
    api.get_client = lambda: fake
    return TestClient(api.app)


def test_endpoint_returns_tool_calls_with_finish_reason():
    c = _client('```json\n{"tool_calls":[{"name":"browser_navigate",'
                '"arguments":{"url":"https://www.youtube.com"}}]}\n```')
    choice = c.post("/v1/chat/completions", json={
        "model": "deepseek-chat", "tools": TOOLS,
        "messages": [{"role": "user", "content": "go to youtube"}],
    }).json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert _names(choice["message"]["tool_calls"]) == ["browser_navigate"]


def test_endpoint_final_answer_still_stops_normally():
    c = _client("YouTube loaded fine.")
    choice = c.post("/v1/chat/completions", json={
        "model": "deepseek-chat", "tools": TOOLS,
        "messages": [{"role": "user", "content": "what happened"}],
    }).json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "YouTube loaded fine."
    assert "tool_calls" not in choice["message"]


def test_streaming_emits_a_tool_calls_delta():
    c = _client('{"tool_calls":[{"name":"browser_navigate","arguments":{"url":"https://x.com"}}]}')
    body = c.post("/v1/chat/completions", json={
        "model": "deepseek-chat", "tools": TOOLS, "stream": True,
        "messages": [{"role": "user", "content": "go"}],
    }).text
    assert '"tool_calls"' in body
    assert '"finish_reason": "tool_calls"' in body
    assert body.rstrip().endswith("data: [DONE]")


def test_tool_choice_none_suppresses_tools():
    c = _client("Just answering in text.")
    choice = c.post("/v1/chat/completions", json={
        "model": "deepseek-chat", "tools": TOOLS, "tool_choice": "none",
        "messages": [{"role": "user", "content": "hi"}],
    }).json()["choices"][0]
    assert choice["finish_reason"] == "stop"
