"""Thin wrapper around the LLM provider.

The agent loop only calls chat() with provider-neutral messages, so it never
touches an SDK: swapping providers (e.g. Ollama) means a new wrapper, and tests
can replace chat() with a scripted fake.

Messages:
    {"role": "user", "text": str}
    {"role": "assistant", "raw": ...}     # response.as_message(): the provider's own turn, unchanged
    {"role": "tool", "results": [{"id": str | None, "name": str, "result": dict}, ...]}
                                          # one entry per tool call of the previous assistant turn

Config comes from env vars: LLM_PROVIDER (only "gemini" so far), LLM_MODEL, GEMINI_KEY.
LLM_CACHE=0 turns the response cache off (e.g. to measure real latency).
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors, types

RETRY_CODES = {429, 500, 502, 503, 504}  # rate limit / transient server errors
MAX_RETRIES = 4
MAX_WAIT_S = 60      # if the API asks us to wait longer (a daily quota), give up at once
TIMEOUT_MS = 60_000
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "llm"


class QuotaExhausted(RuntimeError):
    """The free-tier quota is used up; retrying now would only waste time."""


@dataclass
class ToolCall:
    id: str | None  # some providers/models don't return call ids
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str  # final text; "" when the model only made tool calls
    tool_calls: list[ToolCall]
    usage: dict = field(default_factory=dict)
    raw: Any = None  # provider message, must go back into the history unchanged
    cached: bool = False  # served from the cache: no API call, so latency is ~0

    def as_message(self) -> dict:
        # Not rebuilt from text/tool_calls: Gemini attaches "thought signatures" to
        # its function-call parts and rejects a history where they are missing.
        return {"role": "assistant", "raw": self.raw}


_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        provider = os.environ.get("LLM_PROVIDER", "gemini")
        if provider != "gemini":
            raise NotImplementedError(f"LLM_PROVIDER={provider!r} is not supported yet")
        _client = genai.Client(api_key=os.environ["GEMINI_KEY"],
                               http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    return _client


def _to_contents(messages: list[dict]) -> list[types.Content]:
    contents = []
    for m in messages:
        if m["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=m["text"])]))
        elif m["role"] == "assistant":
            contents.append(m["raw"])
        elif m["role"] == "tool":
            # Gemini expects tool results as a user turn made of function_response parts.
            contents.append(types.Content(role="user", parts=[
                types.Part(function_response=types.FunctionResponse(id=r["id"], name=r["name"], response=r["result"]))
                for r in m["results"]
            ]))
        else:
            raise ValueError(f"unknown role {m['role']!r}")
    return contents


def _config(system: str, tools: list[dict] | None, temperature: float,
            force_tool: str | None = None) -> types.GenerateContentConfig:
    declarations = [
        types.FunctionDeclaration(name=t["name"], description=t["description"],
                                  parameters_json_schema=t["parameters"])
        for t in tools or []
    ]
    return types.GenerateContentConfig(
        system_instruction=system,
        tools=[types.Tool(function_declarations=declarations)] if declarations else None,
        # Otherwise the SDK may run the tool loop itself
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        # mode ANY: the answer must be a call to this tool, never free text. None is dropped
        # from the cache key (exclude_none), so callers that don't force keep their entries.
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
            mode="ANY", allowed_function_names=[force_tool])) if force_tool else None,
        temperature=temperature,
    )


def _retry_delay(e: errors.APIError) -> float | None:
    """Seconds the API asks us to wait, if it says (a google.rpc.RetryInfo detail like '77656s')."""
    error = e.details.get("error", {}) if isinstance(e.details, dict) else {}
    for d in error.get("details", []):
        if d.get("@type", "").endswith("RetryInfo") and "retryDelay" in d:
            return float(d["retryDelay"].rstrip("s"))
    return None


def _generate(model: str, contents, config) -> types.GenerateContentResponse:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return _get_client().models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            if e.code not in RETRY_CODES:
                raise
            delay = _retry_delay(e) or min(2 ** attempt, 30)  # else exponential backoff: 1, 2, 4, 8 s
            if delay > MAX_WAIT_S:
                raise QuotaExhausted(f"{model}: quota exhausted, the API says retry in {delay / 3600:.1f} h") from e
            if attempt == MAX_RETRIES:
                raise
            time.sleep(delay)


def _cache_key(model: str, config, contents) -> str:
    """Hash of everything that determines the answer. The system prompt and tools are
    inside config, so editing the prompt automatically misses the old entries."""
    payload = {"model": model,
               "config": config.model_dump(mode="json", exclude_none=True),
               "contents": [c.model_dump(mode="json", exclude_none=True) for c in contents]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _parse(content: types.Content, usage: dict, cached: bool) -> LLMResponse:
    parts = content.parts or []
    return LLMResponse(
        # skip thought parts: they are the model's reasoning, not the reply
        text="".join(p.text for p in parts if p.text and not p.thought),
        tool_calls=[ToolCall(id=p.function_call.id, name=p.function_call.name, args=dict(p.function_call.args or {}))
                    for p in parts if p.function_call],
        usage=usage,
        raw=content,
        cached=cached,
    )


def chat(system: str, messages: list[dict], tools: list[dict] | None = None,
         temperature: float = 0.0, model: str | None = None, force_tool: str | None = None) -> LLMResponse:
    model = model or os.environ["LLM_MODEL"]
    config = _config(system, tools, temperature, force_tool)
    contents = _to_contents(messages)

    # Free-tier quotas are tiny, so identical requests are answered from disk.
    # model_dump(mode="json") base64-encodes Gemini's binary thought signatures,
    # so the stored turn round-trips exactly.
    use_cache = os.environ.get("LLM_CACHE", "1") != "0"
    path = CACHE_DIR / f"{_cache_key(model, config, contents)}.json"
    if use_cache and path.exists():
        hit = json.loads(path.read_text())
        return _parse(types.Content.model_validate(hit["content"]), hit["usage"], cached=True)

    response = _generate(model, contents, config)
    if not response.candidates or response.candidates[0].content is None:
        # e.g. blocked by a safety filter: there is no turn to append to the history
        raise RuntimeError(f"empty response: {response.prompt_feedback}")

    content = response.candidates[0].content
    u = response.usage_metadata
    usage = {
        "input_tokens": u.prompt_token_count or 0,
        "output_tokens": u.candidates_token_count or 0,
        "thinking_tokens": u.thoughts_token_count or 0,
    } if u else {}
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"content": content.model_dump(mode="json", exclude_none=True), "usage": usage}))
    return _parse(content, usage, cached=False)


def tool_results_message(results: list[dict]) -> dict:
    """Build the history entry answering one assistant turn's tool calls."""
    return {"role": "tool", "results": results}
