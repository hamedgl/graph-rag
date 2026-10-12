"""One structured-output call that works for every model behind the gateway, plus ops tracing.

Tested 2026-10-12 against the gateway (LLM_BASE_URL):
  cx/*  (gpt):    Responses API enforces json_schema; chat json_schema is IGNORED
  ag/*  (gemini): Responses API broken (empty output); chat json_schema IGNORED; forced tool call ENFORCED

Tracing (Stage 2.7): every LLM call / DB query / embedding records {kind, purpose, model, seconds, tokens}
  - into the current thread's trace (start_trace() / get_trace(): one per eval job, jobs run 1-per-thread)
  - as one JSON line in stage2/out/logs/calls.jsonl (structured log)
"""
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

import env  # noqa: F401  (loads .env)

# Normal calls take 1-3 s; one gateway call once hung 180 s and killed an eval run (seen in calls.jsonl).
# Timeout + SDK retries turn a hang into a fast retry instead.
client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"], timeout=30, max_retries=2)
QUERY_MODEL = os.environ["LLM_MODEL"]                               # answering, Text2Cypher, routing
INGEST_MODEL = os.environ.get("INGEST_MODEL", QUERY_MODEL)          # extraction, entity resolution

LOG = Path(__file__).parent.parent / "stage2" / "out" / "logs" / "calls.jsonl"
_local, _log_lock = threading.local(), threading.Lock()


def start_trace():
    _local.trace = []


def get_trace():
    return getattr(_local, "trace", [])


def record(kind, purpose, seconds, model=None, tokens_in=0, tokens_out=0):
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, "purpose": purpose, "model": model,
           "seconds": round(seconds, 3), "tokens_in": tokens_in, "tokens_out": tokens_out}
    if hasattr(_local, "trace"):
        _local.trace.append(rec)
    with _log_lock:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")


def call_json(model, instructions, user_input, name, schema):
    """Return a dict that conforms to `schema` (strict JSON schema). `name` doubles as the trace purpose."""
    t = time.perf_counter()
    # ponytail: transport chosen by gateway prefix; add a branch when a new provider behaves differently
    if model.startswith("ag/"):
        # Forced tool calls are not 100%: gemini occasionally answers with fenced JSON text instead.
        # Retry rather than parse that text (it bypassed schema enforcement).
        for _ in range(3):
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": instructions}, {"role": "user", "content": user_input}],
                tools=[{"type": "function", "function": {"name": name, "strict": True, "parameters": schema}}],
                tool_choice={"type": "function", "function": {"name": name}})
            u = r.usage
            record("llm", name, time.perf_counter() - t, model, getattr(u, "prompt_tokens", 0) or 0,
                   getattr(u, "completion_tokens", 0) or 0)
            if calls := r.choices[0].message.tool_calls:
                return json.loads(calls[0].function.arguments)
            t = time.perf_counter()
        raise ValueError(f"{model} returned no tool call in 3 tries: {r.choices[0].message.content!r:.200}")
    r = client.responses.create(model=model, instructions=instructions, input=user_input,
                                text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}})
    u = r.usage
    record("llm", name, time.perf_counter() - t, model, getattr(u, "input_tokens", 0) or 0,
           getattr(u, "output_tokens", 0) or 0)
    return json.loads(r.output_text)
