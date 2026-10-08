#!/usr/bin/env python3
"""Validate basic reasoning and tool calling on the DP8+EP8 GLM-5.3-Flash server."""
import json, sys, time, urllib.request, urllib.error

BASE = "http://127.0.0.1:8000"
MODEL = "glm-5-3-flash-fp8"
results = []


def post(path, payload, timeout=900):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


# ---- 1. model listing ----
try:
    m = get("/v1/models")
    ids = [d["id"] for d in m.get("data", [])]
    record("models endpoint", MODEL in ids, f"served ids={ids}")
except Exception as e:
    record("models endpoint", False, repr(e))
    sys.exit(1)

# ---- 2. basic completion ----
try:
    d, dt = post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly the word: OK"}],
        "max_tokens": 2048, "temperature": 0,
    })
    msg = d["choices"][0]["message"]
    txt = (msg.get("content") or "").strip()
    record("basic completion", bool(txt), f"{dt:.1f}s content={txt[:80]!r}")
except Exception as e:
    record("basic completion", False, repr(e))

# ---- 3. reasoning (glm45 reasoning parser should populate reasoning_content) ----
try:
    d, dt = post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content":
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
            "the ball. How much does the ball cost? Think it through, then give "
            "the final answer."}],
        "max_tokens": 4096, "temperature": 0,
    })
    msg = d["choices"][0]["message"]
    content = (msg.get("content") or "")
    reasoning = (msg.get("reasoning_content") or "")
    correct = "0.05" in content or "5 cent" in content.lower() \
        or "0.05" in reasoning or "5 cent" in reasoning.lower()
    record("reasoning: correct answer", correct,
           f"{dt:.1f}s answer_has_0.05={correct}")
    record("reasoning: parser populated reasoning_content", bool(reasoning),
           f"reasoning_content len={len(reasoning)} content len={len(content)}")
    print("   --- reasoning_content (first 400) ---")
    print("   " + reasoning[:400].replace("\n", "\n   "))
    print("   --- content (first 400) ---")
    print("   " + content[:400].replace("\n", "\n   "))
except Exception as e:
    record("reasoning", False, repr(e))

# ---- 4. tool calling (glm47 tool parser) ----
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]

try:
    d, dt = post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": "What's the weather in Austin, Texas? Use celsius."}],
        "tools": TOOLS, "tool_choice": "auto",
        "max_tokens": 4096, "temperature": 0,
    })
    msg = d["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    ok = bool(calls) and calls[0]["function"]["name"] == "get_weather"
    args = {}
    if calls:
        try:
            args = json.loads(calls[0]["function"]["arguments"])
        except Exception:
            args = {"_raw": calls[0]["function"]["arguments"]}
    record("tool call: emitted", ok,
           f"{dt:.1f}s finish={d['choices'][0].get('finish_reason')} calls={len(calls)}")
    record("tool call: args parsed", "city" in args, f"args={args}")

    # ---- 5. round trip: feed the tool result back ----
    if calls:
        d2, dt2 = post("/v1/chat/completions", {
            "model": MODEL,
            "messages": [
                {"role": "user",
                 "content": "What's the weather in Austin, Texas? Use celsius."},
                {"role": "assistant", "tool_calls": calls,
                 "content": msg.get("content") or ""},
                {"role": "tool", "tool_call_id": calls[0]["id"],
                 "content": json.dumps({"city": "Austin", "temp_c": 31,
                                        "condition": "sunny"})},
            ],
            "tools": TOOLS,
            "max_tokens": 4096, "temperature": 0,
        })
        final = (d2["choices"][0]["message"].get("content") or "")
        good = "31" in final
        record("tool call: result round-trip", good,
               f"{dt2:.1f}s final={final[:140]!r}")
except Exception as e:
    record("tool calling", False, repr(e))

# ---- summary ----
print("\n" + "=" * 60)
p = sum(1 for _, ok, _ in results if ok)
print(f"SUMMARY: {p}/{len(results)} checks passed")
for n, ok, det in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {n}")
sys.exit(0 if p == len(results) else 1)
