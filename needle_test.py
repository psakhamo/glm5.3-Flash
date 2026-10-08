#!/usr/bin/env python3
"""Needle-in-a-haystack long-context retrieval test for GLM-5.3-Flash.

Probes whether the sparse indexer retrieves information deep in long prompts.
vLLM PR #59412 ("page-aligned kernel blocks for pooled indexers") reportedly
fixes a ROCm bug where the indexer read the wrong keys on long prompts.
Another team measured 4-7/50 before the fix vs 50/50 after at 4K-64K.

Usage:  python3 needle_test.py [--quick]
"""
import json, random, sys, time, urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "glm-5-3-flash-fp8"

# Varied filler so the context cannot be trivially compressed/deduplicated.
TOPICS = [
    "The maintenance crew inspected the northern conveyor belt for wear.",
    "Quarterly rainfall in the coastal district exceeded the ten-year average.",
    "Librarians catalogued the donated manuscripts by acquisition date.",
    "The orchestra rehearsed the second movement until the brass settled.",
    "Soil samples from the eastern ridge showed elevated mineral content.",
    "Ferry schedules shifted by twenty minutes during the winter timetable.",
    "The bakery switched suppliers after the flour shipment was delayed.",
    "Surveyors marked the boundary stones before the frost set in.",
    "Apprentice welders practised on scrap plate before touching the hull.",
    "The observatory logged unusually clear seeing conditions all week.",
]


SUBJ = ["The harbour office", "A field technician", "The regional archive",
        "Warehouse staff", "The survey team", "A night supervisor",
        "The logistics desk", "Inspectors", "The depot manager",
        "A visiting auditor", "The planning board", "Maintenance records"]
VERB = ["logged", "revised", "archived", "flagged", "approved", "deferred",
        "cross-checked", "reissued", "annotated", "escalated"]
OBJ = ["the loading schedule", "the humidity readings", "the shift roster",
       "the calibration notes", "the freight manifest", "the sample register",
       "the inspection form", "the delivery window", "the parts inventory",
       "the access log"]
TAIL = ["before the audit window closed.", "pending supervisor review.",
        "after the second inspection.", "without further comment.",
        "ahead of the quarterly return.", "once the backlog cleared.",
        "subject to a follow-up visit.", "in line with the revised policy."]


def build_prompt(target_tokens: int, depth: float, city: str, code: str, rng):
    """~4 chars/token for English prose; we report the ACTUAL token count.

    Filler lines are combinatorially unique (numbered entries with varying
    subject/verb/object/date), not a small set repeated thousands of times.
    Degenerate repeated filler makes the model comment on the repetition
    instead of answering, which would confound a retrieval measurement.
    """
    needle = f"The secret passcode for {city} is {code}."
    target_chars = target_tokens * 4
    lines, n, i = [], 0, 0
    while n < target_chars:
        i += 1
        s = (f"Entry {i:05d} ({rng.randint(2015,2025)}-{rng.randint(1,12):02d}-"
             f"{rng.randint(1,28):02d}): {rng.choice(SUBJ)} {rng.choice(VERB)} "
             f"{rng.choice(OBJ)} for site {rng.randint(100,999)} "
             f"{rng.choice(TAIL)}")
        lines.append(s)
        n += len(s) + 1
    insert_at = max(0, min(len(lines) - 1, int(len(lines) * depth)))
    lines.insert(insert_at, needle)
    body = "\n".join(lines)
    return (
        "The following is a long document. Read it carefully.\n\n"
        f"{body}\n\n"
        f"Question: What is the secret passcode for {city}? "
        "Answer with only the number."
    ), needle


def ask(prompt, timeout=900):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    text = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
    return text, d.get("usage", {}), time.time() - t0


def main():
    quick = "--quick" in sys.argv
    rng = random.Random(1234)
    lengths = [4000, 16000] if quick else [4000, 16000, 64000, 118000]
    depths = [0.1, 0.5, 0.9]
    cities = ["Lisbon", "Osaka", "Calgary", "Nairobi", "Helsinki", "Bogota",
              "Perth", "Tallinn", "Quito", "Dakar", "Bergen", "Pune"]

    print(f"{'ctx_target':>10} {'ctx_actual':>10} {'depth':>6} {'found':>6} "
          f"{'code':>8} {'got':>22} {'sec':>7}")
    print("-" * 78)
    results = {}
    ci = 0
    for L in lengths:
        for dep in depths:
            city = cities[ci % len(cities)]; ci += 1
            code = f"{rng.randint(100000, 999999)}"
            prompt, _ = build_prompt(L, dep, city, code, rng)
            try:
                text, usage, dt = ask(prompt)
                ptok = usage.get("prompt_tokens", -1)
                ok = code in text
                snippet = " ".join(text.split())[:20]
            except Exception as e:
                ptok, ok, snippet, dt = -1, False, f"ERR {type(e).__name__}", 0.0
            results.setdefault(L, []).append(ok)
            print(f"{L:>10} {ptok:>10} {dep:>6.1f} {'YES' if ok else 'NO':>6} "
                  f"{code:>8} {snippet:>22} {dt:>7.1f}")

    print("\n" + "=" * 78)
    print("RETRIEVAL ACCURACY BY CONTEXT LENGTH")
    total_ok = total = 0
    for L, rs in results.items():
        ok, n = sum(rs), len(rs)
        total_ok += ok; total += n
        bar = "#" * ok + "." * (n - ok)
        print(f"  ~{L:>7} tokens : {ok}/{n}  [{bar}]")
    print(f"\n  OVERALL: {total_ok}/{total}")
    if total_ok < total:
        print("\n  >>> RETRIEVAL FAILURES DETECTED. Consistent with the bug that")
        print("      vLLM PR #59412 fixes (pooled-indexer block table addressing).")
    else:
        print("\n  >>> All needles retrieved on this build.")


if __name__ == "__main__":
    main()
