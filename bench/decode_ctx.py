#!/usr/bin/env python3
"""Agent-like decode test: N simultaneous requests, each with its own ~CTX-token code context (local vLLM sources)
and a coding instruction, thinking on at the given effort, recommended sampling. Measures what drafting is for:
per-request decode tok/s and aggregate output tok/s with long contexts, at several concurrency levels.

usage: decode_ctx.py CTX_TOKENS LEVEL [LEVEL ...]   (env: EFFORT=low, MAX_TOKENS=1024, LLM_TAG)
Each request gets a unique cache_salt, so nothing is reused across levels or runs (prefill is cold every time).
"""
import asyncio, glob, os, sys, time, uuid
import numpy as np
from common import chat, save

CTX = int(sys.argv[1]); LEVELS = [int(x) for x in sys.argv[2:]] or [1, 4, 8]
EFFORT = os.environ.get("EFFORT", "low"); MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
SRC = sorted(glob.glob(os.environ.get("DOC_TREE", "../vllm-swarm") + "/vllm/v1/**/*.py", recursive=True))
TASKS = ["Add type-checked input validation to the most complex public function above and show the full new function.",
         "Find one real bug or race condition in the code above, explain it, and write the fix as a unified diff.",
         "Write pytest unit tests for the scheduler-related class above, covering at least four edge cases.",
         "Refactor the longest function above into smaller helpers; show the complete refactored code.",
         "Explain the data flow through the code above, then write a docstring for each public class.",
         "Write a benchmark script that measures the hot path in the code above and prints a summary table.",
         "Identify the three most expensive operations above and rewrite them to be faster, with code.",
         "Port the most self-contained helper above to Rust, keeping the same behaviour, with tests."]


def context(i):
    """~CTX tokens (3.3 chars/token for this code) of source files, a different slice per request index."""
    want = int(CTX * 3.3); parts = []; n = 0; k = (i * 37) % len(SRC)
    while n < want:
        txt = open(SRC[k % len(SRC)]).read(); parts.append(f"# file: {SRC[k % len(SRC)]}\n{txt}"); n += len(txt); k += 1
    return "".join(parts)[:want]


async def level(n):
    prompts = [[{"role": "user", "content": f"{context(i)}\n\n{TASKS[i % len(TASKS)]}"}] for i in range(n)]
    t0 = time.perf_counter()
    res = await asyncio.gather(*(chat(p, max_tokens=MAX_TOKENS, thinking=True, effort=EFFORT, cache_salt=uuid.uuid4().hex)
                                 for p in prompts))
    wall = time.perf_counter() - t0
    out = sum(r["completion_tokens"] for r in res); dec = np.array([r["decode_tps"] for r in res])
    ttft = np.array([r["ttft"] for r in res])
    row = {"ctx": CTX, "concurrency": n, "wall_s": wall, "output_tokens": out, "aggregate_tps": out / wall,
           "per_request_decode_tps_mean": float(dec.mean()), "per_request_decode_tps_min": float(dec.min()),
           "ttft_p50": float(np.percentile(ttft, 50)), "ttft_max": float(ttft.max()),
           "prompt_tokens_mean": float(np.mean([r.get("prompt_tokens") or 0 for r in res]))}
    print(f"ctx {CTX//1000}K N={n}: per-request decode {row['per_request_decode_tps_mean']:6.1f} tok/s "
          f"(min {row['per_request_decode_tps_min']:5.1f}), aggregate {row['aggregate_tps']:6.1f} tok/s, "
          f"TTFT p50 {row['ttft_p50']:5.1f}s, {out} tokens in {wall:5.1f}s", flush=True)
    return row


async def main():
    rows = [await level(n) for n in LEVELS]
    tag = os.environ.get("LLM_TAG")
    print("saved", save((f"{tag}_" if tag else "") + f"decode_ctx{CTX//1000}k", rows))

asyncio.run(main())
