#!/usr/bin/env python3
"""Concurrency sweep: N simultaneous streams, each generating up to 512 tokens (thinking off).

Reports aggregate output tok/s, per-stream decode tok/s and TTFT percentiles per level.
"""
import asyncio, os, sys, time
import numpy as np
from common import chat, save

LEVELS = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8, 16, 32]
TOPICS = ["a binary search tree in Python with insert, delete and in-order traversal",
          "the causes and consequences of the 1929 stock market crash",
          "a REST API in Go with JWT authentication",
          "how photosynthesis works, for a high-school student",
          "a SQL schema and queries for a library lending system",
          "the trade-offs between microservices and monoliths",
          "a bash script that rotates and compresses log files",
          "the history of the Byzantine Empire in brief"]


async def level(n):
    prompts = [[{"role": "user", "content": f"(request {i}) Explain or write {TOPICS[i % len(TOPICS)]}."}] for i in range(n)]
    t0 = time.perf_counter()
    res = await asyncio.gather(*(chat(p, max_tokens=512, temperature=0.7) for p in prompts))
    wall = time.perf_counter() - t0
    out = sum(r["completion_tokens"] for r in res)
    ttft = np.array([r["ttft"] for r in res]); dec = np.array([r["decode_tps"] for r in res])
    row = {"concurrency": n, "wall_s": wall, "output_tokens": out, "aggregate_tps": out / wall,
           "per_stream_decode_tps_mean": float(dec.mean()), "per_stream_decode_tps_min": float(dec.min()),
           "ttft_p50": float(np.percentile(ttft, 50)), "ttft_p95": float(np.percentile(ttft, 95)),
           "ttft_max": float(ttft.max())}
    print(f"N={n:3d}  aggregate {row['aggregate_tps']:7.1f} tok/s  per-stream {row['per_stream_decode_tps_mean']:6.1f} "
          f"(min {row['per_stream_decode_tps_min']:5.1f})  TTFT p50 {row['ttft_p50']:6.2f}s p95 {row['ttft_p95']:6.2f}s  "
          f"wall {wall:6.1f}s", flush=True)
    return row


async def main():
    await chat([{"role": "user", "content": "hi"}], max_tokens=4)  # warm-up
    # 2026-10-20: one unmeasured short round per level first. A fresh server JIT-compiles Triton kernels on the first
    # batch of each shape ("JIT compilation during inference"), which cost ~25% at N=2-8 on a cold boot. WARMUP=0 skips.
    if os.environ.get("WARMUP", "1") == "1":
        for n in LEVELS:
            await asyncio.gather(*(chat([{"role": "user", "content": f"(warm-up {i}) Explain or write {TOPICS[i % len(TOPICS)]}."}],
                                        max_tokens=128, temperature=0.7) for i in range(n)))
    rows = [await level(n) for n in LEVELS]
    tag = os.environ.get("LLM_TAG")
    print("saved", save((f"{tag}_" if tag else "") + "concurrency", rows))

asyncio.run(main())
