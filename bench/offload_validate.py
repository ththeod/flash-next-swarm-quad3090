#!/usr/bin/env python3
"""Validate CPU KV offloading correctness under concurrent RAM reloads.

1. Cold: 6 long documents (~30K tokens each, unique prefixes), one question each -> reference answers.
2. Flush the GPU prefix cache with ~850K tokens of unrelated prompts.
3. Warm: same 6 questions again (should be served from RAM) plus a second, different question per document,
   all 12 requests concurrently. Repeated questions must match the reference; every answer must look sane.
Then reports external (RAM) prefix hits from the server's metrics.
"""
import asyncio, os, random, re, urllib.request
import pandas as pd
from common import chat

N_DOCS = int(os.environ.get("N_DOCS", 6))
FLUSH_N = int(os.environ.get("FLUSH_N", 24))
text = "\n\n".join(pd.read_parquet("data/mmlu_pro.parquet")["question"].tolist())
rng = random.Random(11)


def metric(name):
    t = urllib.request.urlopen(os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1").rsplit("/v1", 1)[0] + "/metrics").read().decode()
    return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(name) and "_created" not in l)


def garbage(s):
    words = re.findall(r"\w+", s)
    if len(words) < 3:
        return False
    if re.search(r"(\b\w+\b)(?:\W+\1\b){4,}", s) or re.search(r"(\w{3,})\1{4,}", s):
        return True
    return len(set(words)) / len(words) < 0.3


async def ask(doc, q, n=60):
    r = await chat([{"role": "user", "content": doc + "\n\n" + q}], max_tokens=n)
    return r["text"].strip(), r


async def main():
    docs = []
    for i in range(N_DOCS):
        start = rng.randint(0, len(text) - 140_000)
        docs.append(f"[validate doc {i} {rng.random()}]\n" + text[start:start + 130_000])
    q1 = "In one sentence, what is the first question in the text above about?"
    q2 = "In one sentence, what is the last question in the text above about?"

    cold = await asyncio.gather(*(ask(d, q1) for d in docs))
    print("cold done; TTFTs:", [round(r["ttft"], 1) for _, r in cold], flush=True)

    fill = [f"[fill {i} {random.random()}]\n" + text[(150_000 * (i + 1)) % (len(text) - 150_000):][:150_000]
            for i in range(1, FLUSH_N + 1)]
    for k in range(0, FLUSH_N, 8):
        await asyncio.gather(*(ask(f, "Say OK.", 4) for f in fill[k:k + 8]))
    print("flush done", flush=True)

    ext0, gpu0 = metric("vllm:external_prefix_cache_hits_total"), metric("vllm:prefix_cache_hits_total")
    warm = await asyncio.gather(*([ask(d, q1) for d in docs] + [ask(d, q2) for d in docs]))
    ext1, gpu1 = metric("vllm:external_prefix_cache_hits_total"), metric("vllm:prefix_cache_hits_total")

    same = bad = 0
    for i in range(N_DOCS):
        a_cold, a_warm, a_q2 = cold[i][0], warm[i][0], warm[N_DOCS + i][0]
        match = a_cold[:80] == a_warm[:80]
        g = garbage(a_warm) or garbage(a_q2) or garbage(a_cold)
        same += match; bad += g
        print(f"doc {i}: repeat-match={match} garbage={g} TTFT warm {warm[i][1]['ttft']:.1f}s | "
              f"cold={a_cold[:70]!r} | warm={a_warm[:70]!r} | q2={a_q2[:60]!r}", flush=True)
    print(f"RESULT: repeat matches {same}/{N_DOCS}, garbage answers {bad}, "
          f"RAM hits +{ext1 - ext0:.0f} tok, GPU hits +{gpu1 - gpu0:.0f} tok", flush=True)

asyncio.run(main())
