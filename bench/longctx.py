#!/usr/bin/env python3
"""Long-context test: prefill speed, decode speed and needle-in-a-haystack retrieval at several lengths and depths.

The haystack is real, varied text (MMLU-Pro questions), so the model cannot spot the needle by pattern alone.
Every request starts with a unique nonce so prefix caching cannot shortcut the prefill.
"""
import asyncio, os, random, sys
import pandas as pd
from common import chat, save

SIZES = [int(x) for x in sys.argv[1:]] or [16_000, 64_000, 128_000, 250_000]
# Models that always think (GLM-5.3-Flash) need room to reason before the passphrase: NEEDLE_EFFORT=low
# answers with thinking on at that effort, NEEDLE_MAXTOK raises the 48-token answer budget.
EFFORT = os.environ.get("NEEDLE_EFFORT")
NEEDLE_MAXTOK = int(os.environ.get("NEEDLE_MAXTOK", 48))
TAG = os.environ.get("LLM_TAG")
DEPTHS = [0.1, 0.5, 0.9]
CHARS_PER_TOKEN = 4.2  # rough for English; actual token counts come back from the server

rng = random.Random(7)
qs = pd.read_parquet("data/mmlu_pro.parquet")["question"].tolist()
rng.shuffle(qs)
corpus = "\n\n".join(qs)


def haystack(tokens, depth, needle):
    body = corpus[: int(tokens * CHARS_PER_TOKEN)]
    cut = body.rfind("\n\n", 0, int(len(body) * depth)) or int(len(body) * depth)
    return body[:cut] + "\n\n" + needle + "\n\n" + body[cut:]


async def main():
    rows = []
    for size in SIZES:
        for depth in DEPTHS:
            code = f"{rng.choice(['amber', 'cobalt', 'violet', 'saffron', 'teal'])}-{rng.randint(1000, 9999)}"
            needle = f"IMPORTANT NOTE: the secret passphrase for the archive vault is {code}."
            doc = haystack(size, depth, needle)
            prompt = (f"[run {rng.random():.6f}]\nBelow is a long collection of exam questions. One line in it is "
                      f"different.\n\n{doc}\n\nWhat is the secret passphrase for the archive vault? Reply with the "
                      f"passphrase only.")
            r = await chat([{"role": "user", "content": prompt}], max_tokens=NEEDLE_MAXTOK, thinking=bool(EFFORT), effort=EFFORT)
            ok = code in r["text"]
            row = {"target_tokens": size, "depth": depth, "prompt_tokens": r["prompt_tokens"], "ttft": r["ttft"],
                   "prefill_tps": r["prompt_tokens"] / r["ttft"], "found": ok, "answer": r["text"].strip()[:80],
                   "expected": code}
            rows.append(row)
            print(f"{r['prompt_tokens']:7d} tok  depth {int(depth*100):2d}%  TTFT {r['ttft']:6.1f}s  "
                  f"prefill {row['prefill_tps']:6,.0f} tok/s  needle {'FOUND' if ok else 'MISSED'} ({row['answer']!r})",
                  flush=True)
        # decode speed with this much context already in the cache-free prompt
        r = await chat([{"role": "user", "content": f"[d {rng.random()}]\n" + corpus[: int(size * CHARS_PER_TOKEN)]
                         + "\n\nWrite a 400-word essay about the main subject areas these questions cover."}],
                       max_tokens=512)
        rows.append({"target_tokens": size, "decode_test": True, "prompt_tokens": r["prompt_tokens"],
                     "decode_tps": r["decode_tps"], "completion_tokens": r["completion_tokens"]})
        print(f"{r['prompt_tokens']:7d} tok  decode after long prompt: {r['decode_tps']:.1f} tok/s "
              f"({r['completion_tokens']} tokens)", flush=True)
    print("saved", save((f"{TAG}_" if TAG else "") + "longctx", rows))

asyncio.run(main())
