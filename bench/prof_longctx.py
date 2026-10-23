#!/usr/bin/env python3
"""Long-context decode probe (W37, 2026-10-22).

N concurrent requests with CTX-token prompts (unique from the first token, MMLU-Pro corpus text like swarm_bench)
decode together; aggregate decode tok/s, running requests and draft acceptance are read from the server's counters
over a steady window (all N past their first token). Optional: the torch profiler over the window (PROFILE=1), and
one extra fresh prefill of MIX tokens started at the window's start (the window then lasts until that prefill ends,
so it measures decode under an interleaved prefill and the prefill's speed under decode).
Request shape = swarm_bench: no thinking, ignore_eos, temperature 0.7 / top-p 0.8 / top-k 20.

Env: N (8; 0 = only the MIX prefill, alone), CTX (80000), WIN (20 s), PROFILE (0), MIX (0), MAXTOK (8000), TAG.  Prints one PROBE {json} line.
"""
import asyncio, json, os, time, urllib.request
import pandas as pd
from openai import AsyncOpenAI

BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
ROOT = BASE.rsplit("/v1", 1)[0]
MODEL = os.environ.get("LLM_MODEL", "flash-next")
N = int(os.environ.get("N", 8)); CTX = int(os.environ.get("CTX", 80000)); WIN = float(os.environ.get("WIN", 20))
PROFILE = os.environ.get("PROFILE", "0") == "1"; MIX = int(os.environ.get("MIX", 0))
MAXTOK = int(os.environ.get("MAXTOK", 8000)); TAG = os.environ.get("TAG", "probe")
CPT = 4.33                                  # corpus characters per token (Qwen3.8 tokenizer, as in swarm_bench)
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True, "top_k": 20}
AUTH = {"Authorization": "Bearer " + os.environ["LLM_API_KEY"]} if os.environ.get("LLM_API_KEY") else {}
client = AsyncOpenAI(base_url=BASE, api_key=os.environ.get("LLM_API_KEY", "none"), timeout=3600, max_retries=0)
TEXT = "\n\n".join(pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mmlu_pro.parquet"))["question"].tolist())


def messages(label, i, n_tok):
    n = max(200, int(n_tok * CPT))
    off = (i * 1_000_003 + n_tok * 7) % max(1, len(TEXT) - n)
    return [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"[{label} {i}]\n" + TEXT[off:off + n]
             + "\n\nContinue writing about the topics above, in detail."}]


def metrics():
    out = {}
    for line in urllib.request.urlopen(f"{ROOT}/metrics", timeout=30).read().decode().splitlines():
        if not line.startswith("vllm:"):
            continue
        name, _, val = line.rpartition(" ")
        key = name.split("{")[0]
        try:
            out[key] = out.get(key, 0.0) + float(val)
        except ValueError:
            pass
    return out


def post(path):
    urllib.request.urlopen(urllib.request.Request(f"{ROOT}{path}", data=b"", method="POST", headers=AUTH), timeout=600).read()


async def decode(msgs, i, firsts, stop):
    stream = await client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=MAXTOK, stream=True,
                                                  temperature=0.7, top_p=0.8, extra_body=EXTRA)
    async for ev in stream:
        if ev.choices and ev.choices[0].delta.content and i not in firsts:
            firsts[i] = time.time()
        if stop.is_set():
            break
    await stream.close()


async def main():
    msgs = [messages(f"probe-{CTX}", i, CTX) for i in range(N)]
    t = time.time()                          # prefill phase (a cache hit when the same CTX ran before)
    r = await asyncio.gather(*(client.chat.completions.create(model=MODEL, messages=m, max_tokens=1, temperature=0.0,
                                                              extra_body=EXTRA) for m in msgs))
    prefill_s = time.time() - t; prompt = sum(x.usage.prompt_tokens for x in r)
    firsts, stop = {}, asyncio.Event()
    tasks = [asyncio.create_task(decode(m, i, firsts, stop)) for i, m in enumerate(msgs)]
    t = time.time()
    while len(firsts) < N and time.time() - t < 900:
        await asyncio.sleep(0.2)
    await asyncio.sleep(3)
    m1 = metrics(); t1 = time.time()
    if PROFILE:
        post("/start_profile")
    mix_s = None
    if MIX:
        mm = messages(f"mix-{time.time_ns()}", 0, MIX)
        await client.chat.completions.create(model=MODEL, messages=mm, max_tokens=1, temperature=0.0, extra_body=EXTRA)
        mix_s = time.time() - t1
    else:
        await asyncio.sleep(WIN)
    m2 = metrics(); t2 = time.time()
    if PROFILE:
        post("/stop_profile")
    stop.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    d = {k: m2.get(k, 0) - m1.get(k, 0) for k in m2}
    dt = t2 - t1
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
    res = {"tag": TAG, "n": N, "ctx": CTX, "prompt_tokens": prompt, "prefill_phase_s": round(prefill_s, 1),
           "window_s": round(dt, 2), "profiled": PROFILE, "mix": MIX, "mix_prefill_s": mix_s and round(mix_s, 2),
           "mix_prefill_tok_s": mix_s and round(MIX / mix_s),
           "decode_tok_s": round(d.get("vllm:generation_tokens_total", 0) / dt, 1),
           "per_stream_tok_s": round(d.get("vllm:generation_tokens_total", 0) / dt / max(N, 1), 1),
           "running": [m1.get("vllm:num_requests_running"), m2.get("vllm:num_requests_running")],
           "mean_accept_len": round(1 + d.get("vllm:spec_decode_num_accepted_tokens_total", 0) / drafts, 2) if drafts else None,
           "draft_tokens_per_draft": round(d.get("vllm:spec_decode_num_draft_tokens_total", 0) / drafts, 2) if drafts else None,
           "kv_pct": round(100 * m2.get("vllm:kv_cache_usage_perc", 0), 1)}
    print("PROBE " + json.dumps(res), flush=True)


asyncio.run(main())
