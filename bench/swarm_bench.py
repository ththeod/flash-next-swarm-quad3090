#!/usr/bin/env python3
"""Agent-swarm benchmark: N concurrent agent sessions with long, growing contexts (multi-turn, shared system prompt,
tool outputs, tool time, parents idling while subagents run, sessions ending and new ones starting).

Shape calibrated from aggregate statistics of a 2026-10-13 capture of a real coding-agent harness (23 sessions, 559 requests; only aggregates were kept): prompt p50 114K /
p90 218K; new tokens per turn p50 ~600 / p90 ~7.4K; answer p50 ~800 / p90 ~2.9K tokens; tool time p50 3 s / p90 23 s,
parents up to ~20 min; fresh subagents start at ~7K tokens. No captured content is used: text comes from the MMLU-Pro
question corpus, unique per session through a header line (prefix hashes chain, so later blocks differ too).

Phases: warm-up (every session's first turn, i.e. the cold prefill of the initial contexts), then a measured window of
DURATION seconds. Reports server-side counter deltas over the window (computed / GPU-hit / RAM-loaded prompt tokens,
queue time, TTFT, offload bytes) plus client-side turn stats. Same SEED -> same session plans.

Env: N_AGENTS (40), DURATION (1200 s), SEED (1), START_MIN/START_MAX (7000/150000 initial context tokens),
END_MIN/END_MAX (120000/235000 context at which a session ends), PAUSE_FRAC (0.15), TAG (results/<TAG>_swarm.json),
WARM_TURNS (1: the window opens once every slot finished its first turn; 2 also waits out the burst of second turns the
cold start queues, whose 200-450 s waits otherwise land in the window's first ~100 s).
turns_per_min_late counts only completions after the window's first 200 s (comparable across warm-up settings).
"""
import asyncio, json, math, os, random, time, urllib.request
import pandas as pd
from openai import AsyncOpenAI
from common import save

BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("LLM_MODEL", "flash-next")
N_AGENTS = int(os.environ.get("N_AGENTS", 40))
DURATION = float(os.environ.get("DURATION", 1200))
SEED = int(os.environ.get("SEED", 1))
START_MIN, START_MAX = int(os.environ.get("START_MIN", 7000)), int(os.environ.get("START_MAX", 150000))
END_MIN, END_MAX = int(os.environ.get("END_MIN", 120000)), int(os.environ.get("END_MAX", 235000))
PAUSE_FRAC = float(os.environ.get("PAUSE_FRAC", 0.15))
TAG = os.environ.get("TAG", "swarm")
WARM_TURNS = int(os.environ.get("WARM_TURNS", 1))
CPT = 4.33                                  # corpus characters per token (Qwen3.8 tokenizer, measured)
client = AsyncOpenAI(base_url=BASE, api_key=os.environ.get("LLM_API_KEY", "none"), timeout=7200, max_retries=0)
TEXT = "\n\n".join(pd.read_parquet("data/mmlu_pro.parquet")["question"].tolist())
SYSTEM = ("You are a coding agent working in a large repository. Use the tools to inspect files and run commands.\n\n"
          + TEXT[:int(6500 * CPT)])         # shared by every session, like an agent harness's system prompt + tool schemas


def lognorm(rng, p50, p90, lo, hi):
    sigma = math.log(p90 / p50) / 1.2816
    return int(min(hi, max(lo, rng.lognormvariate(math.log(p50), sigma))))


def chunk(rng, tokens):
    n = int(tokens * CPT)
    start = rng.randint(int(6500 * CPT), len(TEXT) - n - 1) if n < len(TEXT) - 7000 * CPT else int(6500 * CPT)
    return TEXT[start:start + n]


def metrics():
    t = urllib.request.urlopen(BASE.replace("/v1", "") + "/metrics", timeout=30).read().decode()
    out = {}
    for line in t.splitlines():
        if line.startswith("#") or "_created" in line or "_bucket" in line:
            continue
        name, _, val = line.rpartition(" ")
        key = name.split("{")[0]
        if 'source="' in name:
            key += ":" + name.split('source="')[1].split('"')[0]
        try:
            out[key] = out.get(key, 0.0) + float(val)
        except ValueError:
            pass
    return out


class Stats:
    def __init__(self):
        self.turns = []
        self.measuring = False
        self.errors = 0


async def turn(messages, max_tokens, rng):
    t0 = time.perf_counter(); first = None; parts = []; usage = None
    stream = await client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=max_tokens, stream=True, temperature=0.7, top_p=0.8,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True, "top_k": 20})
    async for ev in stream:
        if ev.usage:
            usage = ev.usage
        if ev.choices and ev.choices[0].delta.content:
            first = first or time.perf_counter()
            parts.append(ev.choices[0].delta.content)
    end = time.perf_counter()
    det = getattr(usage, "prompt_tokens_details", None)
    return "".join(parts), {"prompt": usage.prompt_tokens, "out": usage.completion_tokens,
                            "cached": (getattr(det, "cached_tokens", 0) or 0) if det else 0,
                            "ttft": (first or end) - t0, "total": end - t0}


async def session_loop(slot, stats, stop_at, warm_done):
    """One agent slot: runs sessions back to back until stop_at."""
    rng = random.Random(SEED * 1000 + slot)
    gen = 0
    first_session = True
    while time.time() < stop_at:
        sid = f"{SEED}-{slot}-{gen}"; gen += 1
        if first_session:
            start = int(START_MIN + (START_MAX - START_MIN) * rng.random())
        else:  # replacement sessions: mostly fresh subagents, some forked/resumed bigger contexts
            start = lognorm(rng, 1500, 30000, 300, 70000) + 6500
        end_ctx = rng.randint(END_MIN, END_MAX)
        pause_turn = rng.randint(2, 12) if rng.random() < PAUSE_FRAC else -1
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"[session {sid}]\nTask context follows.\n\n" + chunk(rng, max(200, start - 6600))}]
        ctx = start; k = 0
        while time.time() < stop_at:
            out_n = lognorm(rng, 800, 2900, 60, 4000)
            try:
                text, r = await turn(msgs, out_n, rng)
            except Exception as e:  # noqa: BLE001
                stats.errors += 1
                print(f"[{slot}] error: {e!r}"[:200], flush=True)
                await asyncio.sleep(5)
                break
            r.update(slot=slot, sid=sid, k=k, t=time.time(), measured=stats.measuring)
            stats.turns.append(r)
            if first_session and k == WARM_TURNS - 1:
                warm_done.add(slot)
            ctx = r["prompt"] + r["out"]; k += 1
            new = lognorm(rng, 600, 7400, 20, 40000)
            if ctx + new + 4200 > min(END_MAX, end_ctx) or k >= 70:
                warm_done.add(slot)         # a first session shorter than WARM_TURNS still counts as warm
                break
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user", "content": f"Tool output ({sid} #{k}):\n" + chunk(rng, new)})
            gap = 600 * rng.random() + 240 if k == pause_turn else lognorm(rng, 3.0, 23.0, 0.1, 120)
            await asyncio.sleep(gap)
        first_session = False


async def main():
    stats = Stats(); warm_done = set()
    t_start = time.time()
    stop_at = t_start + 7200 + DURATION     # generous; the measured window ends the run below
    tasks = [asyncio.create_task(session_loop(i, stats, stop_at, warm_done)) for i in range(N_AGENTS)]
    while len(warm_done) < N_AGENTS:        # warm-up: every slot's first (cold) turn
        await asyncio.sleep(5)
        if time.time() - t_start > 7200:
            break
    warm_s = time.time() - t_start
    print(f"warm-up done in {warm_s:.0f} s ({len(stats.turns)} turns, WARM_TURNS={WARM_TURNS}); measuring {DURATION:.0f} s",
          flush=True)
    m0 = metrics(); t0 = time.time(); stats.measuring = True
    running, kv = [], []
    while time.time() - t0 < DURATION:
        await asyncio.sleep(30)
        done = [x for x in stats.turns if x["measured"]]
        g = metrics()
        running.append(g.get("vllm:num_requests_running", 0)); kv.append(g.get("vllm:kv_cache_usage_perc", 0))
        print(f"  t+{time.time() - t0:.0f}s turns {len(done)} running {running[-1]:.0f}", flush=True)
    m1 = metrics(); dt = time.time() - t0
    for t in tasks:
        t.cancel()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    turns = [x for x in stats.turns if x["measured"]]
    n = max(1, len(turns))
    comp = d.get("vllm:prompt_tokens_by_source_total:local_compute", 0)
    gpu = d.get("vllm:prompt_tokens_by_source_total:local_cache_hit", 0)
    ram = d.get("vllm:prompt_tokens_by_source_total:external_kv_transfer", 0)
    fin = d.get("vllm:request_success_total", 0) or n
    res = {"tag": TAG, "agents": N_AGENTS, "seed": SEED, "window_s": round(dt), "warmup_s": round(warm_s),
           "turns": len(turns), "turns_per_min": round(60 * len(turns) / dt, 2),
           "turns_per_min_late": round(60 * sum(1 for x in turns if x["t"] - t0 >= 200) / max(1, dt - 200), 2),
           "warm_turns": WARM_TURNS,
           "gen_tok_s": round(d.get("vllm:generation_tokens_total", 0) / dt, 1),
           "prompt_tok_per_turn": round((comp + gpu + ram) / max(fin, 1)),
           "computed_per_turn": round(comp / max(fin, 1)), "gpu_hit_per_turn": round(gpu / max(fin, 1)),
           "ram_per_turn": round(ram / max(fin, 1)),
           "computed_share": round(comp / max(1, comp + gpu + ram), 4),
           "queue_s": round(d.get("vllm:request_queue_time_seconds_sum", 0) / max(1, d.get("vllm:request_queue_time_seconds_count", 0)), 1),
           "ttft_s": round(d.get("vllm:time_to_first_token_seconds_sum", 0) / max(1, d.get("vllm:time_to_first_token_seconds_count", 0)), 1),
           "turn_total_s_mean": round(sum(x["total"] for x in turns) / n, 1),
           "ram_load_GBps": round(d.get("vllm:kv_offload_load_bytes_total", 0) / dt / 1e9, 2),
           "ram_store_GBps": round(d.get("vllm:kv_offload_store_bytes_total", 0) / dt / 1e9, 2),
           "ram_rows_stored": d.get("vllm:kv_offload_cpu_allocation_size_sum", 0),
           "preemptions": d.get("vllm:num_preemptions_total", 0), "errors": stats.errors,
           "mean_prompt_k": round(sum(x["prompt"] for x in turns) / n / 1000, 1),
           "mean_running": round(sum(running) / max(1, len(running)), 2),
           "mean_kv_pct": round(100 * sum(kv) / max(1, len(kv)), 1),
           "deferred_lookups": d.get("vllm:kv_offload_lookup_async_delay_seconds_count", 0),
           "deferred_mean_s": round(d.get("vllm:kv_offload_lookup_async_delay_seconds_sum", 0)
                                    / max(1, d.get("vllm:kv_offload_lookup_async_delay_seconds_count", 0)), 1),
           "discarded_rows": d.get("vllm:kv_offload_discarded_rows_total", 0)}
    print("SWARM " + json.dumps(res), flush=True)
    save(f"{TAG}_swarm", {"summary": res, "turns": stats.turns, "metric_delta": d})

asyncio.run(main())
