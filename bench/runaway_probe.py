#!/usr/bin/env python3
"""Garbage / runaway-generation gate: the first N HumanEval problems (env N=32; N=8 for the quick check), thinking on at
EFFORT (xhigh), the recommended thinking sampling (temperature 1.0, top_p 0.95, top_k 20), CONC requests in flight (8),
a MAX_TOKENS cap (12288). Stops early once STOP_AFTER requests (4) hit the cap. Per request: tokens, pass/fail, the
repeated-8-gram ratio of the reasoning (whole and last 2000 words) and the reasoning tail, so a runaway can be read, not
inferred. Every POLL s (20): speculative-decoding acceptance over that window from the server's counters. Optional
baseline: BASELINE=<a quality.py humaneval result JSON> adds that run's token count per problem for comparison.

usage: runaway_probe.py   (env N, CONC, EFFORT, MAX_TOKENS, STOP_AFTER, BASELINE, LLM_BASE_URL, LLM_API_KEY, LLM_TAG)
"""
import asyncio, json, os, re, subprocess, sys, tempfile, time, urllib.request
import pandas as pd
from common import BASE_URL, chat, save

N = int(os.environ.get("N", "32")); CONC = int(os.environ.get("CONC", "8")); EFFORT = os.environ.get("EFFORT", "xhigh")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "12288")); STOP_AFTER = int(os.environ.get("STOP_AFTER", "4"))
POLL = 20
ROOT = BASE_URL.rsplit("/v1", 1)[0]
PROD = ({r["task"]: r["completion_tokens"] for r in json.load(open(os.environ["BASELINE"]))["items"]}
        if os.environ.get("BASELINE") else {})


def run_humaneval_test(prompt, completion, test, entry):  # same as quality.py (which runs on import)
    code = prompt + "\n" + completion if f"def {entry}" in completion else prompt + completion
    prog = code + "\n\n" + test + f"\n\ncheck({entry})\n"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        open(p, "w").write(prog)
        try:
            res = subprocess.run([sys.executable, p], cwd=d, capture_output=True, timeout=20, env={"PATH": "/usr/bin:/bin"})
            return res.returncode == 0, res.stderr.decode()[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def counters():
    out = {}
    for line in urllib.request.urlopen(ROOT + "/metrics", timeout=30).read().decode().splitlines():
        for key in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total",
                    "vllm:generation_tokens_total"):
            if line.startswith(key + "{") or line.startswith(key + " "):
                out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
        if line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            pos = line.split('position="')[1].split('"')[0]
            out[f"pos{pos}"] = out.get(f"pos{pos}", 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def window(a, b):
    d = {k: b.get(k, 0) - a.get(k, 0) for k in b}
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
    pos = [round(d[f"pos{i}"] / drafts, 2) for i in range(16) if f"pos{i}" in d] if drafts else []
    return {"accept": round(1 + d.get("vllm:spec_decode_num_accepted_tokens_total", 0) / drafts, 2) if drafts else None,
            "per_pos": pos, "gen_tokens": int(d.get("vllm:generation_tokens_total", 0))}


def rep8(text):  # fraction of 8-word windows repeating an earlier window: ~0.0-0.1 normal, high for loops
    w = text.split(); grams = [tuple(w[i:i + 8]) for i in range(max(0, len(w) - 7))]
    return round(1 - len(set(grams)) / len(grams), 3) if grams else 0.0


async def main():
    rows = pd.read_parquet("data/humaneval.parquet").head(N).to_dict("records")
    await chat([{"role": "user", "content": "hi"}], max_tokens=4)
    sem = asyncio.Semaphore(CONC); capped = []; done = []; stop = asyncio.Event(); t0 = time.time()

    async def one(row):
        async with sem:
            if stop.is_set():
                return None
            r = await chat([{"role": "user", "content": "Complete this Python function. Return the full function in one "
                             "```python code block, no tests.\n\n```python\n" + row["prompt"] + "```"}],
                           max_tokens=MAX_TOKENS, thinking=True, effort=EFFORT)
        m = re.findall(r"```(?:python)?\n(.*?)```", r["text"], re.S)
        ok, err = await asyncio.to_thread(run_humaneval_test, row["prompt"], m[-1] if m else r["text"], row["test"],
                                          row["entry_point"])
        reasoning = r["reasoning"]; words = reasoning.split()
        rec = {"task": row["task_id"], "tokens": r["completion_tokens"], "prod_tokens": PROD.get(row["task_id"]),
               "finish": r["finish"], "ok": ok, "decode_tps": round(r["decode_tps"], 1), "rep8": rep8(reasoning),
               "rep8_tail": rep8(" ".join(words[-2000:])), "reasoning_chars": len(reasoning),
               "reasoning_head": reasoning[:1500], "reasoning_tail": reasoning[-4000:], "text": r["text"][-3000:]}
        done.append(rec)
        print(f"  {rec['task']:14s} {rec['tokens']:6d} tok (prod {rec['prod_tokens']}) {rec['finish']:6s} ok={ok!s:5s} "
              f"rep8 {rec['rep8']} tail {rec['rep8_tail']}  {rec['decode_tps']} tok/s  t+{time.time() - t0:.0f}s", flush=True)
        if r["finish"] == "length":
            capped.append(rec["task"])
            if len(capped) >= STOP_AFTER and not stop.is_set():
                print(f"  -- {len(capped)} requests hit the {MAX_TOKENS}-token cap; not starting more", flush=True)
                stop.set()
        return rec

    windows = []
    async def poll():
        prev = await asyncio.to_thread(counters)
        while True:
            await asyncio.sleep(POLL)
            cur = await asyncio.to_thread(counters); w = window(prev, cur); prev = cur
            w["t"] = round(time.time() - t0); windows.append(w)
            print(f"  [t+{w['t']:4d}s] acceptance {w['accept']} per-position {w['per_pos']} ({w['gen_tokens']} tokens)", flush=True)

    c0 = counters(); poller = asyncio.create_task(poll())
    await asyncio.gather(*(one(r) for r in rows))
    poller.cancel(); total = window(c0, counters())
    tok = sum(r["tokens"] for r in done); ptok = sum(r["prod_tokens"] or 0 for r in done)
    summ = {"n_done": len(done), "n_planned": N, "concurrency": CONC, "effort": EFFORT, "max_tokens": MAX_TOKENS,
            "tokens": tok, "prod_tokens_same_tasks": ptok, "ratio_vs_prod": round(tok / ptok, 2) if ptok else None,
            "hit_cap": len(capped), "passed": sum(r["ok"] for r in done), "acceptance": total["accept"],
            "per_pos": total["per_pos"], "wall_s": round(time.time() - t0),
            "rep8_median": sorted(r["rep8"] for r in done)[len(done) // 2] if done else None}
    print(f"SUMMARY {json.dumps(summ)}", flush=True)
    tag = os.environ.get("LLM_TAG")
    print("saved", save((f"{tag}_" if tag else "") + "runaway_probe", {"summary": summ, "windows": windows, "items": done}))

asyncio.run(main())
