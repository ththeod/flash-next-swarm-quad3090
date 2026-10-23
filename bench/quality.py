#!/usr/bin/env python3
"""Quality benchmarks against the live server.

  gsm8k      first N test problems, exact numeric match on the final answer
  humaneval  all 164 problems, generated code executed against the official tests (subprocess, timeout)
  mmlu_pro   stratified sample across the 14 categories, letter-match on "the answer is (X)"
  tools      tool-calling and JSON-schema checks

Usage: quality.py <bench> [n] [effort] [max_tokens]   effort: low | medium | xhigh (default medium); thinking is on.
"""
import asyncio, json, os, re, subprocess, sys, tempfile, time
import pandas as pd
from common import chat, run_all, save, client, MODEL, SAMPLING

BENCH = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 0
EFFORT = sys.argv[3] if len(sys.argv) > 3 else "medium"
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # 0 = per-benchmark default
CONC = int(os.environ.get("CONC", 8))  # concurrent requests; the server runs at most 8 at once


def last_number(s):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return nums[-1].replace(",", "").rstrip(".") if nums else None


async def gsm8k():
    df = pd.read_parquet("data/gsm8k.parquet").head(N or 200)
    async def one(row):
        r = await chat([{"role": "user", "content": row["question"] + "\n\nSolve it, then give the final answer on the "
                         "last line as: #### <number>"}], max_tokens=MAXTOK or 8192, thinking=True, effort=EFFORT)
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        m = re.search(r"####\s*(-?[\d,]*\.?\d+)", r["text"])
        pred = (m.group(1).replace(",", "") if m else last_number(r["text"])) or ""
        ok = pred != "" and float(pred) == float(gold)
        return {**r, "gold": gold, "pred": pred, "ok": ok, "reasoning": r["reasoning"][-400:]}
    return await run_all(df.to_dict("records"), one, CONC)


def run_humaneval_test(prompt, completion, test, entry):
    # Keep the prompt: some problems define helpers there (e.g. encode_cyclic). A full redefinition of the
    # entry function in the completion simply replaces the prompt's docstring-only stub.
    code = prompt + "\n" + completion if f"def {entry}" in completion else prompt + completion
    prog = code + "\n\n" + test + f"\n\ncheck({entry})\n"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        open(p, "w").write(prog)
        try:
            res = subprocess.run([sys.executable, p], cwd=d, capture_output=True, timeout=20,
                                 env={"PATH": "/usr/bin:/bin"})
            return res.returncode == 0, res.stderr.decode()[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


async def humaneval():
    df = pd.read_parquet("data/humaneval.parquet")
    if N:
        df = df.head(N)
    async def one(row):
        r = await chat([{"role": "user", "content": "Complete this Python function. Return the full function in one "
                         "```python code block, no tests.\n\n```python\n" + row["prompt"] + "```"}],
                       max_tokens=MAXTOK or 8192, thinking=True, effort=EFFORT)
        m = re.findall(r"```(?:python)?\n(.*?)```", r["text"], re.S)
        completion = m[-1] if m else r["text"]
        ok, err = await asyncio.to_thread(run_humaneval_test, row["prompt"], completion, row["test"], row["entry_point"])
        return {**r, "task": row["task_id"], "ok": ok, "err": err, "reasoning": r["reasoning"][-300:]}
    return await run_all(df.to_dict("records"), one, CONC)


async def mmlu_pro():
    df = pd.read_parquet("data/mmlu_pro.parquet")
    per = max(1, (N or 280) // df["category"].nunique())
    df = df.groupby("category").sample(n=per, random_state=0)
    if os.environ.get("ONLY"):  # comma-separated positions in that sample (e.g. another model's misses)
        df = df.iloc[[int(x) for x in os.environ["ONLY"].split(",")]]
    async def one(row):
        opts = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(row["options"]))
        r = await chat([{"role": "user", "content": f"{row['question']}\n\n{opts}\n\nThink it through, then end with: "
                         "The answer is (X)"}], max_tokens=MAXTOK or 12288, thinking=True, effort=EFFORT)
        m = re.findall(r"answer is \(?([A-J])\)?", r["text"])
        pred = m[-1] if m else None
        return {**r, "category": row["category"], "gold": row["answer"], "pred": pred, "ok": pred == row["answer"],
                "reasoning": r["reasoning"][-300:]}
    return await run_all(df.to_dict("records"), one, CONC)


async def tools():
    out = []
    weather = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"},
               "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
    calc = {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount",
            "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from": {"type": "string"},
            "to": {"type": "string"}}, "required": ["amount", "from", "to"]}}}
    cases = [("What's the weather in Athens in celsius?", "get_weather", lambda a: "athens" in a.get("city", "").lower()),
             ("Convert 250 euros to US dollars.", "convert_currency", lambda a: float(a.get("amount", 0)) == 250),
             ("Tell me a one-line joke.", None, None)]
    for q, want, check in cases:
        t0 = time.perf_counter()
        r = await client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": q}],
                                                 tools=[weather, calc], max_tokens=1024, temperature=0,
                                                 extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        m = r.choices[0].message
        calls = [(c.function.name, json.loads(c.function.arguments or "{}")) for c in (m.tool_calls or [])]
        ok = (not calls) if want is None else bool(calls) and calls[0][0] == want and check(calls[0][1])
        out.append({"case": q, "calls": calls, "content": (m.content or "")[:200], "ok": ok, "s": time.perf_counter() - t0})
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "year": {"type": "integer"},
              "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "year", "tags"],
              "additionalProperties": False}
    r = await client.chat.completions.create(model=MODEL, max_tokens=512, temperature=0,
            messages=[{"role": "user", "content": "Describe the Eiffel Tower as JSON."}],
            response_format={"type": "json_schema", "json_schema": {"name": "landmark", "schema": schema}},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    try:
        d = json.loads(r.choices[0].message.content)
        ok = set(d) == {"name", "year", "tags"} and isinstance(d["year"], int) and isinstance(d["tags"], list)
    except Exception as e:
        d, ok = repr(e), False
    out.append({"case": "json_schema", "value": d, "ok": ok})
    return out


async def main():
    t0 = time.time()
    res = await {"gsm8k": gsm8k, "humaneval": humaneval, "mmlu_pro": mmlu_pro, "tools": tools}[BENCH]()
    ok = [r for r in res if r.get("ok")]; err = [r for r in res if "error" in r]
    toks = sum(r.get("completion_tokens", 0) for r in res)
    summary = {"bench": BENCH, "model_tag": os.environ.get("LLM_TAG", "vllm-merlin"), "effort": EFFORT, "sampling": SAMPLING, "concurrency": CONC, "max_tokens": MAXTOK or "default", "n": len(res), "correct": len(ok), "accuracy": len(ok) / len(res),
               "errors": len(err), "truncated": sum(1 for r in res if r.get("finish") == "length"),
               "completion_tokens": toks, "wall_s": time.time() - t0}
    if BENCH == "mmlu_pro":
        cats = {}
        for r in res:
            c = cats.setdefault(r.get("category"), [0, 0]); c[0] += bool(r.get("ok")); c[1] += 1
        summary["by_category"] = {k: f"{a}/{b}" for k, (a, b) in sorted(cats.items())}
    print(json.dumps(summary, indent=1), flush=True)
    tag = os.environ.get("LLM_TAG")
    save((f"{tag}_" if tag else "") + f"{BENCH}_{EFFORT}" + (f"_{MAXTOK // 1024}k" if MAXTOK else "") + ("_rec" if SAMPLING == "recommended" else "") + (f"_c{CONC}" if CONC != 8 else ""), {"summary": summary, "items": res})

asyncio.run(main())
