import asyncio, uuid, pandas as pd
from common import chat
rows = pd.read_parquet("data/humaneval.parquet").head(8).to_dict("records")
def msg(r): return [{"role": "user", "content": "Complete this Python function. Return the full function in one "
                     "```python code block, no tests.\n\n```python\n" + r["prompt"] + "```"}]
async def one(label, r, **kw):
    x = await chat(msg(r), max_tokens=200, thinking=True, effort="xhigh", **kw)
    out = x["reasoning"] + x["text"]
    print(f"{label:28s} {r['task_id']:13s} cached {x['cached_tokens']:5d}/{x['prompt_tokens']:5d}  bang {out.count('!'):4d}/{len(out):4d}  {out[:60]!r}", flush=True)
async def main():
    for r in rows[1:3]:
        await one("A sampled, cached prefix", r, temperature=1.0, top_p=0.95)
        await one("B sampled, unique salt", r, temperature=1.0, top_p=0.95, cache_salt=uuid.uuid4().hex)
        await one("C greedy, cached prefix", r, temperature=0.0)
        await one("D greedy, unique salt", r, temperature=0.0, cache_salt=uuid.uuid4().hex)
asyncio.run(main())
