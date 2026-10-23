"""Shared helpers for the llm-eval suite: one streaming request with timing, and a bounded-concurrency runner."""
import asyncio, json, os, time
from openai import AsyncOpenAI

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("LLM_MODEL", "flash-next")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

client = AsyncOpenAI(base_url=BASE_URL, api_key=os.environ.get("LLM_API_KEY", "none"), timeout=3600, max_retries=0)


# SAMPLING=recommended uses Qwen's model-card settings instead of the per-call temperature (default 0.0):
# thinking: temperature 1.0, top_p 0.95, top_k 20; non-thinking: 0.7 / 0.80 / 20 with presence_penalty 1.5.
SAMPLING = os.environ.get("SAMPLING", "greedy")
RECOMMENDED = {True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
               False: {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "presence_penalty": 1.5}}
# SAMPLING=glm: GLM-5.3-Flash's generation_config (temperature 1.0, top_p 0.95, no top_k). GLM always thinks;
# its efforts are low / high / max (anything else renders as max).
if SAMPLING == "glm":
    RECOMMENDED = {t: {"temperature": 1.0, "top_p": 0.95, "top_k": 0} for t in (True, False)}
# SAMPLING=deepseek: DeepSeek-V4.1-Flash's generation_config (temperature 1.0, top_p 1.0, no top_k).
if SAMPLING == "deepseek":
    RECOMMENDED = {t: {"temperature": 1.0, "top_p": 1.0, "top_k": 0} for t in (True, False)}


async def chat(messages, max_tokens=1024, thinking=False, effort=None, temperature=0.0, cache_salt=None, **extra):
    """Stream one chat completion. Returns text, reasoning, token counts and timings.
    cache_salt: a unique value keeps the request from reusing (or seeding) any cached prefix."""
    kwargs = {"enable_thinking": thinking}
    if thinking and effort:
        kwargs["reasoning_effort"] = effort
    body = {"chat_template_kwargs": kwargs}
    if cache_salt:
        body["cache_salt"] = cache_salt
    sampling = {"temperature": temperature}
    if SAMPLING in ("recommended", "glm", "deepseek"):
        rec = RECOMMENDED[bool(thinking)]
        sampling = {k: v for k, v in rec.items() if k in ("temperature", "top_p", "presence_penalty")}
        body["top_k"] = rec["top_k"]
    t0 = time.perf_counter(); first = None; text = []; reasoning = []; usage = None; finish = None
    stream = await client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=max_tokens, stream=True, **sampling,
        stream_options={"include_usage": True}, extra_body=body, **extra)
    async for ev in stream:
        if ev.usage:
            usage = ev.usage
        if not ev.choices:
            continue
        d = ev.choices[0].delta
        r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
        if (d.content or r) and first is None:
            first = time.perf_counter()
        if d.content:
            text.append(d.content)
        if r:
            reasoning.append(r)
        if ev.choices[0].finish_reason:
            finish = ev.choices[0].finish_reason
    end = time.perf_counter()
    first = first or end
    out = usage.completion_tokens if usage else 0
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    return {"text": "".join(text), "reasoning": "".join(reasoning), "prompt_tokens": usage.prompt_tokens if usage else 0,
            "cached_tokens": (getattr(details, "cached_tokens", None) or 0) if details else 0,
            "completion_tokens": out, "ttft": first - t0, "total": end - t0, "finish": finish,
            "decode_tps": (out - 1) / (end - first) if out > 1 and end > first else 0.0}


async def run_all(items, fn, concurrency):
    """Run fn(item) for every item with at most `concurrency` in flight; keeps input order."""
    sem = asyncio.Semaphore(concurrency)
    async def one(i, it):
        async with sem:
            try:
                return i, await fn(it)
            except Exception as e:  # keep going; record the failure
                return i, {"error": repr(e)}
    res = await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    return [r for _, r in sorted(res, key=lambda x: x[0])]


def save(name, data):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
        f.flush(); os.fsync(f.fileno())
    return path
