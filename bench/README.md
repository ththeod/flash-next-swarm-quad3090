# Benchmarks and checks

Client scripts for the OpenAI-compatible API; they run from this directory in any Python 3.10+ with
`pip install openai pandas pyarrow numpy`. Server: `LLM_BASE_URL` (default `http://127.0.0.1:8000/v1`), `LLM_MODEL` (default `flash-next`), `LLM_API_KEY` when the
server has one. `bench/check.sh` runs the quick checks and sets these from `config/local.env`.

**`quality.py humaneval` runs model-written code on this machine, unsandboxed.** Run it in a throwaway container or VM
if that matters to you.
Results go to `bench/results/` (`LLM_TAG` names a run).

| Script | What it measures |
|---|---|
| `swarm_bench.py` | `N_AGENTS` (40) agent sessions with 100K-token contexts for `DURATION` s (1200) after warm-up, `SEED` (1); turns/min, queue, TTFT, prompt tokens by source (computed / GPU cache / RAM tier) |
| `concurrency.py 1 8 16` | total generation tok/s at each concurrency (short prompts, 512-token answers) |
| `prof_longctx.py` | 8 requests decoding at 80K context (`N`, `CTX`); `MIX=20000` adds a 20K prompt arriving next to them |
| `longctx.py` | prefill tok/s and needle retrieval from 10K to 260K tokens |
| `decode_ctx.py` | decode speed with 32K of context at 1/4/8 requests (`DOC_TREE` = the engine source, used as text) |
| `quality.py <gsm8k\|humaneval\|mmlu_pro> [n] [effort] [max_tokens]` | accuracy; `CONC=16 python quality.py humaneval 0 xhigh 65536` and `... mmlu_pro 280 xhigh 65536` reproduce our runs |
| `ab_compare.py` | paired comparison of two quality runs |
| `kv_ppl.py` | teacher-forced log-likelihood of long documents through the KV cache (needs prompt logprobs: start the server with `VLLM_EXP_REJECT_PROMPT_LOGPROBS=0` and a smaller `KV_CACHE_MEMORY`) |
| `runaway_probe.py` | garbage / runaway-generation gate (`N=8` for the quick gate) |
| `nan_poke.py` | non-finite logits probe |
| `offload_validate.py` | RAM tier: a reloaded turn matches the original |
