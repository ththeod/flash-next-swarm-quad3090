# Results: flash-next-swarm-quad3090 v1.0

All numbers from the reference host (see [recipe.md](recipe.md)), 2026-10-22/23, with the release configuration
(`config/swarm.env`): gpu-unlock loaded, 1400 MHz clock lock, caps 265/255/255/240 W, RAM tier 130 GiB. Ranges are
repeated runs. A same-day comparison against halt95's v2.5.2 and Merlin 3.0.0 releases will be added here.

## 40-agent swarm

`bench/swarm_bench.py`: 40 concurrent agent sessions, 1,200 s measured after warm-up, seed 1. Session shapes are
calibrated from aggregate statistics of a real coding-agent harness (prompt median 114K tokens, p90 218K; answer
median ~800 tokens; tool time median 3 s; subagents starting at ~7K tokens). The text is MMLU-Pro questions; requests
run with thinking off and fixed output lengths (`ignore_eos`), so every run generates the same amount and runs compare.
Real agents with thinking on generate more per turn.

| Metric | Value |
|---|---|
| Completed turns per minute | 26.63 (counting only turns after the first 200 s: 26.26); a fresh install from this repository: 26.48 |
| Mean turn time | 76.5 s |
| Mean queue time | 52.5 s |
| Generation | 528.6 tok/s |
| Prompt tokens per turn | ~98K: 96% reloaded from the RAM tier, ~4% computed (reused tokens all arrive through the RAM tier in this configuration; the GPU prefix cache reports 0 hits) |
| Preemptions | 189 (resumed from the RAM tier) |
| Errors / OOM / Xid | 0 / 0 / 0 |

The swarm is bound by KV capacity (95% of the pool in use): it queues, it does not fail.

## Generation (tok/s, total over all requests)

`bench/concurrency.py` (short prompts, 512-token answers, temperature 0.7) and `bench/prof_longctx.py`.

| Requests | tok/s | Runs |
|---|---|---|
| 1 | 343 | 300-343 |
| 8 | 782 | 739-782 |
| 16 | 1,144 | 1,122-1,144 |
| 8 at 80K context | 815 | 794-815 |
| 1, after a 10K / 50K / 260K prompt (512 tokens, exam-question text) | 114 / 116 / 132 | one run each |

Single-request speed depends on the text: DFlash 2 accepts more of predictable output (code, lists) than of prose.

## Prompt processing (tok/s)

| Prompt | tok/s | Runs |
|---|---|---|
| 10K | 7,570 | 7,171-7,570 |
| 20K | 7,793 | 7,760-7,793 |
| 50K | 8,490 | 8,227-8,490 |
| 260K | 7,747 | 7,700-7,747 |
| 8 × 80K at once | 72.3 s for 640K tokens | 72.3-73.0 s |
| 20K while 8 requests decode at 80K | 5,621 | 5,402-5,621 |

The first mixed batch on a fresh install compiles the sampler kernels once (a ~6 s stall; 2,097 tok/s in that probe);
later runs use the cache.

Prefill is power-bound with the unlock: see [power-and-clocks.md](power-and-clocks.md).

## Speculative decoding and decode steps

| Measurement | Value |
|---|---|
| Accepted tokens per step, DFlash 2 vs MTP (K=3) | 3.4-3.6 vs 3.1 |
| One request, greedy code prompt, 200 W caps: MTP → DFlash 2 → skinny GEMM → fused HC → custom AR + MoE kernels | 158 → 242 → 264 → 282 → 297 tok/s |
| Decode step at one request: stage 0 / hand-off / stage 1 | 8.6 / 0.12 / 11.6 ms |
| Balanced pipeline micro-batches, 16 requests | 789 → 1,078 tok/s (+34%) |
| All-reduce, 4.8 MB, custom across PCIe root complexes vs NCCL | 900 vs 240 µs |
| NCCL all-reduce, ~5 MB, LL vs Simple | 556 vs 233 µs |
| Mixed prefill+decode step, before vs after FlashInfer's PCIe all-reduce | 158.5 vs 145.2 ms |

## Capacity

| | Value |
|---|---|
| KV pool (VRAM) | 1,212,416 tokens (FP4 attention KV; `--kv-cache-memory 3900000000`) |
| Context per request | 262,144 tokens |
| RAM tier | 130 GiB (3,328 rows of 40 MiB) |
| KV pool history | halt95 v2.5.2 FP8 924,993 → DFlash 2 first try 306,900 → drafter scratch window 958,895 → FP4 1,228,800 → FP4 + INT8 activations 1,212,416 |

## Quality (`xhigh` reasoning, 64K output budget, 16 concurrent)

| Eval | Result |
|---|---|
| HumanEval | 161/164 (repeat runs on this build family: 159-162) |
| MMLU-Pro, 280-question set | 236/280 (BF16 activations: 239; FP8 KV: 232; run-to-run noise is about ±4) |
| Needles | 6/6, up to 260K tokens |
| Teacher-forced perplexity, INT8 vs BF16 activations | code +0.9% / +1.5%, prose +0.03% / -0.04%; top-1 agreement 0.970-0.979 |
| Garbage / runaway gate, NaN probe | clean on every run |

One HumanEval item looped on a repeated character until the 64K cap; loops like this occurred on other
configurations too.

## How we got here (our own history)

Each row was the best configuration at that point. Power settings changed along the way (the unlock arrived at the
INT8 row), so this is the order of the work, not a clean ablation; the ablation will come with the halt95 comparison.

| Step | Swarm turns/min |
|---|---|
| RAM tier (130 GiB, ARC) + DFlash 2 + sm_86 kernels, FP8 KV | 15.94 |
| RAM-tier resume at the turn's end, side-cache layout, FCFS for deferred lookups, adaptive Mamba retention, balanced PP decode, custom all-reduce within one root complex | 23.13 |
| FP4 attention KV | 23.34 |
| INT8 MoE activations + gpu-unlock | 24.24 |
| NCCL LL/Simple protocols | 25.73 |
| FlashInfer 0.7 PCIe all-reduce | 26.63 |
