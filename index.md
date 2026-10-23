---
layout: default
title: "Four RTX 3090s, one 125B MoE, forty coding agents"
---

# Four RTX 3090s, one 125B MoE, forty coding agents

*How we made Qwen3.8-Flash-Next serve an agent swarm on consumer GPUs without giving up quality. ththeod, October 2026.
Code, patches and benchmarks: [github.com/ththeod/flash-next-swarm-quad3090](https://github.com/ththeod/flash-next-swarm-quad3090)*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.svg">
  <img src="assets/hero-light.svg" alt="1,144 tok/s at 16 requests; 8,490 tok/s prefill; 1.21M-token KV cache plus a RAM tier; 26.6 turns per minute with 40 agents" width="100%">
</picture>

**The short version.** Swift 1.5, a post-train of Qwen3.8-Flash-Next (a 125B-parameter MoE), on four RTX 3090s:
40 concurrent coding agents with ~100K-token contexts complete 26.6 turns per minute, with 96% of their prompt tokens
reloaded from a RAM tier instead of recomputed. Up to 16 requests decode at once (1,144 tok/s in total), prefill runs
at 8,490 tok/s, and the KV cache holds 1.21M tokens in VRAM. Quality holds: HumanEval 161/164 and MMLU-Pro 236/280 at
`xhigh` reasoning, all needles found up to 260K tokens. These numbers come from one machine with the optional GeForce
BF16 unlock and a 1400 MHz clock lock; the conditions are in the [results page](https://github.com/ththeod/flash-next-swarm-quad3090/blob/swarm-v1.0/docs/swarm/results.md).

This is a fork of halt95's [Flash-Next engine](https://github.com/halt95/merlin-engine) (v2.5.2), the reason a model
this size runs on 3090s at all. This post covers the 40 commits we added, how we found what to change, and how we
checked that nothing broke.

## The goal

halt95's release is already built for agents: follow-up turns resume from a prefix cache kept on a 64-token grid, and
with 8 agents it serves 93% of follow-up tokens from cache. Our workload is bigger. A coding-agent harness fans out into
dozens of subagents, each carrying a median prompt of about 114K tokens, and the conversations together far exceed one
924,993-token VRAM pool. Whatever does not fit is evicted and later recomputed at a few thousand tokens per second.

So the goal was a server that keeps many long conversations warm, decodes many requests at once, and lets a fresh
prompt in without stalling everyone else, while answering exactly as well as before.

## How we measured

**Two benchmarks shaped every decision.**

- **A 40-agent swarm** (`bench/swarm_bench.py`). Forty simulated agents run for 20 minutes after a warm-up. Their shape
  (prompt sizes, output lengths, subagent fan-out, tool time) comes from aggregate statistics of a real coding-agent
  harness; the text is MMLU-Pro questions, and output lengths are fixed so runs are comparable. The metric is
  completed turns per minute, with queue time, time to first token and where each prompt token came from (computed,
  GPU cache or RAM tier).
- **A long-context probe** (`bench/prof_longctx.py`). Eight requests decode at 80K context while a 20K prompt
  arrives, which is exactly the moment a swarm stalls.

**Quality gates ran on every candidate.** A garbage and runaway detector, a NaN probe, needles from 10K to 260K,
HumanEval and MMLU-Pro at `xhigh` with a 64K budget, teacher-forced perplexity on long documents for precision changes,
and bit-exact tests that a conversation resumed from the RAM tier produces the same tokens as one resumed from the GPU
cache.
A change that failed a gate was dropped, however fast it was.

**Every change ran in an isolated test window.** We ran 64 of them over four days: a clean boot of the candidate on
all four GPUs, a fixed list of probes and gates, the same day as its baseline, and production restored afterwards.
Where two changes interacted, we ran them alone and together.

**Profiling told us where to look.** We captured torch-profiler traces on both pipeline stages and wrote small tools on
top of them: a step anatomy that splits one decode step into each stage's compute, the hand-off between stages and the
all-reduce waits, and per-kernel tables with NCCL waits removed. Suspicious kernels got their own microbenchmarks:
FP4 attention, Marlin MoE block sizes, all-reduce by message size and PCIe root complex, and a GEMM tuner that
replays shapes with idle gaps (back-to-back replays under a power cap throttled the SM clock to ~330 MHz and gave
misleading plans).

## What the profiles showed, and what we changed

The patch numbers below are in `release/swarm-v1.0/patches/`; each commit message carries its own measurement.

### 1. Keep agent conversations in RAM (patches 0099, 0121-0130)

vLLM can offload KV blocks to CPU memory, but Flash-Next is a hybrid: its DeltaNet layers carry a recurrent state
that must match the exact token where the next turn resumes, and halt95's RecoverSSM rebuilds that state after
speculative decoding. We made the two work together:

- **A uniform offload stride under pipeline parallelism (0099).** The two pipeline stages hold different layers and
  derived different bytes-per-block, so their slots in the shared region overlapped and reloads came back corrupted.
- **Resume at the previous turn's end (0122, 0124).** Lookups stopped at the newest aligned state, one or two blocks
  before the prompt end, so each agent turn recomputed ~15K tokens. Anchoring at the stored tail cut that to ~4K.
- **Store fences (0127)** for blocks that are re-allocated while a store is still in flight.
- **A side-cache layout (0128)** that stores each per-layer cache once instead of twice, after an 80 GiB tier filled
  up under a subagent swarm. With ARC eviction instead of LRU, a burst of short subagents no longer flushes the
  long-lived parents. The tier is now 130 GiB.
- **Scheduling under pressure (0129, 0130).** A deferred RAM lookup kept losing its place in the queue: one build had 17
  turns stalled 50-160 s mid-answer. It now keeps its place (0 stalls), and Mamba-state checkpoints are kept only while
  requests are queued.

Effect: the 40-agent swarm went from 15.9 to 23.1 turns per minute with these fixes and the scheduling and
all-reduce changes below.

### 2. A bigger KV pool: FP4 attention KV (patches 0104-0107, 0133-0135)

Switching speculative decoding to DFlash 2 first *shrank* the pool to 306,900 tokens: the drafter's sliding-window
layers made every block cost 40 MiB on the last stage. A scratch window cache for the drafter (0104) brought it back
to 958,895 tokens, slightly above halt95's 924,993.

FP4 attention KV with per-16 FP8 scales (0106, 0107) holds 26-28% more tokens than FP8, but the first version was slow in two ways the profile
made obvious. Decoding E2M1 through `exp2` in FP32 ran prefill 4.5× slower than FP8; building the value bit by bit in
BF16 (0133) is exact for all 16 codes and removed that. Then the sparse attention kernel still ran 5.7× slower: the
tuned prefill configuration used one warp, and FP4's int32 tiles spilled registers. A warp floor (0134) and a dedicated
split-K kernel with contiguous loads (0135) brought FP4 to FP8 speed. With INT8 activations, the pool is now
**1,212,416 tokens**.

### 3. A better drafter: DFlash 2 (patches 0100-0103, 0108-0113)

incoai's [DFlash 2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) predicts a block of tokens from hidden
states of five target layers. Wiring it into a two-stage pipeline meant relaying those hidden states from the first
stage (0100) and fixing what broke:

- a torch.compile recompile that deadlocked with seven drafts and the RAM tier (0108);
- NaN drafts after RAM-tier hits, because async loads skipped zeroing the drafter's window (0110, 0111);
- a prompt-logprobs request that ran the last stage out of memory (0112);
- the drafter's own embedding, moved to pinned host memory to free ~636 MB per GPU on the tight stage (0113).

Acceptance rose from 3.1 tokens per step with MTP to 3.4-3.6. With 7 drafts for one request, 3 for 2-8 and 2 for
9-16, a single request on our code prompt went from 158 tok/s (MTP) to 242 in back-to-back windows.

### 4. Shorter decode steps (patches 0114-0118, 0131)

The step anatomy at one request: stage 0 computes for 8.6 ms, the hand-off takes 0.12 ms, stage 1 (23 layers, the LM
head, the sampler and the drafter) takes 11.6 ms. In those steps, dense BF16 GEMMs were 51% of compute: at 8 rows,
CUTLASS's 16×16 tensor-core tiles are mostly empty.

- **A skinny GEMM for sm_86 (0114, 0115)**, Triton, with a plan tuned per weight shape and a fused down/SiLU for the
  hyper-connection layers: one request +9-16%, 2-12 requests +6-18%.
- **The custom all-reduce, back on.** An inherited flag had disabled it: +4.5-4.8% at one request.
- **Small-batch MoE kernels (0116)** and a cheaper tail of the step (0118: sparse logits gather, an INT8 draft head, a
  top-k-first sampler), which saved ~0.55 ms per step.

On the code prompt, one request went 242 → 264 → 282 → 297 tok/s through these steps.

At 8-16 requests the trace showed a different problem: stage 1 was the bottleneck, and decode micro-batches alternated
6/2 or 7/1 between the stages. Balancing them (0131) gave **+34% at 16 requests** (789 → 1,078 tok/s) and +16% at 8.

### 5. Let a new prompt in without stopping everyone (patch 0132, NCCL and FlashInfer settings)

The long-context probe found the worst moment for a swarm. While one fresh 20K prompt prefilled next to eight 80K
decodes, decode fell to 27 tok/s for 13.5 seconds, and the prefill itself ran at 1.5K tok/s instead of 4.9K. The
profile pointed at communication:

- **The custom all-reduce is fast only within one PCIe root complex.** On this board, GPUs 0 and 1 share one, and the
  other two each have their own. Across root complexes a 4.8 MB all-reduce took 900 µs against NCCL's 240 µs. Keeping
  it within a root complex and below 128 KB (0132) made a 20K prefill 12% faster.
- **NCCL's LL protocol is slow for prefill-sized messages:** 556 µs versus 233 µs with Simple for ~5 MB. We allow both
  (`NCCL_PROTO=LL,Simple`), so small decode messages keep LL.
- **FlashInfer 0.7's PCIe IPC all-reduce for small messages,** switched on as halt95 does in Merlin 3.0.0: the mixed
  step went from 158.5 to 145.2 ms, and generation at 8-16 requests rose 9-11%.

A 20K prompt now prefills at 5,400-5,600 tok/s next to eight 80K decodes.

### 6. Use the tensor cores GeForce gives at full rate (patches 0136, 0137)

GeForce Ampere runs BF16 with FP32 accumulate at half the workstation rate, but INT8 at full rate. vLLM's Marlin
kernels can quantize MoE activations to INT8 per token; on this checkpoint that path produced random tokens from the
first token on. We found two bugs: the compressed-tensors MoE method never passed the INT8 global activation scales
(0136), and the kernel read normalized group scales as unsigned, while this checkpoint's scales are 50% negative
(0137). Fixes by other authors for the same bugs are waiting upstream (vLLM PRs #61405, #48926 and #59895). With them,
INT8 activations made prefill 11-12% faster on stock cards.

### 7. Hardware: the BF16 unlock and the PSU

The third-party [gpu-unlock](https://github.com/Zagitta/gpu-unlock) module lifts GeForce's half-rate BF16 limit
(it is not part of this project; use it at your own risk). With it, prefill gained another 8-9% and became
power-bound. Our 1500 W PSU tripped, taking the whole machine down, at a 1650 MHz clock lock and again at 1455 MHz:
the averaged power was the same (~1,016 W for the four GPUs) at every step, so the trip came from a spike at the start
of a burst. Power caps react too slowly to stop it; the clock lock contains it. 1400 MHz has run for days.

## Keeping quality

Speed changes are easy to make and easy to get wrong, so every one of them went through the gates above. The release
configuration, against the same evals at `xhigh` with a 64K budget:

| Check | Result |
|---|---|
| HumanEval | 161/164 (repeat runs on this build family: 159-162) |
| MMLU-Pro, 280 questions | 236/280; BF16 activations 239, FP8 KV 232 (run-to-run noise about ±4) |
| Needles, 10K to 260K | all found |
| INT8 vs BF16 activations, teacher-forced perplexity on 64K documents | code +0.9% / +1.5%, prose ±0.04% |
| RAM-tier resume vs GPU-cache resume, greedy | bit-exact (4/4) |
| Garbage / runaway gate, NaN probe | clean on every run |

INT8 activations cost a little on code perplexity and nothing measurable on the evals; one setting turns them off.

## Results

| Metric | Value |
|---|---|
| 40-agent swarm (~100K-token contexts) | 26.6 turns/min, 0 errors in 20 minutes |
| Generation, 1 / 8 / 16 requests | 343 / 782 / 1,144 tok/s |
| Generation, 8 requests at 80K context | 815 tok/s |
| Prefill, 10K / 50K / 260K prompt | 7,570 / 8,490 / 7,747 tok/s |
| KV cache | 1,212,416 tokens in VRAM, plus a 130 GiB RAM tier |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/swarm-steps-dark.svg">
  <img src="assets/swarm-steps-light.svg" alt="Bar chart: 40-agent swarm turns per minute after each step: 15.9, 23.1, 23.3, 24.2, 25.7, 26.6" width="100%">
</picture>

Generation figures are short prompts with thinking off, summed over all requests. Power settings changed along the way
(the unlock arrived at the INT8 step), so the chart shows the order of the work rather than a clean ablation. A
same-day comparison with halt95's current Merlin 3.0.0 on this machine follows in a separate post.

## What did not work

- Larger prefill chunks (2,048 and 4,096 tokens): out of memory on the last pipeline stage.
- BF16 instead of FP32 DeltaNet state: no capacity gain, because the state page is padded back to the same size.
- 8 warps for the FP4 attention kernel, a smaller Marlin MoE block size, token-space MoE alignment: no measurable gain
  (savings of ~2 µs per layer are below the noise floor of an end-to-end A/B).
- Moving a layer between pipeline stages: it would cost ~20% of the KV pool.

## Run it yourself

`./setup.sh` checks the machine, shows every download, builds halt95's v2.5.2 and our 40 commits on top (verified by
commit and tree hash), downloads and verifies the models, and sizes the RAM tier. The minimum is 96 GB of RAM, because
the n-gram embedding table alone is 47.7 GiB; what is left after the table and the server (about 84 GiB together)
becomes the RAM tier. We measured 130 GiB on 256 GB.
`bench/check.sh` runs the quick checks, and the same benchmarks are in `bench/`.

## Thanks

This is halt95's engine with our changes on top, and it inherits the work halt95 credits: Intel (AutoRound experts),
RadixArk (the FP8 n-gram table), DominikBucko (MTP INT4 packing), the peakcrosser7 Flash-Next branch behind vLLM PRs
#53896 and #53899, CoolShades (the v2.5.2 idle-power fix) and aikitoria (P2P on consumer Ampere). Thanks also to Qwen,
UkisAI (Swift 1.5), incoai (DFlash 2), Juntian Liu (vLLM #57632, backported here), ntovas, lazypool and plawanrath
(the Marlin INT8 fixes upstream), Zagitta (gpu-unlock), and the vLLM and FlashInfer teams. Built with Claude Code.
