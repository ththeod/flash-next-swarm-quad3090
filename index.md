---
layout: default
title: "Four RTX 3090s, one 125B MoE, forty coding agents"
---

# Four RTX 3090s, one 125B MoE, forty coding agents

*A fork of halt95's Flash-Next engine that serves coding-agent swarms on consumer GPUs without giving up quality. ththeod, October 2026.
Code, patches and benchmarks: [github.com/ththeod/flash-next-swarm-quad3090](https://github.com/ththeod/flash-next-swarm-quad3090)*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.svg">
  <img src="assets/hero-light.svg" alt="1,144 tok/s at 16 requests; 8,490 tok/s prefill; 1.21M-token KV cache plus a RAM tier; 26.6 turns per minute with 40 agents" width="100%">
</picture>

A coding-agent harness can fan out into dozens of subagents, each carrying a conversation of 100K tokens or more.
halt95's [Flash-Next engine](https://github.com/halt95/merlin-engine) is the reason a 125B-parameter model like
Qwen3.8-Flash-Next runs on four RTX 3090s at all, but its GPU cache fits fewer than ten conversations that long. In a
swarm of forty, almost every turn starts by recomputing its prompt from scratch.

flash-next-swarm-quad3090 is a fork of halt95's v2.5.2 with 40 changes for that workload. It serves Swift 1.5, a
fine-tune of Qwen3.8-Flash-Next, and adds:

- a RAM tier that keeps conversations warm;
- the DFlash 2 drafter;
- an FP4 KV cache that holds 1.2M tokens;
- shorter decode steps;
- faster communication between the GPUs.

On the same hardware and the same day, a 40-agent benchmark ran at 26.5 turns a minute on the fork and 4.2 on
halt95's latest release, Merlin 3.0.0. Both builds scored the same on HumanEval and MMLU-Pro.

The numbers in this post were measured with the optional third-party BF16 unlock and a 1400 MHz clock lock. The
[results page](https://github.com/ththeod/flash-next-swarm-quad3090/blob/swarm/docs/swarm/results.md) lists the
conditions, and numbers for stock cards will follow. Your numbers will depend on your board, PCIe layout and RAM.

## The problem

In an agent conversation, each turn resends everything so far, plus a little more: the latest tool output and the
next instruction. If the server still has the earlier part cached, it only processes the new part. halt95's engine
does this well. With 8 agents, 93% of follow-up tokens come from its cache.

A swarm is bigger than that. Forty subagents with 100K-token prompts need several million tokens of cache between
them, and four 3090s hold about 925K. Whatever doesn't fit is evicted and later recomputed, at a few thousand tokens
per second, while the agent waits.

So the fork aims to keep many long conversations warm, decode many requests at once, and let a fresh prompt in
without stalling everyone else. And every answer has to be as good as before.

## How we measured

Two benchmarks, both in `bench/`, drove the work.

The **40-agent swarm** (`swarm_bench.py`) runs forty simulated agents for 20 minutes after a warm-up. Their prompt
sizes, answer lengths, subagent fan-out and tool time follow aggregate statistics from a real coding-agent harness.
The text is MMLU-Pro questions, and answer lengths are fixed, so runs compare. It reports completed turns per minute,
queue time, time to first token, and where each prompt token came from: recomputed, the GPU cache or the RAM tier.

The **long-context probe** (`prof_longctx.py`) has eight requests decode at 80K context while a new 20K prompt
arrives. That is the moment a swarm stalls.

Every change also had to pass quality gates:

- a detector for garbage and runaway output;
- a NaN probe;
- needle-in-a-haystack tests from 10K to 260K tokens;
- HumanEval and MMLU-Pro at `xhigh` reasoning with a 64K-token budget;
- perplexity checks for precision changes;
- a check that a conversation resumed from RAM produces exactly the same tokens as one resumed from the GPU cache.

A change that failed was dropped, however fast it was.

Each change was tested from a clean boot against a baseline measured the same day. Changes that interacted were also
tested alone and together.

To decide what to try, we profiled. The four GPUs run as two pipeline stages of two GPUs each. Small tools on top of
torch-profiler traces split each decode step into three parts:

- each stage's compute;
- the hand-off between stages;
- the time spent waiting on all-reduces, the step where GPUs combine their partial results.

Kernels that looked slow got their own microbenchmarks. One tip if you tune kernels on 3090s: leave idle gaps between
timing runs. Back to back under a power cap, the cards throttle so hard that the timings pick the wrong plans.

## What the fork changes

Each section covers one area. The patches are in `release/swarm-v1.0/patches/`, and each commit message includes its
own measurement.

### 1. Agent conversations stay in RAM

vLLM can copy KV-cache blocks to system RAM and load them back later. Flash-Next makes that harder, because it is a
hybrid. Alongside attention, it has DeltaNet layers, which don't keep a cache per token. Instead, they keep one
running state that sums up everything so far, and that state only exists where it was saved. Resuming a conversation
from RAM needs two things: the attention blocks, and a saved state at exactly the token where the next turn starts.
It also has to keep working with halt95's RecoverSSM, which rebuilds that state after speculative decoding.

The fork makes the RAM tier work for this model:

- **The two pipeline stages get a uniform layout in the shared RAM region.** The stages hold different layers, so
  they computed different block sizes, overwrote each other's slots, and reloaded conversations came back corrupted.
- **A turn resumes from the state saved at the end of the previous turn.** Lookups used to stop one or two blocks
  earlier, so every turn recomputed about 15K tokens. Now it is about 4K.
- **Per-layer state is stored once instead of twice,** and eviction uses ARC instead of LRU. A burst of short
  subagents no longer pushes out the long-lived parent conversations.
- **A request waiting for its RAM lookup keeps its place in line.** It used to jump ahead of older requests and, if
  it couldn't get memory, block everything behind it. That stalled preempted requests for minutes in the middle of an
  answer.
- **A fence stops a block from being reused while its copy to RAM is still running.**
- **Extra saved states are dropped while memory is short.** Each one costs a running request about one block of GPU
  memory. They come back after a quiet spell.

In the swarm benchmark, 96% of prompt tokens come back from RAM instead of being recomputed. The tier is sized from
your RAM (see [Run it yourself](#run-it-yourself)).

*Patches 0099, 0121–0130.*

### 2. A better drafter: DFlash 2

Speculative decoding uses a drafter to guess the next few tokens, and the big model checks all the guesses in one
step. halt95's build uses the model's own MTP head, which guesses three tokens at a time. incoai's
[DFlash 2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) is a separate drafter. It predicts a block of
tokens from the hidden states of five layers of the big model.

The drafter runs on the second pipeline stage, so the fork passes those hidden states over from the first stage. It
also fixes what broke:

- **A torch.compile deadlock** with seven drafts and the RAM tier.
- **NaN drafts after RAM-tier hits.** The asynchronous loads skipped zeroing the drafter's window.
- **Out-of-memory errors on the last stage.** Prompt log-probability requests caused them, so the server now refuses
  those. The drafter's embedding table also moved to pinned host memory, which frees about 636 MB per GPU on that
  stage.

DFlash 2 gets more of its guesses right: 3.4–3.6 accepted tokens per step, against 3.1 for MTP. On a fixed code
prompt, that made single-request decode about 50% faster. The fork uses 7 drafts for a single request and fewer as
load grows, down to 2 at 9–16 requests, because wrong guesses cost more when the GPUs are busy.

*Patches 0100–0103, 0108–0113.*

### 3. A bigger KV cache: FP4 attention KV

DFlash 2's sliding-window layers would cut the KV cache to a third, because they make every cache block much bigger
on the last stage. The fork gives the drafter its own small scratch cache instead, so the pool stays a little above
halt95's.

Then the fork stores attention KV in FP4, with one FP8 scale per 16 values. That holds 26–28% more tokens than FP8.
The first version was slow, and the profile showed two reasons:

- **Unpacking the values made prefill 4.5× slower than FP8.** Each FP4 value went through an `exp2` in FP32. Building
  the value bit by bit in BF16 is exact for all 16 FP4 codes and removes that cost.
- **The sparse attention kernel was still 5.7× slower.** The tuned prefill setup used a single warp, so FP4's tiles
  ran out of registers. A minimum warp count and a dedicated split-K kernel bring FP4 up to FP8 speed.

With the INT8 activations from section 6, the pool holds **1,212,416 tokens**.

*Patches 0104–0107, 0133–0135.*

### 4. Shorter decode steps

The profile of a decode step at one request showed that the hand-off between stages costs almost nothing. The second
stage takes the larger share, because it also runs the output head, the sampler and the drafter. Half of the compute
was dense BF16 matrix multiplies on only 8 rows: one token plus seven drafts. The GPU's tensor-core tiles are 16 rows
tall, so they ran mostly empty. The fork changes three things:

- **A Triton matrix multiply for that thin shape.** Its plans are tuned per weight shape, and it adds a fused kernel
  for the hyper-connection layers. It makes one request 9–16% faster.
- **The custom all-reduce is back on.** An inherited setting had turned it off. It is worth about 5% at one request.
- **Small-batch MoE kernels and a cheaper end of each step.** The end of each step now uses a sparse logits gather, an
  INT8 draft head and a sampler that takes the top-k first. These save about 0.55 ms per step.

Together, these make single-request decode about 23% faster.

With 8–16 requests, the profile showed a different problem. The two stages take turns on two micro-batches, and each
request stayed in whichever micro-batch it joined first. With 8 requests, the split could be 6/2 or 7/1 instead of
4/4, and the bigger micro-batch set the pace of every step. Balancing them makes 16 requests **34% faster** and 8
requests 16% faster.

*Patches 0114–0118, 0131.*

### 5. A new prompt no longer stalls everyone

The long-context probe found the worst moment for a swarm. Before these changes, eight requests were decoding at 80K
context when a new 20K prompt arrived. The eight dropped to 27 tok/s between them for 13.5 seconds, and the new
prompt prefilled at under a third of its normal speed. The profile pointed at communication between the GPUs:

- **The custom all-reduce is fast only when the GPUs share a PCIe root complex,** the part of the CPU that their slots
  connect to. Many boards spread four GPUs over more than one root complex, and across them, a prefill-sized
  all-reduce took almost four times as long as NCCL's. The fork checks where each GPU sits at startup. It uses the
  custom all-reduce only within one root complex and only for messages under 128 KB.
- **NCCL's LL protocol is slow for prefill-sized messages,** more than twice as slow as its Simple protocol. The fork
  allows both (`NCCL_PROTO=LL,Simple`), so small decode messages still use LL.
- **FlashInfer 0.7 has a PCIe all-reduce for small messages,** which halt95 also turned on in Merlin 3.0.0. It makes
  a mixed prefill-and-decode step 8% shorter and generation at 8–16 requests 9–11% faster.

Now the same new prompt is done in 3.6 seconds instead of 13.5, at about 5,600 tok/s. The eight running requests
keep decoding at 100–115 tok/s through it.

*Patch 0132, plus NCCL and FlashInfer settings.*

### 6. INT8, which GeForce runs at full speed

GeForce Ampere cards run BF16 math with FP32 accumulation at half the rate of the workstation cards, but INT8 at the
full rate. vLLM's Marlin kernels can quantize the MoE activations to INT8 per token. On this checkpoint, though, that
path produced garbage from the very first token, because of two bugs:

- the compressed-tensors MoE method never passed the INT8 global activation scales to the kernel;
- the kernel read the normalized group scales as unsigned, but half of this checkpoint's scales are negative.

Other authors have fixes for the same bugs waiting upstream (vLLM PRs #61405, #48926 and #59895). With both bugs
fixed, INT8 activations make prefill 11–12% faster on stock cards. The cost is a small rise in perplexity on code,
covered in the quality section.

*Patches 0136, 0137.*

### 7. Optional: the BF16 unlock

The third-party [gpu-unlock](https://github.com/Zagitta/gpu-unlock) kernel module removes GeForce's half-rate BF16
limit. It is not part of this project, and you use it at your own risk. It adds another 8–9% to prefill and makes the
cards power-bound.

Unlocked cards draw short spikes at the start of each burst of work. These spikes can trip a power supply even when
the average draw is well within its rating. Power caps react too slowly to stop them, but a clock lock contains them.

On stock cards, use halt95's 220 W per card. With the unlock, we recommend 240 W per card and a 1400 MHz lock.
`run-swarm.sh` refuses to start on unlocked cards without the lock. The
[power guide](https://github.com/ththeod/flash-next-swarm-quad3090/blob/swarm/docs/swarm/power-and-clocks.md) has the
details.

## Keeping quality

Speed changes are easy to make and easy to get wrong, so every change went through the gates above. Here are the
release configuration's results at `xhigh` with a 64K-token budget. HumanEval and MMLU-Pro also show halt95's Merlin
3.0.0, run on the same checkpoint on the same day:

| Check | This fork | Merlin 3.0.0 |
|---|---|---|
| HumanEval | 161/164 | 161/164 |
| MMLU-Pro, 280 questions | 236/280 | 237/280 |
| Needles, 10K to 260K tokens | all found | not run |
| RAM-tier resume vs GPU-cache resume, greedy | bit-exact (4/4) | no RAM tier |
| Garbage / runaway gate, NaN probe | clean on every run | not run |

Repeat runs scored 159–162 on HumanEval. MMLU-Pro moves by about ±4 between runs: BF16 activations scored 239, FP8 KV
232. So the differences in the table are noise.

The one cost we could measure is from INT8 activations. Perplexity on code is 0.9–1.5% higher than with BF16, prose
is unchanged, and the evals show no difference. To turn INT8 off, leave `VLLM_MARLIN_INPUT_DTYPE` empty in
`config/swarm.env`.

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

Generation figures are for short prompts with thinking off, added up over all requests.

The chart shows how much each group of changes added to the swarm. It starts from the first working RAM tier. Power
settings changed between steps, so read it as a rough guide, not a clean ablation. The next section compares against
halt95's release, which has no RAM tier.

## Against halt95's latest: Merlin 3.0.0

We ran halt95's current release next to the fork on the same hardware, on the same day, with the same benchmarks.
Merlin used its own launcher for this checkpoint: MTP with 3 drafts, FP8 KV, at most 8 requests at once and no RAM
tier. The fork's figures in this section come from the same day's run, so they differ from the results table above.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/vs-merlin-swarm-dark.svg">
  <img src="assets/vs-merlin-swarm-light.svg" alt="40-agent swarm, turns per minute: flash-next-swarm-quad3090 26.5, Merlin 3.0.0 4.2" width="100%">
</picture>

The swarm is where the two designs differ most. Merlin's 925K-token cache holds fewer than ten of the forty
conversations. It recomputes 99% of every prompt, and requests wait 425 seconds on average for a slot. With the RAM
tier, the fork recomputes 4% and requests wait 52 seconds. That's 26.5 turns per minute against 4.2.

To be fair to Merlin, this is a workload it doesn't claim to serve. Its own bench card measures eight agents with
16K-token conversations.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/vs-merlin-dark.svg">
  <img src="assets/vs-merlin-light.svg" alt="Ours relative to Merlin 3.0.0: 16 requests 2.02x, 1 request short prompt 1.92x, 8 requests 1.30x, 8 at 80K 1.22x, 1 after 260K 1.14x, 8 agents 1.11x, prefill 50K 1.11x, prefill 260K 1.10x, 1 request thinking on at 4K 0.93x" width="100%">
</picture>

On the smaller tests, the fork is ahead by 10–30% on prefill, long-context decode and 8 requests. At 16 requests the
gap is about 2×, because Merlin serves at most 8 requests at a time and the rest wait. A single short request is 1.9×
faster too. That comes mostly from the drafter and the shorter decode steps (sections 2 and 4).

Merlin wins one test, by 7%: a single request with thinking on and a 4K prompt. Thinking text is harder to guess than
code, and the fork spends 7 drafts on a single request. Most likely, Merlin's 3-draft MTP wastes less work there.

halt95 doesn't publish the bench card's drivers, so we re-implemented its tests. Merlin reproduces the published
number in our runs: 223.7 tok/s with 2 agents, against 223.9 on the card. The fork is 11–56% ahead with 2–8 agents.
Quality is the same: HumanEval 161/164 for both, and MMLU-Pro 236 against 237.

The full table and its caveats are in the
[README](https://github.com/ththeod/flash-next-swarm-quad3090#compared-with-merlin-300). Both builds ran with the
unlock here. halt95 publishes Merlin's numbers on stock cards, and we'll add those for both builds.

## What did not work

- **Bigger prefill chunks** (2,048 and 4,096 tokens) ran the last pipeline stage out of memory.
- **Keeping the DeltaNet state in BF16** instead of FP32 gained no capacity, because the state page is padded back to
  the same size.
- **Three kernel tweaks gave no measurable gain:** 8 warps for the FP4 attention kernel, a smaller Marlin MoE block
  size, and a different way of sorting tokens by expert. Savings of about 2 µs per layer disappear in the noise of an
  end-to-end test.
- **Moving a layer from one pipeline stage to the other** would cost about 20% of the KV cache.

## Run it yourself

`./setup.sh` checks your machine and shows every download before it starts. It then builds halt95's v2.5.2 with the
fork's 40 commits on top, verified by commit and tree hash, downloads and verifies the models, and sizes the RAM tier.
Afterwards, `bench/check.sh` runs the quick checks. Every benchmark in this post is in `bench/`, so you can compare
your numbers with ours.

You need four RTX 3090s and at least 96 GB of RAM. The model's n-gram embedding table takes 47.7 GiB and has to stay
in RAM. Together with the server, that comes to about 84 GiB, and whatever is left becomes the RAM tier. By default
the tier is capped at 130 GiB, the largest size we tested, and `--ram-tier` overrides the cap. The
[RAM sizing guide](https://github.com/ththeod/flash-next-swarm-quad3090/blob/swarm/docs/swarm/ram-sizing.md) has
sizes for common RAM amounts.

## What's next

- Numbers for stock cards for both builds, measured the way halt95 publishes Merlin's.
- Fewer drafts for short requests with thinking on, the one test where Merlin is ahead.

If you run the fork, please [open an issue](https://github.com/ththeod/flash-next-swarm-quad3090/issues) with your
board, RAM and numbers. We especially want to hear if they disagree with ours.

## Thanks

This is halt95's engine with our changes on top, and it carries the work that halt95 credits:

- Intel, for the AutoRound experts;
- RadixArk, for the FP8 n-gram table;
- DominikBucko, for MTP INT4 packing;
- the peakcrosser7 Flash-Next branch behind vLLM PRs #53896 and #53899;
- CoolShades, for the v2.5.2 idle-power fix;
- aikitoria, for P2P on consumer Ampere.

Thanks also to:

- Qwen;
- UkisAI, for Swift 1.5;
- incoai, for DFlash 2;
- Juntian Liu, for vLLM #57632, which we backported;
- ntovas, lazypool and plawanrath, for the Marlin INT8 fixes upstream;
- Zagitta, for gpu-unlock;
- the vLLM and FlashInfer teams.

Built with Claude Code.
