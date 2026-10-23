---
layout: default
title: "Four RTX 3090s, one 125B MoE, forty coding agents"
---

# Four RTX 3090s, one 125B MoE, forty coding agents

*flash-next-swarm-quad3090: coding-agent swarms on consumer GPUs without giving up quality, built on halt95's Flash-Next engine. ththeod, October 2026.
Code, patches and benchmarks: [github.com/ththeod/flash-next-swarm-quad3090](https://github.com/ththeod/flash-next-swarm-quad3090)*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.svg">
  <img src="assets/hero-light.svg" alt="1,144 tok/s at 16 requests; 8,490 tok/s prefill; 1.21M-token KV cache plus a RAM tier; 26.6 turns per minute with 40 agents" width="100%">
</picture>

A coding-agent harness can fan out into dozens of subagents, each carrying a conversation of 100K tokens or more.
halt95's [Flash-Next engine](https://github.com/halt95/merlin-engine) is the reason a 125B-parameter model like
Qwen3.8-Flash-Next runs on four RTX 3090s at all, but its GPU cache fits fewer than ten conversations that long. In a
swarm of forty, almost every turn starts by recomputing its prompt from scratch.

flash-next-swarm-quad3090 adds 40 changes for that workload on top of halt95's v2.5.2. It serves Swift 1.5, a
fine-tune of Qwen3.8-Flash-Next, and adds:

- a RAM tier that keeps conversations warm;
- a better drafter, which guesses tokens ahead (DFlash 2);
- a 4-bit KV cache, the model's working memory, which holds 1.2M tokens;
- less wasted work per generated token;
- faster communication between the GPUs.

On the same hardware and the same day, a 40-agent benchmark ran at 26.5 turns a minute on flash-next-swarm-quad3090 and 4.2 on
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

So the goal is to keep many long conversations warm, generate for many requests at once, and let a fresh prompt in
without stalling everyone else. And every answer has to be as good as before.

## How we measured

Two benchmarks, both in `bench/`, drove the work.

The **40-agent swarm** (`swarm_bench.py`) runs forty simulated agents for 20 minutes after a warm-up. Their prompt
sizes, answer lengths, subagent fan-out and tool time follow aggregate statistics from a real coding-agent harness.
The text is MMLU-Pro questions, and answer lengths are fixed, so runs compare. It reports completed turns per minute,
queue time, time to first token, and where each prompt token came from: recomputed, the GPU cache or the RAM tier.

The **long-context probe** (`prof_longctx.py`) has eight requests generate at 80K context while a new 20K prompt
arrives. That is the moment a swarm stalls.

Every change also had to pass quality gates:

- a detector for garbage and runaway output;
- a check for numerical errors (NaN) in the output;
- needle-in-a-haystack tests from 10K to 260K tokens;
- HumanEval and MMLU-Pro at `xhigh` reasoning with a 64K-token budget;
- for changes to number precision, a perplexity check: how well the model predicts known text;
- a check that a conversation resumed from RAM produces exactly the same tokens as one resumed from the GPU cache.

A change that failed was dropped, however fast it was.

Each change was tested from a clean boot against a baseline measured the same day. Changes that interacted were also
tested alone and together.

To decide what to try, we profiled: we recorded where every millisecond of a generation step goes on every GPU,
and timed slow pieces of GPU code on their own. One tip if you tune GPU code on 3090s: leave idle gaps between timing
runs. Back to back under a power cap, the cards throttle so hard that the timings pick the wrong settings.

## What we changed

Each section says what was slow before, what changed, and why that helps. The patch numbers point
to the code in `release/swarm-v1.0/patches/`.

One thing helps with the rest: the model runs in two halves, each on a pair of GPUs. The halves work like an assembly
line, and within a pair the two GPUs split each layer's math between them.

### 1. Conversations stay in RAM

halt95's build keeps conversations only in GPU memory. When that fills up, the oldest are thrown away. When an agent
comes back, its whole prompt is recomputed, which takes over ten seconds for 100K tokens even at full speed.

vLLM can already park conversations in system RAM, but not for this model. Flash-Next mixes two kinds of layers.
Attention layers keep a note for every token, and those notes can be copied out and back in pieces. DeltaNet layers
keep one running summary of the whole conversation instead, and that summary only exists where it was saved. To
resume from RAM, you need a summary saved at exactly the token where the last turn ended.

The server now saves that summary at the end of every turn and resumes from there. Loading a conversation back from RAM is
much faster than recomputing it, so in the swarm benchmark 96% of prompt tokens now come from RAM. We also fixed
the bugs that showed up under real swarms:

- the two GPU pairs overwrote each other's data in RAM;
- some data was stored twice;
- a request waiting for its data could jump the queue and block everyone behind it.

Long-lived parent conversations are now protected from being pushed out by bursts of short subagents.

*Patches 0099, 0121–0130.*

### 2. Better guesses: the DFlash 2 drafter

When a GPU generates text one token at a time, it spends most of each step reading the model's weights for that one
token. Speculative decoding makes better use of that work. A small, fast drafter guesses several tokens ahead, and the
big model checks all the guesses in one step, keeping the ones it agrees with.

halt95's build uses the model's built-in drafter (MTP), which guesses 3 tokens at a time. We switched to incoai's
[DFlash 2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2). It is a separately trained drafter that sees
more of the big model's internal state and guesses up to 7 tokens. It is right more often: 3.4–3.6 tokens are kept
per step, against 3.1. So every step produces more text, and a single request on a code prompt decodes about 50%
faster.

With many requests the GPUs are already busy, so wrong guesses cost more. The server makes fewer guesses as load grows,
down to 2.

*Patches 0100–0103, 0108–0113.*

### 3. More room: a 4-bit KV cache

The KV cache is the model's working memory: notes on every token of every conversation it holds. Its size decides how
many tokens fit in VRAM at once. halt95's build stores those notes as 8-bit numbers. We store them as 4-bit
numbers, with a shared scale for every 16 values, so the same VRAM holds 26–28% more tokens.

The first 4-bit version made prompt processing several times slower, for two reasons:

- unpacking each 4-bit number used slow floating-point math;
- the attention code was tuned for 8-bit data.

The new version unpacks with simple bit operations, which give exactly the same values, and uses attention code written for
4-bit data. It now runs as fast as 8-bit, and the scores didn't drop (see [Keeping quality](#keeping-quality)).

DFlash 2 also gets its own small cache. Otherwise the drafter's memory needs would have cut the main cache
to a third. Altogether the cache holds 1.21M tokens, against 925K in halt95's build.

*Patches 0104–0107, 0133–0135.*

### 4. Less wasted work per step

Profiling one generation step showed that half the time went to matrix multiplications. The GPU's math units work on
blocks of 16 rows, but a single request only gives them 8 (one token plus seven guesses), so the blocks ran half empty.
We changed three things:

- new matrix code sized for these small batches;
- a fast method for adding up partial results between GPUs, switched back on after the inherited configuration had
  turned it off;
- less work in picking the next token.

A single request now decodes about 23% faster.

With several requests at once, a different problem showed up. The assembly line works on two groups of requests and
moves at the pace of the bigger group, and the groups were often uneven: 6 and 2 instead of 4 and 4. The server now
keeps them even, which makes 16 requests 34% faster.

*Patches 0114–0118, 0131.*

### 5. A new prompt no longer freezes everyone

When a long new prompt arrived while other requests were generating, they all froze for seconds. In the long-context
probe, eight requests generating at 80K context dropped to 27 tok/s combined for 13.5 seconds while a 20K prompt was
processed.

The cause was traffic between the GPUs. The two GPUs in a pair add up their partial results after every layer. The
fast method from section 4 is only fast when both GPUs connect to the same PCIe branch of the CPU. On many boards they
don't, and for the large transfers of prompt processing, the fast method took almost four times as long as NVIDIA's
standard library, NCCL. We changed three things:

- the server checks how the GPUs are connected at startup, and uses the fast method only for small transfers on a
  shared branch;
- NCCL may now switch to its faster mode for large transfers;
- small transfers use FlashInfer's newer method, as in halt95's Merlin 3.0.0.

With these changes and the faster prompt processing from sections 6 and 7, the freeze is down to 3.6 seconds. The
other requests keep generating at 100–115 tok/s through it.

*Patch 0132, plus NCCL and FlashInfer settings.*

### 6. 8-bit math, which GeForce runs at full speed

GeForce 3090s are deliberately limited. The 16-bit math this model normally uses (BF16) runs at half the speed it
does on the same chip in a workstation card, but 8-bit integer math (INT8) runs at full speed.

vLLM can convert the numbers going into the model's expert layers to INT8 on the fly. With this model, though, it
produced garbage from the first token, because of two bugs in how it handled the scale factors that keep 8-bit values
accurate. We fixed both. Fixes by other people for the same bugs are waiting in vLLM (PRs #61405, #48926 and
#59895).

Prompt processing is 11–12% faster on stock cards, at a small cost on code (see [Keeping quality](#keeping-quality)).

*Patches 0136, 0137.*

### 7. Optional: the BF16 unlock

The third-party [gpu-unlock](https://github.com/Zagitta/gpu-unlock) kernel module removes that half-speed BF16 limit.
It is not part of this project, and you use it at your own risk. It makes prompt processing another 8–9% faster, and
the cards then run at their power limits.

Unlocked cards draw short power spikes at the start of each burst of work. These spikes can trip a power supply even
when the average draw is well within its rating. Power caps react too slowly to catch them, but a clock lock keeps
them in check.

On stock cards, use halt95's 220 W per card. With the unlock, we recommend 240 W per card and a 1400 MHz lock.
`run-swarm.sh` refuses to start on unlocked cards without the lock. The
[power guide](https://github.com/ththeod/flash-next-swarm-quad3090/blob/swarm/docs/swarm/power-and-clocks.md) has the
details.

## Keeping quality

Speed changes are easy to make and easy to get wrong, so every change went through the gates above. Here are the
release configuration's results, with reasoning at `xhigh` and a 64K-token budget. For HumanEval and MMLU-Pro, the
table also shows halt95's Merlin 3.0.0, run on the same model on the same day:

| Check | flash-next-swarm-quad3090 | Merlin 3.0.0 |
|---|---|---|
| HumanEval | 161/164 | 161/164 |
| MMLU-Pro, 280 questions | 236/280 | 237/280 |
| Needles, 10K to 260K tokens | all found | not run |
| Conversation resumed from RAM vs from GPU memory | identical output (4/4) | no RAM tier |
| Garbage, runaway and NaN checks | clean on every run | not run |

Repeat runs scored 159–162 on HumanEval, and MMLU-Pro moves by about ±4 between runs. Other configurations landed in
the same range: 239 with 16-bit math instead of INT8, and 232 with the 8-bit KV cache. So the differences in the table
are noise.

The one cost we could measure is from INT8 math. Perplexity on code is 0.9–1.5% higher (worse) than with BF16, prose
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

Generation figures are for short prompts with thinking off, added up over all requests. Prefill is prompt
processing.

The chart shows how much each group of changes added to the swarm. It starts from the first working RAM tier. Power
settings changed between steps, so read it as a rough guide, not an exact measurement of each change. The next section compares against
halt95's release, which has no RAM tier.

## Against halt95's latest: Merlin 3.0.0

We ran halt95's current release next to ours on the same hardware, on the same day, with the same benchmarks.
Merlin used its own launcher for this checkpoint: MTP with 3 drafts, FP8 KV, at most 8 requests at once and no RAM
tier. Our figures in this section come from the same day's run, so they differ from the results table above.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/vs-merlin-swarm-dark.svg">
  <img src="assets/vs-merlin-swarm-light.svg" alt="40-agent swarm, turns per minute: flash-next-swarm-quad3090 26.5, Merlin 3.0.0 4.2" width="100%">
</picture>

The swarm is where the two designs differ most. Merlin's 925K-token cache holds fewer than ten of the forty
conversations. It recomputes 99% of every prompt, and requests wait 425 seconds on average for a slot. With the RAM
tier, flash-next-swarm-quad3090 recomputes 4% and requests wait 52 seconds. That's 26.5 turns per minute against 4.2.

To be fair to Merlin, this is a workload it doesn't claim to serve. Its own bench card measures eight agents with
16K-token conversations.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/vs-merlin-dark.svg">
  <img src="assets/vs-merlin-light.svg" alt="Ours relative to Merlin 3.0.0: 16 requests 2.02x, 1 request short prompt 1.92x, 8 requests 1.30x, 8 at 80K 1.22x, 1 after 260K 1.14x, 8 agents 1.11x, prefill 50K 1.11x, prefill 260K 1.10x, 1 request thinking on at 4K 0.93x" width="100%">
</picture>

On the smaller tests, flash-next-swarm-quad3090 is ahead by 10–30% on prefill, long-context decode and 8 requests. At 16 requests the
gap is about 2×, because Merlin serves at most 8 requests at a time and the rest wait. A single short request is 1.9×
faster too. That comes mostly from the drafter and the less wasteful generation steps (sections 2 and 4).

Merlin wins one test, by 7%: a single request with thinking on and a 4K prompt. Thinking text is harder to guess than
code, and for a single request ours makes 7 guesses per step. Most likely, Merlin's 3 guesses waste less work
there.

halt95 doesn't publish the bench card's drivers, so we re-implemented its tests. Merlin reproduces the published
number in our runs: 223.7 tok/s with 2 agents, against 223.9 on the card. Ours is 11–56% ahead with 2–8 agents.
Quality is the same: HumanEval 161/164 for both, and MMLU-Pro 236 against 237.

The full table and its caveats are in the
[README](https://github.com/ththeod/flash-next-swarm-quad3090#compared-with-merlin-300). Both builds ran with the
unlock here. halt95 publishes Merlin's numbers on stock cards, and we'll add those for both builds.

## What did not work

- **Bigger chunks for prompt processing** ran the second GPU pair out of memory.
- **Storing the DeltaNet summary in 16-bit** instead of 32-bit saved no memory, because the memory is reserved in
  fixed-size pages either way.
- **Several smaller tweaks to the GPU code** saved about 2 µs per layer, too little to measure in real use.
- **Moving a layer from one GPU pair to the other** would have cost about 20% of the KV cache.

## Run it yourself

`./setup.sh` checks your machine and shows every download before it starts. It then builds halt95's v2.5.2 with the
our 40 commits on top, verified by commit and tree hash, downloads and verifies the models, and sizes the RAM tier.
Afterwards, `bench/check.sh` runs the quick checks. Every benchmark in this post is in `bench/`, so you can compare
your numbers with ours.

You need four RTX 3090s and at least 96 GB of RAM. The model's n-gram embedding table takes 47.7 GiB and has to stay
in RAM. Together with the server, that comes to about 84 GiB, and whatever is left becomes the RAM tier. By default
the tier is capped at 130 GiB, the largest size we tested, and `--ram-tier` overrides the cap. The
[RAM sizing guide](https://github.com/ththeod/flash-next-swarm-quad3090/blob/swarm/docs/swarm/ram-sizing.md) has
sizes for common RAM amounts.

## What's next

- Numbers for stock cards for both builds, measured the way halt95 publishes Merlin's.
- Fewer guesses for short requests with thinking on, the one test where Merlin is ahead.

If you try flash-next-swarm-quad3090, please [open an issue](https://github.com/ththeod/flash-next-swarm-quad3090/issues) with your
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
