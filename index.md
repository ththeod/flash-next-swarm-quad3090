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
Every turn resends the whole conversation, so a server is only fast if it still has the earlier part cached.
halt95's [Flash-Next engine](https://github.com/halt95/merlin-engine) is the reason a 125B-parameter model like
Qwen3.8-Flash-Next runs on four RTX 3090s at all, but its GPU cache fits fewer than ten conversations that long. In a
swarm of forty, almost every turn starts by recomputing its prompt from scratch.

Our version, flash-next-swarm-quad3090, adds 40 changes for that workload on top of halt95's v2.5.2. It serves
Swift 1.5, a fine-tune of Qwen3.8-Flash-Next, and adds:

- a RAM tier that keeps conversations warm;
- a better drafter, which guesses tokens ahead (DFlash 2);
- a 4-bit KV cache, the model's working memory, which holds 1.2M tokens;
- less wasted work per generated token;
- faster communication between the GPUs.

On the same hardware and the same day, a 40-agent benchmark ran at 26.5 turns a minute on our version and 4.2 on
halt95's latest release, Merlin 3.0.0. Both scored the same on HumanEval and MMLU-Pro.

The numbers in this post were measured with Zagitta's optional BF16 unlock and a 1400 MHz clock lock (see
[Run it yourself](#run-it-yourself)). Numbers for stock cards will follow, and yours will depend on your board, PCIe
layout and RAM.

## How fast is it

The main benchmark is a 40-agent swarm (`bench/swarm_bench.py`). Forty simulated coding agents with ~100K-token
conversations, subagents and tool calls run for 20 minutes, shaped after a real harness. It counts completed turns
per minute.

<figure class="gif">
  <iframe src="https://giphy.com/embed/mPOGx4hJtOWSA" title="A huge crowd of cheering Minions" frameborder="0" loading="lazy" allowfullscreen></iframe>
  <figcaption>The swarm benchmark, artist's impression · <a href="https://giphy.com/gifs/despicable-me-minions-cute-mPOGx4hJtOWSA">via GIPHY</a></figcaption>
</figure>

| Metric | Value |
|---|---|
| 40-agent swarm | 26.6 turns/min, 0 errors in 20 minutes |
| Generation, 1 / 8 / 16 requests | 343 / 782 / 1,144 tok/s |
| Generation, 8 requests at 80K context | 815 tok/s |
| Prefill, 10K / 50K / 260K prompt | 7,570 / 8,490 / 7,747 tok/s |
| KV cache | 1,212,416 tokens in VRAM, plus a 130 GiB RAM tier |

Generation figures are for short prompts with thinking off, added up over all requests. Prefill is prompt
processing.

### Against halt95's Merlin 3.0.0, same day

We ran halt95's current release next to ours on the same hardware, on the same day, with the same benchmarks. Merlin
used its own launcher for this model: the built-in drafter, which guesses 3 tokens ahead, an 8-bit KV cache and at
most 8 requests at once. These are that day's numbers for both builds, so ours differ from the table above.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/vs-merlin-swarm-dark.svg">
  <img src="assets/vs-merlin-swarm-light.svg" alt="40-agent swarm, turns per minute: flash-next-swarm-quad3090 26.5, Merlin 3.0.0 4.2" width="100%">
</picture>

Merlin's cache holds fewer than ten of the forty conversations, so it recomputes 99% of every prompt, and requests
wait 425 seconds on average for a slot. Ours recomputes 4%, and requests wait 52 seconds. To be fair to Merlin, it
doesn't claim to serve this workload: its own bench card measures eight agents with 16K-token conversations.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/vs-merlin-dark.svg">
  <img src="assets/vs-merlin-light.svg" alt="Ours relative to Merlin 3.0.0: 16 requests 2.02x, 1 request short prompt 1.92x, 8 requests 1.30x, 8 at 80K 1.22x, 1 after 260K 1.14x, 8 agents 1.11x, prefill 50K 1.11x, prefill 260K 1.10x, 1 request thinking on at 4K 0.93x" width="100%">
</picture>

On the smaller tests, ours is ahead by 10–30%, and by about 2× at 16 requests, where Merlin serves 8 at a time and the
rest wait. A single short request is 1.9× faster too.

Merlin wins one test, by 7%: a single request with thinking on and a short prompt. Thinking text is harder to guess
than code, and Merlin's 3 guesses per step most likely waste less work there than our 7.

Quality is the same: HumanEval 161/164 for both, and MMLU-Pro 236 against 237. We also re-implemented the tests on
halt95's bench card. Merlin reproduces its published numbers in our runs, and ours is 11–56% ahead with 2–8 agents.
The [README](https://github.com/ththeod/flash-next-swarm-quad3090#compared-with-merlin-300) has the full table and
its caveats.

## How it works

Each part below says what was slow in halt95's build, what we changed, and why that helps. We found what to change by
profiling, which records where every millisecond of a generation step goes on each GPU. The patch numbers point to
the code in `release/swarm-v1.0/patches/`.

One thing helps with the rest: the model runs in two halves, each on a pair of GPUs. The halves work like an assembly
line, and within a pair the two GPUs split each layer's math between them.

### 1. Conversations stay in RAM

halt95's build keeps conversations only in GPU memory. When that fills up, the oldest are thrown away. When an agent
comes back, its whole prompt is recomputed, which takes over ten seconds for 100K tokens even at full speed.

vLLM can already park conversations in system RAM, but not for this model. Flash-Next mixes two kinds of layers.
Attention layers keep a note for every token, and those notes can be copied out and back in pieces. DeltaNet layers
keep one running summary of the whole conversation instead, and that summary only exists where it was saved. To
resume from RAM, you need a summary saved at exactly the token where the last turn ended.

Our system saves that summary at the end of every turn and resumes from there. Loading a conversation back from RAM
is much faster than recomputing it, so in the swarm benchmark 96% of prompt tokens come from RAM. It also protects
long-lived parent conversations, so a burst of short subagents doesn't push them out.

*Patches 0099, 0121–0130.*

### 2. Better guesses: the DFlash 2 drafter

When a GPU generates text one token at a time, it spends most of each step reading the model's weights for that one
token. Speculative decoding makes better use of that work. A small, fast drafter guesses several tokens ahead, and the
big model checks all the guesses in one step, keeping the ones it agrees with.

halt95's build uses the model's built-in drafter (MTP), which guesses 3 tokens at a time. We switched to incoai's
[DFlash 2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2). It is a separately trained drafter that sees
more of the big model's internal state and guesses up to 7 tokens. It is right more often: 3.4–3.6 tokens are kept
per step, against 3.1. So every step produces more text, and a single request on a code prompt generates about 50%
faster.

With many requests the GPUs are already busy, so wrong guesses cost more. Our system makes fewer guesses as load
grows, down to 2.

*Patches 0100–0103, 0108–0113.*

### 3. More room: a 4-bit KV cache

The KV cache is the model's working memory: notes on every token of every conversation it holds. Its size decides how
many tokens fit in VRAM at once. halt95's build stores those notes as 8-bit numbers. We store them as 4-bit numbers,
with a shared scale for every 16 values, so the same VRAM holds 26–28% more tokens.

The GPU has to unpack 4-bit numbers before it can use them. Our system does that with simple bit operations, which
give exactly the same values as the slow way, and uses attention code written for 4-bit data. So it runs as fast as
8-bit, and the scores didn't drop.

DFlash 2 also gets its own small cache. Otherwise the drafter's memory needs would cut the main cache to a third.
Altogether the cache holds 1.21M tokens, against 925K in halt95's build.

*Patches 0104–0107, 0133–0135.*

### 4. Less wasted work per step

Profiling one generation step showed that half the time went to matrix multiplications. The GPU's math units work on
blocks of 16 rows, but a single request only gives them 8 (one token plus seven guesses), so the blocks ran half empty.
We changed three things:

- **Matrix code sized for these small batches,** so the GPU stops computing empty rows.
- **A faster way for the two GPUs in a pair to combine their results** after each layer (more on that in section 5).
- **A cheaper last step.** When a request's sampling settings allow it, picking the next token looks only at the top
  candidates instead of all ~250,000 possible tokens.

A single request now generates about 23% faster.

With several requests at once, a different problem showed up. The assembly line works on two groups of requests and
moves at the pace of the bigger group, and the groups were often uneven: 6 and 2 instead of 4 and 4. Our system
keeps them even, which makes 16 requests 34% faster.

*Patches 0114–0118, 0131.*

### 5. A new prompt doesn't freeze everyone

The two GPUs in a pair combine their partial results after every layer, so a long new prompt means a lot of traffic
between them. The fastest method for this only works well when both GPUs connect to the same PCIe branch of the CPU.
On many boards they don't, and then large transfers take almost four times as long as with NVIDIA's standard library,
NCCL.

Our system checks how the GPUs are connected at startup, and uses the fast method only where it is fast: small
transfers on a shared branch. Large transfers go through NCCL, in its mode for large messages. FlashInfer's newer
PCIe method, which halt95's Merlin 3.0.0 also uses, speeds up the small ones.

In the long-context probe (`bench/prof_longctx.py`), eight requests are generating at 80K context when a 20K prompt
arrives. The new prompt is done in 3.6 seconds, about a second more than on an idle server, and the eight keep
generating through it.

*Patch 0132, plus NCCL and FlashInfer settings.*

### 6. 8-bit math, which GeForce runs at full speed

GeForce 3090s are deliberately limited. The 16-bit math this model normally uses (BF16) runs at half the speed it does
on the same chip in a workstation card, but 8-bit integer math (INT8) runs at full speed.

Our system converts the numbers going into the model's expert layers to INT8 on the fly. vLLM has this feature, but
it doesn't work with this model yet; ours does. Prompt processing is 11–12% faster on stock cards, at a small cost on
code (see [Quality](#quality)).

*Patches 0136, 0137.*

### Adding it up

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/swarm-steps-dark.svg">
  <img src="assets/swarm-steps-light.svg" alt="Bar chart: 40-agent swarm turns per minute after each group of changes: RAM tier, DFlash 2 and small-batch math 15.9; smarter RAM resume and scheduling 23.1; 4-bit KV cache 23.3; 8-bit math and BF16 unlock 24.2; faster GPU traffic with NCCL 25.7; faster GPU traffic with FlashInfer 26.6" width="100%">
</picture>

The chart shows the swarm benchmark after each group of changes, in the order we made them. Power settings changed
along the way, so read it as a rough guide.

## Quality

Every change had to pass the same checks before it stayed:

- HumanEval and MMLU-Pro, with reasoning at `xhigh`;
- needle-in-a-haystack tests up to 260K tokens;
- a check that a conversation resumed from RAM gives exactly the same output as one resumed from GPU memory;
- detectors for garbage, runaway and numerical errors.

A change that failed was dropped, however fast it was.

| Check | Our version | Merlin 3.0.0 |
|---|---|---|
| HumanEval | 161/164 | 161/164 |
| MMLU-Pro, 280 questions | 236/280 | 237/280 |
| Needles, 10K to 260K tokens | all found | — |
| Resumed from RAM vs from GPU memory | identical output | no RAM tier |

Scores move by a few points between runs (HumanEval 159–162, MMLU-Pro about ±4), so these differences are noise.

The one cost we could measure is from the 8-bit math in section 6. The model predicts known code 0.9–1.5% less well
(perplexity), with no difference on prose or on the evals. One setting turns it off.

## Which one should you run?

Merlin 3.0.0 supports seven models and many more GPU setups. Ours runs one model, Swift 1.5, on four RTX 3090s. On
that setup it's faster at nearly everything, and most of all with many agents or long conversations.

## Run it yourself

You need:

- four RTX 3090s with peer-to-peer enabled;
- at least 96 GB of RAM;
- about 160 GB of free disk;
- a recent Linux.

```bash
git clone --branch swarm-v1.0 https://github.com/ththeod/flash-next-swarm-quad3090.git
cd flash-next-swarm-quad3090
./setup.sh            # checks the machine, shows each download, asks once
scripts/run-swarm.sh  # starts the server
bench/check.sh        # quick checks (~10 min)
```

`setup.sh` builds halt95's v2.5.2 with our 40 commits on top, verified by commit and tree hash. It downloads and
verifies the models and sizes the RAM tier. The first start takes about 10 minutes while the server compiles its GPU
code, and later starts take about 5.5. You then have an OpenAI-compatible API at `http://127.0.0.1:8000/v1`, which any
agent harness can use, with 16 requests at once.

**RAM.** The model's n-gram embedding table takes 47.7 GiB and has to stay in RAM. Together with the server, that
comes to about 84 GiB, and whatever is left becomes the RAM tier. By default the tier is capped at 130 GiB, the largest
size we tested. The
[RAM sizing guide](https://github.com/ththeod/flash-next-swarm-quad3090/blob/swarm/docs/swarm/ram-sizing.md) has
sizes for common RAM amounts.

**Optional: the BF16 unlock.** Zagitta's [gpu-unlock](https://github.com/Zagitta/gpu-unlock) kernel module removes the
half-speed BF16 limit from section 6. It is not part of this project, and you use it at your own risk. It makes prompt
processing another 8–9% faster, and the cards then run at their power limits. Unlocked cards also draw short power
spikes that can trip a power supply even when the average draw is well within its rating. Power caps react too slowly
to catch them, but a clock lock keeps them in check.

**Power settings.** On stock cards, use halt95's 220 W per card. With the unlock, we recommend 240 W per card and a
1400 MHz lock. `run-swarm.sh` refuses to start on unlocked cards without the lock. The
[power guide](https://github.com/ththeod/flash-next-swarm-quad3090/blob/swarm/docs/swarm/power-and-clocks.md) has the
details.

## What's next

- Numbers for stock cards for both builds, measured the way halt95 publishes Merlin's.
- Fewer guesses for short requests with thinking on, the one test where Merlin is ahead.

If you try it, please [open an issue](https://github.com/ththeod/flash-next-swarm-quad3090/issues) with your
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
