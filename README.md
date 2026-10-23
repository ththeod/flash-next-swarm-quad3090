# flash-next-swarm-quad3090

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/hero-dark.svg">
  <img src="docs/images/hero-light.svg" alt="flash-next-swarm-quad3090: Swift 1.5 on four RTX 3090s. 1,144 tok/s at 16 requests; 8,490 tok/s prefill; 1.21M-token KV cache plus a RAM tier; 26.6 turns per minute with 40 agents." width="100%">
</picture>

**Run a coding-agent swarm on four RTX 3090s without giving up answer quality.** This is a fork of
[halt95's Merlin engine](https://github.com/halt95/merlin-engine) (Flash-Next v2.5.2) that serves halt95's 4-bit build of
[Swift 1.5](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next), a post-train of Qwen3.8-Flash-Next (125B MoE),
tuned for many long conversations at once.

**[Results](#results) · [What it changes](#what-this-fork-changes) · [Quality](#quality) · [Quick start](#quick-start) ·
[Talk to it](#talk-to-it) · [RAM sizing](#ram-sizing) · [Recipe](docs/swarm/recipe.md) ·
[Blog post: how we did it](https://ththeod.github.io/flash-next-swarm-quad3090/)**

- **Built for swarms.** A RAM KV tier keeps agent conversations warm: in a 40-agent run with ~100K-token contexts, 96%
  of prompt tokens are reloaded from RAM instead of recomputed, and the swarm completes 26.6 turns per minute.
- **More at once.** 16 requests decode together (1,144 tok/s in total), in a 1.21M-token VRAM pool.
- **Prompts get in fast.** 8,490 tok/s prefill, and a new 20K prompt still runs at 5,400-5,600 tok/s while eight
  80K-context requests are decoding.
- **Same answers.** Every change passed quality gates before it was kept: HumanEval 161/164 and MMLU-Pro 236/280 at
  `xhigh`, all needles to 260K, bit-exact resumes from the RAM tier.
- **One command to install.** `./setup.sh` checks your machine, sizes the RAM tier for it, and verifies everything it
  downloads.

**This fork or Merlin 3.0.0?** This fork is built on Flash-Next v2.5.2, the release before halt95's current
[Merlin 3.0.0](https://github.com/halt95/merlin-engine) (vLLM 0.32, seven checkpoints, many more GPU setups). Use Merlin
3.0.0 for its checkpoints and hardware range. Use this fork to run many coding agents with Swift 1.5 on 4× RTX 3090.
Porting these changes onto Merlin 3.0.0 is our next step.

## Results

| Metric | Value |
|---|---|
| 40-agent swarm (~100K-token contexts, 20 min) | **26.6 turns/min**, 0 errors |
| Generation, 1 / 8 / 16 requests | **343 / 782 / 1,144 tok/s** (total over all requests) |
| Generation, 8 requests at 80K context | **815 tok/s** |
| Prefill, 10K / 50K / 260K prompt | **7,570 / 8,490 / 7,747 tok/s** |
| KV cache | **1,212,416 tokens** in VRAM (262K per request), plus a **130 GiB** RAM tier |
| Quality, `xhigh`, 64K budget | HumanEval 161/164, MMLU-Pro 236/280, all needles to 260K |

<sub>One machine, 2026-10-22/23, release configuration: ASUS WRX80E-SAGE SE, Threadripper PRO 3955WX, 8×32 GB DDR4-3200,
4× RTX 3090 with P2P, the optional gpu-unlock active, 1400 MHz clock lock, power caps 265/255/255/240 W (tuned to these
boards; we recommend 240 W per card for unlocked cards). Generation:
short prompts, thinking off, 512-token answers. The 80K row is a steady-state window from a different harness, which is
why it reads higher than 8 short requests. Swarm: 40 simulated agents shaped like a real coding-agent harness (MMLU-Pro
text, thinking off, fixed output lengths), measured for 1,200 s after warm-up; a turn took 76.5 s on average, 52.5 s of
it queueing for a slot. Prefill: one cold request. These methods differ from those in [README-halt95.md](README-halt95.md)
(thinking on, 220 W), so the two sets of numbers are not comparable; a same-day comparison with Merlin 3.0.0 on this
machine is coming. Every run and its spread: [docs/swarm/results.md](docs/swarm/results.md).</sub>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/swarm-steps-dark.svg">
  <img src="docs/images/swarm-steps-light.svg" alt="Bar chart: 40-agent swarm turns per minute after each step of the work: 15.9, 23.1, 23.3, 24.2, 25.7, 26.6" width="100%">
</picture>

## What this fork changes

40 commits on halt95's v2.5.2, one change each, with its measurement in the commit message
(`release/swarm-v1.0/patches/0099..0138`). How each was found and measured: [the blog post](https://ththeod.github.io/flash-next-swarm-quad3090/).

| Area | Change | Effect |
|---|---|---|
| RAM KV tier | vLLM's CPU KV offloading made to work with this hybrid model, whose DeltaNet layers carry a recurrent state: one offload stride on both pipeline stages, resume at the previous turn's end, fences for in-flight stores, ARC eviction, each per-layer cache stored once | agent follow-ups reload from RAM instead of recomputing (~4K computed tokens per turn instead of ~15K) |
| Speculative decoding | the [DFlash 2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) drafter instead of MTP: 7 drafts for one request, 3 for 2-8, 2 for 9-16 | 3.4-3.6 tokens accepted per step instead of 3.1; one request +53% on code |
| KV pool | FP4 attention KV with dedicated sm_86 kernels | 26-28% more tokens than FP8, at FP8 speed |
| MoE activations | INT8 activations for the INT4 experts (W4A8 Marlin), two Marlin bugs fixed | prefill +11-12% on stock cards; small quality cost ([Quality](#quality)) |
| Decode kernels | a skinny GEMM for sm_86, small-batch MoE routing, sparse logits, an INT8 draft head, a top-k-first sampler | one request 242 → 297 tok/s on our code prompt |
| Pipeline | decode micro-batches balanced across the two stages | +34% at 16 requests |
| Communication | custom all-reduce only within one PCIe root complex; NCCL LL for small and Simple for large messages; vLLM's FlashInfer PCIe IPC all-reduce, switched on with FlashInfer 0.7 as halt95 does in Merlin 3.0.0 | a new prompt no longer stalls running decodes; +9-11% generation at 8-16 requests |
| Scheduling | a deferred RAM lookup keeps its place in the queue; Mamba-state checkpoints kept only while requests queue | no mid-answer stalls under a swarm |
| Stability | NaN-draft guard, a pipeline crash on one-token chunks, a KV-copy OOM, a torch.compile deadlock with the RAM tier | 0 errors in our swarm runs |

## Quality

Each change went through the same gates before it was kept: a garbage and runaway detector, a NaN probe, needles, and
quality evals at `xhigh` reasoning with a 64K budget.

| Check | Result |
|---|---|
| HumanEval | 161/164 (repeat runs on this build family: 159-162) |
| MMLU-Pro, 280 questions | 236/280; BF16 activations 239, FP8 KV 232 (run-to-run noise about ±4) |
| Needles, 10K to 260K | all found |
| INT8 vs BF16 activations, teacher-forced perplexity on 64K documents | code +0.9% / +1.5%, prose ±0.04% |
| RAM-tier resume vs GPU-cache resume, greedy | bit-exact |

INT8 activations cost a little on code perplexity and nothing measurable on the evals. `VLLM_MARLIN_INPUT_DTYPE=` (empty)
in `config/local.env` restores BF16.

## Quick start

```bash
git clone --branch swarm-v1.0 https://github.com/ththeod/flash-next-swarm-quad3090.git && cd flash-next-swarm-quad3090
./setup.sh            # checks the machine, shows every download with its size, asks once
scripts/run-swarm.sh  # OpenAI-compatible API on 127.0.0.1:8000
bench/check.sh        # quick checks against the running server (~10 min)
```

`setup.sh` runs halt95's v2.5.2 build script unchanged and then this fork's delta, both verified by commit and tree
hash. It downloads the checkpoint and the drafter at pinned revisions, checks them against pinned SHA-256 lists, sizes
the RAM tier for your machine and writes `config/local.env`. Useful options: `--ckpt DIR` reuses a checkpoint you
already have, `--host 0.0.0.0` serves your network with a generated API key, `--systemd` installs the service,
`--build-own` compiles the engine extensions on older systems, and `--dry-run` shows the plan without changing
anything. `scripts/gpu-status.sh` shows the cards, peer-to-peer, power caps, the clock lock and the BF16 unlock.

Every step by hand, plus BIOS settings and the P2P driver: [docs/swarm/recipe.md](docs/swarm/recipe.md).

## Talk to it

OpenAI-compatible chat completions at `http://127.0.0.1:8000/v1`, model name `flash-next-v2` (also `flash-next`):

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "flash-next-v2",
  "messages": [{"role": "user", "content": "Write a haiku about four GPUs."}],
  "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "xhigh"}
}'
```

- **Reasoning.** Thinking is on by default with `reasoning_effort` `"low"`, as on the reference host. Ask for `"high"` or
  `"xhigh"` per request as above, or turn it off with `"enable_thinking": false`. `REASONING_EFFORT=xhigh` in
  `config/local.env` changes the server default; halt95's recipes default to `xhigh`, and our quality numbers use it.
- **Network.** `./setup.sh --host 0.0.0.0` serves your LAN and writes an API key to `config/local.env`; clients send
  `Authorization: Bearer <key>`.
- **Agents.** Point any OpenAI-compatible harness at the URL. 16 requests run at once and more queue. Tool calls
  (`qwen3_coder` parser) and reasoning are parsed.

## What you need

| | |
|---|---|
| GPUs | 4× RTX 3090 (24 GB), headless, PCIe Gen4 x16 each, with peer-to-peer ([P2P driver](docs/swarm/recipe.md#p2p)); no NVLink. Tested only with P2P. |
| System RAM | **96 GB or more.** The n-gram embedding table (47.7 GiB, FP8) must stay in RAM; what is left becomes the RAM tier. [RAM sizing](#ram-sizing) |
| Disk | about 160 GB free (checkpoint 115 GiB, drafter, engine and caches) |
| OS | Linux x86-64 with glibc 2.38 or newer for the prebuilt extension (Ubuntu 24.04, Debian 13); older systems: `./setup.sh --build-own` |
| Driver | NVIDIA with CUDA 13.0 or newer (R580+); tested on 610.57.04 (open, P2P build) |
| Software | Python 3.13 with headers (e.g. `uv python install 3.13`), git, curl, a C/C++ compiler, ninja |
| Power | stock cards: 220 W per card (halt95's reference); unlocked cards: 240 W per card and a 1400 MHz clock lock, with a PSU that has transient headroom ([power and clocks](docs/swarm/power-and-clocks.md)) |

## RAM sizing

**RAM tier ≈ visible RAM − 49 GiB (n-gram table) − 32 GiB (server, OS, headroom) − whatever else the machine runs.**
`/dev/shm` must hold the tier plus 5 GiB. One GiB of tier holds about 100K tokens of context, roughly one long agent
conversation. `setup.sh` applies the rule and caps it at the 130 GiB we measured unless you pass `--ram-tier`.

| Installed RAM | RAM tier | Good for |
|---|---|---|
| 64 GB | does not fit | |
| 96 GB | ~10 GiB | 1-8 agents |
| 128 GB | ~42 GiB | ~10-20 agents |
| 192 GB | ~104 GiB | large swarms |
| 256 GB | 130 GiB, measured (the rule allows up to ~166 on a dedicated machine) | 40-agent swarm |

Only the 256 GB row is measured so far. Details, and what a 64 GB machine would need: [docs/swarm/ram-sizing.md](docs/swarm/ram-sizing.md).

## Hardware tuning (optional)

GeForce Ampere runs BF16/FP16 tensor math with FP32 accumulate at half the workstation rate. The third-party
[gpu-unlock](https://github.com/Zagitta/gpu-unlock) module (Zagitta; not part of this project, use only this URL) lifts
that limit. It is volatile and needs a cold power cycle. On the same build it added 8-9% prefill, which then became
power-bound. How we set it up, check it and undo it: [docs/swarm/gpu-unlock.md](docs/swarm/gpu-unlock.md).

**Unlocked cards need a clock lock.** With unlocked cards our 1500 W PSU tripped (whole machine off) at a 1650 MHz lock,
and again at 1455 MHz about a minute into a step test, at the start of a 4-card BF16 burst. Power caps react too slowly
to stop that spike; the clock lock contains it, and 1400 MHz has run for days. For unlocked cards we recommend 240 W per
card with the 1400 MHz lock; for stock cards, halt95's 220 W and no lock. `scripts/run-swarm.sh` refuses to start on
unlocked cards without the lock. Details: [docs/swarm/power-and-clocks.md](docs/swarm/power-and-clocks.md).

## Known limitations

- Single-request decode depends on the text: 300-343 tok/s on short prompts, about 115-130 tok/s right after a long
  prompt of exam questions. The same-day comparison will show how Merlin 3.0.0 does on the same test.
- Rare generation loops can run to the token cap (1 of 164 HumanEval items at 64K), as on every build we tested.
- The first batch that mixes greedy and sampled requests after a fresh install or a configuration change stalls once
  for ~6 s while sampler kernels compile.
- A full swarm preempts requests (about 190 in 20 minutes at 40 agents); they resume from the RAM tier.
- Prompt logprobs are refused with HTTP 400: a 1024-token chunk of them runs the last pipeline stage out of memory.
- The server uses the checkpoint's own chat template. Merlin 3.0.0 serves a fixed Qwen3.8 template (froggeric's) that
  keeps the prefix cache when an agent adds a system message after the first turn; adopting it is planned once it has
  passed our swarm and quality runs.
- `MemorySwapMax=0`, which keeps the n-gram table out of swap, applies when the server runs as the systemd service, not
  when you start `scripts/run-swarm.sh` by hand on a machine with swap.
- Image input is enabled as in halt95's release but was not part of our tests with the RAM tier and DFlash 2.
- Tested on one machine, with peer-to-peer: ASUS Pro WS WRX80E-SAGE SE, Threadripper PRO 3955WX, 8×32 GB DDR4-3200,
  4× RTX 3090 (Gigabyte ×2, ZOTAC, Founders Edition), be quiet! Dark Power Pro 12 1500 W in single-rail mode.

More, including what we tried and dropped: [docs/swarm/troubleshooting.md](docs/swarm/troubleshooting.md).

## Credits

- **halt95**: the Merlin engine and the Flash-Next recipe ([merlin-engine](https://github.com/halt95/merlin-engine)), the
  [Merlin checkpoint](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin), the
  [Swift 1.5 build](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin) and its
  [recipe](https://github.com/halt95/Swift1.5-qwen38-flash-next-3090s), and the KV scales. This fork starts at halt95's
  v2.5.2 release, its build runs halt95's build script unchanged, halt95's 98 engine commits keep their authorship, and
  halt95's README is kept as [README-halt95.md](README-halt95.md). The RecoverSSM boundary fix (patch 0138) is ported
  from Merlin 3.0.0, and FlashInfer's PCIe all-reduce is switched on as halt95 does there.
- Everyone halt95 credits for v2.5.2, whose work this fork inherits ([README-halt95.md](README-halt95.md#credits)),
  among them the [peakcrosser7 Flash-Next branch](https://github.com/peakcrosser7/vllm/commits/release/qwen38next_offload)
  behind vLLM PRs #53896 and #53899, and [CoolShades](https://github.com/CoolShades) (the idle-power fix in v2.5.2).
- **Qwen** ([Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)), **UkisAI**
  ([Swift 1.5](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next)), **incoai**
  ([DFlash 2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)), **Intel** (AutoRound INT4 experts),
  **RadixArk** (FP8 n-gram table), **DominikBucko** (MTP INT4 packing recipe), **Zagitta**
  ([gpu-unlock](https://github.com/Zagitta/gpu-unlock)), **aikitoria** (P2P kernel modules),
  [vLLM](https://github.com/vllm-project/vllm) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer).
- Juntian Liu: vLLM #57632 (DFlash context K/V precompute in the draft CUDA graph), backported as patch 0109.
- The two W4A8-INT8 Marlin fixes (patches 0136 and 0137) were found independently; fixes by other authors are open
  upstream: vLLM #61405 by ntovas (the missing global scales), #48926 by lazypool (signed group scales) and #59895 by
  plawanrath (absolute-maximum normalization).
- This fork: [ththeod](https://github.com/ththeod), developed with [Claude Code](https://claude.com/claude-code).

This is an independent derivative. It is not made, reviewed or endorsed by halt95, UkisAI, Qwen or incoai. If this
work helps you, consider supporting the engine it is built on: [halt95 on Ko-fi](https://ko-fi.com/halt95).

## License

Code: Apache-2.0 (`LICENSE`), as halt95's engine and vLLM. `NOTICE` lists every file this fork adds or modifies
relative to halt95's v2.5.2 release (Apache-2.0 §4(b)). `bench/data/` holds GSM8K, HumanEval and MMLU-Pro (MIT; see
`bench/data/README.md`). This repository ships no model weights. The checkpoint is under the Swift Open License v1.0
and the Qwen Community License 1.0; read halt95's
[licence summary](https://github.com/halt95/Swift1.5-qwen38-flash-next-3090s#license) before commercial use. The DFlash 2
drafter is Apache-2.0. This is not legal advice; the licence texts govern.
