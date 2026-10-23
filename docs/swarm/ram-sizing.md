# RAM sizing

## What uses system RAM

Measured on the reference host while serving a 40-agent workload (247 GiB visible, RAM tier 130 GiB):

| Use | Size | Notes |
|---|---|---|
| n-gram (PLE) embedding table, FP8 | 48.6 GiB | the 47.7 GiB table plus its process (the PLE offload worker); must never swap |
| RAM KV tier | 130 GiB | `/dev/shm`, pinned with cudaHostRegister; the only part that scales |
| vLLM workers, engine core, API server | ~17 GiB | |
| model weights while loading | page cache only | the checkpoint (~115 GiB) is read through the page cache; nothing extra stays resident |

## The rule

```
RAM tier = visible RAM - 49 GiB (n-gram table) - 32 GiB (server, OS, first-boot compile, headroom) - other software
/dev/shm >= RAM tier + 5 GiB
```

`setup.sh` and `scripts/size-ram-tier.sh [OTHER_GIB]` apply it to your machine and cap the result at 130 GiB, the size we
measured (`setup.sh --ram-tier` sets it yourself). "Visible RAM" is what `free -g` reports; it is a few
percent below the installed size.

One GiB of tier holds about 100K tokens of cached context (rows of 40 MiB per 4,096 tokens, plus state rows), which is
roughly one long agent conversation. In our 40-agent benchmark 96% of all prompt tokens were reloaded from the tier
instead of recomputed; with the real agents on the reference host it was 86%.

## By installed RAM

| Installed | Visible (approx.) | RAM tier | Good for | Status |
|---|---|---|---|---|
| 64 GB | ~61 GiB | does not fit | | the n-gram table plus the server need about 84 GiB |
| 96 GB | ~92 GiB | ~10 GiB | 1-8 agents | from the rule, not yet measured |
| 128 GB | ~123 GiB | ~42 GiB | ~10-20 agents | from the rule, not yet measured |
| 192 GB | ~185 GiB | ~104 GiB | large swarms | from the rule, not yet measured |
| 256 GB | ~247 GiB | 130 GiB | 40-agent swarm | **measured** (with ~13 GiB of other services; the rule allows ~166 on a dedicated machine) |

With less tier, a swarm still runs correctly; requests whose context was evicted are recomputed at 7-8K tok/s.

## 64 GB machines

Not supported in v1.0. Two ways it could work, neither built yet:

- keep the n-gram table on NVMe and read it through mmap (speed unknown: every token reads rows of the table);
- store the n-gram table in FP4 (~25 GiB; NVIDIA's FP4 checkpoint of the base model does this).

## /dev/shm

Ubuntu sizes `/dev/shm` at half the RAM by default. For a 130 GiB tier, add to `/etc/fstab`:

```
tmpfs /dev/shm tmpfs defaults,size=140G 0 0
```

then `sudo mount -o remount /dev/shm`. `scripts/run-swarm.sh` refuses to start when `/dev/shm` is too small.

## Swap

Keep swap if you want it, but never for the server: `systemd/flash-next-swarm.service` sets `MemorySwapMax=0`. A
swapped-out n-gram row stalls every request that touches it.
