# Recipe: flash-next-swarm-quad3090 v1.0 from a bare machine

`./setup.sh` runs steps 3-5 for you (and checks what steps 1-2 need). This page is every step by hand; each is what the
reference host runs. Steps marked *optional* change speed, not correctness.

Reference host: ASUS Pro WS WRX80E-SAGE SE WIFI (BIOS 1501), Threadripper PRO 3955WX, 8×32 GB DDR4-3200 (one DIMM per
channel), 4× RTX 3090 on CPU-attached PCIe Gen4 x16 slots, be quiet! Dark Power Pro 12 1500 W, Ubuntu 26.04.1,
kernel 7.0, NVIDIA open driver 610.57.04 with the P2P patch.

## 1. Hardware and BIOS

- Four RTX 3090s, each in a Gen4 x16 slot wired to the CPU (not the chipset). Check after boot:
  `nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.gen.max,pcie.link.width.max --format=csv`.
- BIOS: **Above 4G Decoding** on, **Re-Size BAR** on (each card then shows a 32 GB BAR1), **IOMMU** on, **ACS** off
  (needed for peer-to-peer). Kernel command line: `amd_iommu=on iommu=pt` (Intel: `intel_iommu=on iommu=pt`).
- PSU: put a multi-rail PSU in **single-rail** mode (on the Dark Power Pro 12 that is the OC key). Read
  [power-and-clocks.md](power-and-clocks.md) before raising any power setting.
- RAM: see [ram-sizing.md](ram-sizing.md). Fill every memory channel; bandwidth matters for the n-gram table lookups.

<a id="p2p"></a>

## 2. Driver, peer-to-peer and CUDA

- An NVIDIA driver with CUDA 13.0 or newer. For peer-to-peer on GeForce cards, build the open kernel modules with the
  P2P patch for your exact driver version (we used aikitoria's `610.57.04-p2p-v3` branch of
  `open-gpu-kernel-modules`), install them under `/lib/modules/$(uname -r)/updates/`, run `depmod -a` and
  `update-initramfs -u`, and reboot.
- Check: `nvidia-smi topo -p2p r` (or `scripts/gpu-status.sh`) shows `OK` between every pair. This project was tested only
  with P2P; its all-reduce settings assume it. halt95's release also runs without P2P, with slower prefill.
- A kernel update needs the modules rebuilt. Keep the stock modules installed as the fallback.

## 3. Build the engine

Python 3.13 with venv and headers on `PATH` as `python3.13` (Ubuntu before 26.04: `uv python install 3.13` and add
`~/.local/bin` to `PATH`, or the deadsnakes PPA), git, curl, a C/C++ compiler and ninja.

```bash
git clone --branch swarm-v1.0 https://github.com/ththeod/flash-next-swarm-quad3090.git
cd flash-next-swarm-quad3090
# halt95's v2.5.2 release assets (hash-checked by halt95's build script)
curl -fLO https://github.com/halt95/merlin-engine/releases/download/v2.5.2/v2.5.2-from-upstream-v0.30.0.bundle.gz
curl -fLO https://github.com/halt95/merlin-engine/releases/download/v2.2.0/build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz
# this project's MoE extension (downloaded automatically if absent)
curl -fLO https://github.com/ththeod/flash-next-swarm-quad3090/releases/download/swarm-v1.0/moe-C-sm86-py313-cu134-swarm-v1.0.tar.gz
scripts/build-swarm.sh vllm-swarm venv-swarm
```

`build-swarm.sh` runs halt95's `scripts/build-v2.5.sh` unchanged (vLLM v0.30.0 + halt95's 98 commits, pinned venv,
halt95's build products), then fetches this project's bundle, verifies the commit and tree against
`upstream/PIN-swarm-v1.0`, installs the rebuilt MoE extension and FlashInfer 0.7.0.post1, and re-hashes every build
product. The prebuilt extension needs glibc 2.38 or newer (Ubuntu 24.04, Debian 13). On older systems, or to build
everything yourself, use `BUILD_OWN=1 scripts/build-swarm.sh ...` (15-30 min; needs cmake >= 3.26; halt95's tarball
still supplies the unchanged stock vLLM build products).

## 4. Model files (about 125 GB)

`hf` is in the engine's venv after step 3.

```bash
venv-swarm/bin/hf download halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin --revision f3a8182d39702b7637b45a23c613da72d9ab800c \
  --local-dir models/Swift1.5-Merlin
(cd models/Swift1.5-Merlin && grep -v '^#' ../../release/swarm-v1.0/checkpoint.sha256 | sha256sum -c)
venv-swarm/bin/hf download incoai/Qwen3.8-Flash-Next-DFlash2 --revision a22a1744e85c17364dcc54a6a420af7135d4c7f6 \
  --local-dir models/DFlash2
(cd models/DFlash2 && grep -v '^#' ../../release/swarm-v1.0/drafter.sha256 | sha256sum -c)
venv-swarm/bin/python scripts/graft-drafter-embed.py models/DFlash2 models/Swift1.5-Merlin models/DFlash2-swiftembed
```

The graft copies the checkpoint's input embedding next to the drafter: under pipeline parallelism the drafter runs on
the last stage, which has no target embedding to share. No weights are changed and this repository ships none. The
weights are under the Swift Open License v1.0 and the Qwen Community License 1.0 (see halt95's licence summary).

## 5. RAM tier and /dev/shm

```bash
scripts/size-ram-tier.sh 0 >> config/local.env    # argument: GiB that other software on the machine needs
```

It appends `RAM_TIER_GIB` (at most 130 GiB, the size we measured) and prints on the terminal the `/etc/fstab` line for
`/dev/shm` if it is too small (the tier lives there). Add your
paths to `config/local.env` if they differ from the defaults in `config/swarm.env` (TREE, VENV, CKPT, DRAFTER,
CACHE_ROOT). The n-gram table must never swap; the systemd unit sets `MemorySwapMax=0`. Details:
[ram-sizing.md](ram-sizing.md).

## 6. Power and clocks

Copy `systemd/gpu-power.example` to `/etc/default/gpu-power`, adjust the bus ids and caps, and install
`systemd/gpu-power.service`. *Optional:* Zagitta's BF16 unlock, with the clock lock it requires:
[gpu-unlock.md](gpu-unlock.md). Background and measurements: [power-and-clocks.md](power-and-clocks.md).

## 7. Start

```bash
scripts/run-swarm.sh            # or: install systemd/flash-next-swarm.service
```

The first boot compiles kernels and CUDA graphs (about 10 minutes); later boots take about 5.5 minutes. The API is
OpenAI-compatible on 127.0.0.1:8000; to serve a network, set `HOST=0.0.0.0` and `API_KEY=...` in `config/local.env`
(`./setup.sh --host 0.0.0.0` does both). Model names: `flash-next-v2` and `flash-next` (`MODEL_NAME` in
`scripts/serve-swarm.sh`).

<a id="validate"></a>

## 8. Validate

`bench/check.sh` runs the quick ones with the right URL and API key (it creates `bench/.venv` the first time). By hand,
from `bench/` with `openai pandas pyarrow numpy` installed and `LLM_BASE_URL` / `LLM_API_KEY` set if needed:

| Check | Command | Pass |
|---|---|---|
| Garbage / runaway gate | `N=8 python3 runaway_probe.py` | no runaway, no garbage |
| NaN probe | `python3 nan_poke.py` | no non-finite logits |
| RAM tier reload | `python3 offload_validate.py` | reloaded turns match |
| Generation | `python3 concurrency.py 1 8 16` | ~300 / ~760 / ~1,120 tok/s with the unlock |
| Prefill, needles | `python3 longctx.py 10000 50000 250000` | ~7,200 / ~8,400 / ~7,700 tok/s with the unlock; needles found |

Also check `journalctl -k | grep -c Xid` stays 0 and the server log has no `CUDA out of memory`.

## 9. Benchmarks

`bench/swarm_bench.py` (40 agents, 20 minutes, seed 1) and `bench/prof_longctx.py` (8 requests at 80K; `MIX=20000` adds the 20K
prompt). [results.md](results.md) has the reference numbers and how each was run.
