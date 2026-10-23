# Troubleshooting, and what did not work

## Known behaviours

| Symptom | Cause | What to do |
|---|---|---|
| Box powers off under load, no kernel message | PSU transient trip | lower `GPU_MAX_CLOCK` ([power-and-clocks.md](power-and-clocks.md)) |
| ~6 s stall on the first batch mixing greedy and sampled requests after a config change | Triton sampler kernels compile into a new cache directory | none; it happens once per configuration |
| First boot takes ~10 min | torch.compile and CUDA graph capture | later boots take ~5.5 min; keep `CACHE_ROOT` |
| Requests queue with the pool at 95% | KV capacity, not a fault | lower concurrency, or more RAM tier so evicted contexts reload instead of recompute |
| Prompt-logprobs request returns 400 | refused on purpose: a 1024-token chunk of prompt logits OOMs the last pipeline stage | `VLLM_EXP_REJECT_PROMPT_LOGPROBS=0` with `KV_CACHE_MEMORY=3400000000` (we measured perplexity this way) |
| A generation loops to the token cap | model behaviour, rare (1 of 164 HumanEval items at 64K) | set a sensible `max_tokens`; retry with another seed |
| `run-swarm.sh` refuses to start: /dev/shm too small | the RAM tier lives in /dev/shm | [ram-sizing.md](ram-sizing.md) |
| `run-swarm.sh` refuses to start: UNSAFE unlocked cards | the BF16 unlock is active without an active clock lock <= 1400 MHz | [gpu-unlock.md](gpu-unlock.md), step 2 |
| `gpu-status.sh` shows `gpu_unlock.targets` set but no card unlocked | the machine was rebooted, not powered off | shut down, switch off, start |
| Corrected PCIe AER errors in `journalctl -k` | marginal Gen4 link (riser, seating) | harmless while corrected; reseat if uncorrected errors or Xid appear |
| Server output is random multilingual tokens with `VLLM_MARLIN_INPUT_DTYPE=int8` | an extension without this fork's Marlin fixes | `sha256sum vllm-swarm/vllm/_moe_C_stable_libtorch.abi3.so` must match `moe_C_sha256` in `upstream/PIN-swarm-v1.0`; re-run `scripts/build-swarm.sh` |
| CUDA OOM at boot or on the first long prompt | `KV_CACHE_MEMORY` above 3.9e9, or other processes on the GPUs | keep 3.9e9; GPUs must be otherwise empty |

## Settings that must not be combined

- `VLLM_EXP_SW_BLOCK_SIZE` without `VLLM_EXP_SW_SCRATCH=1`: prompts above ~9K tokens are never admitted.
- `--use-replayssm` with the RAM tier: use `REPLAYSSM=1`, which sets the RecoverSSM offload mode.

## Tried and rejected

| Experiment | Result |
|---|---|
| 2,048- or 4,096-token prefill chunks | OOM on the last pipeline stage at long context; would need ~12% less KV |
| BF16 instead of FP32 DeltaNet state | no capacity gain: the state page is padded back to the same size |
| 8 warps for the FP4 attention kernel | no gain (260K prefill 7,566-7,582 vs 7,700-7,724 with 4) |
| Marlin MoE block size 8 | no gain |
| Plain 3-gram prompt lookup with MTP | 10% fewer accepted tokens on agent traffic; gated lookup +1-8% at wide verify (not built) |
| NVIDIA's FP4 checkpoint of the base model | same quality, 2-3× slower per task: the base model thinks much longer than Swift |
| DeepSeek-V4.1-Flash 2.9 bpw with experts split between GPU and CPU | ~31 tok/s, one request at a time |
| Clock lock above 1440 MHz with the unlock | PSU trips |
