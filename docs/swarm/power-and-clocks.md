# Power and clocks

## The short version

- Power caps set the sustained draw. **The clock lock is what protects the PSU**: it limits the millisecond spikes at
  the start of a burst, which power caps react to too slowly.
- With the BF16 unlock, prefill becomes power-bound: under prefill the cards sit at ~930-1,035 MHz regardless of the
  lock, and sustained BF16 throughput under these caps (70-75 TFLOPS per card) does not change with the lock. (GA102's
  stock BF16 rate is a 71 TFLOPS *peak* at boost clock; at these caps stock cards sustain less, not measured here.)
- **Recommended settings** (`systemd/gpu-power.example`):

  | Cards | Power cap | Clock lock |
  |---|---|---|
  | Stock (locked) | 220 W on every card (halt95's reference) | none |
  | Unlocked with gpu-unlock | 240 W on every card | **1400 MHz** |

- Our measured runs used 265 / 255 / 255 / 240 W, tuned to this machine's boards (two of them are 390 W-class cards with
  more idle draw; at a flat 200 W one ran at ~1000 MHz). All are well below the 350-390 W stock limits.

## The BF16 unlock (optional)

GeForce Ampere runs BF16/FP16 tensor math with FP32 accumulate at half the workstation rate (71 vs ~155 TFLOPS on
GA102). [gpu-unlock](https://github.com/Zagitta/gpu-unlock) by Zagitta is a kernel module that lifts this before the
NVIDIA driver loads. It is volatile (no flash or fuse writes) and needs a **cold power cycle** to take effect. Use only
that URL: look-alike repositories exist. Setup, checks and undo: [gpu-unlock.md](gpu-unlock.md).

What it bought here, on the same build (INT8 MoE activations, measured before the NCCL and FlashInfer all-reduce
changes, same caps):

| | Prefill 10K / 50K (tok/s) |
|---|---|
| stock (locked) cards, BF16 MoE activations | 5,589 / 6,314 |
| stock cards, INT8 MoE activations | 6,201 / 7,063 (+11-12%) |
| unlocked, 1400 MHz lock, INT8 | 6,739 / 7,630 (+8-9% more) |

The current release on unlocked cards reaches 7,570 / 8,490. INT8 tensor math runs at full rate on GeForce
with or without the unlock, so INT8 activations help stock cards too. Generation is latency-bound and was not
compared separately.

## PSU trips we hit (be quiet! Dark Power Pro 12 1500 W, single rail)

| Lock with unlocked cards | Result |
|---|---|
| 1650 MHz | whole box lost power during a 4-card BF16 burst |
| 1410 / 1425 / 1440 MHz | pass: 4 cards × 20 bursts each, 0 Xid |
| 1455 MHz | whole box lost power about 1 minute into the step |
| 1400 MHz | days of benchmarks and production, no trip |

The 100 ms average was the same (~1,016 W for the four GPUs) at every step; the trip comes from a spike in the first
milliseconds of a burst. More clock headroom needs a PSU with more transient capacity, not other settings.

## Cap sweep (unlocked, 1400 MHz lock)

| Caps vs reference | Prefill | Generation | 4-GPU peak |
|---|---|---|---|
| reference (265/255/255/240) | 7,374 / 7,672 / 8,428 tok/s at 10K / 20K / 50K | 1,030 tok/s at 16 requests | 996 W |
| -15 W | -3 to -5% | 0 | 948 W |
| -30 W | -12 to -15% | -7% | 887 W |
| -45 W | -22 to -27% | -19% | 830 W |

(Measured before the FlashInfer all-reduce change, with FlashInfer 0.6.18; prefill is the same within run-to-run noise
on the release, and the relative steps still apply.) If power matters, -15 W saves
~50 W at peak for ~4% prefill.

## Clock steps and generation

Generation is latency-bound and gains roughly 1% per 40 MHz of lock. Without the unlock, 1650 MHz gave +6% decode and
+13% prefill over 1400 MHz at ~105 W more under load.

## Watch for

- `journalctl -k | grep -E 'Xid|AER'`. Corrected PCIe AER errors on one slot (Bad TLP, Bad DLLP, replay timer) point at
  a marginal riser or seating; act if uncorrected errors or Xid appear.
- A box that powers off with no kernel message is a PSU trip: lower the clock lock first.
