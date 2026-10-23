# The BF16 unlock (optional, third-party)

**gpu-unlock is not part of this project.** It is [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock), a kernel
module that runs before the NVIDIA driver and lifts GeForce Ampere's half-rate BF16/FP16 tensor math (FP32
accumulate). We use it and document what worked for us; we do not maintain it, ship it or support it. It changes how
your cards run: read its README, and use it at your own risk. **Use only that URL: look-alike repositories exist.**

Everything in this project works without it. With it, prefill becomes power-bound and gets 8-9% faster on our machine
(see [power-and-clocks.md](power-and-clocks.md)); generation, which is latency-bound, was not compared separately.

## Before you start

- Cards: GA102/GA104/GA106 (RTX 3090 and others; the upstream README lists them). We tested 4× RTX 3090.
- **Secure Boot off**, or enroll the DKMS signing key yourself: the module is unsigned and will not load otherwise.
  `mokutil --sb-state` shows the state.
- Kernel headers for every installed kernel, and DKMS.
- A PSU with transient headroom. With unlocked cards our 1500 W PSU tripped (the whole box went off) at a 1650 MHz
  and a 1455 MHz clock lock; 1400 MHz held. **Set up the clock lock before the first unlocked boot** (step 2).

## 1. Install gpu-unlock (upstream's installer)

Upstream's `make install` detects Debian/Ubuntu, Arch or Fedora/RHEL, builds a package locally (its NVIDIA inputs are
fetched and SHA-256 checked) and adds initramfs hooks (initramfs-tools, dracut or mkinitcpio). We used commit
`649b36c27178fdf9fcaafd767d60dfbb78b6caa2` (2026-10-22):

```bash
git clone https://github.com/Zagitta/gpu-unlock && cd gpu-unlock
git checkout 649b36c27178fdf9fcaafd767d60dfbb78b6caa2     # optional: the commit we tested
make install
```

Then opt in on the kernel command line with `gpu_unlock.targets=all` (the module does nothing without it):

| Boot setup | How |
|---|---|
| GRUB, Debian/Ubuntu/Mint | add it inside `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, then `sudo update-grub` |
| GRUB, Arch/CachyOS | same edit, then `sudo grub-mkconfig -o /boot/grub/grub.cfg` |
| Fedora/RHEL | `sudo grubby --update-kernel=ALL --args="gpu_unlock.targets=all"` |
| systemd-boot | append it to the `options` line of your entry (`bootctl status` shows where) |
| UKI / generated entries | add it to `/etc/kernel/cmdline` and regenerate the images |

## 2. Clock lock first

```bash
sudo cp systemd/gpu-power.example /etc/default/gpu-power     # switch to the unlocked lines: 240 W, GPU_MAX_CLOCK=1400
sudo cp systemd/gpu-power.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now gpu-power.service
```

`scripts/run-swarm.sh` refuses to start on unlocked cards (or cards it cannot prove are locked) unless this service is
active with `GPU_MAX_CLOCK` at or below 1400 MHz. With a bigger PSU, raise it in small steps under load and set
`UNLOCK_MAX_CLOCK` to match; if you lock clocks some other way, `UNLOCK_CLOCK_OK=1` skips the check.

## 3. Cold power cycle

**Shut down and switch the power off** (a reboot is not enough), then start the machine.

## 4. Check

```bash
scripts/gpu-status.sh
```

Every card should show `BF16: UNLOCKED` and the clock lock should read 1400. The kernel log has one
`gpu_unlock: unlocked` line per card (`journalctl -k -b | grep gpu_unlock`; reading it needs the `adm` group).
If `gpu_unlock.targets` is set but no card is unlocked, the machine was rebooted rather than powered off.

## Undo

Remove `gpu_unlock.targets=...` from the kernel command line (for one boot, delete it in the bootloader's edit screen)
and power-cycle. Removing the package also removes the module and its initramfs hooks. The unlock is volatile: nothing
is written to the cards.

## Notes from our machine

- Ubuntu 26.04, kernel 7.0, GRUB, driver 610.57.04 (open, P2P build), 4× RTX 3090 (Gigabyte ×2, ZOTAC, Founders
  Edition): all four unlocked on every cold boot.
- The kernel is tainted by the out-of-tree, unsigned module; that is expected.
- Sustained BF16 throughput under the power caps was 70-75 TFLOPS per card at every clock lock from 1400 to 1440 MHz:
  the cards are power-bound, so a higher lock buys generation speed (~1% per 40 MHz) and PSU risk, not prefill.
