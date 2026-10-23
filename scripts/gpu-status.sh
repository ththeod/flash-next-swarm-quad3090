#!/usr/bin/env bash
# GPU status for flash-next-swarm-quad3090: cards, peer-to-peer, power caps, the clock lock and the BF16 unlock.
#   scripts/gpu-status.sh            report
#   scripts/gpu-status.sh --guard    exit 1 when the cards are (or may be) unlocked but no safe clock lock is configured;
#                                    scripts/run-swarm.sh runs this before every start
# The unlock is the third-party gpu-unlock module (https://github.com/Zagitta/gpu-unlock); see docs/swarm/gpu-unlock.md.
# It fails closed: when the module is loaded or gpu_unlock.targets is set and the kernel log cannot show that every
# card stayed locked, the cards count as unlocked.
# Env: UNLOCK_MAX_CLOCK (MHz, default 1400: the lock we run with unlocked cards; 1410-1440 also passed our PSU test,
#      1455 tripped it), GPU_POWER_FILE (default /etc/default/gpu-power, then /etc/default/gpu-power-limit),
#      UNLOCK_CLOCK_OK=1 (you lock clocks some other way: skips the guard).
set -uo pipefail
GUARD=0; [ "${1:-}" = --guard ] && GUARD=1
MAXC="${UNLOCK_MAX_CLOCK:-1400}"
PF="${GPU_POWER_FILE:-}"
if [ -z "$PF" ]; then for f in /etc/default/gpu-power /etc/default/gpu-power-limit; do [ -r "$f" ] && { PF="$f"; break; }; done; fi
lock=0
if [ -n "$PF" ] && [ -r "$PF" ]; then lock="$(sed -n 's/^GPU_MAX_CLOCK=\([0-9]*\).*/\1/p' "$PF" | tail -1)"; lock="${lock:-0}"; fi
svc_active=0
for u in gpu-power gpu-power-limit; do systemctl is-active --quiet "$u" 2>/dev/null && svc_active=1; done

module=0; [ -d /sys/module/gpu_unlock ] && module=1
cmdline_opt=0; grep -q 'gpu_unlock.targets=' /proc/cmdline && cmdline_opt=1
klog="$( { journalctl -k -b --no-pager 2>/dev/null || dmesg 2>/dev/null; } | grep 'gpu_unlock' || true)"
log_has_module=0; [ -n "$klog" ] && log_has_module=1

n=0; n_unlocked=0; rows=""
while IFS=, read -r idx bus name pl; do
  [ -n "$idx" ] || continue
  n=$((n + 1))
  b="$(echo "$bus" | tr -d ' ' | tr 'A-F' 'a-f' | sed 's/^0000\(0000:\)/\1/; s/^00000000:/0000:/')"
  if echo "$klog" | grep -qi "${b#0000:}: gpu_unlock: unlocked"; then st="UNLOCKED"; n_unlocked=$((n_unlocked + 1))
  elif [ "$module" = 0 ] && [ "$cmdline_opt" = 0 ]; then st="stock"
  elif [ "$log_has_module" = 1 ]; then st="stock (module present, card not unlocked this boot)"
  else st="unknown (kernel log unreadable: add your user to the adm group)"; fi
  rows+="$(printf '  GPU%s %s %-26s cap %-9s BF16: %s' "$idx" "$b" "$(echo $name)" "$(echo $pl) W" "$st")"$'\n'
done < <(nvidia-smi --query-gpu=index,pci.bus_id,name,power.limit --format=csv,noheader,nounits 2>/dev/null)

# unlocked = proven by the log; maybe = the module could be active but the log cannot tell
maybe=0
if [ "$n_unlocked" -eq 0 ] && { [ "$module" = 1 ] || [ "$cmdline_opt" = 1 ]; } && [ "$log_has_module" = 0 ]; then maybe=1; fi
if [ "$n_unlocked" -gt 0 ]; then summary="$n_unlocked/$n cards unlocked"
elif [ "$maybe" = 1 ]; then summary="unknown (module present, kernel log unreadable): treated as unlocked"
elif [ "$cmdline_opt" = 1 ]; then summary="not active this boot (gpu_unlock.targets is set: power the machine fully off, not a reboot)"
else summary="not active"; fi
lock_ok=0; [ "$svc_active" = 1 ] && [ "$lock" -gt 0 ] && [ "$lock" -le "$MAXC" ] && lock_ok=1

if [ "$GUARD" = 0 ]; then
  echo "GPUs: $n"
  printf '%s' "$rows"
  p2p_ok="$(nvidia-smi topo -p2p r 2>/dev/null | grep -E '^[[:space:]]*GPU[0-9]+[[:space:]]+(X|OK|NS|CNS|TNS|NA)' | grep -oE '\b(OK|NS|CNS|TNS|NA)\b' | sort | uniq -c | xargs)"
  echo "P2P read between card pairs: ${p2p_ok:-unknown} (OK on every pair = peer-to-peer works; docs/swarm/recipe.md#p2p)"
  echo "clock lock: GPU_MAX_CLOCK=${lock} in ${PF:-<no gpu-power file>}; power service $([ $svc_active = 1 ] && echo active || echo not active)"
  if [ "$n_unlocked" -gt 0 ] || [ "$maybe" = 1 ]; then echo "BF16 unlock: $summary$([ $lock_ok = 1 ] && echo ", ${lock} MHz clock lock")"
  else echo "BF16 unlock: $summary"; fi
fi

if [ "$n_unlocked" -gt 0 ] || [ "$maybe" = 1 ]; then
  [ "${UNLOCK_CLOCK_OK:-0}" = 1 ] && exit 0
  [ "$lock_ok" = 1 ] && exit 0
  echo "UNSAFE: BF16 unlock $summary, and no active power service locks the clock at <= ${MAXC} MHz."
  echo "  Unlocked cards draw their transient peaks together; our 1500 W PSU tripped (whole box off) at 1455 and 1650 MHz."
  echo "  Fix: install systemd/gpu-power.service with GPU_MAX_CLOCK=${MAXC} (systemd/gpu-power.example), then"
  echo "       sudo systemctl enable --now gpu-power.service"
  echo "  A bigger PSU: UNLOCK_MAX_CLOCK=<MHz>. Locked some other way: UNLOCK_CLOCK_OK=1. Details: docs/swarm/gpu-unlock.md"
  exit 1
fi
exit 0
