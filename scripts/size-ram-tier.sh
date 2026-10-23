#!/usr/bin/env bash
# Print the RAM_TIER_GIB line for this machine (docs/swarm/ram-sizing.md); hints go to the terminal (stderr).
#   scripts/size-ram-tier.sh [OTHER_GIB] >> config/local.env      (setup.sh does this for you)
# OTHER_GIB: RAM the rest of your machine needs (databases, containers, desktop); default 0.
#
# RAM tier = visible RAM - 49 GiB (n-gram table, FP8, never swapped) - 32 GiB (vLLM processes ~17, OS, first-boot compile and page-cache headroom)
#            - OTHER_GIB, rounded down to a multiple of 2 GiB, at most 130 GiB (the size we measured).
#            /dev/shm must hold the tier + 5 GiB.
set -euo pipefail
other="${1:-0}"; tested=130
total_gib=$(( $(awk '/^MemTotal:/{print $2}' /proc/meminfo) / 1048576 ))
rule=$(( (total_gib - 49 - 32 - other) / 2 * 2 ))
tier=$rule; [ "$tier" -gt "$tested" ] && tier=$tested
shm_gib=$(( $(df -k --output=size /dev/shm | tail -1) / 1048576 ))
echo "# visible RAM ${total_gib} GiB, other uses ${other} GiB (scripts/size-ram-tier.sh, $(date -I))"
if [ "$rule" -lt 4 ]; then
  echo "RAM_TIER_GIB=0"
  echo "no room for a RAM tier: the n-gram table (49 GiB) and the server need about 84 GiB of visible RAM (docs/swarm/ram-sizing.md)" >&2
else
  echo "RAM_TIER_GIB=$tier"
  [ "$rule" -gt "$tested" ] && echo "note: the rule allows ${rule} GiB; ${tested} GiB is the size we measured. Set RAM_TIER_GIB=${rule} to use more." >&2
  need=$(( tier + 5 ))
  if [ "$shm_gib" -lt "$need" ]; then
    echo "/dev/shm is ${shm_gib} GiB; it needs >= ${need} GiB. In /etc/fstab:" >&2
    echo "  tmpfs /dev/shm tmpfs defaults,size=${need}G 0 0      then: sudo mount -o remount /dev/shm" >&2
  fi
fi
[ "$total_gib" -ge 84 ] || echo "warning: ${total_gib} GiB visible RAM is below the ~84 GiB the server needs (docs/swarm/ram-sizing.md)" >&2
