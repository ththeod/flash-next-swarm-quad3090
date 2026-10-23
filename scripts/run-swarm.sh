#!/usr/bin/env bash
# Start the server with config/swarm.env + config/local.env (the same command systemd/flash-next-swarm.service runs).
#   scripts/run-swarm.sh [extra vllm args]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
set -a
# shellcheck disable=SC1091
. config/swarm.env
# shellcheck disable=SC1091
[ -f config/local.env ] && . config/local.env
set +a
# an empty value means "use the default": unset it (vLLM rejects VLLM_MARLIN_INPUT_DTYPE="")
for v in VLLM_MARLIN_INPUT_DTYPE NCCL_PROTO_OVERRIDE; do [ -n "${!v:-}" ] || unset "$v"; done
abs(){ case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$HERE/${1#./}" ;; esac; }
TREE="$(abs "$TREE")"; VENV="$(abs "$VENV")"; CKPT="$(abs "$CKPT")"; DRAFTER="$(abs "$DRAFTER")"
CACHE_ROOT="$(abs "$CACHE_ROOT")"; VLLM_EXP_SKINNY_SM86_PLANS="$(abs "$VLLM_EXP_SKINNY_SM86_PLANS")"
export TREE VENV CKPT CACHE_ROOT VLLM_EXP_SKINNY_SM86_PLANS
export SCALES="${SCALES:-$CKPT/qsa_kv_scales_swift.json}"
[ -f "$SCALES" ] || { echo "KV scales not found: $SCALES"; exit 1; }
[ -f "$DRAFTER/embed_tokens.safetensors" ] || { echo "$DRAFTER has no embed_tokens.safetensors: run scripts/graft-drafter-embed.py"; exit 1; }

# the RAM tier lives in /dev/shm: it must hold the tier plus a few GiB
shm_gib=$(( $(df -k --output=size /dev/shm | tail -1) / 1048576 ))
[ "$shm_gib" -ge $(( RAM_TIER_GIB + 5 )) ] || { echo "/dev/shm is ${shm_gib} GiB; RAM_TIER_GIB=$RAM_TIER_GIB needs >= $(( RAM_TIER_GIB + 5 )) (docs/swarm/ram-sizing.md)"; exit 1; }

# unlocked cards need a clock lock (docs/swarm/gpu-unlock.md); refuse to start without one
"$HERE/scripts/gpu-status.sh" --guard || exit 1

export SPEC_CONFIG="{\"method\":\"dflash\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\",\"kv_cache_dtype\":\"auto\",\"num_speculative_tokens_per_batch_size\":$SPEC_DRAFTS}"
args=(--max-num-seqs "$MAX_NUM_SEQS" --kv-cache-memory "$KV_CACHE_MEMORY")
if [ "${RAM_TIER_GIB:-0}" -gt 0 ]; then
  args+=(--kv-offloading-backend native --kv-offloading-size "$RAM_TIER_GIB"
         --kv-transfer-config '{"kv_connector_extra_config":{"eviction_policy":"arc"}}')
fi
if [ -n "${REASONING_EFFORT:-}" ]; then
  args+=(--default-chat-template-kwargs "{\"enable_thinking\": true, \"reasoning_effort\": \"$REASONING_EFFORT\"}")
fi
# HOST other than 127.0.0.1 exposes the API to the network: give it a key (setup.sh generates one)
# (passed as VLLM_API_KEY, not --api-key, so it does not show in the process list)
if [ -n "${API_KEY:-}" ]; then export VLLM_API_KEY="$API_KEY"; unset API_KEY
elif [ "${HOST:-127.0.0.1}" != 127.0.0.1 ]; then echo "warning: HOST=$HOST with no API_KEY: the API is open to the network"; fi
exec "$HERE/scripts/serve-swarm.sh" "$CKPT" "${args[@]}" "$@"
