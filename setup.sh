#!/usr/bin/env bash
# One-command install for flash-next-swarm-quad3090.
#
#   git clone --branch swarm-v1.0 https://github.com/ththeod/flash-next-swarm-quad3090.git && cd flash-next-swarm-quad3090
#   ./setup.sh
#
# Checks the machine, shows every download with its size and asks once. Then it builds the engine (halt95's v2.5.2 build
# script, unchanged, plus this fork's delta, both verified by commit and tree hash), downloads the checkpoint and the
# drafter at pinned revisions and checks them against pinned SHA-256 lists, sizes the RAM tier and writes
# config/local.env. Safe to re-run: finished steps are verified, not redone, and your local.env settings are kept.
# It does not touch BIOS, drivers, peer-to-peer, power settings or /dev/shm (it tells you when one needs attention),
# installs nothing system-wide unless you pass --systemd, and starts nothing unless you pass --start.
# Disk: about 125 GiB for the models and 16 GiB for the engine.
#
# Options:
#   --yes              no questions
#   --dry-run          checks and plan only; downloads and writes nothing
#   --models-dir DIR   where the model folders go (default ./models)
#   --ckpt DIR         use an existing Swift 1.5 Merlin checkpoint folder (verified, not re-downloaded)
#   --drafter DIR      use an existing incoai/Qwen3.8-Flash-Next-DFlash2 folder (verified)
#   --other-ram GIB    RAM other software on this machine needs (default 0); lowers the RAM tier
#   --ram-tier GIB     set the RAM KV tier yourself (default: from the RAM rule, at most 130 GiB, the size we measured)
#   --host HOST        127.0.0.1 (default, this machine only) or 0.0.0.0 (network; an API key is generated)
#   --port PORT        default 8000
#   --api-key KEY      API key the server requires
#   --build-own        compile the two engine extensions here (cmake >= 3.26, nvcc >= 13.0; 15-30 min)
#   --systemd          also install systemd/flash-next-swarm.service for this user and path (uses sudo)
#   --start            start the server at the end (foreground; Ctrl+C stops it)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$HERE"

YES=0 DRY=0 MODELS="$HERE/models" CKPT="" DRAFTER_SRC="" OTHER=0 TIER_SET="" HOST="" PORT="" API_KEY="" BUILD_OWN=0 SYSTEMD=0 START=0
while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) YES=1 ;;
    --dry-run) DRY=1 ;;
    --models-dir) MODELS="$2"; shift ;;
    --ckpt) CKPT="$2"; shift ;;
    --drafter) DRAFTER_SRC="$2"; shift ;;
    --other-ram) OTHER="$2"; shift ;;
    --ram-tier) TIER_SET="$2"; shift ;;
    --host) HOST="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --api-key) API_KEY="$2"; shift ;;
    --build-own) BUILD_OWN=1 ;;
    --systemd) SYSTEMD=1 ;;
    --start) START=1 ;;
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (--help lists them)"; exit 1 ;;
  esac
  shift
done
# a re-run keeps the network settings already in config/local.env unless a flag changes them
prev(){ [ -f config/local.env ] && sed -n "s/^$1=//p" config/local.env | tail -1 || true; }
HOST="${HOST:-$(prev HOST)}"; HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-$(prev PORT)}"; PORT="${PORT:-8000}"
[ -n "$API_KEY" ] || API_KEY="$(prev API_KEY)"
abs(){ case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$HERE/$1" ;; esac; }
CKPT_GIVEN=0; [ -n "$CKPT" ] && CKPT_GIVEN=1
MODELS="$(abs "$MODELS")"
CKPT="$(abs "${CKPT:-$MODELS/Swift1.5-Merlin}")"
DRAFTER_SRC="$(abs "${DRAFTER_SRC:-$MODELS/DFlash2}")"
DRAFTER="$MODELS/DFlash2-swiftembed"
CKPT_REPO=halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin; CKPT_REV=f3a8182d39702b7637b45a23c613da72d9ab800c
DRAFT_REPO=incoai/Qwen3.8-Flash-Next-DFlash2; DRAFT_REV=a22a1744e85c17364dcc54a6a420af7135d4c7f6
CKPT_SUMS="$HERE/release/swarm-v1.0/checkpoint.sha256"; DRAFT_SUMS="$HERE/release/swarm-v1.0/drafter.sha256"
STATE="$HERE/.setup"; TREE="$HERE/vllm-swarm"; VENV="$HERE/venv-swarm"
TESTED_TIER=130

fails=0 warnings=()
ok(){ printf '  ok    %s\n' "$*"; }
warn(){ printf '  WARN  %s\n' "$*"; warnings+=("$*"); }
fail(){ printf '  FAIL  %s\n' "$*"; fails=$((fails + 1)); }
fix(){ printf '        fix: %s\n' "$*"; }
gib_free(){ local d="$1"; while [ ! -e "$d" ]; do d="$(dirname "$d")"; done; echo $(( $(df -k --output=avail "$d" | tail -1) / 1048576 )); }
pinned_ok(){ (cd "$1" && grep -v '^#' "$2" | sha256sum -c --quiet --strict >/dev/null 2>&1); }   # dir, list

echo "flash-next-swarm-quad3090 v1.0 setup ($HERE)"
echo "Checks:"
# --- platform and tools ---
[ "$(uname -s)/$(uname -m)" = Linux/x86_64 ] && ok "Linux x86-64" || fail "Linux x86-64 only (this is $(uname -s)/$(uname -m))"
missing=""; for t in git curl tar gzip sha256sum cc c++ ninja python3; do command -v "$t" >/dev/null || missing="$missing $t"; done
if [ -z "$missing" ]; then ok "tools: git curl tar gzip sha256sum cc c++ ninja python3"
else fail "missing tools:$missing"; fix "Debian/Ubuntu: sudo apt install git curl tar gzip coreutils build-essential ninja-build python3"; fi
PY313="$(command -v python3.13 || true)"
if [ -z "$PY313" ] && command -v uv >/dev/null; then PY313="$(uv python find 3.13 2>/dev/null || true)"; fi
if [ -n "$PY313" ] && "$PY313" -c 'import venv, sysconfig, os; assert os.path.exists(sysconfig.get_paths()["include"] + "/Python.h")' 2>/dev/null; then
  ok "Python 3.13: $PY313"; export PATH="$(dirname "$PY313"):$PATH"
else
  fail "Python 3.13 with venv and headers not found"
  fix "uv python install 3.13 (https://docs.astral.sh/uv/), or on Ubuntu the deadsnakes PPA: python3.13 python3.13-venv python3.13-dev"
fi
glibc="$(ldd --version 2>/dev/null | sed -n '1s/.* \([0-9][0-9]*\.[0-9][0-9]*\)$/\1/p' || true)"
if [ "$BUILD_OWN" = 1 ]; then
  cm="$(cmake --version 2>/dev/null | sed -n 's/^cmake version \([0-9][0-9.]*\).*/\1/p' || true)"
  if [ -n "$cm" ] && [ "$(printf '%s\n3.26\n' "$cm" | sort -V | head -1)" = 3.26 ]; then ok "--build-own: cmake $cm, glibc ${glibc:-?}"
  else fail "--build-own needs cmake >= 3.26 (found ${cm:-none})"; fix "pip install 'cmake>=3.26,<4' and put it first on PATH"; fi
elif [ -n "$glibc" ] && [ "$(printf '%s\n2.38\n' "$glibc" | sort -V | head -1)" = 2.38 ]; then ok "glibc $glibc (>= 2.38 for the prebuilt extension)"
else fail "glibc ${glibc:-unknown} is older than 2.38 (the prebuilt MoE extension needs it)"; fix "./setup.sh --build-own (needs cmake >= 3.26; 15-30 min)"; fi
# --- GPUs, driver, power ---
if ! command -v nvidia-smi >/dev/null; then
  fail "nvidia-smi not found"; fix "install an NVIDIA driver with CUDA 13.0 or newer (R580+) and reboot"
elif ! nvidia-smi -L >/dev/null 2>&1; then
  fail "nvidia-smi cannot talk to the driver"; fix "after a kernel update, rebuild the driver modules (the P2P build too: docs/swarm/recipe.md#p2p) and reboot"
else
  cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9][0-9.]*\).*/\2/p' | head -1 || true)"
  if [ -n "$cuda" ] && [ "$(printf '%s\n13.0\n' "$cuda" | sort -V | head -1)" = 13.0 ]; then ok "NVIDIA driver, CUDA $cuda"
  else fail "driver CUDA version ${cuda:-unknown} < 13.0"; fix "update the NVIDIA driver (R580 or later)"; fi
  ngpu="$(nvidia-smi -L | wc -l || true)"; n3090="$(nvidia-smi -L | grep -c 'RTX 3090' || true)"
  if [ "$n3090" -eq 4 ] && [ "$ngpu" -eq 4 ]; then ok "4 x RTX 3090"
  else fail "found $ngpu GPU(s), $n3090 of them RTX 3090: v1.0 supports exactly four RTX 3090s"; fi
  busy="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1024{n++} END{print n+0}' || true)"
  [ "${busy:-0}" -eq 0 ] && ok "GPUs idle" || warn "${busy} GPU(s) already hold more than 1 GiB; stop that before starting the server"
  p2p_bad="$(nvidia-smi topo -p2p r 2>/dev/null | grep -E '^[[:space:]]*GPU[0-9]' | grep -oE '\b(NS|CNS|TNS|NA)\b' | wc -l || true)"
  [ "${p2p_bad:-1}" -eq 0 ] && ok "peer-to-peer between all cards" || { warn "no peer-to-peer between some cards: this fork was tested only with P2P"; fix "docs/swarm/recipe.md#p2p"; }
  at_default="$(nvidia-smi --query-gpu=power.limit,power.default_limit --format=csv,noheader,nounits | awk -F', ' '$1+0>=$2+0{n++} END{print n+0}' || true)"
  if [ "${at_default:-0}" -gt 0 ]; then warn "$at_default card(s) at the stock power limit: four 3090s at full power can trip a PSU (stock cards: 220 W, halt95's reference; unlocked: 240 W + a 1400 MHz lock)"; fix "systemd/gpu-power.example and docs/swarm/power-and-clocks.md"
  else ok "power caps below stock ($(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | awk '{printf "%d ", $1}')W)"; fi
fi
# --- RAM, /dev/shm, disk ---
ram_gib=$(( $(awk '/^MemTotal:/{print $2}' /proc/meminfo) / 1048576 ))
rule=$(( (ram_gib - 49 - 32 - OTHER) / 2 * 2 ))
if [ -n "$TIER_SET" ]; then tier="$TIER_SET"; tier_note="set with --ram-tier"
elif [ "$rule" -gt "$TESTED_TIER" ]; then tier=$TESTED_TIER; tier_note="the rule allows ${rule}; ${TESTED_TIER} is the size we measured (--ram-tier ${rule} to use more)"
else tier=$rule; tier_note="from the RAM rule"; fi
if [ "$ram_gib" -lt 84 ]; then fail "${ram_gib} GiB RAM visible; the n-gram table (49 GiB) and the server need about 84 GiB"; fix "docs/swarm/ram-sizing.md"; tier=0
elif [ "$tier" -lt 4 ]; then warn "${ram_gib} GiB RAM: no room for a RAM KV tier (RAM_TIER_GIB=0; agent follow-ups are recomputed)"; tier=0
else ok "${ram_gib} GiB RAM: RAM KV tier ${tier} GiB (${tier_note}; docs/swarm/ram-sizing.md)"; fi
shm_gib=$(( $(df -k --output=size /dev/shm | tail -1) / 1048576 ))
if [ "$tier" -gt 0 ] && [ "$shm_gib" -lt $((tier + 5)) ]; then
  warn "/dev/shm is ${shm_gib} GiB; the ${tier} GiB tier needs $((tier + 5)) GiB (the server will not start until then)"
  fix "add to /etc/fstab: tmpfs /dev/shm tmpfs defaults,size=$((tier + 5))G 0 0   then: sudo mount -o remount /dev/shm"
else ok "/dev/shm ${shm_gib} GiB"; fi
ckpt_ok=0; [ -f "$STATE/ckpt-verified" ] && [ "$(cat "$STATE/ckpt-verified")" = "$CKPT $(sha256sum "$CKPT_SUMS" | cut -c1-16)" ] && ckpt_ok=1
need_models=0
[ "$ckpt_ok" = 1 ] || [ "$CKPT_GIVEN" = 1 ] || need_models=$((need_models + 116))
[ -f "$DRAFTER_SRC/model.safetensors" ] || need_models=$((need_models + 2))
[ -f "$DRAFTER/embed_tokens.safetensors" ] || need_models=$((need_models + 3))
free_m="$(gib_free "$MODELS")"; free_r="$(gib_free "$HERE")"
[ "$free_m" -ge "$need_models" ] && ok "disk for models: ${free_m} GiB free, up to ${need_models} GiB needed ($MODELS)" || fail "disk for models: ${free_m} GiB free, ${need_models} GiB needed ($MODELS)"
[ -x "$VENV/bin/python" ] && need_r=1 || need_r=16
[ "$free_r" -ge "$need_r" ] && ok "disk for the engine: ${free_r} GiB free, ${need_r} GiB needed" || fail "disk for the engine: ${free_r} GiB free, ${need_r} GiB needed"
# --- unlock (third-party, optional) ---
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  if scripts/gpu-status.sh --guard >/dev/null 2>&1; then
    st="$(scripts/gpu-status.sh 2>/dev/null | sed -n 's/^BF16 unlock: //p' | head -1 || true)"
    ok "BF16 unlock: ${st:-not active} (optional, third-party: docs/swarm/gpu-unlock.md)"
  else
    fail "BF16 unlock possibly active without a safe clock lock"; scripts/gpu-status.sh --guard | sed 's/^/        /' || true
  fi
fi
# --- network ---
if [ "$HOST" != 127.0.0.1 ] && [ -z "$API_KEY" ]; then
  API_KEY="$(head -c 24 /dev/urandom | base64 | tr -d '/+=')"
  ok "HOST=$HOST PORT=$PORT: an API key will be generated (config/local.env, mode 600)"
else ok "HOST=$HOST PORT=$PORT$([ -n "$API_KEY" ] && echo ', API key set')"; fi

if [ "$fails" -gt 0 ]; then echo "$fails check(s) failed; nothing was downloaded or changed."; exit 1; fi

# --- plan ---
echo "Plan:"
H25=https://github.com/halt95/merlin-engine/releases/download
built=0; [ -f "$TREE/.swarm-build" ] && built=1
if [ "$built" = 0 ]; then
  [ -f v2.5.2-from-upstream-v0.30.0.bundle.gz ] || [ -f v2.5.2-from-upstream-v0.30.0.bundle ] || echo "  download  halt95 v2.5.2 source bundle (1 MB)"
  echo "  download  vLLM v0.30.0, one commit, shallow (from github.com/vllm-project/vllm)"
  [ -f build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz ] || echo "  download  halt95 v2.2.0 build products (229 MB)"
  [ "$BUILD_OWN" = 1 ] || [ -f moe-C-sm86-py313-cu134-swarm-v1.0.tar.gz ] || echo "  download  this release's MoE extension (18 MB)"
  [ -x "$VENV/bin/python" ] || echo "  download  Python packages into venv-swarm/ (about 10 GB installed: PyTorch, CUDA 13.0 wheels, FlashInfer)"
  echo "  build     scripts/build-swarm.sh vllm-swarm venv-swarm$([ "$BUILD_OWN" = 1 ] && echo ', compiling both extensions (15-30 min)')"
else echo "  verify    the existing engine build"; fi
if [ "$ckpt_ok" = 1 ]; then echo "  keep      checkpoint $CKPT (verified earlier)"
elif [ "$CKPT_GIVEN" = 1 ]; then echo "  verify    checkpoint $CKPT against release/swarm-v1.0/checkpoint.sha256 (~115 GiB read)"
else echo "  download  $CKPT_REPO @ ${CKPT_REV:0:12} -> $CKPT (115 GiB; resumes), then verify"; fi
[ -f "$DRAFTER_SRC/model.safetensors" ] && echo "  verify    drafter $DRAFTER_SRC" || echo "  download  $DRAFT_REPO @ ${DRAFT_REV:0:12} -> $DRAFTER_SRC (1.2 GB), then verify"
[ -f "$DRAFTER/embed_tokens.safetensors" ] || echo "  write     $DRAFTER (drafter + the checkpoint's input embedding, 2.5 GB)"
echo "  write     config/local.env (paths, RAM_TIER_GIB=$tier, HOST=$HOST, PORT=$PORT$([ -n "$API_KEY" ] && echo ', API_KEY'))"
[ "$SYSTEMD" = 1 ] && echo "  install   /etc/systemd/system/flash-next-swarm.service (sudo; enabled, not started)"
[ "$START" = 1 ] && echo "  start     scripts/run-swarm.sh (the first start compiles kernels for about 10 minutes)"
[ "$DRY" = 1 ] && { echo "dry run: nothing downloaded or written."; exit 0; }
if [ "$YES" = 0 ]; then read -r -p "Proceed? [Y/n] " a; case "$a" in n|N|no) echo "stopped; nothing changed."; exit 0 ;; esac; fi

mkdir -p "$STATE" "$MODELS"
fetch(){ [ -f "$2" ] || { curl -fL --retry 3 -# -o "$2.part" "$1" && mv "$2.part" "$2"; }; }
if [ "$built" = 0 ]; then
  fetch "$H25/v2.5.2/v2.5.2-from-upstream-v0.30.0.bundle.gz" v2.5.2-from-upstream-v0.30.0.bundle.gz
  fetch "$H25/v2.2.0/build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz" build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz
  if [ "$BUILD_OWN" = 0 ]; then . upstream/PIN-swarm-v1.0; fetch "$moe_C_url" "$moe_C_tar"; fi
fi
BUILD_OWN="$BUILD_OWN" scripts/build-swarm.sh "$TREE" "$VENV"
HF="$VENV/bin/hf"
if [ "$ckpt_ok" = 0 ]; then
  [ "$CKPT_GIVEN" = 1 ] || "$HF" download "$CKPT_REPO" --revision "$CKPT_REV" --local-dir "$CKPT"   # skips complete files
  echo "verifying the checkpoint against release/swarm-v1.0/checkpoint.sha256 (40 files; a few minutes)"
  if ! (cd "$CKPT" && grep -v '^#' "$CKPT_SUMS" | sha256sum -c --quiet --strict); then
    echo "checkpoint verification failed. Run ./setup.sh again (the download resumes and replaces damaged files)$([ "$CKPT_GIVEN" = 1 ] && echo ', or drop --ckpt to download a fresh copy')."; exit 1
  fi
  echo "$CKPT $(sha256sum "$CKPT_SUMS" | cut -c1-16)" > "$STATE/ckpt-verified"
fi
[ -f "$DRAFTER_SRC/model.safetensors" ] || "$HF" download "$DRAFT_REPO" --revision "$DRAFT_REV" --local-dir "$DRAFTER_SRC"
pinned_ok "$DRAFTER_SRC" "$DRAFT_SUMS" || { echo "drafter files in $DRAFTER_SRC do not match release/swarm-v1.0/drafter.sha256"; exit 1; }
[ -f "$DRAFTER/embed_tokens.safetensors" ] || "$VENV/bin/python" scripts/graft-drafter-embed.py "$DRAFTER_SRC" "$CKPT" "$DRAFTER"

umask 077
managed="TREE VENV CKPT DRAFTER CACHE_ROOT RAM_TIER_GIB HOST PORT API_KEY"
{
  echo "# written by setup.sh $(date -I); settings here override config/swarm.env. Lines setup.sh does not manage are kept."
  echo "TREE=$TREE"; echo "VENV=$VENV"; echo "CKPT=$CKPT"; echo "DRAFTER=$DRAFTER"; echo "CACHE_ROOT=$HERE/cache"
  echo "RAM_TIER_GIB=$tier"; echo "HOST=$HOST"; echo "PORT=$PORT"
  [ -z "$API_KEY" ] || echo "API_KEY=$API_KEY"
  if [ -f config/local.env ]; then grep -vE "^(# written by setup.sh|($(echo $managed | tr ' ' '|'))=)" config/local.env || true; fi
} > config/local.env.new && mv config/local.env.new config/local.env
umask 022
echo "wrote config/local.env"
[ -z "$API_KEY" ] || echo "API key (also in config/local.env): $API_KEY"

if [ "$SYSTEMD" = 1 ]; then
  sed -e "s#^User=.*#User=$(id -un)#; s#^Group=.*#Group=$(id -gn)#; s#/opt/flash-next-swarm-quad3090#$HERE#g" \
    systemd/flash-next-swarm.service > "$STATE/flash-next-swarm.service"
  sudo install -m 644 "$STATE/flash-next-swarm.service" /etc/systemd/system/flash-next-swarm.service
  sudo systemctl daemon-reload && sudo systemctl enable flash-next-swarm.service
  echo "installed flash-next-swarm.service (enabled; start: sudo systemctl start flash-next-swarm)"
fi

echo
if [ "${#warnings[@]}" -gt 0 ]; then echo "Open warnings:"; for w in "${warnings[@]}"; do echo "  - $w"; done; echo; fi
echo "Done. Start:   scripts/run-swarm.sh   (or the systemd service; ready at 'Application startup complete')"
echo "      Check:   bench/check.sh          (about 10 minutes; compare with docs/swarm/results.md)"
echo "      Status:  scripts/gpu-status.sh"
[ "$START" = 1 ] && exec scripts/run-swarm.sh
exit 0
