#!/usr/bin/env bash
# Quick checks after install, against the running server (about 10 minutes):
#   garbage / runaway gate (8 HumanEval problems, thinking on), NaN probe, generation at 1 / 8 / 16 requests,
#   prefill and needles at 10K and 50K.
#   bench/check.sh
# First use creates bench/.venv (openai, pandas, pyarrow, numpy). The server URL, port and API key come from
# config/local.env; override with LLM_BASE_URL / LLM_API_KEY. Reference numbers: docs/swarm/results.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$HERE"
val(){ sed -n "s/^$1=//p" ../config/local.env 2>/dev/null | tail -1; }
PORT="$(val PORT)"; KEY="$(val API_KEY)"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:${PORT:-8000}/v1}" LLM_API_KEY="${LLM_API_KEY:-$KEY}" LLM_MODEL="${LLM_MODEL:-flash-next-v2}"
if [ ! -x .venv/bin/python ]; then
  echo "creating bench/.venv"; python3 -m venv .venv && .venv/bin/pip install -q openai pandas pyarrow numpy
fi
curl -sf -m 10 ${LLM_API_KEY:+-H "Authorization: Bearer $LLM_API_KEY"} "$LLM_BASE_URL/models" >/dev/null \
  || { echo "no server at $LLM_BASE_URL (start it with scripts/run-swarm.sh and wait for 'Application startup complete')"; exit 1; }
echo "== garbage / runaway gate (expect: no runaway, no garbage)"; N=8 .venv/bin/python runaway_probe.py
echo "== NaN probe";                                         .venv/bin/python nan_poke.py
echo "== generation, 1 / 8 / 16 requests";                   .venv/bin/python concurrency.py 1 8 16
echo "== prefill and needles, 10K / 50K";                    .venv/bin/python longctx.py 10000 50000
echo "done: compare with docs/swarm/results.md"
