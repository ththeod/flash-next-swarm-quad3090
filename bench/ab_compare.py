#!/usr/bin/env python3
"""Side-by-side table of two runs of the same suite (fp4-cycle.sh): concurrency, long context, quality.
usage: ab_compare.py <tag A> <tag B> [effort, default medium]   (reads results/<tag>_concurrency.json, _longctx.json, _<bench>_medium*.json)"""
import glob, json, os, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
A, B = sys.argv[1], sys.argv[2]
EFFORT = sys.argv[3] if len(sys.argv) > 3 else "medium"


def load(tag, name):
    hits = sorted(glob.glob(os.path.join(R, f"{tag}_{name}*.json")), key=os.path.getmtime)
    return json.load(open(hits[-1])) if hits else None


def pct(a, b):
    return f"{(b / a - 1) * 100:+.0f}%" if a and b else ""


print(f"## Concurrency (thinking off, 512 tokens): {A} vs {B}\n")
print("| Requests | A agg tok/s | B agg tok/s | change | A per-request | B per-request | A TTFT p50 | B TTFT p50 |")
print("|---|---|---|---|---|---|---|---|")
ca, cb = load(A, "concurrency") or [], load(B, "concurrency") or []
for ra in ca:
    rb = next((x for x in cb if x["concurrency"] == ra["concurrency"]), {})
    print(f"| {ra['concurrency']} | {ra['aggregate_tps']:.0f} | {rb.get('aggregate_tps', 0):.0f} | "
          f"{pct(ra['aggregate_tps'], rb.get('aggregate_tps'))} | {ra['per_stream_decode_tps_mean']:.0f} | "
          f"{rb.get('per_stream_decode_tps_mean', 0):.0f} | {ra['ttft_p50']:.2f} s | {rb.get('ttft_p50', 0):.2f} s |")

print(f"\n## Long context (unique prefix, needle at 10/50/90%)\n")
print("| Target | A prefill tok/s | B prefill tok/s | change | A needles | B needles |")
print("|---|---|---|---|---|---|")
la, lb = load(A, "longctx") or [], load(B, "longctx") or []
for size in sorted({r["target_tokens"] for r in la + lb}):
    ra = [r for r in la if r["target_tokens"] == size]
    rb = [r for r in lb if r["target_tokens"] == size]
    pa = sum(r.get("prefill_tps") or 0 for r in ra) / max(len(ra), 1)
    pb = sum(r.get("prefill_tps") or 0 for r in rb) / max(len(rb), 1)
    print(f"| {size // 1000}K | {pa:,.0f} | {pb:,.0f} | {pct(pa, pb)} | {sum(bool(r.get('found')) for r in ra)}/{len(ra)} "
          f"| {sum(bool(r.get('found')) for r in rb)}/{len(rb)} |")

print(f"\n## Quality (thinking on, {EFFORT} effort, recommended sampling)\n")
print("| Benchmark | A score | B score | A tokens/answer | B tokens/answer | A truncated | B truncated |")
print("|---|---|---|---|---|---|---|")
for bench in ("tools", "gsm8k", "humaneval", "mmlu_pro"):
    qa, qb = load(A, f"{bench}_{EFFORT}"), load(B, f"{bench}_{EFFORT}")
    sa, sb = (qa or {}).get("summary", {}), (qb or {}).get("summary", {})
    tok = lambda s: f"{s['completion_tokens'] / s['n']:.0f}" if s.get("n") and s.get("completion_tokens") else "–"
    sc = lambda s: f"{s['correct']}/{s['n']} = {s['accuracy'] * 100:.1f}%" if s.get("n") else "–"
    print(f"| {bench} | {sc(sa)} | {sc(sb)} | {tok(sa)} | {tok(sb)} | {sa.get('truncated', '–')} | {sb.get('truncated', '–')} |")
