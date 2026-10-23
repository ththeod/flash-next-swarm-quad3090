#!/usr/bin/env python3
"""Teacher-forced log-likelihood on long local documents, for KV-cache precision A/B (FP8 vs FP4 fake quant).

Sends each document as exact prompt token ids to /v1/completions with prompt_logprobs=1 and max_tokens=1, so every
position's next-token distribution is computed through the paged (quantized) KV cache. Saves, per document, the
actual-token logprob and whether the actual token was the model's top-1 at every position, so two runs can be compared
position by position (compare: kv_ppl.py --compare TAG_A TAG_B).

usage: kv_ppl.py [TOKENS_PER_DOC=65536]      (env LLM_TAG names the run)
       kv_ppl.py --compare TAG_A TAG_B
"""
import glob, json, math, os, sys, urllib.request
import numpy as np
from common import BASE_URL, MODEL, RESULTS, save

BUCKETS = [(0, 4096), (4096, 16384), (16384, 65536), (65536, 1 << 30)]


def docs():
    """Two code documents and two prose documents, all local files (no private data)."""
    code1 = sorted(glob.glob(os.environ.get("DOC_TREE", "../vllm-swarm") + "/vllm/v1/core/**/*.py", recursive=True))
    code2 = sorted(glob.glob(os.environ.get("DOC_TREE", "../vllm-swarm") + "/vllm/model_executor/layers/**/*.py", recursive=True))
    prose = sorted(glob.glob(os.environ.get("DOC_TREE", "../vllm-swarm") + "/docs/**/*.md", recursive=True))
    cat = lambda fs: "\n\n".join(open(f, errors="replace").read() for f in fs)
    half = len(prose) // 2
    return {"code-core": cat(code1), "code-layers": cat(code2), "prose-docs-a": cat(prose[:half]),
            "prose-docs-b": cat(prose[half:])}


def run(n_tok):
    root = BASE_URL.rsplit("/v1", 1)[0]
    out = {}
    for name, text in docs().items():
        req = urllib.request.Request(root + "/tokenize", json.dumps(
            {"model": MODEL, "prompt": text[: n_tok * 6], "add_special_tokens": False}).encode(), {"Content-Type": "application/json", **({"Authorization": "Bearer " + os.environ["LLM_API_KEY"]} if os.environ.get("LLM_API_KEY") else {})})
        ids = json.load(urllib.request.urlopen(req, timeout=600))["tokens"][:n_tok]
        body = {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1}
        req = urllib.request.Request(BASE_URL + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json", **({"Authorization": "Bearer " + os.environ["LLM_API_KEY"]} if os.environ.get("LLM_API_KEY") else {})})
        res = json.load(urllib.request.urlopen(req, timeout=3600))
        pl = res["choices"][0]["prompt_logprobs"]
        lp, top1 = [], []
        for i in range(1, len(ids)):
            d = pl[i]; e = d.get(str(ids[i]))
            lp.append(e["logprob"]); top1.append(e.get("rank") == 1)
        lp = np.array(lp); top1 = np.array(top1)
        row = {"tokens": len(ids), "logprob": lp.tolist(), "top1": top1.astype(int).tolist()}
        out[name] = row
        msg = "  ".join(f"[{a//1024}K-{min(b, len(ids))//1024}K] nll {-lp[a:b].mean():.4f}"
                        for a, b in BUCKETS if a < len(lp))
        print(f"{name}: {len(ids)} tok, ppl {math.exp(-lp.mean()):.4f}, top1 {top1.mean():.4f}   {msg}", flush=True)
    tag = os.environ.get("LLM_TAG")
    print("saved", save((f"{tag}_" if tag else "") + "kv_ppl", out))


def compare(a, b):
    A = json.load(open(os.path.join(RESULTS, f"{a}_kv_ppl.json"))); B = json.load(open(os.path.join(RESULTS, f"{b}_kv_ppl.json")))
    for name in A:
        la, lb = np.array(A[name]["logprob"]), np.array(B[name]["logprob"])
        ta, tb = np.array(A[name]["top1"]), np.array(B[name]["top1"])
        n = min(len(la), len(lb)); la, lb, ta, tb = la[:n], lb[:n], ta[:n], tb[:n]
        parts = []
        for lo, hi in BUCKETS:
            if lo >= n:
                break
            d = (-lb[lo:hi].mean()) - (-la[lo:hi].mean())
            parts.append(f"[{lo//1024}K-{min(hi, n)//1024}K] dNLL {d:+.4f} ({100*(math.exp(d)-1):+.2f}% ppl)")
        print(f"{name}: {b} vs {a}: ppl {math.exp(-la.mean()):.4f} -> {math.exp(-lb.mean()):.4f}, "
              f"top1 {ta.mean():.4f} -> {tb.mean():.4f}, top1 agreement-of-hits {np.mean(ta == tb):.4f}\n    " + "  ".join(parts))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--compare"]:
        compare(sys.argv[2], sys.argv[3])
    else:
        run(int(sys.argv[1]) if len(sys.argv) > 1 else 65536)
