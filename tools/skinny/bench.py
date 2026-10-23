# Tune / verify the sm_86 skinny GEMM (vllm/models/qwen4_exp/nvidia/ops/skinny_sm86.py) against F.linear (cuBLAS)
# under CUDA graph replay with a cold L2 (weights rotate through enough copies to exceed the 6 MB L2, as in a decode
# step that streams every layer's weights once). Replays are timed one at a time with idle gaps in between: the
# 3090s are capped at 200 W / 1400 MHz and back-to-back streaming throttled the SM clock to ~330 MHz in W18, while
# a decode step keeps each GPU busy only ~half the time.
#   tune:    bench.py tune OUT.jsonl [shape ...]   (env MS=8,16,32,48,64)
#   tunemix: bench.py tunemix OUT.jsonl            (fused HC up projection + gate mix, hc_up shape)
#   plans:   bench.py plans OUT.json IN.jsonl ...  (keep configs >= MIN_GAIN faster than the unfused/cuBLAS path)
#   stress:  bench.py stress PLANS.json [reps]     (bit-exact repeatability of every plan, counters reset)
#   hctest:  bench.py hctest PLANS.json            (fused HC chain == unfused skinny chain, close to the cuBLAS chain)
# run: env PYTHONPATH=<tree> CUDA_VISIBLE_DEVICES=<gpu> VENV/bin/python bench.py ...
import itertools, json, math, os, statistics, sys, time
import torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.nvidia.ops.hc import _hc_gate_mix, _hc_silu
from vllm.models.qwen4_exp.nvidia.ops.skinny_sm86 import launch, launch_up_mix, scratch_size

dev = torch.device("cuda")
SHAPES = {  # name: (N, K) per GPU at TP2 (PP2 x TP2), target model then DFlash 2 drafter
    "qsa_qkv": (6656, 2560), "qsa_o": (2560, 3072), "idx_qk": (640, 2560), "gdn_ba": (48, 2560),
    "router": (512, 2560), "hc_down": (336, 10240), "hc_up": (10240, 320), "hc_fdown": (320, 10240),
    "ple_kv": (12800, 2560), "lm_head": (124160, 2560), "sh_down": (2560, 320),
    "d_qkv": (3072, 2560), "d_o": (2560, 2048), "d_gu": (7680, 2560), "d_down": (2560, 3840),
    "d_fc": (2560, 12800),
}
HC, HS = 4, 2560
MS = [int(x) for x in os.environ.get("MS", "8,16,32,48,64").split(",")]
MIN_GAIN = float(os.environ.get("MIN_GAIN", "1.05"))
GAP_S = float(os.environ.get("GAP_S", "0.010"))


class Case:
    def __init__(self, n, k, m):
        wb = n * k * 2
        self.copies = max(2, min(256, math.ceil(64 * 2**20 / wb)))
        self.ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(self.copies)]
        self.xs = [torch.randn(m, k, device=dev, dtype=torch.bfloat16) for _ in range(self.copies)]
        self.n, self.k, self.m = n, k, m
        self.ref = F.linear(self.xs[0].float(), self.ws[0].float())


def graph_time(calls, reps=5):
    """Median us per call over `reps` graph replays of one pass over `calls`, with idle gaps between replays."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls:
            c()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in calls:
            c()
    g.replay(); g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        time.sleep(GAP_S)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / len(calls))
    del g
    return statistics.median(ts)


def time_cublas(case):
    return graph_time([(lambda i=i: F.linear(case.xs[i], case.ws[i])) for i in range(case.copies)])


def scratch(cfg, n, m=64):
    e, t = scratch_size(cfg, n, m)
    return torch.empty(max(e, 1), device=dev, dtype=torch.float32), torch.zeros(max(t, 1), device=dev, dtype=torch.int32)


def time_skinny(case, cfg):
    try:
        ws, cnt = scratch(cfg, case.n)
        outs = [torch.empty(case.m, case.n, device=dev, dtype=torch.bfloat16) for _ in range(case.copies)]
        y = launch(case.xs[0], case.ws[0], cfg, ws, cnt, outs[0]).clone()
        err = (y.float() - case.ref).abs().max().item() / (case.ref.abs().max().item() + 1e-6)
        if not err < 1e-2:
            print("  WRONG", cfg, err, flush=True)
            return float("nan")
        t = graph_time([(lambda i=i: launch(case.xs[i], case.ws[i], cfg, ws, cnt, outs[i])) for i in range(case.copies)])
        if not torch.equal(outs[0], y) or cnt.any():
            print("  NONDETERMINISTIC / counter not reset", cfg, flush=True)
            return float("nan")
        return t
    except Exception as e:
        if "out of resource" not in str(e).lower():
            print("  skip", cfg, type(e).__name__, str(e)[:160], flush=True)
        return float("nan")


def floor_us(n, k, m):
    return (n * k * 2 + m * k * 2 + m * n * 2) / 900e9 * 1e6   # ~900 GB/s practical peak (936 nominal)


def bms_for(m):   # M tile sizes worth trying: one tile, or 2-4 smaller tiles that share W through L2
    return [16] if m <= 16 else [16, 32] if m <= 32 else [16, 32, 64]


def fits(c, m, n, k):
    bm, bn, bk, sk, warps, stages = c
    ctas = math.ceil(m / bm) * math.ceil(n / bn) * sk
    return k % (bk * sk) == 0 and stages * (bm + bn) * bk * 2 <= 96 * 1024 and 8 <= ctas <= 8192


def tune(name, m, top=5):
    n, k = SHAPES[name]
    case = Case(n, k, m)
    base = time_cublas(case)
    res, seen = [], set()
    for bm, bn, bk, sk in itertools.product(bms_for(m), [16, 32, 64, 128], [64, 128, 256], [1, 2, 4, 5, 8, 10, 20]):
        c = (bm, bn, bk, sk, 4, 3)
        if sk > 2 and math.ceil(m / bm) * math.ceil(n / bn) >= 164:   # two waves of tiles already: split-K only adds work
            continue
        if fits(c, m, n, k):
            seen.add(c)
            t = time_skinny(case, c)
            if t == t:
                res.append((t, c))
    res.sort()
    for _, c in res[:top]:
        for warps, stages in itertools.product([2, 4, 8], [2, 3, 4, 5]):
            c2 = c[:4] + (warps, stages)
            if c2 in seen or not fits(c2, m, n, k):
                continue
            seen.add(c2)
            t = time_skinny(case, c2)
            if t == t:
                res.append((t, c2))
    res.sort()
    del case; torch.cuda.empty_cache()
    return base, res[:3]


class MixCase:
    """HC up projection (x [M, 320] -> gate [M, HC*HS]) + hc_gate_mix with xn [M, HC*HS]."""
    def __init__(self, m):
        n, k = HC * HS, 320
        self.copies = max(2, math.ceil(64 * 2**20 / (n * k * 2)))
        self.ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.05 for _ in range(self.copies)]
        self.xs = [torch.randn(m, k, device=dev, dtype=torch.bfloat16) for _ in range(self.copies)]
        self.xn = [torch.randn(m, n, device=dev, dtype=torch.bfloat16) for _ in range(self.copies)]
        self.m = m
        self.ref = _hc_gate_mix(self.xn[0], F.linear(self.xs[0], self.ws[0]), HC).float()


def time_mix(case, cfg):
    try:
        outs = [torch.empty(case.m, HS, device=dev, dtype=torch.bfloat16) for _ in range(case.copies)]
        y = launch_up_mix(case.xs[0], case.ws[0], case.xn[0], cfg, HC, outs[0]).clone()
        err = (y.float() - case.ref).abs().max().item() / (case.ref.abs().max().item() + 1e-6)
        if not err < 2e-2:
            print("  WRONG mix", cfg, err, flush=True)
            return float("nan")
        return graph_time([(lambda i=i: launch_up_mix(case.xs[i], case.ws[i], case.xn[i], cfg, HC, outs[i])) for i in range(case.copies)])
    except Exception as e:
        if "out of resource" not in str(e).lower():
            print("  skip mix", cfg, type(e).__name__, str(e)[:160], flush=True)
        return float("nan")


def tune_mix(m, gemm_cfg):
    case = MixCase(m)
    base = graph_time([(lambda i=i: _hc_gate_mix(case.xn[i], F.linear(case.xs[i], case.ws[i]), HC)) for i in range(case.copies)])
    unfused = float("nan")
    if gemm_cfg:   # current path: skinny up GEMM + hc_gate_mix kernel
        ws, cnt = scratch(gemm_cfg, HC * HS)
        unfused = graph_time([(lambda i=i: _hc_gate_mix(case.xn[i], launch(case.xs[i], case.ws[i], gemm_cfg, ws, cnt), HC)) for i in range(case.copies)])
    res = []
    for bm, bn, bk, warps, stages in itertools.product(bms_for(m), [16, 32, 64, 128], [32, 64], [2, 4, 8], [2, 3, 4]):
        c = (bm, bn, bk, 1, warps, stages)
        if stages * (bm + bn) * bk * 2 > 96 * 1024:
            continue
        t = time_mix(case, c)
        if t == t:
            res.append((t, c))
    res.sort()
    del case; torch.cuda.empty_cache()
    return base, unfused, res[:3]


def main():
    mode = sys.argv[1]
    if mode == "tune":
        out_path = sys.argv[2]
        for name in sys.argv[3:] or list(SHAPES):
            n, k = SHAPES[name]
            for m in MS:
                t0 = time.time()
                base, best = tune(name, m)
                bt, bc = best[0] if best else (float("nan"), None)
                print(f"{name:9s} {n:6d}x{k:<5d} M={m:2d} floor {floor_us(n,k,m):7.1f} us | cublas {base:7.1f} | "
                      f"skinny {bt:7.1f} {bc} | x{base/bt:4.2f} [{time.time()-t0:3.0f}s]", flush=True)
                with open(out_path, "a") as f:
                    f.write(json.dumps({"name": name, "n": n, "k": k, "m": m, "cublas_us": base,
                                        "floor_us": floor_us(n, k, m), "best": [[t, list(c)] for t, c in best]}) + "\n")
    elif mode == "tunemix":
        out_path = sys.argv[2]
        gemm = {}
        for p in sys.argv[3:]:   # GEMM tune results for hc_up (10240x320): the unfused baseline config per M
            for line in open(p):
                r = json.loads(line)
                if r["name"] == "hc_up" and r["best"]:
                    gemm[r["m"]] = tuple(r["best"][0][1])
        for m in MS:
            t0 = time.time()
            base, unfused, best = tune_mix(m, gemm.get(m))
            bt, bc = best[0] if best else (float("nan"), None)
            print(f"hc_upmix  {HC*HS:6d}x320   M={m:2d} | cublas+mix {base:7.1f} us | skinny+mix {unfused:7.1f} | fused {bt:7.1f} {bc} "
                  f"| x{base/bt:4.2f} vs cublas, x{unfused/bt:4.2f} vs skinny [{time.time()-t0:3.0f}s]", flush=True)
            with open(out_path, "a") as f:
                f.write(json.dumps({"name": "hc_upmix", "n": HC * HS, "k": 320, "m": m, "cublas_us": base,
                                    "unfused_us": unfused, "best": [[t, list(c)] for t, c in best]}) + "\n")
    elif mode == "plans":
        plans = {}
        for path in sys.argv[3:]:
            for line in open(path):
                r = json.loads(line)
                if not r["best"]:
                    continue
                t = r["best"][0][0]
                if r["name"] in ("hc_upmix", "w8_head"):   # beat the faster of cuBLAS and the skinny bf16 path
                    ref = min(x for x in (r["cublas_us"], r.get("unfused_us", float("inf"))) if x == x)
                    prefix = "mix" if r["name"] == "hc_upmix" else "w8"
                    if ref / t >= MIN_GAIN:
                        plans.setdefault(f'{prefix}:{r["n"]}x{r["k"]}', {})[str(r["m"])] = r["best"][0][1]
                elif r["cublas_us"] / t >= MIN_GAIN:
                    plans.setdefault(f'{r["n"]}x{r["k"]}', {})[str(r["m"])] = r["best"][0][1]
        json.dump(plans, open(sys.argv[2], "w"), indent=1)
        print(f"{len(plans)} shapes, {sum(len(v) for v in plans.values())} (shape, M) plans -> {sys.argv[2]}")
    elif mode == "stress":
        plans = json.load(open(sys.argv[2])); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        bad = 0
        for key, by_m in plans.items():
            if key.startswith("mix:"):
                continue   # no split-K: deterministic by construction (checked in hctest)
            n, k = map(int, key.split("x"))
            for m, c in by_m.items():
                c = tuple(c)
                for mm in sorted({int(m), max(1, int(m) - 3)}):   # bucket top and a ragged M inside it
                    x = torch.randn(mm, k, device=dev, dtype=torch.bfloat16)
                    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
                    ws, cnt = scratch(c, n)
                    out = torch.empty(mm, n, device=dev, dtype=torch.bfloat16)
                    ref = launch(x, w, c, ws, cnt).clone()
                    err = (ref.float() - F.linear(x.float(), w.float())).abs().max().item()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        launch(x, w, c, ws, cnt, out)
                    mism = 0
                    for _ in range(reps):
                        g.replay()
                        mism += int(not torch.equal(out, ref))
                    torch.cuda.synchronize()
                    ok = mism == 0 and not cnt.any() and err < 0.05
                    bad += not ok
                    print(f"{'ok ' if ok else 'BAD'} {key:12s} M={mm:2d} {c} abs err {err:.4f} mismatches {mism}/{reps}", flush=True)
        print("STRESS", "PASS" if bad == 0 else f"FAIL ({bad})")
        sys.exit(1 if bad else 0)
    elif mode == "hctest":
        plans = json.load(open(sys.argv[2])); bad = 0
        mix = {int(m): tuple(c) for m, c in plans.get(f"mix:{HC*HS}x320", {}).items()}
        for key, act_n in ((f"336x{HC*HS}", 320), (f"320x{HC*HS}", 320)):
            dplan = {int(m): tuple(c) for m, c in plans.get(key, {}).items()}
            nd = int(key.split("x")[0])
            for m in (1, 3, 8, 13, 16, 24, 32, 40, 48, 64):
                bucket = next(b for b in (8, 16, 32, 48, 64) if m <= b)
                dc, mc = dplan.get(bucket), mix.get(bucket)
                if not dc:
                    continue
                xn = torch.randn(m, HC * HS, device=dev, dtype=torch.bfloat16)
                wd = torch.randn(nd, HC * HS, device=dev, dtype=torch.bfloat16) * 0.01
                wu = torch.randn(HC * HS, 320, device=dev, dtype=torch.bfloat16) * 0.05
                ws, cnt = scratch(dc, nd)
                # unfused skinny chain: GEMM -> hc_silu -> up GEMM (same tiles as the fused kernel) -> hc_gate_mix
                y = launch(xn, wd, dc, ws, cnt)
                lora = _hc_silu(y[:, :act_n], HC)
                d = launch(xn, wd, dc, ws, cnt, act_n=act_n, hc=HC)
                ok_down = torch.equal(d[:, :act_n], lora) and torch.equal(d[:, act_n:], y[:, act_n:])
                line = f"{key:10s} M={m:2d} down+silu {'bit-exact' if ok_down else 'MISMATCH'}"
                bad += not ok_down
                # cuBLAS chain for tolerance
                ref = _hc_gate_mix(xn, F.linear(_hc_silu(F.linear(xn, wd)[:, :act_n], HC), wu), HC).float()
                if mc:
                    out = launch_up_mix(d[:, :act_n], wu, xn, mc, HC)
                    bm, bn, bk, _, wp, st = mc
                    unf = _hc_gate_mix(xn, launch(lora, wu, (bm, bn, bk, 1, wp, st), *scratch((bm, bn, bk, 1, wp, st), HC * HS)), HC)
                    exact = torch.equal(out, unf)
                    err = (out.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
                    ok = err < 2e-2
                    bad += not ok
                    line += f" | up+mix vs unfused skinny {'bit-exact' if exact else 'differs (%.2e)' % (out.float() - unf.float()).abs().max().item()}, vs cuBLAS rel err {err:.1e} {'ok' if ok else 'BAD'}"
                print(line, flush=True)
        print("HCTEST", "PASS" if bad == 0 else f"FAIL ({bad})")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
