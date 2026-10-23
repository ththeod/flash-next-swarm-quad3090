#!/usr/bin/env python3
"""Blog charts as HTML includes (Jekyll _includes/): bars are divs, so text stays sharp at any width, colors come
from the page's CSS variables (light/dark), and assets/js/charts.js animates them in when scrolled into view."""
import html, os, sys

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
esc = lambda s: html.escape(str(s), quote=True)


def pct(v, lo, hi):
    return 100.0 * (v - lo) / (hi - lo)


def axis(ticks, lo, hi, fmt, title):
    marks = "".join(f'<span class="tick" style="left:{pct(t, lo, hi):.3f}%">{esc(fmt(t))}</span>' for t in ticks)
    return f'<div class="axis"><div class="ticks">{marks}</div><div class="axis-title">{esc(title)}</div></div>'


def grid(ticks, lo, hi, base=None):
    """Gridlines drawn inside each track, so they stay aligned when labels stack above the bars on phones."""
    return "".join(f'<span class="gl{" base" if t == base else ""}" style="left:{pct(t, lo, hi):.3f}%"></span>' for t in ticks)


def figure(cid, title, sub, body, legend="", lw=280):
    return (f'<figure class="chart" id="{cid}" style="--lw:{lw}px" aria-label="{esc(title)}">'
            f'<figcaption class="chart-head"><span class="chart-title">{esc(title)}</span>'
            f'<span class="chart-sub">{esc(sub)}</span></figcaption>{legend}{body}</figure>\n')


def steps():
    data = [("RAM tier, DFlash 2, small-batch math", 15.94),
            ("Smarter RAM resume and scheduling", 23.13),
            ("4-bit KV cache", 23.34),
            ("8-bit math + BF16 unlock", 24.24),
            ("Faster GPU traffic: NCCL", 25.73),
            ("Faster GPU traffic: FlashInfer", 26.63)]
    lo, hi, ticks = 0, 30, (0, 10, 20, 30)
    rows, prev = [], 0.0
    for i, (name, v) in enumerate(data):
        gain = v - prev
        tip_label = name if i == 0 else f"{name}: +{gain:.1f} turns/min"
        segs = ""
        if prev:
            segs += f'<span class="seg tint" style="left:0;width:{pct(prev, lo, hi):.3f}%"></span>'
        gap = "2px" if prev else "0px"
        segs += (f'<span class="seg solid" style="left:calc({pct(prev, lo, hi):.3f}% + {gap});'
                 f'width:calc({pct(v, lo, hi) - pct(prev, lo, hi):.3f}% - {gap})"></span>')
        gain_txt = f'<span class="gain">+{gain:.1f}</span>' if prev else ""
        rows.append(f'<div class="row" tabindex="0" style="--i:{i}" data-tip-value="{v:.1f} turns/min" data-tip-label="{esc(tip_label)}">'
                    f'<div class="label">{esc(name)}</div><div class="track">{grid(ticks, lo, hi, base=0)}{segs}'
                    f'<span class="val" style="left:{pct(v, lo, hi):.3f}%">{v:.1f}{gain_txt}</span></div></div>')
        prev = v
    legend = ('<div class="legend"><span class="key"><i class="sw tint"></i>before this change</span>'
              '<span class="key"><i class="sw solid"></i>added by this change</span></div>')
    body = f'<div class="rows">{"".join(rows)}</div>{axis(ticks, lo, hi, str, "completed turns per minute")}'
    return figure("chart-steps", "What each change added",
                  "40-agent swarm with ~100K-token conversations, 20 minutes per run. The BF16 unlock came with the 8-bit math.",
                  body, legend, lw=260)


def swarm():
    data = [("Our version", 26.48, "ours"), ("Merlin 3.0.0", 4.15, "merlin")]
    lo, hi, ticks = 0, 33, (0, 10, 20, 30)   # headroom on the right for the value and the 6.4x pill
    rows = []
    for i, (name, v, cls) in enumerate(data):
        extra = f'<span class="callout">{data[0][1] / data[1][1]:.1f}×</span>' if cls == "ours" else ""
        rows.append(f'<div class="row" tabindex="0" style="--i:{i}" data-tip-value="{v:.1f} turns/min" data-tip-label="{esc(name)}">'
                    f'<div class="label">{esc(name)}</div><div class="track">{grid(ticks, lo, hi, base=0)}'
                    f'<span class="seg {cls}" style="left:0;width:{pct(v, lo, hi):.3f}%"></span>'
                    f'<span class="val" style="left:{pct(v, lo, hi):.3f}%">{v:.1f}{extra}</span></div></div>')
    body = f'<div class="rows tall">{"".join(rows)}</div>{axis(ticks, lo, hi, str, "completed turns per minute")}'
    return figure("chart-swarm", "40-agent swarm: completed turns per minute",
                  "Same machine, same day. Merlin recomputes 99% of each prompt, ours 4%.", body, lw=140)


def ratios():
    data = [("Generation, 16 requests", 1124.0, 556.8, "tok/s"),
            ("Generation, 1 request, short prompt", 299.6, 156.0, "tok/s"),
            ("Generation, 8 requests", 763.9, 586.2, "tok/s"),
            ("Generation, 8 requests at 80K context", 802.3, 659.2, "tok/s"),
            ("1 request after a 260K prompt", 131.5, 114.9, "tok/s"),
            ("8 agents at once (halt95's test)", 792.6, 716.5, "tok/s"),
            ("Prefill, 50K prompt", 8256, 7418, "tok/s"),
            ("Prefill, 260K prompt", 7737, 7003, "tok/s"),
            ("1 request, thinking on, 4K (halt95's test)", 170.6, 182.9, "tok/s")]
    lo, hi, ticks = 0.8, 2.2, (1.0, 1.5, 2.0)
    b = pct(1.0, lo, hi)
    rows = []
    for i, (name, ours, merlin, unit) in enumerate(data):
        r = ours / merlin
        x = pct(r, lo, hi)
        cls = "up" if r >= 1 else "down"
        left, width = (b, x - b) if r >= 1 else (x, b - x)
        val_style = f"left:{x:.3f}%" if r >= 1 else f"left:{b:.3f}%"
        rows.append(f'<div class="row" tabindex="0" style="--i:{i}" data-tip-value="{r:.2f}×"'
                    f' data-tip-label="Ours {ours:,.0f} vs Merlin {merlin:,.0f} {unit}">'
                    f'<div class="label">{esc(name)}</div><div class="track">{grid(ticks, lo, hi, base=1.0)}'
                    f'<span class="seg {cls}" style="left:{left:.3f}%;width:{width:.3f}%"></span>'
                    f'<span class="val" style="{val_style}">{r:.2f}×</span></div></div>')
    body = f'<div class="rows">{"".join(rows)}</div>{axis(ticks, lo, hi, lambda t: f"{t:.1f}×", "ours ÷ Merlin 3.0.0")}'
    return figure("chart-ratios", "Our version relative to Merlin 3.0.0",
                  "Same machine, same day, same settings. 1.0× is Merlin; right of the line, ours is faster.", body, lw=290)


def hero():
    tiles = [("Generation, 16 requests", 1144, 0, "", "tok/s"),
             ("Prefill, 50K-token prompt", 8490, 0, "", "tok/s"),
             ("KV cache in VRAM", 1.21, 2, "M", "tokens, plus a RAM tier"),
             ("40-agent swarm", 26.6, 1, "", "turns per minute")]
    out = []
    for i, (label, v, dec, suf, unit) in enumerate(tiles):
        shown = f"{v:,.{dec}f}{suf}"
        out.append(f'<div class="kpi" style="--i:{i}"><span class="kpi-label">{esc(label)}</span>'
                   f'<span class="kpi-value" data-count="{v}" data-dec="{dec}" data-suffix="{suf}">{shown}</span>'
                   f'<span class="kpi-unit">{esc(unit)}</span></div>')
    return ('<figure class="hero chart" aria-label="Headline numbers: 1,144 tok/s at 16 requests; 8,490 tok/s prefill; '
            '1.21M-token KV cache plus a RAM tier; 26.6 turns per minute with 40 agents">'
            f'<div class="kpis">{"".join(out)}</div></figure>\n')


for name, fn in [("chart-hero", hero), ("chart-steps", steps), ("chart-swarm", swarm), ("chart-ratios", ratios)]:
    open(os.path.join(OUT, f"{name}.html"), "w").write(fn())
print("ok", sorted(os.listdir(OUT)))
