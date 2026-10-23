#!/usr/bin/env python3
"""Hero banner (KPI row of stat tiles) and the swarm-progress bar chart, light and dark, as static SVG for GitHub."""
import os, sys
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781", grid="#e1e0d9", base="#c3c2b7",
                  border="rgba(11,11,11,0.10)", series="#2a78d6", tile="#f5f5f2"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#898781", grid="#2c2c2a", base="#383835",
                 border="rgba(255,255,255,0.10)", series="#3987e5", tile="#222221"),
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def hero(t):
    W, H = 1200, 300
    tiles = [("Generation, 16 requests", "1,144", "tok/s"),
             ("Prefill, 50K-token prompt", "8,490", "tok/s"),
             ("KV cache in VRAM", "1.21M", "tokens + RAM tier"),
             ("40-agent swarm", "26.6", "turns per minute")]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="flash-next-swarm-quad3090: Swift 1.5 on 4x RTX 3090. Generation 1,144 tok/s at 16 requests; prefill 8,490 tok/s on a 50K prompt; 1.21M-token KV cache plus a RAM tier; 26.6 turns per minute with 40 agents.">',
         f'<rect width="{W}" height="{H}" rx="16" fill="{t["surface"]}"/>',
         f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="16" fill="none" stroke="{t["border"]}"/>',
         f'<text x="48" y="70" font-family=\'{FONT}\' font-size="34" font-weight="650" fill="{t["text"]}">flash-next-swarm-quad3090</text>',
         f'<text x="48" y="104" font-family=\'{FONT}\' font-size="19" fill="{t["text2"]}">Swift 1.5 (Qwen3.8-Flash-Next, 125B MoE) on four RTX 3090s, tuned for coding-agent swarms</text>',
         f'<rect x="48" y="122" width="56" height="3" rx="1.5" fill="{t["series"]}"/>']
    tw, gap, x0, y0, th = 258, 20, 48, 150, 112
    for i, (label, value, unit) in enumerate(tiles):
        x = x0 + i * (tw + gap)
        o += [f'<rect x="{x}" y="{y0}" width="{tw}" height="{th}" rx="12" fill="{t["tile"]}"/>',
              f'<text x="{x+20}" y="{y0+32}" font-family=\'{FONT}\' font-size="15" fill="{t["text2"]}">{esc(label)}</text>',
              f'<text x="{x+20}" y="{y0+80}" font-family=\'{FONT}\' font-size="44" font-weight="650" fill="{t["text"]}">{esc(value)}</text>',
              f'<text x="{x+20}" y="{y0+102}" font-family=\'{FONT}\' font-size="14" fill="{t["muted"]}">{esc(unit)}</text>']
    o.append('</svg>')
    return "\n".join(o)


def swarm_chart(t):
    steps = [("RAM tier, DFlash 2, small-batch math", 15.94),
             ("Smarter RAM resume and scheduling", 23.13),
             ("4-bit KV cache", 23.34),
             ("8-bit math + BF16 unlock", 24.24),
             ("Faster GPU traffic: NCCL", 25.73),
             ("Faster GPU traffic: FlashInfer", 26.63)]
    W, H = 1200, 456
    L, R, T = 360, 1110, 118          # plot left, right, top
    row, bar = 44, 22
    xmax = 30.0
    sx = lambda v: L + (R - L) * v / xmax
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Bar chart: 40-agent swarm, completed turns per minute after each step: ' +
         "; ".join(f"{esc(n)} {v:.2f}" for n, v in steps) + '.">',
         f'<rect width="{W}" height="{H}" rx="16" fill="{t["surface"]}"/>',
         f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="16" fill="none" stroke="{t["border"]}"/>',
         f'<text x="48" y="58" font-family=\'{FONT}\' font-size="22" font-weight="650" fill="{t["text"]}">What each change added: 40-agent swarm, turns per minute</text>',
         f'<text x="48" y="86" font-family=\'{FONT}\' font-size="15" fill="{t["text2"]}">40 agents with ~100K-token conversations for 20 minutes, same benchmark at every step; the BF16 unlock came with 8-bit math</text>']
    yb = T + len(steps) * row
    for v in (0, 10, 20, 30):     # recessive hairline grid + ticks
        x = sx(v)
        o.append(f'<line x1="{x:.1f}" y1="{T-6}" x2="{x:.1f}" y2="{yb}" stroke="{t["grid"] if v else t["base"]}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{yb+22}" text-anchor="middle" font-family=\'{FONT}\' font-size="13" fill="{t["muted"]}">{v}</text>')
    for i, (name, v) in enumerate(steps):
        y = T + i * row + (row - bar) / 2
        x1 = sx(v); r = 4
        # square at the baseline, 4px rounded data end
        path = (f'M{L},{y} H{x1-r:.1f} Q{x1:.1f},{y} {x1:.1f},{y+r} V{y+bar-r} Q{x1:.1f},{y+bar} {x1-r:.1f},{y+bar} H{L} Z')
        o.append(f'<path d="{path}" fill="{t["series"]}"><title>{esc(name)}: {v:.2f} turns/min</title></path>')
        o.append(f'<text x="{L-14}" y="{y+bar/2+5}" text-anchor="end" font-family=\'{FONT}\' font-size="15" fill="{t["text2"]}">{esc(name)}</text>')
        o.append(f'<text x="{x1+10:.1f}" y="{y+bar/2+5}" font-family=\'{FONT}\' font-size="15" font-weight="600" fill="{t["text"]}">{v:.1f}</text>')
    o.append(f'<text x="{(L+R)/2:.0f}" y="{yb+48}" text-anchor="middle" font-family=\'{FONT}\' font-size="13" fill="{t["muted"]}">turns per minute</text>')
    o.append('</svg>')
    return "\n".join(o)


for name, t in THEMES.items():
    open(os.path.join(OUT, f"hero-{name}.svg"), "w").write(hero(t))
    open(os.path.join(OUT, f"swarm-steps-{name}.svg"), "w").write(swarm_chart(t))
print("wrote", sorted(os.listdir(OUT)))
