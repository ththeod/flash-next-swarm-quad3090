#!/usr/bin/env python3
"""Same-day comparison graphics vs Merlin 3.0.0: a diverging ratio chart (ours / Merlin, baseline 1.0x) and the swarm
pair, light and dark, static SVG for GitHub and Pages."""
import os, sys
OUT = sys.argv[1]
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
T = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781", grid="#e1e0d9", base="#898781",
                  border="rgba(11,11,11,0.10)", up="#2a78d6", down="#e34948", merlin="#eb6834", pill="#f3f2ee"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#898781", grid="#2c2c2a", base="#898781",
                 border="rgba(255,255,255,0.10)", up="#3987e5", down="#e66767", merlin="#d95926", pill="#242423"),
}
RATIOS = [  # (label, ours, merlin)
    ("Generation, 16 requests", 1124.0, 556.8),
    ("Generation, 1 request, short prompt", 299.6, 156.0),
    ("Generation, 8 requests", 763.9, 586.2),
    ("Generation, 8 requests at 80K context", 802.3, 659.2),
    ("1 request after a 260K prompt", 131.5, 114.9),
    ("8 agents at once (halt95's test)", 792.6, 716.5),
    ("Prefill, 50K prompt", 8256, 7418),
    ("Prefill, 260K prompt", 7737, 7003),
    ("1 request, thinking on, 4K (halt95's test)", 170.6, 182.9),
]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def frame(t, W, H, title, sub, label):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label)}">',
            f'<rect width="{W}" height="{H}" rx="16" fill="{t["surface"]}"/>',
            f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="16" fill="none" stroke="{t["border"]}"/>',
            f'<text x="48" y="58" font-family=\'{FONT}\' font-size="22" font-weight="650" fill="{t["text"]}">{esc(title)}</text>',
            f'<text x="48" y="86" font-family=\'{FONT}\' font-size="15" fill="{t["text2"]}">{esc(sub)}</text>']


def ratio_chart(t):
    W, row, bar, T0 = 1200, 40, 20, 118
    H = T0 + len(RATIOS) * row + 70
    L, R = 440, 1110
    lo, hi = 0.8, 2.2
    sx = lambda v: L + (R - L) * (v - lo) / (hi - lo)
    label = "Ours relative to Merlin 3.0.0, same machine and day: " + "; ".join(f"{n} {o/m:.2f}x" for n, o, m in RATIOS)
    o = frame(t, W, H, "flash-next-swarm-quad3090 relative to Merlin 3.0.0", "Same machine, same day, same settings; 1.0× = Merlin 3.0.0; right of the line = ours is faster", label)
    yb = T0 + len(RATIOS) * row
    for v in (1.0, 1.5, 2.0):
        x = sx(v)
        o.append(f'<line x1="{x:.1f}" y1="{T0-6}" x2="{x:.1f}" y2="{yb}" stroke="{t["base"] if v == 1.0 else t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{yb+22}" text-anchor="middle" font-family=\'{FONT}\' font-size="13" fill="{t["muted"]}">{v:.1f}×</text>')
    x1 = sx(1.0)
    for i, (name, ours, merlin) in enumerate(RATIOS):
        r = ours / merlin
        y = T0 + i * row + (row - bar) / 2
        x2 = sx(r); rr = 4
        if r >= 1:
            d = f'M{x1:.1f},{y} H{x2-rr:.1f} Q{x2:.1f},{y} {x2:.1f},{y+rr} V{y+bar-rr} Q{x2:.1f},{y+bar} {x2-rr:.1f},{y+bar} H{x1:.1f} Z'
            col, lx, anc = t["up"], x2 + 10, "start"
        else:
            d = f'M{x1:.1f},{y} H{x2+rr:.1f} Q{x2:.1f},{y} {x2:.1f},{y+rr} V{y+bar-rr} Q{x2:.1f},{y+bar} {x2+rr:.1f},{y+bar} H{x1:.1f} Z'
            col, lx, anc = t["down"], x1 + 10, "start"
        o.append(f'<path d="{d}" fill="{col}"><title>{esc(name)}: ours {ours:,.1f}, Merlin {merlin:,.1f} ({r:.2f}×)</title></path>')
        o.append(f'<text x="{L-14}" y="{y+bar/2+5}" text-anchor="end" font-family=\'{FONT}\' font-size="15" fill="{t["text2"]}">{esc(name)}</text>')
        o.append(f'<text x="{lx:.1f}" y="{y+bar/2+5}" text-anchor="{anc}" font-family=\'{FONT}\' font-size="15" font-weight="600" fill="{t["text"]}">{r:.2f}×</text>')
    o.append(f'<text x="{(L+R)/2:.0f}" y="{yb+48}" text-anchor="middle" font-family=\'{FONT}\' font-size="13" fill="{t["muted"]}">ours ÷ Merlin 3.0.0 (tok/s)</text>')
    o.append('</svg>')
    return "\n".join(o)


def swarm_chart(t):
    W, H = 1200, 300
    rows = [("flash-next-swarm-quad3090", 26.48, t["up"]), ("Merlin 3.0.0", 4.15, t["merlin"])]
    L, R, T0, row, bar = 300, 1110, 122, 56, 24
    sx = lambda v: L + (R - L) * v / 33.0   # headroom for the value and the pill
    label = "40-agent swarm, turns per minute, same machine and day: ours 26.5, Merlin 3.0.0 4.2"
    o = frame(t, W, H, "40-agent swarm: completed turns per minute", "40 agents with ~100K-token contexts for 20 minutes, same machine and day; Merlin recomputes 99% of each prompt, ours 4%", label)
    yb = T0 + len(rows) * row
    for v in (0, 10, 20, 30):
        x = sx(v)
        o.append(f'<line x1="{x:.1f}" y1="{T0-6}" x2="{x:.1f}" y2="{yb}" stroke="{t["grid"] if v else t["base"]}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{yb+22}" text-anchor="middle" font-family=\'{FONT}\' font-size="13" fill="{t["muted"]}">{v}</text>')
    for i, (name, v, col) in enumerate(rows):
        y = T0 + i * row + (row - bar) / 2; x2 = sx(v); r = 4
        d = f'M{L},{y} H{x2-r:.1f} Q{x2:.1f},{y} {x2:.1f},{y+r} V{y+bar-r} Q{x2:.1f},{y+bar} {x2-r:.1f},{y+bar} H{L} Z'
        o.append(f'<path d="{d}" fill="{col}"><title>{esc(name)}: {v:.2f} turns/min</title></path>')
        o.append(f'<text x="{L-14}" y="{y+bar/2+5}" text-anchor="end" font-family=\'{FONT}\' font-size="15" fill="{t["text2"]}">{esc(name)}</text>')
        o.append(f'<text x="{x2+10:.1f}" y="{y+bar/2+5}" font-family=\'{FONT}\' font-size="15" font-weight="600" fill="{t["text"]}">{v:.1f}</text>')
        if i == 0:   # the ratio, as a pill after our value (the blog chart shows the same)
            px = x2 + 56
            o.append(f'<rect x="{px:.1f}" y="{y+bar/2-12}" width="52" height="24" rx="12" fill="{t["pill"]}" stroke="{t["border"]}"/>')
            o.append(f'<text x="{px+26:.1f}" y="{y+bar/2+5}" text-anchor="middle" font-family=\'{FONT}\' font-size="13" font-weight="650" fill="{t["text"]}">{rows[0][1] / rows[1][1]:.1f}×</text>')
    o.append(f'<text x="{(L+R)/2:.0f}" y="{yb+46}" text-anchor="middle" font-family=\'{FONT}\' font-size="13" fill="{t["muted"]}">turns per minute</text>')
    o.append('</svg>')
    return "\n".join(o)


for name, t in T.items():
    open(os.path.join(OUT, f"vs-merlin-{name}.svg"), "w").write(ratio_chart(t))
    open(os.path.join(OUT, f"vs-merlin-swarm-{name}.svg"), "w").write(swarm_chart(t))
print("ok")
