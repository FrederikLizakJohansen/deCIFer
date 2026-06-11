#!/usr/bin/env python3
"""Graphical abstract v2 (RSC TOC entry, 80 x 40 mm) for the deCIFer evaluation paper.

Story (left to right): real-world PXRD arrives in two regimes (informative /
degraded); one PXRD-conditioned generative model, with optional composition and
space-group conditioning, maps them to candidate structures; the outcome differs:
a confident match with a narrow lattice distribution, or diverse candidates with
a wide one — accuracy plus calibrated uncertainty.
"""
import math
import random

W, H = 80.0, 40.0

SOFTBLUE = "#6699CC"
SOFTBLUE_D = "#4A7FB5"
SOFTORANGE = "#FFB75E"
ORANGE_D = "#E09A3E"
SOFTGREEN = "#AADE87"
GREEN_D = "#5A9B3C"
SOFTRED = "#FF8282"
RED_D = "#C0504D"
TEXTGRAY = "#505050"
CREAM = "#DED9C9"

random.seed(11)


def fmt(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------ primitives
def trace(panel, peaks, sigma, background, noise, n=170):
    x0, y0, x1, y1 = panel
    w, h = x1 - x0, y1 - y0
    vals = []
    for i in range(n):
        t = i / (n - 1)
        v = background(t)
        for p, a in peaks:
            v += a * math.exp(-0.5 * ((t - p) / sigma) ** 2)
        v += random.gauss(0.0, noise)
        vals.append(max(v, 0.0))
    vmax = max(vals)
    return [
        (x0 + (i / (n - 1)) * w, y1 - (v / vmax) * (h - 0.4))
        for i, v in enumerate(vals)
    ]


def polyline(pts):
    return " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)


def area_path(pts, ybase):
    d = f"M{fmt(pts[0][0])},{fmt(ybase)} "
    d += " ".join(f"L{fmt(x)},{fmt(y)}" for x, y in pts)
    d += f" L{fmt(pts[-1][0])},{fmt(ybase)} Z"
    return d


def arrow(p0, c1, c2, p3, color, sw=0.5, head=1.7, half=0.95):
    """Cubic-bezier arrow with a hand-built head: the shaft is shortened so it
    never pokes out in front of the tip."""
    dx, dy = p3[0] - c2[0], p3[1] - c2[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    base = (p3[0] - ux * head, p3[1] - uy * head)        # head base on the path
    shaft_end = (p3[0] - ux * (head - 0.05), p3[1] - uy * (head - 0.05))
    nx, ny = -uy, ux
    w1 = (base[0] + nx * half, base[1] + ny * half)
    w2 = (base[0] - nx * half, base[1] - ny * half)
    return (
        f'<path d="M{fmt(p0[0])},{fmt(p0[1])} C{fmt(c1[0])},{fmt(c1[1])} '
        f'{fmt(c2[0])},{fmt(c2[1])} {fmt(shaft_end[0])},{fmt(shaft_end[1])}" '
        f'fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="butt"/>'
        f'<path d="M{fmt(p3[0])},{fmt(p3[1])} L{fmt(w1[0])},{fmt(w1[1])} '
        f'L{fmt(w2[0])},{fmt(w2[1])} Z" fill="{color}"/>'
    )


def cube(cx, cy, s, d, edge, fill_a, fill_b, opacity=1.0, sw=0.28, skew=1.0):
    sx = s * skew
    fx0, fy0 = cx - sx / 2 - d / 2, cy - s / 2 + d / 2
    fx1, fy1 = fx0 + sx, fy0 + s
    bx0, by0 = fx0 + d, fy0 - d
    bx1, by1 = bx0 + sx, by0 + s
    g = [f'<g opacity="{opacity}">']
    g.append(
        f'<path d="M{fmt(bx0)},{fmt(by0)} H{fmt(bx1)} V{fmt(by1)} H{fmt(bx0)} Z" '
        f'fill="none" stroke="{edge}" stroke-width="{sw * 0.8}" opacity="0.65"/>'
    )
    for (px, py, qx, qy) in [(fx0, fy0, bx0, by0), (fx1, fy0, bx1, by0),
                             (fx1, fy1, bx1, by1), (fx0, fy1, bx0, by1)]:
        g.append(
            f'<line x1="{fmt(px)}" y1="{fmt(py)}" x2="{fmt(qx)}" y2="{fmt(qy)}" '
            f'stroke="{edge}" stroke-width="{sw * 0.8}" opacity="0.65"/>'
        )
    g.append(
        f'<path d="M{fmt(fx0)},{fmt(fy0)} H{fmt(fx1)} V{fmt(fy1)} H{fmt(fx0)} Z" '
        f'fill="none" stroke="{edge}" stroke-width="{sw}"/>'
    )
    for (px, py) in [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]:
        g.append(f'<circle cx="{fmt(px)}" cy="{fmt(py)}" r="0.58" fill="{fill_a}" opacity="0.6"/>')
    g.append(
        f'<circle cx="{fmt((fx0 + bx1) / 2)}" cy="{fmt((fy0 + by1) / 2)}" r="1.0" '
        f'fill="{fill_b}" stroke="{edge}" stroke-width="0.15"/>'
    )
    for (px, py) in [(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1)]:
        g.append(
            f'<circle cx="{fmt(px)}" cy="{fmt(py)}" r="0.8" fill="{fill_a}" '
            f'stroke="{edge}" stroke-width="0.15"/>'
        )
    g.append("</g>")
    return "".join(g)


def distribution(x0, x1, ybase, height, sigma, stroke, fill, n=60):
    """Small gaussian glyph on a baseline: the lattice-parameter spread."""
    pts = []
    for i in range(n):
        t = i / (n - 1)
        v = math.exp(-0.5 * ((t - 0.5) / sigma) ** 2)
        pts.append((x0 + t * (x1 - x0), ybase - v * height))
    g = (
        f'<path d="{area_path(pts, ybase)}" fill="{fill}" opacity="0.30"/>'
        f'<polyline points="{polyline(pts)}" fill="none" stroke="{stroke}" '
        f'stroke-width="0.35" stroke-linejoin="round"/>'
        f'<line x1="{fmt(x0)}" y1="{fmt(ybase)}" x2="{fmt(x1)}" y2="{fmt(ybase)}" '
        f'stroke="#B9B9B9" stroke-width="0.22"/>'
    )
    return g


# ------------------------------------------------------------------ layout
TOP_BAND = (1.0, 4.8, 79.0, 20.4)     # x0, y0, x1, y1
BOT_BAND = (1.0, 21.6, 79.0, 38.8)
TOP_PANEL = (4.0, 6.6, 23.0, 15.4)
BOT_PANEL = (4.0, 23.6, 23.0, 32.4)
BOX = (29.5, 10.4, 48.5, 29.6)        # deCIFer box

PEAKS = [(0.08, 0.95), (0.20, 0.50), (0.36, 1.00), (0.50, 0.42),
         (0.64, 0.58), (0.78, 0.33), (0.90, 0.22)]

sharp = trace(TOP_PANEL, PEAKS, 0.012, lambda t: 0.02, 0.0)
broad = trace(BOT_PANEL, PEAKS, 0.042, lambda t: 0.22 * math.exp(-1.8 * t) + 0.11, 0.015)

svg = []
svg.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="80mm" height="40mm" '
    f'viewBox="0 0 {fmt(W)} {fmt(H)}" font-family="Helvetica, Arial, sans-serif">'
)
svg.append("<defs>")
svg.append(
    f'<linearGradient id="boxGrad" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0" stop-color="{SOFTBLUE}"/>'
    f'<stop offset="1" stop-color="{SOFTBLUE_D}"/></linearGradient>'
)
svg.append("</defs>")

svg.append(f'<rect x="0" y="0" width="{fmt(W)}" height="{fmt(H)}" fill="#ffffff"/>')

# regime bands
for (x0, y0, x1, y1), fill in [(TOP_BAND, "#EFF6EA"), (BOT_BAND, "#FCEFEB")]:
    svg.append(
        f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(x1 - x0)}" '
        f'height="{fmt(y1 - y0)}" rx="2" fill="{fill}"/>'
    )

# headers, aligned with the content margins (x = 4 .. 78)
svg.append(
    f'<text x="4" y="3.5" font-size="2.5" font-weight="bold" fill="{TEXTGRAY}">'
    "Real-world PXRD</text>"
)
svg.append(
    f'<text x="78" y="3.5" font-size="2.5" font-weight="bold" fill="{TEXTGRAY}" '
    'text-anchor="end">Candidate structures</text>'
)

# ---------------------------------------------------------------- left panels
for (x0, y0, x1, y1) in (TOP_PANEL, BOT_PANEL):
    svg.append(
        f'<path d="M{fmt(x0)},{fmt(y0)} V{fmt(y1)} H{fmt(x1)}" fill="none" '
        f'stroke="#B9B9B9" stroke-width="0.25"/>'
    )
svg.append(f'<path d="{area_path(sharp, TOP_PANEL[3])}" fill="{SOFTBLUE}" opacity="0.18"/>')
svg.append(
    f'<polyline points="{polyline(sharp)}" fill="none" stroke="{SOFTBLUE_D}" '
    f'stroke-width="0.42" stroke-linejoin="round"/>'
)
svg.append(f'<path d="{area_path(broad, BOT_PANEL[3])}" fill="{SOFTRED}" opacity="0.18"/>')
svg.append(
    f'<polyline points="{polyline(broad)}" fill="none" stroke="{RED_D}" '
    f'stroke-width="0.42" stroke-linejoin="round"/>'
)

# panel labels: one bold row per band (top band) / bold + sub row (bottom band),
# all with >= 1.3 mm clearance from the band edges
svg.append(
    f'<text x="13.5" y="18.3" font-size="2.0" fill="{GREEN_D}" text-anchor="middle" '
    'font-weight="bold">informative</text>'
)
svg.append(
    f'<text x="13.5" y="35.2" font-size="2.0" fill="{RED_D}" text-anchor="middle" '
    'font-weight="bold">degraded</text>'
)
svg.append(
    f'<text x="13.5" y="37.4" font-size="1.55" fill="{RED_D}" text-anchor="middle" '
    'opacity="0.85">noise &#183; broadening &#183; background</text>'
)

# ---------------------------------------------------------------- model box
bx0, by0, bx1, by1 = BOX
svg.append(
    f'<rect x="{fmt(bx0 + 0.4)}" y="{fmt(by0 + 0.4)}" width="{fmt(bx1 - bx0)}" '
    f'height="{fmt(by1 - by0)}" rx="1.6" fill="#000000" opacity="0.10"/>'
)
svg.append(
    f'<rect x="{fmt(bx0)}" y="{fmt(by0)}" width="{fmt(bx1 - bx0)}" '
    f'height="{fmt(by1 - by0)}" rx="1.6" fill="url(#boxGrad)"/>'
)
cx = (bx0 + bx1) / 2
svg.append(
    f'<text x="{fmt(cx)}" y="15.6" font-size="3.3" font-weight="bold" fill="#ffffff" '
    'text-anchor="middle">deCIFer</text>'
)
svg.append(
    f'<text x="{fmt(cx)}" y="18.3" font-size="1.75" fill="#ffffff" text-anchor="middle" '
    'opacity="0.9">PXRD &#8594; CIF</text>'
)
# token row with attention arcs (autoregressive generation)
tok_y = 22.9
tok_xs = [bx0 + 1.9 + i * 2.05 for i in range(7)]
for i, tx in enumerate(tok_xs[:-1]):
    lx = tok_xs[-1] + 0.7
    svg.append(
        f'<path d="M{fmt(tx + 0.7)},{fmt(tok_y)} Q{fmt((tx + lx) / 2)},'
        f'{fmt(tok_y - 2.4 - 0.22 * i)} {fmt(lx)},{fmt(tok_y)}" fill="none" '
        f'stroke="#ffffff" stroke-width="0.18" opacity="0.55"/>'
    )
for i, tx in enumerate(tok_xs):
    fill = SOFTORANGE if i == len(tok_xs) - 1 else "#ffffff"
    op = "1" if i == len(tok_xs) - 1 else "0.75"
    svg.append(
        f'<rect x="{fmt(tx)}" y="{fmt(tok_y)}" width="1.4" height="1.4" rx="0.3" '
        f'fill="{fill}" opacity="{op}"/>'
    )
# the study's framing, inside the box so it never straddles a band edge
svg.append(
    f'<text x="{fmt(cx)}" y="27.9" font-size="1.6" font-style="italic" fill="#ffffff" '
    'text-anchor="middle" opacity="0.92">artefact-aware evaluation</text>'
)

# optional conditioning chips feeding the model (the user's levers)
for (cx0, cw, label) in [(26.2, 12.3, "+ composition"), (39.3, 12.3, "+ space group")]:
    ccx = cx0 + cw / 2
    svg.append(
        f'<rect x="{fmt(cx0)}" y="32.4" width="{fmt(cw)}" height="3.1" rx="1.0" '
        f'fill="#ffffff" stroke="{SOFTBLUE_D}" stroke-width="0.25"/>'
    )
    svg.append(
        f'<text x="{fmt(ccx)}" y="34.55" font-size="1.6" fill="{SOFTBLUE_D}" '
        f'text-anchor="middle">{label}</text>'
    )
    svg.append(arrow((ccx, 32.2), (ccx, 31.6), (ccx, 31.0), (ccx, 30.0),
                     SOFTBLUE_D, sw=0.35, head=1.1, half=0.65))

# ---------------------------------------------------------------- flow arrows
svg.append(arrow((23.6, 10.6), (26.8, 10.6), (27.4, 12.8), (29.1, 13.5), SOFTBLUE_D))
svg.append(arrow((23.6, 29.4), (26.8, 29.4), (27.4, 27.2), (29.1, 26.5), RED_D))
svg.append(arrow((48.9, 13.5), (51.0, 12.8), (51.8, 10.8), (54.4, 10.5), GREEN_D))
svg.append(arrow((48.9, 26.5), (51.0, 27.2), (51.8, 29.2), (54.4, 29.5), ORANGE_D))

# ---------------------------------------------------------------- outputs
# top: confident match — one cell, check, narrow lattice distribution
svg.append(cube(61.0, 11.4, 6.0, 2.2, TEXTGRAY, SOFTBLUE, SOFTORANGE))
svg.append(
    f'<path d="M66.9,10.0 l1.4,1.7 2.6,-3.5" fill="none" stroke="{GREEN_D}" '
    'stroke-width="0.9" stroke-linecap="round" stroke-linejoin="round"/>'
)
svg.append(distribution(72.4, 78.0, 15.6, 7.2, 0.055, GREEN_D, SOFTGREEN))
svg.append(
    f'<text x="65.8" y="18.3" font-size="2.0" fill="{GREEN_D}" text-anchor="middle" '
    'font-weight="bold">confident match</text>'
)

# bottom: diverse candidates — ghost ensemble, wide lattice distribution
svg.append(cube(58.0, 27.6, 4.4, 1.6, RED_D, SOFTRED, CREAM, opacity=0.45, skew=0.82))
svg.append(cube(66.0, 28.8, 4.4, 1.6, RED_D, SOFTRED, CREAM, opacity=0.45, skew=1.22))
svg.append(cube(62.0, 28.1, 4.7, 1.7, RED_D, SOFTRED, SOFTORANGE, opacity=0.85))
svg.append(distribution(70.6, 78.0, 32.4, 4.6, 0.24, RED_D, SOFTRED))
svg.append(
    f'<text x="65.8" y="35.2" font-size="2.0" fill="{RED_D}" text-anchor="middle" '
    'font-weight="bold">diverse candidates</text>'
)
svg.append(
    f'<text x="65.8" y="37.4" font-size="1.55" fill="{RED_D}" text-anchor="middle" '
    'opacity="0.85">calibrated uncertainty</text>'
)

svg.append("</svg>")

out = "/home/frederik/phd/papers/tackling/deCIFer_realCSP/graphical_abstract_v2.svg"
with open(out, "w") as f:
    f.write("\n".join(svg) + "\n")

import xml.etree.ElementTree as ET
ET.parse(out)
print("valid SVG written:", out)
