#!/usr/bin/env python3
"""Graphical abstract (RSC TOC entry, 80 x 40 mm) for the deCIFer evaluation paper."""
import math
import random

W, H = 80.0, 40.0

# paper palette (from the manuscript preamble)
SOFTBLUE = "#6699CC"
SOFTBLUE_D = "#4A7FB5"
SOFTORANGE = "#FFB75E"
SOFTGREEN = "#AADE87"
GREEN_D = "#5A9B3C"
SOFTRED = "#FF8282"
RED_D = "#C0504D"
TEXTGRAY = "#505050"
CREAM = "#DED9C9"

random.seed(11)


def fmt(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def trace(panel, peaks, sigma, background, noise, n=170):
    """Return polyline points for a PXRD trace inside panel=(x0,y0,x1,y1)."""
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
    pts = []
    for i, v in enumerate(vals):
        t = i / (n - 1)
        px = x0 + t * w
        py = y1 - (v / vmax) * (h - 0.4)
        pts.append((px, py))
    return pts


def polyline(pts):
    return " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)


def area_path(pts, ybase):
    d = f"M{fmt(pts[0][0])},{fmt(ybase)} "
    d += " ".join(f"L{fmt(x)},{fmt(y)}" for x, y in pts)
    d += f" L{fmt(pts[-1][0])},{fmt(ybase)} Z"
    return d


PEAKS = [(0.08, 0.95), (0.20, 0.50), (0.36, 1.00), (0.50, 0.42),
         (0.64, 0.58), (0.78, 0.33), (0.90, 0.22)]

# --- panels -------------------------------------------------------------------
TOP_PANEL = (3.5, 6.2, 22.5, 15.8)
BOT_PANEL = (3.5, 23.2, 22.5, 32.8)

sharp = trace(TOP_PANEL, PEAKS, 0.012, lambda t: 0.02, 0.0)
broad = trace(BOT_PANEL, PEAKS, 0.042, lambda t: 0.22 * math.exp(-1.8 * t) + 0.11, 0.015)


def cube(cx, cy, s, d, edge, fill_a, fill_b, opacity=1.0, sw=0.28, skew=1.0):
    """Axonometric unit cell centred at (cx, cy); skew stretches width."""
    sx = s * skew
    fx0, fy0 = cx - sx / 2 - d / 2, cy - s / 2 + d / 2
    fx1, fy1 = fx0 + sx, fy0 + s
    bx0, by0 = fx0 + d, fy0 - d
    bx1, by1 = bx0 + sx, by0 + s
    g = [f'<g opacity="{opacity}">']
    # back square + connectors
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
    # front square
    g.append(
        f'<path d="M{fmt(fx0)},{fmt(fy0)} H{fmt(fx1)} V{fmt(fy1)} H{fmt(fx0)} Z" '
        f'fill="none" stroke="{edge}" stroke-width="{sw}"/>'
    )
    # atoms: back corners, centre, front corners
    for (px, py) in [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]:
        g.append(f'<circle cx="{fmt(px)}" cy="{fmt(py)}" r="0.62" fill="{fill_a}" opacity="0.6"/>')
    g.append(
        f'<circle cx="{fmt((fx0 + bx1) / 2)}" cy="{fmt((fy0 + by1) / 2)}" r="1.05" '
        f'fill="{fill_b}" stroke="{edge}" stroke-width="0.15"/>'
    )
    for (px, py) in [(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1)]:
        g.append(
            f'<circle cx="{fmt(px)}" cy="{fmt(py)}" r="0.85" fill="{fill_a}" '
            f'stroke="{edge}" stroke-width="0.15"/>'
        )
    g.append("</g>")
    return "".join(g)


svg = []
svg.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="80mm" height="40mm" '
    f'viewBox="0 0 {fmt(W)} {fmt(H)}" font-family="Helvetica, Arial, sans-serif">'
)

# defs: arrow markers + gradients
svg.append("<defs>")
for name, color in [("aBlue", SOFTBLUE_D), ("aRed", RED_D), ("aGreen", GREEN_D), ("aOrange", "#E09A3E")]:
    svg.append(
        f'<marker id="{name}" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4.5" '
        f'markerHeight="4.5" orient="auto-start-reverse">'
        f'<path d="M0,0.4 L5.6,3 L0,5.6 Z" fill="{color}"/></marker>'
    )
svg.append(
    f'<linearGradient id="boxGrad" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0" stop-color="{SOFTBLUE}"/>'
    f'<stop offset="1" stop-color="{SOFTBLUE_D}"/></linearGradient>'
)
svg.append("</defs>")

# background
svg.append(f'<rect x="0" y="0" width="{fmt(W)}" height="{fmt(H)}" fill="#ffffff"/>')

# regime bands
svg.append('<rect x="1" y="4.6" width="78" height="15.4" rx="2" fill="#EFF6EA"/>')
svg.append('<rect x="1" y="21.2" width="78" height="16.4" rx="2" fill="#FCEFEB"/>')

# headers
svg.append(
    f'<text x="3.5" y="3.4" font-size="2.5" font-weight="bold" fill="{TEXTGRAY}">'
    "Real-world PXRD</text>"
)
svg.append(
    f'<text x="76.5" y="3.4" font-size="2.5" font-weight="bold" fill="{TEXTGRAY}" '
    'text-anchor="end">Candidate structures</text>'
)

# --- left: PXRD panels ---------------------------------------------------------
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

svg.append(
    f'<text x="13" y="18.6" font-size="2.0" fill="{GREEN_D}" text-anchor="middle" '
    'font-weight="bold">informative</text>'
)
svg.append(
    f'<text x="13" y="35.6" font-size="2.0" fill="{RED_D}" text-anchor="middle" '
    'font-weight="bold">degraded</text>'
)
svg.append(
    f'<text x="13" y="37.9" font-size="1.55" fill="{RED_D}" text-anchor="middle" '
    'opacity="0.85">noise &#183; broadening &#183; background</text>'
)

# --- middle: deCIFer box -------------------------------------------------------
svg.append('<rect x="30.4" y="11.9" width="18" height="17" rx="1.6" fill="#000000" opacity="0.10"/>')
svg.append('<rect x="30" y="11.5" width="18" height="17" rx="1.6" fill="url(#boxGrad)"/>')
svg.append(
    '<text x="39" y="17.1" font-size="3.3" font-weight="bold" fill="#ffffff" '
    'text-anchor="middle">deCIFer</text>'
)
svg.append(
    '<text x="39" y="19.9" font-size="1.75" fill="#ffffff" text-anchor="middle" '
    'opacity="0.9">PXRD &#8594; CIF</text>'
)
# token row with attention arcs
tok_y = 24.6
tok_xs = [31.9 + i * 2.05 for i in range(7)]
for i, tx in enumerate(tok_xs[:-1]):
    # attention arcs into the next-token slot
    lx = tok_xs[-1] + 0.7
    svg.append(
        f'<path d="M{fmt(tx + 0.7)},{fmt(tok_y)} Q{fmt((tx + lx) / 2)},'
        f'{fmt(tok_y - 2.6 - 0.25 * i)} {fmt(lx)},{fmt(tok_y)}" fill="none" '
        f'stroke="#ffffff" stroke-width="0.18" opacity="0.55"/>'
    )
for i, tx in enumerate(tok_xs):
    fill = SOFTORANGE if i == len(tok_xs) - 1 else "#ffffff"
    op = "1" if i == len(tok_xs) - 1 else "0.75"
    svg.append(
        f'<rect x="{fmt(tx)}" y="{fmt(tok_y)}" width="1.4" height="1.4" rx="0.3" '
        f'fill="{fill}" opacity="{op}"/>'
    )
svg.append(
    f'<text x="39" y="31.6" font-size="1.75" font-style="italic" fill="{TEXTGRAY}" '
    'text-anchor="middle">artefact-aware evaluation</text>'
)

# --- flow arrows ---------------------------------------------------------------
svg.append(
    f'<path d="M23.2,11 C26.5,11 27,13 29.3,13.6" fill="none" stroke="{SOFTBLUE_D}" '
    'stroke-width="0.5" marker-end="url(#aBlue)"/>'
)
svg.append(
    f'<path d="M23.2,28.6 C26.5,28.6 27,27 29.3,26.4" fill="none" stroke="{RED_D}" '
    'stroke-width="0.5" marker-end="url(#aRed)"/>'
)
svg.append(
    f'<path d="M48.7,13.6 C51.5,13 52,11.2 55,10.9" fill="none" stroke="{GREEN_D}" '
    'stroke-width="0.5" marker-end="url(#aGreen)"/>'
)
svg.append(
    f'<path d="M48.7,26.4 C51.5,27 52,28.6 55,28.9" fill="none" stroke="#E09A3E" '
    'stroke-width="0.5" marker-end="url(#aOrange)"/>'
)

# --- right, top: confident match ------------------------------------------------
svg.append(cube(63.5, 11.3, 6.2, 2.3, TEXTGRAY, SOFTBLUE, SOFTORANGE))
svg.append(
    f'<path d="M70.6,10.6 l1.5,1.8 2.8,-3.8" fill="none" stroke="{GREEN_D}" '
    'stroke-width="0.95" stroke-linecap="round" stroke-linejoin="round"/>'
)
svg.append(
    f'<text x="66" y="18.6" font-size="2.0" fill="{GREEN_D}" text-anchor="middle" '
    'font-weight="bold">confident match</text>'
)

# --- right, bottom: diverse candidates ------------------------------------------
svg.append(cube(60.6, 28.0, 4.6, 1.7, RED_D, SOFTRED, CREAM, opacity=0.45, skew=0.82))
svg.append(cube(69.4, 29.6, 4.6, 1.7, RED_D, SOFTRED, CREAM, opacity=0.45, skew=1.22))
svg.append(cube(65.0, 28.8, 4.9, 1.8, RED_D, SOFTRED, SOFTORANGE, opacity=0.85))
svg.append(
    f'<text x="74.6" y="28.4" font-size="3.4" font-weight="bold" fill="{RED_D}" '
    'text-anchor="middle">&#177;</text>'
)
svg.append(
    f'<text x="66" y="35.6" font-size="2.0" fill="{RED_D}" text-anchor="middle" '
    'font-weight="bold">diverse candidates</text>'
)
svg.append(
    f'<text x="66" y="37.9" font-size="1.55" fill="{RED_D}" text-anchor="middle" '
    'opacity="0.85">calibrated uncertainty</text>'
)

svg.append("</svg>")

out = "/home/frederik/phd/papers/tackling/deCIFer_realCSP/graphical_abstract.svg"
with open(out, "w") as f:
    f.write("\n".join(svg) + "\n")

# validate well-formedness
import xml.etree.ElementTree as ET
ET.parse(out)
print("valid SVG written:", out)

sentence = (
    "Artefact-aware tests and experimental benchmarks show that PXRD-conditioned "
    "generative crystal structure prediction is accurate when diffraction data are "
    "informative, and flags its own uncertainty when the pattern underdetermines the "
    "structure."
)
print("TOC sentence chars:", len(sentence))
print(sentence)
