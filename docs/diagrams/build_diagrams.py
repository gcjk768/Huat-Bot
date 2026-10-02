"""Build the README diagrams.

One layout model produces three files per diagram:

* ``<name>.drawio``      editable draw.io source (open in diagrams.net or the VS Code draw.io extension)
* ``<name>-light.svg``   rendered for GitHub light mode
* ``<name>-dark.svg``    rendered for GitHub dark mode

Brand logos are the official marks from Simple Icons (CC0, see icons/SIMPLE_ICONS_LICENSE.md),
drawn in each brand's own colour. The small line glyphs are drawn here.

Run:  python docs/diagrams/build_diagrams.py
Only the standard library is needed.
"""
from __future__ import annotations

import base64
import html
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
ICONS = HERE / "icons"

FONT = "'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Helvetica, Arial, sans-serif"

# Brand colours published with each logo.
BRAND = {
    "docker": "#2496ED",
    "telegram": "#26A5E4",
    "obsidian": "#7C3AED",
    "python": "#3776AB",
    "claude": "#D97757",
    "pandas": "#150458",
    "numpy": "#013243",
}

ACCENT = {
    "pools": "#E11D48",
    "scrape": "#0EA5E9",
    "tickets": "#8B5CF6",
    "wait": "#F59E0B",
    "outlook": "#10B981",
    "signal": "#EAB308",
    "alert": "#EF4444",
    "vault": "#7C3AED",
    "post": "#26A5E4",
    "docker": "#2496ED",
    "claude": "#D97757",
    "nas": "#64748B",
}

THEMES = {
    "light": dict(
        bg1="#FFFFFF", bg2="#EEF2F7", dots="#CBD5E1", glow="#FDE68A",
        card="#FFFFFF", stroke="#E2E8F0", text="#0F172A", muted="#475569", faint="#94A3B8",
        box="#F8FAFC", box_stroke="#CBD5E1", tile="#FFFFFF", edge="#64748B",
        shadow="#0F172A", shadow_op=0.09, tint=0.12, title_a="#DC2626", title_b="#D97706",
    ),
    "dark": dict(
        bg1="#0B1220", bg2="#111A2E", dots="#1F2A40", glow="#78350F",
        card="#141D2F", stroke="#263449", text="#F1F5F9", muted="#A8B5C8", faint="#64748B",
        box="#0F1728", box_stroke="#334155", tile="#F8FAFC", edge="#94A3B8",
        shadow="#000000", shadow_op=0.45, tint=0.20, title_a="#F87171", title_b="#FBBF24",
    ),
}

# Simple line glyphs on a 24 x 24 grid (stroke drawn in the accent colour).
GLYPHS = {
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/>'
             '<path d="M12 3c3 3.2 3 14.8 0 18M12 3c-3 3.2-3 14.8 0 18"/>',
    "nas": '<rect x="5" y="2.5" width="14" height="19" rx="2.2"/><path d="M8.5 7h7M8.5 10.5h7M8.5 14h7"/>'
           '<circle cx="9" cy="18" r="0.9" fill="currentColor" stroke="none"/>'
           '<circle cx="12" cy="18" r="0.9" fill="currentColor" stroke="none"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5.2l3.4 2"/>',
    "ball": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.2"/>',
    "bell": '<path d="M6 16.5v-5a6 6 0 0 1 12 0v5l1.6 2H4.4z"/><path d="M10 21a2.2 2.2 0 0 0 4 0"/>',
    "download": '<path d="M12 3.5v11M7 10l5 5 5-5M4.5 20h15"/>',
    "chart": '<path d="M4 20.5h16.5"/><path d="M6.5 17v-5M11 17V6.5M15.5 17v-8M20 17v-3"/>',
    "history": '<path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1L3.5 8.5"/><path d="M3.5 3.5v5h5"/><path d="M12 8v4.3l3 1.7"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/>'
              '<circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none"/>',
    "ticket": '<path d="M3 8.5a2.5 2.5 0 0 0 0 7V18a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1v-2.5a2.5 2.5 0 0 1 0-7V6'
              'a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1z"/><path d="M14 5v14" stroke-dasharray="2 2"/>',
    "send": '<path d="M21 3 10.5 13.5M21 3l-6.5 18-4-7.5-7.5-4z"/>',
}


# draw.io hosted copies of the same official logos, used by the lightweight variant that is
# opened through the draw.io MCP server (keeps that XML small).
DRAWIO_LIBRARY = {
    "docker": "https://app.diagrams.net/img/lib/mscae/Docker.svg",
    "telegram": "https://icons.diagrams.net/icon-cache1/Brands_Pack-2317/telegram-1286.svg",
    "obsidian": "https://icons.diagrams.net/assets/font-awesome/1/Obsidian_brand.svg",
    "claude": "https://icons.diagrams.net/assets/font-awesome/1/Claude_brand.svg",
    "python": "https://icons.diagrams.net/icon-cache1/Scripting_and_programming_languages-2101/Python_logo-1301.svg",
}


def _brand_path(name: str) -> str:
    text = (ICONS / f"{name}.svg").read_text(encoding="utf-8")
    return re.search(r'<path d="([^"]+)"', text).group(1)


def brand_svg(name: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            f'<path fill="{BRAND[name]}" d="{_brand_path(name)}"/></svg>')


def glyph_svg(name: str, color: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" color="{color}">'
            f'{GLYPHS[name]}</svg>')


def data_uri(svg: str) -> str:
    # draw.io convention: no ";base64" marker because ";" separates style keys.
    return "data:image/svg+xml," + base64.b64encode(svg.encode()).decode()


def text_width(s: str, size: float, bold: bool = False) -> float:
    """Rough width of sans serif text, good enough to size pills."""
    w = 0.0
    for ch in s:
        if ch in "il.,:;|!'":
            w += 0.28
        elif ch in "mwMW":
            w += 0.86
        elif ch.isupper() or ch.isdigit():
            w += 0.64
        elif ch == " ":
            w += 0.28
        else:
            w += 0.53
    return w * size * (1.06 if bold else 1.0)


def esc(s: str) -> str:
    return html.escape(s, quote=True)


@dataclass
class Box:
    id: str
    x: float
    y: float
    w: float
    h: float


@dataclass
class Diagram:
    name: str
    w: int
    h: int
    t: dict
    svg: list[str] = field(default_factory=list)
    cells: list[str] = field(default_factory=list)
    boxes: dict[str, Box] = field(default_factory=dict)
    linked: bool = False  # True: reference draw.io hosted logos and skip glyph images
    _n: int = 0

    # helpers

    def nid(self, prefix: str = "c") -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def vertex(self, style: str, x, y, w, h, value: str = "", cid: str | None = None) -> str:
        cid = cid or self.nid()
        self.cells.append(
            f'<mxCell id="{cid}" value="{esc(value)}" style="{esc(style)}" vertex="1" parent="1">'
            f'<mxGeometry x="{round(x)}" y="{round(y)}" width="{math.ceil(w)}" height="{math.ceil(h)}" '
            f'as="geometry"/></mxCell>'
        )
        return cid

    def text(self, x, y, s, size=13, weight=400, color=None, anchor="start", fill=None, spacing=0.0,
             cell=True):
        color = color or self.t["text"]
        ls = f' letter-spacing="{spacing:g}"' if spacing else ""
        self.svg.append(
            f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" font-weight="{weight}" '
            f'fill="{fill or color}" text-anchor="{anchor}"{ls}>{esc(s)}</text>'
        )
        if cell:
            width = text_width(s, size, weight >= 600) + 8
            left = {"start": x, "middle": x - width / 2, "end": x - width}[anchor]
            align = {"start": "left", "middle": "center", "end": "right"}[anchor]
            self.vertex(
                f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;"
                f"spacing=0;fontSize={size:g};fontColor={color};fontStyle={1 if weight >= 600 else 0};"
                f"fontFamily=Helvetica;",
                left, y - size, width, size * 1.35, s,
            )

    def icon(self, x, y, size, *, brand: str | None = None, glyph: str | None = None,
             color: str | None = None, tile: bool = True, tile_size: float | None = None):
        """Icon centred in a rounded tile whose top left corner is (x, y)."""
        t = self.t
        ts = tile_size or size + 18
        if tile:
            if brand:
                fill, stroke, op = t["tile"], t["stroke"], 1.0
            else:
                fill, stroke, op = color, "none", t["tint"]
            self.svg.append(
                f'<rect x="{x:g}" y="{y:g}" width="{ts:g}" height="{ts:g}" rx="{ts * 0.27:g}" '
                f'fill="{fill}" fill-opacity="{op:g}" stroke="{stroke}"/>'
            )
            tint = fill if brand else _mix(color, "#FFFFFF", 1 - 0.12)
            self.vertex(
                f"rounded=1;arcSize=27;whiteSpace=wrap;html=1;fillColor={tint};"
                f"strokeColor={t['stroke'] if brand else 'none'};",
                x, y, ts, ts,
            )
        ix, iy = (x + (ts - size) / 2, y + (ts - size) / 2) if tile else (x, y)
        s = size / 24
        if brand:
            self.svg.append(
                f'<g transform="translate({ix:g},{iy:g}) scale({s:g})">'
                f'<path fill="{BRAND[brand]}" d="{_brand_path(brand)}"/></g>'
            )
            img = brand_svg(brand)
            if self.linked:
                if brand not in DRAWIO_LIBRARY:
                    return
                self.vertex(f"shape=image;html=1;aspect=fixed;imageAspect=0;image={DRAWIO_LIBRARY[brand]};",
                            ix, iy, size, size)
                return
        else:
            self.svg.append(
                f'<g transform="translate({ix:g},{iy:g}) scale({s:g})" fill="none" stroke="{color}" '
                f'stroke-width="{1.8:g}" stroke-linecap="round" stroke-linejoin="round" color="{color}">'
                f"{GLYPHS[glyph]}</g>"
            )
            img = glyph_svg(glyph, color)
            if self.linked:
                return
        self.vertex(
            f"shape=image;html=1;aspect=fixed;imageAspect=0;image={data_uri(img)};",
            ix, iy, size, size,
        )

    def card(self, cid, x, y, w, h, title, lines=(), *, accent, brand=None, glyph=None, dashed=False,
             title_size=16, line_gap=21, step=None, icon_right=False, line_size=13):
        t = self.t
        self.boxes[cid] = Box(cid, x, y, w, h)
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        self.svg.append(
            f'<g filter="url(#shadow)"><rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="14" '
            f'fill="{t["card"]}" stroke="{accent if dashed else t["stroke"]}" stroke-width="{1.4 if dashed else 1}"'
            f"{dash}/></g>"
        )
        clip = self.nid("clip")
        self.svg.append(
            f'<clipPath id="{clip}"><rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="14"/></clipPath>'
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="4" fill="{accent}" clip-path="url(#{clip})"/>'
        )
        self.vertex(
            f"rounded=1;arcSize={max(4, round(1400 / min(w, h)))};whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
            f"strokeColor={accent if dashed else '#E2E8F0'};shadow=1;{'dashed=1;' if dashed else ''}",
            x, y, w, h, cid=cid,
        )
        self.vertex(f"rounded=0;html=1;fillColor={accent};strokeColor=none;", x + 8, y, w - 16, 4)
        has_icon = bool(brand or glyph)
        if step is not None:
            # numbered badge top left, icon top right, title and lines below
            self.svg.append(
                f'<circle cx="{x + 32:g}" cy="{y + 36:g}" r="15" fill="{accent}"/>'
                f'<text x="{x + 32:g}" y="{y + 41:g}" font-size="14" font-weight="700" fill="#FFFFFF" '
                f'text-anchor="middle">{step}</text>'
            )
            self.vertex(
                f"ellipse;html=1;fillColor={accent};strokeColor=none;fontColor=#FFFFFF;fontStyle=1;fontSize=14;",
                x + 17, y + 21, 30, 30, str(step),
            )
            if has_icon:
                self.icon(x + w - 52, y + 18, 22, brand=brand, glyph=glyph, color=accent, tile_size=36)
            tx, ty, lx = x + 18, y + 84, x + 18
        elif has_icon and icon_right:
            self.icon(x + w - 52, y + 16, 22, brand=brand, glyph=glyph, color=accent, tile_size=36)
            tx, ty, lx = x + 18, y + 42, x + 18
        elif has_icon:
            # icon tile left, title beside it, lines below the tile
            self.icon(x + 16, y + 20, 26, brand=brand, glyph=glyph, color=accent, tile_size=44)
            tx, ty, lx = x + 72, y + 48, x + 18
        else:
            tx, ty, lx = x + 18, y + 40, x + 18
        self.text(tx, ty, title, size=title_size, weight=700)
        ly = ty + line_gap + 4 if lx == tx else y + 90
        for line in lines:
            self.text(lx, ly, line, size=line_size, color=t["muted"])
            ly += line_gap

    def container(self, cid, x, y, w, h, label, *, glyph=None, brand=None, accent=None):
        t = self.t
        self.boxes[cid] = Box(cid, x, y, w, h)
        self.svg.append(
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="22" fill="{t["box"]}" '
            f'stroke="{t["box_stroke"]}" stroke-width="1.5" stroke-dasharray="2 0"/>'
        )
        self.vertex(
            "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;strokeWidth=1.5;",
            x, y, w, h, cid=cid,
        )
        cx = x + 18
        if glyph or brand:
            self.icon(cx, y + 13, 18, glyph=glyph, brand=brand, color=accent or t["muted"], tile=False)
            cx += 28
        self.text(cx, y + 27, label, size=12, weight=700, color=t["muted"], spacing=1.4)

    def pill(self, cx, cy, label, color=None, size=11.5):
        t = self.t
        w = text_width(label, size, True) + 18
        h = size + 11
        self.svg.append(
            f'<rect x="{cx - w / 2:g}" y="{cy - h / 2:g}" width="{w:g}" height="{h:g}" rx="{h / 2:g}" '
            f'fill="{t["card"]}" stroke="{color or t["stroke"]}"/>'
            f'<text x="{cx:g}" y="{cy + size * 0.36:g}" font-size="{size:g}" font-weight="600" '
            f'fill="{color or t["muted"]}" text-anchor="middle">{esc(label)}</text>'
        )
        self.vertex(
            f"rounded=1;arcSize=50;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor={color or '#E2E8F0'};"
            f"fontSize={size:g};fontStyle=1;fontColor={color or '#475569'};",
            cx - w / 2, cy - h / 2, w, h, label,
        )

    def edge(self, src, tgt, pts, *, color=None, dashed=False, both=False, label=None, label_at=None,
             label_color=None):
        """Orthogonal connector through the given points (first on src, last on tgt)."""
        t = self.t
        color = color or t["edge"]
        d = _rounded_path(pts, 10)
        marker_end = f'url(#arrow-{_mid(color)})'
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        start = f' marker-start="{marker_end}"' if both else ""
        self.svg.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round" marker-end="{marker_end}"{start}{dash}/>'
        )
        self._markers.add(color)
        s, g = self.boxes[src], self.boxes[tgt]
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2;endArrow=block;endFill=1;"
            f"strokeColor={color};exitX={(x0 - s.x) / s.w:.3f};exitY={(y0 - s.y) / s.h:.3f};exitDx=0;exitDy=0;"
            f"entryX={(x1 - g.x) / g.w:.3f};entryY={(y1 - g.y) / g.h:.3f};entryDx=0;entryDy=0;"
            + ("dashed=1;" if dashed else "")
            + ("startArrow=block;startFill=1;" if both else "startArrow=none;")
        )
        way = "".join(f'<mxPoint x="{px:g}" y="{py:g}"/>' for px, py in pts[1:-1])
        self.cells.append(
            f'<mxCell id="{self.nid("e")}" style="{esc(style)}" edge="1" parent="1" source="{src}" target="{tgt}">'
            f'<mxGeometry relative="1" as="geometry">{f"<Array as={chr(34)}points{chr(34)}>{way}</Array>" if way else ""}'
            f"</mxGeometry></mxCell>"
        )
        if label:
            lx, ly = label_at or _midpoint(pts)
            self.pill(lx, ly, label, color=label_color)

    def diamond(self, cid, x, y, w, h, label, accent):
        t = self.t
        self.boxes[cid] = Box(cid, x, y, w, h)
        cx, cy = x + w / 2, y + h / 2
        pts = f"{cx:g},{y:g} {x + w:g},{cy:g} {cx:g},{y + h:g} {x:g},{cy:g}"
        self.svg.append(
            f'<g filter="url(#shadow)"><polygon points="{pts}" fill="{t["card"]}" stroke="{accent}" '
            f'stroke-width="2" stroke-linejoin="round"/></g>'
        )
        self.vertex(
            f"rhombus;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor={accent};strokeWidth=2;shadow=1;",
            x, y, w, h, cid=cid,
        )
        self.text(cx, cy + 5, label, size=14, weight=700, anchor="middle")

    # output

    _markers: set = None  # type: ignore[assignment]

    def __post_init__(self):
        self._markers = set()

    def svg_text(self) -> str:
        t = self.t
        markers = "".join(
            f'<marker id="arrow-{_mid(c)}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0.8 L9.5,5 L0,9.2 Z" fill="{c}"/></marker>'
            for c in sorted(self._markers)
        )
        defs = f"""<defs>
<linearGradient id="bg" x1="0" y1="0" x2="0.4" y2="1"><stop offset="0" stop-color="{t['bg1']}"/><stop offset="1" stop-color="{t['bg2']}"/></linearGradient>
<radialGradient id="glow" cx="0.08" cy="0.02" r="0.55"><stop offset="0" stop-color="{t['glow']}" stop-opacity="0.55"/><stop offset="1" stop-color="{t['glow']}" stop-opacity="0"/></radialGradient>
<linearGradient id="title" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{t['title_a']}"/><stop offset="1" stop-color="{t['title_b']}"/></linearGradient>
<pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="{t['dots']}"/></pattern>
<filter id="shadow" x="-15%" y="-15%" width="130%" height="150%"><feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="{t['shadow']}" flood-opacity="{t['shadow_op']}"/></filter>
{markers}
</defs>"""
        frame = (
            f'<rect width="{self.w}" height="{self.h}" rx="26" fill="url(#bg)"/>'
            f'<rect width="{self.w}" height="{self.h}" rx="26" fill="url(#dots)" opacity="0.7"/>'
            f'<rect width="{self.w}" height="{self.h}" rx="26" fill="url(#glow)"/>'
            f'<rect x="0.5" y="0.5" width="{self.w - 1}" height="{self.h - 1}" rx="26" fill="none" stroke="{t["stroke"]}"/>'
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
            f'height="{self.h}" font-family="{esc(FONT)}" role="img" aria-label="{esc(self.name)}">'
            f"<title>{esc(self.name)}</title>{defs}{frame}{''.join(self.svg)}</svg>\n"
        )

    def drawio_text(self) -> str:
        bg = self.vertex(
            "rounded=1;arcSize=3;whiteSpace=wrap;html=1;fillColor=#FFFFFF;gradientColor=#EEF2F7;"
            "strokeColor=#E2E8F0;",
            0, 0, self.w, self.h, cid="bg",
        )
        cells = [self.cells.pop()] + self.cells  # background first so it sits at the back
        return (
            f'<mxfile host="huatbot" type="device"><diagram id="{self.name.replace(" ", "_")}" '
            f'name="{esc(self.name)}"><mxGraphModel dx="1000" dy="700" grid="0" gridSize="10" guides="1" '
            f'tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="{self.w}" '
            f'pageHeight="{self.h}" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
            f"{''.join(cells)}</root></mxGraphModel></diagram></mxfile>\n"
        )


def _mid(color: str) -> str:
    return color.lstrip("#").lower()


def _mix(a: str, b: str, k: float) -> str:
    """Blend colour a towards b by k (0 keeps a)."""
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * k):02X}" for x, y in zip(pa, pb))


def _midpoint(pts):
    # midpoint along the polyline
    segs = list(zip(pts, pts[1:]))
    total = sum(math.dist(p, q) for p, q in segs)
    half, acc = total / 2, 0.0
    for p, q in segs:
        d = math.dist(p, q)
        if acc + d >= half:
            k = (half - acc) / d if d else 0
            return p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k
        acc += d
    return pts[-1]


def _rounded_path(pts, r):
    if len(pts) == 2:
        (x0, y0), (x1, y1) = pts
        return f"M{x0:g},{y0:g} L{x1:g},{y1:g}"
    d = [f"M{pts[0][0]:g},{pts[0][1]:g}"]
    for i in range(1, len(pts) - 1):
        (xa, ya), (xb, yb), (xc, yc) = pts[i - 1], pts[i], pts[i + 1]
        la, lc = math.dist((xa, ya), (xb, yb)), math.dist((xb, yb), (xc, yc))
        rr = min(r, la / 2, lc / 2)
        p1 = (xb + (xa - xb) * rr / la, yb + (ya - yb) * rr / la)
        p2 = (xb + (xc - xb) * rr / lc, yb + (yc - yb) * rr / lc)
        d.append(f"L{p1[0]:g},{p1[1]:g} Q{xb:g},{yb:g} {p2[0]:g},{p2[1]:g}")
    d.append(f"L{pts[-1][0]:g},{pts[-1][1]:g}")
    return " ".join(d)


# Diagram 1: how it fits together

def architecture(theme: str, linked: bool = False) -> Diagram:
    t = THEMES[theme]
    g = Diagram("Huat Bot architecture", 1000, 720, t, linked=linked)

    # header
    g.svg.append(
        f'<text x="40" y="66" font-size="38" font-weight="800" fill="url(#title)" letter-spacing="-0.5">Huat Bot</text>'
    )
    g.vertex("text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=38;"
             "fontStyle=1;fontColor=#DC2626;fontFamily=Helvetica;", 36, 28, 220, 50, "Huat Bot")
    g.text(42, 96, "Singapore Pools TOTO: what the next draws will do and the next big prize", size=15.5, color=t["muted"])
    g.text(764, 44, "BUILT WITH", size=10.5, weight=700, color=t["faint"], spacing=1.6)
    for i, b in enumerate(("python", "pandas", "numpy", "docker")):
        g.icon(764 + i * 52, 54, 20, brand=b, tile_size=38)
    g.svg.append(f'<path d="M40 126 H960" stroke="{t["stroke"]}" stroke-width="1"/>')

    # left column
    g.card("pools", 30, 160, 232, 196, "Singapore Pools",
           ["Result page for every draw", "Next draw and jackpot", "Cascade, Hongbao, special", "Official prize rules"],
           accent=ACCENT["pools"], glyph="globe")
    g.card("claude", 30, 466, 232, 178, "Claude, optional",
           ["claude -p writes a short", "comment from the figures.", "Python computes every number."],
           accent=ACCENT["claude"], brand="claude", dashed=True)

    # NAS
    g.container("nas", 296, 146, 408, 518, "UGREEN NAS", glyph="nas", accent=t["muted"])
    g.card("docker", 318, 186, 364, 214, "huat-bot container",
           ["Fetches only the missing draws", "Checks your tickets",
            "Projects the jackpot to the cascade", "Posts two Telegram messages"],
           accent=ACCENT["docker"], brand="docker")
    # schedule pill inside the docker card
    sx, sy, sw = 336, 354, 328
    g.svg.append(
        f'<rect x="{sx}" y="{sy}" width="{sw}" height="30" rx="15" fill="{ACCENT["docker"]}" fill-opacity="{t["tint"]}"/>'
    )
    g.vertex(f"rounded=1;arcSize=50;html=1;fillColor={_mix(ACCENT['docker'], '#FFFFFF', 0.86)};strokeColor=none;",
             sx, sy, sw, 30)
    g.icon(sx + 12, sy + 6, 18, glyph="clock", color=ACCENT["docker"] if theme == "light" else "#7CC4FA", tile=False)
    g.text(sx + 38, sy + 20, "7.30pm SGT on draw days, retries for 2 hours", size=12.5, weight=600,
           color=ACCENT["docker"] if theme == "light" else "#7CC4FA")

    g.card("vault", 318, 466, 364, 178, "Obsidian vault", [], accent=ACCENT["vault"], brand="obsidian")
    col1, col2, top = 336, 492, 550
    g.text(col1, top, "YOU EDIT", size=10.5, weight=700, color=t["faint"], spacing=1.4)
    g.text(col2, top, "BOT WRITES", size=10.5, weight=700, color=t["faint"], spacing=1.4)
    for i, s in enumerate(("Settings.md", "Tickets.md")):
        g.text(col1, top + 24 + i * 22, s, size=13, weight=600, color=t["text"])
    for i, s in enumerate(("Dashboard and Ledger", "Draw notes and reports", "Activity log, data CSVs")):
        g.text(col2, top + 24 + i * 22, s, size=13, color=t["muted"])
    g.svg.append(f'<path d="M476 540 V630" stroke="{t["stroke"]}"/>')

    # right column
    g.card("telegram", 738, 160, 232, 184, "Telegram channel", [], accent=ACCENT["post"], brand="telegram")
    for i, s in enumerate(("Result and ticket check", "Next draw and big prize")):
        y = 232 + i * 46
        g.svg.append(
            f'<rect x="754" y="{y}" width="200" height="36" rx="10" fill="{ACCENT["post"]}" fill-opacity="{t["tint"]}"/>'
            f'<circle cx="772" cy="{y + 18}" r="10" fill="{ACCENT["post"]}"/>'
            f'<text x="772" y="{y + 22}" font-size="11.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">{i + 1}</text>'
        )
        g.vertex(f"rounded=1;arcSize=25;html=1;fillColor={_mix(ACCENT['post'], '#FFFFFF', 0.86)};strokeColor=none;"
                 "align=left;spacingLeft=30;fontSize=12.5;fontStyle=1;fontColor=#0F172A;", 754, y, 200, 36, s)
        g.vertex(f"ellipse;html=1;fillColor={ACCENT['post']};strokeColor=none;fontColor=#FFFFFF;fontStyle=1;"
                 "fontSize=11.5;", 762, y + 8, 20, 20, str(i + 1))
        g.svg.append(
            f'<text x="790" y="{y + 22.5}" font-size="12.5" font-weight="600" fill="{t["text"]}">{esc(s)}</text>'
        )
    g.card("devices", 738, 466, 232, 178, "Your phone and PC",
           ["Obsidian with the synced", "vault: edit tickets, read", "the dashboard and ledger"],
           accent=ACCENT["vault"], brand="obsidian")

    # connectors
    g.edge("pools", "docker", [(262, 258), (318, 258)], color=ACCENT["pools"])
    g.pill(290, 236, "HTTPS", color=ACCENT["pools"], size=10.5)
    g.edge("docker", "telegram", [(682, 275), (738, 275)], color=ACCENT["post"])
    g.edge("docker", "vault", [(450, 400), (450, 466)], color=ACCENT["vault"], label="writes", label_at=(450, 433))
    g.edge("vault", "docker", [(560, 466), (560, 400)], color=ACCENT["vault"], label="reads", label_at=(560, 433))
    g.edge("vault", "devices", [(682, 555), (738, 555)], color=ACCENT["vault"], both=True)
    g.pill(710, 533, "sync", color=ACCENT["vault"], size=10.5)
    g.edge("claude", "docker", [(146, 466), (146, 420), (300, 420), (300, 330), (318, 330)],
           color=ACCENT["claude"], dashed=True)

    g.text(500, 694, "Every figure comes from Python. Draws are independent, so the bot never suggests numbers.",
           size=12.5, color=t["muted"], anchor="middle")
    return g


# Diagram 2: what happens on a draw day

def draw_day(theme: str, linked: bool = False) -> Diagram:
    t = THEMES[theme]
    g = Diagram("Huat Bot draw day", 1000, 600, t, linked=linked)
    g.text(40, 58, "What happens on a draw day", size=28, weight=800)
    g.text(42, 86, "TOTO on Monday and Thursday at 6.30pm, plus Hongbao and special draws",
           size=14.5, color=t["muted"])
    g.svg.append(f'<path d="M40 110 H960" stroke="{t["stroke"]}" stroke-width="1"/>')

    # row 1: timing and retry loop
    r1 = dict(icon_right=True, line_gap=20, line_size=12.5)
    g.card("draw", 30, 150, 176, 116, "Draw", ["6.30pm at", "Singapore Pools"], accent=ACCENT["pools"],
           glyph="ball", **r1)
    g.card("wake", 228, 150, 176, 116, "Bot wakes", ["7.30pm", "Singapore time"], accent=ACCENT["scrape"],
           glyph="clock", **r1)
    g.diamond("check", 428, 146, 164, 124, "Result out?", ACCENT["wait"])
    g.card("wait", 616, 150, 166, 116, "Wait", ["10 minutes, then", "check again"], accent=ACCENT["wait"],
           glyph="clock", **r1)
    g.card("alert", 804, 150, 166, 116, "Alert", ["after 2 hours,", "on Telegram"], accent=ACCENT["alert"],
           glyph="bell", **r1)

    g.edge("draw", "wake", [(206, 208), (228, 208)])
    g.edge("wake", "check", [(404, 208), (428, 208)])
    g.edge("check", "wait", [(592, 208), (616, 208)], label="no", label_at=(604, 186), label_color=ACCENT["wait"])
    g.edge("wait", "check", [(699, 150), (699, 128), (510, 128), (510, 146)], color=ACCENT["wait"],
           label="retry", label_at=(604, 128), label_color=ACCENT["wait"])
    g.edge("wait", "alert", [(782, 208), (804, 208)], color=ACCENT["alert"])

    # row 2: the run
    steps = [
        ("fetch", "Fetch", ["only the draws", "that are missing"], ACCENT["scrape"], "download", None),
        ("tickets", "Tickets", ["check each one,", "update the ledger"], ACCENT["tickets"], "ticket", None),
        ("outlook", "Outlook", ["jackpot draw by", "draw to cascade"], ACCENT["outlook"], "chart", None),
        ("signal", "Big prize", ["buy signal and", "the next big prize"], ACCENT["signal"], "target", None),
        ("vaultw", "Vault", ["tickets, ledger,", "notes and log"], ACCENT["vault"], None, "obsidian"),
        ("post", "Post", ["two messages", "to your channel"], ACCENT["post"], None, "telegram"),
    ]
    x, w, gap, y, h = 32, 136, 24, 336, 150
    for i, (cid, title, lines, accent, glyph, brand) in enumerate(steps):
        g.card(cid, x + i * (w + gap), y, w, h, title, lines, accent=accent, glyph=glyph, brand=brand,
               step=i + 1, icon_right=True, title_size=16, line_gap=20, line_size=12.5)
    for i in range(len(steps) - 1):
        x0 = x + i * (w + gap) + w
        g.edge(steps[i][0], steps[i + 1][0], [(x0, y + h / 2), (x0 + gap, y + h / 2)])
    g.edge("check", "fetch", [(510, 270), (510, 302), (100, 302), (100, 336)], color=ACCENT["outlook"],
           label="yes", label_at=(510, 294), label_color=ACCENT["outlook"])

    # footer
    fy = 516
    g.svg.append(
        f'<rect x="30" y="{fy}" width="940" height="52" rx="14" fill="{t["card"]}" stroke="{t["stroke"]}"/>'
    )
    g.vertex("rounded=1;arcSize=25;html=1;fillColor=#FFFFFF;strokeColor=#E2E8F0;", 30, fy, 940, 52)
    g.icon(44, fy + 8, 22, brand="docker", tile_size=36)
    g.text(92, fy + 31, "Dry run prints the two messages instead of posting. Each message stays under "
                        "4,000 characters, and a draw is never posted twice.", size=13, color=t["muted"])
    return g


def mcp_model(fn) -> str:
    """mxGraphModel XML with linked logos, small enough to send to the draw.io MCP server."""
    text = fn("light", linked=True).drawio_text()
    return text[text.index("<mxGraphModel"):text.index("</mxGraphModel>") + len("</mxGraphModel>")]


def build() -> list[Path]:
    out = []
    for fn in (architecture, draw_day):
        slug = fn.__name__.replace("_", "-")
        for theme in ("light", "dark"):
            d = fn(theme)
            p = HERE / f"{slug}-{theme}.svg"
            p.write_text(d.svg_text(), encoding="utf-8")
            out.append(p)
            if theme == "light":
                q = HERE / f"{slug}.drawio"
                q.write_text(d.drawio_text(), encoding="utf-8")
                out.append(q)
    return out


if __name__ == "__main__":
    for p in build():
        print(p.relative_to(HERE.parent.parent))
