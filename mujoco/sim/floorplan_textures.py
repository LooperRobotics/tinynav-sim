#!/usr/bin/env python3
"""Per-segment wall textures for the floorplan scene, generated as PNG bytes.

The ring corridor used to be eight runs of flat paint over one global
checkerboard: every wall face looked like every other, so reloc retrieval
aliased freely along the ring.  Following the gazebo side's per-instance
albedo maps (models/textures/crate_0..7.png), each wall segment here gets
its own texture: a distinct paint hue, a large bitmap-font label, and a
seeded dot/stripe field so local high-frequency features differ at every
position.  Everything is deterministic in the segment name -- no binary
assets in the repo, the PNGs are regenerated on scene build and handed to
MjSpec through its assets dict.
"""
from __future__ import annotations

import colorsys
import zlib
from io import BytesIO

# (segment key, big label, small tag, base hue in degrees).  outer walls use
# uppercase labels, well walls lowercase -- two visually distinct classes so
# opposite walls never share a look even at matching hues.
SEGMENTS = [
    ("outer_w", "W", "0", 15.0),
    ("outer_n", "N", "1", 45.0),
    ("outer_e", "E", "2", 110.0),
    ("outer_s", "S", "3", 205.0),
    ("well_w", "w", "0", 285.0),
    ("well_n", "n", "1", 325.0),
    ("well_e", "e", "2", 170.0),
    ("well_s", "s", "3", 60.0),
]

_W = 1024
_H = 256

# 5x7 bitmap font, rows top->bottom, bit 4 leftmost.
_FONT = {
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "N": ["10001", "11001", "11001", "10101", "10011", "10011", "10001"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "w": ["00000", "00000", "10001", "10001", "10101", "10101", "01010"],
    "n": ["00000", "00000", "11110", "10001", "10001", "10001", "10001"],
    "e": ["00000", "00000", "01110", "10001", "11111", "10000", "01111"],
    "s": ["00000", "00000", "01111", "10000", "01110", "00001", "11110"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
}


def _seed(name: str):
    import random

    return random.Random(zlib.crc32(name.encode()))


def _hls(h: float, l: float, s: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def _draw_glyph(draw, ch: str, x: int, y: int, px: int, fill, outline):
    rows = _FONT[ch]
    for dy, row in enumerate(rows):
        for dx, bit in enumerate(row):
            if bit == "1":
                draw.rectangle(
                    [x + dx * px, y + dy * px, x + (dx + 1) * px - 1, y + (dy + 1) * px - 1],
                    fill=fill,
                    outline=outline,
                    width=2,
                )


def wall_textures() -> dict[str, bytes]:
    """segment key -> PNG bytes, one file per wall segment."""
    from PIL import Image, ImageDraw

    out: dict[str, bytes] = {}
    for key, label, tag, hue in SEGMENTS:
        rng = _seed(key)
        img = Image.new("RGB", (_W, _H), _hls(hue, 0.62, 0.34))
        d = ImageDraw.Draw(img)

        # skirting board and top trim break the vertical centre line
        d.rectangle([0, int(_H * 0.85), _W, _H], fill=_hls(hue, 0.38, 0.30))
        d.rectangle([0, 0, _W, 10], fill=_hls(hue, 0.46, 0.30))

        # corridor-code band at dog-eye height: the segment label repeats
        # every ~1 m so ANY view along a wall carries its code -- two
        # geometrically twin corridors (N vs S flight) then never embed
        # alike, which is what false loop closures feed on.  Band pixel
        # rows 70..180 of 256 map to roughly z 0.6..1.4 m on a 2 m wall.
        band_top, band_bot = 70, 180
        d.rectangle([0, band_top, _W, band_bot], fill=_hls(hue, 0.86, 0.20))
        px = 14
        gw, gh = 5 * px, 7 * px
        step = gw + 46
        xg = 20
        while xg + gw < _W:
            _draw_glyph(d, label, xg, band_top + 6, px,
                        _hls(hue, 0.22, 0.35), _hls(hue, 0.95, 0.10))
            xg += step

        # vertical stripe accents at seeded x positions
        for _ in range(rng.randint(3, 5)):
            sx = rng.randint(40, _W - 60)
            sw = rng.randint(6, 12)
            d.rectangle([sx, 10, sx + sw, int(_H * 0.85)], fill=_hls(hue, 0.50, 0.32))

        # seeded dot field: local high-frequency features everywhere on the
        # face, dark/light alternating so every patch of wall is distinctive
        step = 32
        for gy in range(10, _H, step):
            if band_top - 8 <= gy <= band_bot + 8:
                continue
            for gx in range(step // 2, _W, step):
                if rng.random() < 0.42:
                    continue
                r = rng.randint(3, 6)
                light = rng.random() < 0.5
                col = _hls(hue, 0.78 if light else 0.34, 0.30)
                jx = gx + rng.randint(-6, 6)
                jy = gy + rng.randint(-6, 6)
                d.ellipse([jx - r, jy - r, jx + r, jy + r], fill=col)

        # big label: letter (uppercase=outer, lowercase=well) + small tag,
        # dark fill with a light halo so it reads against any hue
        dark = _hls(hue, 0.22, 0.35)
        halo = _hls(hue, 0.92, 0.15)
        px_big, px_small = 26, 15
        lw, lh = 5 * px_big, 7 * px_big
        sw_, sh_ = 5 * px_small, 7 * px_small
        total = lw + 40 + sw_
        x0 = (_W - total) // 2
        y0 = (_H - lh) // 2 + 8
        _draw_glyph(d, label, x0, y0, px_big, dark, halo)
        _draw_glyph(d, tag, x0 + lw + 40, y0 + (lh - sh_) // 2, px_small, dark, halo)

        # thin frame ties the segments to the wall geometry edges
        d.rectangle([0, 0, _W - 1, _H - 1], outline=_hls(hue, 0.30, 0.30), width=4)

        buf = BytesIO()
        img.save(buf, format="PNG")
        out[f"fp_{key}.png"] = buf.getvalue()
    return out


def floor_material_assets(params) -> tuple[str, list[dict]]:
    """Per-region floor materials: (XML fragment, geom specs).

    One global checker made every region embed alike; region-tinted
    checkers at different scales give each corridor its own ground -- the
    floor is a large FOV share at dog-eye height, so this is the strongest
    single aliasing breaker for the N/S flight pair.
    """
    t = params.wall_thickness
    lx2 = params.free_len_x / 2.0
    ly2 = params.free_len_y / 2.0
    p = params.platform_len
    fx0 = lx2 - p  # inner edge of the platform strips

    regions = [
        ("floor_n", (0.0, +(ly2 - params.flight_width / 2.0)),
         (fx0, params.flight_width / 2.0), "0.55 0.62 0.55", "0.82 0.88 0.78", "6 2"),
        ("floor_s", (0.0, -(ly2 - params.flight_width / 2.0)),
         (fx0, params.flight_width / 2.0), "0.60 0.55 0.66", "0.85 0.80 0.90", "8 2"),
        ("floor_w", (-(lx2 + fx0) / 2.0, 0.0),
         (p / 2.0, ly2), "0.66 0.60 0.50", "0.90 0.86 0.78", "2 4"),
        ("floor_e", (+(lx2 + fx0) / 2.0, 0.0),
         (p / 2.0, ly2), "0.50 0.60 0.66", "0.78 0.88 0.92", "3 4"),
    ]
    frag = ""
    geoms = []
    for name, pos, half, rgb1, rgb2, rep in regions:
        frag += (
            f'<texture type="2d" name="{name}_tex" builtin="checker" '
            f'rgb1="{rgb1}" rgb2="{rgb2}" width="512" height="512"/>'
            f'<material name="{name}" texture="{name}_tex" '
            f'texrepeat="{rep}" rgba="1 1 1 1"/>'
        )
        geoms.append({"name": name, "pos": (pos[0], pos[1], -params.floor_thickness / 2.0),
                      "half": (half[0], half[1], params.floor_thickness / 2.0),
                      "material": name})
    return frag, geoms


def material_assets() -> tuple[str, dict[str, bytes]]:
    """(XML asset block fragment, {file name: png bytes}) for MjSpec."""
    frag = "".join(
        f'<texture type="2d" name="fp_{key}_tex" file="fp_{key}.png"/>'
        f'<material name="fp_{key}" texture="fp_{key}_tex" '
        'texrepeat="1 1" rgba="1 1 1 1" specular="0.1" shininess="0.3"/>'
        for key, _, _, _ in SEGMENTS
    )
    return frag, wall_textures()
