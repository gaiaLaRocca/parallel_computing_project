#!/usr/bin/env python3
"""Render an escape-time dump (``--raw``) as a PNG figure for the report.

Colouring is deliberately *not* done by the C++ ``save_ppm``: the shared writer
normalises on a fixed ``PALETTE_RANGE = 256`` so that colours stay comparable
across ``max_iter`` values, which is exactly what the benchmark contract needs
and exactly what a zoom figure cannot use — past the first few magnifications
almost every exterior pixel escapes after more than 256 iterations and would be
painted black, indistinguishable from the interior.

Here the mapping is instead renormalised per image, so every panel of a zoom
sequence uses the full colour range whatever its iteration histogram looks like:
by the log of the escape count between the panel's own extremes (default), or by
cumulative frequency (``--map equalise``), which spreads the palette by pixel
area instead. The numbers themselves still come from the serial kernel; only the
number -> colour mapping lives in this script.

Depends on the standard library only (``zlib`` + ``struct`` write the PNG), so
it runs anywhere the repository is checked out.
"""

import argparse
import array
import math
import struct
import sys
import zlib
from pathlib import Path

# Classic "Ultra Fractal" ramp: deep blue -> blue -> white -> amber -> black.
# Wide luminance excursion, so the bands stay legible in a printed grayscale
# copy of the report as well.
PALETTE_STOPS = (
    (0.0000, (0, 7, 100)),
    (0.1600, (32, 107, 203)),
    (0.4200, (237, 255, 255)),
    (0.6425, (255, 170, 0)),
    (0.8575, (0, 2, 0)),
    (1.0000, (0, 7, 100)),
)

INTERIOR_COLOR = (0, 0, 0)

# Off the palette on purpose: the highlighted pixels must read as an annotation,
# not as another band of the ramp.
HIGHLIGHT_COLOR = (230, 30, 40)


def main(argv=None):
    args = parse_arguments(argv)

    counts = read_escape_counts(args.raw, args.width, args.height)
    rgb = colorise(counts, args.max_iter, args.map, args.gamma,
                   parse_count_range(args.count_range), args.highlight_above)
    if args.frame:
        if not args.view:
            raise SystemExit("--frame needs --view: the box can only be placed "
                             "once the panel's own viewport is known")
        rgb = outline_window(rgb, args.width, args.height,
                             parse_window(args.view), parse_window(args.frame))
    write_png(args.out, args.width, args.height, rgb)

    interior = sum(1 for c in counts if c >= args.max_iter)
    stats = (f"{args.out}  {args.width}x{args.height}  max_iter={args.max_iter}  "
             f"interior={interior} ({100.0 * interior / len(counts):.3f}%)  "
             f"escape_max={max((c for c in counts if c < args.max_iter), default=0)}")
    if args.highlight_above is not None:
        # The report quotes this count: it is the number of pixels a cap of
        # --highlight-above would have misclassified as interior.
        fringe = sum(1 for c in counts if args.highlight_above <= c < args.max_iter)
        stats += (f"  fringe[{args.highlight_above},{args.max_iter})={fringe} "
                  f"({100.0 * fringe / len(counts):.3f}%)")
    print(stats)
    return 0


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="escape-time dump written by --raw")
    parser.add_argument("out", type=Path, help="destination PNG")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--max-iter", type=int, required=True,
                        help="interior threshold: pixels at this count are painted black")
    parser.add_argument("--map", choices=("log", "equalise"), default="log",
                        help="log: colour by log of the escape count (default); "
                             "equalise: spread the counts by cumulative frequency")
    parser.add_argument("--view", metavar="CX,CY,SPAN",
                        help="the viewport this dump was computed on; only "
                             "needed together with --frame")
    parser.add_argument("--frame", metavar="CX,CY,SPAN",
                        help="outline this window on the image, to show a reader "
                             "which portion the next panel of a zoom sequence "
                             "magnifies")
    parser.add_argument("--highlight-above", type=int, metavar="N",
                        help="paint in red the pixels that escape at N or later "
                             "but still before max_iter: exactly the pixels a "
                             "cap of N would have misclassified as interior. "
                             "Turns the effect of max_iter into something "
                             "visible, since it is a one-pixel fringe")
    parser.add_argument("--count-range", metavar="MIN,MAX",
                        help="freeze the log normalisation on this escape-count "
                             "interval instead of the panel's own extremes. "
                             "Required when several panels must be compared: "
                             "with per-panel normalisation a change of max_iter "
                             "recolours the whole image and hides the only "
                             "difference that matters, which pixels went black")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="exponent applied to the normalised value; <1 brightens "
                             "the slow-escaping fringe, >1 compresses it")
    return parser.parse_args(argv)


def read_escape_counts(path, width, height):
    """Little-endian int32, row-major — the layout save_raw emits."""
    expected = width * height
    counts = array.array("i")
    with path.open("rb") as dump:
        counts.frombytes(dump.read())
    if sys.byteorder != "little":
        counts.byteswap()
    if len(counts) != expected:
        raise SystemExit(f"{path}: holds {len(counts)} values, "
                         f"{width}x{height} needs {expected}")
    return counts


def parse_window(text):
    """"CX,CY,SPAN" -> the same triple as floats."""
    center_real, center_imag, span = (float(value) for value in text.split(","))
    return center_real, center_imag, span


def outline_window(rgb, width, height, view, frame):
    """Draw the frame window as a rectangle over the rendered pixels.

    The zoom factor between two panels is a number in a caption; the rectangle
    is the same statement the reader can see, and it is what makes a sequence
    read as nested rather than as three unrelated pictures.
    """
    view_real, view_imag, view_span = view
    frame_real, frame_imag, frame_span = frame

    scale = width / view_span
    left = round((frame_real - 0.5 * frame_span - (view_real - 0.5 * view_span)) * scale)
    right = round((frame_real + 0.5 * frame_span - (view_real - 0.5 * view_span)) * scale)
    # Rows run downwards from y_max, so the imaginary axis flips.
    view_top = view_imag + 0.5 * view_span * height / width
    top = round((view_top - (frame_imag + 0.5 * frame_span * height / width)) * scale)
    bottom = round((view_top - (frame_imag - 0.5 * frame_span * height / width)) * scale)

    # A thin box vanishes once the figure is scaled down to a column width; a
    # box smaller than its own border would swallow what it points at.
    thickness = max(2, round(0.003 * width))
    if right - left < 4 * thickness or bottom - top < 4 * thickness:
        thickness = max(1, min(right - left, bottom - top) // 4)

    framed = bytearray(rgb)
    for row in range(max(top - thickness, 0), min(bottom + thickness, height)):
        for col in range(max(left - thickness, 0), min(right + thickness, width)):
            on_vertical = col < left + thickness or col >= right - thickness
            on_horizontal = row < top + thickness or row >= bottom - thickness
            if on_vertical or on_horizontal:
                framed[3 * (row * width + col):3 * (row * width + col) + 3] = \
                    bytes(HIGHLIGHT_COLOR)
    return bytes(framed)


def parse_count_range(text):
    """"MIN,MAX" -> (min, max); None when the panel normalises on itself."""
    if text is None:
        return None
    low, high = (int(value) for value in text.split(","))
    if low >= high:
        raise SystemExit(f"--count-range {text}: MIN must be below MAX")
    return low, high


def colorise(counts, max_iter, mapping, gamma, count_range, highlight_above=None):
    """Map escape counts to RGB bytes through the chosen normalisation."""
    shade = (equalised_shades(counts, max_iter, gamma) if mapping == "equalise"
             else logarithmic_shades(counts, max_iter, gamma, count_range))
    highlight = max_iter if highlight_above is None else highlight_above

    rgb = bytearray(3 * len(counts))
    for i, count in enumerate(counts):
        if count >= max_iter:
            r, g, b = INTERIOR_COLOR
        elif count >= highlight:
            r, g, b = HIGHLIGHT_COLOR
        else:
            r, g, b = shade[count]
        rgb[3 * i] = r
        rgb[3 * i + 1] = g
        rgb[3 * i + 2] = b
    return bytes(rgb)


def logarithmic_shades(counts, max_iter, gamma, count_range=None):
    """One colour per exterior escape count, on a log scale of the count itself.

    The escape counts of a Mandelbrot view are extremely skewed — most pixels
    leave in a handful of iterations — so a linear ramp would spend the whole
    palette on the featureless outside. The log rescales per image, which is
    what makes the panels of a zoom sequence individually readable.
    """
    exterior = {count for count in counts if count < max_iter}
    if not exterior:
        return {}

    # Rescaled between the panel's own extremes, not from zero: a deep zoom has
    # no fast-escaping pixel at all, and anchoring at zero would leave the cold
    # half of the palette permanently unused. An explicit range overrides this,
    # which is what makes two panels comparable.
    low, high = count_range if count_range else (min(exterior), max(exterior))
    shallowest = math.log1p(low)
    span = math.log1p(high) - shallowest
    if span == 0.0:
        return {count: palette_color(0.0) for count in exterior}
    return {count: palette_color(((math.log1p(count) - shallowest) / span) ** gamma)
            for count in exterior}


def equalised_shades(counts, max_iter, gamma):
    """One colour per exterior escape count, spread by cumulative frequency.

    Histogram equalisation is what keeps a zoom sequence readable: the deep
    panels concentrate their counts in a narrow high band that any fixed
    normalisation would flatten into a single tone.
    """
    histogram = {}
    for count in counts:
        if count < max_iter:
            histogram[count] = histogram.get(count, 0) + 1

    exterior = sum(histogram.values())
    if exterior == 0:
        return {}

    shades = {}
    cumulative = 0
    for count in sorted(histogram):
        # Mid-bin rank, so the rarest (deepest) counts do not collapse onto the
        # very end of the palette.
        position = (cumulative + 0.5 * histogram[count]) / exterior
        shades[count] = palette_color(position ** gamma)
        cumulative += histogram[count]
    return shades


def palette_color(t):
    """Linear interpolation between the palette stops, t clamped to [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    for (t0, c0), (t1, c1) in zip(PALETTE_STOPS, PALETTE_STOPS[1:]):
        if t <= t1:
            u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return tuple(round(a + u * (b - a)) for a, b in zip(c0, c1))
    return PALETTE_STOPS[-1][1]


def write_png(path, width, height, rgb):
    """Minimal 8-bit truecolour PNG: no filtering, one zlib stream."""
    stride = 3 * width
    raw = bytearray()
    for row in range(height):
        raw.append(0)  # filter type 0 (None)
        raw += rgb[row * stride:(row + 1) * stride]

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with path.open("wb") as png:
        png.write(b"\x89PNG\r\n\x1a\n")
        png.write(png_chunk(b"IHDR", header))
        png.write(png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
        png.write(png_chunk(b"IEND", b""))


def png_chunk(tag, payload):
    return (struct.pack(">I", len(payload)) + tag + payload +
            struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


if __name__ == "__main__":
    raise SystemExit(main())
