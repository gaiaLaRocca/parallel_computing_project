#!/bin/bash
# Regenerate the two illustrative figures of the Frattali chapter, both produced
# by the serial kernel and coloured by analysis/render_fractal.py. They are
# figures, not measurements: run them anywhere (laptop included), while the
# timings quoted in the report come from batch jobs on the cluster.
#
# The two figures deliberately vary ONE thing each, because the two properties
# of the definition are independent and mixing them explains neither.
#
#   A. self-similarity   -> the viewport shrinks, N_max is FIXED.
#      Three nested views: each panel outlines the window the next one
#      magnifies, so the nesting is visible and not just asserted.
#
#   B. cost of detail    -> N_max varies, the viewport is FIXED, and fixed at
#      the window the serial table was measured on, so each image is the
#      picture of one row of that table rather than a separate experiment.
#      Rendered on a shared colour range, otherwise each panel renormalises on
#      its own extremes and the recolouring hides the only real difference.
#      The outcome is worth stating plainly in the report: over 500..5000 the
#      three images are indistinguishable at figure size (the pixels whose
#      classification changes are a one-pixel fringe, 0.116% of the grid) while
#      the work grows 9.57x. N_max is a cost knob, not an image-quality knob.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$REPO/mandelbrot/src"
DUMPS="$REPO/mandelbrot/data"          # git-ignored: the raw dumps are scratch
FIGURES="$REPO/latex_report/Immagini/frattali"
RENDER="$REPO/mandelbrot/analysis/render_fractal.py"

RESOLUTION=1024x1024
PIXELS=1024                            # both axes, for the renderer

mkdir -p "$DUMPS" "$FIGURES"
make -C "$SRC" mandelbrot

# Compute one view and colour it. The renderer takes whatever extra flags the
# caller appends (--frame, --count-range, ...).
render_panel() {
    local name=$1 center_re=$2 center_im=$3 span=$4 nmax=$5
    shift 5

    "$SRC/mandelbrot" --resolution "$RESOLUTION" --max-iter "$nmax" \
        --center-re "$center_re" --center-im "$center_im" --span "$span" \
        --raw "$DUMPS/figure_${name}.raw"

    python3 "$RENDER" "$DUMPS/figure_${name}.raw" "$FIGURES/${name}.png" \
        --width "$PIXELS" --height "$PIXELS" --max-iter "$nmax" "$@"
}

# --- A. self-similarity: viewport shrinks, N_max fixed ---------------------
# Four nested panels, x1 -> x6 -> x50 -> x1875 on the full view's 3.0 span.
# Each panel outlines the window the next one magnifies.
#
# All four share one colour range, and that matters twice over. With per-panel
# normalisation the same colour would mean a different escape count in each
# panel, so the backgrounds would differ for a reason no reader could infer.
# Fixed, the warming of the background along the sequence becomes a measurement:
# the deeper the window, the higher its escape-count floor.
ZOOM_NMAX=1000
ZOOM_RANGE=1,1000

# The island on the antenna, and the island on *its* antenna. Panel 3 is
# centred between the two, so that both its own body and the window of panel 4
# fall inside it: the sub-island is 40x smaller than the copy that carries it,
# so the outlined box is necessarily tiny — which is the point being made.
ISLAND=-1.7548776
SUBISLAND=-1.7859250
PANEL3_CENTER=-1.7608776

render_panel zoom1_insieme  -0.75             0.0 3.0000 "$ZOOM_NMAX" \
    --count-range "$ZOOM_RANGE" --view=-0.75,0.0,3.0 --frame="$ISLAND,0.0,0.50"
render_panel zoom2_antenna "$ISLAND"          0.0 0.5000 "$ZOOM_NMAX" \
    --count-range "$ZOOM_RANGE" --view="$ISLAND,0.0,0.5" \
    --frame="$PANEL3_CENTER,0.0,0.06"
render_panel zoom3_isola   "$PANEL3_CENTER"   0.0 0.0600 "$ZOOM_NMAX" \
    --count-range "$ZOOM_RANGE" --view="$PANEL3_CENTER,0.0,0.06" \
    --frame="$SUBISLAND,0.0,0.0016"
render_panel zoom4_sottoisola "$SUBISLAND"    0.0 0.0016 "$ZOOM_NMAX" \
    --count-range "$ZOOM_RANGE"

# --- B. cost of detail: N_max varies, viewport fixed ------------------------
# The window is the one the report's serial table (tab:serial_var_Nmax) was
# measured on: the binary's default viewport, which these flags reproduce
# exactly — same checksum, same W as the cluster job. That is deliberate. A
# deeper window would stress N_max harder, but it would also be a second
# configuration the reader has to keep apart from the table; here every number
# in the section belongs to one and the same run.
BENCH_CENTER=-0.75
BENCH_SPAN=2.5

# The shared range spans the deepest panel's counts, so a panel that cannot
# reach them simply stops earlier on the same ramp.
for NMAX in 500 1000 5000; do
    render_panel "nmax${NMAX}" "$BENCH_CENTER" 0.0 "$BENCH_SPAN" "$NMAX" \
        --count-range 1,5000
done

# Where N_max actually acts: the pixels a cap of 500 would call interior even
# though they do escape. A one-pixel fringe on the boundary — the visual answer
# to why the three panels above look identical.
python3 "$RENDER" "$DUMPS/figure_nmax5000.raw" "$FIGURES/nmax_frangia.png" \
    --width "$PIXELS" --height "$PIXELS" --max-iter 5000 \
    --count-range 1,5000 --highlight-above 500

# The same fringe across three grids, which is the table the report quotes: its
# pixel count quadruples with the grid while its *fraction* stays put. That is
# the boundary having box dimension 2 — the same reason lambda and CoV are
# resolution-invariant in the Level 1 sweep.
echo "--- fringe [500,5000) vs resolution, benchmark window ---"
for GRID in 512 1024 2048; do
    "$SRC/mandelbrot" --resolution "${GRID}x${GRID}" --max-iter 5000 \
        --center-re "$BENCH_CENTER" --center-im 0.0 --span "$BENCH_SPAN" \
        --raw "$DUMPS/figure_fringe_${GRID}.raw" > /dev/null

    python3 "$RENDER" "$DUMPS/figure_fringe_${GRID}.raw" \
        "$DUMPS/figure_fringe_${GRID}.png" \
        --width "$GRID" --height "$GRID" --max-iter 5000 \
        --count-range 1,5000 --highlight-above 500
done
