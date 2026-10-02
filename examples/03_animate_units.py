"""Animate unit activity over time (synthetic activity generated here).

    xvfb-run -a python examples/03_animate_units.py --out render_output/units.mp4

Scaling options: per-unit min-max to [0, 1], then fixed colour
limits vmin=0.2, vmax=1.4 (so the brightest colour is never reached -> softer look). Use
--no-scale-per-unit to colour raw values (limits default to the 5th/95th percentile).
"""

import argparse

from _synthetic import synthetic_activity, synthetic_units
from np_rendering.colors import minmax_per_unit
from np_rendering.coords import bregma_to_ccf

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="render_output/units.mp4")
ap.add_argument("--n-units", type=int, default=300)
ap.add_argument("--n-bins", type=int, default=120)
ap.add_argument("--bins-per-second", type=float, default=36.0, help="playback speed")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--no-scale-per-unit", action="store_true")
ap.add_argument("--vmin", type=float, default=None, help="default 0.2 if scaled per unit, else percentile")
ap.add_argument("--vmax", type=float, default=None, help="default 1.4 if scaled per unit, else percentile")
ap.add_argument("--cmap", default="hot")
ap.add_argument("--rotate-degrees", type=float, default=60.0)
ap.add_argument("--light", action="store_true", help="light background instead of dark mode")
ap.add_argument("--crf", type=int, default=15)
ap.add_argument("--no-render", action="store_true")
a = ap.parse_args()

xyz_um, _ = synthetic_units(a.n_units)
ccf = bregma_to_ccf(xyz_um, mode="clip")
act = synthetic_activity(a.n_units, a.n_bins)
if not a.no_scale_per_unit:
    act = minmax_per_unit(act)
vmin, vmax = a.vmin, a.vmax
if not a.no_scale_per_unit:
    vmin = 0.2 if vmin is None else vmin
    vmax = 1.4 if vmax is None else vmax
print(f"{ccf.shape[0]} units x {act.shape[1]} bins; colour limits {vmin}..{vmax}")

if not a.no_render:
    from np_rendering.animation import animate_neurons

    animate_neurons(
        ccf, act, a.out, bins_per_second=a.bins_per_second, fps=a.fps, cmap=a.cmap, vmin=vmin, vmax=vmax,
        point_size=7, rotate_degrees=a.rotate_degrees, zoom=1.5, dark_mode=not a.light,
        timer_prefix="T from onset:\n", timer_start=0.0, time_per_bin=1.0, crf=a.crf,
    )
