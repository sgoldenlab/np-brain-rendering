"""Units coloured by a metric (synthetic positions + metric generated here).

    python examples/02_render_units_by_metric.py --screenshots --cmap viridis
    python examples/02_render_units_by_metric.py --csv units.csv --columns ml ap dv metric --screenshots

CSV mode expects bregma-relative um (ML, AP, DV) columns + a metric column (NaN allowed).
Units outside the brain (checked against the atlas annotation) are hidden.
"""

import argparse

import numpy as np

from _synthetic import synthetic_units
from np_rendering.colors import metric_to_rgba
from np_rendering.coords import bregma_to_ccf

ap = argparse.ArgumentParser()
ap.add_argument("--csv", help="optional table with unit coordinates and a metric")
ap.add_argument("--columns", nargs=4, metavar=("ML", "AP", "DV", "METRIC"), default=["ml", "ap", "dv", "metric"])
ap.add_argument("--cmap", default="viridis")
ap.add_argument("--vmin", type=float, default=None)
ap.add_argument("--vmax", type=float, default=None)
ap.add_argument("--point-size", type=float, default=10)
ap.add_argument("--keep-outside", action="store_true", help="draw units even if outside the brain")
ap.add_argument("--screenshots", action="store_true")
ap.add_argument("--out-dir", default="render_output/units")
ap.add_argument("--views", nargs="+", default=["frontal", "sagittal"])
ap.add_argument("--no-render", action="store_true")
a = ap.parse_args()

if a.csv:
    import pandas as pd  # only needed for the CSV option

    df = pd.read_csv(a.csv)
    ml, apc, dv, mcol = a.columns
    xyz_um = df[[ml, apc, dv]].to_numpy(float)
    metric = df[mcol].to_numpy(float)
else:
    xyz_um, metric = synthetic_units()

ccf = bregma_to_ccf(xyz_um, input_order="mlapdv", ccf_order="apdvml", mode="clip")  # brainrender order
rgba = metric_to_rgba(metric, a.cmap, a.vmin, a.vmax)
print(f"{len(ccf)} units; metric range {np.nanmin(metric):.2f}..{np.nanmax(metric):.2f}; "
      f"CCF AP {ccf[:, 0].min():.0f}..{ccf[:, 0].max():.0f} um")

if not a.no_render:
    from np_rendering.neurons import render_neurons

    paths = render_neurons(
        ccf, point_color=rgba, point_size=a.point_size, hide_outside=not a.keep_outside,
        screenshots=a.screenshots, out_dir=a.out_dir, ss_label="units_metric", views=a.views,
    )
    print("saved:", *paths, sep="\n  ")
