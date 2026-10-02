"""Probe tracks in the Allen CCF from xyz_picks JSON files (synthetic tracks generated here).

    python examples/01_render_probe_tracks.py --screenshots --out-dir render_output/probes
    python examples/01_render_probe_tracks.py --picks-dir my_xyz_picks --screenshots   # your own files

Without --screenshots an interactive window opens. Headless Linux: prefix with ``xvfb-run -a``.
The first run downloads the BrainGlobe atlas (allen_mouse_25um, ~ hundreds of MB).
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from _synthetic import synthetic_xyz_picks
from np_rendering.colors import categorical_colors
from np_rendering.coords import load_xyz_picks, track_endpoints_ccf
from np_rendering.probes import render_probes

ap = argparse.ArgumentParser()
ap.add_argument("--picks-dir", help="folder of *xyz_picks*.json; default: generate synthetic ones")
ap.add_argument("--filter", default=None, help="keep files whose name contains this text")
ap.add_argument("--color-by", choices=["animal", "recording", "black"], default="animal")
ap.add_argument("--screenshots", action="store_true")
ap.add_argument("--out-dir", default="render_output/probes")
ap.add_argument("--views", nargs="+", default=["frontal", "sagittal"])
ap.add_argument("--regions", nargs="*", default=None, help="region acronyms to draw, e.g. BLA CEA")
ap.add_argument("--no-render", action="store_true", help="only load + convert (no brainrender)")
a = ap.parse_args()

picks_dir = Path(a.picks_dir) if a.picks_dir else Path(tempfile.mkdtemp(prefix="syn_picks_"))
if not a.picks_dir:
    synthetic_xyz_picks(picks_dir)
files = sorted(picks_dir.glob("*.json"), key=lambda p: p.stem)
if a.filter:
    files = [f for f in files if a.filter in f.stem]
assert files, "no JSON files found"

# file naming convention: <animal>_<recording>_...json  (change the parsing for your own names)
recs = ["_".join(f.stem.split("_")[:2]) for f in files]
animals = [r.split("_")[0] for r in recs]
tracks = [track_endpoints_ccf(load_xyz_picks(f)) for f in files]  # (2,3) CCF um, AP/DV/ML
print(f"{len(tracks)} tracks; first top/bottom in CCF (AP, DV, ML) um:\n{np.round(tracks[0], 1)}")

key = {"animal": animals, "recording": recs, "black": ["all"] * len(files)}[a.color_by]
cmap = {"black": {"all": "#333333"}}.get(a.color_by) or categorical_colors(key)
colors = [cmap[k] for k in key]

if not a.no_render:
    paths = render_probes(
        tracks, labels=recs, colors=colors,
        legend_labels=list(cmap), legend_colors=list(cmap.values()),
        regions=a.regions, region_colors=["red"] * len(a.regions or []),
        screenshots=a.screenshots, out_dir=a.out_dir, ss_label="probe_tracks", views=a.views,
    )
    print("saved:", *paths, sep="\n  ")
