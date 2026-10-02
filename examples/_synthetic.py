"""Synthetic inputs shared by the examples (no animal data; everything is random, seeded)."""

import json
from pathlib import Path

import numpy as np


def synthetic_xyz_picks(out_dir, n_tracks=6, seed=0):
    """Write ``<out_dir>/SYN01_R<i>_xyz_picks.json`` files in the lab format (um relative to bregma, ML/AP/DV).

    Tracks start just below the surface and run ~3.5 mm ventrally with a small tilt, spread over a
    forebrain-sized box. They are NOT guaranteed to lie inside every structure; they only need to be
    inside the CCF volume.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n_tracks):
        ml = rng.uniform(-3000, 3000)
        ap = rng.uniform(-3000, 1500)
        top = np.array([ml, ap, -300.0])
        bottom = top + np.array([rng.normal(0, 200), rng.normal(0, 200), -3500.0])
        picks = np.linspace(top, bottom, 8).round(1).tolist()
        f = out_dir / f"SYN{1 + i // 3:02d}_R{1 + i % 3}_xyz_picks.json"
        f.write_text(json.dumps({"xyz_picks": picks}))
        files.append(f)
    return files


def synthetic_units(n=1500, seed=0):
    """(n, 3) unit positions (um relative to bregma, ML/AP/DV) in an ellipsoid + a metric that varies along AP."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    pts /= np.maximum(1.0, np.linalg.norm(pts, axis=1, keepdims=True))  # keep inside unit ball
    pts = pts * np.array([3000, 3000, 2200]) + np.array([0, -500, -3000])
    metric = (pts[:, 1] - pts[:, 1].min()) / np.ptp(pts[:, 1]) + rng.normal(0, 0.15, n)
    metric[rng.random(n) < 0.02] = np.nan  # a few units without a value
    return pts, metric


def synthetic_activity(n_units=300, n_bins=120, seed=0):
    """(n, T) activity: a wave travelling along AP, plus noise (arbitrary units)."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, (n_units, 1))
    t = np.arange(n_bins)[None, :]
    return np.clip(np.sin(t / 8.0 - phase) + rng.normal(0, 0.2, (n_units, n_bins)), 0, None)
