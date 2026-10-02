"""Coordinate conventions and conversions (bregma-relative <-> Allen CCF).

Convention table
================

========================  ============================  ====================  =========================
Space                     Axes / order                  Units                 Origin / sign
========================  ============================  ====================  =========================
IBL / xyz_picks ("xyz")   ML, AP, DV  (``mlapdv``)      um in JSON files,     bregma. +ML right,
                                                        metres for iblatlas   +AP anterior, +DV dorsal
                                                                              (ventral = negative)
Allen CCF, IBL order      ML, AP, DV  (``mlapdv``)      um (index * res)      front-left-top corner of
                                                                              the volume
Allen CCF, brainrender    AP, DV, ML  (``apdvml``)      um                    same corner; AP and DV
order (and Allen MCC                                                          increase posterior and
mesh vertices)                                                                ventral
========================  ============================  ====================  =========================

Bregma in the Allen CCF (IBL definition) is ``[ML, AP, DV] = [5739, 5400, 332]`` um, so
bregma (0, 0, 0) maps to brainrender coordinates ``(AP, DV, ML) = (5400, 332, 5739)``.
Anterior points have SMALLER CCF-AP, ventral points LARGER CCF-DV.

Everything here goes through ``iblatlas.atlas.AllenAtlas.xyz2ccf`` so the offsets
and axis flips are exactly IBL's. A "mock" AllenAtlas (geometry only, no image volume)
is used, so no atlas download is needed for the conversion.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

# Allen CCF landmark used by iblatlas (ML, AP, DV) in um.
BREGMA_CCF_MLAPDV_UM = (5739.0, 5400.0, 332.0)
# Approximate geometric centre of the 25 um CCF in brainrender (AP, DV, ML) order; used as camera focal point.
BRAIN_CENTER_CCF_APDVML_UM = (6600.0, 4000.0, 5700.0)

_AXES = ("ml", "ap", "dv")


def _parse_order(order: str) -> list[int]:
    """'apdvml' -> indices into (ml, ap, dv) for each position of the string."""
    order = order.lower()
    if len(order) != 6 or sorted(order[i : i + 2] for i in (0, 2, 4)) != sorted(_AXES):
        raise ValueError(f"order must be a permutation of 'ml','ap','dv' (e.g. 'mlapdv'); got {order!r}")
    return [_AXES.index(order[i : i + 2]) for i in (0, 2, 4)]


@lru_cache(maxsize=4)
def _allen_atlas(res_um: int):
    """Geometry-only IBL AllenAtlas (lazy import; cached)."""
    from iblatlas.atlas import AllenAtlas

    try:
        return AllenAtlas(res_um, mock=True)  # no volume download
    except TypeError:  # very old iblatlas without `mock`
        return AllenAtlas(res_um)


def bregma_to_ccf(
    xyz_um,
    input_order: str = "mlapdv",
    ccf_order: str = "apdvml",
    res_um: int = 25,
    mode: str = "raise",
) -> np.ndarray:
    """Convert bregma-relative coordinates (um) to Allen CCF coordinates (um).

    Parameters
    ----------
    xyz_um : array-like (3,) or (N, 3)
        Positions in um relative to bregma, columns ordered as ``input_order``.
    input_order : str, default 'mlapdv'
        Column order of the input. Change if your table is e.g. (AP, DV, ML).
    ccf_order : str, default 'apdvml'
        Column order of the output. ``'apdvml'`` is what brainrender/vedo meshes use;
        ``'mlapdv'`` is IBL order.
    res_um : int, default 25
        Atlas resolution (25 matches brainrender ``allen_mouse_25um``).
    mode : {'raise', 'clip', 'wrap'}
        Outside-the-volume behaviour (iblatlas): 'raise' errors, 'clip' moves the point
        to the nearest voxel, 'wrap' returns it as is.

    Returns
    -------
    numpy.ndarray, same leading shape as the input, float CCF um.
    """
    xyz = np.asarray(xyz_um, dtype=float)
    if xyz.shape[-1] != 3:
        raise ValueError(f"expected last dimension 3, got shape {xyz.shape}")
    idx = _parse_order(input_order)
    # reorder columns to (ml, ap, dv)
    ml_ap_dv = np.empty_like(xyz)
    for pos, axis in enumerate(idx):
        ml_ap_dv[..., axis] = xyz[..., pos]
    atlas = _allen_atlas(int(res_um))
    return np.asarray(atlas.xyz2ccf(ml_ap_dv / 1e6, ccf_order=ccf_order, mode=mode))


def ccf_to_bregma(ccf_um, ccf_order: str = "apdvml", res_um: int = 25) -> np.ndarray:
    """Inverse of :func:`bregma_to_ccf`; returns um relative to bregma in (ML, AP, DV) order."""
    ccf = np.asarray(ccf_um, dtype=float)
    atlas = _allen_atlas(int(res_um))
    return np.asarray(atlas.ccf2xyz(ccf, ccf_order=ccf_order)) * 1e6


def load_xyz_picks(path) -> np.ndarray:
    """Read an ``xyz_picks`` JSON (``{"xyz_picks": [[x, y, z], ...]}``, um relative to bregma, ML/AP/DV).

    Returns an (N, 3) float array.
    """
    with open(Path(path), "r") as f:
        picks = np.asarray(json.load(f)["xyz_picks"], dtype=float)
    if picks.ndim != 2 or picks.shape[1] != 3 or len(picks) < 2:
        raise ValueError(f"{path}: expected >=2 points of 3 coordinates, got shape {picks.shape}")
    return picks


def track_endpoints_ccf(xyz_picks_um, mode: str = "raise") -> np.ndarray:
    """Top and bottom pick of a track as a (2, 3) CCF array (AP, DV, ML)."""
    p = np.asarray(xyz_picks_um, dtype=float)
    return bregma_to_ccf(np.vstack([p[0], p[-1]]), ccf_order="apdvml", mode=mode)


def unit_position_from_deltas(
    peak_channel,
    delta_axial: float,
    delta_lateral: float,
    delta_perp: float,
    channel_data: dict,
) -> np.ndarray:
    """Shift a unit away from its peak channel along the probe's local axes.

    Places a unit using template-derived offsets instead of snapping it to a channel position.

    Parameters
    ----------
    peak_channel : hashable
        Key of the unit's peak channel in ``channel_data``.
    delta_axial, delta_lateral, delta_perp : float
        Offsets in um along the probe shank (positive = increasing local ``axial``),
        across the shank face (increasing local ``lateral``), and perpendicular to both.
    channel_data : dict
        ``{channel: {'local': [axial, lateral], 'ccf': [x, y, z]}}`` for the channels
        of ONE recording. ``'ccf'`` coordinates may be in any consistent frame
        (e.g. bregma-relative um); the result is in the same frame.
        Needs at least two channels with different ``axial`` positions.
    """
    base_local = np.asarray(channel_data[peak_channel]["local"], dtype=float)
    base = np.asarray(channel_data[peak_channel]["ccf"], dtype=float)
    chans = list(channel_data)

    ax_n = next((c for c in chans if channel_data[c]["local"][0] != base_local[0]), None)
    if ax_n is None:
        raise ValueError(f"No axial neighbour found for channel {peak_channel!r}.")
    lat_n = next((c for c in chans if channel_data[c]["local"][1] != base_local[1]), None)

    d_ax = np.asarray(channel_data[ax_n]["ccf"], dtype=float) - base
    axial = d_ax / (channel_data[ax_n]["local"][0] - base_local[0])
    axial /= np.linalg.norm(axial)

    if lat_n is not None:
        d_lat = np.asarray(channel_data[lat_n]["ccf"], dtype=float) - base
        lat_raw = d_lat / (channel_data[lat_n]["local"][1] - base_local[1])
        lat_raw = lat_raw - axial * (channel_data[lat_n]["local"][0] - base_local[0])
        lat = lat_raw - np.dot(lat_raw, axial) * axial  # orthogonalise
        lat /= np.linalg.norm(lat)
        perp = np.cross(axial, lat)
        perp /= np.linalg.norm(perp)
        return base + delta_axial * axial + delta_lateral * lat + delta_perp * perp

    # single-column probe: no lateral direction available, lateral delta ignored
    perp = np.cross(axial, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(axial, np.array([0.0, 1.0, 0.0]))
    perp /= np.linalg.norm(perp)
    return base + delta_axial * axial + delta_perp * perp
