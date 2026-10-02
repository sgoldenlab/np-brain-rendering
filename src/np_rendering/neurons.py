"""Render units (points) in the Allen CCF, coloured per unit; outside-brain check.

Adapted from the Golden Lab ``render_neurons.py`` / ``identify_outside_brain_new.py`` /
``final_rendering_neurons.py`` (brainrender, BSD-3-Clause; vedo, MIT).

Input coordinates must be brainrender CCF um (AP, DV, ML); convert bregma-relative data with
:func:`np_rendering.coords.bregma_to_ccf`.
"""

from __future__ import annotations

import numpy as np

from .colors import to_rgba_uint8
from .scene import add_legend, make_scene, render_settings, save_views


def identify_outside_brain(ccf_um, atlas=None, annotation=None, res_um: float = 25.0) -> dict:
    """Flag points that fall outside the brain (annotation value 0 / outside the volume).

    Provide either ``atlas`` (a BrainGlobeAtlas, e.g. ``scene.atlas``; gives region acronyms) or a
    raw ``annotation`` volume (integer array, AP x DV x ML, 0 = outside) -- the latter needs no
    atlas download and is what the unit tests use. Voxel 'root' (id 997) counts as INSIDE.

    Returns a dict with ``outside_indices``, ``inside_indices``, ``outside_mask``, ``region_ids``
    (-1 where outside the volume), ``regions`` (acronyms of inside points, when ``atlas`` given),
    ``n_outside``, ``n_inside``, ``percent_outside``.
    """
    pts = np.asarray(ccf_um, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"ccf_um must be (N, 3); got {pts.shape}")
    if annotation is None:
        if atlas is None:
            raise ValueError("pass `atlas` or `annotation`")
        annotation = atlas.annotation
        res_um = float(atlas.resolution[0]) if hasattr(atlas, "resolution") else res_um
    annotation = np.asarray(annotation)
    idx = np.floor(pts / res_um).astype(np.int64)
    in_vol = np.all((idx >= 0) & (idx < np.array(annotation.shape)) & np.isfinite(pts), axis=1)
    ids = np.full(len(pts), -1, dtype=np.int64)
    ids[in_vol] = annotation[tuple(idx[in_vol].T)]
    outside = ~in_vol | (ids == 0)
    result = {
        "outside_mask": outside,
        "outside_indices": np.flatnonzero(outside),
        "inside_indices": np.flatnonzero(~outside),
        "outside_coords": pts[outside],
        "region_ids": ids,
        "n_outside": int(outside.sum()),
        "n_inside": int((~outside).sum()),
        "percent_outside": float(outside.mean() * 100) if len(pts) else 0.0,
    }
    if atlas is not None:
        result["regions"] = [atlas.structures[int(i)]["acronym"] for i in ids[~outside]]
    return result


def detect_regions(atlas, ccf_um, hierarchy_level: int = 6, exclude=("root",)) -> list[str]:
    """Acronyms of the structures (coarsened to ``hierarchy_level``) containing the points."""
    res = identify_outside_brain(ccf_um, atlas=atlas)
    found = set()
    for i in np.unique(res["region_ids"][res["inside_indices"]]):
        path = atlas.structures[int(i)]["structure_id_path"]
        rid = path[hierarchy_level] if len(path) > hierarchy_level else int(i)
        acr = atlas.structures[int(rid)]["acronym"]
        if acr not in exclude:
            found.add(acr)
    return sorted(found)


def render_neurons(
    coordinates,
    point_color="darkred",
    point_size: float = 50,
    point_alpha: float = 1.0,
    coord_system: str = "ccf",
    coord_order: str = "mlapdv",
    title: str = "Neuron locations",
    regions=None,
    region_colors=None,
    region_alpha: float = 0.1,
    include_regions: bool = False,
    legend_labels=None,
    legend_colors=None,
    hide_outside: bool = False,
    highlight_outside: bool = False,
    outside_color="black",
    outside=None,
    screenshots: bool = False,
    out_dir="render_output/neurons",
    ss_label: str = "neurons",
    views=("frontal", "sagittal"),
    root_alpha=None,
    shader_style: str = "cartoon",
    zoom: float = 1.25,
    scale: float = 1.5,
    offscreen: bool = True,
):
    """Draw units as points.

    Parameters
    ----------
    coordinates : (N, 3)
        Brainrender CCF um (AP, DV, ML) if ``coord_system='ccf'`` (default); or bregma-relative um
        in ``coord_order`` if ``coord_system='bregma'`` (converted with iblatlas, mode='clip').
    point_color : colour name, or (N,3)/(N,4) per-unit colours (0-1 floats or 0-255)
        e.g. the output of :func:`np_rendering.colors.metric_to_rgba`.
    point_size : float
        vedo point radius parameter (default 50; 7-10 suits many units).
    hide_outside / highlight_outside : bool
        Make units outside the brain transparent / recolour them with ``outside_color``.
        ``outside`` may be the dict from :func:`identify_outside_brain` or a boolean mask; if omitted it is
        computed from the scene's atlas.
    include_regions, regions, region_colors, region_alpha
        Draw regions (acronyms) as transparent meshes. ``regions=None`` auto-detects from the unit positions.
    screenshots : bool
        True -> save ``<out_dir>/<ss_label>_<view>.png`` per view; False -> interactive window.
    root_alpha : float or dict, optional  (default 0.05 for three_quarters, 0.03 otherwise)

    Returns
    -------
    list of str : screenshot paths (empty if interactive).
    """
    import vedo
    from brainrender.actor import Actor

    pts = np.asarray(coordinates, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"coordinates must be (N, 3); got {pts.shape}")
    if coord_system == "bregma":
        from .coords import bregma_to_ccf

        pts = bregma_to_ccf(pts, input_order=coord_order, ccf_order="apdvml", mode="clip")
    elif coord_system != "ccf":
        raise ValueError("coord_system must be 'ccf' or 'bregma'")
    root_alpha = {"three_quarters": 0.05, "default": 0.03} if root_alpha is None else root_alpha

    with render_settings(shader_style=shader_style, offscreen=offscreen if screenshots else False):
        scene = make_scene(title=title, camera="three_quarters")
        try:
            outside_mask = None
            if hide_outside or highlight_outside:
                if outside is None:
                    outside = identify_outside_brain(pts, atlas=scene.atlas)
                outside_mask = np.asarray(outside["outside_mask"] if isinstance(outside, dict) else outside, bool)
                if len(outside_mask) != len(pts):
                    raise ValueError("`outside` does not match the number of units")

            mesh = vedo.Points(pts, r=point_size)
            rgba = to_rgba_uint8(point_color, n=len(pts), alpha=point_alpha)
            if len(rgba) != len(pts):
                raise ValueError("per-unit colours must have one row per unit")
            if outside_mask is not None and outside_mask.any():
                if hide_outside:
                    rgba[outside_mask, 3] = 0
                else:
                    rgba[outside_mask] = to_rgba_uint8(outside_color, n=1)[0]
            mesh.pointcolors = rgba
            scene.add(Actor(mesh, name="Neurons", br_class="Points"))
            scene.root._needs_silhouette = True

            if include_regions:
                if regions is None:
                    regions = detect_regions(scene.atlas, pts)
                rc = list(region_colors) if region_colors else [
                    "blue", "red", "green", "orange", "purple", "cyan", "magenta", "lime", "pink", "teal"
                ]
                for i, reg in enumerate(regions):
                    col = rc[i % len(rc)]
                    try:
                        obj = scene.add_brain_region(reg, alpha=region_alpha, color=col, silhouette=False)
                        obj._needs_silhouette = False
                        if screenshots:
                            scene.add_label(obj, reg, size=150, radius=0, color=col, zoffset=-1000, xoffset=150, yoffset=500)
                    except Exception as e:  # unknown acronym etc.: skip
                        print(f"Could not add region {reg}: {e}")

            if legend_labels is not None and legend_colors is not None:
                add_legend(scene, legend_labels, legend_colors)

            if screenshots:
                return save_views(scene, views, out_dir, ss_label, zoom=zoom, scale=scale, root_alpha=root_alpha)
            scene.render(camera="three_quarters", interactive=True)
            return []
        finally:
            scene.close()
