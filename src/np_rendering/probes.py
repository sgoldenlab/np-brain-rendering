"""Render probe tracks (straight cylinders between top and bottom pick) in the Allen CCF.

Adapted from the Golden Lab ``render_probes.py`` / ``final_rendering_probe_tracks.py``
(brainrender, BSD-3-Clause; vedo, MIT). Input tracks must already be in brainrender CCF um
(AP, DV, ML) -- use :func:`np_rendering.coords.bregma_to_ccf`.
"""

from __future__ import annotations

from itertools import cycle

import numpy as np

from .scene import add_legend, make_scene, render_settings, save_views

DEFAULT_ROOT_ALPHA = {"three_quarters": 0.5, "default": 0.05}


def probe_view_hook(all_points_ccf, margin_um: float = 10.0):
    """Root-slicing hook used for the probe figures.

    sagittal: cut away everything lateral of the most lateral probe (so tracks are visible);
    top: keep only the dorsal half. Other views are left unsliced.
    """
    pts = np.asarray(all_points_ccf, dtype=float)

    def hook(scene, view):
        root = scene.root
        if view == "sagittal":
            min_z = pts[:, 2].min()
            plane = scene.atlas.get_plane(pos=[0, 0, min_z - margin_um], plane="sagittal", norm=(0, 0, -1))
            scene.slice(plane, actors=[root], invert=False)
        elif view == "top":
            scene.slice("horizontal", actors=[root], invert=True)
            mesh = getattr(root, "mesh", None) or root._mesh  # brainrender >=2.2 / 2.1.x
            b = mesh.bounds()
            plane = scene.atlas.get_plane([b[1], b[3] - 1, b[5]], norm=(0, 1, 0))
            scene.slice(plane, actors=[root], invert=False)

    return hook


def render_probes(
    tracks,
    labels=None,
    colors=("darkred",),
    legend_labels=None,
    legend_colors=None,
    title: str = "Probe trajectories",
    show_labels: bool = False,
    regions=None,
    region_colors=None,
    region_alpha: float = 0.1,
    radius_um: float = 25,
    screenshots: bool = False,
    out_dir="render_output/probe_tracks",
    ss_label: str = "probe_tracks",
    views=("frontal", "sagittal"),
    root_alpha=None,
    shader_style: str = "plastic",
    zoom: float = 1.25,
    scale: float = 1.5,
    offscreen: bool = True,
):
    """Draw probe tracks.

    Parameters
    ----------
    tracks : list of (2, 3) arrays
        Top and bottom point of each track in brainrender CCF um (AP, DV, ML).
    labels : list of str, optional
        One per track (only drawn if ``show_labels``).
    colors : list of colours
        One per track; cycled if shorter.
    legend_labels, legend_colors : optional
        One sphere + text per entry (e.g. one per animal).
    regions, region_colors : list of str, optional
        Acronyms (e.g. ``['BLA','CEA']``) of brain regions to draw as transparent meshes.
    radius_um : float, default 25
        Cylinder radius.
    screenshots : bool
        True -> save ``<out_dir>/<ss_label>_<view>.png`` for each of ``views``; False -> open the
        interactive window.
    views : subset of 'frontal', 'sagittal', 'top', 'three_quarters'.
    root_alpha : float or dict, optional
        Brain-shell opacity per view (default: 0.5 for three_quarters, 0.05 otherwise).
    shader_style : str, default 'plastic' (probe figures); 'cartoon' for neurons.
    offscreen : bool, default True
        Offscreen screenshots (needed headless). Ignored for interactive display.

    Returns
    -------
    list of str : screenshot paths (empty if interactive).
    """
    import vedo
    from brainrender.actor import Actor

    tracks = [np.asarray(t, dtype=float) for t in tracks]
    if not tracks:
        raise ValueError("no tracks given")
    labels = list(labels) if labels is not None else [f"probe_{i}" for i in range(len(tracks))]
    if len(labels) != len(tracks):
        raise ValueError("length of labels does not match number of tracks")
    colors = list(colors) if not isinstance(colors, str) else [colors]
    if len(colors) < len(tracks):
        colors = list(cycle(colors))[: len(tracks)]
    root_alpha = DEFAULT_ROOT_ALPHA if root_alpha is None else root_alpha

    with render_settings(shader_style=shader_style, offscreen=offscreen if screenshots else False):
        scene = make_scene(title=title, camera="frontal")
        try:
            for trk, lab, col in zip(tracks, labels, colors):
                cyl = vedo.shapes.Cylinder(trk, r=radius_um, res=24, cap=False)
                actor = Actor(cyl, name="Cylinder", br_class="Cylinder", color=col)
                scene.add(actor, color=col)
                if show_labels:
                    scene.add_label(actor, lab, size=200, radius=0, color=col, zoffset=300, xoffset=-400)
            scene.root._needs_silhouette = True  # cartoon outline

            if regions:
                from itertools import islice

                rcols = list(region_colors) if region_colors else ["red"] * len(regions)
                for reg, rc in zip(regions, islice(cycle(rcols), len(regions))):
                    obj = scene.add_brain_region(reg, alpha=region_alpha, color=rc, silhouette=False)
                    obj._needs_silhouette = False
                    if show_labels:
                        scene.add_label(obj, reg, size=150, radius=0, color=rc, zoffset=-1000, xoffset=150, yoffset=500)

            if legend_labels is not None and legend_colors is not None:
                add_legend(scene, legend_labels, legend_colors)

            if screenshots:
                return save_views(
                    scene, views, out_dir, ss_label, zoom=zoom, scale=scale, root_alpha=root_alpha,
                    view_hook=probe_view_hook(np.vstack(tracks)),
                )
            scene.render(camera="frontal", interactive=True)
            return []
        finally:
            scene.close()
