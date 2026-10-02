"""brainrender scene helpers shared by probes / neurons / animation.

Built on brainrender (BSD-3-Clause) and vedo (MIT). Global styling (shader, line width,
root alpha, vedo backend ...) is applied inside :func:`render_settings`, only while a
function runs.

brainrender/vedo are imported lazily.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

DEFAULT_ATLAS = "allen_mouse_25um"
# The lab's custom 'frontal' camera position (brainrender CCF um, x/y/z = AP/DV/ML) used instead of
# brainrender's own preset; set to None to use the stock preset.
FRONTAL_POS_DEFAULT = (-28919.2, 3172.74, -5425.18)
VIEWS = ("frontal", "sagittal", "top", "three_quarters")


@contextmanager
def render_settings(
    shader_style: str = "cartoon",
    root_alpha: float = 0.3,
    line_width: float = 2,
    show_axes: bool = False,
    offscreen: bool = True,
    background_color: str | None = None,
    screenshot_scale: float | None = None,
):
    """Temporarily apply brainrender/vedo global settings (restored on exit).

    Defaults: 'cartoon' shader,
    ROOT_ALPHA 0.3, LW 2, axes hidden, OFFSCREEN screenshots, vtk backend.
    """
    import vedo
    from brainrender import settings

    names = {
        "SHADER_STYLE": shader_style,
        "ROOT_ALPHA": root_alpha,
        "LW": line_width,
        "SHOW_AXES": show_axes,
        "OFFSCREEN": offscreen,
    }
    if background_color is not None:
        names["BACKGROUND_COLOR"] = background_color
    if screenshot_scale is not None:
        names["SCREENSHOT_SCALE"] = screenshot_scale
    old = {k: getattr(settings, k) for k in names if hasattr(settings, k)}
    old_backend = vedo.settings.default_backend
    old_fxaa = getattr(vedo.settings, "use_fxaa", None)
    for k, v in names.items():
        setattr(settings, k, v)
    vedo.settings.default_backend = "vtk"
    vedo.settings.use_fxaa = True
    try:
        yield settings
    finally:
        for k, v in old.items():
            setattr(settings, k, v)
        vedo.settings.default_backend = old_backend
        if old_fxaa is not None:
            vedo.settings.use_fxaa = old_fxaa


def make_scene(title: str | None = None, atlas_name: str = DEFAULT_ATLAS, camera: str | None = "three_quarters"):
    """Create a brainrender Scene (no inset) and set the camera early (avoids silhouette warnings)."""
    from brainrender import Scene
    from brainrender.camera import set_camera
    from brainrender.cameras import cameras

    scene = Scene(title=title, atlas_name=atlas_name, inset=False)
    if camera:
        set_camera(scene, cameras[camera])
    return scene


def get_camera(scene, view: str, frontal_pos=FRONTAL_POS_DEFAULT) -> dict:
    """Camera dict for a named view (``frontal``/``sagittal``/``top``/``three_quarters``).

    Applies it to the scene and returns a COPY of the dict. For 'frontal' the lab's
    custom camera position is used unless ``frontal_pos`` is None.
    """
    from brainrender.camera import set_camera

    cam = dict(set_camera(scene, view))
    if view == "frontal" and frontal_pos is not None:
        cam["pos"] = tuple(frontal_pos)
        set_camera(scene, cam)
    return cam


def add_legend(scene, labels, colors, origin=(-15000, 8000, 0), step=800, marker_r=200, text_size=300):
    """Legend as small spheres + text labels placed beside the brain."""
    import vedo
    from brainrender.actor import Actor

    for i, (label, color) in enumerate(zip(labels, colors)):
        if isinstance(color, (list, tuple, np.ndarray)) and len(color) >= 3:
            c = np.asarray(color[:3], dtype=float)
            color = tuple(int(x) for x in (c * 255 if c.max() <= 1.0 else c))
        marker = vedo.Sphere(r=marker_r, c=color)
        marker.pos([origin[0], origin[1] - i * step, origin[2]])
        actor = Actor(marker, name=f"legend_{i}", br_class="Sphere")
        scene.add(actor)
        scene.add_label(
            actor, str(label), size=text_size, radius=0, color=color, xoffset=500, yoffset=0, zoffset=-700
        )


def _alpha_for(root_alpha, view):
    if isinstance(root_alpha, dict):
        return root_alpha.get(view, root_alpha.get("default", 0.05))
    return float(root_alpha)


def save_views(
    scene,
    views,
    out_dir,
    label: str,
    zoom: float = 1.25,
    scale: float = 1.5,
    root_alpha=0.05,
    silhouette_lw: float = 8,
    frontal_pos=FRONTAL_POS_DEFAULT,
    view_hook=None,
    ext: str = "png",
) -> list[str]:
    """Render the scene from several named views and save ``<out_dir>/<label>_<view>.<ext>``.

    The root (whole-brain) mesh is (re)added per view with ``root_alpha`` (float or
    ``{view: alpha, 'default': alpha}``), black silhouette for every view except
    'three_quarters', then removed again. ``view_hook(scene, view)`` runs after the root
    is added (probes use it to slice the root). Paths use ``os.path``/pathlib so this works
    on Windows and Linux.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # drop the scene's own root so every view gets the same, explicit root settings
    for a in list(scene.get_actors("root") or []):
        scene.remove(a)
    saved = []
    for view in views:
        sil = view != "three_quarters"
        root = scene.add_brain_region("root", alpha=_alpha_for(root_alpha, view), silhouette=sil)
        scene.root = root
        if sil:
            scene.add_silhouette(root, lw=silhouette_lw, color="k")
        if view_hook is not None:
            view_hook(scene, view)
        cam = get_camera(scene, view, frontal_pos=frontal_pos)
        scene.render(camera=cam, interactive=False, zoom=zoom)
        path = out_dir / f"{label}_{view}.{ext}"
        scene.screenshot(name=os.fspath(path), scale=scale)
        saved.append(os.fspath(path))
        scene.remove(root)
    return saved
