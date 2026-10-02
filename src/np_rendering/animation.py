"""Animate unit activity (colour per time bin) with a rotating camera and write an mp4.

Adapted from ``animate_neurons`` in the Golden Lab ``render_neurons.py`` (current version only),
built on brainrender (BSD-3-Clause), vedo (MIT) and an external ``ffmpeg``.

Pipeline: (units x time bins) matrix -> RGBA per bin (:func:`precompute_frame_colors`) ->
per-frame offscreen render with camera Azimuth rotation -> PNG frames -> libx264 mp4
(:func:`frames_to_video`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .colors import metric_to_rgba


def clean_units(coordinates, values):
    """Drop units with any NaN in their coordinates or in their time series.

    Returns ``(coordinates, values, kept_mask)``.
    """
    coords = np.asarray(coordinates, dtype=float)
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 2 or len(vals) != len(coords):
        raise ValueError(f"values must be (N units, T bins) matching coordinates; got {vals.shape} vs {coords.shape}")
    keep = ~np.isnan(coords).any(axis=1) & ~np.isnan(vals).any(axis=1)
    return coords[keep], vals[keep], keep


def color_limits(values, vmin=None, vmax=None, percentile=(5, 95)):
    """Colour limits: given vmin/vmax, else the 5th/95th percentile of all values."""
    flat = np.asarray(values, dtype=float).ravel()
    lo = float(np.nanpercentile(flat, percentile[0])) if vmin is None else float(vmin)
    hi = float(np.nanpercentile(flat, percentile[1])) if vmax is None else float(vmax)
    if hi <= lo:
        hi = float(np.nanmax(flat))
        if hi <= lo:
            hi = lo + 1e-5
    return lo, hi


def precompute_frame_colors(values, cmap="hot", vmin=None, vmax=None, percentile=(5, 95)):
    """RGBA uint8 array (N, 4) for every time bin -> list of length T."""
    vals = np.asarray(values, dtype=float)
    lo, hi = color_limits(vals, vmin, vmax, percentile)
    # NaN -> colour of vmin
    return [metric_to_rgba(vals[:, t], cmap, lo, hi, nan_color=None) for t in range(vals.shape[1])], (lo, hi)


def frame_to_bin(frame: int, total_frames: int, n_bins: int) -> int:
    """Map a video frame to a time-bin index (linear, first->0, last->T-1)."""
    idx = int((frame / max(1, total_frames - 1)) * (n_bins - 1))
    return min(max(idx, 0), n_bins - 1)


def find_ffmpeg() -> str:
    """`ffmpeg` on PATH, else the binary shipped with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`") from e


def frames_to_video(frame_dir, output_path, fps: int = 30, crf: int = 15, size=None, pattern="%d.png"):
    """Encode numbered PNG frames (0-based, ``pattern``) into an H.264 mp4 (yuv420p).

    size : (w, h) or None. None keeps the render size (rounded down to even numbers, required by
    yuv420p).
    Raises RuntimeError if ffmpeg fails.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-r", str(fps), "-start_number", "0",
        "-i", os.fspath(Path(frame_dir) / pattern),
        "-vcodec", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
    ]  # fmt: skip
    if size is not None:
        cmd += ["-s", f"{int(size[0])}x{int(size[1])}"]
    else:
        cmd += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    cmd.append(os.fspath(out))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (code {res.returncode}):\n{res.stderr}")
    return os.fspath(out.resolve())


def animate_neurons(
    coordinates,
    values,
    output_path,
    duration: float = 10.0,
    bins_per_second: float | None = None,
    fps: int = 30,
    cmap: str = "hot",
    point_size: float = 30.0,
    vmin: float | None = None,
    vmax: float | None = None,
    rotate: bool = True,
    rotate_degrees: float = 180.0,
    regions=None,
    region_alpha: float = 0.15,
    title: str = "Neural activity",
    zoom: float = 1.25,
    show_timer: bool = True,
    timer_pos: str = "top-right",
    timer_color: str = "black",
    timer_size: float = 1.5,
    timer_start: float = 0.0,
    timer_prefix: str = "Time: ",
    time_per_bin: float = 1.0,
    dark_mode: bool = False,
    root_alpha: float | None = None,
    crf: int = 15,
    video_size=None,
    keep_frames_dir=None,
    verbose: bool = True,
) -> str:
    """Render an activity animation of units on the Allen CCF brain.

    Parameters
    ----------
    coordinates : (N, 3)
        brainrender CCF um (AP, DV, ML); see :mod:`np_rendering.coords`.
    values : (N, T)
        Activity per unit and time bin (rows align with ``coordinates``). Units with any NaN are dropped.
        Scale it yourself first if wanted (e.g. :func:`np_rendering.colors.minmax_per_unit`; per-unit scaling to [0, 1]
        with vmin=0.2, vmax=1.4 gives a softer look).
    duration : float, default 10.0 (s)
        Video length. If ``bins_per_second`` is given, ``duration = T / bins_per_second`` instead
        (36 bins per second of video is a common choice).
    vmin, vmax : float or None
        Colour limits; None -> 5th/95th percentile of all values.
    rotate, rotate_degrees : turn the camera by this many degrees (azimuth) over the whole video.
    dark_mode : black background, white faint brain, white timer text.
    root_alpha : brain-shell opacity (default 0.05 light / 0.015 dark).
    crf : H.264 quality (lower = better; default 15).
    video_size : (w, h) or None.  ``(1666, 1080)`` is a common choice.
    keep_frames_dir : keep the PNG frames in this folder instead of a temp dir.

    Returns
    -------
    str : absolute path of the mp4.

    Notes
    -----
    Needs a working OpenGL context (offscreen is used; on headless Linux run under ``xvfb-run``).
    """
    import vedo
    from brainrender.actor import Actor
    from brainrender.cameras import cameras

    from .scene import make_scene, render_settings

    coords, vals, keep = clean_units(coordinates, values)
    n_units, n_bins = vals.shape
    if n_units == 0:
        raise ValueError("No valid units left to render after dropping NaNs")
    if verbose:
        print(f"Dropped {len(keep) - n_units} units with NaNs; rendering {n_units} units, {n_bins} bins.")
    colors, (lo, hi) = precompute_frame_colors(vals, cmap, vmin, vmax)
    if verbose:
        print(f"Colour limits: vmin={lo:.4f}, vmax={hi:.4f}")

    if bins_per_second:
        duration = n_bins / float(bins_per_second)
    total_frames = int(duration * fps)
    az_step = rotate_degrees / max(1, total_frames - 1) if rotate else 0.0
    if dark_mode and timer_color == "black":
        timer_color = "white"

    own_tmp = keep_frames_dir is None
    tmp = tempfile.TemporaryDirectory() if own_tmp else None
    frame_dir = Path(tmp.name if own_tmp else keep_frames_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    with render_settings(
        shader_style="cartoon", offscreen=True, background_color="black" if dark_mode else None
    ):
        scene = make_scene(title=title, camera="three_quarters")
        try:
            if scene.root is not None:
                active = (0.015 if dark_mode else 0.05) if root_alpha is None else root_alpha
                scene.root.color("white" if dark_mode else "black")
                scene.root.alpha(active)
                scene.root._needs_silhouette = False  # silhouette adds a grey overlay
            if regions:
                rcols = ["blue", "red", "green", "orange", "purple", "cyan", "magenta"]
                for reg, rc in zip(regions, rcols):
                    try:
                        scene.add_brain_region(reg, alpha=region_alpha, color=rc, silhouette=False)
                    except Exception as e:
                        print(f"Could not add region {reg}: {e}")

            mesh = vedo.Points(coords, r=point_size)
            mesh.pointcolors = colors[0]
            actor = Actor(mesh, name="Neurons", br_class="Points")
            scene.add(actor)

            timer = None
            if show_timer:
                timer = vedo.Text2D(
                    f"{timer_prefix}{timer_start:.1f}s", pos=timer_pos, s=timer_size, c=timer_color, font="Calco"
                )
                scene.add(timer)

            scene.render(camera=cameras["three_quarters"], interactive=False, zoom=zoom)
            if verbose:
                print(f"Rendering {total_frames} frames ({duration:.1f}s @ {fps} fps)...")
            for i in range(total_frames):
                b = frame_to_bin(i, total_frames, n_bins)
                scene.remove(actor)
                mesh = vedo.Points(coords, r=point_size)
                mesh.pointcolors = colors[b]
                actor = Actor(mesh, name="Neurons", br_class="Points")
                scene.add(actor)
                if timer is not None:
                    t = timer_start + (i / max(1, total_frames - 1)) * (n_bins - 1) * time_per_bin
                    timer.text(f"{timer_prefix}{t:.1f}s")
                if rotate:
                    scene.plotter.camera.Azimuth(az_step)
                scene.plotter.window.Render()
                scene.plotter.screenshot(os.fspath(frame_dir / f"{i}.png"))
            out = frames_to_video(frame_dir, output_path, fps=fps, crf=crf, size=video_size)
        finally:
            scene.close()
            if tmp is not None:
                tmp.cleanup()
    if verbose:
        print(f"Video saved to: {out}")
    return out
