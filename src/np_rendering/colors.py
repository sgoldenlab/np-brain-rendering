"""Colour helpers (numpy + matplotlib only)."""

from __future__ import annotations

import numpy as np

# plotly "G10" qualitative palette (used for animals/recordings).
G10 = [
    "#3366CC", "#DC3912", "#FF9900", "#109618", "#990099",
    "#0099C6", "#DD4477", "#66AA00", "#B82E2E", "#316395",
]  # fmt: skip


def _get_cmap(cmap):
    import matplotlib

    if not isinstance(cmap, str):
        return cmap
    try:
        return matplotlib.colormaps[cmap]
    except AttributeError:  # matplotlib < 3.5
        import matplotlib.cm as cm

        return cm.get_cmap(cmap)


def metric_to_rgba(
    values,
    cmap="viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    percentile: tuple[float, float] | None = None,
    alpha: float = 1.0,
    nan_color=(160, 160, 160),
) -> np.ndarray:
    """Colour units by a metric (e.g. firing rate, delta, aperiodic exponent).

    Parameters
    ----------
    values : (N,) array-like
        Metric per unit; NaN allowed.
    cmap : str or matplotlib Colormap
    vmin, vmax : float or None
        Colour limits. None -> from the data (see ``percentile``).
    percentile : (lo, hi) or None
        If vmin/vmax are None and this is given, use these percentiles of the finite
        values (e.g. (5, 95)); otherwise min/max.
    alpha : float in [0, 1]
        Opacity of coloured points.
    nan_color : RGB tuple (0-255) or None
        Colour for NaN values; None makes NaN units fully transparent.

    Returns
    -------
    (N, 4) uint8 RGBA array, ready for ``render_neurons(point_color=...)``.
    """
    v = np.asarray(values, dtype=float).ravel()
    finite = v[np.isfinite(v)]
    lo, hi = vmin, vmax
    if (lo is None or hi is None) and finite.size:
        if percentile is not None:
            plo, phi = np.percentile(finite, percentile)
        else:
            plo, phi = finite.min(), finite.max()
        lo = plo if lo is None else lo
        hi = phi if hi is None else hi
    if lo is None or hi is None:  # all NaN
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1e-12
    x = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    rgba = _get_cmap(cmap)(np.where(np.isfinite(x), x, 0.0))  # (N, 4) floats
    rgba[:, 3] = alpha
    out = np.round(rgba * 255).astype(np.uint8)
    bad = ~np.isfinite(v)
    if bad.any():
        if nan_color is None:
            out[bad, 3] = 0
        else:
            out[bad, :3] = np.asarray(nan_color, dtype=np.uint8)[:3]
    return out


def minmax_per_unit(matrix) -> np.ndarray:
    """Scale each row (unit) of an (N units, T bins) matrix to [0, 1] over time.

    Equivalent to ``MinMaxScaler().fit_transform(m.T).T``:
    constant rows become 0, NaN become 0.
    """
    m = np.asarray(matrix, dtype=float)
    lo = np.nanmin(m, axis=1, keepdims=True)
    rng = np.nanmax(m, axis=1, keepdims=True) - lo
    rng = np.where(rng == 0, 1.0, rng)
    return np.nan_to_num((m - lo) / rng, nan=0.0)


def categorical_colors(keys, palette=None) -> dict:
    """Stable mapping ``{key: hex colour}`` (sorted keys, palette cycled)."""
    palette = list(palette or G10)
    uniq = sorted(set(keys))
    return {k: palette[i % len(palette)] for i, k in enumerate(uniq)}


def to_rgba_uint8(colors, n: int | None = None, alpha: float = 1.0) -> np.ndarray:
    """Normalise colours to an (N, 4) uint8 array.

    Accepts (N,3)/(N,4) arrays in 0-1 floats or 0-255 ints, a single colour name/hex,
    or a list of names/hex.
    """
    from matplotlib.colors import to_rgba

    if isinstance(colors, str) or (
        np.ndim(colors) == 1 and len(colors) in (3, 4) and not isinstance(colors[0], str)
    ):
        colors = [colors] * (n or 1)
    arr = None
    if not isinstance(colors, np.ndarray) and len(colors) and isinstance(colors[0], str):
        arr = np.array([to_rgba(c) for c in colors])
    else:
        a = np.asarray(colors, dtype=float)
        if a.ndim == 1:
            a = np.tile(a, (n or 1, 1))
        if a.shape[1] == 3:
            a = np.hstack([a, np.full((len(a), 1), alpha if a.max() <= 1 else alpha * 255)])
        arr = a if a.max() <= 1.0 else a / 255.0
    return np.round(np.clip(arr, 0, 1) * 255).astype(np.uint8)
