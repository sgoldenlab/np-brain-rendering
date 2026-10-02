import json
import subprocess
import sys

import numpy as np
import pytest

import np_rendering as npr
from np_rendering import animation, colors, coords
from np_rendering.neurons import identify_outside_brain

iblatlas = pytest.importorskip("iblatlas")


def test_import_is_light():
    code = (
        "import sys\nfor m in ('brainrender','vedo','iblatlas','vtk'):"
        " sys.modules[m]=None\nimport np_rendering"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_bregma_maps_to_allen_landmark():
    out = coords.bregma_to_ccf([0, 0, 0])
    np.testing.assert_allclose(out, [5400, 332, 5739])  # AP, DV, ML


def test_bregma_axes_signs():
    # +ML -> larger CCF ML, +AP (anterior) -> SMALLER CCF AP, -DV (ventral) -> LARGER CCF DV
    out = coords.bregma_to_ccf([1000, 2000, -3000])
    np.testing.assert_allclose(out, [3400, 3332, 6739])
    # a simple abs(AP)+5400 approximation would give 7400 for the anterior point
    assert out[0] < 5400


def test_orders_and_roundtrip():
    a = coords.bregma_to_ccf([[1000, -1000, -500]], input_order="mlapdv", ccf_order="mlapdv")
    b = coords.bregma_to_ccf([[-1000, -500, 1000]], input_order="apdvml", ccf_order="mlapdv")
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(coords.ccf_to_bregma(a, ccf_order="mlapdv"), [[1000, -1000, -500]], atol=1e-6)
    with pytest.raises(ValueError):
        coords.bregma_to_ccf([0, 0, 0], input_order="mlapml")


def test_outside_volume_modes():
    with pytest.raises(ValueError):
        coords.bregma_to_ccf([0, -9000, 0], mode="raise")
    assert np.isfinite(coords.bregma_to_ccf([0, -9000, 0], mode="clip")).all()


def test_xyz_picks_roundtrip(tmp_path):
    f = tmp_path / "x_xyz_picks.json"
    f.write_text(json.dumps({"xyz_picks": [[0, 0, -100], [100, -200, -3000]]}))
    p = coords.load_xyz_picks(f)
    assert p.shape == (2, 3)
    t = coords.track_endpoints_ccf(p)
    assert t.shape == (2, 3)
    f.write_text(json.dumps({"xyz_picks": [[0, 0, 0]]}))
    with pytest.raises(ValueError):
        coords.load_xyz_picks(f)


def test_unit_position_from_deltas_straight_probe():
    # probe along +DV (index 2) in the 'ccf' frame; channel local = [axial, lateral]
    ch = {
        0: {"local": [0, 0], "ccf": [0, 0, 0]},
        1: {"local": [20, 0], "ccf": [0, 0, 20]},
        2: {"local": [0, 32], "ccf": [0, 32, 0]},
    }
    out = coords.unit_position_from_deltas(0, 10, 5, 0, ch)
    np.testing.assert_allclose(out, [0, 5, 10], atol=1e-9)
    with pytest.raises(ValueError):
        coords.unit_position_from_deltas(0, 1, 1, 1, {0: ch[0]})


def test_metric_to_rgba():
    out = colors.metric_to_rgba([0, 0.5, 1, np.nan], "viridis", vmin=0, vmax=1)
    assert out.shape == (4, 4) and out.dtype == np.uint8
    np.testing.assert_array_equal(out[0], [68, 1, 84, 255])
    np.testing.assert_array_equal(out[3, :3], [160, 160, 160])
    assert colors.metric_to_rgba([np.nan], nan_color=None)[0, 3] == 0
    # clipping and alpha
    o2 = colors.metric_to_rgba([-5, 5], "viridis", 0, 1, alpha=0.5)
    assert o2[0, 3] == o2[1, 3] == 128
    np.testing.assert_array_equal(o2[0, :3], [68, 1, 84])
    # percentile limits
    o3 = colors.metric_to_rgba(np.arange(101.0), "viridis", percentile=(5, 95))
    np.testing.assert_array_equal(o3[0, :3], o3[5, :3])


def test_minmax_per_unit_matches_sklearn():
    m = np.array([[1.0, 2, 3], [5, 5, 5], [0, 10, np.nan]])
    out = colors.minmax_per_unit(m)
    np.testing.assert_allclose(out[0], [0, 0.5, 1])
    np.testing.assert_allclose(out[1], [0, 0, 0])
    assert out[2, 2] == 0
    sk = pytest.importorskip("sklearn.preprocessing")
    ref = sk.MinMaxScaler().fit_transform(m[:2].T).T
    np.testing.assert_allclose(out[:2], ref)


def test_categorical_and_rgba_helpers():
    d = colors.categorical_colors(["b", "a", "b"])
    assert list(d) == ["a", "b"] and d["a"] == colors.G10[0]
    assert colors.to_rgba_uint8("red", n=2).shape == (2, 4)
    assert colors.to_rgba_uint8(np.array([[1.0, 0, 0]]))[0].tolist() == [255, 0, 0, 255]
    assert colors.to_rgba_uint8(np.array([[255, 0, 0]], dtype=np.uint8))[0].tolist() == [255, 0, 0, 255]


def test_identify_outside_with_mock_annotation():
    ann = np.zeros((10, 8, 6), dtype=np.int32)
    ann[2:8, 1:7, 1:5] = 997  # 'root' counts as inside
    ann[4, 3, 2] = 5
    res = 25.0
    pts = np.array([
        [4.5 * res, 3.5 * res, 2.5 * res],   # inside, region 5
        [0.5 * res, 0.5 * res, 0.5 * res],   # in volume, id 0 -> outside
        [-1.0, 50.0, 50.0],                  # negative -> outside
        [9999.0, 50.0, 50.0],                # beyond volume -> outside
        [3.5 * res, 3.5 * res, 2.5 * res],   # root -> inside
    ])
    r = identify_outside_brain(pts, annotation=ann, res_um=res)
    assert r["outside_indices"].tolist() == [1, 2, 3]
    assert r["inside_indices"].tolist() == [0, 4]
    assert r["region_ids"].tolist() == [5, 0, -1, -1, 997]
    assert r["n_outside"] == 3 and r["percent_outside"] == pytest.approx(60.0)


def test_identify_outside_with_fake_atlas():
    class A:
        resolution = (25, 25, 25)
        annotation = np.full((4, 4, 4), 997)
        structures = {997: {"acronym": "root"}}

    r = identify_outside_brain([[30.0, 30.0, 30.0]], atlas=A())
    assert r["regions"] == ["root"]


def test_animation_helpers():
    assert animation.frame_to_bin(0, 100, 60) == 0
    assert animation.frame_to_bin(99, 100, 60) == 59
    assert animation.frame_to_bin(0, 1, 60) == 0
    lo, hi = animation.color_limits(np.arange(101.0))
    assert (lo, hi) == (5.0, 95.0)
    assert animation.color_limits([1, 1, 1])[1] > 1
    c, _, kept = animation.clean_units(np.ones((3, 3)), np.array([[1.0, 2], [np.nan, 1], [1, 1]]))
    assert c.shape == (2, 3) and kept.tolist() == [True, False, True]
    cols, (lo, hi) = animation.precompute_frame_colors(np.random.rand(5, 4), vmin=0, vmax=1)
    assert len(cols) == 4 and cols[0].shape == (5, 4)
    with pytest.raises(ValueError):
        animation.clean_units(np.ones((3, 3)), np.ones((2, 4)))


def test_frames_to_video(tmp_path):
    iio = pytest.importorskip("imageio.v3")
    try:
        animation.find_ffmpeg()
    except RuntimeError:
        pytest.skip("no ffmpeg")
    for i in range(6):
        iio.imwrite(tmp_path / f"{i}.png", np.full((33, 47, 3), i * 40, dtype=np.uint8))  # odd size
    out = animation.frames_to_video(tmp_path, tmp_path / "o" / "a.mp4", fps=6)
    assert (tmp_path / "o" / "a.mp4").stat().st_size > 0 and out.endswith("a.mp4")
    with pytest.raises(RuntimeError):
        animation.frames_to_video(tmp_path / "missing", tmp_path / "b.mp4")


def test_package_exports():
    assert npr.__version__
