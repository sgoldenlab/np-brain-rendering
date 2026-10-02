"""Control-flow tests with a fake brainrender Scene (no atlas download, real vedo offscreen).

These do NOT validate brainrender's atlas/mesh behaviour (needs the BrainGlobe atlas, which
cannot be downloaded in restricted sandboxes); they validate argument handling, per-view loops,
paths, colours and the animation frame/video pipeline.
"""

import numpy as np
import pytest

vedo = pytest.importorskip("vedo")
pytest.importorskip("brainrender")
from brainrender.actor import Actor  # noqa: E402

import np_rendering.animation as anim  # noqa: E402
import np_rendering.neurons as neurons  # noqa: E402
import np_rendering.probes as probes  # noqa: E402
import np_rendering.scene as scene_mod  # noqa: E402


class FakeAtlas:
    resolution = (25, 25, 25)
    annotation = np.full((528, 320, 456), 997, dtype=np.int32)
    structures = {997: {"acronym": "root", "structure_id_path": [997]}}

    def get_plane(self, pos=None, plane=None, norm=None):
        return vedo.Plane()


class FakeScene:
    def __init__(self):
        self.plotter = vedo.Plotter(offscreen=True, size=(160, 120), axes=0)
        self.atlas = FakeAtlas()
        self.actors = []
        self.calls = []
        self.root = self.add_brain_region("root", alpha=0.3)

    def add(self, *a, **kw):
        self.actors.extend(a)

    def remove(self, *a):
        for x in a:
            if x in self.actors:
                self.actors.remove(x)

    def get_actors(self, name=None, br_class=None):
        return [a for a in self.actors if getattr(a, "name", None) == name]

    def add_brain_region(self, name, alpha=1, color=None, silhouette=None, **kw):
        self.calls.append(("region", name, alpha, silhouette))
        act = Actor(vedo.Sphere(r=3000, pos=(6600, 4000, 5700)), name=name, br_class="brain region")
        self.actors.append(act)
        return act

    def add_silhouette(self, *a, **kw):
        self.calls.append(("silhouette",))

    def add_label(self, *a, **kw):
        pass

    def slice(self, plane, actors=None, invert=False):
        self.calls.append(("slice", plane if isinstance(plane, str) else "plane"))

    def render(self, camera=None, interactive=False, zoom=None, **kw):
        self.plotter.show([a.mesh for a in self.actors if hasattr(a, "mesh")], interactive=False)

    def screenshot(self, name=None, scale=None, **kw):
        self.render()
        self.plotter.screenshot(name)
        self.calls.append(("shot", name))
        return name

    def close(self):
        self.calls.append(("close",))


@pytest.fixture
def fake(monkeypatch):
    holder = {}

    def make(title=None, atlas_name=None, camera=None):
        holder["scene"] = FakeScene()
        return holder["scene"]

    for mod in (probes, neurons, scene_mod):
        monkeypatch.setattr(mod, "make_scene", make)
    return holder


def test_render_probes_screenshots(fake, tmp_path):
    tracks = [np.array([[5000, 500, 5000], [5200, 4000, 5100.0]]), np.array([[4000, 500, 6500], [4100, 3500, 6600.0]])]
    paths = probes.render_probes(
        tracks, colors=["red", "blue"], legend_labels=["A", "B"], legend_colors=["red", "blue"],
        screenshots=True, out_dir=tmp_path / "pt", ss_label="t", views=["frontal", "sagittal", "top", "three_quarters"],
    )
    assert [p.endswith(f"t_{v}.png") for p, v in zip(paths, ["frontal", "sagittal", "top", "three_quarters"])] == [True] * 4
    assert all((tmp_path / "pt" / f"t_{v}.png").exists() for v in ["frontal", "sagittal", "top", "three_quarters"])
    calls = fake["scene"].calls
    assert ("slice", "horizontal") in calls and ("close",) in calls
    assert sum(c[0] == "slice" for c in calls) == 3  # sagittal(1) + top(2)
    # one root per view, alpha per view (three_quarters 0.5, others 0.05)
    root_alphas = [c[2] for c in calls if c[0] == "region" and c[1] == "root"][1:]
    assert root_alphas == [0.05, 0.05, 0.05, 0.5]


def test_render_neurons_colors_and_outside(fake, tmp_path):
    pts = np.array([[6000.0, 3000, 5000], [6100, 3100, 5200], [-5000, 3000, 5000]])  # last: outside volume
    out = neurons.render_neurons(
        pts, point_color=np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]), hide_outside=True,
        screenshots=True, out_dir=tmp_path, ss_label="n", views=["frontal"],
    )
    assert (tmp_path / "n_frontal.png").exists() and len(out) == 1
    with pytest.raises(ValueError):
        neurons.render_neurons(pts[:, :2], screenshots=True)
    with pytest.raises(ValueError):
        neurons.render_neurons(pts, point_color=np.ones((2, 3)), screenshots=True, out_dir=tmp_path)


def test_animate_neurons_pipeline(fake, tmp_path):
    try:
        anim.find_ffmpeg()
    except RuntimeError:
        pytest.skip("no ffmpeg")
    rng = np.random.default_rng(0)
    coords = np.column_stack([rng.uniform(4000, 9000, 20), rng.uniform(2000, 5000, 20), rng.uniform(3000, 8000, 20)])
    vals = rng.random((20, 8))
    vals[3, 2] = np.nan  # unit 3 gets dropped
    out = anim.animate_neurons(
        coords, vals, tmp_path / "v" / "a.mp4", duration=1.0, fps=6, show_timer=True, dark_mode=True,
        point_size=7, verbose=False, keep_frames_dir=tmp_path / "frames",
    )
    assert out.endswith("a.mp4") and (tmp_path / "v" / "a.mp4").stat().st_size > 0
    assert len(list((tmp_path / "frames").glob("*.png"))) == 6
