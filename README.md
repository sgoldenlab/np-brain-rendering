# np-brain-rendering (`np_rendering`)

3D mouse-brain figures and videos for Neuropixels projects, built on brainrender/vedo: probe tracks,
units coloured by any metric, rotating activity animations, plus an atlas-mesh export helper.
Everything takes explicit paths and arguments; nothing is tied to lab drives or project names.
Repo: `https://github.com/sgoldenlab/np-brain-rendering`. See the lab protocol docs for how xyz picks and unit positions are produced upstream.

## Install

```bash
uv venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[all]"                    # or pick: .[atlas] .[render] .[video]
```
`import np_rendering` needs only numpy + matplotlib; heavy packages are imported inside functions.
Videos need `ffmpeg` on PATH (or `.[video]`, which provides a binary). First render downloads the BrainGlobe
atlas `allen_mouse_25um`. Headless Linux: run scripts as `xvfb-run -a python ...`.

## Pipeline

```
xyz_picks JSON (bregma-relative um, ML/AP/DV)  --coords.load_xyz_picks / track_endpoints_ccf-->  CCF tracks --probes.render_probes--> PNGs
unit table (bregma-relative um + metric)       --coords.bregma_to_ccf, colors.metric_to_rgba--> neurons.render_neurons --> PNGs
units x time-bin matrix                        --colors.minmax_per_unit (optional)--> animation.animate_neurons --> mp4
```
Render functions expect **brainrender CCF um in (AP, DV, ML) order**. Convention table: `src/np_rendering/coords.py`
(bregma = [ML 5739, AP 5400, DV 332] um in the CCF; bregma (0,0,0) -> CCF (AP,DV,ML) = (5400, 332, 5739); anterior = smaller CCF-AP, ventral = larger CCF-DV).

## Modules

| Function | Goal | Key parameters (default; change for a new project) |
|---|---|---|
| `coords.bregma_to_ccf(xyz_um, ...)` | bregma-relative um -> Allen CCF um via iblatlas `xyz2ccf` (no atlas download) | `input_order='mlapdv'` (match your table), `ccf_order='apdvml'`, `res_um=25`, `mode='raise'` (`'clip'` for noisy unit positions) |
| `coords.ccf_to_bregma` | inverse | `ccf_order`, `res_um` |
| `coords.load_xyz_picks`, `track_endpoints_ccf` | read `{"xyz_picks": [[x,y,z],...]}`; top/bottom pick as CCF | `mode` |
| `coords.unit_position_from_deltas` | shift a unit off its peak channel along probe axes | `channel_data` `{ch: {'local': [axial, lateral], 'ccf': [x,y,z]}}` per recording |
| `colors.metric_to_rgba` | colour units by a metric | `cmap='viridis'`, `vmin/vmax=None` (data range), `percentile`, `alpha`, `nan_color` |
| `colors.minmax_per_unit` | scale each unit's time series to [0,1] | - |
| `colors.categorical_colors` | stable animal/recording colours | `palette` (plotly G10) |
| `probes.render_probes` | cylinders (r=25 um) per track, legend, optional region meshes | `colors`, `regions`, `views=('frontal','sagittal')`, `screenshots`, `out_dir`, `shader_style='plastic'`, `root_alpha` |
| `neurons.identify_outside_brain` | flag units outside the brain (annotation 0 / outside volume; `root` = inside); needs `atlas` or a raw `annotation` array | `res_um=25` |
| `neurons.render_neurons` | points with per-unit RGBA; hide/recolour outside units; optional regions | `point_color`, `point_size=50`, `hide_outside`, `coord_system='ccf'`, `include_regions`, `views`, `root_alpha`, `shader_style='cartoon'` |
| `animation.animate_neurons` | per-bin colours + rotating camera -> mp4 (libx264, yuv420p) | `duration=10` or `bins_per_second`, `fps=30`, `cmap='hot'`, `vmin/vmax=None` (5th/95th pct), `rotate_degrees=180`, `dark_mode`, `crf=15`, `video_size=None` |
| `animation.frames_to_video` | PNG frames -> mp4 | `fps`, `crf`, `size` |
| `scene.render_settings / save_views / get_camera` | styling (cartoon shader, LW 2, root alpha 0.3, offscreen, vtk) applied only while rendering; multi-view screenshots | `frontal_pos` (lab's custom frontal camera) |
| `atlas_download.download_atlas` / `np-render-download-atlas` | export Allen `root` mesh to .obj (not needed by brainrender itself) | `output_path`, `structure` |

## Quick start

```python
import numpy as np
from np_rendering.coords import bregma_to_ccf
from np_rendering.colors import metric_to_rgba
from np_rendering.neurons import render_neurons

xyz = np.array([[1000, -1500, -2500], [-800, 500, -3000.0]])        # um from bregma, ML/AP/DV
ccf = bregma_to_ccf(xyz, mode="clip")                                 # -> (AP, DV, ML) CCF um
render_neurons(ccf, point_color=metric_to_rgba([0.2, 0.9], "magma"),
               screenshots=True, out_dir="plots/units", views=["frontal"])
```
Examples (all generate their own synthetic data; `--no-render` checks data prep only):
`examples/01_render_probe_tracks.py`, `02_render_units_by_metric.py`, `03_animate_units.py`, `04_download_atlas.py`.
Screenshots are saved as `<out_dir>/<label>_<view>.png` (views: frontal, sagittal, top, three_quarters; scale 1.5).


## Credits & citations

Code adapted from the Golden Lab rendering scripts built on these packages; please cite them:

- **brainrender**: Claudi F, Tyson AL, Petrucco L, Margrie TW, Portugues R, Branco T (2021). Visualizing anatomically registered data with brainrender. *eLife* 10:e65751. https://doi.org/10.7554/eLife.65751 (preferred citation per the brainrender README).
- **BrainGlobe Atlas API**: Claudi F, Petrucco L, Tyson AL, Branco T, Margrie TW, Portugues R (2020). BrainGlobe Atlas API: a common interface for neuroanatomical atlases. *JOSS* 5(54):2668. https://doi.org/10.21105/joss.02668 (title, venue, DOI verified; full author order not shown in the README <!-- TODO verify author order -->). Also cite the specific atlas used.
- **vedo**: Musy M et al. (2021). vedo, a python module for scientific analysis and visualization of 3D objects and point clouds. Zenodo. https://doi.org/10.5281/zenodo.7019968 (per vedo README).
- **iblatlas** (International Brain Laboratory, MIT): cite as in its `CITATION.cff` ("iblatlas: IBL Atlas Module", International Brain Laboratory; DOI listed there was still a placeholder when checked). Also IBL et al. for the atlas/coordinate conventions.
- **Allen CCFv3**: Wang Q et al. (2020). The Allen Mouse Brain Common Coordinate Framework: a 3D reference atlas. *Cell* 181(4):936-953. https://doi.org/10.1016/j.cell.2020.04.007
