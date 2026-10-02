"""Download the Allen CCF 25 um whole-brain ('root') mesh via BrainGlobe and save it as .obj.

brainrender does NOT need this file (it fetches its own atlas); it is useful for opening the brain
outline in other tools (napari, Blender, Lasagna ...).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def download_atlas(output_path="allen_root.obj", atlas_name: str = "allen_mouse_25um", structure: str = "root") -> str:
    """Fetch ``atlas_name`` (downloads on first use) and write the mesh of ``structure`` to ``output_path``."""
    from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

    atlas = BrainGlobeAtlas(atlas_name)
    mesh = atlas.mesh_from_structure(structure)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.write(str(out))
    return str(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("-o", "--output", default="allen_root.obj")
    p.add_argument("--atlas", default="allen_mouse_25um")
    p.add_argument("--structure", default="root", help="acronym of the structure mesh to export")
    a = p.parse_args(argv)
    print("wrote", download_atlas(a.output, a.atlas, a.structure))


if __name__ == "__main__":
    main()
