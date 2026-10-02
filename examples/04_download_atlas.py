"""Export the Allen 'root' brain mesh to .obj (optional; brainrender itself does not need it).

    python examples/04_download_atlas.py --out allen_root.obj     # needs a BrainGlobe atlas download (~network)
"""

import argparse

from np_rendering.atlas_download import download_atlas

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="allen_root.obj")
a = ap.parse_args()
print("mesh ->", download_atlas(a.out))
