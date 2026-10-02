"""np_rendering: 3D mouse-brain rendering helpers for Neuropixels projects.

Importing this package needs only numpy + matplotlib. brainrender, vedo, iblatlas and other heavy
packages are imported lazily inside the functions that need them.
"""

__version__ = "0.1.0"

from .colors import categorical_colors, metric_to_rgba, minmax_per_unit  # noqa: F401
from .coords import (  # noqa: F401
    BREGMA_CCF_MLAPDV_UM,
    bregma_to_ccf,
    ccf_to_bregma,
    load_xyz_picks,
)
