"""Run Dream Labs policies on MakerMods Metal and Maker arms."""

import os

# The IK solves many small matrices: BLAS worker threads gain nothing there and starve the servo and arm loops.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from .core.robot import Observation, Robot  # noqa: E402

__all__ = ["Robot", "Observation"]
