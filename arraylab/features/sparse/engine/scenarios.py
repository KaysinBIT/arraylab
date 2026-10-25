"""Discrete scan/frequency cases; coordinates are fixed in reference wavelengths."""
from dataclasses import replace
import numpy as np

from arraylab.shared.numerics.scenarios import LIGHT_MM_GHZ, scenarios




def case_config(c,case):
    """Angular sampling only; do not use this scaled config to move physical layout."""
    return replace(c,theta=case['theta'],phi=case['phi'],spacing=c.spacing*case['frequency_ratio'])
