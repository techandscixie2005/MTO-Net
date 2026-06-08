"""MTO-Net: Molecular Tensor Orbital Network on top of DetaNet."""

# Core modules always available
from .compat import *  # noqa: F401
from .valence import molecular_valence_electrons, VALENCE_ELECTRONS

# Lazy imports for optional deps
try:
    from .molecule_builder import make_ethanol, make_formaldehyde, make_water
except ImportError:
    pass

try:
    from .mto_module import ValenceAdaptiveMTO
except ImportError:
    pass

try:
    from .visualization import plot_mto_map, plot_mto_summary
except ImportError:
    pass

try:
    from .detanet_adapter import DetaNetBackboneAdapter
except ImportError:
    pass
