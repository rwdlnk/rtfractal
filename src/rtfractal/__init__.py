"""rtfractal: fractal and mixing analysis of Rayleigh-Taylor interfaces.

Consolidates the analysis path of FractalParameterAI (analyze_rt.py,
analyze_rt_3d.py), FractalAnalyzer (2D box counting) and 3D-FractalParameterAI
(fractal3d) into one package, reproducing their results bit for bit.
"""
__version__ = "0.1.0"

from .io.case import open_case  # noqa: F401
from .physics import RTPhysics  # noqa: F401
