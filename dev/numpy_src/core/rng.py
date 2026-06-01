"""Shared random number generator used throughout SimTorch.

The module exposes a single NumPy random generator instance so stochastic operations can
be coordinated and seeded consistently across the framework.
"""
from numpy.random import default_rng
rng = default_rng()