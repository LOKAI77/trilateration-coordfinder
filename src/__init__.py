"""
Trilateration Coordinate Finder
A high-precision geospatial trilateration solver using multiple optimization methods.
"""

__version__ = "1.0.0"
__author__ = "LOKAI77"

from .solver import (
    trilateration_objective,
    multi_stage_optimization,
    alternative_optimization,
    geometric_approach,
    refine_position,
    verify_solution,
    vincenty_distance
)

from .cli import main

__all__ = [
    'trilateration_objective',
    'multi_stage_optimization',
    'alternative_optimization',
    'geometric_approach',
    'refine_position',
    'verify_solution',
    'vincenty_distance',
    'main'
]
