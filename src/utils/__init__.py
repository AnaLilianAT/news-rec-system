"""
Módulo de utilitários compartilhados.
"""

from .dim_grid import (
    build_dims,
    validate_dims_list,
    compute_d_min_heuristic
)

__all__ = [
    'build_dims',
    'validate_dims_list',
    'compute_d_min_heuristic'
]
