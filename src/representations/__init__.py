"""
Módulo de representações de itens.

Provê interface unificada para carregar representações de itens (notícias),
sejam elas binárias ou embeddings densos.
"""

from .item_representation import (
    get_item_representation,
    prepare_item_vectors,
    ItemRepresentation
)

__all__ = [
    'get_item_representation',
    'prepare_item_vectors',
    'ItemRepresentation'
]
