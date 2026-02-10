"""
Módulo para geração de embeddings via Autoencoders.

Exporta as principais funções para treino e extração de embeddings.
"""

from .autoencoder import BinaryAutoencoder
from .train_embeddings import (
    train_and_export_embeddings,
    load_embedding_cache
)

__all__ = [
    'BinaryAutoencoder',
    'train_and_export_embeddings',
    'load_embedding_cache'
]
