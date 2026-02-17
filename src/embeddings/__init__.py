"""
Módulo para geração de embeddings via Autoencoders e TruncatedSVD.

Exporta as principais funções para treino e extração de embeddings.
"""

from .autoencoder import BinaryAutoencoder
from .train_embeddings import (
    train_and_export_embeddings,
    load_embedding_cache
)
from .truncated_svd import (
    fit_transform_features,
    train_and_export_svd_embeddings
)
from .embedding_factory import (
    generate_embeddings,
    validate_embedding_method,
    get_embedding_info,
    list_available_methods
)

__all__ = [
    'BinaryAutoencoder',
    'train_and_export_embeddings',
    'load_embedding_cache',
    'fit_transform_features',
    'train_and_export_svd_embeddings',
    'generate_embeddings',
    'validate_embedding_method',
    'get_embedding_info',
    'list_available_methods'
]
