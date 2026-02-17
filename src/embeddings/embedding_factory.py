"""
Factory e roteador para geração de embeddings.

Suporta múltiplos métodos de geração de embeddings (autoencoder, TruncatedSVD, etc.)
com interface unificada.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

from ..config import (
    AUTOENCODER_CONFIG,
    SVD_CONFIG,
    EMBEDDING_METHODS,
    DEFAULT_EMBEDDING_METHOD
)


def validate_embedding_method(method: str) -> str:
    """
    Valida e normaliza o nome do método de embedding.
    
    Args:
        method: Nome do método ('autoencoder', 'truncated_svd', etc.)
    
    Returns:
        Nome normalizado do método
    
    Raises:
        ValueError: Se método não for suportado
    """
    method_lower = method.lower().strip()
    
    # Aliases
    aliases = {
        'ae': 'autoencoder',
        'svd': 'truncated_svd',
        'lsa': 'truncated_svd',
        'tsvd': 'truncated_svd'
    }
    
    method_normalized = aliases.get(method_lower, method_lower)
    
    if method_normalized not in EMBEDDING_METHODS:
        raise ValueError(
            f"Método de embedding não suportado: '{method}'\n"
            f"Métodos disponíveis: {EMBEDDING_METHODS}\n"
            f"Aliases: {list(aliases.keys())}"
        )
    
    return method_normalized


def generate_embeddings(
    method: str = DEFAULT_EMBEDDING_METHOD,
    data_dir: str = "outputs",
    embedding_dim: Optional[int] = None,
    seed: Optional[int] = None,
    force_retrain: bool = False,
    verbose: bool = True,
    **kwargs
) -> Tuple[Path, Path]:
    """
    Gera embeddings usando o método especificado.
    
    Interface unificada para geração de embeddings com diferentes métodos.
    Cada método pode ter seus próprios parâmetros específicos passados via kwargs.
    
    Args:
        method: Método de embedding ('autoencoder', 'truncated_svd')
        data_dir: Diretório com canonical_features.parquet e canonical_topics.parquet
        embedding_dim: Dimensão dos embeddings (n_components para SVD)
        seed: Seed para reprodutibilidade (random_state para SVD)
        force_retrain: Se True, força retreinamento (apenas autoencoder)
        verbose: Se True, imprime progresso
        **kwargs: Parâmetros específicos do método
    
    Returns:
        Tupla (features_path, topics_path) com caminhos dos embeddings gerados
    
    Raises:
        ValueError: Se método não for válido ou parâmetros inválidos
        FileNotFoundError: Se dados canônicos não existirem
    
    Examples:
        >>> # Autoencoder
        >>> generate_embeddings('autoencoder', embedding_dim=32, seed=42, epochs=100)
        
        >>> # TruncatedSVD
        >>> generate_embeddings('truncated_svd', embedding_dim=32, seed=42)
        
        >>> # Com parâmetros específicos
        >>> generate_embeddings('truncated_svd', embedding_dim=64, n_iter=10)
    """
    method = validate_embedding_method(method)
    
    if verbose:
        print("\n" + "█"*80)
        print(f" "*25 + f"EMBEDDING FACTORY")
        print("█"*80)
        print(f"\nMétodo selecionado: {method}")
        print(f"Diretório de dados: {data_dir}")
        print(f"Dimensão: {embedding_dim or 'padrão'}")
        print(f"Seed: {seed or 'padrão'}")
    
    # Roteamento por método
    if method == 'autoencoder':
        return _generate_autoencoder_embeddings(
            data_dir=data_dir,
            embedding_dim=embedding_dim,
            seed=seed,
            force_retrain=force_retrain,
            verbose=verbose,
            **kwargs
        )
    
    elif method == 'truncated_svd':
        return _generate_svd_embeddings(
            data_dir=data_dir,
            n_components=embedding_dim,
            random_state=seed,
            verbose=verbose,
            **kwargs
        )
    
    else:
        # Não deveria chegar aqui devido à validação
        raise ValueError(f"Método não implementado: {method}")


def _generate_autoencoder_embeddings(
    data_dir: str,
    embedding_dim: Optional[int] = None,
    seed: Optional[int] = None,
    force_retrain: bool = False,
    verbose: bool = True,
    **kwargs
) -> Tuple[Path, Path]:
    """Gera embeddings usando autoencoder."""
    from .train_embeddings import train_and_export_embeddings
    
    # Usar configuração default se não especificado
    cfg = AUTOENCODER_CONFIG
    embedding_dim = embedding_dim if embedding_dim is not None else cfg['embedding_dim']
    seed = seed if seed is not None else cfg['seed']
    
    # Parâmetros adicionais via kwargs ou config
    params = {
        'hidden_dim': kwargs.get('hidden_dim', cfg.get('hidden_dim')),
        'dropout_rate': kwargs.get('dropout_rate', cfg.get('dropout_rate')),
        'epochs': kwargs.get('epochs', cfg.get('epochs')),
        'batch_size': kwargs.get('batch_size', cfg.get('batch_size')),
        'learning_rate': kwargs.get('learning_rate', cfg.get('learning_rate')),
        'pos_weight_mode': kwargs.get('pos_weight_mode', cfg.get('pos_weight_mode')),
        'denoising_prob': kwargs.get('denoising_prob', cfg.get('denoising_prob')),
        'weight_decay': kwargs.get('weight_decay', cfg.get('weight_decay')),
        'early_stopping_patience': kwargs.get('early_stopping_patience', cfg.get('early_stopping_patience')),
        'l2_normalize': kwargs.get('l2_normalize', cfg.get('l2_normalize', True)),
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print("CONFIGURAÇÃO DO AUTOENCODER")
        print(f"{'='*70}")
        print(f"  - embedding_dim: {embedding_dim}")
        print(f"  - seed: {seed}")
        print(f"  - epochs: {params['epochs']}")
        print(f"  - batch_size: {params['batch_size']}")
        print(f"  - force_retrain: {force_retrain}")
    
    return train_and_export_embeddings(
        data_dir=data_dir,
        embedding_dim=embedding_dim,
        seed=seed,
        force_retrain=force_retrain,
        verbose=verbose,
        **params
    )


def _generate_svd_embeddings(
    data_dir: str,
    n_components: Optional[int] = None,
    random_state: Optional[int] = None,
    verbose: bool = True,
    **kwargs
) -> Tuple[Path, Path]:
    """Gera embeddings usando TruncatedSVD."""
    from .truncated_svd import train_and_export_svd_embeddings
    
    # Usar configuração default se não especificado
    cfg = SVD_CONFIG
    n_components = n_components if n_components is not None else cfg['n_components']
    random_state = random_state if random_state is not None else cfg['random_state']
    
    # Parâmetros adicionais via kwargs ou config
    params = {
        'n_iter': kwargs.get('n_iter', cfg.get('n_iter', 5)),
        'continuous_cols': kwargs.get('continuous_cols', cfg.get('continuous_cols')),
        'normalize_l2': kwargs.get('normalize_l2', cfg.get('normalize_l2', True)),
    }
    
    # Validar que colunas contínuas existem
    data_path = Path(data_dir)
    features_path = data_path / 'canonical_features.parquet'
    
    if features_path.exists() and params['continuous_cols']:
        df_sample = pd.read_parquet(features_path)
        available_cols = [col for col in params['continuous_cols'] if col in df_sample.columns]
        
        if len(available_cols) < len(params['continuous_cols']):
            missing = set(params['continuous_cols']) - set(available_cols)
            if verbose:
                print(f"\n[AVISO] Colunas contínuas não encontradas: {missing}")
                print(f"        Usando apenas: {available_cols}")
            params['continuous_cols'] = available_cols if available_cols else None
    
    if verbose:
        print(f"\n{'='*70}")
        print("CONFIGURAÇÃO DO TRUNCATED SVD")
        print(f"{'='*70}")
        print(f"  - n_components: {n_components}")
        print(f"  - random_state: {random_state}")
        print(f"  - n_iter: {params['n_iter']}")
        print(f"  - continuous_cols: {params['continuous_cols']}")
    
    return train_and_export_svd_embeddings(
        data_dir=data_dir,
        n_components=n_components,
        random_state=random_state,
        verbose=verbose,
        **params
    )


def get_embedding_info(method: str) -> Dict[str, Any]:
    """
    Retorna informações sobre um método de embedding.
    
    Args:
        method: Nome do método
    
    Returns:
        Dicionário com informações do método
    """
    method = validate_embedding_method(method)
    
    info = {
        'autoencoder': {
            'name': 'Autoencoder (Neural Network)',
            'type': 'non-linear',
            'deterministic': False,
            'training_time': 'slow (~minutes)',
            'config_key': 'AUTOENCODER_CONFIG',
            'default_dim': AUTOENCODER_CONFIG['embedding_dim'],
            'description': 'Rede neural com bottleneck para compressão não-linear'
        },
        'truncated_svd': {
            'name': 'TruncatedSVD (Latent Semantic Analysis)',
            'type': 'linear',
            'deterministic': True,
            'training_time': 'fast (~seconds)',
            'config_key': 'SVD_CONFIG',
            'default_dim': SVD_CONFIG['n_components'],
            'description': 'Decomposição SVD para redução linear de dimensionalidade'
        }
    }
    
    return info[method]


def list_available_methods(verbose: bool = True) -> List[str]:
    """
    Lista métodos de embedding disponíveis.
    
    Args:
        verbose: Se True, imprime informações detalhadas
    
    Returns:
        Lista de nomes de métodos
    """
    if verbose:
        print("\n" + "="*70)
        print("MÉTODOS DE EMBEDDING DISPONÍVEIS")
        print("="*70)
        
        for method in EMBEDDING_METHODS:
            info = get_embedding_info(method)
            print(f"\n{method}:")
            print(f"  - Nome: {info['name']}")
            print(f"  - Tipo: {info['type']}")
            print(f"  - Determinístico: {info['deterministic']}")
            print(f"  - Tempo: {info['training_time']}")
            print(f"  - Dimensão padrão: {info['default_dim']}")
            print(f"  - Descrição: {info['description']}")
    
    return EMBEDDING_METHODS


if __name__ == '__main__':
    """CLI para geração de embeddings com qualquer método."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Geração unificada de embeddings (Autoencoder, TruncatedSVD, etc.)"
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default=DEFAULT_EMBEDDING_METHOD,
        choices=EMBEDDING_METHODS + ['ae', 'svd', 'lsa', 'tsvd'],  # Incluir aliases
        help=f"Método de embedding (default: {DEFAULT_EMBEDDING_METHOD})"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='outputs',
        help='Diretório com canonical_features.parquet e canonical_topics.parquet'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=None,
        help='Dimensão dos embeddings (se None, usa padrão do método)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed para reprodutibilidade (se None, usa padrão do método)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Força retreinamento (apenas autoencoder)'
    )
    parser.add_argument(
        '--list-methods',
        action='store_true',
        help='Lista métodos disponíveis e sai'
    )
    
    # Parâmetros específicos do autoencoder
    ae_group = parser.add_argument_group('Autoencoder')
    ae_group.add_argument('--ae-epochs', type=int, help='Número de épocas')
    ae_group.add_argument('--ae-batch-size', type=int, help='Tamanho do batch')
    ae_group.add_argument('--ae-learning-rate', type=float, help='Taxa de aprendizado')
    
    # Parâmetros específicos do SVD
    svd_group = parser.add_argument_group('TruncatedSVD')
    svd_group.add_argument('--svd-n-iter', type=int, help='Número de iterações do SVD')
    svd_group.add_argument('--svd-no-continuous', action='store_true', 
                          help='Desabilita concatenação de features contínuas')
    
    args = parser.parse_args()
    
    # Listar métodos e sair
    if args.list_methods:
        list_available_methods(verbose=True)
        exit(0)
    
    # Preparar kwargs específicos do método
    kwargs = {}
    
    if args.method in ['autoencoder', 'ae']:
        if args.ae_epochs:
            kwargs['epochs'] = args.ae_epochs
        if args.ae_batch_size:
            kwargs['batch_size'] = args.ae_batch_size
        if args.ae_learning_rate:
            kwargs['learning_rate'] = args.ae_learning_rate
    
    elif args.method in ['truncated_svd', 'svd', 'lsa', 'tsvd']:
        if args.svd_n_iter:
            kwargs['n_iter'] = args.svd_n_iter
        if args.svd_no_continuous:
            kwargs['continuous_cols'] = None
    
    # Gerar embeddings
    features_path, topics_path = generate_embeddings(
        method=args.method,
        data_dir=args.data_dir,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
        force_retrain=args.force,
        verbose=True,
        **kwargs
    )
    
    print("\n" + "="*80)
    print("✅ EMBEDDINGS GERADOS COM SUCESSO")
    print("="*80)
    print(f"\nArquivos:")
    print(f"  - Features: {features_path.name}")
    print(f"  - Topics: {topics_path.name}")
