"""
Script para treinar autoencoders e gerar embeddings.

Carrega matrizes binárias (features e tópicos), treina AEs separados,
e salva embeddings com cache + metadados para reuso.
"""

import argparse
import json
import hashlib
import random
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from .autoencoder import train_autoencoder, extract_embeddings, extract_embeddings_with_continuous
from .cache_utils import (
    make_embedding_cache_key,
    get_embedding_paths,
    build_cache_metadata,
    validate_cache_metadata,
    compute_item_ids_hash
)
from ..representations.item_representation import get_item_representation
from ..config import AUTOENCODER_CONFIG


def set_seeds(seed: int):
    """
    Fixa seeds para reprodutibilidade completa.
    
    Args:
        seed: Seed para random, numpy e torch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Garante determinismo em CUDA (pode impactar performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_data_hash(X: np.ndarray) -> str:
    """
    Calcula hash MD5 dos dados para detectar mudanças.
    
    Args:
        X: Matriz numpy
    
    Returns:
        String com hash hexadecimal
    """
    data_bytes = X.tobytes()
    return hashlib.md5(data_bytes).hexdigest()


def get_ae_config_dict(
    hidden_dim: Optional[int],
    dropout_rate: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pos_weight_mode: str = 'auto',
    denoising_prob: float = 0.0,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 0,
    pos_weight_clip: list = None,
    include_continuous: bool = False,
    continuous_cols: list = None,
    concat_continuous_after: bool = True,
    l2_normalize: bool = True
) -> Dict[str, Any]:
    """
    Extrai apenas a configuração do AE (sem d e seed que são explícitos).
    
    Returns:
        Dicionário com config do autoencoder
    """
    return {
        'hidden_dim': hidden_dim,
        'dropout_rate': dropout_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'pos_weight_mode': pos_weight_mode,
        'denoising_prob': denoising_prob,
        'weight_decay': weight_decay,
        'early_stopping_patience': early_stopping_patience,
        'pos_weight_clip': pos_weight_clip or [1.0, 10.0],
        'include_continuous': include_continuous,
        'continuous_cols': continuous_cols or ['polaridade', 'subjetividade'],
        'concat_continuous_after': concat_continuous_after,
        'l2_normalize': l2_normalize
    }


# Função check_cache_valid removida - usando validate_cache_metadata do cache_utils


def save_embeddings_with_metadata(
    embeddings: np.ndarray,
    item_ids: np.ndarray,
    metadata: Dict[str, Any],
    output_path: Path,
    json_path: Path,
    verbose: bool = True
):
    """
    Salva embeddings em parquet + metadados em JSON.
    
    Args:
        embeddings: Matriz (n_items, embedding_dim)
        item_ids: Array com news_id
        metadata: Dicionário de metadados (já construído com build_cache_metadata)
        output_path: Caminho para parquet
        json_path: Caminho para JSON
    """
    # Criar DataFrame com news_id + embeddings
    n_dims = embeddings.shape[1]
    cols = ['news_id'] + [f'emb_{i}' for i in range(n_dims)]
    
    df_emb = pd.DataFrame(
        data=np.column_stack([item_ids, embeddings]),
        columns=cols
    )
    df_emb['news_id'] = df_emb['news_id'].astype(int)
    
    # Salvar parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_emb.to_parquet(output_path, index=False)
    print(f"Embeddings salvos: {output_path}")
    print(f"  - Shape: {embeddings.shape}")
    print(f"  - Tamanho: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"  - Seed: {metadata.get('seed')}")
    print(f"  - Cache key: {metadata.get('cache_key')}")
    
    # Salvar metadados (com timestamp)
    import datetime
    metadata['export_timestamp'] = datetime.datetime.now().isoformat()
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadados salvos: {json_path}")
    
    # Log de verificação L2
    if verbose and embeddings.shape[0] > 0:
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"  L2 norms: mean={norms.mean():.4f}, std={norms.std():.4f}, min={norms.min():.4f}, max={norms.max():.4f}")


def load_embedding_cache(
    cache_path: Path,
    json_path: Path,
    verbose: bool = True
) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Carrega embeddings do cache se existirem.
    
    Args:
        cache_path: Caminho para parquet de embeddings
        json_path: Caminho para JSON de metadados
        verbose: Se True, imprime mensagens
    
    Returns:
        Tupla (DataFrame, metadados) ou None se cache não existir
    """
    if not cache_path.exists() or not json_path.exists():
        return None
    
    try:
        df = pd.read_parquet(cache_path)
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        if verbose:
            print(f"Cache carregado: {cache_path}")
            print(f"  - Shape: ({len(df)}, {len(df.columns) - 1})")
        
        return df, metadata
    
    except Exception as e:
        if verbose:
            print(f"Erro ao carregar cache: {e}")
        return None


def train_and_export_embeddings(
    data_dir: str = "outputs",
    embedding_dim: int = None,
    hidden_dim: Optional[int] = None,
    dropout_rate: float = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    seed: int = None,
    pos_weight_mode: str = None,
    denoising_prob: float = None,
    weight_decay: float = None,
    early_stopping_patience: int = None,
    pos_weight_clip: list = None,
    include_continuous: bool = None,
    continuous_cols: list = None,
    concat_continuous_after: bool = None,
    l2_normalize: bool = None,
    force_retrain: bool = False,
    verbose: bool = True
) -> Tuple[Path, Path]:
    """
    Treina autoencoders para features e tópicos, gerando embeddings.
    
    Implementa cache: se embeddings com mesma config já existem, reutiliza.
    Usa configuração centralizada de config.py quando parâmetros são None.
    
    Args:
        data_dir: Diretório com canonical_features.parquet e canonical_topics.parquet
        embedding_dim: Dimensão dos embeddings (None = usar config.py)
        hidden_dim: Dimensão da camada oculta (None = auto ou config.py)
        dropout_rate: Taxa de dropout (None = usar config.py)
        epochs: Número de épocas (None = usar config.py)
        batch_size: Tamanho do batch (None = usar config.py)
        learning_rate: Taxa de aprendizado (None = usar config.py)
        seed: Seed para reprodutibilidade (None = usar config.py)
        pos_weight_mode: Modo de peso positivo (None = usar config.py)
        denoising_prob: Probabilidade de denoising (None = usar config.py)
        force_retrain: Se True, ignora cache e retreina
        verbose: Se True, imprime progresso
    
    Returns:
        Tupla com (path_features, path_topics) dos embeddings gerados
    """
    # Usar configuração centralizada como padrão
    cfg = AUTOENCODER_CONFIG
    
    embedding_dim = embedding_dim if embedding_dim is not None else cfg['embedding_dim']
    hidden_dim = hidden_dim if hidden_dim is not None else cfg['hidden_dim']
    dropout_rate = dropout_rate if dropout_rate is not None else cfg['dropout_rate']
    epochs = epochs if epochs is not None else cfg['epochs']
    batch_size = batch_size if batch_size is not None else cfg['batch_size']
    learning_rate = learning_rate if learning_rate is not None else cfg['learning_rate']
    seed = seed if seed is not None else cfg['seed']
    pos_weight_mode = pos_weight_mode if pos_weight_mode is not None else cfg['pos_weight_mode']
    denoising_prob = denoising_prob if denoising_prob is not None else cfg['denoising_prob']
    weight_decay = weight_decay if weight_decay is not None else cfg.get('weight_decay', 1e-5)
    early_stopping_patience = early_stopping_patience if early_stopping_patience is not None else cfg.get('early_stopping_patience', 0)
    pos_weight_clip = pos_weight_clip if pos_weight_clip is not None else cfg.get('pos_weight_clip', [1.0, 10.0])
    include_continuous = include_continuous if include_continuous is not None else cfg.get('include_continuous', False)
    continuous_cols = continuous_cols if continuous_cols is not None else cfg.get('continuous_cols', ['polaridade', 'subjetividade'])
    concat_continuous_after = concat_continuous_after if concat_continuous_after is not None else cfg.get('concat_continuous_after', True)
    l2_normalize = l2_normalize if l2_normalize is not None else cfg.get('l2_normalize', True)
    
    # Fixar seeds para reprodutibilidade
    set_seeds(seed)
    
    if verbose:
        print("\n" + "="*60)
        print("CONFIGURAÇÃO DO AUTOENCODER")
        print("="*60)
        print(f"Embedding dim: {embedding_dim}")
        print(f"Hidden dim: {hidden_dim if hidden_dim else 'auto'}")
        print(f"Dropout rate: {dropout_rate}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Seed: {seed}")
        print(f"Pos weight mode: {pos_weight_mode}")
        print(f"Pos weight clip: {pos_weight_clip}")
        print(f"Denoising prob: {denoising_prob}")
        print(f"Include continuous: {include_continuous}")
        if include_continuous:
            print(f"Continuous cols: {continuous_cols}")
            print(f"Concat after: {concat_continuous_after}")
        print(f"L2 normalize: {l2_normalize}")
        print(f"Cache: {'Desabilitado (--force)' if force_retrain else 'Habilitado'}")
    
    data_path = Path(data_dir)
    
    # Construir dicionário de config do AE (sem d e seed)
    ae_config = get_ae_config_dict(
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pos_weight_mode=pos_weight_mode,
        denoising_prob=denoising_prob,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        pos_weight_clip=pos_weight_clip,
        include_continuous=include_continuous,
        continuous_cols=continuous_cols,
        concat_continuous_after=concat_continuous_after,
        l2_normalize=l2_normalize
    )
    
    # ========================
    # FEATURES
    # ========================
    if verbose:
        print("\n" + "="*60)
        print("TREINANDO AUTOENCODER PARA FEATURES")
        print("="*60)
    
    # Carregar dados binários usando loader existente
    repr_features = get_item_representation(
        kind='bin_features',
        output_dir=str(data_path)
    )
    X_features = repr_features.matrix[repr_features.feature_names].values
    item_ids_features = repr_features.matrix['news_id'].values
    
    # Calcular hashes
    data_hash_features = compute_data_hash(X_features)
    item_ids_hash_features = compute_item_ids_hash(item_ids_features)
    
    # Gerar cache key e paths
    cache_key_features = make_embedding_cache_key(
        d=embedding_dim,
        seed=seed,
        ae_config=ae_config,
        datahash=data_hash_features
    )
    features_output, features_json = get_embedding_paths(
        base_dir=data_path,
        representation_type='ae_features',
        cache_key=cache_key_features
    )
    
    # Verificar cache
    cache_valid_features = False
    if features_json.exists():
        try:
            with open(features_json, 'r') as f:
                cached_metadata = json.load(f)
            cache_valid_features = validate_cache_metadata(
                cached_metadata,
                expected_d=embedding_dim,
                expected_seed=seed,
                expected_ae_config=ae_config,
                expected_datahash=data_hash_features
            )
        except Exception:
            cache_valid_features = False
    
    if cache_valid_features and not force_retrain:
        if verbose:
            print("\n[OK] Cache válido encontrado. Reutilizando embeddings existentes.")
            print(f"  Arquivo: {features_output}")
            print(f"  Cache key: {cache_key_features}")
    else:
        if force_retrain and verbose:
            print("\n[INFO] --force especificado. Retreinando modelo...")
        elif verbose:
            print("\n[INFO] Cache inválido ou inexistente. Treinando novo modelo...")
            print(f"  Cache key: {cache_key_features}")
        
        # Treinar autoencoder
        model_features, train_metadata_features = train_autoencoder(
            X=X_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            pos_weight_mode=pos_weight_mode,
            pos_weight_clip=pos_weight_clip,
            denoising_prob=denoising_prob,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            val_split=0.2,  # Split treino/val padrão
            seed=seed,
            verbose=verbose
        )
        
        # Extrair embeddings (com L2 normalization baseado na config)
        # Se concat_continuous_after=True, concatenar polaridade/subjetividade
        if concat_continuous_after and not include_continuous:
            if verbose:
                print(f"\n[INFO] Concatenando features contínuas após embedding...")
            
            # Carregar features contínuas do dataset completo
            # Assumindo que repr_features.matrix contém todas as colunas
            X_continuous = None
            if all(col in repr_features.matrix.columns for col in continuous_cols):
                X_continuous = repr_features.matrix[continuous_cols].values
                if verbose:
                    print(f"  Colunas contínuas: {continuous_cols}")
                    print(f"  Shape contínuas: {X_continuous.shape}")
            else:
                if verbose:
                    print(f"  [WARNING] Colunas {continuous_cols} não encontradas. Continuando sem concatenação.")
            
            Z_features = extract_embeddings_with_continuous(
                model=model_features,
                X_binary=X_features,
                X_continuous=X_continuous,
                normalize_l2=l2_normalize,
                concat_continuous_after=True
            )
            
            if verbose and X_continuous is not None:
                print(f"  Shape final (AE + continuous): {Z_features.shape}")
        else:
            # Extração padrão sem concatenação
            Z_features = extract_embeddings(
                model=model_features,
                X=X_features,
                normalize_l2=l2_normalize
            )
        
        # Construir metadados completos (incluindo metadados de treino)
        metadata_features = build_cache_metadata(
            d=embedding_dim,
            seed=seed,
            ae_config=ae_config,
            datahash=data_hash_features,
            representation_type='ae_features',
            cache_key=cache_key_features,
            shape=Z_features.shape,
            item_ids_hash=item_ids_hash_features
        )
        # Adicionar metadados de treino
        metadata_features['training'] = train_metadata_features
        
        # Adicionar info sobre features contínuas
        metadata_features['continuous_features'] = {
            'included_in_ae': include_continuous,
            'concatenated_after': concat_continuous_after and not include_continuous,
            'columns': continuous_cols if (concat_continuous_after and not include_continuous) else [],
            'final_dim': Z_features.shape[1]
        }
        
        # Salvar com metadados
        save_embeddings_with_metadata(
            embeddings=Z_features,
            item_ids=item_ids_features,
            metadata=metadata_features,
            output_path=features_output,
            json_path=features_json,
            verbose=verbose
        )
    
    # ========================
    # TÓPICOS
    # ========================
    if verbose:
        print("\n" + "="*60)
        print("TREINANDO AUTOENCODER PARA TÓPICOS")
        print("="*60)
    
    # Carregar dados binários usando loader existente
    repr_topics = get_item_representation(
        kind='bin_topics',
        output_dir=str(data_path)
    )
    X_topics = repr_topics.matrix[repr_topics.feature_names].values
    item_ids_topics = repr_topics.matrix['news_id'].values
    
    # Calcular hashes
    data_hash_topics = compute_data_hash(X_topics)
    item_ids_hash_topics = compute_item_ids_hash(item_ids_topics)
    
    # Gerar cache key e paths
    cache_key_topics = make_embedding_cache_key(
        d=embedding_dim,
        seed=seed,
        ae_config=ae_config,
        datahash=data_hash_topics
    )
    topics_output, topics_json = get_embedding_paths(
        base_dir=data_path,
        representation_type='ae_topics',
        cache_key=cache_key_topics
    )
    
    # Verificar cache
    cache_valid_topics = False
    if topics_json.exists():
        try:
            with open(topics_json, 'r') as f:
                cached_metadata = json.load(f)
            cache_valid_topics = validate_cache_metadata(
                cached_metadata,
                expected_d=embedding_dim,
                expected_seed=seed,
                expected_ae_config=ae_config,
                expected_datahash=data_hash_topics
            )
        except Exception:
            cache_valid_topics = False
    
    if cache_valid_topics and not force_retrain:
        if verbose:
            print("\n[OK] Cache válido encontrado. Reutilizando embeddings existentes.")
            print(f"  Arquivo: {topics_output}")
            print(f"  Cache key: {cache_key_topics}")
    else:
        if force_retrain and verbose:
            print("\n[INFO] --force especificado. Retreinando modelo...")
        elif verbose:
            print("\n[INFO] Cache inválido ou inexistente. Treinando novo modelo...")
            print(f"  Cache key: {cache_key_topics}")
        
        # Treinar autoencoder
        model_topics, train_metadata_topics = train_autoencoder(
            X=X_topics,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            pos_weight_mode=pos_weight_mode,
            pos_weight_clip=pos_weight_clip,
            denoising_prob=denoising_prob,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            val_split=0.2,  # Split treino/val padrão
            seed=seed,
            verbose=verbose
        )
        
        # Extrair embeddings (com L2 normalization baseado na config)
        # Tópicos são sempre binários, sem features contínuas
        Z_topics = extract_embeddings(
            model=model_topics,
            X=X_topics,
            normalize_l2=l2_normalize
        )
        
        # Construir metadados completos (incluindo metadados de treino)
        metadata_topics = build_cache_metadata(
            d=embedding_dim,
            seed=seed,
            ae_config=ae_config,
            datahash=data_hash_topics,
            representation_type='ae_topics',
            cache_key=cache_key_topics,
            shape=Z_topics.shape,
            item_ids_hash=item_ids_hash_topics
        )
        # Adicionar metadados de treino
        metadata_topics['training'] = train_metadata_topics
        
        # Adicionar info sobre features contínuas (para tópicos, sempre False)
        metadata_topics['continuous_features'] = {
            'included_in_ae': False,
            'concatenated_after': False,
            'columns': [],
            'final_dim': Z_topics.shape[1]
        }
        
        # Salvar com metadados
        save_embeddings_with_metadata(
            embeddings=Z_topics,
            item_ids=item_ids_topics,
            metadata=metadata_topics,
            output_path=topics_output,
            json_path=topics_json,
            verbose=verbose
        )
    
    if verbose:
        print("\n" + "="*60)
        print("EMBEDDINGS GERADOS COM SUCESSO")
        print("="*60)
        print(f"Features: {features_output}")
        print(f"Tópicos: {topics_output}")
    
    return features_output, topics_output


def main():
    """CLI para treinar autoencoders e gerar embeddings."""
    cfg = AUTOENCODER_CONFIG
    
    parser = argparse.ArgumentParser(
        description="Treina autoencoders e gera embeddings de features e tópicos"
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
        help=f"Dimensão dos embeddings (padrão config: {cfg['embedding_dim']})"
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=None,
        help=f"Dimensão da camada oculta (padrão config: {cfg['hidden_dim']})"
    )
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=None,
        help=f"Taxa de dropout para denoising (padrão config: {cfg['dropout_rate']})"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f"Número de épocas (padrão config: {cfg['epochs']})"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f"Tamanho do batch (padrão config: {cfg['batch_size']})"
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help=f"Taxa de aprendizado (padrão config: {cfg['learning_rate']})"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help=f"Seed para reprodutibilidade (padrão config: {cfg['seed']})"
    )
    parser.add_argument(
        '--pos-weight-mode',
        type=str,
        default=None,
        help=f"Modo de pos_weight para BCE (padrão config: {cfg['pos_weight_mode']})"
    )
    parser.add_argument(
        '--denoising-prob',
        type=float,
        default=None,
        help=f"Probabilidade de denoising (padrão config: {cfg['denoising_prob']})"
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=None,
        help=f"Weight decay para regularização L2 (padrão config: {cfg.get('weight_decay', 1e-5)})"
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=None,
        help=f"Paciência para early stopping (padrão config: {cfg.get('early_stopping_patience', 0)})"
    )
    parser.add_argument(
        '--no-l2-normalize',
        action='store_true',
        help='Desabilita normalização L2 dos embeddings (padrão: habilitado)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Força retreinamento mesmo se cache válido existir'
    )
    
    args = parser.parse_args()
    
    train_and_export_embeddings(
        data_dir=args.data_dir,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        pos_weight_mode=args.pos_weight_mode,
        denoising_prob=args.denoising_prob,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        l2_normalize=not args.no_l2_normalize if not args.no_l2_normalize else None,
        force_retrain=args.force,
        verbose=True
    )


if __name__ == '__main__':
    main()
