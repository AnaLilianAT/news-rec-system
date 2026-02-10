"""
Script para treinar autoencoders e gerar embeddings.

Carrega matrizes binárias (features e tópicos), treina AEs separados,
e salva embeddings com cache + metadados para reuso.
"""

import argparse
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from .autoencoder import train_autoencoder, extract_embeddings
from ..representations.item_representation import get_item_representation


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


def get_cache_metadata(
    embedding_dim: int,
    hidden_dim: Optional[int],
    dropout_rate: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    data_hash: str,
    representation_type: str
) -> Dict[str, Any]:
    """
    Cria dicionário de metadados para cache.
    
    Returns:
        Dicionário com config e hash dos dados
    """
    return {
        'representation_type': representation_type,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'dropout_rate': dropout_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'seed': seed,
        'data_hash': data_hash
    }


def check_cache_valid(
    cache_json_path: Path,
    current_metadata: Dict[str, Any]
) -> bool:
    """
    Verifica se cache existente é válido (mesma config + data_hash).
    
    Args:
        cache_json_path: Caminho para arquivo JSON de metadados
        current_metadata: Metadados atuais para comparar
    
    Returns:
        True se cache é válido
    """
    if not cache_json_path.exists():
        return False
    
    try:
        with open(cache_json_path, 'r') as f:
            cached_metadata = json.load(f)
        
        # Comparar todos os campos relevantes
        for key in ['embedding_dim', 'hidden_dim', 'dropout_rate', 'epochs',
                    'batch_size', 'learning_rate', 'seed', 'data_hash']:
            if cached_metadata.get(key) != current_metadata.get(key):
                return False
        
        return True
    
    except Exception as e:
        print(f"Erro ao ler cache: {e}")
        return False


def save_embeddings_with_metadata(
    embeddings: np.ndarray,
    item_ids: np.ndarray,
    metadata: Dict[str, Any],
    output_path: Path,
    json_path: Path
):
    """
    Salva embeddings em parquet + metadados em JSON.
    
    Args:
        embeddings: Matriz (n_items, embedding_dim)
        item_ids: Array com news_id
        metadata: Dicionário de metadados
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
    
    # Salvar metadados
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadados salvos: {json_path}")


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
    embedding_dim: int = 32,
    hidden_dim: Optional[int] = None,
    dropout_rate: float = 0.1,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 42,
    force_retrain: bool = False,
    verbose: bool = True
) -> Tuple[Path, Path]:
    """
    Treina autoencoders para features e tópicos, gerando embeddings.
    
    Implementa cache: se embeddings com mesma config já existem, reutiliza.
    
    Args:
        data_dir: Diretório com canonical_features.parquet e canonical_topics.parquet
        embedding_dim: Dimensão dos embeddings
        hidden_dim: Dimensão da camada oculta (None = auto)
        dropout_rate: Taxa de dropout para denoising
        epochs: Número de épocas
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        seed: Seed para reprodutibilidade
        force_retrain: Se True, ignora cache e retreina
        verbose: Se True, imprime progresso
    
    Returns:
        Tupla com (path_features, path_topics) dos embeddings gerados
    """
    data_path = Path(data_dir)
    embeddings_dir = data_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Definir caminhos de saída
    features_output = embeddings_dir / f"ae_features_dim{embedding_dim}.parquet"
    features_json = embeddings_dir / f"ae_features_dim{embedding_dim}.json"
    topics_output = embeddings_dir / f"ae_topics_dim{embedding_dim}.parquet"
    topics_json = embeddings_dir / f"ae_topics_dim{embedding_dim}.json"
    
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
    
    # Calcular hash e metadados
    data_hash_features = compute_data_hash(X_features)
    metadata_features = get_cache_metadata(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        data_hash=data_hash_features,
        representation_type='ae_features'
    )
    
    # Verificar cache
    cache_valid_features = check_cache_valid(features_json, metadata_features)
    
    if cache_valid_features and not force_retrain:
        if verbose:
            print("\n✓ Cache válido encontrado. Reutilizando embeddings existentes.")
            print(f"  Arquivo: {features_output}")
    else:
        if force_retrain and verbose:
            print("\n⚠ --force especificado. Retreinando modelo...")
        elif verbose:
            print("\n✗ Cache inválido ou inexistente. Treinando novo modelo...")
        
        # Treinar autoencoder
        model_features = train_autoencoder(
            X=X_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            verbose=verbose
        )
        
        # Extrair embeddings (com L2 normalization)
        Z_features = extract_embeddings(
            model=model_features,
            X=X_features,
            normalize_l2=True
        )
        
        # Salvar com metadados
        save_embeddings_with_metadata(
            embeddings=Z_features,
            item_ids=item_ids_features,
            metadata=metadata_features,
            output_path=features_output,
            json_path=features_json
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
    
    # Calcular hash e metadados
    data_hash_topics = compute_data_hash(X_topics)
    metadata_topics = get_cache_metadata(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        data_hash=data_hash_topics,
        representation_type='ae_topics'
    )
    
    # Verificar cache
    cache_valid_topics = check_cache_valid(topics_json, metadata_topics)
    
    if cache_valid_topics and not force_retrain:
        if verbose:
            print("\n✓ Cache válido encontrado. Reutilizando embeddings existentes.")
            print(f"  Arquivo: {topics_output}")
    else:
        if force_retrain and verbose:
            print("\n⚠ --force especificado. Retreinando modelo...")
        elif verbose:
            print("\n✗ Cache inválido ou inexistente. Treinando novo modelo...")
        
        # Treinar autoencoder
        model_topics = train_autoencoder(
            X=X_topics,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            verbose=verbose
        )
        
        # Extrair embeddings (com L2 normalization)
        Z_topics = extract_embeddings(
            model=model_topics,
            X=X_topics,
            normalize_l2=True
        )
        
        # Salvar com metadados
        save_embeddings_with_metadata(
            embeddings=Z_topics,
            item_ids=item_ids_topics,
            metadata=metadata_topics,
            output_path=topics_output,
            json_path=topics_json
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
        default=32,
        help='Dimensão dos embeddings (padrão: 32)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=None,
        help='Dimensão da camada oculta (padrão: média entre input e embedding)'
    )
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0.1,
        help='Taxa de dropout para denoising (padrão: 0.1)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Número de épocas (padrão: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamanho do batch (padrão: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Taxa de aprendizado (padrão: 0.001)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed para reprodutibilidade (padrão: 42)'
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
        force_retrain=args.force,
        verbose=True
    )


if __name__ == '__main__':
    main()
