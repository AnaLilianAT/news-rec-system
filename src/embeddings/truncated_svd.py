"""
Gerador de embeddings usando TruncatedSVD (Latent Semantic Analysis).

Implementa redução de dimensionalidade para features binárias/esparsas,
com suporte a concatenação de features contínuas padronizadas.
"""

import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize
from pathlib import Path


def fit_transform_features(
    df: pd.DataFrame,
    id_col: str = 'news_id',
    binary_cols: Optional[List[str]] = None,
    continuous_cols: Optional[List[str]] = None,
    n_components: int = 32,
    random_state: int = 42,
    n_iter: int = 5,
    normalize_l2: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica TruncatedSVD em features binárias e concatena features contínuas padronizadas.
    
    Pipeline:
    1. Extrai IDs das notícias
    2. Separa features binárias e contínuas
    3. Aplica TruncatedSVD nas features binárias
    4. Padroniza features contínuas (StandardScaler)
    5. Concatena [svd_embedding | continuous_scaled]
    6. Normaliza L2 o embedding final (para uso com cosine similarity)
    
    Args:
        df: DataFrame com features das notícias
        id_col: Nome da coluna de identificação (padrão: 'news_id')
        binary_cols: Lista de colunas binárias. Se None, infere automaticamente
                    (colunas com valores únicos {0, 1})
        continuous_cols: Lista de colunas contínuas. Se None, usa []
        n_components: Número de componentes do SVD (dimensão do embedding)
        random_state: Seed para reprodutibilidade
        n_iter: Número de iterações do algoritmo SVD
        normalize_l2: Se True, aplica normalização L2 no embedding final
        verbose: Se True, imprime informações de progresso
    
    Returns:
        Tupla (ids, embeddings):
        - ids: Array numpy (n_items,) com identificadores das notícias
        - embeddings: Array numpy (n_items, n_components + len(continuous_cols))
                     com embeddings L2-normalizados
    
    Raises:
        ValueError: Se coluna de ID não existir ou dados inválidos
    
    Example:
        >>> df = pd.read_parquet('outputs/canonical_features.parquet')
        >>> ids, emb = fit_transform_features(
        ...     df,
        ...     id_col='news_id',
        ...     continuous_cols=['polaridade', 'subjetividade'],
        ...     n_components=32
        ... )
        >>> print(f"Shape: {emb.shape}, Norma média: {np.linalg.norm(emb, axis=1).mean():.3f}")
    """
    if verbose:
        print("\n" + "="*70)
        print("TRUNCATED SVD - GERAÇÃO DE EMBEDDINGS")
        print("="*70)
    
    # Validar presença da coluna de ID
    if id_col not in df.columns:
        raise ValueError(f"Coluna de ID '{id_col}' não encontrada. Colunas disponíveis: {df.columns.tolist()}")
    
    # Extrair IDs
    ids = df[id_col].values
    n_items = len(ids)
    
    if verbose:
        print(f"\nDataFrame de entrada:")
        print(f"  - Número de itens: {n_items}")
        print(f"  - Colunas totais: {len(df.columns)}")
        print(f"  - Coluna ID: '{id_col}'")
    
    # Inferir colunas binárias se não fornecidas
    if binary_cols is None:
        # Excluir coluna de ID e contínuas
        exclude_cols = {id_col}
        if continuous_cols:
            exclude_cols.update(continuous_cols)
        
        candidate_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Detectar colunas binárias (valores únicos são subconjunto de {0, 1, NaN})
        binary_cols = []
        for col in candidate_cols:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 2 and all(v in [0, 1, 0.0, 1.0] for v in unique_vals):
                binary_cols.append(col)
        
        if verbose:
            print(f"  - Colunas binárias detectadas: {len(binary_cols)}")
    else:
        if verbose:
            print(f"  - Colunas binárias fornecidas: {len(binary_cols)}")
    
    # Validar colunas contínuas
    if continuous_cols is None:
        continuous_cols = []
    
    if verbose:
        print(f"  - Colunas contínuas: {len(continuous_cols)}")
        if continuous_cols:
            print(f"    → {continuous_cols}")
    
    # Validar que todas as colunas existem
    missing_binary = [col for col in binary_cols if col not in df.columns]
    missing_continuous = [col for col in continuous_cols if col not in df.columns]
    
    if missing_binary:
        raise ValueError(f"Colunas binárias não encontradas: {missing_binary}")
    if missing_continuous:
        raise ValueError(f"Colunas contínuas não encontradas: {missing_continuous}")
    
    # === PARTE 1: TruncatedSVD nas features binárias ===
    if verbose:
        print(f"\n{'='*70}")
        print("ETAPA 1: TruncatedSVD nas features binárias")
        print(f"{'='*70}")
    
    X_binary = df[binary_cols].values.astype(float)
    
    if verbose:
        print(f"  - Shape da matriz binária: {X_binary.shape}")
        print(f"  - Esparsidade: {(X_binary == 0).sum() / X_binary.size * 100:.1f}% zeros")
        print(f"  - n_components: {n_components}")
        print(f"  - random_state: {random_state}")
        print(f"  - n_iter: {n_iter}")
    
    # Validar que n_components não excede dimensionalidade
    max_components = min(X_binary.shape) - 1
    if n_components > max_components:
        if verbose:
            print(f"\n  [AVISO] n_components={n_components} excede máximo permitido={max_components}")
            print(f"          Ajustando para n_components={max_components}")
        n_components = max_components
    
    # Treinar TruncatedSVD
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
        n_iter=n_iter,
        algorithm='randomized'
    )
    
    Z_svd = svd.fit_transform(X_binary)
    
    if verbose:
        explained_var = svd.explained_variance_ratio_.sum()
        print(f"\n  ✓ SVD treinado com sucesso")
        print(f"  - Shape do embedding SVD: {Z_svd.shape}")
        print(f"  - Variância explicada: {explained_var*100:.2f}%")
        print(f"  - Variância por componente (top 5): {svd.explained_variance_ratio_[:5]}")
    
    # === PARTE 2: Padronizar features contínuas ===
    Z_continuous = None
    if continuous_cols:
        if verbose:
            print(f"\n{'='*70}")
            print("ETAPA 2: Padronização de features contínuas")
            print(f"{'='*70}")
        
        X_continuous = df[continuous_cols].values.astype(float)
        
        if verbose:
            print(f"  - Shape da matriz contínua: {X_continuous.shape}")
            for i, col in enumerate(continuous_cols):
                print(f"    → {col}: média={X_continuous[:, i].mean():.3f}, "
                      f"std={X_continuous[:, i].std():.3f}")
        
        # Aplicar StandardScaler
        scaler = StandardScaler()
        Z_continuous = scaler.fit_transform(X_continuous)
        
        if verbose:
            print(f"\n  ✓ Padronização concluída")
            print(f"  - Shape do embedding contínuo: {Z_continuous.shape}")
            for i, col in enumerate(continuous_cols):
                print(f"    → {col} (scaled): média={Z_continuous[:, i].mean():.3f}, "
                      f"std={Z_continuous[:, i].std():.3f}")
    
    # === PARTE 3: Concatenar embeddings ===
    if verbose:
        print(f"\n{'='*70}")
        print("ETAPA 3: Concatenação de embeddings")
        print(f"{'='*70}")
    
    if Z_continuous is not None:
        Z_final = np.hstack([Z_svd, Z_continuous])
        if verbose:
            print(f"  - Shape SVD: {Z_svd.shape}")
            print(f"  - Shape contínuo: {Z_continuous.shape}")
            print(f"  - Shape concatenado: {Z_final.shape}")
    else:
        Z_final = Z_svd
        if verbose:
            print(f"  - Sem features contínuas. Shape final: {Z_final.shape}")
    
    # === PARTE 4: Normalização L2 ===
    if normalize_l2:
        if verbose:
            print(f"\n{'='*70}")
            print("ETAPA 4: Normalização L2")
            print(f"{'='*70}")
        
        # Calcular normas antes
        norms_before = np.linalg.norm(Z_final, axis=1)
        
        # Normalizar
        Z_final = normalize(Z_final, norm='l2', axis=1)
        
        # Calcular normas depois
        norms_after = np.linalg.norm(Z_final, axis=1)
        
        if verbose:
            print(f"  - Norma antes: média={norms_before.mean():.3f}, min={norms_before.min():.3f}, max={norms_before.max():.3f}")
            print(f"  - Norma depois: média={norms_after.mean():.6f}, min={norms_after.min():.6f}, max={norms_after.max():.6f}")
            print(f"  ✓ Normalização L2 concluída")
    
    # === VALIDAÇÃO FINAL ===
    if verbose:
        print(f"\n{'='*70}")
        print("VALIDAÇÃO")
        print(f"{'='*70}")
        
        nan_count = np.isnan(Z_final).sum()
        inf_count = np.isinf(Z_final).sum()
        
        print(f"  - Shape final: {Z_final.shape}")
        print(f"  - NaN: {nan_count}")
        print(f"  - Inf: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠️  AVISO: Valores inválidos detectados!")
        else:
            print(f"  ✓ Embeddings válidos")
    
    return ids, Z_final


def train_and_export_svd_embeddings(
    data_dir: str = "outputs",
    n_components: int = 32,
    random_state: int = 42,
    n_iter: int = 5,
    continuous_cols: Optional[List[str]] = None,
    normalize_l2: bool = True,
    verbose: bool = True
) -> Tuple[Path, Path]:
    """
    Pipeline completo: carrega features canônicas, treina SVD e salva embeddings.
    
    Gera embeddings para features e tópicos separadamente, seguindo o padrão
    do autoencoder existente.
    
    Args:
        data_dir: Diretório com canonical_features.parquet e canonical_topics.parquet
        n_components: Dimensão do embedding SVD
        random_state: Seed para reprodutibilidade
        n_iter: Número de iterações do SVD
        continuous_cols: Colunas contínuas a incluir (ex: ['polaridade', 'subjetividade'])
        normalize_l2: Se True, normaliza L2 os embeddings
        verbose: Se True, imprime progresso
    
    Returns:
        Tupla (features_output_path, topics_output_path) com caminhos dos parquets salvos
    """
    data_path = Path(data_dir)
    
    if continuous_cols is None:
        continuous_cols = ['polaridade', 'subjetividade']
    
    # Criar diretório de embeddings
    embeddings_dir = data_path / 'embeddings'
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================
    # FEATURES
    # ========================
    if verbose:
        print("\n" + "█"*70)
        print(" "*15 + "PROCESSANDO FEATURES")
        print("█"*70)
    
    features_path = data_path / 'canonical_features.parquet'
    if not features_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {features_path}\n"
            "Execute 'python -m src.build_canonical_tables' primeiro"
        )
    
    df_features = pd.read_parquet(features_path)
    
    # Detectar colunas contínuas disponíveis
    available_continuous = [col for col in continuous_cols if col in df_features.columns]
    
    if verbose and len(available_continuous) < len(continuous_cols):
        missing = set(continuous_cols) - set(available_continuous)
        print(f"\n[AVISO] Colunas contínuas não encontradas: {missing}")
        print(f"        Usando apenas: {available_continuous}")
    
    # Gerar embeddings
    ids_features, Z_features = fit_transform_features(
        df=df_features,
        id_col='news_id',
        binary_cols=None,  # Auto-detectar
        continuous_cols=available_continuous if available_continuous else None,
        n_components=n_components,
        random_state=random_state,
        n_iter=n_iter,
        normalize_l2=normalize_l2,
        verbose=verbose
    )
    
    # Salvar embeddings de features
    features_output = embeddings_dir / f'svd_features_d{n_components}_seed{random_state}.parquet'
    features_json = embeddings_dir / f'svd_features_d{n_components}_seed{random_state}.json'
    
    n_dims = Z_features.shape[1]
    cols = ['news_id'] + [f'emb_{i}' for i in range(n_dims)]
    df_emb_features = pd.DataFrame(
        data=np.column_stack([ids_features, Z_features]),
        columns=cols
    )
    df_emb_features['news_id'] = df_emb_features['news_id'].astype(int)
    
    df_emb_features.to_parquet(features_output, index=False, engine='pyarrow')
    
    # Salvar metadados JSON
    cache_key = f'd{n_components}_seed{random_state}_svd'
    metadata_features = {
        'cache_key': cache_key,
        'method': 'truncated_svd',
        'n_components': n_components,
        'random_state': random_state,
        'n_iter': n_iter,
        'normalize_l2': normalize_l2,
        'shape': Z_features.shape,
        'continuous_cols': available_continuous if available_continuous else [],
        'representation_type': 'svd_features'
    }
    with open(features_json, 'w') as f:
        json.dump(metadata_features, f, indent=2)
    
    if verbose:
        file_size = features_output.stat().st_size / 1024
        print(f"\n✓ Embeddings de features salvos em: {features_output.name}")
        print(f"  Tamanho: {file_size:.1f} KB")
    
    # ========================
    # TOPICS
    # ========================
    if verbose:
        print("\n" + "█"*70)
        print(" "*15 + "PROCESSANDO TOPICS")
        print("█"*70)
    
    topics_path = data_path / 'canonical_topics.parquet'
    if not topics_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {topics_path}\n"
            "Execute 'python -m src.build_canonical_tables' primeiro"
        )
    
    df_topics = pd.read_parquet(topics_path)
    
    # Topics não têm features contínuas
    ids_topics, Z_topics = fit_transform_features(
        df=df_topics,
        id_col='news_id',
        binary_cols=None,  # Auto-detectar (Topic0..Topic15)
        continuous_cols=None,
        n_components=min(n_components, 15),  # Máximo 15 para 16 tópicos
        random_state=random_state,
        n_iter=n_iter,
        normalize_l2=normalize_l2,
        verbose=verbose
    )
    
    # Salvar embeddings de topics  
    topics_output = embeddings_dir / f'svd_topics_d{n_components}_seed{random_state}.parquet'
    topics_json = embeddings_dir / f'svd_topics_d{n_components}_seed{random_state}.json'
    
    n_dims = Z_topics.shape[1]
    cols = ['news_id'] + [f'emb_{i}' for i in range(n_dims)]
    df_emb_topics = pd.DataFrame(
        data=np.column_stack([ids_topics, Z_topics]),
        columns=cols
    )
    df_emb_topics['news_id'] = df_emb_topics['news_id'].astype(int)
    
    df_emb_topics.to_parquet(topics_output, index=False, engine='pyarrow')
    
    # Salvar metadados JSON
    metadata_topics = {
        'cache_key': cache_key,
        'method': 'truncated_svd',
        'n_components': min(n_components, 15),
        'random_state': random_state,
        'n_iter': n_iter,
        'normalize_l2': normalize_l2,
        'shape': Z_topics.shape,
        'continuous_cols': [],
        'representation_type': 'svd_topics'
    }
    with open(topics_json, 'w') as f:
        json.dump(metadata_topics, f, indent=2)
    
    if verbose:
        file_size = topics_output.stat().st_size / 1024
        print(f"\n✓ Embeddings de topics salvos em: {topics_output.name}")
        print(f"  Tamanho: {file_size:.1f} KB")
    
    # === RESUMO FINAL ===
    if verbose:
        print("\n" + "="*70)
        print("PIPELINE CONCLUÍDO")
        print("="*70)
        print(f"\nArquivos gerados:")
        print(f"  1. {features_output.name}")
        print(f"     → {len(ids_features)} itens × {Z_features.shape[1]} dims")
        print(f"  2. {topics_output.name}")
        print(f"     → {len(ids_topics)} itens × {Z_topics.shape[1]} dims")
        print(f"\nPara usar nos experimentos:")
        print(f"  python -m src.run_generate_reclists_assigned \\")
        print(f"    --feature-representation svd_features \\")
        print(f"    --topic-representation svd_topics \\")
        print(f"    --embedding-dim {n_components} \\")
        print(f"    --seed {random_state}")
    
    return features_output, topics_output


if __name__ == '__main__':
    """CLI para geração de embeddings SVD."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gera embeddings usando TruncatedSVD (LSA)"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='outputs',
        help='Diretório com canonical_features.parquet e canonical_topics.parquet'
    )
    parser.add_argument(
        '--n-components',
        type=int,
        default=32,
        help='Número de componentes do SVD (dimensão do embedding)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Seed para reprodutibilidade'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=5,
        help='Número de iterações do algoritmo SVD'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Desabilita normalização L2 dos embeddings'
    )
    
    args = parser.parse_args()
    
    train_and_export_svd_embeddings(
        data_dir=args.data_dir,
        n_components=args.n_components,
        random_state=args.random_state,
        n_iter=args.n_iter,
        normalize_l2=not args.no_normalize,
        verbose=True
    )
