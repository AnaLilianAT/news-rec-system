"""
Módulo para construir tabelas canônicas a partir dos dados brutos.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import sys

from .io_loaders import load_all_datasets


def build_interactions(ratings: pd.DataFrame, news_ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói tabela canônica de interações (avaliações explícitas).
    
    Args:
        ratings: DataFrame com avaliações (id, rating, rating_when, user_id)
        news_ratings: DataFrame com relação notícia-avaliação (id, news_id, ratings_id)
    
    Returns:
        DataFrame com colunas: user_id, news_id, rating, rating_when
    """
    print("\n" + "="*70)
    print("CONSTRUINDO DF_INTERACTIONS")
    print("="*70)
    
    # Join: ratings.id -> news_ratings.ratings_id
    df = ratings.merge(
        news_ratings[['news_id', 'ratings_id']], 
        left_on='id', 
        right_on='ratings_id',
        how='inner'
    )
    
    # Selecionar e renomear colunas
    df_interactions = df[['user_id', 'news_id', 'rating', 'rating_when']].copy()
    
    # Garantir tipos corretos
    df_interactions['user_id'] = df_interactions['user_id'].astype(int)
    df_interactions['news_id'] = df_interactions['news_id'].astype(int)
    df_interactions['rating'] = df_interactions['rating'].astype(float)
    df_interactions['rating_when'] = pd.to_datetime(df_interactions['rating_when'])
    
    # Ordenar por timestamp
    df_interactions = df_interactions.sort_values('rating_when').reset_index(drop=True)
    
    print(f"Interações criadas: {len(df_interactions)} registros")
    print(f"  → Usuários únicos: {df_interactions['user_id'].nunique()}")
    print(f"  → Notícias únicas: {df_interactions['news_id'].nunique()}")
    print(f"  → Período: {df_interactions['rating_when'].min()} a {df_interactions['rating_when'].max()}")
    print(f"  → Rating médio: {df_interactions['rating'].mean():.2f}")
    
    return df_interactions


def build_rec_sessions(recLists: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói tabela canônica de sessões de recomendação.
    
    Agrupa recLists por (user_id, generated_when, diversifed) e produz sessões com:
    - user_id, generated_when, diversifed, list_size, news_ids (lista ordenada)
    
    Args:
        recLists: DataFrame com listas de recomendação
    
    Returns:
        DataFrame com sessões de recomendação
    """
    print("\n" + "="*70)
    print("CONSTRUINDO DF_REC_SESSIONS")
    print("="*70)
    
    # Converter tipos
    recLists = recLists.copy()
    recLists['generated_when'] = pd.to_datetime(recLists['generated_when'])
    recLists['user_id'] = recLists['user_id'].astype(int)
    recLists['news_id'] = recLists['news_id'].astype(int)
    
    # Ordenar por predicted_rating descendente para ter ranking
    recLists = recLists.sort_values(
        ['user_id', 'generated_when', 'diversifed', 'predicted_rating'],
        ascending=[True, True, True, False]
    )
    
    # Agrupar por sessão
    sessions = recLists.groupby(['user_id', 'generated_when', 'diversifed']).agg(
        list_size=('news_id', 'count'),
        news_ids=('news_id', lambda x: list(x))
    ).reset_index()
    
    # Ordenar por timestamp
    sessions = sessions.sort_values('generated_when').reset_index(drop=True)
    
    print(f"Sessões criadas: {len(sessions)} registros")
    print(f"  → Usuários únicos: {sessions['user_id'].nunique()}")
    print(f"  → Período: {sessions['generated_when'].min()} a {sessions['generated_when'].max()}")
    print(f"\n  Distribuição de list_size:")
    list_size_dist = sessions['list_size'].value_counts().sort_index(ascending=False).head(10)
    for size, count in list_size_dist.items():
        print(f"    {size:3d} itens: {count:5d} sessões ({count/len(sessions)*100:.1f}%)")
    
    return sessions


def build_features_and_topics(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constrói tabelas canônicas de features e topics.
    
    Args:
        features_df: DataFrame com features das notícias
    
    Returns:
        Tupla com (df_features, df_topics)
    """
    print("\n" + "="*70)
    print("CONSTRUINDO DF_FEATURES E DF_TOPICS")
    print("="*70)
    
    features_df = features_df.copy()
    
    # Garantir que 'id' é news_id
    if 'id' in features_df.columns:
        features_df['news_id'] = features_df['id'].astype(int)
        features_df = features_df.drop(columns=['id'])
    
    # Identificar colunas de tópicos (Topic0..Topic15)
    topic_cols = [col for col in features_df.columns if col.startswith('Topic')]
    topic_cols_sorted = sorted(topic_cols, key=lambda x: int(x.replace('Topic', '')))
    
    # Extrair df_topics
    df_topics = features_df[['news_id'] + topic_cols_sorted].copy()
    
    # Garantir que topics são inteiros (0 ou 1)
    for col in topic_cols_sorted:
        df_topics[col] = df_topics[col].astype(int)
    
    print(f"Topics extraídos: {len(df_topics)} notícias")
    print(f"  → Colunas de topics: {len(topic_cols_sorted)} ({topic_cols_sorted[0]}..{topic_cols_sorted[-1]})")
    
    # Contar quantos topics por notícia
    topic_counts = df_topics[topic_cols_sorted].sum(axis=1)
    print(f"  → Topics por notícia (média): {topic_counts.mean():.2f}")
    print(f"  → Topics por notícia (mediana): {topic_counts.median():.0f}")
    
    # Construir df_features (todas as colunas numéricas exceto news_id)
    # Incluir: Topics, polaridade, subjetividade, e demais features numéricas
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Garantir que news_id está incluso
    if 'news_id' not in numeric_cols:
        feature_cols = ['news_id'] + numeric_cols
    else:
        feature_cols = numeric_cols
        # Mover news_id para primeira coluna
        feature_cols = ['news_id'] + [col for col in feature_cols if col != 'news_id']
    
    df_features = features_df[feature_cols].copy()
    
    # Verificar se polaridade e subjetividade estão presentes
    has_polaridade = 'polaridade' in df_features.columns
    has_subjetividade = 'subjetividade' in df_features.columns
    
    print(f"\nFeatures construídas: {len(df_features)} notícias, {len(df_features.columns)-1} features")
    print(f"  → Colunas principais: news_id, {', '.join(df_features.columns[1:6].tolist())}...")
    if has_polaridade:
        print(f"  → Polaridade: média={df_features['polaridade'].mean():.3f}, std={df_features['polaridade'].std():.3f}")
    if has_subjetividade:
        print(f"  → Subjetividade: média={df_features['subjetividade'].mean():.3f}, std={df_features['subjetividade'].std():.3f}")
    
    # Verificar valores nulos
    null_counts = df_features.isnull().sum()
    if null_counts.sum() > 0:
        print(f"\n  Valores nulos encontrados:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"    {col}: {count} ({count/len(df_features)*100:.1f}%)")
    
    return df_features, df_topics


def save_canonical_tables(
    df_interactions: pd.DataFrame,
    df_rec_sessions: pd.DataFrame,
    df_features: pd.DataFrame,
    df_topics: pd.DataFrame,
    output_dir: str = 'outputs'
):
    """
    Salva tabelas canônicas em formato Parquet.
    
    Args:
        df_interactions: Tabela de interações
        df_rec_sessions: Tabela de sessões de recomendação
        df_features: Tabela de features
        df_topics: Tabela de topics
        output_dir: Diretório de saída
    """
    print("\n" + "="*70)
    print("SALVANDO TABELAS CANÔNICAS")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Salvar cada tabela
    tables = {
        'canonical_interactions.parquet': df_interactions,
        'canonical_rec_sessions.parquet': df_rec_sessions,
        'canonical_features.parquet': df_features,
        'canonical_topics.parquet': df_topics
    }
    
    for filename, df in tables.items():
        filepath = output_path / filename
        df.to_parquet(filepath, index=False, engine='pyarrow')
        file_size = filepath.stat().st_size / 1024  # KB
        print(f"Salvo {filename}: {len(df)} linhas, {file_size:.1f} KB")
    
    print("\n" + "="*70)
    print("SALVAMENTO CONCLUÍDO")
    print("="*70)


def main():
    """
    Função principal: carrega dados, constrói tabelas canônicas e salva.
    """
    print("\n" + "█"*70)
    print(" "*15 + "PIPELINE: BUILD CANONICAL TABLES")
    print("█"*70)
    
    # Carregar todos os datasets
    datasets = load_all_datasets()
    
    # Construir tabelas canônicas
    df_interactions = build_interactions(datasets['ratings'], datasets['news_ratings'])
    df_rec_sessions = build_rec_sessions(datasets['recLists'])
    df_features, df_topics = build_features_and_topics(datasets['features'])
    
    # Salvar tabelas
    save_canonical_tables(df_interactions, df_rec_sessions, df_features, df_topics)
    
    # Executar verificações de sanidade
    from .sanity_checks import run_sanity_checks
    run_sanity_checks(
        df_interactions=df_interactions,
        df_rec_sessions=df_rec_sessions,
        df_features=df_features,
        df_topics=df_topics,
        news_df=datasets['news']
    )
    
    print("\n" + "█"*70)
    print(" "*20 + "PIPELINE CONCLUÍDO COM SUCESSO!")
    print("█"*70 + "\n")


if __name__ == '__main__':
    main()
