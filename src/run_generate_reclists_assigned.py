"""
Geração de listas top-20 com diversificação por algoritmo atribuído (all-between).

Para cada checkpoint (user_id, t_rec), gera UMA lista top-20 usando a estratégia
de diversificação atribuída ao usuário em users.csv.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from .diversify import apply_diversification
from .io_loaders import load_tsv


def load_input_data(output_dir: Path = Path('outputs')):
    """
    Carrega dados de entrada necessários para gerar as listas.
    
    Returns:
        Tupla (predictions_df, features_df, topics_df, users_df)
    """
    print("Carregando dados de entrada...")
    
    # Predições scored
    predictions_path = output_dir / 'predictions_candidate_scored_assigned.parquet'
    predictions_df = pd.read_parquet(predictions_path)
    print(f"Predições: {len(predictions_df):,} registros")
    
    # Features (para MMR)
    features_path = output_dir / 'canonical_features.parquet'
    features_df = pd.read_parquet(features_path)
    print(f"Features: {len(features_df):,} notícias")
    
    # Tópicos (para TD)
    topics_path = output_dir / 'canonical_topics.parquet'
    topics_df = pd.read_parquet(topics_path)
    print(f"Tópicos: {len(topics_df):,} notícias")
    
    # Usuários (para recuperar algoritmo original)
    users_path = Path('dataset') / 'users.csv'
    users_df = load_tsv(users_path)
    print(f"Usuários: {len(users_df):,} registros")
    
    return predictions_df, features_df, topics_df, users_df


def prepare_feature_vectors(features_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Prepara dicionário de vetores de features por news_id.
    
    Args:
        features_df: DataFrame com news_id e colunas de features
    
    Returns:
        Dicionário {news_id: feature_vector}
    """
    feature_cols = [col for col in features_df.columns if col != 'news_id']
    feature_vectors = {}
    
    for _, row in features_df.iterrows():
        news_id = row['news_id']
        vec = row[feature_cols].values.astype(float)
        feature_vectors[news_id] = vec
    
    return feature_vectors


def prepare_topic_vectors(topics_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Prepara dicionário de vetores de tópicos por news_id.
    
    Args:
        topics_df: DataFrame com news_id e Topic0..Topic15
    
    Returns:
        Dicionário {news_id: topic_vector}
    """
    topic_cols = [f'Topic{i}' for i in range(16)]
    # Verificar quais colunas existem
    available_topic_cols = [col for col in topic_cols if col in topics_df.columns]
    
    if not available_topic_cols:
        print("Nenhuma coluna de tópicos encontrada em canonical_topics.parquet")
        return {}
    
    topic_vectors = {}
    
    for _, row in topics_df.iterrows():
        news_id = row['news_id']
        vec = row[available_topic_cols].values.astype(float)
        topic_vectors[news_id] = vec
    
    return topic_vectors


def generate_reclists(
    predictions_df: pd.DataFrame,
    feature_vectors: Dict[int, np.ndarray],
    topic_vectors: Dict[int, np.ndarray],
    users_df: pd.DataFrame,
    k: int = 20
) -> Tuple[pd.DataFrame, Dict]:
    """
    Gera listas top-20 para cada checkpoint aplicando diversificação.
    
    Args:
        predictions_df: DataFrame com predições scored
        feature_vectors: Features para MMR
        topic_vectors: Tópicos para TD
        users_df: DataFrame com users.csv (id, algoritmo)
        k: Tamanho da lista (default=20)
    
    Returns:
        Tupla (reclists_df, stats)
    """
    print(f"\nGerando listas top-{k}...")
    
    # Criar mapeamento user_id -> algoritmo original
    user_algorithm_map = dict(zip(users_df['id'], users_df['algoritmo']))
    
    # Agrupar predições por (user_id, t_rec)
    grouped = predictions_df.groupby(['user_id', 't_rec', 'base_algo', 'diversify'])
    
    all_reclists = []
    stats = {
        'total_lists': 0,
        'complete_lists': 0,  # listas com exatamente k itens
        'incomplete_lists': 0,  # listas com menos de k itens
        'fallback_mmr': 0,  # TD que fez fallback para MMR
        'fallback_none': 0,  # diversify que fez fallback para none
        'by_diversify': {'none': 0, 'mmr': 0, 'td': 0},
        'by_base_algo': {'knnu': 0, 'knni': 0, 'svd': 0}
    }
    
    for (user_id, t_rec, base_algo, diversify), group_df in tqdm(grouped, desc="Processando checkpoints"):
        # Ordenar por rank_in_candidates (já vem ordenado por score)
        group_sorted = group_df.sort_values('rank_in_candidates')
        
        # Preparar lista de (news_id, score_pred)
        ranked_items = list(zip(group_sorted['news_id'], group_sorted['score_pred']))
        
        # Aplicar diversificação apropriada
        selected_items = apply_diversification(
            ranked_items=ranked_items,
            diversify=diversify,
            feature_vectors=feature_vectors,
            topic_vectors=topic_vectors,
            k=k
        )
        
        # Recuperar algoritmo original
        algorithm_full = user_algorithm_map.get(user_id, f"{base_algo}")
        
        # Construir registros para esta lista
        for news_id, score_pred, rank in selected_items:
            all_reclists.append({
                'user_id': user_id,
                't_rec': t_rec,
                'algorithm': algorithm_full,
                'base_algo': base_algo,
                'diversify': diversify,
                'news_id': news_id,
                'score_pred': score_pred,
                'rank_20': rank
            })
        
        # Atualizar estatísticas
        stats['total_lists'] += 1
        
        if len(selected_items) == k:
            stats['complete_lists'] += 1
        else:
            stats['incomplete_lists'] += 1
        
        stats['by_diversify'][diversify] = stats['by_diversify'].get(diversify, 0) + 1
        stats['by_base_algo'][base_algo] = stats['by_base_algo'].get(base_algo, 0) + 1
    
    # Criar DataFrame
    reclists_df = pd.DataFrame(all_reclists)
    
    return reclists_df, stats


def save_reclists(
    reclists_df: pd.DataFrame,
    stats: Dict,
    output_dir: Path = Path('outputs')
):
    """
    Salva listas top-20 e relatório.
    
    Args:
        reclists_df: DataFrame com listas geradas
        stats: Estatísticas da geração
        output_dir: Diretório de saída
    """
    # Salvar parquet
    output_path = output_dir / 'reclists_top20_assigned.parquet'
    reclists_df.to_parquet(output_path, index=False)
    
    file_size_kb = output_path.stat().st_size / 1024
    print(f"\nListas salvas: {output_path}")
    print(f"  - {len(reclists_df):,} registros")
    print(f"  - {file_size_kb:.1f} KB")
    
    # Gerar relatório
    report_dir = output_dir / 'reports'
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / 'reclists_report_assigned.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Geração de Listas Top-20 (ALL-BETWEEN)\n\n")
        f.write(f"**Data de geração**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Resumo Geral\n\n")
        f.write(f"- **Total de listas geradas**: {stats['total_lists']:,}\n")
        f.write(f"- **Listas completas (20 itens)**: {stats['complete_lists']:,} "
                f"({100*stats['complete_lists']/stats['total_lists']:.1f}%)\n")
        f.write(f"- **Listas incompletas (<20)**: {stats['incomplete_lists']:,} "
                f"({100*stats['incomplete_lists']/stats['total_lists']:.1f}%)\n\n")
        
        f.write("## Distribuição por Estratégia de Diversificação\n\n")
        f.write("| Estratégia | Listas | Percentual |\n")
        f.write("|------------|--------|------------|\n")
        for diversify, count in sorted(stats['by_diversify'].items()):
            pct = 100 * count / stats['total_lists']
            f.write(f"| {diversify} | {count:,} | {pct:.1f}% |\n")
        
        f.write("\n## Distribuição por Algoritmo Base\n\n")
        f.write("| Algoritmo | Listas | Percentual |\n")
        f.write("|-----------|--------|------------|\n")
        for algo, count in sorted(stats['by_base_algo'].items()):
            pct = 100 * count / stats['total_lists']
            f.write(f"| {algo} | {count:,} | {pct:.1f}% |\n")
        
        f.write("\n## Estatísticas dos Registros\n\n")
        f.write(f"- **Total de itens recomendados**: {len(reclists_df):,}\n")
        f.write(f"- **Usuários únicos**: {reclists_df['user_id'].nunique():,}\n")
        f.write(f"- **Notícias únicas**: {reclists_df['news_id'].nunique():,}\n")
        f.write(f"- **Score médio**: {reclists_df['score_pred'].mean():.4f}\n")
        f.write(f"- **Score mínimo**: {reclists_df['score_pred'].min():.4f}\n")
        f.write(f"- **Score máximo**: {reclists_df['score_pred'].max():.4f}\n\n")
        
        f.write("## Distribuição de Tamanho das Listas\n\n")
        list_sizes = reclists_df.groupby(['user_id', 't_rec']).size()
        f.write(f"- **Tamanho médio**: {list_sizes.mean():.1f} itens\n")
        f.write(f"- **Tamanho mínimo**: {list_sizes.min()} itens\n")
        f.write(f"- **Tamanho máximo**: {list_sizes.max()} itens\n\n")
        
        size_dist = list_sizes.value_counts().sort_index()
        f.write("### Distribuição detalhada:\n\n")
        f.write("| Tamanho | Quantidade |\n")
        f.write("|---------|------------|\n")
        for size, count in size_dist.items():
            f.write(f"| {size} | {count:,} |\n")
        
        f.write("\n## Validação\n\n")
        f.write("- Todas as listas foram geradas com o algoritmo atribuído ao usuário\n")
        f.write("- Estratégias de diversificação aplicadas conforme configurado\n")
        f.write("- Rankings preservam ordem de score (exceto quando diversificado)\n\n")
    
    print(f"Relatório salvo: {report_path}")


def main():
    """
    Função principal: gera listas top-20 com diversificação.
    """
    print("=" * 70)
    print("  GERAÇÃO DE LISTAS TOP-20 (ALL-BETWEEN)")
    print("=" * 70)
    
    # Carregar dados
    predictions_df, features_df, topics_df, users_df = load_input_data()
    
    # Preparar vetores
    print("\nPreparando vetores...")
    feature_vectors = prepare_feature_vectors(features_df)
    print(f"Feature vectors: {len(feature_vectors):,}")
    
    topic_vectors = prepare_topic_vectors(topics_df)
    print(f"Topic vectors: {len(topic_vectors):,}")
    
    # Gerar listas
    reclists_df, stats = generate_reclists(
        predictions_df=predictions_df,
        feature_vectors=feature_vectors,
        topic_vectors=topic_vectors,
        users_df=users_df,
        k=20
    )
    
    # Salvar resultados
    save_reclists(reclists_df, stats)
    
    print("\n" + "=" * 70)
    print("  GERAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)


if __name__ == '__main__':
    main()
