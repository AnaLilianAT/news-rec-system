"""
Geração de listas top-20 com diversificação por algoritmo atribuído (all-between).

Para cada checkpoint (user_id, t_rec), gera UMA lista top-20 usando a estratégia
de diversificação atribuída ao usuário em users.csv.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from .diversify import apply_diversification
from .io_loaders import load_tsv
from .representations import get_item_representation, prepare_item_vectors


def load_input_data(
    output_dir: Path = Path('outputs'),
    feature_representation: str = 'bin_features',
    topic_representation: str = 'bin_topics',
    embedding_dim: Optional[int] = None,
    seed: Optional[int] = None
):
    """
    Carrega dados de entrada necessários para gerar as listas.
    
    Args:
        output_dir: Diretório com outputs do pipeline
        feature_representation: Tipo de representação de features ('bin_features' ou 'ae_features')
        topic_representation: Tipo de representação de tópicos ('bin_topics' ou 'ae_topics')
        embedding_dim: Dimensão dos embeddings (apenas para ae_features/ae_topics)
        seed: Seed do embedding (apenas para ae_features/ae_topics)
    
    Returns:
        Tupla (predictions_df, features_df, topics_df, users_df)
    """
    print("Carregando dados de entrada...")
    
    # Predições scored
    predictions_path = output_dir / 'predictions_candidate_scored_assigned.parquet'
    predictions_df = pd.read_parquet(predictions_path)
    print(f"Predições: {len(predictions_df):,} registros")
    
    # Features (via nova interface de representações)
    print(f"Carregando representação de features: {feature_representation}")
    features_rep = get_item_representation(
        feature_representation, 
        output_dir=str(output_dir),
        embedding_dim=embedding_dim,
        seed=seed
    )
    features_df = features_rep.matrix
    print(f"Features: {len(features_df):,} notícias, {len(features_rep.feature_names)} dimensões")
    
    # Tópicos (via nova interface de representações)
    print(f"Carregando representação de tópicos: {topic_representation}")
    topics_rep = get_item_representation(
        topic_representation, 
        output_dir=str(output_dir),
        embedding_dim=embedding_dim,
        seed=seed
    )
    topics_df = topics_rep.matrix
    print(f"Tópicos: {len(topics_df):,} notícias, {len(topics_rep.feature_names)} dimensões")
    
    # Usuários (para recuperar algoritmo original)
    users_path = Path('dataset') / 'users.csv'
    users_df = load_tsv(users_path)
    print(f"Usuários: {len(users_df):,} registros")
    
    return predictions_df, features_df, topics_df, users_df


def prepare_feature_vectors(features_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Prepara dicionário de vetores de features por news_id.
    
    NOTA: Esta função é mantida por compatibilidade com o código existente.
    Para novos desenvolvimentos, usar prepare_item_vectors() do módulo representations.
    
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
    
    NOTA: Esta função é mantida por compatibilidade com o código existente.
    Para novos desenvolvimentos, usar prepare_item_vectors() do módulo representations.
    
    Args:
        topics_df: DataFrame com news_id e Topic0..Topic15
    
    Returns:
        Dicionário {news_id: topic_vector}
    """
    topic_cols = [f'Topic{i}' for i in range(16)]
    # Verificar quais colunas existem
    available_topic_cols = [col for col in topic_cols if col in topics_df.columns]
    
    if not available_topic_cols:
        # Tentar colunas genéricas (para suporte a embeddings no futuro)
        available_topic_cols = [col for col in topics_df.columns if col != 'news_id']
    
    if not available_topic_cols:
        print("AVISO: Nenhuma coluna de features encontrada no DataFrame de tópicos")
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
    output_dir: Path = Path('outputs'),
    representation_suffix: str = None
):
    """
    Salva listas top-20 e relatório.
    
    Args:
        reclists_df: DataFrame com listas geradas
        stats: Estatísticas da geração
        output_dir: Diretório de saída
        representation_suffix: Sufixo para diferenciar representações (ex: 'ae_features')
    """
    # Determinar nome do arquivo
    if representation_suffix:
        output_path = output_dir / f'reclists_top20_assigned_{representation_suffix}.parquet'
    else:
        output_path = output_dir / 'reclists_top20_assigned.parquet'
    
    # Salvar parquet
    reclists_df.to_parquet(output_path, index=False)
    
    file_size_kb = output_path.stat().st_size / 1024
    print(f"\nListas salvas: {output_path}")
    print(f"  - {len(reclists_df):,} registros")
    print(f"  - {file_size_kb:.1f} KB")
    
    # DESABILITADO: Relatório não é mais gerado
    # report_dir = output_dir / 'reports'
    # report_dir.mkdir(exist_ok=True)
    # 
    # if representation_suffix:
    #     report_path = report_dir / f'reclists_report_assigned_{representation_suffix}.md'
    # else:
    #     report_path = report_dir / 'reclists_report_assigned.md'
    # 
    # # with open(report_path, 'w', encoding='utf-8') as f:
    #     f.write("# Relatório de Geração de Listas Top-20 (ALL-BETWEEN)\n\n")
    #     f.write(f"**Data de geração**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    #     
    #     f.write("## Resumo Geral\n\n")
    #     f.write(f"- **Total de listas geradas**: {stats['total_lists']:,}\n")
    #     f.write(f"- **Listas completas (20 itens)**: {stats['complete_lists']:,} "
    #             f"({100*stats['complete_lists']/stats['total_lists']:.1f}%)\n")
    #     f.write(f"- **Listas incompletas (<20)**: {stats['incomplete_lists']:,} "
    #             f"({100*stats['incomplete_lists']/stats['total_lists']:.1f}%)\n\n")
    #     
    #     f.write("## Distribuição por Estratégia de Diversificação\n\n")
    #     f.write("| Estratégia | Listas | Percentual |\n")
    #     f.write("|------------|--------|------------|\n")
    #     for diversify, count in sorted(stats['by_diversify'].items()):
    #         pct = 100 * count / stats['total_lists']
    #         f.write(f"| {diversify} | {count:,} | {pct:.1f}% |\n")
    #     
    #     f.write("\n## Distribuição por Algoritmo Base\n\n")
    #     f.write("| Algoritmo | Listas | Percentual |\n")
    #     f.write("|-----------|--------|------------|\n")
    #     for algo, count in sorted(stats['by_base_algo'].items()):
    #         pct = 100 * count / stats['total_lists']
    #         f.write(f"| {algo} | {count:,} | {pct:.1f}% |\n")
    #     
    #     f.write("\n## Estatísticas dos Registros\n\n")
    #     f.write(f"- **Total de itens recomendados**: {len(reclists_df):,}\n")
    #     f.write(f"- **Usuários únicos**: {reclists_df['user_id'].nunique():,}\n")
    #     f.write(f"- **Notícias únicas**: {reclists_df['news_id'].nunique():,}\n")
    #     f.write(f"- **Score médio**: {reclists_df['score_pred'].mean():.4f}\n")
    #     f.write(f"- **Score mínimo**: {reclists_df['score_pred'].min():.4f}\n")
    #     f.write(f"- **Score máximo**: {reclists_df['score_pred'].max():.4f}\n\n")
    #     
    #     f.write("## Distribuição de Tamanho das Listas\n\n")
    #     list_sizes = reclists_df.groupby(['user_id', 't_rec']).size()
    #     f.write(f"- **Tamanho médio**: {list_sizes.mean():.1f} itens\n")
    #     f.write(f"- **Tamanho mínimo**: {list_sizes.min()} itens\n")
    #     f.write(f"- **Tamanho máximo**: {list_sizes.max()} itens\n\n")
    #     
    #     size_dist = list_sizes.value_counts().sort_index()
    #     f.write("### Distribuição detalhada:\n\n")
    #     f.write("| Tamanho | Quantidade |\n")
    #     f.write("|---------|------------|\n")
    #     for size, count in size_dist.items():
    #         f.write(f"| {size} | {count:,} |\n")
    #     
    #     f.write("\n## Validação\n\n")
    #     f.write("- Todas as listas foram geradas com o algoritmo atribuído ao usuário\n")
    #     f.write("- Estratégias de diversificação aplicadas conforme configurado\n")
    #     f.write("- Rankings preservam ordem de score (exceto quando diversificado)\n\n")
    # 
    # print(f"Relatório salvo: {report_path}")


def main():
    """
    Função principal: gera listas top-20 com diversificação.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Gera listas top-20 com diversificação por algoritmo atribuído'
    )
    parser.add_argument(
        '--feature-representation',
        type=str,
        default='bin_features',
        choices=['bin_features', 'ae_features'],
        help='Tipo de representação de features (default: bin_features). Ignorado se --representations usado.'
    )
    parser.add_argument(
        '--topic-representation',
        type=str,
        default='bin_topics',
        choices=['bin_topics', 'ae_topics'],
        help='Tipo de representação de tópicos (default: bin_topics). Ignorado se --representations usado.'
    )
    parser.add_argument(
        '--representations',
        type=str,
        nargs='+',
        choices=['bin_features', 'ae_features', 'bin_topics', 'ae_topics'],
        help='Rodar com múltiplas representações (ex: --representations bin_features ae_features)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Diretório base dos outputs (default: outputs)'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=None,
        help='Dimensão dos embeddings para representações ae_* (default: None = usa dim32)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed do embedding para representações ae_* (default: None = usa primeiro encontrado)'
    )
    
    args = parser.parse_args()
    
    # Determinar quais representações processar
    if args.representations:
        # Modo múltiplas representações - APENAS pares homogêneos
        # Gera apenas: ae_features+ae_topics e bin_features+bin_topics
        representations_to_process = []
        
        # Organizar em pares (features, topics)
        feature_reps = [r for r in args.representations if 'features' in r]
        topic_reps = [r for r in args.representations if 'topics' in r]
        
        # Garantir que temos pelo menos uma de cada
        if not feature_reps:
            feature_reps = ['bin_features']
        if not topic_reps:
            topic_reps = ['bin_topics']
        
        # NOVA LÓGICA: Criar apenas pares homogêneos (mesma representação para features e topics)
        # Se ae_features está presente e ae_topics também, gera ae_features+ae_topics
        if 'ae_features' in feature_reps and 'ae_topics' in topic_reps:
            representations_to_process.append(('ae_features', 'ae_topics'))
        
        # Se bin_features está presente e bin_topics também, gera bin_features+bin_topics
        if 'bin_features' in feature_reps and 'bin_topics' in topic_reps:
            representations_to_process.append(('bin_features', 'bin_topics'))
    else:
        # Modo single representação (default)
        representations_to_process = [(args.feature_representation, args.topic_representation)]
    
    print("=" * 70)
    print("  GERAÇÃO DE LISTAS TOP-20 (ALL-BETWEEN)")
    print("=" * 70)
    
    if len(representations_to_process) > 1:
        print(f"\n⚙ Modo múltiplas representações: {len(representations_to_process)} combinações")
    
    # Processar cada representação
    for idx, (feat_rep, topic_rep) in enumerate(representations_to_process, 1):
        if len(representations_to_process) > 1:
            print(f"\n{'='*70}")
            print(f"  PROCESSANDO [{idx}/{len(representations_to_process)}]: {feat_rep} + {topic_rep}")
            print(f"{'='*70}")
        
        print(f"\nConfigurações:")
        print(f"  - Representação de features: {feat_rep}")
        print(f"  - Representação de tópicos: {topic_rep}")
        print(f"  - Diretório de saída: {args.output_dir}")
        
        # Carregar dados (com representação escolhida)
        predictions_df, features_df, topics_df, users_df = load_input_data(
            output_dir=Path(args.output_dir),
            feature_representation=feat_rep,
            topic_representation=topic_rep,
            embedding_dim=getattr(args, 'embedding_dim', None),
            seed=getattr(args, 'seed', None)
        )
        
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
        
        # Determinar sufixo para arquivos
        if len(representations_to_process) > 1:
            # Modo múltiplas: usar sufixo composto
            suffix = f"{feat_rep}+{topic_rep}"
        else:
            # Modo single: só usar sufixo se não for default
            if feat_rep == 'bin_features' and topic_rep == 'bin_topics':
                suffix = None  # Default: sem sufixo
            else:
                suffix = f"{feat_rep}+{topic_rep}"
        
        # Salvar resultados
        save_reclists(reclists_df, stats, output_dir=Path(args.output_dir), representation_suffix=suffix)
    
    print("\n" + "=" * 70)
    print("  GERAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)


if __name__ == '__main__':
    main()
