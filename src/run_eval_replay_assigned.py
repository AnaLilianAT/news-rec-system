"""
Avaliação de métricas no replay temporal usando algoritmos atribuídos (all-between).

Métricas:
- RMSE: erro entre rating_real e score_pred para itens expostos (na top-20)
- GH (homogeneidade): similaridade média entre itens da lista top-20
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, List
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

from .similarity import cosine_similarity


def load_eval_data(
    output_dir: Path = Path('outputs'),
    representation_suffix: str = None
):
    """
    Carrega dados necessários para avaliação.
    
    Args:
        output_dir: Diretório com outputs do pipeline
        representation_suffix: Sufixo da representação (ex: 'ae_features+ae_topics')
    
    Returns:
        Tupla (interactions_df, checkpoints_df, reclists_df, features_df, topics_df)
    """
    print("Carregando dados para avaliação...")
    
    # Interações canônicas
    interactions_path = output_dir / 'canonical_interactions.parquet'
    interactions_df = pd.read_parquet(interactions_path)
    interactions_df['rating_when'] = pd.to_datetime(interactions_df['rating_when'], utc=True)
    print(f"Interações: {len(interactions_df):,}")
    
    # Checkpoints replay
    checkpoints_path = output_dir / 'replay_checkpoints.parquet'
    checkpoints_df = pd.read_parquet(checkpoints_path)
    checkpoints_df['t_rec'] = pd.to_datetime(checkpoints_df['t_rec'], utc=True)
    checkpoints_df['t_next_rec'] = pd.to_datetime(checkpoints_df['t_next_rec'], utc=True)
    print(f"Checkpoints: {len(checkpoints_df):,}")
    
    # Listas top-20 (com sufixo se aplicável)
    if representation_suffix:
        reclists_path = output_dir / f'reclists_top20_assigned_{representation_suffix}.parquet'
    else:
        reclists_path = output_dir / 'reclists_top20_assigned.parquet'
    reclists_df = pd.read_parquet(reclists_path)
    reclists_df['t_rec'] = pd.to_datetime(reclists_df['t_rec'], utc=True)
    print(f"Listas top-20: {len(reclists_df):,}")
    
    # Features
    features_path = output_dir / 'canonical_features.parquet'
    features_df = pd.read_parquet(features_path)
    print(f"Features: {len(features_df):,}")
    
    # Tópicos
    topics_path = output_dir / 'canonical_topics.parquet'
    topics_df = pd.read_parquet(topics_path)
    print(f"Tópicos: {len(topics_df):,}")
    
    return interactions_df, checkpoints_df, reclists_df, features_df, topics_df


def build_eval_pairs(
    interactions_df: pd.DataFrame,
    checkpoints_df: pd.DataFrame,
    reclists_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Constrói pares de avaliação: interações do teste que estão na top-20.
    
    Args:
        interactions_df: Interações canônicas
        checkpoints_df: Checkpoints com t_rec e t_next_rec
        reclists_df: Listas top-20 geradas
    
    Returns:
        DataFrame com eval_pairs
    """
    print("\nConstruindo eval_pairs...")
    
    # Criar índice para reclists: (user_id, t_rec, news_id) -> (algorithm, score_pred)
    reclists_idx = reclists_df.set_index(['user_id', 't_rec', 'news_id'])[['algorithm', 'score_pred']]
    
    eval_pairs = []
    total_test_interactions = 0
    total_exposed = 0
    
    for _, checkpoint in tqdm(checkpoints_df.iterrows(), total=len(checkpoints_df), desc="Processando checkpoints"):
        user_id = checkpoint['user_id']
        t_rec = checkpoint['t_rec']
        t_next_rec = checkpoint['t_next_rec']
        
        # Filtrar interações do teste: (t_rec, t_next_rec]
        test_mask = (
            (interactions_df['user_id'] == user_id) &
            (interactions_df['rating_when'] > t_rec) &
            (interactions_df['rating_when'] <= t_next_rec)
        )
        test_interactions = interactions_df[test_mask]
        total_test_interactions += len(test_interactions)
        
        # Para cada interação do teste, verificar se está na top-20
        for _, interaction in test_interactions.iterrows():
            news_id = interaction['news_id']
            
            # Verificar se (user_id, t_rec, news_id) está nas reclists
            try:
                reclist_info = reclists_idx.loc[(user_id, t_rec, news_id)]
                algorithm = reclist_info['algorithm']
                score_pred = reclist_info['score_pred']
                
                eval_pairs.append({
                    'user_id': user_id,
                    't_rec': t_rec,
                    'algorithm': algorithm,
                    'news_id': news_id,
                    'rating_real': interaction['rating'],
                    'score_pred': score_pred,
                    'rating_when': interaction['rating_when']
                })
                total_exposed += 1
            except KeyError:
                # news_id não está na top-20 deste checkpoint
                pass
    
    eval_pairs_df = pd.DataFrame(eval_pairs)
    
    exposure_rate = 100 * total_exposed / total_test_interactions if total_test_interactions > 0 else 0
    print(f"Total de interações no teste: {total_test_interactions:,}")
    print(f"Interações expostas (na top-20): {total_exposed:,} ({exposure_rate:.2f}%)")
    print(f"Eval pairs criados: {len(eval_pairs_df):,}")
    
    return eval_pairs_df, total_test_interactions, total_exposed


def calculate_rmse_metrics(eval_pairs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula métricas RMSE global e por usuário.
    
    Args:
        eval_pairs_df: DataFrame com pares de avaliação
    
    Returns:
        Tupla (rmse_by_algorithm, rmse_by_user)
    """
    print("\nCalculando RMSE...")
    
    # RMSE global por algorithm
    rmse_results = []
    
    for algorithm in eval_pairs_df['algorithm'].unique():
        algo_pairs = eval_pairs_df[eval_pairs_df['algorithm'] == algorithm]
        
        if len(algo_pairs) == 0:
            continue
        
        # Calcular erro quadrático médio
        mse = ((algo_pairs['rating_real'] - algo_pairs['score_pred']) ** 2).mean()
        rmse = np.sqrt(mse)
        
        rmse_results.append({
            'algorithm': algorithm,
            'rmse': rmse,
            'n_pairs': len(algo_pairs),
            'mean_rating_real': algo_pairs['rating_real'].mean(),
            'mean_score_pred': algo_pairs['score_pred'].mean()
        })
    
    rmse_by_algorithm = pd.DataFrame(rmse_results).sort_values('rmse')
    
    # RMSE por usuário (agregando por algorithm)
    user_rmse_results = []
    
    for (user_id, algorithm), group in eval_pairs_df.groupby(['user_id', 'algorithm']):
        if len(group) < 2:  # Mínimo 2 pares para calcular RMSE
            continue
        
        mse = ((group['rating_real'] - group['score_pred']) ** 2).mean()
        rmse = np.sqrt(mse)
        
        user_rmse_results.append({
            'user_id': user_id,
            'algorithm': algorithm,
            'rmse': rmse,
            'n_pairs': len(group)
        })
    
    rmse_by_user = pd.DataFrame(user_rmse_results)
    
    print(f"RMSE calculado para {len(rmse_by_algorithm)} algoritmos")
    print(f"RMSE por usuário: {len(rmse_by_user)} registros")
    
    return rmse_by_algorithm, rmse_by_user


def prepare_feature_vectors(features_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Prepara vetores de features."""
    feature_cols = [col for col in features_df.columns if col != 'news_id']
    return {
        row['news_id']: row[feature_cols].values.astype(float)
        for _, row in features_df.iterrows()
    }


def prepare_topic_vectors(topics_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Prepara vetores de tópicos."""
    topic_cols = [f'Topic{i}' for i in range(16)]
    available_topic_cols = [col for col in topic_cols if col in topics_df.columns]
    
    if not available_topic_cols:
        return {}
    
    return {
        row['news_id']: row[available_topic_cols].values.astype(float)
        for _, row in topics_df.iterrows()
    }


def calculate_gh_cosine(news_ids: List[int], feature_vectors: Dict[int, np.ndarray]) -> float:
    """
    Calcula homogeneidade usando similaridade cosseno entre features.
    
    Args:
        news_ids: Lista de news_ids da top-20
        feature_vectors: Dicionário com vetores de features
    
    Returns:
        GH médio (similaridade cosseno média entre todos os pares)
    """
    # Filtrar apenas news_ids que têm features
    valid_ids = [nid for nid in news_ids if nid in feature_vectors]
    
    if len(valid_ids) < 2:
        return np.nan
    
    # Calcular similaridade entre todos os pares
    similarities = []
    for nid1, nid2 in combinations(valid_ids, 2):
        vec1 = feature_vectors[nid1]
        vec2 = feature_vectors[nid2]
        sim = cosine_similarity(vec1, vec2)
        similarities.append(sim)
    
    return np.mean(similarities) if similarities else np.nan


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calcula similaridade de Jaccard entre dois conjuntos."""
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def calculate_gh_jaccard(news_ids: List[int], topic_vectors: Dict[int, np.ndarray], threshold: float = 0.1) -> float:
    """
    Calcula homogeneidade usando Jaccard entre tópicos binários.
    
    Args:
        news_ids: Lista de news_ids da top-20
        topic_vectors: Dicionário com vetores de tópicos
        threshold: Limiar para binarizar tópicos
    
    Returns:
        GH médio (Jaccard médio entre todos os pares)
    """
    # Filtrar apenas news_ids que têm tópicos
    valid_ids = [nid for nid in news_ids if nid in topic_vectors]
    
    if len(valid_ids) < 2:
        return np.nan
    
    # Converter tópicos em conjuntos binários (índices dos tópicos dominantes)
    topic_sets = {}
    for nid in valid_ids:
        vec = topic_vectors[nid]
        # Tópicos acima do threshold
        dominant_topics = set(np.where(vec > threshold)[0])
        topic_sets[nid] = dominant_topics
    
    # Calcular Jaccard entre todos os pares
    similarities = []
    for nid1, nid2 in combinations(valid_ids, 2):
        set1 = topic_sets[nid1]
        set2 = topic_sets[nid2]
        sim = jaccard_similarity(set1, set2)
        similarities.append(sim)
    
    return np.mean(similarities) if similarities else np.nan


def calculate_gh_metrics(
    reclists_df: pd.DataFrame,
    feature_vectors: Dict[int, np.ndarray],
    topic_vectors: Dict[int, np.ndarray]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula métricas de homogeneidade (GH) para cada lista top-20.
    
    Args:
        reclists_df: Listas top-20
        feature_vectors: Features para GH_COSINE
        topic_vectors: Tópicos para GH_JACCARD
    
    Returns:
        Tupla (gh_cosine_by_algorithm, gh_jaccard_by_algorithm)
    """
    print("\nCalculando GH (homogeneidade)...")
    
    gh_results = []
    
    # Agrupar por (user_id, t_rec, algorithm)
    for (user_id, t_rec, algorithm), group in tqdm(
        reclists_df.groupby(['user_id', 't_rec', 'algorithm']),
        desc="Processando listas"
    ):
        news_ids = group['news_id'].tolist()
        
        # GH_COSINE_FEATURES
        gh_cosine = calculate_gh_cosine(news_ids, feature_vectors)
        
        # GH_JACCARD_TOPICS
        gh_jaccard = calculate_gh_jaccard(news_ids, topic_vectors)
        
        gh_results.append({
            'user_id': user_id,
            't_rec': t_rec,
            'algorithm': algorithm,
            'gh_cosine': gh_cosine,
            'gh_jaccard': gh_jaccard,
            'list_size': len(news_ids)
        })
    
    gh_df = pd.DataFrame(gh_results)
    
    # Agregar por algorithm - GH_COSINE
    gh_cosine_by_algo = gh_df.groupby('algorithm').agg({
        'gh_cosine': ['mean', 'median', 'std', 'count']
    }).round(4)
    gh_cosine_by_algo.columns = ['mean_gh_cosine', 'median_gh_cosine', 'std_gh_cosine', 'n_lists']
    gh_cosine_by_algo = gh_cosine_by_algo.reset_index()
    
    # Agregar por algorithm - GH_JACCARD
    gh_jaccard_by_algo = gh_df.groupby('algorithm').agg({
        'gh_jaccard': ['mean', 'median', 'std', 'count']
    }).round(4)
    gh_jaccard_by_algo.columns = ['mean_gh_jaccard', 'median_gh_jaccard', 'std_gh_jaccard', 'n_lists']
    gh_jaccard_by_algo = gh_jaccard_by_algo.reset_index()
    
    print(f"GH calculado para {len(gh_df)} listas")
    
    return gh_cosine_by_algo, gh_jaccard_by_algo, gh_df


def aggregate_user_metrics(
    rmse_by_user: pd.DataFrame,
    gh_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Agrega métricas por usuário e algorithm.
    
    Args:
        rmse_by_user: RMSE por usuário
        gh_df: GH por lista
    
    Returns:
        DataFrame com métricas agregadas por usuário
    """
    print("\nAgregando métricas por usuário...")
    
    # Agregar GH por usuário
    gh_by_user = gh_df.groupby(['user_id', 'algorithm']).agg({
        'gh_cosine': 'mean',
        'gh_jaccard': 'mean',
        'list_size': 'count'  # número de listas
    }).reset_index()
    gh_by_user.rename(columns={'list_size': 'n_lists'}, inplace=True)
    
    # Merge com RMSE
    user_metrics = rmse_by_user.merge(
        gh_by_user,
        on=['user_id', 'algorithm'],
        how='outer'
    )
    
    print(f"Métricas agregadas para {user_metrics['user_id'].nunique()} usuários")
    
    return user_metrics


def save_metrics(
    eval_pairs_df: pd.DataFrame,
    rmse_by_algorithm: pd.DataFrame,
    gh_cosine_by_algo: pd.DataFrame,
    gh_jaccard_by_algo: pd.DataFrame,
    user_metrics: pd.DataFrame,
    total_test_interactions: int,
    total_exposed: int,
    output_dir: Path = Path('outputs'),
    representation_suffix: str = None
):
    """
    Salva todos os outputs de métricas.
    
    Args:
        eval_pairs_df: Pares de avaliação
        rmse_by_algorithm: RMSE por algorithm
        gh_cosine_by_algo: GH cosine por algorithm
        gh_jaccard_by_algo: GH Jaccard por algorithm
        user_metrics: Métricas por usuário
        total_test_interactions: Total de interações no teste
        total_exposed: Total de interações expostas
        output_dir: Diretório de saída
        representation_suffix: Sufixo da representação (ex: 'ae_features+ae_topics')
    """
    print("\nSalvando métricas...")
    
    # Determinar sufixo para arquivos
    suffix = f"_{representation_suffix}" if representation_suffix else ""
    
    # Salvar eval_pairs (necessário para run_export_thesis_tables.py)
    eval_pairs_path = output_dir / f'eval_pairs_assigned{suffix}.parquet'
    eval_pairs_df.to_parquet(eval_pairs_path, index=False)
    print(f"Eval pairs: {eval_pairs_path} ({len(eval_pairs_df):,} registros)")


def generate_report(
    eval_pairs_df: pd.DataFrame,
    rmse_by_algorithm: pd.DataFrame,
    gh_cosine_by_algo: pd.DataFrame,
    gh_jaccard_by_algo: pd.DataFrame,
    user_metrics: pd.DataFrame,
    total_test_interactions: int,
    total_exposed: int,
    output_dir: Path,
    representation_suffix: str = None
):
    """
    Gera relatório markdown com resultados da avaliação.
    """
    report_dir = output_dir / 'reports'
    report_dir.mkdir(exist_ok=True)
    
    suffix = f"_{representation_suffix}" if representation_suffix else ""
    report_path = report_dir / f'eval_report_assigned{suffix}.md'
    
    exposure_rate = 100 * total_exposed / total_test_interactions if total_test_interactions > 0 else 0
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Avaliação - Replay Temporal (ALL-BETWEEN)\n\n")
        f.write(f"**Data de geração**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Exposição\n\n")
        f.write(f"- **Total de interações no teste**: {total_test_interactions:,}\n")
        f.write(f"- **Interações expostas (na top-20)**: {total_exposed:,}\n")
        f.write(f"- **Taxa de exposição**: {exposure_rate:.2f}%\n\n")
        
        f.write("## RMSE por Algoritmo\n\n")
        f.write("RMSE mede o erro de predição entre score_pred e rating_real para itens expostos.\n\n")
        f.write("| Algoritmo | RMSE | N Pares | Média Rating Real | Média Score Pred |\n")
        f.write("|-----------|------|---------|-------------------|------------------|\n")
        for _, row in rmse_by_algorithm.iterrows():
            f.write(f"| {row['algorithm']} | {row['rmse']:.4f} | {row['n_pairs']:,} | "
                   f"{row['mean_rating_real']:.3f} | {row['mean_score_pred']:.3f} |\n")
        
        f.write("\n### Estatísticas RMSE por Usuário\n\n")
        if len(user_metrics[user_metrics['rmse'].notna()]) > 0:
            rmse_stats = user_metrics.groupby('algorithm')['rmse'].agg(['mean', 'median', 'std', 'count'])
            f.write("| Algoritmo | Média RMSE | Mediana RMSE | Std RMSE | N Usuários |\n")
            f.write("|-----------|------------|--------------|----------|------------|\n")
            for algo, row in rmse_stats.iterrows():
                f.write(f"| {algo} | {row['mean']:.4f} | {row['median']:.4f} | "
                       f"{row['std']:.4f} | {int(row['count'])} |\n")
        else:
            f.write("Dados insuficientes para calcular RMSE por usuário.\n")
        
        f.write("\n## GH Cosine (Homogeneidade por Features)\n\n")
        f.write("GH_COSINE mede a similaridade média (cosseno) entre itens da lista top-20.\n\n")
        f.write("| Algoritmo | Média GH | Mediana GH | Std GH | N Listas |\n")
        f.write("|-----------|----------|------------|--------|----------|\n")
        for _, row in gh_cosine_by_algo.iterrows():
            f.write(f"| {row['algorithm']} | {row['mean_gh_cosine']:.4f} | "
                   f"{row['median_gh_cosine']:.4f} | {row['std_gh_cosine']:.4f} | "
                   f"{int(row['n_lists'])} |\n")
        
        f.write("\n## GH Jaccard (Homogeneidade por Tópicos)\n\n")
        f.write("GH_JACCARD mede a similaridade Jaccard média entre tópicos dominantes dos itens.\n\n")
        f.write("| Algoritmo | Média GH | Mediana GH | Std GH | N Listas |\n")
        f.write("|-----------|----------|------------|--------|----------|\n")
        for _, row in gh_jaccard_by_algo.iterrows():
            f.write(f"| {row['algorithm']} | {row['mean_gh_jaccard']:.4f} | "
                   f"{row['median_gh_jaccard']:.4f} | {row['std_gh_jaccard']:.4f} | "
                   f"{int(row['n_lists'])} |\n")
        
        f.write("\n## Resumo Geral\n\n")
        f.write(f"- **Algoritmos avaliados**: {len(rmse_by_algorithm)}\n")
        f.write(f"- **Usuários com métricas**: {user_metrics['user_id'].nunique()}\n")
        f.write(f"- **Total de eval_pairs**: {len(eval_pairs_df):,}\n")
        f.write(f"- **Listas avaliadas (GH)**: {gh_cosine_by_algo['n_lists'].sum():.0f}\n\n")
        
        # Top 5 algoritmos por RMSE
        f.write("### Top 5 Algoritmos (Menor RMSE)\n\n")
        top5_rmse = rmse_by_algorithm.head(5)
        for idx, row in top5_rmse.iterrows():
            f.write(f"{idx + 1}. **{row['algorithm']}**: RMSE = {row['rmse']:.4f} ({row['n_pairs']:,} pares)\n")
        
        # Top 5 algoritmos por menor GH cosine (mais diversificados)
        f.write("\n### Top 5 Algoritmos (Menor GH Cosine = Mais Diversos)\n\n")
        top5_diverse = gh_cosine_by_algo.sort_values('mean_gh_cosine').head(5)
        for idx, (_, row) in enumerate(top5_diverse.iterrows(), 1):
            f.write(f"{idx}. **{row['algorithm']}**: GH = {row['mean_gh_cosine']:.4f} ({int(row['n_lists'])} listas)\n")
        
        f.write("\n## Interpretação\n\n")
        f.write("- **RMSE baixo**: Predições mais precisas\n")
        f.write("- **GH alto**: Lista mais homogênea (menos diversa)\n")
        f.write("- **GH baixo**: Lista mais diversificada\n")
        f.write("- **Taxa de exposição**: Percentual de avaliações no teste que foram recomendadas\n")
    
    print(f"Relatório: {report_path}")


def main():
    """
    Função principal: avalia métricas no replay temporal.
    """
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(
        description='Avalia métricas no replay temporal usando algoritmos atribuídos'
    )
    parser.add_argument(
        '--representations',
        type=str,
        nargs='+',
        help='Sufixos de representações a processar (ex: bin_features+bin_topics ae_features+ae_topics)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Diretório de saída (default: outputs)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  AVALIAÇÃO - REPLAY TEMPORAL (ALL-BETWEEN)")
    print("=" * 70)
    
    # Determinar quais representações processar
    output_path = Path(args.output_dir)
    
    if args.representations:
        # Processar representações especificadas
        suffixes_to_process = args.representations
    else:
        # Auto-detectar arquivos reclists disponíveis
        reclists_files = list(output_path.glob('reclists_top20_assigned*.parquet'))
        suffixes_to_process = []
        
        for file in reclists_files:
            filename = file.stem  # reclists_top20_assigned ou reclists_top20_assigned_XXX
            if filename == 'reclists_top20_assigned':
                suffixes_to_process.append(None)  # Default
            else:
                # Extrair sufixo
                suffix = filename.replace('reclists_top20_assigned_', '')
                suffixes_to_process.append(suffix)
        
        if not suffixes_to_process:
            print("ERRO: Nenhum arquivo reclists_top20_assigned*.parquet encontrado")
            return
    
    if len(suffixes_to_process) > 1:
        print(f"\n[INFO] Processando {len(suffixes_to_process)} representações")
    
    # Processar cada representação
    for idx, suffix in enumerate(suffixes_to_process, 1):
        if len(suffixes_to_process) > 1:
            print(f"\n{'='*70}")
            print(f"  PROCESSANDO [{idx}/{len(suffixes_to_process)}]: {suffix or 'default'}")
            print(f"{'='*70}")
        
        # Carregar dados
        interactions_df, checkpoints_df, reclists_df, features_df, topics_df = load_eval_data(
            output_dir=output_path,
            representation_suffix=suffix
        )
        
        # Construir eval_pairs
        eval_pairs_df, total_test_interactions, total_exposed = build_eval_pairs(
            interactions_df, checkpoints_df, reclists_df
        )
        
        if len(eval_pairs_df) == 0:
            print("\nNenhum eval_pair encontrado. Não há interações expostas para avaliar.")
            continue
        
        # Calcular RMSE
        rmse_by_algorithm, rmse_by_user = calculate_rmse_metrics(eval_pairs_df)
        
        # Preparar vetores para GH
        print("\nPreparando vetores para GH...")
        feature_vectors = prepare_feature_vectors(features_df)
        print(f"Feature vectors: {len(feature_vectors):,}")
        
        topic_vectors = prepare_topic_vectors(topics_df)
        print(f"Topic vectors: {len(topic_vectors):,}")
        
        # Calcular GH
        gh_cosine_by_algo, gh_jaccard_by_algo, gh_df = calculate_gh_metrics(
            reclists_df, feature_vectors, topic_vectors
        )
        
        # Agregar métricas por usuário
        user_metrics = aggregate_user_metrics(rmse_by_user, gh_df)
        
        # Salvar tudo
        save_metrics(
            eval_pairs_df,
            rmse_by_algorithm,
            gh_cosine_by_algo,
            gh_jaccard_by_algo,
            user_metrics,
            total_test_interactions,
            total_exposed,
            output_dir=output_path,
            representation_suffix=suffix
        )
    
    print("\n" + "=" * 70)
    print("  AVALIAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)

if __name__ == '__main__':
    main()
