"""
Pipeline de treino e score com replay temporal usando Surprise.
Cada usuário recebe predições apenas do algoritmo atribuído a ele.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .config import (
    RANDOM_SEED, MIN_CANDIDATE_SIZE, BATCH_SIZE, 
    CACHE_BY_TREC, MAX_ITEMS_TO_RANK, VERBOSE
)
from .models_surprise import (
    create_model, parse_user_algorithm, rank_candidates_by_score
)
from .io_loaders import load_tsv


def load_replay_data(base_path='outputs', dataset_path='dataset'):
    """
    Carrega dados necessários para replay.
    """
    print("\n" + "="*70)
    print("CARREGANDO DADOS")
    print("="*70)
    
    base_path = Path(base_path)
    dataset_path = Path(dataset_path)
    
    df_checkpoints = pd.read_parquet(base_path / 'replay_checkpoints.parquet')
    df_interactions = pd.read_parquet(base_path / 'canonical_interactions.parquet')
    df_users = load_tsv(dataset_path / 'users.csv')
    
    print(f"Checkpoints: {len(df_checkpoints):,}")
    print(f"Interações: {len(df_interactions):,}")
    print(f"Usuários: {len(df_users):,}")
    
    return df_checkpoints, df_interactions, df_users


def parse_user_algorithms(df_users: pd.DataFrame) -> pd.DataFrame:
    """
    Parseia algoritmos dos usuários.
    """
    print("\n" + "="*70)
    print("PARSEANDO ALGORITMOS DOS USUÁRIOS")
    print("="*70)
    
    # Aplicar parsing
    algo_parsed = df_users['algoritmo'].apply(parse_user_algorithm)
    df_users['base_algo'] = algo_parsed.apply(lambda x: x[0])
    df_users['diversify'] = algo_parsed.apply(lambda x: x[1])
    
    # Estatísticas
    print(f"\nDistribuição de base_algo:")
    for algo, count in df_users['base_algo'].value_counts().items():
        pct = count / len(df_users) * 100
        print(f"  {algo:5s}: {count:3d} usuários ({pct:5.1f}%)")
    
    print(f"\nDistribuição de diversify:")
    for div, count in df_users['diversify'].value_counts().items():
        pct = count / len(df_users) * 100
        print(f"  {div:5s}: {count:3d} usuários ({pct:5.1f}%)")
    
    return df_users[['id', 'base_algo', 'diversify']].rename(columns={'id': 'user_id'})


def prepare_checkpoints(df_checkpoints: pd.DataFrame, df_user_algos: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara checkpoints com informação de algoritmo.
    """
    print("\n" + "="*70)
    print("PREPARANDO CHECKPOINTS")
    print("="*70)
    
    # Filtrar apenas checkpoints com candidate pool válido
    valid_checkpoints = df_checkpoints[
        (df_checkpoints['has_candidate_pool'] == True) &
        (df_checkpoints['candidate_size'] >= MIN_CANDIDATE_SIZE)
    ].copy()
    
    print(f"Checkpoints válidos: {len(valid_checkpoints):,}/{len(df_checkpoints):,}")
    
    # Join com algoritmos dos usuários
    checkpoints_with_algo = valid_checkpoints.merge(
        df_user_algos,
        on='user_id',
        how='inner'
    )
    
    print(f"Checkpoints com algoritmo: {len(checkpoints_with_algo):,}")
    
    # Ordenar por t_rec
    checkpoints_with_algo = checkpoints_with_algo.sort_values('t_rec').reset_index(drop=True)
    
    return checkpoints_with_algo


def group_checkpoints_by_trec(df_checkpoints: pd.DataFrame) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Agrupa checkpoints por t_rec.
    """
    print("\n" + "="*70)
    print("AGRUPANDO CHECKPOINTS POR T_REC")
    print("="*70)
    
    grouped = {}
    for t_rec, group in df_checkpoints.groupby('t_rec'):
        grouped[t_rec] = group
    
    print(f"Timestamps únicos: {len(grouped):,}")
    print(f"  → Checkpoints por t_rec (média): {len(df_checkpoints) / len(grouped):.1f}")
    
    # Estatísticas sobre algoritmos necessários por t_rec
    algos_per_trec = []
    for t_rec, group in grouped.items():
        algos_needed = group['base_algo'].nunique()
        algos_per_trec.append(algos_needed)
    
    print(f"\n  Algoritmos necessários por t_rec:")
    print(f"    Média: {np.mean(algos_per_trec):.1f}")
    print(f"    Mediana: {np.median(algos_per_trec):.0f}")
    print(f"    Máximo: {max(algos_per_trec)}")
    
    return grouped


def train_models_for_trec(
    t_rec: pd.Timestamp,
    algos_needed: Set[str],
    df_interactions: pd.DataFrame
) -> Dict[str, object]:
    """
    Treina modelos necessários para um t_rec específico.
    """
    # Preparar dados de treino (todas interações antes de t_rec)
    train_data = df_interactions[df_interactions['rating_when'] < t_rec].copy()
    
    if len(train_data) == 0:
        return {}
    
    # Treinar apenas modelos necessários
    models = {}
    for algo in algos_needed:
        if VERBOSE:
            print(f"    Treinando {algo}... ", end='', flush=True)
        
        model = create_model(algo)
        model.fit(train_data[['user_id', 'news_id', 'rating']])
        models[algo] = model
        
        if VERBOSE:
            print(f"({len(train_data):,} interações)")
    
    return models


def score_checkpoint(
    checkpoint: pd.Series,
    model: object
) -> List[Dict]:
    """
    Gera scores para um checkpoint.
    """
    user_id = checkpoint['user_id']
    t_rec = checkpoint['t_rec']
    base_algo = checkpoint['base_algo']
    diversify = checkpoint['diversify']
    candidate_pool = checkpoint['candidate_news_ids']
    candidate_size = checkpoint['candidate_size']
    
    # Predizer scores para todos os candidatos
    scores = model.predict_batch(user_id, candidate_pool)
    
    # Ranquear por score
    ranked = rank_candidates_by_score(scores, max_items=MAX_ITEMS_TO_RANK)
    
    # Preparar resultados
    results = []
    for news_id, score, rank in ranked:
        results.append({
            'user_id': user_id,
            't_rec': t_rec,
            'base_algo': base_algo,
            'diversify': diversify,
            'news_id': news_id,
            'score_pred': score,
            'rank_in_candidates': rank,
            'candidate_size': candidate_size
        })
    
    return results


def process_checkpoints_batch(
    checkpoints_grouped: Dict[pd.Timestamp, pd.DataFrame],
    df_interactions: pd.DataFrame,
    output_path: str = 'outputs/predictions_candidate_scored_assigned.parquet'
) -> Dict:
    """
    Processa checkpoints em batch, treinando modelos por t_rec.
    """
    print("\n" + "="*70)
    print("PROCESSANDO CHECKPOINTS")
    print("="*70)
    
    all_predictions = []
    stats = {
        'total_checkpoints': 0,
        'total_trecs': len(checkpoints_grouped),
        'total_trainings': 0,
        'trainings_by_algo': {'knnu': 0, 'knni': 0, 'svd': 0},
        'avg_train_size': [],
        'skipped_checkpoints': 0
    }
    
    # Processar por t_rec
    sorted_trecs = sorted(checkpoints_grouped.keys())
    
    for i, t_rec in enumerate(tqdm(sorted_trecs, desc="Processando t_rec", disable=not VERBOSE)):
        group = checkpoints_grouped[t_rec]
        stats['total_checkpoints'] += len(group)
        
        # Identificar algoritmos necessários
        algos_needed = set(group['base_algo'].unique())
        
        # Treinar modelos
        if VERBOSE:
            print(f"\n  t_rec {i+1}/{len(sorted_trecs)}: {t_rec.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    Checkpoints: {len(group)}")
            print(f"    Algoritmos necessários: {algos_needed}")
        
        models = train_models_for_trec(t_rec, algos_needed, df_interactions)
        
        if len(models) == 0:
            if VERBOSE:
                print(f"    Sem dados de treino, pulando...")
            stats['skipped_checkpoints'] += len(group)
            continue
        
        # Atualizar estatísticas de treino
        train_size = len(df_interactions[df_interactions['rating_when'] < t_rec])
        stats['avg_train_size'].append(train_size)
        stats['total_trainings'] += len(models)
        for algo in models.keys():
            stats['trainings_by_algo'][algo] += 1
        
        # Gerar scores para cada checkpoint
        for _, checkpoint in group.iterrows():
            algo = checkpoint['base_algo']
            
            if algo not in models:
                stats['skipped_checkpoints'] += 1
                continue
            
            # Score checkpoint
            predictions = score_checkpoint(checkpoint, models[algo])
            all_predictions.extend(predictions)
    
    # Salvar predições
    print(f"\nTotal de predições geradas: {len(all_predictions):,}")
    
    if len(all_predictions) > 0:
        df_predictions = pd.DataFrame(all_predictions)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_predictions.to_parquet(output_path, index=False, engine='pyarrow')
        
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"Salvo: {output_path.name} ({file_size:.1f} KB)")
    
    # Calcular estatísticas finais
    if len(stats['avg_train_size']) > 0:
        stats['avg_train_size'] = np.mean(stats['avg_train_size'])
    else:
        stats['avg_train_size'] = 0
    
    return stats


def generate_train_score_report(
    stats: Dict,
    output_path: str = 'outputs/reports/train_score_report_assigned.md'
):
    """
    Gera relatório de treino e score.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Treino e Score (Assigned)\n\n")
        f.write(f"**Data de geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 1. Resumo do Processamento\n\n")
        f.write(f"- **Total de checkpoints processados:** {stats['total_checkpoints']:,}\n")
        f.write(f"- **Timestamps únicos (t_rec):** {stats['total_trecs']:,}\n")
        f.write(f"- **Checkpoints pulados:** {stats['skipped_checkpoints']:,}\n")
        f.write(f"- **Taxa de sucesso:** {(stats['total_checkpoints'] - stats['skipped_checkpoints'])/stats['total_checkpoints']*100:.1f}%\n\n")
        
        f.write("## 2. Treinamentos Executados\n\n")
        f.write(f"- **Total de treinamentos:** {stats['total_trainings']:,}\n")
        f.write(f"- **Tamanho médio do treino:** {stats['avg_train_size']:.0f} interações\n\n")
        
        f.write("**Treinamentos por algoritmo:**\n")
        for algo in ['knnu', 'knni', 'svd']:
            count = stats['trainings_by_algo'][algo]
            pct = count / stats['total_trainings'] * 100 if stats['total_trainings'] > 0 else 0
            f.write(f"- **{algo.upper()}:** {count:,} treinos ({pct:.1f}%)\n")
        
        f.write("\n## 3. Otimização\n\n")
        f.write("Este pipeline implementa otimização inteligente:\n\n")
        f.write("- Agrupa checkpoints por `t_rec`\n")
        f.write("- Treina apenas algoritmos necessários por `t_rec`\n")
        f.write("- Cada usuário recebe predições do algoritmo atribuído\n")
        f.write("- Evita treinos desnecessários\n\n")
        
        max_trainings = stats['total_trecs'] * 3  # 3 algoritmos por t_rec
        saved = max_trainings - stats['total_trainings']
        f.write(f"**Economia de treinamentos:**\n")
        f.write(f"- Máximo possível (sem otimização): {max_trainings:,} treinos\n")
        f.write(f"- Executados (com otimização): {stats['total_trainings']:,} treinos\n")
        f.write(f"- **Economia: {saved:,} treinos ({saved/max_trainings*100:.1f}%)**\n\n")
        
        f.write("---\n\n")
        f.write("*Relatório gerado automaticamente pelo pipeline de treino/score*\n")
    
    print(f"Relatório salvo: {output_path}")


def main():
    """
    Função principal.
    """
    print("\n" + "█"*70)
    print(" "*10 + "PIPELINE: TRAIN & SCORE - REPLAY ASSIGNED")
    print("█"*70)
    
    # Carregar dados
    df_checkpoints, df_interactions, df_users = load_replay_data()
    
    # Parsear algoritmos
    df_user_algos = parse_user_algorithms(df_users)
    
    # Preparar checkpoints
    checkpoints_prepared = prepare_checkpoints(df_checkpoints, df_user_algos)
    
    # Agrupar por t_rec
    checkpoints_grouped = group_checkpoints_by_trec(checkpoints_prepared)
    
    # Processar em batch
    stats = process_checkpoints_batch(checkpoints_grouped, df_interactions)
    
    # Gerar relatório
    generate_train_score_report(stats)
    
    print("\n" + "█"*70)
    print(" "*20 + "PIPELINE CONCLUÍDO COM SUCESSO!")
    print("█"*70 + "\n")


if __name__ == '__main__':
    main()
