# -*- coding: utf-8 -*-
"""
Agrega resultados do sweep dimensao x seed por (d, algorithm).

Entrada: outputs/experiments/embedding_dim_seed_sweep_runs.parquet
Saida: outputs/experiments/embedding_dim_seed_sweep_agg.parquet

Calcula media, std e IC95 para ranking metric (RMSE ou NDCG) e ILS por (d, algorithm).
IC95 = mean +/- 1.96 * (std / sqrt(n))
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import warnings


def calculate_ci95(mean: float, std: float, n: int) -> tuple:
    """
    Calcula intervalo de confianca 95%.
    
    Args:
        mean: Media da metrica
        std: Desvio padrao
        n: Numero de observacoes
    
    Returns:
        (ci95_low, ci95_high)
    """
    if n <= 0 or pd.isna(std):
        return (np.nan, np.nan)
    
    margin = 1.96 * (std / np.sqrt(n))
    return (mean - margin, mean + margin)


def aggregate_sweep_results(df_runs: pd.DataFrame, expected_n_seeds: int = 20) -> pd.DataFrame:
    """
    Agrega resultados por (d, algorithm).
    
    Args:
        df_runs: DataFrame de runs individuais
        expected_n_seeds: Numero esperado de seeds (default: 20)
    
    Returns:
        DataFrame agregado com estatisticas
    """
    print(f"\n[>] Agregando resultados...")
    print(f"    Total de runs: {len(df_runs)}")
    print(f"    Dimensoes: {sorted(df_runs['d'].unique())}")
    print(f"    Algoritmos: {sorted(df_runs['algorithm'].unique())}")
    
    # Detectar coluna de ranking metric
    if 'rmse' in df_runs.columns:
        ranking_col = 'rmse'
        print(f"    Métrica de ranking: RMSE")
    elif 'ndcg' in df_runs.columns:
        ranking_col = 'ndcg'
        print(f"    Métrica de ranking: NDCG")
    else:
        raise ValueError("DataFrame deve conter coluna 'rmse' ou 'ndcg'")
    
    # Agrupa por (d, algorithm)
    agg_stats = df_runs.groupby(['d', 'algorithm']).agg({
        'seed': 'count',  # n_seeds
        ranking_col: ['mean', 'std'],
        'ils_list': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    agg_stats.columns = ['d', 'algorithm', 'n_seeds', 
                         f'{ranking_col}_mean', f'{ranking_col}_std', 
                         'ils_mean', 'ils_std']
    
    # Validar n_seeds
    invalid_seeds = agg_stats[agg_stats['n_seeds'] != expected_n_seeds]
    if len(invalid_seeds) > 0:
        warnings.warn(
            f"\n[!] AVISO: {len(invalid_seeds)} combinacoes (d, algorithm) com n_seeds != {expected_n_seeds}:"
        )
        for _, row in invalid_seeds.iterrows():
            print(f"    d={int(row['d'])}, algorithm={row['algorithm']}: n_seeds={int(row['n_seeds'])}")
    
    # Calcular IC95 para ranking metric
    ci95_results = agg_stats.apply(
        lambda row: calculate_ci95(row[f'{ranking_col}_mean'], row[f'{ranking_col}_std'], row['n_seeds']),
        axis=1
    )
    agg_stats[f'{ranking_col}_ci95_low'] = [x[0] for x in ci95_results]
    agg_stats[f'{ranking_col}_ci95_high'] = [x[1] for x in ci95_results]
    
    # Calcular IC95 para ILS
    ci95_results_ils = agg_stats.apply(
        lambda row: calculate_ci95(row['ils_mean'], row['ils_std'], row['n_seeds']),
        axis=1
    )
    agg_stats['ils_ci95_low'] = [x[0] for x in ci95_results_ils]
    agg_stats['ils_ci95_high'] = [x[1] for x in ci95_results_ils]
    
    # Reordenar colunas
    agg_stats = agg_stats[[
        'd', 'algorithm', 'n_seeds',
        f'{ranking_col}_mean', f'{ranking_col}_std', f'{ranking_col}_ci95_low', f'{ranking_col}_ci95_high',
        'ils_mean', 'ils_std', 'ils_ci95_low', 'ils_ci95_high'
    ]]
    
    print(f"[OK] Agregacao completa: {len(agg_stats)} linhas")
    
    return agg_stats


def print_summary(df_agg: pd.DataFrame):
    """Imprime resumo das estatisticas agregadas."""
    print("\n" + "="*70)
    print(" RESUMO DA AGREGACAO")
    print("="*70)
    
    print(f"\nTotal de combinacoes (d, algorithm): {len(df_agg)}")
    print(f"Dimensoes: {sorted(df_agg['d'].unique())}")
    print(f"Algoritmos: {sorted(df_agg['algorithm'].unique())}")
    
    # Estatisticas de n_seeds
    print(f"\nDistribuicao de n_seeds:")
    print(df_agg['n_seeds'].value_counts().sort_index())
    
    # Detectar métrica (RMSE ou NDCG)
    if 'rmse_mean' in df_agg.columns:
        metric_col = 'rmse'
        metric_label = 'RMSE'
        minimize = True
    elif 'ndcg_mean' in df_agg.columns:
        metric_col = 'ndcg'
        metric_label = 'NDCG'
        minimize = False
    else:
        print("[!] Nenhuma métrica de ranking encontrada")
        return
    
    mean_col = f'{metric_col}_mean'
    std_col = f'{metric_col}_std'
    ci95_low_col = f'{metric_col}_ci95_low'
    ci95_high_col = f'{metric_col}_ci95_high'
    
    # Melhor algoritmo por dimensao
    print(f"\n{'-'*70}")
    if minimize:
        print(f"MELHOR ALGORITMO POR DIMENSAO (menor {metric_label} medio)")
    else:
        print(f"MELHOR ALGORITMO POR DIMENSAO (maior {metric_label} medio)")
    print(f"{'-'*70}")
    
    if minimize:
        best_per_dim = df_agg.loc[df_agg.groupby('d')[mean_col].idxmin()]
    else:
        best_per_dim = df_agg.loc[df_agg.groupby('d')[mean_col].idxmax()]
    
    for _, row in best_per_dim.iterrows():
        print(f"d={int(row['d']):2d}: {row['algorithm']:<12} "
              f"{metric_label}={row[mean_col]:.4f} +/- {row[std_col]:.4f} "
              f"(CI95: [{row[ci95_low_col]:.4f}, {row[ci95_high_col]:.4f}])")
    
    # Algoritmos mais estaveis (menor CV)
    print(f"\n{'-'*70}")
    print(f"ALGORITMOS MAIS ESTAVEIS (menor CV do {metric_label})")
    print(f"{'-'*70}")
    
    df_agg_cv = df_agg.copy()
    cv_col = f'{metric_col}_cv'
    df_agg_cv[cv_col] = df_agg_cv[std_col] / df_agg_cv[mean_col]
    most_stable = df_agg_cv.nsmallest(5, cv_col)
    
    for _, row in most_stable.iterrows():
        cv_pct = row[cv_col] * 100
        print(f"d={int(row['d']):2d}, {row['algorithm']:<12}: "
              f"{metric_label}={row[mean_col]:.4f} +/- {row[std_col]:.4f} "
              f"(CV={cv_pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Agrega resultados do sweep por (d, algorithm)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='outputs/experiments/embedding_dim_seed_sweep_runs.parquet',
        help='Path do parquet de runs'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/experiments/embedding_dim_seed_sweep_agg.parquet',
        help='Path do parquet agregado'
    )
    
    parser.add_argument(
        '--expected-n-seeds',
        type=int,
        default=20,
        help='Numero esperado de seeds por combinacao (default: 20)'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Nao imprime resumo detalhado'
    )
    
    args = parser.parse_args()
    
    # Validar input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[X] Erro: Arquivo nao encontrado: {input_path}")
        return 1
    
    # Carregar runs
    print("="*70)
    print(" AGREGACAO: EMBEDDING DIMENSION x SEED SWEEP")
    print("="*70)
    print(f"\n[>] Carregando: {input_path}")
    
    df_runs = pd.read_parquet(input_path)
    print(f"[OK] {len(df_runs)} runs carregadas")
    
    # Validar colunas necessarias
    required_cols = ['d', 'algorithm', 'seed', 'ils_list']
    if 'rmse' not in df_runs.columns and 'ndcg' not in df_runs.columns:
        print(f"[X] Erro: DataFrame deve conter coluna 'rmse' ou 'ndcg'")
        return 1
    
    # Detectar coluna de ranking
    ranking_col = 'rmse' if 'rmse' in df_runs.columns else 'ndcg'
    required_cols.append(ranking_col)
    
    missing_cols = [col for col in required_cols if col not in df_runs.columns]
    if missing_cols:
        print(f"[X] Erro: Colunas ausentes: {missing_cols}")
        return 1
    
    # Agregar
    df_agg = aggregate_sweep_results(df_runs, args.expected_n_seeds)
    
    # Salvar
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_parquet(output_path, index=False)
    
    print(f"\n[OK] Parquet agregado salvo: {output_path}")
    print(f"    Linhas: {len(df_agg)}")
    print(f"    Colunas: {list(df_agg.columns)}")
    
    # Resumo
    if not args.no_summary:
        print_summary(df_agg)
    
    # Estatisticas finais
    print("\n" + "="*70)
    print(" ESTATISTICAS FINAIS")
    print("="*70)
    
    # Detectar coluna de ranking
    if 'rmse_mean' in df_agg.columns:
        ranking_col = 'rmse'
        metric_label = 'RMSE'
        print(f"\n{metric_label}:")
        print(f"  Media global: {df_agg[f'{ranking_col}_mean'].mean():.4f}")
        print(f"  Melhor (min): {df_agg[f'{ranking_col}_mean'].min():.4f}")
        print(f"  Pior (max): {df_agg[f'{ranking_col}_mean'].max():.4f}")
    elif 'ndcg_mean' in df_agg.columns:
        ranking_col = 'ndcg'
        metric_label = 'NDCG'
        print(f"\n{metric_label}:")
        print(f"  Media global: {df_agg[f'{ranking_col}_mean'].mean():.4f}")
        print(f"  Melhor (max): {df_agg[f'{ranking_col}_mean'].max():.4f}")
        print(f"  Pior (min): {df_agg[f'{ranking_col}_mean'].min():.4f}")
    
    print(f"\nILS (diversidade):")
    print(f"  Media global: {df_agg['ils_mean'].mean():.4f}")
    print(f"  Melhor (max): {df_agg['ils_mean'].max():.4f}")
    print(f"  Pior (min): {df_agg['ils_mean'].min():.4f}")
    
    print("\n" + "="*70)
    print(" AGREGACAO CONCLUIDA")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
