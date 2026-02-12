# -*- coding: utf-8 -*-
"""
Analise de resultados do sweep dimensao x seed.

Gera estatisticas e visualizacoes da variabilidade de RMSE e GH
em funcao da dimensao do embedding e do algoritmo.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_sweep_results(parquet_path: str = 'outputs/experiments/embedding_dim_seed_sweep_runs.parquet') -> pd.DataFrame:
    """Carrega resultados do sweep."""
    path = Path(parquet_path)
    if not path.exists():
        print(f"[X] Arquivo nao encontrado: {path}")
        sys.exit(1)
    
    df = pd.read_parquet(path)
    print(f"[OK] Carregado: {len(df)} linhas")
    print(f"    Dimensoes: {sorted(df['d'].unique())}")
    print(f"    Seeds: {len(df['seed'].unique())} valores")
    print(f"    Algoritmos: {len(df['algorithm'].unique())} tipos")
    
    return df


def compute_variability_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estatisticas de variabilidade por (dimensao, algoritmo)."""
    stats = df.groupby(['d', 'algorithm']).agg({
        'rmse': ['mean', 'std', 'min', 'max', 'count'],
        'gh_list': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
    
    # Coeficiente de variacao (CV = std/mean)
    stats['rmse_cv'] = stats['rmse_std'] / stats['rmse_mean']
    stats['gh_list_cv'] = stats['gh_list_std'] / stats['gh_list_mean']
    
    return stats


def find_optimal_dimension(stats: pd.DataFrame, algorithm: str = 'knnu') -> pd.DataFrame:
    """Identifica dimensao otima para um algoritmo."""
    algo_stats = stats[stats['algorithm'] == algorithm].copy()
    
    # Score: minimizar RMSE medio e variabilidade
    # Normaliza RMSE e CV para [0, 1] e soma
    rmse_norm = (algo_stats['rmse_mean'] - algo_stats['rmse_mean'].min()) / \
                (algo_stats['rmse_mean'].max() - algo_stats['rmse_mean'].min())
    cv_norm = (algo_stats['rmse_cv'] - algo_stats['rmse_cv'].min()) / \
              (algo_stats['rmse_cv'].max() - algo_stats['rmse_cv'].min())
    
    algo_stats['score'] = rmse_norm + cv_norm
    algo_stats = algo_stats.sort_values('score')
    
    return algo_stats[['d', 'rmse_mean', 'rmse_std', 'rmse_cv', 'score']].head(5)


def compare_algorithms(stats: pd.DataFrame, d: int) -> pd.DataFrame:
    """Compara algoritmos para uma dimensao fixa."""
    dim_stats = stats[stats['d'] == d].copy()
    dim_stats = dim_stats.sort_values('rmse_mean')
    
    return dim_stats[['algorithm', 'rmse_mean', 'rmse_std', 'rmse_cv', 
                      'gh_list_mean', 'gh_list_std', 'gh_list_cv']]


def print_summary(df: pd.DataFrame):
    """Imprime resumo geral."""
    print("\n" + "="*70)
    print(" RESUMO DO SWEEP")
    print("="*70)
    
    print(f"\nTotal de execucoes: {len(df)}")
    print(f"Dimensoes testadas: {sorted(df['d'].unique())}")
    print(f"Seeds por dimensao: {df.groupby('d')['seed'].nunique().iloc[0]}")
    print(f"Algoritmos: {sorted(df['algorithm'].unique())}")
    
    print("\n" + "-"*70)
    print("ESTATISTICAS GLOBAIS (todos algoritmos, todas dims)")
    print("-"*70)
    print(f"RMSE medio: {df['rmse'].mean():.4f} +/- {df['rmse'].std():.4f}")
    print(f"GH medio: {df['gh_list'].mean():.4f} +/- {df['gh_list'].std():.4f}")
    print(f"Runtime medio: {df['runtime_sec'].mean():.1f}s")


def print_variability_table(stats: pd.DataFrame, top_n: int = 10):
    """Imprime tabela de variabilidade."""
    print("\n" + "="*70)
    print(f" TOP {top_n}: MENOR VARIABILIDADE (RMSE CV)")
    print("="*70)
    
    top_stable = stats.nsmallest(top_n, 'rmse_cv')
    
    print(f"\n{'Dim':<5} {'Algorithm':<12} {'RMSE':<12} {'CV':<8} {'Seeds':<7}")
    print("-"*70)
    for _, row in top_stable.iterrows():
        print(f"{int(row['d']):<5} {row['algorithm']:<12} "
              f"{row['rmse_mean']:.4f}+/-{row['rmse_std']:.4f}  "
              f"{row['rmse_cv']:.4f}  {int(row['rmse_count']):<7}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analisa resultados do sweep')
    parser.add_argument('--input', type=str, 
                       default='outputs/experiments/embedding_dim_seed_sweep_runs.parquet')
    parser.add_argument('--output-stats', type=str,
                       default='outputs/experiments/dim_seed_variability_stats.csv')
    parser.add_argument('--algorithm', type=str, default='knnu',
                       help='Algoritmo para analise de dimensao otima')
    parser.add_argument('--dimension', type=int, default=None,
                       help='Dimensao para comparacao de algoritmos')
    
    args = parser.parse_args()
    
    # 1. Carrega dados
    df = load_sweep_results(args.input)
    
    # 2. Resumo geral
    print_summary(df)
    
    # 3. Calcula estatisticas de variabilidade
    stats = compute_variability_stats(df)
    
    # 4. Salva estatisticas
    output_path = Path(args.output_stats)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_path, index=False)
    print(f"\n[OK] Estatisticas salvas: {output_path}")
    
    # 5. Tabela de variabilidade
    print_variability_table(stats, top_n=10)
    
    # 6. Dimensao otima para algoritmo
    print("\n" + "="*70)
    print(f" DIMENSOES OTIMAS PARA: {args.algorithm.upper()}")
    print("="*70)
    optimal = find_optimal_dimension(stats, args.algorithm)
    print(optimal.to_string(index=False))
    
    # 7. Comparacao de algoritmos
    if args.dimension:
        print("\n" + "="*70)
        print(f" COMPARACAO DE ALGORITMOS (d={args.dimension})")
        print("="*70)
        comparison = compare_algorithms(stats, args.dimension)
        print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()
