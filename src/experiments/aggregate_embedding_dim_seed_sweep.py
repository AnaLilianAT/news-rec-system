# -*- coding: utf-8 -*-
"""
Agrega resultados do sweep dimensao x seed por (d, algorithm).

Entrada: outputs/experiments/embedding_dim_seed_sweep_runs.parquet
Saida: outputs/experiments/embedding_dim_seed_sweep_agg.parquet

Calcula media, std e IC95 para RMSE e GH por (d, algorithm).
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
    
    # Agrupa por (d, algorithm)
    agg_stats = df_runs.groupby(['d', 'algorithm']).agg({
        'seed': 'count',  # n_seeds
        'rmse': ['mean', 'std'],
        'gh_list': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    agg_stats.columns = ['d', 'algorithm', 'n_seeds', 'rmse_mean', 'rmse_std', 
                         'gh_mean', 'gh_std']
    
    # Validar n_seeds
    invalid_seeds = agg_stats[agg_stats['n_seeds'] != expected_n_seeds]
    if len(invalid_seeds) > 0:
        warnings.warn(
            f"\n[!] AVISO: {len(invalid_seeds)} combinacoes (d, algorithm) com n_seeds != {expected_n_seeds}:"
        )
        for _, row in invalid_seeds.iterrows():
            print(f"    d={int(row['d'])}, algorithm={row['algorithm']}: n_seeds={int(row['n_seeds'])}")
    
    # Calcular IC95 para RMSE
    ci95_results = agg_stats.apply(
        lambda row: calculate_ci95(row['rmse_mean'], row['rmse_std'], row['n_seeds']),
        axis=1
    )
    agg_stats['rmse_ci95_low'] = [x[0] for x in ci95_results]
    agg_stats['rmse_ci95_high'] = [x[1] for x in ci95_results]
    
    # Calcular IC95 para GH
    ci95_results_gh = agg_stats.apply(
        lambda row: calculate_ci95(row['gh_mean'], row['gh_std'], row['n_seeds']),
        axis=1
    )
    agg_stats['gh_ci95_low'] = [x[0] for x in ci95_results_gh]
    agg_stats['gh_ci95_high'] = [x[1] for x in ci95_results_gh]
    
    # Reordenar colunas
    agg_stats = agg_stats[[
        'd', 'algorithm', 'n_seeds',
        'rmse_mean', 'rmse_std', 'rmse_ci95_low', 'rmse_ci95_high',
        'gh_mean', 'gh_std', 'gh_ci95_low', 'gh_ci95_high'
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
    
    # Melhor algoritmo por dimensao (menor RMSE medio)
    print(f"\n{'-'*70}")
    print("MELHOR ALGORITMO POR DIMENSAO (menor RMSE medio)")
    print(f"{'-'*70}")
    
    best_per_dim = df_agg.loc[df_agg.groupby('d')['rmse_mean'].idxmin()]
    for _, row in best_per_dim.iterrows():
        print(f"d={int(row['d']):2d}: {row['algorithm']:<12} "
              f"RMSE={row['rmse_mean']:.4f} +/- {row['rmse_std']:.4f} "
              f"(CI95: [{row['rmse_ci95_low']:.4f}, {row['rmse_ci95_high']:.4f}])")
    
    # Algoritmos mais estaveis (menor CV)
    print(f"\n{'-'*70}")
    print("ALGORITMOS MAIS ESTAVEIS (menor CV do RMSE)")
    print(f"{'-'*70}")
    
    df_agg_cv = df_agg.copy()
    df_agg_cv['rmse_cv'] = df_agg_cv['rmse_std'] / df_agg_cv['rmse_mean']
    most_stable = df_agg_cv.nsmallest(5, 'rmse_cv')
    
    for _, row in most_stable.iterrows():
        cv_pct = row['rmse_cv'] * 100
        print(f"d={int(row['d']):2d}, {row['algorithm']:<12}: "
              f"RMSE={row['rmse_mean']:.4f} +/- {row['rmse_std']:.4f} "
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
    required_cols = ['d', 'algorithm', 'seed', 'rmse', 'gh_list']
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
    
    print(f"\nRMSE:")
    print(f"  Media global: {df_agg['rmse_mean'].mean():.4f}")
    print(f"  Melhor (min): {df_agg['rmse_mean'].min():.4f}")
    print(f"  Pior (max): {df_agg['rmse_mean'].max():.4f}")
    
    print(f"\nGH (diversidade):")
    print(f"  Media global: {df_agg['gh_mean'].mean():.4f}")
    print(f"  Melhor (max): {df_agg['gh_mean'].max():.4f}")
    print(f"  Pior (min): {df_agg['gh_mean'].min():.4f}")
    
    print("\n" + "="*70)
    print(" AGREGACAO CONCLUIDA")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
