# -*- coding: utf-8 -*-
"""
Análise do trade-off entre NDCG e ILS (diversidade).

Lê os resultados agregados do sweep de dimensão x seed e calcula o trade-off
entre acurácia (NDCG) e diversidade (ILS) para diferentes valores de alpha.

Para cada algoritmo:
1. Normaliza NDCG e ILS usando min-max normalization (escala 0-1)
2. Calcula trade-off: NDCG_norm(d) - α * ILS_norm(d)
3. Encontra dimensão ótima que maximiza o trade-off para cada α
4. Gera tabela com resultados por valor de α

Uso:
    python src/experiments/analyze_ndcg_gh_tradeoff.py
    python src/experiments/analyze_ndcg_gh_tradeoff.py --input outputs/experiments/embedding_dim_seed_sweep_agg.parquet
    python src/experiments/analyze_ndcg_gh_tradeoff.py --alphas 0.0 0.25 0.5 0.75 1.0
"""

import sys
from pathlib import Path

# Adicionar raiz do projeto ao path para imports absolutos
root_dir = str(Path(__file__).resolve().parents[2])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import argparse
import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


def normalize_metric(values: pd.Series) -> pd.Series:
    """
    Normaliza métrica usando min-max normalization para escala [0, 1].
    
    normalized(x) = (x - min(x)) / (max(x) - min(x))
    
    Args:
        values: Série com valores da métrica
    
    Returns:
        Série com valores normalizados em [0, 1]
    """
    min_val = values.min()
    max_val = values.max()
    
    # Evitar divisão por zero se todos os valores forem iguais
    if max_val == min_val:
        return pd.Series([0.5] * len(values), index=values.index)
    
    normalized = (values - min_val) / (max_val - min_val)
    return normalized


def calculate_tradeoff(
    df_algo: pd.DataFrame,
    alphas: List[float],
    metric_col: str = 'ndcg_mean',
    diversity_col: str = 'ils_mean',
    metric_label: str = 'NDCG@20'
) -> pd.DataFrame:
    """
    Calcula trade-off entre métrica de ranking e diversidade (ILS) para um algoritmo.
    
    Para cada valor de alpha, encontra a dimensão d* que maximiza:
        trade-off ponderado(d) = (1 - alpha) * metric_norm(d) - alpha * diversity_norm(d)
    
    Args:
        df_algo: DataFrame filtrado para um algoritmo com colunas 'd', metric_col, diversity_col
        alphas: Lista de valores de alpha para testar
        metric_col: Nome da coluna da métrica de ranking (ex: 'ndcg_mean')
        diversity_col: Nome da coluna de diversidade (ex: 'ils_mean')
        metric_label: Label da métrica para nome da coluna (ex: 'NDCG@20', 'RMSE')
    
    Returns:
        DataFrame com colunas:
            - alpha: Valor de alpha
            - tradeoff: Valor do trade-off na dimensão ótima
            - [metric_label]: Valor original da métrica na dimensão ótima
            - ILS: Valor original da diversidade na dimensão ótima
            - dim_optimal: Dimensão ótima que maximiza trade-off
    """
    # Ordenar por dimensão
    df_algo = df_algo.sort_values('d').copy()
    
    # Normalizar métricas (min-max para [0, 1])
    df_algo['metric_norm'] = normalize_metric(df_algo[metric_col])
    df_algo['diversity_norm'] = normalize_metric(df_algo[diversity_col])
    
    results = []
    
    for alpha in alphas:
        # Calcular trade-off ponderado para cada dimensão
        df_algo['tradeoff'] = (1 - alpha) * df_algo['metric_norm'] - alpha * df_algo['diversity_norm']
        
        # Encontrar dimensão que maximiza trade-off
        idx_optimal = df_algo['tradeoff'].idxmax()
        row_optimal = df_algo.loc[idx_optimal]
        
        results.append({
            'alpha': alpha,
            'tradeoff': row_optimal['tradeoff'],
            metric_label: row_optimal[metric_col],
            'ILS': row_optimal[diversity_col],
            'dim_optimal': int(row_optimal['d'])
        })
    
    return pd.DataFrame(results)


def analyze_tradeoff(
    input_path: str,
    alphas: List[float],
    output_dir: str = 'outputs/experiments',
    ndcg_cutoff: int = 20,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Analisa trade-off entre NDCG e ILS para todos os algoritmos.
    
    Args:
        input_path: Path do arquivo parquet agregado
        alphas: Lista de valores de alpha
        output_dir: Diretório para salvar resultados
        ndcg_cutoff: Valor de N para NDCG@N (default: 20)
        verbose: Se True, imprime progresso
    
    Returns:
        Dicionário {algoritmo: DataFrame de trade-off}
    """
    if verbose:
        print("="*70)
        print(" ANÁLISE DE TRADE-OFF: NDCG vs ILS")
        print("="*70)
    
    # Carregar dados
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")
    
    if verbose:
        print(f"\n[>] Carregando: {input_path}")
    
    df = pd.read_parquet(input_path)
    
    # Detectar métrica de ranking
    if 'ndcg_mean' in df.columns:
        metric_col = 'ndcg_mean'
        metric_label = f'NDCG@{ndcg_cutoff}'
        metric_print = f'NDCG@{ndcg_cutoff}'
    elif 'rmse_mean' in df.columns:
        metric_col = 'rmse_mean'
        metric_label = 'RMSE'
        metric_print = 'RMSE'
    else:
        raise ValueError("Nenhuma métrica de ranking encontrada (ndcg_mean ou rmse_mean)")
    
    diversity_col = 'ils_mean'
    
    if verbose:
        print(f"[OK] {len(df)} combinações carregadas")
        print(f"    Métrica: {metric_print}")
        print(f"    Diversidade: ILS")
        print(f"    Algoritmos: {sorted(df['algorithm'].unique())}")
        print(f"    Dimensões: {sorted(df['d'].unique())}")
    
    # Analisar cada algoritmo
    algorithms = sorted(df['algorithm'].unique())
    results_by_algo = {}
    
    if verbose:
        print(f"\n[>] Calculando trade-off para {len(algorithms)} algoritmos...")
        print(f"    Valores de alpha: {alphas}")
    
    for i, algorithm in enumerate(algorithms, 1):
        if verbose:
            print(f"\n[{i}/{len(algorithms)}] {algorithm}")
        
        # Filtrar dados do algoritmo
        df_algo = df[df['algorithm'] == algorithm].copy()
        
        if len(df_algo) < 2:
            if verbose:
                print(f"  [!] Apenas {len(df_algo)} ponto(s), pulando...")
            continue
        
        # Calcular trade-off
        df_tradeoff = calculate_tradeoff(
            df_algo=df_algo,
            alphas=alphas,
            metric_col=metric_col,
            diversity_col=diversity_col,
            metric_label=metric_label
        )
        
        results_by_algo[algorithm] = df_tradeoff
        
        if verbose:
            print(f"  [OK] {len(df_tradeoff)} valores de alpha analisados")
            # Mostrar resumo
            for _, row in df_tradeoff.iterrows():
                print(f"    α={row['alpha']:.2f}: d*={int(row['dim_optimal']):2d}, "
                      f"{metric_print}={row[metric_label]:.4f}, "
                      f"ILS={row['ILS']:.4f}, "
                      f"trade-off={row['tradeoff']:.4f}")
    
    # Salvar resultados
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n[>] Salvando tabelas de trade-off...")
    
    for algorithm, df_tradeoff in results_by_algo.items():
        # Sanitizar nome do algoritmo para filename
        algo_safe = algorithm.replace(' ', '_').replace('/', '_')
        
        # Salvar CSV
        csv_path = output_dir_obj / f'tradeoff_{algo_safe}.csv'
        df_tradeoff.to_csv(csv_path, index=False, float_format='%.4f')
        
        if verbose:
            print(f"  [OK] {csv_path.name}")
    
    # Salvar arquivo consolidado (todos os algoritmos)
    if verbose:
        print(f"\n[>] Criando tabela consolidada...")
    
    df_consolidated = []
    for algorithm, df_tradeoff in results_by_algo.items():
        df_temp = df_tradeoff.copy()
        df_temp.insert(0, 'algorithm', algorithm)
        df_consolidated.append(df_temp)
    
    if df_consolidated:
        df_consolidated = pd.concat(df_consolidated, ignore_index=True)
        
        # Salvar CSV consolidado
        csv_consolidated_path = output_dir_obj / 'tradeoff_all_algorithms.csv'
        df_consolidated.to_csv(csv_consolidated_path, index=False, float_format='%.4f')
        
        if verbose:
            print(f"  [OK] {csv_consolidated_path.name}")
        
        # Salvar parquet consolidado
        parquet_consolidated_path = output_dir_obj / 'tradeoff_all_algorithms.parquet'
        df_consolidated.to_parquet(parquet_consolidated_path, index=False)
        
        if verbose:
            print(f"  [OK] {parquet_consolidated_path.name}")
    
    if verbose:
        print("\n" + "="*70)
        print(" ANÁLISE CONCLUÍDA")
        print("="*70)
        print(f"\nResultados salvos em: {output_dir_obj.absolute()}")
        print(f"  - {len(results_by_algo)} tabelas por algoritmo")
        print(f"  - 1 tabela consolidada")
        
        # Resumo: dimensão ótima mais comum por alpha
        if df_consolidated is not None and len(df_consolidated) > 0:
            print("\n" + "-"*70)
            print("DIMENSÃO ÓTIMA MAIS COMUM POR ALPHA (MODA)")
            print("-"*70)
            for alpha in alphas:
                df_alpha = df_consolidated[df_consolidated['alpha'] == alpha]
                if len(df_alpha) > 0:
                    mode_d = df_alpha['dim_optimal'].mode()
                    if len(mode_d) > 0:
                        mode_d = int(mode_d.iloc[0])
                        count = (df_alpha['dim_optimal'] == mode_d).sum()
                        print(f"α={alpha:.2f}: d*={mode_d:2d} ({count}/{len(df_alpha)} algoritmos)")
    
    return results_by_algo


def main():
    parser = argparse.ArgumentParser(
        description='Análise de trade-off entre NDCG e ILS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='outputs/experiments/embedding_dim_seed_sweep_agg.parquet',
        help='Path do parquet agregado (default: outputs/experiments/embedding_dim_seed_sweep_agg.parquet)'
    )
    
    parser.add_argument(
        '--alphas',
        type=float,
        nargs='+',
        default=[0.00, 0.25, 0.50, 0.75, 1.00],
        help='Valores de alpha para trade-off (default: 0.00 0.25 0.50 0.75 1.00)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/experiments',
        help='Diretório para salvar resultados (default: outputs/experiments)'
    )
    
    parser.add_argument(
        '--ndcg-cutoff',
        type=int,
        default=20,
        help='Valor de N para NDCG@N (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Validar alphas
    for alpha in args.alphas:
        if alpha < 0 or alpha > 1:
            print(f"[!] Aviso: Alpha {alpha} fora do intervalo [0, 1]")
    
    # Executar análise
    try:
        analyze_tradeoff(
            input_path=args.input,
            alphas=sorted(args.alphas),
            output_dir=args.output_dir,
            ndcg_cutoff=args.ndcg_cutoff,
            verbose=True
        )
        return 0
    except Exception as e:
        print(f"\n[X] Erro: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
