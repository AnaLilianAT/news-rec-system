# -*- coding: utf-8 -*-
"""
Gera graficos do sweep dimensao x seed, um por algoritmo.

Cada grafico mostra:
- Eixo Y esquerdo: Ranking metric (RMSE ou NDCG) mean + banda (std ou IC95)
- Eixo Y direito: ILS_mean + banda (std ou IC95)
- Eixo X: dimensao do embedding (d)

Output: 1 PNG por algoritmo
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def sanitize_filename(name: str) -> str:
    """Sanitiza nome de algoritmo para usar como filename."""
    return name.replace(' ', '_').replace('/', '_').replace('\\', '_')


def plot_algorithm(df_algo, algorithm, band_type='ci95', output_path=None, 
                  ranking_avg=None, ils_avg=None, ranking_metric='rmse', ndcg_cutoff=20):
    """
    Plota resultados de um algoritmo com dois eixos Y.
    
    Args:
        df_algo: DataFrame filtrado para o algoritmo
        algorithm: Nome do algoritmo
        band_type: 'ci95', 'std', ou 'none'
        output_path: Path para salvar PNG
        ranking_avg: Média geral da métrica de ranking para linha tracejada (opcional)
        ils_avg: Média geral do ILS para linha tracejada (opcional)
        ranking_metric: 'rmse' ou 'ndcg'
        ndcg_cutoff: Cutoff N para NDCG@N
    """
    # Ordenar por dimensao
    df_algo = df_algo.sort_values('d')
    
    # Criar figura e eixos
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Cores
    color_ranking = '#d62728'  # vermelho
    color_ils = '#2ca02c'       # verde
    
    # Nomes de colunas dinâmicos
    metric_col = ranking_metric
    mean_col = f'{metric_col}_mean'
    std_col = f'{metric_col}_std'
    ci95_low_col = f'{metric_col}_ci95_low'
    ci95_high_col = f'{metric_col}_ci95_high'
    
    # Label da métrica
    if ranking_metric == 'ndcg':
        metric_label = f'NDCG@{ndcg_cutoff}'
    else:
        metric_label = 'RMSE'
    
    # ===== EIXO ESQUERDO: RANKING METRIC =====
    ax1.set_xlabel('Dimensao do Embedding (d)', fontsize=12)
    ax1.set_ylabel(metric_label, fontsize=12, color=color_ranking)
    ax1.tick_params(axis='y', labelcolor=color_ranking)
    
    # Linha principal da métrica
    line1 = ax1.plot(df_algo['d'], df_algo[mean_col], 
                     color=color_ranking, marker='o', linewidth=2, 
                     markersize=6, label=metric_label)
    
    # Linha tracejada da média (se fornecida)
    if ranking_avg is not None:
        ax1.axhline(y=ranking_avg, color=color_ranking, linestyle='--', 
                    linewidth=2, alpha=0.7, label=f'{metric_label} binary features')
    
    # Banda da métrica
    if band_type == 'ci95' and ci95_low_col in df_algo.columns:
        ax1.fill_between(df_algo['d'], 
                         df_algo[ci95_low_col], 
                         df_algo[ci95_high_col],
                         color=color_ranking, alpha=0.2, label=f'{metric_label} IC95')
    elif band_type == 'std' and std_col in df_algo.columns:
        low = df_algo[mean_col] - df_algo[std_col]
        high = df_algo[mean_col] + df_algo[std_col]
        ax1.fill_between(df_algo['d'], low, high,
                         color=color_ranking, alpha=0.2, label=f'{metric_label} +/- std')
    
    # ===== EIXO DIREITO: ILS =====
    ax2 = ax1.twinx()
    ax2.set_ylabel('ILS (diversidade)', fontsize=12, color=color_ils)
    ax2.tick_params(axis='y', labelcolor=color_ils)
    
    # Linha principal ILS
    line2 = ax2.plot(df_algo['d'], df_algo['ils_mean'], 
                     color=color_ils, marker='s', linewidth=2, 
                     markersize=6, label='ILS')
    
    # Linha tracejada ILS média (se fornecida)
    if ils_avg is not None:
        ax2.axhline(y=ils_avg, color=color_ils, linestyle='--', 
                    linewidth=2, alpha=0.7, label='ILS binary features')
    
    # Banda ILS
    if band_type == 'ci95' and 'ils_ci95_low' in df_algo.columns:
        ax2.fill_between(df_algo['d'], 
                         df_algo['ils_ci95_low'], 
                         df_algo['ils_ci95_high'],
                         color=color_ils, alpha=0.2, label='ILS IC95')
    elif band_type == 'std' and 'ils_std' in df_algo.columns:
        ils_low = df_algo['ils_mean'] - df_algo['ils_std']
        ils_high = df_algo['ils_mean'] + df_algo['ils_std']
        ax2.fill_between(df_algo['d'], ils_low, ils_high,
                         color=color_ils, alpha=0.2, label='ILS +/- std')
    
    # ===== TITULO E LEGENDA =====
    #title = f'Algoritmo: {algorithm.upper()}'
    #plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Coletar todos os handles e labels dos dois eixos
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    # Combinar sem duplicatas
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    
    # Criar legenda com todos os elementos
    ax1.legend(all_handles, all_labels, loc='upper left', framealpha=0.9)
    
    # Grid e layout
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    fig.tight_layout()
    
    # Salvar
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] {output_path.name}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plota resultados do sweep por algoritmo'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='outputs/experiments/embedding_dim_seed_sweep_agg.parquet',
        help='Path do parquet agregado'
    )
    
    parser.add_argument(
        '--band',
        type=str,
        choices=['ci95', 'std', 'none'],
        default='ci95',
        help='Tipo de banda: ci95, std, ou none (default: ci95)'
    )
    
    parser.add_argument(
        '--outdir',
        type=str,
        default='outputs/plots/embedding_dim_seed_sweep',
        help='Diretorio de output para PNGs'
    )
    
    parser.add_argument(
        '--algorithms',
        type=str,
        nargs='+',
        help='Lista de algoritmos especificos (default: todos)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Mostrar plots ao inves de salvar'
    )
    
    parser.add_argument(
        '--ranking-table',
        type=str,
        default='outputs/tabela_6_3_RMSE_bin_features+bin_topics.csv',
        help='Path da tabela CSV com médias de ranking metric por algoritmo'
    )
    
    parser.add_argument(
        '--ranking-metric',
        type=str,
        choices=['rmse', 'ndcg'],
        default='rmse',
        help='Métrica de ranking: rmse ou ndcg (default: rmse)'
    )
    
    parser.add_argument(
        '--ndcg-cutoff',
        type=int,
        default=20,
        help='Cutoff N para NDCG@N (default: 20)'
    )
    
    parser.add_argument(
        '--ils-table',
        type=str,
        default='outputs/tabela_6_6_ILS_listas_bin_features+bin_topics.csv',
        help='Path da tabela CSV com médias de ILS por algoritmo'
    )
    
    args = parser.parse_args()
    
    # Validar input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[X] Erro: Arquivo nao encontrado: {input_path}")
        return 1
    
    # Carregar tabelas de médias
    ranking_averages = {}
    ils_averages = {}
    
    # Determinar coluna de ranking na tabela
    if args.ranking_metric == 'ndcg':
        metric_label = f'NDCG@{args.ndcg_cutoff}'
    else:
        metric_label = 'RMSE'
    
    # As tabelas de RMSE e NDCG têm a mesma estrutura, coluna 'Média'
    ranking_col_name = 'Média'
    
    ranking_table_path = Path(args.ranking_table)
    if ranking_table_path.exists():
        print(f"[>] Carregando médias de {metric_label}: {ranking_table_path}")
        df_ranking = pd.read_csv(ranking_table_path)
        # Mapear algoritmo -> média
        for _, row in df_ranking.iterrows():
            algo = row['Algoritmo'].strip().lower()
            ranking_averages[algo] = row[ranking_col_name]
        print(f"[OK] {len(ranking_averages)} médias de {metric_label} carregadas")
    else:
        print(f"[!] Aviso: Tabela de {metric_label} não encontrada: {ranking_table_path}")
    
    ils_table_path = Path(args.ils_table)
    if ils_table_path.exists():
        print(f"[>] Carregando médias de ILS: {ils_table_path}")
        df_ils = pd.read_csv(ils_table_path)
        # Mapear algoritmo -> média (coluna 'Média')
        for _, row in df_ils.iterrows():
            algo = row['Algoritmo'].strip().lower()
            ils_averages[algo] = row['Média']
        print(f"[OK] {len(ils_averages)} médias de ILS carregadas")
    else:
        print(f"[!] Aviso: Tabela de ILS não encontrada: {ils_table_path}")
    
    # Carregar dados
    print("="*70)
    print(" PLOTS: EMBEDDING DIMENSION x SEED SWEEP")
    print("="*70)
    print(f"\n[>] Carregando: {input_path}")
    
    df = pd.read_parquet(input_path)
    print(f"[OK] {len(df)} combinacoes carregadas")
    print(f"    Algoritmos: {sorted(df['algorithm'].unique())}")
    print(f"    Dimensoes: {sorted(df['d'].unique())}")
    
    # Filtrar algoritmos
    if args.algorithms:
        df = df[df['algorithm'].isin(args.algorithms)]
        print(f"\n[>] Filtrado para algoritmos: {args.algorithms}")
    
    algorithms = sorted(df['algorithm'].unique())
    print(f"\n[>] Gerando {len(algorithms)} plots...")
    print(f"    Banda: {args.band}")
    print(f"    Output: {args.outdir}")
    
    # Gerar plots
    output_dir = Path(args.outdir)
    
    for i, algorithm in enumerate(algorithms, 1):
        df_algo = df[df['algorithm'] == algorithm].copy()
        
        if len(df_algo) == 0:
            print(f"[!] {algorithm}: sem dados, pulando")
            continue
        
        print(f"\n[{i}/{len(algorithms)}] {algorithm}: {len(df_algo)} pontos")
        
        # Buscar médias do algoritmo
        algo_key = algorithm.strip().lower()
        ranking_avg = ranking_averages.get(algo_key, None)
        ils_avg = ils_averages.get(algo_key, None)
        
        if ranking_avg is not None:
            print(f"  -> {metric_label} média: {ranking_avg:.3f}")
        if ils_avg is not None:
            print(f"  -> ILS média: {ils_avg:.3f}")
        
        if args.show:
            output_path = None
        else:
            filename = sanitize_filename(algorithm) + '.png'
            output_path = output_dir / filename
        
        try:
            plot_algorithm(df_algo, algorithm, args.band, output_path, 
                          ranking_avg, ils_avg, args.ranking_metric, args.ndcg_cutoff)
        except Exception as e:
            print(f"[X] Erro ao plotar {algorithm}: {e}")
            continue
    
    # Resumo final
    if not args.show:
        saved_files = list(output_dir.glob('*.png'))
        print("\n" + "="*70)
        print(" PLOTS CONCLUIDOS")
        print("="*70)
        print(f"\nTotal de arquivos salvos: {len(saved_files)}")
        print(f"Diretorio: {output_dir.absolute()}")
        
        if len(saved_files) > 0:
            print(f"\nArquivos gerados:")
            for f in sorted(saved_files):
                print(f"  - {f.name}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
