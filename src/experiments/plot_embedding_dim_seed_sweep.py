# -*- coding: utf-8 -*-
"""
Gera graficos do sweep dimensao x seed, um por algoritmo.

Cada grafico mostra:
- Eixo Y esquerdo: RMSE_mean + banda (std ou IC95)
- Eixo Y direito: GH_mean + banda (std ou IC95)
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


def plot_algorithm(df_algo, algorithm, band_type='ci95', output_path=None, rmse_avg=None, gh_avg=None):
    """
    Plota resultados de um algoritmo com dois eixos Y.
    
    Args:
        df_algo: DataFrame filtrado para o algoritmo
        algorithm: Nome do algoritmo
        band_type: 'ci95', 'std', ou 'none'
        output_path: Path para salvar PNG
        rmse_avg: Média geral do RMSE para linha tracejada (opcional)
        gh_avg: Média geral do GH para linha tracejada (opcional)
    """
    # Ordenar por dimensao
    df_algo = df_algo.sort_values('d')
    
    # Criar figura e eixos
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Cores
    color_rmse = '#d62728'  # vermelho
    color_gh = '#2ca02c'    # verde
    
    # ===== EIXO ESQUERDO: RMSE =====
    ax1.set_xlabel('Dimensao do Embedding (d)', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12, color=color_rmse)
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    
    # Linha principal RMSE
    line1 = ax1.plot(df_algo['d'], df_algo['rmse_mean'], 
                     color=color_rmse, marker='o', linewidth=2, 
                     markersize=6, label='RMSE')
    
    # Linha tracejada RMSE média (se fornecida)
    if rmse_avg is not None:
        ax1.axhline(y=rmse_avg, color=color_rmse, linestyle='--', 
                    linewidth=2, alpha=0.7, label='RMSE Média')
    
    # Banda RMSE
    if band_type == 'ci95' and 'rmse_ci95_low' in df_algo.columns:
        ax1.fill_between(df_algo['d'], 
                         df_algo['rmse_ci95_low'], 
                         df_algo['rmse_ci95_high'],
                         color=color_rmse, alpha=0.2, label='RMSE IC95')
    elif band_type == 'std' and 'rmse_std' in df_algo.columns:
        rmse_low = df_algo['rmse_mean'] - df_algo['rmse_std']
        rmse_high = df_algo['rmse_mean'] + df_algo['rmse_std']
        ax1.fill_between(df_algo['d'], rmse_low, rmse_high,
                         color=color_rmse, alpha=0.2, label='RMSE +/- std')
    
    # ===== EIXO DIREITO: GH =====
    ax2 = ax1.twinx()
    ax2.set_ylabel('GH (diversidade)', fontsize=12, color=color_gh)
    ax2.tick_params(axis='y', labelcolor=color_gh)
    
    # Linha principal GH
    line2 = ax2.plot(df_algo['d'], df_algo['gh_mean'], 
                     color=color_gh, marker='s', linewidth=2, 
                     markersize=6, label='GH')
    
    # Linha tracejada GH média (se fornecida)
    if gh_avg is not None:
        ax2.axhline(y=gh_avg, color=color_gh, linestyle='--', 
                    linewidth=2, alpha=0.7, label='GH Média')
    
    # Banda GH
    if band_type == 'ci95' and 'gh_ci95_low' in df_algo.columns:
        ax2.fill_between(df_algo['d'], 
                         df_algo['gh_ci95_low'], 
                         df_algo['gh_ci95_high'],
                         color=color_gh, alpha=0.2, label='GH IC95')
    elif band_type == 'std' and 'gh_std' in df_algo.columns:
        gh_low = df_algo['gh_mean'] - df_algo['gh_std']
        gh_high = df_algo['gh_mean'] + df_algo['gh_std']
        ax2.fill_between(df_algo['d'], gh_low, gh_high,
                         color=color_gh, alpha=0.2, label='GH +/- std')
    
    # ===== TITULO E LEGENDA =====
    title = f'Algoritmo: {algorithm.upper()}'
    if band_type != 'none':
        title += f' (banda: {band_type.upper()})'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
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
        '--rmse-table',
        type=str,
        default='outputs/tabela_6_3_RMSE_bin_features+bin_topics.csv',
        help='Path da tabela CSV com médias de RMSE por algoritmo'
    )
    
    parser.add_argument(
        '--gh-table',
        type=str,
        default='outputs/tabela_6_6_GH_listas_bin_features+bin_topics.csv',
        help='Path da tabela CSV com médias de GH por algoritmo'
    )
    
    args = parser.parse_args()
    
    # Validar input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[X] Erro: Arquivo nao encontrado: {input_path}")
        return 1
    
    # Carregar tabelas de médias
    rmse_averages = {}
    gh_averages = {}
    
    rmse_table_path = Path(args.rmse_table)
    if rmse_table_path.exists():
        print(f"[>] Carregando médias de RMSE: {rmse_table_path}")
        df_rmse = pd.read_csv(rmse_table_path)
        # Mapear algoritmo -> média
        for _, row in df_rmse.iterrows():
            algo = row['Algoritmo'].strip().lower()
            rmse_averages[algo] = row['Média']
        print(f"[OK] {len(rmse_averages)} médias de RMSE carregadas")
    else:
        print(f"[!] Aviso: Tabela de RMSE não encontrada: {rmse_table_path}")
    
    gh_table_path = Path(args.gh_table)
    if gh_table_path.exists():
        print(f"[>] Carregando médias de GH: {gh_table_path}")
        df_gh = pd.read_csv(gh_table_path)
        # Mapear algoritmo -> média
        for _, row in df_gh.iterrows():
            algo = row['Algoritmo'].strip().lower()
            gh_averages[algo] = row['Média']
        print(f"[OK] {len(gh_averages)} médias de GH carregadas")
    else:
        print(f"[!] Aviso: Tabela de GH não encontrada: {gh_table_path}")
    
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
        rmse_avg = rmse_averages.get(algo_key, None)
        gh_avg = gh_averages.get(algo_key, None)
        
        if rmse_avg is not None:
            print(f"  -> RMSE média: {rmse_avg:.3f}")
        if gh_avg is not None:
            print(f"  -> GH média: {gh_avg:.3f}")
        
        if args.show:
            output_path = None
        else:
            filename = sanitize_filename(algorithm) + '.png'
            output_path = output_dir / filename
        
        try:
            plot_algorithm(df_algo, algorithm, args.band, output_path, rmse_avg, gh_avg)
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
