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


def plot_algorithm(df_algo, algorithm, band_type='ci95', output_path=None):
    """
    Plota resultados de um algoritmo com dois eixos Y.
    
    Args:
        df_algo: DataFrame filtrado para o algoritmo
        algorithm: Nome do algoritmo
        band_type: 'ci95', 'std', ou 'none'
        output_path: Path para salvar PNG
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
                     markersize=6, linestyle='--', label='GH')
    
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
    
    # Combinar legendas dos dois eixos
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    
    # Adicionar bandas na legenda se existirem
    if band_type == 'ci95':
        labels.extend(['RMSE IC95', 'GH IC95'])
    elif band_type == 'std':
        labels.extend(['RMSE +/- std', 'GH +/- std'])
    
    ax1.legend(lines, labels, loc='upper left', framealpha=0.9)
    
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
    
    args = parser.parse_args()
    
    # Validar input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[X] Erro: Arquivo nao encontrado: {input_path}")
        return 1
    
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
        
        if args.show:
            output_path = None
        else:
            filename = sanitize_filename(algorithm) + '.png'
            output_path = output_dir / filename
        
        try:
            plot_algorithm(df_algo, algorithm, args.band, output_path)
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
