"""
Visualização dos resultados do sweep de dimensão de embedding.

Gera gráficos de linha mostrando como RMSE e GH (listas) variam com a
dimensão do embedding (d), para cada algoritmo.

Cada gráfico possui:
- Eixo X: dimensão do embedding (d)
- Eixo Y esquerdo: RMSE (menor é melhor)
- Eixo Y direito: GH por listas (maior = mais homogêneo)

Uso:
    python -m src.experiments.plot_embedding_dim_sweep
    python -m src.experiments.plot_embedding_dim_sweep --input outputs/my_sweep.parquet
    python -m src.experiments.plot_embedding_dim_sweep --format pdf
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import warnings
warnings.filterwarnings('ignore')


def load_sweep_results(input_path: Path) -> pd.DataFrame:
    """
    Carrega resultados do sweep de dimensão.
    
    Args:
        input_path: Path para arquivo parquet ou csv
    
    Returns:
        DataFrame com colunas [representation, d, algorithm, rmse, gh_list]
    
    Raises:
        FileNotFoundError: Se arquivo não existir
        ValueError: Se schema estiver incorreto
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")
    
    # Detectar formato e carregar
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    elif input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Formato não suportado: {input_path.suffix}. Use .parquet ou .csv")
    
    # Validar schema
    required_cols = ['d', 'algorithm', 'rmse', 'gh_list']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(
            f"Colunas faltando no arquivo: {missing_cols}\n"
            f"Colunas presentes: {df.columns.tolist()}\n"
            f"Schema esperado: {required_cols}"
        )
    
    print(f"✓ Resultados carregados de {input_path}")
    print(f"  Registros: {len(df)}")
    print(f"  Dimensões testadas: {sorted(df['d'].unique().tolist())}")
    print(f"  Algoritmos: {sorted(df['algorithm'].unique().tolist())}")
    
    return df


def plot_algorithm_metrics(
    df_algo: pd.DataFrame,
    algorithm: str,
    output_dir: Path,
    fmt: str = 'png'
):
    """
    Plota RMSE e GH vs dimensão para um algoritmo específico.
    
    Args:
        df_algo: DataFrame filtrado para um algoritmo
        algorithm: Nome do algoritmo
        output_dir: Diretório para salvar gráficos
        fmt: Formato de saída ('png', 'pdf', ou 'both')
    """
    # Ordenar por dimensão
    df_algo = df_algo.sort_values('d').copy()
    
    # Extrair dados
    dims = df_algo['d'].values
    rmse = df_algo['rmse'].values
    gh_list = df_algo['gh_list'].values
    
    # Criar figura com dois eixos Y
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plotar RMSE no eixo esquerdo (azul)
    color_rmse = 'tab:blue'
    line1 = ax1.plot(
        dims, rmse,
        linestyle='-',
        linewidth=1.5,
        color=color_rmse,
        label='RMSE'
    )
    
    # Calcular e plotar linha de tendência para RMSE
    z_rmse = np.polyfit(dims, rmse, 1)
    p_rmse = np.poly1d(z_rmse)
    trend_rmse = ax1.plot(
        dims, p_rmse(dims),
        linestyle='--',
        linewidth=2.5,
        color=color_rmse,
        alpha=0.7,
        label='RMSE trend'
    )
    
    ax1.set_xlabel('Embedding dimension (d)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold', color=color_rmse)
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    
    # Plotar GH no eixo direito (laranja)
    color_gh = 'tab:orange'
    line2 = ax2.plot(
        dims, gh_list,
        linestyle='-',
        linewidth=1.5,
        color=color_gh,
        label='GH (lists)'
    )
    
    # Calcular e plotar linha de tendência para GH
    z_gh = np.polyfit(dims, gh_list, 1)
    p_gh = np.poly1d(z_gh)
    trend_gh = ax2.plot(
        dims, p_gh(dims),
        linestyle='--',
        linewidth=2.5,
        color=color_gh,
        alpha=0.7,
        label='GH trend'
    )
    
    ax2.set_ylabel('GH (lists)', fontsize=12, fontweight='bold', color=color_gh)
    ax2.tick_params(axis='y', labelcolor=color_gh)
    
    # Título
    ax1.set_title(
        f'RMSE and GH vs Embedding Dimension\nAlgorithm: {algorithm}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Grid leve
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Combinar legendas dos dois eixos
    lines = line1 + trend_rmse + line2 + trend_gh
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.9, fontsize=10)
    
    # Ajustar layout
    fig.tight_layout()
    
    # Salvar em formato(s) solicitado(s)
    formats_to_save = []
    if fmt == 'both':
        formats_to_save = ['png', 'pdf']
    else:
        formats_to_save = [fmt]
    
    saved_files = []
    for file_fmt in formats_to_save:
        # Nome do arquivo: substituir espaços e caracteres especiais
        safe_algo_name = algorithm.replace(' ', '_').replace('/', '_')
        output_file = output_dir / f'{safe_algo_name}_rmse_gh_vs_d.{file_fmt}'
        
        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        saved_files.append(output_file)
    
    plt.close(fig)
    
    # Reportar arquivos salvos
    for fpath in saved_files:
        print(f"  ✓ Salvo: {fpath.name} ({fpath.stat().st_size / 1024:.1f} KB)")


def generate_plots(
    input_path: str = 'outputs/experiments/embedding_dim_sweep.parquet',
    output_dir: str = 'outputs/plots/embedding_dim_sweep',
    fmt: str = 'png',
    verbose: bool = True
) -> int:
    """
    Gera todos os gráficos a partir dos resultados do sweep.
    
    Função programática que pode ser chamada por outros scripts.
    
    Args:
        input_path: Path do arquivo de resultados
        output_dir: Diretório para salvar gráficos
        fmt: Formato dos gráficos ('png', 'pdf', ou 'both')
        verbose: Se True, imprime mensagens de progresso
    
    Returns:
        Número de gráficos gerados
    
    Raises:
        FileNotFoundError: Se arquivo de entrada não existir
        ValueError: Se schema estiver incorreto
    """
    if verbose:
        print("="*70)
        print("  VISUALIZAÇÃO DO SWEEP DE DIMENSÃO DE EMBEDDING")
        print("="*70)
    
    # Carregar dados
    input_path_obj = Path(input_path)
    df = load_sweep_results(input_path_obj)
    
    # Preparar diretório de saída
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"\nDiretório de saída: {output_dir_obj}")
    
    # Obter lista de algoritmos
    algorithms = sorted(df['algorithm'].unique().tolist())
    if verbose:
        print(f"\nGerando gráficos para {len(algorithms)} algoritmos...")
    
    # Contador de gráficos gerados
    plots_generated = 0
    
    # Gerar um gráfico por algoritmo
    for idx, algo in enumerate(algorithms, 1):
        if verbose:
            print(f"\n[{idx}/{len(algorithms)}] {algo}")
        
        # Filtrar dados do algoritmo
        df_algo = df[df['algorithm'] == algo].copy()
        
        # Verificar se há dados suficientes
        if len(df_algo) < 2:
            if verbose:
                print(f"  ⚠ Apenas {len(df_algo)} ponto(s), pulando...")
            continue
        
        # Gerar gráfico
        plot_algorithm_metrics(
            df_algo=df_algo,
            algorithm=algo,
            output_dir=output_dir_obj,
            fmt=fmt
        )
        plots_generated += 1
    
    if verbose:
        print("\n" + "="*70)
        print("  VISUALIZAÇÃO CONCLUÍDA")
        print("="*70)
        print(f"Gráficos salvos em: {output_dir_obj}")
        print(f"Total de arquivos: {len(list(output_dir_obj.glob('*')))} ")
        
        # Listar arquivos gerados
        print("\nArquivos gerados:")
        for fpath in sorted(output_dir_obj.glob('*')):
            print(f"  - {fpath.name}")
        
        print("\n✓ Concluído com sucesso!")
    
    return plots_generated


def main():
    """Entrypoint do script de visualização."""
    parser = argparse.ArgumentParser(
        description='Visualização dos resultados do sweep de dimensão de embedding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='outputs/experiments/embedding_dim_sweep.parquet',
        help='Path do arquivo de resultados (parquet ou csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/plots/embedding_dim_sweep',
        help='Diretório para salvar gráficos (default: outputs/plots/embedding_dim_sweep)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['png', 'pdf', 'both'],
        default='png',
        help='Formato dos gráficos (default: png)'
    )
    
    args = parser.parse_args()
    
    # Chamar função programática
    generate_plots(
        input_path=args.input,
        output_dir=args.output_dir,
        fmt=args.format,
        verbose=True
    )


if __name__ == '__main__':
    main()
