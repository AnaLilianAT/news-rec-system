"""
Script helper para consolidar métricas de múltiplas representações.

Lê os arquivos de métricas gerados com diferentes sufixos de representação
e cria um DataFrame consolidado com coluna 'representation' para comparação.
"""
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict


def consolidate_metrics(
    output_dir: Path = Path('outputs'),
    metric_type: str = 'rmse',
    output_file: str = None
) -> pd.DataFrame:
    """
    Consolida métricas de múltiplas representações em um único DataFrame.
    
    Args:
        output_dir: Diretório com arquivos de métricas
        metric_type: Tipo de métrica ('rmse', 'gh_cosine', 'gh_jaccard')
        output_file: Arquivo de saída (opcional)
    
    Returns:
        DataFrame consolidado com coluna 'representation'
    """
    # Padrão do arquivo de acordo com metric_type
    patterns = {
        'rmse': 'metrics_rmse_by_algorithm_assigned*.csv',
        'gh_cosine': 'metrics_gh_cosine_by_algorithm_assigned*.csv',
        'gh_jaccard': 'metrics_gh_jaccard_by_algorithm_assigned*.csv'
    }
    
    if metric_type not in patterns:
        raise ValueError(f"Tipo de métrica inválido: {metric_type}. "
                        f"Tipos válidos: {list(patterns.keys())}")
    
    pattern = patterns[metric_type]
    files = list(output_dir.glob(pattern))
    
    if not files:
        print(f"AVISO: Nenhum arquivo encontrado para padrão: {pattern}")
        return pd.DataFrame()
    
    print(f"\nConsolidando métricas: {metric_type}")
    print(f"Arquivos encontrados: {len(files)}")
    
    # Coletar DataFrames
    dfs = []
    
    for file in files:
        df = pd.read_csv(file)
        
        # Extrair nome da representação do filename
        filename = file.stem  # metrics_rmse_by_algorithm_assigned ou metrics_rmse_by_algorithm_assigned_XXX
        base_name = f'metrics_{metric_type}_by_algorithm_assigned'
        
        if filename == base_name:
            representation = 'bin_features+bin_topics'  # Default
        else:
            representation = filename.replace(f'{base_name}_', '')
        
        df['representation'] = representation
        dfs.append(df)
        
        print(f"  - {representation}: {len(df)} registros")
    
    # Consolidar
    df_consolidated = pd.concat(dfs, ignore_index=True)
    
    # Reordenar colunas (representation primeiro)
    cols = ['representation'] + [col for col in df_consolidated.columns if col != 'representation']
    df_consolidated = df_consolidated[cols]
    
    # Salvar se output_file especificado
    if output_file:
        output_path = output_dir / output_file
        df_consolidated.to_csv(output_path, index=False)
        print(f"\nConsolidado salvo: {output_path}")
    
    return df_consolidated


def main():
    """CLI para consolidar métricas."""
    parser = argparse.ArgumentParser(
        description='Consolida métricas de múltiplas representações em um único CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Diretório com arquivos de métricas (default: outputs)'
    )
    parser.add_argument(
        '--metric-type',
        type=str,
        choices=['rmse', 'gh_cosine', 'gh_jaccard', 'all'],
        default='all',
        help='Tipo de métrica a consolidar (default: all)'
    )
    
    args = parser.parse_args()
    output_path = Path(args.output_dir)
    
    print("=" * 70)
    print("  CONSOLIDAÇÃO DE MÉTRICAS POR REPRESENTAÇÃO")
    print("=" * 70)
    
    if args.metric_type == 'all':
        metrics_to_process = ['rmse', 'gh_cosine', 'gh_jaccard']
    else:
        metrics_to_process = [args.metric_type]
    
    for metric in metrics_to_process:
        output_file = f'metrics_{metric}_consolidated.csv'
        df = consolidate_metrics(
            output_dir=output_path,
            metric_type=metric,
            output_file=output_file
        )
        
        if not df.empty:
            print(f"\nResumo {metric}:")
            print(f"  - Total de registros: {len(df)}")
            print(f"  - Representações: {df['representation'].nunique()}")
            print(f"  - Algoritmos: {df['algorithm'].nunique() if 'algorithm' in df.columns else 'N/A'}")
    
    print("\n" + "=" * 70)
    print("  CONSOLIDAÇÃO CONCLUÍDA!")
    print("=" * 70)


if __name__ == '__main__':
    main()
