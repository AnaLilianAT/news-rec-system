"""
Sweep de dimensão de embedding para análise de trade-offs.

Este script realiza um experimento de varredura (sweep) de diferentes dimensões
de embedding do autoencoder, rodando o pipeline completo para cada dimensão e
coletando métricas RMSE e GH por algoritmo.

Heurística de d_min:
    Usamos d_min = max(4, round(log2(D_bin) * 2))
    Justificativa: Queremos capturar informação suficiente sem redundância.
    log2(D_bin) representa a "entropia informacional" aproximada da dimensão
    binária. Multiplicamos por 2 para dar margem de representação e garantimos
    pelo menos 4 dimensões para evitar embeddings triviais.
    
Exemplo para D_bin=99 (83 features + 16 tópicos):
    d_min = max(4, round(log2(99)*2)) = max(4, round(6.63*2)) = max(4, 13) = 13

Uso:
    python -m src.experiments.embedding_dim_sweep
    python -m src.experiments.embedding_dim_sweep --dmin 8 --dmax 64
    python -m src.experiments.embedding_dim_sweep --dims 8 16 32 64
    python -m src.experiments.embedding_dim_sweep --force
"""

import argparse
import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_d_min_heuristic(d_bin: int) -> int:
    """
    Calcula d_min baseado em heurística informacional.
    
    Heurística: d_min = max(4, round(log2(D_bin) * 2))
    
    Justificativa:
    - log2(D_bin) representa a "entropia informacional" aproximada
    - Multiplicar por 2 dá margem para capturar relações não-lineares
    - Mínimo de 4 evita embeddings triviais demais
    
    Args:
        d_bin: Dimensão total do vetor binário (features + topics)
    
    Returns:
        Dimensão mínima recomendada
    
    Examples:
        >>> compute_d_min_heuristic(99)  # 83 features + 16 topics
        13
        >>> compute_d_min_heuristic(200)
        16
        >>> compute_d_min_heuristic(10)
        7
    """
    if d_bin <= 0:
        raise ValueError(f"d_bin deve ser positivo, recebido: {d_bin}")
    
    d_min = max(4, round(np.log2(d_bin) * 2))
    return d_min


def get_binary_dim() -> int:
    """
    Detecta a dimensão total do vetor binário (features + topics).
    
    Returns:
        Soma de dimensões de canonical_features + canonical_topics
    
    Raises:
        FileNotFoundError: Se arquivos canônicos não existirem
    """
    outputs_dir = Path('outputs')
    
    features_path = outputs_dir / 'canonical_features.parquet'
    topics_path = outputs_dir / 'canonical_topics.parquet'
    
    if not features_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {features_path}\n"
            "Execute 'python -m src.build_canonical_tables' primeiro"
        )
    
    if not topics_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {topics_path}\n"
            "Execute 'python -m src.build_canonical_tables' primeiro"
        )
    
    # Carregar e contar colunas (exceto news_id)
    df_features = pd.read_parquet(features_path)
    
    n_features = len([c for c in df_features.columns if c != 'news_id'])
    
    d_bin = n_features
    
    print(f"Dimensão binária detectada:")
    print(f"  - Total (D_bin): {d_bin}")
    
    return d_bin


def generate_dims_sequence(d_min: int, d_max: int) -> List[int]:
    """
    Gera sequência completa de dimensões para o sweep.
    
    Estratégia:
    - Incrementar de 1 em 1 de d_min até d_max (inclusive)
    
    Args:
        d_min: Dimensão mínima (calculada por heurística)
        d_max: Dimensão máxima (D_bin)
    
    Returns:
        Lista de dimensões de d_min até d_max
    
    Examples:
        >>> generate_dims_sequence(13, 16)
        [13, 14, 15, 16]
        >>> generate_dims_sequence(8, 10)
        [8, 9, 10]
    """
    if d_min > d_max:
        raise ValueError(f"d_min ({d_min}) não pode ser maior que d_max ({d_max})")
    
    # Sequência simples: d_min, d_min+1, ..., d_max
    dims = list(range(d_min, d_max + 1))
    
    return dims


def train_embeddings(d: int, force: bool = False) -> Tuple[bool, str]:
    """
    Treina embeddings para dimensão d (ou reutiliza cache).
    
    Args:
        d: Dimensão do embedding
        force: Se True, força retreinamento mesmo se cache existir
    
    Returns:
        Tupla (sucesso, mensagem)
    """
    print(f"\n{'='*70}")
    print(f"  TREINAMENTO DE EMBEDDINGS (d={d})")
    print(f"{'='*70}")
    
    # Verificar se cache já existe (a menos de --force)
    embeddings_dir = Path('outputs/embeddings')
    ae_features_path = embeddings_dir / f'ae_features_dim{d}.parquet'
    ae_topics_path = embeddings_dir / f'ae_topics_dim{d}.parquet'
    
    if not force and ae_features_path.exists() and ae_topics_path.exists():
        print(f"[OK] Cache válido encontrado para d={d}, pulando treinamento")
        print(f"  - {ae_features_path}")
        print(f"  - {ae_topics_path}")
        return True, f"Cache válido para d={d}"
    
    # Treinar embeddings
    cmd = [
        sys.executable, '-m', 'src.embeddings.train_embeddings',
        '--embedding-dim', str(d)
    ]
    
    if force:
        cmd.append('--force')
    
    print(f"Executando: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return True, f"Embeddings treinados com sucesso para d={d}"
    
    except subprocess.CalledProcessError as e:
        error_msg = f"Erro ao treinar embeddings para d={d}:\n{e.stderr}"
        print(f"✗ {error_msg}")
        return False, error_msg


def run_pipeline_for_dim(d: int) -> Tuple[bool, str]:
    """
    Roda pipeline para dimensão d usando representação de embeddings.
    
    Pipeline:
    1. generate_reclists (ae_features+ae_topics)
    2. eval_replay
    3. export_thesis_tables
    
    Args:
        d: Dimensão do embedding
    
    Returns:
        Tupla (sucesso, mensagem)
    """
    print(f"\n{'='*70}")
    print(f"  RODANDO PIPELINE (d={d})")
    print(f"{'='*70}")
    
    # 1. Generate reclists
    print("\n[1/3] Gerando listas de recomendação...")
    cmd_reclists = [
        sys.executable, '-m', 'src.run_generate_reclists_assigned',
        '--representations', 'ae_features', 'ae_topics',
        '--embedding-dim', str(d)
    ]
    
    try:
        result = subprocess.run(cmd_reclists, check=True, capture_output=True, text=True)
        print(f"[OK] Listas geradas para d={d}")
    except subprocess.CalledProcessError as e:
        error_msg = f"Erro em generate_reclists (d={d}):\n{e.stderr}"
        print(f"✗ {error_msg}")
        return False, error_msg
    
    # 2. Eval replay
    print("\n[2/3] Avaliando listas...")
    cmd_eval = [
        sys.executable, '-m', 'src.run_eval_replay_assigned'
    ]
    
    try:
        result = subprocess.run(cmd_eval, check=True, capture_output=True, text=True)
        print(f"[OK] Avaliação concluída para d={d}")
    except subprocess.CalledProcessError as e:
        error_msg = f"Erro em eval_replay (d={d}):\n{e.stderr}"
        print(f"✗ {error_msg}")
        return False, error_msg
    
    # 3. Export thesis tables
    print("\n[3/3] Exportando tabelas formatadas...")
    cmd_export = [
        sys.executable, '-m', 'src.run_export_thesis_tables',
        '--embedding-dim', str(d)
    ]
    
    try:
        result = subprocess.run(cmd_export, check=True, capture_output=True, text=True)
        print(f"[OK] Tabelas exportadas para d={d}")
    except subprocess.CalledProcessError as e:
        error_msg = f"Erro em export_thesis_tables (d={d}):\n{e.stderr}"
        print(f"✗ {error_msg}")
        return False, error_msg
    
    print(f"\n[OK] Pipeline executado com sucesso para d={d}")
    return True, f"Pipeline OK para d={d}"


def collect_metrics(d: int) -> Optional[pd.DataFrame]:
    """
    Coleta métricas RMSE e GH por algoritmo para dimensão d.
    
    Args:
        d: Dimensão do embedding
    
    Returns:
        DataFrame com colunas [representation, d, algorithm, rmse, gh_list]
        ou None se arquivos não existirem
    """
    outputs_dir = Path('outputs')
    tables_dir = outputs_dir / 'tabelas'
    
    # Determinar sufixo baseado na representação
    # Como rodamos com ae_features+ae_topics, o sufixo é esse
    suffix = f'ae_features+ae_topics_dim{d}'
    
    rmse_path = tables_dir / f'tabela_6_3_RMSE_{suffix}.csv'
    gh_path = tables_dir / f'tabela_6_6_GH_listas_{suffix}.csv'
    
    if not rmse_path.exists():
        print(f"⚠ Arquivo RMSE não encontrado: {rmse_path}")
        return None
    
    if not gh_path.exists():
        print(f"⚠ Arquivo GH não encontrado: {gh_path}")
        return None
    
    # Carregar CSVs
    df_rmse = pd.read_csv(rmse_path)
    df_gh = pd.read_csv(gh_path)
    
    # Verificar colunas esperadas
    if 'Algoritmo' not in df_rmse.columns or 'Média' not in df_rmse.columns:
        print(f"⚠ Schema inesperado em {rmse_path}")
        return None
    
    if 'Algoritmo' not in df_gh.columns or 'Média' not in df_gh.columns:
        print(f"⚠ Schema inesperado em {gh_path}")
        return None
    
    # Mesclar por algoritmo
    df_merged = df_rmse[['Algoritmo', 'Média']].merge(
        df_gh[['Algoritmo', 'Média']],
        on='Algoritmo',
        suffixes=('_rmse', '_gh')
    )
    
    # Renomear para schema final
    df_merged = df_merged.rename(columns={
        'Algoritmo': 'algorithm',
        'Média_rmse': 'rmse',
        'Média_gh': 'gh_list'
    })
    
    # Adicionar colunas de contexto
    df_merged['representation'] = 'ae'
    df_merged['d'] = d
    
    # Reordenar colunas
    df_merged = df_merged[['representation', 'd', 'algorithm', 'rmse', 'gh_list']]
    
    print(f"[OK] Métricas coletadas para d={d}: {len(df_merged)} algoritmos")
    
    return df_merged


def main():
    """Entrypoint do script de sweep."""
    parser = argparse.ArgumentParser(
        description='Sweep de dimensão de embedding para análise de trade-offs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Força retreinamento de embeddings mesmo se cache existir'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        default='outputs/experiments/embedding_dim_sweep.parquet',
        help='Path do arquivo de saída (default: outputs/experiments/embedding_dim_sweep.parquet)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Gerar gráficos automaticamente após o sweep'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("  SWEEP DE DIMENSÃO DE EMBEDDING")
    print("="*70)
    
    # Detectar dimensão binária (D_bin)
    print("\nDetectando dimensão binária (D_bin)...")
    d_bin = get_binary_dim()
    
    # Calcular d_min usando heurística
    d_min = compute_d_min_heuristic(d_bin)
    print(f"\nUsando d_min heurístico: {d_min}")
    print(f"  Heurística: max(4, round(log2({d_bin}) * 2)) = {d_min}")
    
    # d_max é sempre D_bin
    d_max = d_bin
    print(f"Usando d_max: {d_max} (D_bin)")
    
    # Gerar sequência completa (incremento de 1)
    dims = generate_dims_sequence(d_min, d_max)
    print(f"\nSequência de dimensões gerada: {dims[0]} até {dims[-1]} (incremento de 1)")
    print(f"Total de dimensões: {len(dims)}")
    print(f"Força retreinamento: {'Sim' if args.force else 'Não'}")
    print(f"Arquivo de saída: {args.out}")
    
    # Lista para acumular resultados
    all_results = []
    
    # Loop sobre cada dimensão
    for idx, d in enumerate(dims, 1):
        print(f"\n{'#'*70}")
        print(f"  DIMENSÃO {idx}/{len(dims)}: d={d}")
        print(f"{'#'*70}")
        
        # 1. Treinar embeddings
        success, msg = train_embeddings(d, force=args.force)
        if not success:
            print(f"✗ Pulando d={d} devido a erro no treinamento")
            continue
        
        # 2. Rodar pipeline
        success, msg = run_pipeline_for_dim(d)
        if not success:
            print(f"✗ Pulando d={d} devido a erro no pipeline")
            continue
        
        # 3. Coletar métricas
        df_metrics = collect_metrics(d)
        if df_metrics is None:
            print(f"✗ Pulando d={d} devido a erro na coleta de métricas")
            continue
        
        all_results.append(df_metrics)
        
        print(f"\n[OK] d={d} processado com sucesso")
    
    # Consolidar resultados
    if not all_results:
        print("\n✗ Nenhum resultado foi coletado. Verifique os erros acima.")
        sys.exit(1)
    
    df_final = pd.concat(all_results, ignore_index=True)
    
    print(f"\n{'='*70}")
    print(f"  RESULTADOS CONSOLIDADOS")
    print(f"{'='*70}")
    print(f"Total de registros: {len(df_final)}")
    print(f"Dimensões processadas: {sorted(df_final['d'].unique().tolist())}")
    print(f"Algoritmos: {sorted(df_final['algorithm'].unique().tolist())}")
    
    # Salvar resultado
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_final.to_parquet(output_path, index=False)
    print(f"\n[OK] Resultados salvos em: {output_path}")
    print(f"  Tamanho: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Mostrar preview
    print("\nPreview dos resultados:")
    print(df_final.head(10).to_string(index=False))
    
    # Estatísticas resumidas
    print("\n" + "="*70)
    print("  ESTATÍSTICAS POR ALGORITMO")
    print("="*70)
    
    for algo in sorted(df_final['algorithm'].unique()):
        df_algo = df_final[df_final['algorithm'] == algo]
        print(f"\n{algo}:")
        print(f"  RMSE: min={df_algo['rmse'].min():.3f}, max={df_algo['rmse'].max():.3f}, média={df_algo['rmse'].mean():.3f}")
        print(f"  GH:   min={df_algo['gh_list'].min():.3f}, max={df_algo['gh_list'].max():.3f}, média={df_algo['gh_list'].mean():.3f}")
    
    print("\n[OK] Sweep concluído com sucesso!")
    
    # Gerar gráficos se solicitado
    if args.plot:
        print("\n" + "="*70)
        print("  GERANDO GRÁFICOS")
        print("="*70)
        
        try:
            from .plot_embedding_dim_sweep import generate_plots
            
            plots_dir = 'outputs/plots/embedding_dim_sweep'
            print(f"\nImportando módulo de visualização...")
            
            num_plots = generate_plots(
                input_path=str(output_path),
                output_dir=plots_dir,
                fmt='png',
                verbose=True
            )
            
            print(f"\n[OK] {num_plots} gráfico(s) gerado(s) com sucesso!")
            print(f"  Diretório: {plots_dir}")
            
        except Exception as e:
            print(f"\n✗ Erro ao gerar gráficos: {e}")
            print("  Os resultados do sweep foram salvos, mas os gráficos não puderam ser gerados.")
            print("  Execute manualmente: python -m src.experiments.plot_embedding_dim_sweep")


if __name__ == '__main__':
    main()
