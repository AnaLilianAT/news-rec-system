# -*- coding: utf-8 -*-
"""
Sweep de dimensao de embedding x seed para analise de variabilidade.

Suporta autoencoder (ae) e truncated SVD (svd).

Output: outputs/experiments/embedding_dim_seed_sweep_runs.parquet
Colunas: [d, seed, algorithm, rmse, ils_list, embedding_cache_key, runtime_sec, timestamp]

Uso:
    python -m src.experiments.run_embedding_dim_seed_sweep --dims 13 18 --n-seeds 2
    python -m src.experiments.run_embedding_dim_seed_sweep --d-min 13 --d-max 30 --step 5
    python -m src.experiments.run_embedding_dim_seed_sweep --embedding-method svd
    python -m src.experiments.run_embedding_dim_seed_sweep --resume
"""

import argparse
import subprocess
import sys
import time
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Set
from datetime import datetime
import warnings
import json

# Adicionar diretório raiz ao sys.path para permitir execução direta
if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

warnings.filterwarnings('ignore')

from src.utils.dim_grid import build_dims, compute_d_min_heuristic, get_binary_dim
from src.experiments.seed_utils import load_or_create_seeds


def train_embedding(d: int, seed: int, method: str = 'ae', force: bool = False) -> Tuple[bool, str, str]:
    """Treina/carrega embeddings para (d, seed)."""
    print(f"\n{'-'*70}")
    print(f"  EMBEDDINGS: método={method.upper()}, d={d}, seed={seed}")
    print(f"{'-'*70}")
    
    from src.embeddings.cache_utils import find_cached_embedding
    
    if not force:
        rep_type = f'{method}_features'
        features_path, features_json = find_cached_embedding(
            base_dir=Path('outputs'), representation_type=rep_type, d=d, seed=seed
        )
        
        if features_path and features_path.exists():
            with open(features_json, 'r') as f:
                metadata = json.load(f)
            cache_key = metadata.get('cache_key', f'd{d}_seed{seed}_{method}')
            print(f"[OK] Cache encontrado: {cache_key}")
            return True, f"Cache d={d}, seed={seed}, {method}", cache_key
    
    cmd = [sys.executable, '-m', 'src.embeddings.train_embeddings',
           '--embedding-dim', str(d), '--seed', str(seed), 
           '--embedding-method', method, '--data-dir', 'outputs']
    if force:
        cmd.append('--force')
    
    print(f"[>] Treinando...")
    
    # Não capturar output - deixar imprimir diretamente no terminal
    # Verificaremos sucesso pela existência dos arquivos, não pelo returncode
    result = subprocess.run(cmd, cwd=Path.cwd())
    
    # Verificar sucesso pela existência dos arquivos, não pelo returncode
    rep_type = f'{method}_features'
    features_path, features_json = find_cached_embedding(
        base_dir=Path('outputs'), representation_type=rep_type, d=d, seed=seed
    )
    
    if features_path and features_path.exists():
        if features_json and features_json.exists():
            with open(features_json, 'r') as f:
                cache_key = json.load(f).get('cache_key', f'd{d}_seed{seed}_{method}')
        else:
            cache_key = f'd{d}_seed{seed}_{method}'
        print(f"[OK] Treinado: {cache_key}")
        return True, f"OK d={d}, seed={seed}, {method}", cache_key
    else:
        print(f"[X] Falhou: arquivos não foram criados")
        return False, f"Erro d={d}, seed={seed}, {method}", ""


def run_pipeline(d: int, seed: int, method: str = 'ae', aggregate_by_user: bool = True, ranking_metric: str = 'rmse', ndcg_cutoff: int = 20) -> Tuple[bool, str]:
    """Roda pipeline completo para (d, seed)."""
    print(f"\n{'-'*70}")
    print(f"  PIPELINE: método={method.upper()}, d={d}, seed={seed}")
    print(f"  Agregação: {'Por usuário' if aggregate_by_user else 'Global'}")
    print(f"  Métrica de ranking: {ranking_metric.upper()}")
    print(f"{'-'*70}")
    
    feature_rep = f'{method}_features'
    topic_rep = f'{method}_topics'
    
    # 1. Generate reclists
    print("[>] [1/3] Gerando listas...")
    cmd1 = [sys.executable, '-m', 'src.run_generate_reclists_assigned',
            '--representations', feature_rep, topic_rep,
            '--embedding-dim', str(d), '--seed', str(seed)]
    
    result = subprocess.run(cmd1, cwd=Path.cwd())
    if result.returncode == 0:
        print("[OK] Listas geradas")
    else:
        print(f"[X] Erro ao gerar listas (returncode={result.returncode})")
        return False, f"Erro generate_reclists"
    
    # 2. Eval
    print("[>] [2/3] Avaliando...")
    # run_eval_replay_assigned espera representações no formato "feature+topic"
    rep_suffix = f"{feature_rep}+{topic_rep}"
    cmd2 = [sys.executable, '-m', 'src.run_eval_replay_assigned',
            '--representations', rep_suffix,
            '--embedding-dim', str(d),
            '--seed', str(seed),
            '--ranking-metric', ranking_metric]
    
    if ranking_metric == 'ndcg':
        cmd2.extend(['--ndcg-cutoff', str(ndcg_cutoff)])
    
    result = subprocess.run(cmd2, cwd=Path.cwd())
    if result.returncode == 0:
        print("[OK] Avaliacao concluida")
    else:
        print(f"[X] Erro ao avaliar (returncode={result.returncode})")
        return False, f"Erro eval"
    
    # 3. Export
    print("[>] [3/3] Exportando tabelas...")
    cmd3 = [sys.executable, '-m', 'src.run_export_thesis_tables',
            '--embedding-dim', str(d),
            '--ranking-metric', ranking_metric]
    
    if ranking_metric == 'ndcg':
        cmd3.extend(['--ndcg-cutoff', str(ndcg_cutoff)])
    
    # Adicionar flag de agregação
    if not aggregate_by_user:
        cmd3.append('--global-aggregation')
    
    result = subprocess.run(cmd3, cwd=Path.cwd())
    if result.returncode == 0:
        print("[OK] Tabelas exportadas")
    else:
        print(f"[X] Erro ao exportar (returncode={result.returncode})")
        return False, f"Erro export"
    
    print(f"[OK] Pipeline completo")
    return True, f"OK d={d}, seed={seed}"  


def collect_metrics(d: int, seed: int, method: str, cache_key: str, ranking_metric: str = 'rmse', ndcg_cutoff: int = 20) -> Optional[pd.DataFrame]:
    """Coleta metricas de ranking (RMSE ou NDCG) e ILS."""
    print(f"[>] Coletando metricas...")
    
    tables_dir = Path('outputs/tabelas')
    suffix = f'{method}_features+{method}_topics_dim{d}'
    
    # Determinar nome do arquivo de ranking baseado na métrica
    if ranking_metric == 'rmse':
        ranking_path = tables_dir / f'tabela_6_3_RMSE_{suffix}.csv'
        ranking_col_user = 'Média'
        ranking_col_global = 'RMSE'
    else:  # ndcg
        ranking_path = tables_dir / f'tabela_6_3_NDCG@{ndcg_cutoff}_{suffix}.csv'
        ranking_col_user = 'Média'
        ranking_col_global = 'NDCG'
    
    ils_path = tables_dir / f'tabela_6_6_ILS_listas_{suffix}.csv'
    
    if not ranking_path.exists() or not ils_path.exists():
        print(f"[X] Arquivos nao encontrados")
        return None
    
    df_ranking = pd.read_csv(ranking_path)
    df_ils = pd.read_csv(ils_path)
    
    # Detectar se é agregação por usuário (tem 'Média') ou global (tem métrica direto)
    if ranking_col_user in df_ranking.columns:
        # Modo: agregação por usuário
        df_merged = df_ranking[['Algoritmo', ranking_col_user]].merge(
            df_ils[['Algoritmo', 'Média']], on='Algoritmo', suffixes=('_rank', '_ils')
        )
        df_merged = df_merged.rename(columns={
            'Algoritmo': 'algorithm', 
            f'{ranking_col_user}_rank': ranking_metric, 
            f'{ranking_col_user}_ils': 'ils_list'
        })
    else:
        # Modo: agregação global
        df_merged = df_ranking[['Algoritmo', ranking_col_global]].merge(
            df_ils[['Algoritmo', 'ILS_COSINE_LISTS']], on='Algoritmo'
        )
        df_merged = df_merged.rename(columns={
            'Algoritmo': 'algorithm', 
            ranking_col_global: ranking_metric, 
            'ILS_COSINE_LISTS': 'ils_list'
        })
    
    df_merged['d'] = d
    df_merged['seed'] = seed
    df_merged['embedding_cache_key'] = cache_key
    df_merged['timestamp'] = datetime.now().isoformat()
    
    df_merged = df_merged[['d', 'seed', 'algorithm', ranking_metric, 'ils_list', 
                           'embedding_cache_key', 'timestamp']]
    
    print(f"[OK] {len(df_merged)} algoritmos coletados")
    return df_merged


def cleanup_intermediates():
    """Remove arquivos intermediarios."""
    patterns = [
        'predictions_assigned_*.parquet',
        'reclists_top20_assigned_*.parquet',
        'eval_pairs_assigned_*.parquet',
        'ils_by_algorithm*.parquet',
        'ils_jaccard_by_algorithm*.parquet',
        'user_metrics_assigned*.parquet',
        'eval_report_assigned*.md'
    ]
    
    count = 0
    for pattern in patterns:
        for f in Path('outputs').glob(pattern):
            try:
                f.unlink()
                count += 1
            except:
                pass
    
    if count > 0:
        print(f"[OK] {count} arquivos intermediarios removidos")


def main():
    parser = argparse.ArgumentParser(description='Sweep de dimensao x seed')
    
    # Dimensoes
    parser.add_argument('--dims', type=int, nargs='+', help='Lista de dimensoes')
    parser.add_argument('--d-min', type=int, help='Dimensao minima')
    parser.add_argument('--d-max', type=int, help='Dimensao maxima')
    parser.add_argument('--step', type=int, default=5, help='Passo (default: 5)')
    parser.add_argument('--d-bin', type=int, help='Dimensao binaria')
    
    # Seeds
    parser.add_argument('--master-seed', type=int, default=20260211)
    parser.add_argument('--n-seeds', type=int, default=20)
    parser.add_argument('--seeds-file', type=str, default='configs/experiment_seeds.json')
    
    # Controle
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--cleanup-intermediate', action='store_true')
    parser.add_argument('--out', type=str, 
                       default='outputs/experiments/embedding_dim_seed_sweep_runs.parquet')
    parser.add_argument('--embedding-method', type=str, choices=['ae', 'svd'],
                       default='svd', help='Método de embedding: ae ou svd (default: svd)')
    parser.add_argument('--global-aggregation', action='store_true',
                       help='Usar agregação global (sem passar por usuário) para RMSE e ILS')
    parser.add_argument('--ranking-metric', type=str, choices=['rmse', 'ndcg'],
                       default='rmse', help='Métrica de ranking: rmse ou ndcg (default: rmse)')
    parser.add_argument('--ndcg-cutoff', type=int, default=20,
                       help='Cutoff N para NDCG@N (default: 20)')
    
    args = parser.parse_args()
    use_resume = args.resume and not args.no_resume
    
    print("="*70)
    print(" EMBEDDING DIMENSION x SEED SWEEP")
    print("="*70)
    print(f"\nMétodo de embedding: {args.embedding_method.upper()}")
    print(f"  {'Autoencoder' if args.embedding_method == 'ae' else 'Truncated SVD'}")
    print(f"\nMétrica de ranking: {args.ranking_metric.upper()}")
    if args.ranking_metric == 'ndcg':
        print(f"  NDCG@{args.ndcg_cutoff}")
    print(f"\nAgregação: {'Global' if args.global_aggregation else 'Por usuário e depois por algoritmo'}")
    
    # 1. Dimensoes
    if args.dims:
        dims = args.dims
        print(f"\n[>] Dimensoes: {dims}")
    else:
        if args.d_bin is None:
            d_bin = get_binary_dim()
        else:
            d_bin = args.d_bin
        
        d_min = args.d_min if args.d_min else compute_d_min_heuristic(d_bin)
        d_max = args.d_max if args.d_max else d_bin
        
        dims = build_dims(d_min=d_min, d_max=d_max, step=args.step)
        print(f"\n[>] Dimensoes: {len(dims)} valores de {d_min} a {d_max}")
    
    # 2. Seeds
    seeds = load_or_create_seeds(args.seeds_file, args.master_seed, args.n_seeds)
    print(f"[>] Seeds: {len(seeds)} valores")
    
    # 3. Resume
    output_path = Path(args.out)
    completed = set()
    
    if use_resume and output_path.exists():
        df_existing = pd.read_parquet(output_path)
        for _, row in df_existing.iterrows():
            completed.add((int(row['d']), int(row['seed']), str(row['algorithm'])))
        print(f"\n[>] Resume: {len(completed)} combinacoes ja computadas")
    
    # 4. Sweep
    total = len(dims) * len(seeds)
    print(f"\n{'='*70}")
    print(f" EXECUTANDO SWEEP: {len(dims)} dims x {len(seeds)} seeds = {total}")
    print(f"{'='*70}")
    
    all_results = []
    ok_count, skip_count, fail_count = 0, 0, 0
    start_time = time.time()
    
    for idx_d, d in enumerate(dims, 1):
        for idx_s, seed in enumerate(seeds, 1):
            iteration = (idx_d - 1) * len(seeds) + idx_s
            
            print(f"\n{'='*70}")
            print(f" [{iteration}/{total}] d={d}, seed={seed}")
            print(f"{'='*70}")
            
            # Check resume
            if use_resume and any((d, seed, a) in completed for a in ['knnu', 'knni', 'svd']):
                print(f"[O] Ja computado, pulando")
                skip_count += 1
                continue
            
            iter_start = time.time()
            
            # Train
            success_emb, _, cache_key = train_embedding(d, seed, args.embedding_method, args.force)
            if not success_emb:
                fail_count += 1
                continue
            
            # Pipeline
            success_pipe, _ = run_pipeline(d, seed, args.embedding_method, 
                                          aggregate_by_user=not args.global_aggregation,
                                          ranking_metric=args.ranking_metric,
                                          ndcg_cutoff=args.ndcg_cutoff)
            if not success_pipe:
                fail_count += 1
                continue
            
            # Metrics
            df_metrics = collect_metrics(d, seed, args.embedding_method, cache_key,
                                        ranking_metric=args.ranking_metric,
                                        ndcg_cutoff=args.ndcg_cutoff)
            if df_metrics is None:
                fail_count += 1
                continue
            
            df_metrics['runtime_sec'] = time.time() - iter_start
            all_results.append(df_metrics)
            ok_count += 1
            
            print(f"[OK] Iteracao {iteration} concluida em {df_metrics['runtime_sec'].iloc[0]:.1f}s")
            print(f"    Progresso: {ok_count} OK, {skip_count} skip, {fail_count} fail")
            
            # Cleanup
            if args.cleanup_intermediate:
                cleanup_intermediates()
            
            # Save batch
            if len(all_results) >= 5 or iteration == total:
                if all_results:
                    df_batch = pd.concat(all_results, ignore_index=True)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if output_path.exists():
                        df_existing = pd.read_parquet(output_path)
                        df_combined = pd.concat([df_existing, df_batch], ignore_index=True)
                        df_combined.to_parquet(output_path, index=False)
                    else:
                        df_batch.to_parquet(output_path, index=False)
                    
                    all_results = []
                    print(f"[OK] Resultados salvos")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(" SWEEP CONCLUIDO")
    print(f"{'='*70}")
    print(f"Tempo: {total_time/60:.1f} min")
    print(f"OK: {ok_count}, Skip: {skip_count}, Fail: {fail_count}")
    print(f"Output: {output_path}")
    
    if output_path.exists():
        df_final = pd.read_parquet(output_path)
        print(f"\nResumo:")
        print(f"  Linhas: {len(df_final)}")
        print(f"  Dimensoes: {df_final['d'].nunique()}")
        print(f"  Seeds: {df_final['seed'].nunique()}")
        print(f"  Algoritmos: {df_final['algorithm'].nunique()}")


if __name__ == '__main__':
    main()
