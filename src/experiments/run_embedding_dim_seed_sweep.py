# -*- coding: utf-8 -*-
"""
Sweep de dimensao de embedding x seed para analise de variabilidade.

Output: outputs/experiments/embedding_dim_seed_sweep_runs.parquet
Colunas: [d, seed, algorithm, rmse, gh_list, embedding_cache_key, runtime_sec, timestamp]

Uso:
    python -m src.experiments.run_embedding_dim_seed_sweep --dims 13 18 --n-seeds 2
    python -m src.experiments.run_embedding_dim_seed_sweep --d-min 13 --d-max 30 --step 5
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
warnings.filterwarnings('ignore')

from ..utils.dim_grid import build_dims, compute_d_min_heuristic
from .seed_utils import load_or_create_seeds


def train_embedding(d: int, seed: int, force: bool = False) -> Tuple[bool, str, str]:
    """Treina/carrega embeddings para (d, seed)."""
    print(f"\n{'-'*70}")
    print(f"  EMBEDDINGS: d={d}, seed={seed}")
    print(f"{'-'*70}")
    
    from ..embeddings.cache_utils import find_cached_embedding
    
    if not force:
        features_path, features_json = find_cached_embedding(
            base_dir=Path('outputs'), representation_type='ae_features', d=d, seed=seed
        )
        
        if features_path and features_path.exists():
            with open(features_json, 'r') as f:
                metadata = json.load(f)
            cache_key = metadata.get('cache_key', f'd{d}_seed{seed}')
            print(f"[OK] Cache encontrado: {cache_key}")
            return True, f"Cache d={d}, seed={seed}", cache_key
    
    cmd = [sys.executable, '-m', 'src.embeddings.train_embeddings',
           '--embedding-dim', str(d), '--seed', str(seed), '--data-dir', 'outputs']
    if force:
        cmd.append('--force')
    
    print(f"[>] Treinando...")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=Path.cwd())
        features_path, features_json = find_cached_embedding(
            base_dir=Path('outputs'), representation_type='ae_features', d=d, seed=seed
        )
        if features_json and features_json.exists():
            with open(features_json, 'r') as f:
                cache_key = json.load(f).get('cache_key', f'd{d}_seed{seed}')
        else:
            cache_key = f'd{d}_seed{seed}'
        print(f"[OK] Treinado: {cache_key}")
        return True, f"OK d={d}, seed={seed}", cache_key
    except subprocess.CalledProcessError as e:
        print(f"[X] Erro: {e.stderr[:200]}")
        return False, f"Erro d={d}, seed={seed}", ""


def run_pipeline(d: int, seed: int) -> Tuple[bool, str]:
    """Roda pipeline completo para (d, seed)."""
    print(f"\n{'-'*70}")
    print(f"  PIPELINE: d={d}, seed={seed}")
    print(f"{'-'*70}")
    
    # 1. Generate reclists
    print("[>] [1/3] Gerando listas...")
    cmd1 = [sys.executable, '-m', 'src.run_generate_reclists_assigned',
            '--representations', 'ae_features', 'ae_topics',
            '--embedding-dim', str(d), '--seed', str(seed)]
    
    try:
        subprocess.run(cmd1, check=True, capture_output=True, text=True, cwd=Path.cwd())
        print("[OK] Listas geradas")
    except subprocess.CalledProcessError as e:
        print(f"[X] Erro: {e.stderr[:200]}")
        return False, f"Erro generate_reclists"
    
    # 2. Eval
    print("[>] [2/3] Avaliando...")
    cmd2 = [sys.executable, '-m', 'src.run_eval_replay_assigned']
    
    try:
        subprocess.run(cmd2, check=True, capture_output=True, text=True, cwd=Path.cwd())
        print("[OK] Avaliacao concluida")
    except subprocess.CalledProcessError as e:
        print(f"[X] Erro: {e.stderr[:200]}")
        return False, f"Erro eval"
    
    # 3. Export
    print("[>] [3/3] Exportando tabelas...")
    cmd3 = [sys.executable, '-m', 'src.run_export_thesis_tables',
            '--embedding-dim', str(d)]
    
    try:
        subprocess.run(cmd3, check=True, capture_output=True, text=True, cwd=Path.cwd())
        print("[OK] Tabelas exportadas")
    except subprocess.CalledProcessError as e:
        print(f"[X] Erro: {e.stderr[:200]}")
        return False, f"Erro export"
    
    print(f"[OK] Pipeline completo")
    return True, f"OK d={d}, seed={seed}"


def collect_metrics(d: int, seed: int, cache_key: str) -> Optional[pd.DataFrame]:
    """Coleta metricas RMSE e GH."""
    print(f"[>] Coletando metricas...")
    
    tables_dir = Path('outputs/tabelas')
    suffix = f'ae_features+ae_topics_dim{d}'
    
    rmse_path = tables_dir / f'tabela_6_3_RMSE_{suffix}.csv'
    gh_path = tables_dir / f'tabela_6_6_GH_listas_{suffix}.csv'
    
    if not rmse_path.exists() or not gh_path.exists():
        print(f"[X] Arquivos nao encontrados")
        return None
    
    df_rmse = pd.read_csv(rmse_path)
    df_gh = pd.read_csv(gh_path)
    
    df_merged = df_rmse[['Algoritmo', 'Média']].merge(
        df_gh[['Algoritmo', 'Média']], on='Algoritmo', suffixes=('_rmse', '_gh')
    )
    
    df_merged = df_merged.rename(columns={
        'Algoritmo': 'algorithm', 'Média_rmse': 'rmse', 'Média_gh': 'gh_list'
    })
    
    df_merged['d'] = d
    df_merged['seed'] = seed
    df_merged['embedding_cache_key'] = cache_key
    df_merged['timestamp'] = datetime.now().isoformat()
    
    df_merged = df_merged[['d', 'seed', 'algorithm', 'rmse', 'gh_list', 
                           'embedding_cache_key', 'timestamp']]
    
    print(f"[OK] {len(df_merged)} algoritmos coletados")
    return df_merged


def cleanup_intermediates():
    """Remove arquivos intermediarios."""
    patterns = ['predictions_assigned_*.parquet', 'reclists_top20_assigned_*.parquet',
                'eval_pairs_assigned_*.parquet', 'gh_lists_*.parquet',
                'user_metrics_*.parquet', 'report_assigned_*.md']
    
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
    
    args = parser.parse_args()
    use_resume = args.resume and not args.no_resume
    
    print("="*70)
    print(" EMBEDDING DIMENSION x SEED SWEEP")
    print("="*70)
    
    # 1. Dimensoes
    if args.dims:
        dims = args.dims
        print(f"\n[>] Dimensoes: {dims}")
    else:
        if args.d_bin is None:
            from .embedding_dim_sweep import get_binary_dim
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
            success_emb, _, cache_key = train_embedding(d, seed, args.force)
            if not success_emb:
                fail_count += 1
                continue
            
            # Pipeline
            success_pipe, _ = run_pipeline(d, seed)
            if not success_pipe:
                fail_count += 1
                continue
            
            # Metrics
            df_metrics = collect_metrics(d, seed, cache_key)
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
