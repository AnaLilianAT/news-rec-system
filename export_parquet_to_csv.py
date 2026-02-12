#!/usr/bin/env python3
"""
Exporta arquivos parquet para CSV para visualização.

Converte os arquivos de resultados do sweep de dimensão/seed para formato CSV.
"""

import pandas as pd
from pathlib import Path


def export_parquet_to_csv():
    """Exporta arquivos parquet para CSV."""
    
    # Caminhos dos arquivos
    base_dir = Path(__file__).parent
    experiments_dir = base_dir / "outputs" / "experiments"
    
    runs_parquet = experiments_dir / "embedding_dim_seed_sweep_runs.parquet"
    agg_parquet = experiments_dir / "embedding_dim_seed_sweep_agg.parquet"
    
    runs_csv = experiments_dir / "embedding_dim_seed_sweep_runs.csv"
    agg_csv = experiments_dir / "embedding_dim_seed_sweep_agg.csv"
    
    print("=" * 80)
    print("EXPORTAÇÃO DE PARQUET PARA CSV")
    print("=" * 80)
    
    # Exportar runs
    if runs_parquet.exists():
        print(f"\n📊 Lendo {runs_parquet.name}...")
        df_runs = pd.read_parquet(runs_parquet)
        print(f"   ✓ {len(df_runs)} linhas, {len(df_runs.columns)} colunas")
        
        print(f"\n💾 Salvando em {runs_csv.name}...")
        df_runs.to_csv(runs_csv, index=False, encoding='utf-8')
        print(f"   ✓ Arquivo CSV criado: {runs_csv}")
        print(f"   ✓ Tamanho: {runs_csv.stat().st_size / 1024:.1f} KB")
    else:
        print(f"\n❌ Arquivo não encontrado: {runs_parquet}")
    
    # Exportar agregados
    if agg_parquet.exists():
        print(f"\n📈 Lendo {agg_parquet.name}...")
        df_agg = pd.read_parquet(agg_parquet)
        print(f"   ✓ {len(df_agg)} linhas, {len(df_agg.columns)} colunas")
        
        print(f"\n💾 Salvando em {agg_csv.name}...")
        df_agg.to_csv(agg_csv, index=False, encoding='utf-8')
        print(f"   ✓ Arquivo CSV criado: {agg_csv}")
        print(f"   ✓ Tamanho: {agg_csv.stat().st_size / 1024:.1f} KB")
    else:
        print(f"\n❌ Arquivo não encontrado: {agg_parquet}")
    
    print("\n" + "=" * 80)
    print("✅ EXPORTAÇÃO CONCLUÍDA")
    print("=" * 80)
    print("\nArquivos CSV criados em:")
    print(f"  - {runs_csv}")
    print(f"  - {agg_csv}")
    print("\nVocê pode abrir estes arquivos no Excel, LibreOffice ou qualquer editor de texto.")


if __name__ == "__main__":
    export_parquet_to_csv()
