"""
Script para gerar tabelas no formato exato da tese.

Gera:
- outputs/tabela_6_1_GH_interacao.csv (GH Jaccard - itens interagidos)
- outputs/tabela_6_6_GH_listas.csv (GH Cosseno - listas)
- outputs/tabela_6_3_RMSE.csv (RMSE por usuário)
- outputs/reports/thesis_format_report.md (documentação)
"""

import pandas as pd
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from format_like_thesis import (
    normalize_algorithm_name,
    compute_GH_interaction_jaccard,
    compute_GH_lists_cosine,
    compute_RMSE_user,
    aggregate_like_thesis,
    format_table_for_export
)


def main():
    print("=" * 80)
    print("GERAÇÃO DE TABELAS NO FORMATO DA TESE")
    print("=" * 80)
    
    # Diretórios
    outputs_dir = Path("outputs")
    reports_dir = outputs_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Verificar arquivos necessários
    required_files = {
        'eval_pairs': outputs_dir / "eval_pairs_assigned.parquet",
        'reclists': outputs_dir / "reclists_top20_assigned.parquet",
        'features': outputs_dir / "canonical_features.parquet",
        'topics': outputs_dir / "canonical_topics.parquet"
    }
    
    missing = []
    for name, path in required_files.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
    
    if missing:
        print("\n❌ ERRO: Arquivos necessários não encontrados:")
        for m in missing:
            print(f"  - {m}")
        print("\nExecute as etapas anteriores do pipeline antes de rodar este script.")
        return 1
    
    print("\n✓ Todos os arquivos necessários foram encontrados")
    
    # Carregar dados
    print("\n📂 Carregando dados...")
    eval_pairs = pd.read_parquet(required_files['eval_pairs'])
    reclists = pd.read_parquet(required_files['reclists'])
    features = pd.read_parquet(required_files['features'])
    topics = pd.read_parquet(required_files['topics'])
    
    print(f"  - eval_pairs: {len(eval_pairs)} registros")
    print(f"  - reclists: {len(reclists)} listas")
    print(f"  - features: {len(features)} itens")
    print(f"  - topics: {len(topics)} relações item-tópico")
    
    # Normalizar nomes de algoritmos nos dados
    print("\n🔄 Normalizando nomes de algoritmos...")
    eval_pairs['algorithm'] = eval_pairs['algorithm'].apply(normalize_algorithm_name)
    reclists['algorithm'] = reclists['algorithm'].apply(normalize_algorithm_name)
    
    algorithms = sorted(eval_pairs['algorithm'].unique())
    print(f"  Algoritmos encontrados: {', '.join(algorithms)}")
    
    # Inicializar relatório
    report_lines = ["# Relatório de Geração de Tabelas no Formato da Tese\n"]
    report_lines.append(f"Total de pares de avaliação: {len(eval_pairs)}")
    report_lines.append(f"Total de listas top-20: {len(reclists)}")
    report_lines.append(f"Algoritmos: {', '.join(algorithms)}\n")
    
    # ========================================================================
    # TABELA 6.1: GH (itens recomendados e interagidos) - Jaccard
    # ========================================================================
    print("\n" + "=" * 80)
    print("TABELA 6.1: GH (Jaccard) - Itens Recomendados e Interagidos")
    print("=" * 80)
    
    print("Calculando GH por usuário usando Jaccard entre tópicos...")
    df_gh_interaction = compute_GH_interaction_jaccard(eval_pairs, topics)
    
    if len(df_gh_interaction) == 0:
        print("❌ AVISO: Nenhum usuário com >= 2 itens para calcular GH (interação)")
        report_lines.append("## Tabela 6.1: GH (Jaccard - Interação)\n")
        report_lines.append("⚠️ Nenhum dado disponível\n")
    else:
        print(f"  - {len(df_gh_interaction)} usuários com GH calculado")
        
        # Estatísticas por algoritmo
        table_6_1 = aggregate_like_thesis(
            df_gh_interaction,
            metric_col='gh_jaccard_interaction',
            include_users=True,
            include_minmax=False
        )
        
        # Formatar e exportar
        table_6_1_formatted = format_table_for_export(table_6_1, decimal_places=3)
        output_path_6_1 = outputs_dir / "tabela_6_1_GH_interacao.csv"
        table_6_1_formatted.to_csv(output_path_6_1, index=False)
        
        print(f"\n✓ Tabela salva em: {output_path_6_1}")
        print("\nPreview:")
        print(table_6_1_formatted.to_string(index=False))
        
        # Adicionar ao relatório
        report_lines.append("## Tabela 6.1: GH (Jaccard - Interação)\n")
        report_lines.append(f"Arquivo: `{output_path_6_1.name}`\n")
        report_lines.append("### Usuários por algoritmo:\n")
        for _, row in table_6_1.iterrows():
            algo = row['Algoritmo']
            n_users = int(row['Usuários'])
            report_lines.append(f"- **{algo}**: {n_users} usuários")
        
        # Exclusões
        total_users = eval_pairs['user_id'].nunique()
        included_users = len(df_gh_interaction)
        excluded = total_users - included_users
        report_lines.append(f"\n**Usuários excluídos**: {excluded} (< 2 itens expostos)\n")
    
    # ========================================================================
    # TABELA 6.6: GH (listas de recomendação) - Cosseno
    # ========================================================================
    print("\n" + "=" * 80)
    print("TABELA 6.6: GH (Cosseno) - Listas de Recomendação")
    print("=" * 80)
    
    print("Calculando GH por lista usando cosseno entre features...")
    df_gh_lists = compute_GH_lists_cosine(reclists, features)
    
    if len(df_gh_lists) == 0:
        print("❌ AVISO: Nenhuma lista com >= 2 itens para calcular GH (listas)")
        report_lines.append("## Tabela 6.6: GH (Cosseno - Listas)\n")
        report_lines.append("⚠️ Nenhum dado disponível\n")
    else:
        print(f"  - {len(df_gh_lists)} usuários com GH calculado")
        
        # Estatísticas por algoritmo
        table_6_6 = aggregate_like_thesis(
            df_gh_lists,
            metric_col='gh_cosine_lists',
            include_users=True,
            include_minmax=False
        )
        
        # Formatar e exportar
        table_6_6_formatted = format_table_for_export(table_6_6, decimal_places=3)
        output_path_6_6 = outputs_dir / "tabela_6_6_GH_listas.csv"
        table_6_6_formatted.to_csv(output_path_6_6, index=False)
        
        print(f"\n✓ Tabela salva em: {output_path_6_6}")
        print("\nPreview:")
        print(table_6_6_formatted.to_string(index=False))
        
        # Adicionar ao relatório
        report_lines.append("## Tabela 6.6: GH (Cosseno - Listas)\n")
        report_lines.append(f"Arquivo: `{output_path_6_6.name}`\n")
        report_lines.append("### Usuários por algoritmo:\n")
        for _, row in table_6_6.iterrows():
            algo = row['Algoritmo']
            n_users = int(row['Usuários'])
            report_lines.append(f"- **{algo}**: {n_users} usuários")
        
        # Exclusões
        total_lists = len(reclists)
        total_users_lists = reclists['user_id'].nunique()
        included_users_lists = len(df_gh_lists)
        excluded_lists = total_users_lists - included_users_lists
        report_lines.append(f"\n**Usuários excluídos**: {excluded_lists} (listas com < 2 itens válidos)\n")
    
    # ========================================================================
    # TABELA 6.3: RMSE
    # ========================================================================
    print("\n" + "=" * 80)
    print("TABELA 6.3: RMSE")
    print("=" * 80)
    
    print("Calculando RMSE por usuário...")
    df_rmse = compute_RMSE_user(eval_pairs)
    
    if len(df_rmse) == 0:
        print("❌ AVISO: Nenhum usuário com >= 2 pares para calcular RMSE")
        report_lines.append("## Tabela 6.3: RMSE\n")
        report_lines.append("⚠️ Nenhum dado disponível\n")
    else:
        print(f"  - {len(df_rmse)} usuários com RMSE calculado")
        
        # Estatísticas por algoritmo (sem coluna Usuários, com Min/Max)
        table_6_3 = aggregate_like_thesis(
            df_rmse,
            metric_col='rmse',
            include_users=False,
            include_minmax=True
        )
        
        # Formatar e exportar
        table_6_3_formatted = format_table_for_export(table_6_3, decimal_places=3)
        output_path_6_3 = outputs_dir / "tabela_6_3_RMSE.csv"
        table_6_3_formatted.to_csv(output_path_6_3, index=False)
        
        print(f"\n✓ Tabela salva em: {output_path_6_3}")
        print("\nPreview:")
        print(table_6_3_formatted.to_string(index=False))
        
        # Adicionar ao relatório
        report_lines.append("## Tabela 6.3: RMSE\n")
        report_lines.append(f"Arquivo: `{output_path_6_3.name}`\n")
        report_lines.append("### Usuários por algoritmo:\n")
        for algo in df_rmse['algorithm'].unique():
            n_users = len(df_rmse[df_rmse['algorithm'] == algo])
            algo_normalized = normalize_algorithm_name(algo)
            report_lines.append(f"- **{algo_normalized}**: {n_users} usuários")
        
        # Exclusões
        total_users_rmse = eval_pairs['user_id'].nunique()
        included_users_rmse = len(df_rmse)
        excluded_rmse = total_users_rmse - included_users_rmse
        report_lines.append(f"\n**Usuários excluídos**: {excluded_rmse} (< 2 pares de avaliação)\n")
    
    # ========================================================================
    # Salvar relatório
    # ========================================================================
    report_path = reports_dir / "thesis_format_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print("\n" + "=" * 80)
    print(f"✓ Relatório salvo em: {report_path}")
    print("=" * 80)
    
    print("\n✅ CONCLUÍDO! As três tabelas foram geradas com sucesso.")
    print("\nArquivos gerados:")
    print(f"  1. {output_path_6_1}")
    print(f"  2. {output_path_6_6}")
    print(f"  3. {output_path_6_3}")
    print(f"  4. {report_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
