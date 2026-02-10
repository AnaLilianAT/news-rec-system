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

from . import format_like_thesis

normalize_algorithm_name = format_like_thesis.normalize_algorithm_name
compute_GH_interaction_jaccard = format_like_thesis.compute_GH_interaction_jaccard
compute_GH_lists_cosine = format_like_thesis.compute_GH_lists_cosine
compute_RMSE_user = format_like_thesis.compute_RMSE_user
aggregate_like_thesis = format_like_thesis.aggregate_like_thesis
format_table_for_export = format_like_thesis.format_table_for_export


def process_representation(
    outputs_dir: Path,
    representation_suffix: str = None,
    representation_label: str = None
):
    """
    Processa uma representação e gera tabelas no formato da tese.
    
    Args:
        outputs_dir: Diretório de outputs
        representation_suffix: Sufixo dos arquivos (ex: 'ae_features+ae_topics')
        representation_label: Label para display (ex: 'ae_features+ae_topics')
    
    Returns:
        0 se sucesso, 1 se erro
    """
    reports_dir = outputs_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Determinar sufixo para arquivos
    suffix = f"_{representation_suffix}" if representation_suffix else ""
    label = representation_label or "default (bin_features+bin_topics)"
    
    print(f"\n{'='*80}")
    print(f"PROCESSANDO: {label}")
    print(f"{'='*80}")
    
    # Verificar arquivos necessários
    required_files = {
        'eval_pairs': outputs_dir / f"eval_pairs_assigned{suffix}.parquet",
        'reclists': outputs_dir / f"reclists_top20_assigned{suffix}.parquet",
        'features': outputs_dir / "canonical_features.parquet",
        'topics': outputs_dir / "canonical_topics.parquet"
    }
    
    missing = []
    for name, path in required_files.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
    
    if missing:
        print(f"\nAVISO: Arquivos necessários não encontrados para {label}:")
        for m in missing:
            print(f"  - {m}")
        print(f"Pulando esta representação...")
        return 1
    
    print("\nTodos os arquivos necessários foram encontrados")
    
    # Carregar dados
    print("\nCarregando dados...")
    eval_pairs = pd.read_parquet(required_files['eval_pairs'])
    reclists = pd.read_parquet(required_files['reclists'])
    features = pd.read_parquet(required_files['features'])
    topics = pd.read_parquet(required_files['topics'])
    
    print(f"  - eval_pairs: {len(eval_pairs)} registros")
    print(f"  - reclists: {len(reclists)} listas")
    print(f"  - features: {len(features)} itens")
    print(f"  - topics: {len(topics)} relações item-tópico")
    
    # Normalizar nomes de algoritmos nos dados
    print("\nNormalizando nomes de algoritmos...")
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
        print("AVISO: Nenhum usuário com >= 2 itens para calcular GH (interação)")
        report_lines.append("## Tabela 6.1: GH (Jaccard - Interação)\n")
        report_lines.append("Nenhum dado disponível\n")
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
        output_path_6_1 = outputs_dir / f"tabela_6_1_GH_interacao{suffix}.csv"
        table_6_1_formatted.to_csv(output_path_6_1, index=False)
        
        print(f"\nTabela salva em: {output_path_6_1}")
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
        print("AVISO: Nenhuma lista com >= 2 itens para calcular GH (listas)")
        report_lines.append("## Tabela 6.6: GH (Cosseno - Listas)\n")
        report_lines.append("Nenhum dado disponível\n")
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
        output_path_6_6 = outputs_dir / f"tabela_6_6_GH_listas{suffix}.csv"
        table_6_6_formatted.to_csv(output_path_6_6, index=False)
        
        print(f"\nTabela salva em: {output_path_6_6}")
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
        print("AVISO: Nenhum usuário com >= 2 pares para calcular RMSE")
        report_lines.append("## Tabela 6.3: RMSE\n")
        report_lines.append("Nenhum dado disponível\n")
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
        output_path_6_3 = outputs_dir / f"tabela_6_3_RMSE{suffix}.csv"
        table_6_3_formatted.to_csv(output_path_6_3, index=False)
        
        print(f"\nTabela salva em: {output_path_6_3}")
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
    # DESABILITADO: Relatório não é mais gerado
    # ========================================================================
    # report_path = reports_dir / f"thesis_format_report{suffix}.md"
    # with open(report_path, 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(report_lines))
    # 
    # print("\n" + "=" * 80)
    # print(f"Relatório salvo em: {report_path}")
    # print("=" * 80)
    
    print(f"\nTabelas geradas para {label}:")
    print(f"  1. {output_path_6_1}")
    print(f"  2. {output_path_6_6}")
    print(f"  3. {output_path_6_3}")
    # print(f"  4. {report_path}")
    
    return 0


def main():
    """
    Função principal: gera tabelas no formato da tese para múltiplas representações.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Gera tabelas no formato da tese a partir das métricas calculadas'
    )
    parser.add_argument(
        '--representations',
        type=str,
        nargs='+',
        help='Sufixos de representações a processar (ex: bin_features+bin_topics ae_features+ae_topics)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Diretório de saída (default: outputs)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GERAÇÃO DE TABELAS NO FORMATO DA TESE")
    print("=" * 80)
    
    outputs_dir = Path(args.output_dir)
    
    # Determinar quais representações processar
    if args.representations:
        # Processar representações especificadas
        representations_to_process = args.representations
    else:
        # Auto-detectar arquivos eval_pairs disponíveis
        eval_files = list(outputs_dir.glob('eval_pairs_assigned*.parquet'))
        representations_to_process = []
        
        for file in eval_files:
            filename = file.stem  # eval_pairs_assigned ou eval_pairs_assigned_XXX
            if filename == 'eval_pairs_assigned':
                representations_to_process.append(None)  # Default
            else:
                # Extrair sufixo
                suffix = filename.replace('eval_pairs_assigned_', '')
                representations_to_process.append(suffix)
        
        if not representations_to_process:
            print("\nERRO: Nenhum arquivo eval_pairs_assigned*.parquet encontrado")
            print("Execute as etapas anteriores do pipeline antes de rodar este script.")
            return 1
    
    if len(representations_to_process) > 1:
        print(f"\n⚙ Processando {len(representations_to_process)} representações")
    
    # Processar cada representação
    success_count = 0
    for idx, suffix in enumerate(representations_to_process, 1):
        if len(representations_to_process) > 1:
            label = suffix or "default (bin_features+bin_topics)"
            print(f"\n[{idx}/{len(representations_to_process)}]")
        else:
            label = suffix or "default"
        
        result = process_representation(
            outputs_dir=outputs_dir,
            representation_suffix=suffix,
            representation_label=label
        )
        
        if result == 0:
            success_count += 1
    
    # Resumo final
    print("\n" + "=" * 80)
    if success_count == len(representations_to_process):
        print("CONCLUÍDO! Todas as tabelas foram geradas com sucesso.")
    else:
        print(f"CONCLUÍDO com avisos: {success_count}/{len(representations_to_process)} representações processadas.")
    print("=" * 80)
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
