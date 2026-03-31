"""
Script para gerar tabelas no formato exato da tese.

Gera:
- outputs/tabela_6_1_ILS_interacao.csv (ILS Jaccard - itens interagidos)
- outputs/tabela_6_6_ILS_listas.csv (ILS Cosseno - listas)
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
compute_ILS_interaction_jaccard = format_like_thesis.compute_ILS_interaction_jaccard
compute_ILS_lists_cosine = format_like_thesis.compute_ILS_lists_cosine
compute_ILS_lists_cosine_global = format_like_thesis.compute_ILS_lists_cosine_global
compute_RMSE_user = format_like_thesis.compute_RMSE_user
compute_RMSE_global = format_like_thesis.compute_RMSE_global
compute_NDCG_user = format_like_thesis.compute_NDCG_user
compute_NDCG_global = format_like_thesis.compute_NDCG_global
aggregate_like_thesis = format_like_thesis.aggregate_like_thesis
aggregate_global = format_like_thesis.aggregate_global
format_table_for_export = format_like_thesis.format_table_for_export


def process_representation(
    outputs_dir: Path,
    representation_suffix: str = None,
    representation_label: str = None,
    embedding_dim: int = None,
    aggregate_by_user: bool = True,
    ranking_metric: str = 'rmse',
    ndcg_cutoff: int = 20
):
    """
    Processa uma representação e gera tabelas no formato da tese.
    
    Args:
        outputs_dir: Diretório de outputs
        representation_suffix: Sufixo dos arquivos (ex: 'ae_features+ae_topics')
        representation_label: Label para display (ex: 'ae_features+ae_topics')
        embedding_dim: Dimensão do embedding (opcional, para incluir no nome do arquivo)
        aggregate_by_user: Se True, agrega por usuário primeiro; se False, agrega globalmente
    
    Returns:
        0 se sucesso, 1 se erro
    """
    # Criar diretório tabelas/
    tables_dir = outputs_dir / "tabelas"
    tables_dir.mkdir(exist_ok=True)
    
    reports_dir = outputs_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Determinar sufixo para arquivos de entrada (eval_pairs, reclists)
    # Esses arquivos NÃO incluem a dimensão no nome
    input_suffix = f"_{representation_suffix}" if representation_suffix else ""
    
    # Determinar sufixo para arquivos de saída (tabelas)
    # Esses arquivos incluem a dimensão no nome
    output_suffix = input_suffix
    if embedding_dim is not None:
        output_suffix += f"_dim{embedding_dim}"
    
    label = representation_label or "default (bin_features+bin_topics)"
    
    print(f"\n{'='*80}")
    print(f"PROCESSANDO: {label}")
    print(f"{'='*80}")
    
    # Verificar arquivos necessários
    required_files = {
        'eval_pairs': outputs_dir / f"eval_pairs_assigned{input_suffix}.parquet",
        'reclists': outputs_dir / f"reclists_top20_assigned{input_suffix}.parquet",
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
    # TABELA 6.1: ILS (itens recomendados e interagidos) - Jaccard
    # ========================================================================
    print("\n" + "=" * 80)
    print("TABELA 6.1: ILS (Jaccard) - Itens Recomendados e Interagidos")
    print("=" * 80)
    
    print("Calculando ILS por usuário usando Jaccard entre tópicos...")
    df_ils_interaction = compute_ILS_interaction_jaccard(eval_pairs, topics)
    
    if len(df_ils_interaction) == 0:
        print("AVISO: Nenhum usuário com >= 2 itens para calcular ILS (interação)")
        report_lines.append("## Tabela 6.1: ILS (Jaccard - Interação)\n")
        report_lines.append("Nenhum dado disponível\n")
    else:
        print(f"  - {len(df_ils_interaction)} usuários com ILS calculado")
        
        # Estatísticas por algoritmo
        table_6_1 = aggregate_like_thesis(
            df_ils_interaction,
            metric_col='ils_jaccard_interaction',
            include_users=True,
            include_minmax=False
        )
        
        # Formatar e exportar
        table_6_1_formatted = format_table_for_export(table_6_1, decimal_places=3)
        output_path_6_1 = tables_dir / f"tabela_6_1_ILS_interacao{output_suffix}.csv"
        table_6_1_formatted.to_csv(output_path_6_1, index=False)
        
        print(f"\nTabela salva em: {output_path_6_1}")
        print("\nPreview:")
        print(table_6_1_formatted.to_string(index=False))
        
        # Adicionar ao relatório
        report_lines.append("## Tabela 6.1: ILS (Jaccard - Interação)\n")
        report_lines.append(f"Arquivo: `{output_path_6_1.name}`\n")
        report_lines.append("### Usuários por algoritmo:\n")
        for _, row in table_6_1.iterrows():
            algo = row['Algoritmo']
            n_users = int(row['Usuários'])
            report_lines.append(f"- **{algo}**: {n_users} usuários")
        
        # Exclusões
        total_users = eval_pairs['user_id'].nunique()
        included_users = len(df_ils_interaction)
        excluded = total_users - included_users
        report_lines.append(f"\n**Usuários excluídos**: {excluded} (< 2 itens expostos)\n")
    
    # ========================================================================
    # TABELA 6.6: ILS (listas de recomendação) - Cosseno
    # ========================================================================
    print("\n" + "=" * 80)
    print("TABELA 6.6: ILS (Cosseno) - Listas de Recomendação")
    print("=" * 80)
    
    if aggregate_by_user:
        print("Calculando ILS por usuário usando cosseno entre features...")
        df_ils_lists = compute_ILS_lists_cosine(reclists, features)
    else:
        print("Calculando ILS global usando cosseno entre features...")
        # Primeiro calcula por usuário, depois agrega por algoritmo
        df_ils_lists_per_user = compute_ILS_lists_cosine(reclists, features)
        
        # Agregar por algoritmo (global)
        results = []
        for algorithm, group in df_ils_lists_per_user.groupby('algorithm'):
            # Média ponderada pelo número de listas de cada usuário
            total_lists = group['n_lists'].sum()
            weighted_ils = (group['ils_cosine_lists'] * group['n_lists']).sum() / total_lists
            results.append({
                'algorithm': algorithm,
                'ils_cosine_lists': weighted_ils,
                'n_lists': int(total_lists)
            })
        df_ils_lists = pd.DataFrame(results)
    
    if len(df_ils_lists) == 0:
        print("AVISO: Nenhuma lista com >= 2 itens para calcular ILS (listas)")
        report_lines.append("## Tabela 6.6: ILS (Cosseno - Listas)\n")
        report_lines.append("Nenhum dado disponível\n")
    else:
        if aggregate_by_user:
            print(f"  - {len(df_ils_lists)} usuários com ILS calculado")
            
            # Estatísticas por algoritmo (com coluna Usuários)
            table_6_6 = aggregate_like_thesis(
                df_ils_lists,
                metric_col='ils_cosine_lists',
                include_users=True,
                include_minmax=False
            )
        else:
            print(f"  - ILS global calculado para {len(df_ils_lists)} algoritmos")
            
            # Formatação global
            table_6_6 = aggregate_global(
                df_ils_lists,
                metric_col='ils_cosine_lists'
            )
        
        # Formatar e exportar
        table_6_6_formatted = format_table_for_export(table_6_6, decimal_places=3)
        output_path_6_6 = tables_dir / f"tabela_6_6_ILS_listas{output_suffix}.csv"
        table_6_6_formatted.to_csv(output_path_6_6, index=False)
        
        print(f"\nTabela salva em: {output_path_6_6}")
        print("\nPreview:")
        print(table_6_6_formatted.to_string(index=False))
        
        # Adicionar ao relatório
        report_lines.append("## Tabela 6.6: ILS (Cosseno - Listas)\n")
        report_lines.append(f"Arquivo: `{output_path_6_6.name}`\n")
        
        if aggregate_by_user and 'Usuários' in table_6_6.columns:
            report_lines.append("### Usuários por algoritmo:\n")
            for _, row in table_6_6.iterrows():
                algo = row['Algoritmo']
                n_users = int(row['Usuários'])
                report_lines.append(f"- **{algo}**: {n_users} usuários")
            
            # Exclusões
            total_lists = len(reclists)
            total_users_lists = reclists['user_id'].nunique()
            included_users_lists = len(df_ils_lists)
            excluded_lists = total_users_lists - included_users_lists
            report_lines.append(f"\n**Usuários excluídos**: {excluded_lists} (listas com < 2 itens válidos)\n")
        else:
            report_lines.append("### Agregação global:\n")
            for _, row in table_6_6.iterrows():
                algo = row['Algoritmo']
                n_lists = int(row['N Listas'])
                report_lines.append(f"- **{algo}**: {n_lists} listas")
    
    # ========================================================================
    # TABELA 6.3: RMSE ou NDCG@N
    # ========================================================================
    print("\n" + "=" * 80)
    if ranking_metric == 'rmse':
        print("TABELA 6.3: RMSE")
        metric_label = "RMSE"
        metric_col = 'rmse'
        count_col = 'N Pares'
    else:  # ndcg
        print(f"TABELA 6.3: NDCG@{ndcg_cutoff}")
        metric_label = f"NDCG@{ndcg_cutoff}"
        metric_col = 'ndcg'
        count_col = 'N Usuários'
    print("=" * 80)
    
    if ranking_metric == 'rmse':
        if aggregate_by_user:
            print("Modo: Agregação por usuário")
            print("Calculando RMSE por usuário...")
            df_ranking = compute_RMSE_user(eval_pairs)
        else:
            print("Modo: Agregação global")
            print("Calculando RMSE global...")
            df_ranking = compute_RMSE_global(eval_pairs)
    else:  # ndcg
        if aggregate_by_user:
            print("Modo: Agregação por usuário")
            print(f"Calculando NDCG@{ndcg_cutoff} por usuário...")
            df_ranking = compute_NDCG_user(reclists, eval_pairs, N=ndcg_cutoff)
        else:
            print("Modo: Agregação global")
            print(f"Calculando NDCG@{ndcg_cutoff} global...")
            df_ranking = compute_NDCG_global(reclists, eval_pairs, N=ndcg_cutoff)
    
    if len(df_ranking) == 0:
        report_lines.append(f"## Tabela 6.3: {metric_label}\n")
        report_lines.append("Nenhum dado disponível\n")
    else:
        if aggregate_by_user:
            print(f"  - {len(df_ranking)} usuários com {metric_label} calculado")
            
            # Estatísticas por algoritmo (sem coluna Usuários, com Min/Max)
            table_6_3 = aggregate_like_thesis(
                df_ranking,
                metric_col=metric_col,
                include_users=False,
                include_minmax=True
            )
        else:
            print(f"  - {metric_label} global calculado para {len(df_ranking)} algoritmos")
            
            # Formatação global
            table_6_3 = aggregate_global(
                df_ranking,
                metric_col=metric_col
            )
        
        # Formatar e exportar
        table_6_3_formatted = format_table_for_export(table_6_3, decimal_places=3)
        
        # Nome do arquivo depende da métrica
        if ranking_metric == 'rmse':
            output_path_6_3 = tables_dir / f"tabela_6_3_RMSE{output_suffix}.csv"
        else:
            output_path_6_3 = tables_dir / f"tabela_6_3_NDCG@{ndcg_cutoff}{output_suffix}.csv"
        
        table_6_3_formatted.to_csv(output_path_6_3, index=False)
        
        print(f"\nTabela salva em: {output_path_6_3}")
        print("\nPreview:")
        print(table_6_3_formatted.to_string(index=False))
        
        # Adicionar ao relatório
        report_lines.append(f"## Tabela 6.3: {metric_label}\n")
        report_lines.append(f"Arquivo: `{output_path_6_3.name}`\n")
        
        if aggregate_by_user:
            report_lines.append("### Usuários por algoritmo:\n")
            for algo in df_ranking['algorithm'].unique():
                n_users = len(df_ranking[df_ranking['algorithm'] == algo])
                algo_normalized = normalize_algorithm_name(algo)
                report_lines.append(f"- **{algo_normalized}**: {n_users} usuários")
            
            # Exclusões
            if ranking_metric == 'rmse':
                total_users = eval_pairs['user_id'].nunique()
                included_users = len(df_ranking)
                excluded = total_users - included_users
                report_lines.append(f"\n**Usuários excluídos**: {excluded} (< 2 pares de avaliação)\n")
            else:  # ndcg
                # Para NDCG, os usuários excluídos são aqueles sem itens avaliáveis
                total_users = reclists['user_id'].nunique()
                included_users = len(df_ranking)
                excluded = total_users - included_users
                report_lines.append(f"\n**Usuários excluídos**: {excluded} (sem itens avaliáveis)\n")
        else:
            report_lines.append("### Agregação global:\n")
            for _, row in df_ranking.iterrows():
                algo = normalize_algorithm_name(row['algorithm'])
                if ranking_metric == 'rmse':
                    n_items = int(row['n_pairs'])
                    report_lines.append(f"- **{algo}**: {n_items} pares")
                else:  # ndcg
                    n_users = int(row['n_users'])
                    report_lines.append(f"- **{algo}**: {n_users} usuários")
    
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
    parser.add_argument(
        '--embedding-dim',
        type=int,
        help='Dimensão do embedding (incluída no nome dos arquivos)'
    )
    parser.add_argument(
        '--aggregate-by-user',
        action='store_true',
        default=True,
        help='Agregar métricas por usuário antes de agregar por algoritmo (default: True)'
    )
    parser.add_argument(
        '--global-aggregation',
        action='store_true',
        help='Usar agregação global (sem passar por usuário). Equivalente a --no-aggregate-by-user'
    )
    parser.add_argument(
        '--ranking-metric',
        type=str,
        choices=['rmse', 'ndcg'],
        default='rmse',
        help='Métrica de ranking a processar: rmse ou ndcg (default: rmse)'
    )
    parser.add_argument(
        '--ndcg-cutoff',
        type=int,
        default=20,
        help='Cutoff N para NDCG@N (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Determinar modo de agregação
    aggregate_by_user = not args.global_aggregation if hasattr(args, 'global_aggregation') else args.aggregate_by_user
    
    print("=" * 80)
    print("GERAÇÃO DE TABELAS NO FORMATO DA TESE")
    print("=" * 80)
    print(f"Modo de agregação: {'Por usuário' if aggregate_by_user else 'Global'}")
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
        print(f"\n[INFO] Processando {len(representations_to_process)} representações")
    
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
            representation_label=label,
            embedding_dim=args.embedding_dim,
            aggregate_by_user=aggregate_by_user,
            ranking_metric=args.ranking_metric,
            ndcg_cutoff=args.ndcg_cutoff
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
