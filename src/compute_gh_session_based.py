"""
Script para calcular GH por sessão (abordagem correta da tese).

Calcula GH_session para cada (user_id, t_rec), depois agrega:
GH_user = média(GH_session).

Uso:
    python -m src.compute_gh_session_based
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.format_like_thesis import normalize_algorithm_name


def compute_jaccard(topics_a: set, topics_b: set) -> float:
    """Calcula Jaccard entre dois conjuntos de tópicos."""
    if len(topics_a) == 0 and len(topics_b) == 0:
        return 1.0
    intersection = len(topics_a & topics_b)
    union = len(topics_a | topics_b)
    return intersection / union if union > 0 else 0.0


def build_topics_dict(topics_df: pd.DataFrame) -> dict:
    """Constrói {news_id: set(topics_ativos)}."""
    topics_dict = {}
    topic_cols = [c for c in topics_df.columns if c.startswith('Topic')]
    
    for _, row in topics_df.iterrows():
        news_id = row['news_id']
        active_topics = {col for col in topic_cols if row[col] == 1}
        topics_dict[news_id] = active_topics
    
    return topics_dict


def compute_gh_session(items: list, topics_dict: dict) -> float:
    """
    Calcula GH para uma sessão.
    
    GH_session = (1/|R_session|) × Σ_{i<j} Jaccard(i,j)
    """
    if len(items) < 2:
        return np.nan
    
    jaccards = []
    for item_a, item_b in combinations(items, 2):
        topics_a = topics_dict.get(item_a, set())
        topics_b = topics_dict.get(item_b, set())
        jacc = compute_jaccard(topics_a, topics_b)
        jaccards.append(jacc)
    
    # Normalização: soma / |R_session|
    return np.sum(jaccards) / len(items)


def shapiro_p(values) -> float:
    """Calcula p-valor do teste Shapiro-Wilk."""
    values = np.array(values)
    values = values[~np.isnan(values)]
    
    if len(values) < 3:
        return np.nan
    
    try:
        _, p_value = stats.shapiro(values)
        return p_value
    except Exception:
        return np.nan


def process_representation(outputs_dir: Path = None, representation_suffix: str = None, representation_label: str = None):
    """Processa uma representação específica.
    
    Args:
        outputs_dir: Diretório de outputs (padrão: 'outputs')
        representation_suffix: Sufixo para os arquivos (ex: '_ae_features+ae_topics' ou None para default)
        representation_label: Label para exibição (ex: 'ae_features+ae_topics' ou 'default')
    """
    if outputs_dir is None:
        outputs_dir = Path("outputs")
    
    suffix = representation_suffix if representation_suffix else ""
    label = representation_label if representation_label else "default (bin_features+bin_topics)"
    
    print("=" * 80)
    print(f"PROCESSANDO: {label}")
    print("=" * 80)
    
    # Diretórios
    debug_dir = outputs_dir / "debug"
    reports_dir = outputs_dir / "reports"
    
    debug_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    # Verificar arquivos
    eval_pairs_path = outputs_dir / f"eval_pairs_assigned{suffix}.parquet"
    topics_path = outputs_dir / "canonical_topics.parquet"
    
    if not eval_pairs_path.exists():
        print(f"\nERRO: {eval_pairs_path} não encontrado")
        return 1
    if not topics_path.exists():
        print(f"\nERRO: {topics_path} não encontrado")
        return 1
    
    print("\nArquivos encontrados")
    
    # Carregar dados
    print("\nCarregando dados...")
    eval_pairs = pd.read_parquet(eval_pairs_path)
    topics = pd.read_parquet(topics_path)
    
    print(f"  - eval_pairs: {len(eval_pairs)} registros")
    print(f"  - topics: {len(topics)} notícias")
    
    # Normalizar nomes de algoritmos
    print("\nNormalizando nomes de algoritmos...")
    eval_pairs['algorithm'] = eval_pairs['algorithm'].apply(normalize_algorithm_name)
    print(f"  - Algoritmos: {', '.join(sorted(eval_pairs['algorithm'].unique()))}")
    
    # Construir dicionário de tópicos
    print("\nConstruindo dicionário de tópicos...")
    topics_dict = build_topics_dict(topics)
    print(f"  - {len(topics_dict)} notícias com tópicos")
    
    # ETAPA 1: Calcular GH por sessão
    print("\nETAPA 1: Calculando GH por sessão...")
    print("  (para cada user_id, t_rec, algorithm)")
    
    session_results = []
    
    for (user_id, t_rec, algorithm), group in eval_pairs.groupby(['user_id', 't_rec', 'algorithm']):
        # Itens da sessão (deduplicados)
        items = group['news_id'].unique().tolist()
        
        # Filtrar itens com tópicos válidos
        items_with_topics = [item for item in items if item in topics_dict and len(topics_dict[item]) > 0]
        
        if len(items_with_topics) >= 2:
            gh_session = compute_gh_session(items_with_topics, topics_dict)
            
            session_results.append({
                'user_id': user_id,
                't_rec': t_rec,
                'algorithm': algorithm,
                'n_items_session': len(items_with_topics),
                'gh_session': gh_session
            })
    
    df_sessions = pd.DataFrame(session_results)
    print(f"  {len(df_sessions)} sessões válidas (|R_session| >= 2)")
    
    # DESABILITADO: CSV de debug não é mais gerado
    # session_path = debug_dir / f"gh_sessions_level{suffix}.csv"
    # df_sessions.to_csv(session_path, index=False)
    # print(f"  Dados salvos: {session_path}")
    
    # ETAPA 2: Agregar por usuário (média de sessões)
    print("\nETAPA 2: Agregando por usuário (GH_user = média de GH_session)...")
    
    user_results = []
    
    for (user_id, algorithm), group in df_sessions.groupby(['user_id', 'algorithm']):
        gh_sessions = group['gh_session'].dropna().values
        
        if len(gh_sessions) >= 1:  # Pelo menos 1 sessão válida
            gh_user = np.mean(gh_sessions)
            n_sessions = len(gh_sessions)
            
            user_results.append({
                'user_id': user_id,
                'algorithm': algorithm,
                'gh_user': gh_user,
                'n_sessions_valid': n_sessions,
                'gh_session_std': np.std(gh_sessions, ddof=1) if n_sessions > 1 else 0.0
            })
    
    df_users = pd.DataFrame(user_results)
    print(f"  {len(df_users)} usuários processados")
    
    # DESABILITADO: CSV de debug não é mais gerado
    # user_path = debug_dir / f"gh_user_session_based{suffix}.csv"
    # df_users.to_csv(user_path, index=False)
    # print(f"  Dados salvos: {user_path}")
    
    # ETAPA 3: Agregar por algoritmo (Tabela 6.1)
    print("\nETAPA 3: Agregando por algoritmo (Tabela 6.1)...")
    
    algo_results = []
    
    # Ordem dos algoritmos da tese
    algo_order = ['knnu', 'knnu td', 'knnu mmr', 'knni', 'knni td', 'knni mmr', 'svd', 'svd td', 'svd mmr']
    
    for algo in algo_order:
        subset = df_users[df_users['algorithm'] == algo]
        
        if len(subset) == 0:
            continue
        
        gh_values = subset['gh_user'].dropna().values
        
        if len(gh_values) == 0:
            continue
        
        algo_results.append({
            'Algoritmo': algo,
            'Usuários': len(gh_values),
            'Média': np.mean(gh_values),
            'Mediana': np.median(gh_values),
            'Desvio Padrão': np.std(gh_values, ddof=1) if len(gh_values) > 1 else 0.0,
            'p-valor': shapiro_p(gh_values)
        })
    
    df_table61 = pd.DataFrame(algo_results)
    
    # Formatação
    df_table61['Média'] = df_table61['Média'].apply(lambda x: f'{x:.3f}')
    df_table61['Mediana'] = df_table61['Mediana'].apply(lambda x: f'{x:.3f}')
    df_table61['Desvio Padrão'] = df_table61['Desvio Padrão'].apply(lambda x: f'{x:.3f}')
    df_table61['p-valor'] = df_table61['p-valor'].apply(lambda x: f'{x:.4f}' if not pd.isna(x) else 'NaN')
    
    # Exportar Tabela 6.1
    table61_path = outputs_dir / f"tabela_6_1_GH_interacao_session_based{suffix}.csv"
    df_table61.to_csv(table61_path, index=False)
    
    print("\n" + "=" * 80)
    print("TABELA 6.1 - GH POR INTERAÇÃO (BASEADO EM SESSÕES)")
    print("=" * 80)
    print(df_table61.to_string(index=False))
    
    print(f"\nTabela salva: {table61_path}")
    
    # ETAPA 4: Comparação com abordagem global
    print("\nETAPA 4: Comparação com abordagem global...")
    
    # Carregar dados da auditoria (abordagem global)
    audit_path = debug_dir / f"R_size_audit{suffix}.csv"
    
    if audit_path.exists():
        df_global = pd.read_csv(audit_path)
        
        # Merge para comparar
        df_comparison = df_users.merge(
            df_global[['user_id', 'algorithm', 'gh_global', 'n_R_items']],
            on=['user_id', 'algorithm'],
            how='inner'
        )
        
        print(f"\n  Comparando {len(df_comparison)} usuários...")
        
        comparison_stats = []
        for algo in algo_order:
            subset = df_comparison[df_comparison['algorithm'] == algo]
            
            if len(subset) == 0:
                continue
            
            comparison_stats.append({
                'algorithm': algo,
                'n_users': len(subset),
                'gh_session_mean': subset['gh_user'].mean(),
                'gh_session_std': subset['gh_user'].std(ddof=1),
                'gh_global_mean': subset['gh_global'].mean(),
                'gh_global_std': subset['gh_global'].std(ddof=1),
                'std_reduction': 1 - subset['gh_user'].std(ddof=1) / subset['gh_global'].std(ddof=1)
            })
        
        df_comp_stats = pd.DataFrame(comparison_stats)
        
        print("\n" + "=" * 80)
        print("COMPARAÇÃO: SESSÃO vs GLOBAL")
        print("=" * 80)
        print(df_comp_stats.to_string(index=False))
        
        # DESABILITADO: CSV de comparação não é mais gerado
        # comp_path = debug_dir / f"gh_session_vs_global_comparison{suffix}.csv"
        # df_comp_stats.to_csv(comp_path, index=False)
        # print(f"\nComparação salva: {comp_path}")
    
    # DESABILITADO: Relatório não é mais gerado
    # print("\nGerando relatório...")
    # report_lines = []
    # 
    # report_lines.append("# Tabela 6.1 - GH por Sessão (Abordagem da Tese)\n")
    # report_lines.append("## Metodologia\n")
    # report_lines.append("1. **GH por sessão**: Para cada (user_id, t_rec):")
    # report_lines.append("   - R_session = itens expostos e avaliados naquela sessão")
    # report_lines.append("   - GH_session = (1/|R_session|) × Σ_{i<j} Jaccard(i,j)")
    # report_lines.append("   - Exigir |R_session| >= 2\n")
    # report_lines.append("2. **GH por usuário**: GH_user = média(GH_session)")
    # report_lines.append("   - Exigir pelo menos 1 sessão válida\n")
    # report_lines.append("3. **GH por algoritmo**: Agregação por usuário (média, mediana, DP, p-valor)\n")
    # 
    # report_lines.append("## Dados Processados\n")
    # report_lines.append(f"- **Sessões válidas**: {len(df_sessions)}")
    # report_lines.append(f"- **Usuários válidos**: {len(df_users)}")
    # report_lines.append(f"- **Algoritmos**: {df_users['algorithm'].nunique()}\n")
    # 
    # report_lines.append("## Distribuição de Sessões por Usuário\n")
    # report_lines.append("| Algoritmo | N usuários | Sessões Médias | Sessões Mediana | Sessões Max |")
    # report_lines.append("|-----------|-----------|---------------|----------------|-------------|")
    # for algo in algo_order:
    #     subset = df_users[df_users['algorithm'] == algo]['n_sessions_valid']
    #     if len(subset) > 0:
    #         report_lines.append(
    #             f"| {algo:9s} | {len(subset):10d} | {subset.mean():14.1f} | "
    #             f"{subset.median():15.1f} | {subset.max():11d} |"
    #         )
    # report_lines.append("\n")
    # 
    # report_lines.append("## Tabela 6.1 - GH por Interação (Baseado em Sessões)\n")
    # report_lines.append("```")
    # report_lines.append(df_table61.to_string(index=False))
    # report_lines.append("```\n")
    # 
    # # Análise comparativa
    # if audit_path.exists():
    #     report_lines.append("## Comparação: Sessão vs. Global\n")
    #     report_lines.append("| Algoritmo | GH Sessão (Média±DP) | GH Global (Média±DP) | Redução DP |")
    #     report_lines.append("|-----------|---------------------|---------------------|------------|")
    #     for _, row in df_comp_stats.iterrows():
    #         report_lines.append(
    #             f"| {row['algorithm']:9s} | {row['gh_session_mean']:.3f}±{row['gh_session_std']:.3f} | "
    #             f"{row['gh_global_mean']:.3f}±{row['gh_global_std']:.3f} | "
    #             f"{row['std_reduction']*100:7.1f}% |"
    #         )
    #     report_lines.append("\n")
    #     
    #     # Diagnóstico
    #     report_lines.append("### Diagnóstico\n")
    #     
    #     avg_std_reduction = df_comp_stats['std_reduction'].mean()
    #     
    #     if avg_std_reduction > 0.5:
    #         report_lines.append(f"**Redução média de DP: {avg_std_reduction*100:.1f}%**\n")
    #         report_lines.append("A abordagem por sessão reduziu drasticamente a variância,")
    #         report_lines.append("confirmando que a acumulação de itens de múltiplas sessões")
    #         report_lines.append("estava distorcendo a métrica.\n")
    #     else:
    #         report_lines.append(f"**Redução média de DP: {avg_std_reduction*100:.1f}%**\n")
    #         report_lines.append("A redução de variância foi menor que o esperado.\n")
    #     
    #     # Checar valores GH
    #     gh_session_mean = df_comp_stats['gh_session_mean'].mean()
    #     gh_global_mean = df_comp_stats['gh_global_mean'].mean()
    #     
    #     report_lines.append(f"**Média global GH (sessão)**: {gh_session_mean:.3f}")
    #     report_lines.append(f"**Média global GH (global)**: {gh_global_mean:.3f}")
    #     report_lines.append(f"**Intervalo da tese**: [0.72, 0.76]\n")
    #     
    #     if 0.65 <= gh_session_mean <= 0.85:
    #         report_lines.append("**Escala aproximada da tese alcançada!**\n")
    #     elif gh_session_mean > 1.0:
    #         report_lines.append("**Valores ainda acima de 1.0 - revisar normalização**\n")
    # 
    # # Salvar relatório
    # report_path = reports_dir / f"table61_session_based_report{suffix}.md"
    # report_path.write_text("\n".join(report_lines), encoding='utf-8')
    # 
    # print(f"\nRelatório salvo: {report_path}")
    
    print("\n" + "=" * 80)
    print("PROCESSO CONCLUÍDO")
    print("=" * 80)
    print(f"\nArquivo gerado para {label}:")
    print(f"  1. {table61_path} (Tabela 6.1 final)")
    # print(f"  1. {session_path} (sessões individuais)")
    # print(f"  2. {user_path} (usuários agregados)")
    # print(f"  3. {table61_path} (Tabela 6.1 final)")
    # if audit_path.exists():
    #     print(f"  4. {comp_path} (comparação sessão vs global)")
    # print(f"  5. {report_path} (relatório completo)")
    
    return True


def main():
    """Main com suporte a múltiplas representações."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calcular GH por sessão para diferentes representações'
    )
    parser.add_argument(
        '--representations',
        nargs='+',
        help='Lista de sufixos de representação (ex: ae_features+ae_topics bin_features+ae_topics)'
    )
    parser.add_argument(
        '--output-dir',
        default='outputs',
        help='Diretório de outputs (padrão: outputs)'
    )
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.output_dir)
    
    # Auto-detectar representações se não especificadas
    if args.representations:
        representations = [(f"_{rep}", rep) for rep in args.representations]
    else:
        # Buscar todos os arquivos eval_pairs_assigned*.parquet
        eval_files = list(outputs_dir.glob("eval_pairs_assigned*.parquet"))
        
        if not eval_files:
            print("ERRO: Nenhum arquivo eval_pairs_assigned encontrado")
            return 1
        
        representations = []
        for file in eval_files:
            # Extrair sufixo do nome do arquivo
            filename = file.stem  # eval_pairs_assigned ou eval_pairs_assigned_XXX
            if filename == "eval_pairs_assigned":
                # Arquivo default (sem sufixo)
                representations.append((None, "default"))
            else:
                # Extrair sufixo após 'eval_pairs_assigned_'
                suffix = filename.replace("eval_pairs_assigned_", "")
                representations.append((f"_{suffix}", suffix))
    
    print("=" * 80)
    print("CÁLCULO DE GH POR SESSÃO (Abordagem da Tese)")
    print("=" * 80)
    print(f"\n⚙ Processando {len(representations)} representações\n")
    
    success_count = 0
    
    for idx, (suffix, label) in enumerate(representations, 1):
        print(f"\n[{idx}/{len(representations)}]\n")
        
        try:
            success = process_representation(
                outputs_dir=outputs_dir,
                representation_suffix=suffix,
                representation_label=label
            )
            
            if success:
                success_count += 1
        except Exception as e:
            print(f"\nERRO ao processar {label}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    if success_count == len(representations):
        print("CONCLUÍDO! Todas as representações foram processadas com sucesso.")
    else:
        print(f"CONCLUÍDO COM AVISOS: {success_count}/{len(representations)} processadas com sucesso.")
    print("=" * 80)
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
