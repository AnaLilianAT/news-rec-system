"""
Módulo para verificações de sanidade e geração de relatórios.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional


def generate_ingestion_report(
    datasets: dict,
    output_path: str = 'outputs/reports/ingestion_report.md'
):
    """
    Gera relatório de ingestão de dados.
    
    Args:
        datasets: Dicionário com DataFrames originais
        output_path: Caminho do arquivo de saída
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Ingestão de Dados\n\n")
        f.write(f"**Data de geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 1. Resumo dos Datasets Carregados\n\n")
        f.write("| Dataset | Linhas | Colunas | Tamanho Memória |\n")
        f.write("|---------|--------|---------|------------------|\n")
        
        for name, df in datasets.items():
            mem_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            f.write(f"| {name} | {len(df):,} | {len(df.columns)} | {mem_usage:.2f} MB |\n")
        
        f.write("\n## 2. Estrutura dos Datasets\n\n")
        
        for name, df in datasets.items():
            f.write(f"### 2.{list(datasets.keys()).index(name) + 1} {name}\n\n")
            f.write(f"**Colunas ({len(df.columns)}):** {', '.join(df.columns.tolist())}\n\n")
            
            # Tipos de dados
            f.write("**Tipos de dados:**\n")
            for col, dtype in df.dtypes.items():
                f.write(f"- `{col}`: {dtype}\n")
            f.write("\n")
            
            # Valores nulos
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                f.write("**Valores nulos:**\n")
                for col, count in null_counts[null_counts > 0].items():
                    pct = count / len(df) * 100
                    f.write(f"- `{col}`: {count:,} ({pct:.1f}%)\n")
                f.write("\n")
            else:
                f.write("**Valores nulos:** Nenhum\n\n")
        
        f.write("---\n\n")
        f.write("*Relatório gerado automaticamente pelo pipeline de ingestão*\n")
    
    print(f"✓ Relatório de ingestão salvo em: {output_path}")


def generate_data_quality_report(
    df_interactions: pd.DataFrame,
    df_rec_sessions: pd.DataFrame,
    df_features: pd.DataFrame,
    df_topics: pd.DataFrame,
    news_df: pd.DataFrame,
    output_path: str = 'outputs/reports/data_quality_report.md'
):
    """
    Gera relatório de qualidade de dados e validações.
    
    Args:
        df_interactions: Tabela de interações
        df_rec_sessions: Tabela de sessões de recomendação
        df_features: Tabela de features
        df_topics: Tabela de topics
        news_df: DataFrame original de notícias
        output_path: Caminho do arquivo de saída
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Qualidade de Dados\n\n")
        f.write(f"**Data de geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # ========================================
        # 1. VALIDAÇÃO DE CONSISTÊNCIA
        # ========================================
        f.write("## 1. Validação de Consistência\n\n")
        
        f.write("### 1.1 Cobertura de Notícias\n\n")
        
        news_in_catalog = news_df['id'].nunique()
        news_in_features = df_features['news_id'].nunique()
        news_in_interactions = df_interactions['news_id'].nunique()
        
        f.write(f"- **Notícias no catálogo (news.csv):** {news_in_catalog:,}\n")
        f.write(f"- **Notícias com features:** {news_in_features:,}\n")
        f.write(f"- **Notícias com interações:** {news_in_interactions:,}\n\n")
        
        # Verificar sobreposição
        catalog_ids = set(news_df['id'].unique())
        feature_ids = set(df_features['news_id'].unique())
        interaction_ids = set(df_interactions['news_id'].unique())
        
        features_not_in_catalog = len(feature_ids - catalog_ids)
        interactions_not_in_catalog = len(interaction_ids - catalog_ids)
        interactions_without_features = len(interaction_ids - feature_ids)
        
        f.write("**Análise de sobreposição:**\n")
        f.write(f"- Features sem correspondência no catálogo: {features_not_in_catalog}\n")
        f.write(f"- Interações sem correspondência no catálogo: {interactions_not_in_catalog}\n")
        f.write(f"- Interações sem features: {interactions_without_features}\n\n")
        
        if interactions_not_in_catalog > 0:
            pct = interactions_not_in_catalog / news_in_interactions * 100
            f.write(f"⚠️ **Atenção:** {pct:.1f}% das notícias com interações não estão no catálogo\n\n")
        
        # ========================================
        # 2. DISTRIBUIÇÕES
        # ========================================
        f.write("## 2. Distribuições de Dados\n\n")
        
        f.write("### 2.1 Interações (Avaliações Explícitas)\n\n")
        f.write(f"- **Total de interações:** {len(df_interactions):,}\n")
        f.write(f"- **Usuários únicos:** {df_interactions['user_id'].nunique():,}\n")
        f.write(f"- **Notícias únicas:** {df_interactions['news_id'].nunique():,}\n")
        f.write(f"- **Período:** {df_interactions['rating_when'].min()} a {df_interactions['rating_when'].max()}\n\n")
        
        f.write("**Distribuição de ratings:**\n")
        rating_dist = df_interactions['rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            pct = count / len(df_interactions) * 100
            bar = '█' * int(pct / 2)
            f.write(f"- Rating {rating:.1f}: {count:6,} ({pct:5.1f}%) {bar}\n")
        f.write(f"\n**Estatísticas de rating:**\n")
        f.write(f"- Média: {df_interactions['rating'].mean():.2f}\n")
        f.write(f"- Mediana: {df_interactions['rating'].median():.2f}\n")
        f.write(f"- Desvio padrão: {df_interactions['rating'].std():.2f}\n\n")
        
        # Interações por usuário
        interactions_per_user = df_interactions.groupby('user_id').size()
        f.write("**Interações por usuário:**\n")
        f.write(f"- Média: {interactions_per_user.mean():.1f}\n")
        f.write(f"- Mediana: {interactions_per_user.median():.0f}\n")
        f.write(f"- Mínimo: {interactions_per_user.min()}\n")
        f.write(f"- Máximo: {interactions_per_user.max()}\n\n")
        
        # ========================================
        f.write("### 2.2 Sessões de Recomendação\n\n")
        f.write(f"- **Total de sessões:** {len(df_rec_sessions):,}\n")
        f.write(f"- **Usuários únicos:** {df_rec_sessions['user_id'].nunique():,}\n")
        f.write(f"- **Período:** {df_rec_sessions['generated_when'].min()} a {df_rec_sessions['generated_when'].max()}\n\n")
        
        f.write("**Distribuição de list_size:**\n")
        list_size_dist = df_rec_sessions['list_size'].value_counts().sort_index(ascending=False).head(15)
        for size, count in list_size_dist.items():
            pct = count / len(df_rec_sessions) * 100
            bar = '█' * int(pct / 2)
            f.write(f"- {size:3d} itens: {count:5,} sessões ({pct:5.1f}%) {bar}\n")
        
        f.write("\n**Sessões por flag diversified:**\n")
        div_dist = df_rec_sessions['diversifed'].value_counts()
        for div_flag, count in div_dist.items():
            pct = count / len(df_rec_sessions) * 100
            f.write(f"- diversifed={div_flag}: {count:,} ({pct:.1f}%)\n")
        f.write("\n")
        
        # Sessões por usuário
        sessions_per_user = df_rec_sessions.groupby('user_id').size()
        f.write("**Sessões por usuário:**\n")
        f.write(f"- Média: {sessions_per_user.mean():.1f}\n")
        f.write(f"- Mediana: {sessions_per_user.median():.0f}\n")
        f.write(f"- Mínimo: {sessions_per_user.min()}\n")
        f.write(f"- Máximo: {sessions_per_user.max()}\n\n")
        
        # ========================================
        f.write("### 2.3 Features e Topics\n\n")
        f.write(f"- **Notícias com features:** {len(df_features):,}\n")
        f.write(f"- **Número de features:** {len(df_features.columns) - 1}\n")
        f.write(f"- **Notícias com topics:** {len(df_topics):,}\n")
        
        # Topics
        topic_cols = [col for col in df_topics.columns if col.startswith('Topic')]
        topic_counts = df_topics[topic_cols].sum(axis=1)
        
        f.write(f"\n**Distribuição de topics por notícia:**\n")
        f.write(f"- Média: {topic_counts.mean():.2f}\n")
        f.write(f"- Mediana: {topic_counts.median():.0f}\n")
        f.write(f"- Mínimo: {topic_counts.min()}\n")
        f.write(f"- Máximo: {topic_counts.max()}\n\n")
        
        f.write("**Frequência de cada topic:**\n")
        for topic_col in sorted(topic_cols, key=lambda x: int(x.replace('Topic', ''))):
            count = df_topics[topic_col].sum()
            pct = count / len(df_topics) * 100
            f.write(f"- {topic_col}: {count:,} notícias ({pct:.1f}%)\n")
        
        # Features numéricas
        if 'polaridade' in df_features.columns and 'subjetividade' in df_features.columns:
            f.write("\n**Features de sentimento:**\n")
            f.write(f"- Polaridade: média={df_features['polaridade'].mean():.3f}, "
                   f"std={df_features['polaridade'].std():.3f}, "
                   f"min={df_features['polaridade'].min():.3f}, "
                   f"max={df_features['polaridade'].max():.3f}\n")
            f.write(f"- Subjetividade: média={df_features['subjetividade'].mean():.3f}, "
                   f"std={df_features['subjetividade'].std():.3f}, "
                   f"min={df_features['subjetividade'].min():.3f}, "
                   f"max={df_features['subjetividade'].max():.3f}\n")
        
        f.write("\n")
        
        # ========================================
        # 3. QUALIDADE DE DADOS
        # ========================================
        f.write("## 3. Qualidade de Dados\n\n")
        
        f.write("### 3.1 Valores Ausentes\n\n")
        
        tables = {
            'df_interactions': df_interactions,
            'df_rec_sessions': df_rec_sessions,
            'df_features': df_features,
            'df_topics': df_topics
        }
        
        for name, df in tables.items():
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                f.write(f"**{name}:**\n")
                for col, count in null_counts[null_counts > 0].items():
                    pct = count / len(df) * 100
                    f.write(f"- `{col}`: {count:,} ({pct:.1f}%)\n")
                f.write("\n")
            else:
                f.write(f"**{name}:** ✓ Nenhum valor ausente\n\n")
        
        # ========================================
        f.write("### 3.2 Duplicatas\n\n")
        
        # Verificar duplicatas em interactions
        dup_interactions = df_interactions.duplicated(subset=['user_id', 'news_id', 'rating_when']).sum()
        f.write(f"- **df_interactions:** {dup_interactions} duplicatas\n")
        
        # Verificar duplicatas em rec_sessions
        dup_sessions = df_rec_sessions.duplicated(subset=['user_id', 'generated_when', 'diversifed']).sum()
        f.write(f"- **df_rec_sessions:** {dup_sessions} duplicatas\n")
        
        # Verificar duplicatas em features
        dup_features = df_features.duplicated(subset=['news_id']).sum()
        f.write(f"- **df_features:** {dup_features} duplicatas\n")
        
        # Verificar duplicatas em topics
        dup_topics = df_topics.duplicated(subset=['news_id']).sum()
        f.write(f"- **df_topics:** {dup_topics} duplicatas\n\n")
        
        # ========================================
        # 4. RECOMENDAÇÕES
        # ========================================
        f.write("## 4. Recomendações\n\n")
        
        issues = []
        
        if interactions_not_in_catalog > 0:
            issues.append(f"⚠️ Existem {interactions_not_in_catalog} notícias nas interações que não estão no catálogo")
        
        if interactions_without_features > 0:
            issues.append(f"⚠️ Existem {interactions_without_features} notícias nas interações sem features")
        
        if any(null_counts.sum() > 0 for _, null_counts in [(name, df.isnull().sum()) for name, df in tables.items()]):
            issues.append("⚠️ Existem valores ausentes em algumas tabelas - considere estratégias de imputação")
        
        if dup_interactions > 0 or dup_sessions > 0:
            issues.append("⚠️ Existem registros duplicados - considere deduplificação")
        
        if len(issues) > 0:
            for issue in issues:
                f.write(f"- {issue}\n")
        else:
            f.write("✓ **Nenhum problema crítico detectado!**\n\n")
            f.write("Os dados estão consistentes e prontos para uso no pipeline de recomendação.\n")
        
        f.write("\n---\n\n")
        f.write("*Relatório gerado automaticamente pelo pipeline de validação*\n")
    
    print(f"✓ Relatório de qualidade salvo em: {output_path}")


def run_sanity_checks(
    df_interactions: pd.DataFrame,
    df_rec_sessions: pd.DataFrame,
    df_features: pd.DataFrame,
    df_topics: pd.DataFrame,
    news_df: pd.DataFrame
):
    """
    Executa todas as verificações de sanidade e gera relatórios.
    
    Args:
        df_interactions: Tabela de interações
        df_rec_sessions: Tabela de sessões de recomendação
        df_features: Tabela de features
        df_topics: Tabela de topics
        news_df: DataFrame original de notícias
    """
    print("\n" + "="*70)
    print("EXECUTANDO VERIFICAÇÕES DE SANIDADE")
    print("="*70)
    
    # Gerar relatório de qualidade
    generate_data_quality_report(
        df_interactions=df_interactions,
        df_rec_sessions=df_rec_sessions,
        df_features=df_features,
        df_topics=df_topics,
        news_df=news_df
    )
    
    print("\n" + "="*70)
    print("VERIFICAÇÕES CONCLUÍDAS")
    print("="*70)


if __name__ == '__main__':
    # Este módulo é chamado pelo build_canonical_tables
    print("Este módulo deve ser executado via build_canonical_tables.py")
