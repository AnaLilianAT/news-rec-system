"""
Módulo para construir checkpoints de replay temporal.

Define checkpoints baseados nos momentos reais de recomendação (generated_when),
permitindo replay fiel do comportamento do sistema ao longo do tempo.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple


def load_canonical_data(base_path='outputs'):
    """
    Carrega tabelas canônicas necessárias para replay.
    
    Args:
        base_path: Diretório base dos arquivos parquet
    
    Returns:
        Tupla com (df_rec_sessions, df_interactions)
    """
    base_path = Path(base_path)
    
    print("\n" + "="*70)
    print("CARREGANDO DADOS CANÔNICOS")
    print("="*70)
    
    df_rec_sessions = pd.read_parquet(base_path / 'canonical_rec_sessions.parquet')
    df_interactions = pd.read_parquet(base_path / 'canonical_interactions.parquet')
    
    print(f"✓ Sessões de recomendação: {len(df_rec_sessions):,} registros")
    print(f"✓ Interações: {len(df_interactions):,} registros")
    
    return df_rec_sessions, df_interactions


def identify_candidate_pools(df_rec_sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica sessões que representam candidate pools (não-diversificadas, ~100 itens).
    
    Candidate pools são sessões com:
    - diversifed = False
    - list_size >= 97 (idealmente 100)
    
    Args:
        df_rec_sessions: DataFrame com sessões de recomendação
    
    Returns:
        DataFrame filtrado com apenas candidate pools
    """
    print("\n" + "="*70)
    print("IDENTIFICANDO CANDIDATE POOLS")
    print("="*70)
    
    # Filtrar sessões não-diversificadas com list_size >= 97
    candidate_pools = df_rec_sessions[
        (df_rec_sessions['diversifed'] == False) & 
        (df_rec_sessions['list_size'] >= 97)
    ].copy()
    
    print(f"✓ Candidate pools identificados: {len(candidate_pools):,} sessões")
    print(f"  → De um total de {len(df_rec_sessions):,} sessões")
    print(f"  → Percentual: {len(candidate_pools)/len(df_rec_sessions)*100:.1f}%")
    
    # Distribuição de list_size
    print("\n  Distribuição de candidate_size:")
    size_dist = candidate_pools['list_size'].value_counts().sort_index(ascending=False)
    for size, count in size_dist.items():
        pct = count / len(candidate_pools) * 100
        print(f"    {size:3d} itens: {count:5d} pools ({pct:5.1f}%)")
    
    return candidate_pools


def build_replay_checkpoints(
    df_rec_sessions: pd.DataFrame, 
    df_interactions: pd.DataFrame
) -> pd.DataFrame:
    """
    Constrói checkpoints de replay temporal.
    
    Para cada usuário:
    - Ordena suas sessões por generated_when
    - Define t_next_rec como o próximo generated_when do mesmo usuário
    - Identifica candidate_pool (sessão não-diversificada com list_size >= 97)
    
    Args:
        df_rec_sessions: DataFrame com sessões de recomendação
        df_interactions: DataFrame com interações
    
    Returns:
        DataFrame com checkpoints de replay
    """
    print("\n" + "="*70)
    print("CONSTRUINDO REPLAY CHECKPOINTS")
    print("="*70)
    
    # Identificar candidate pools
    candidate_pools = identify_candidate_pools(df_rec_sessions)
    
    # Criar um dicionário para acesso rápido aos candidate pools
    # Key: (user_id, generated_when), Value: news_ids e list_size
    pool_dict = {}
    for _, row in candidate_pools.iterrows():
        key = (row['user_id'], row['generated_when'])
        pool_dict[key] = {
            'news_ids': row['news_ids'],
            'list_size': row['list_size']
        }
    
    print(f"\n✓ Dicionário de candidate pools criado: {len(pool_dict):,} entradas")
    
    # Agrupar sessões por usuário e obter timestamps únicos
    user_sessions = df_rec_sessions.groupby('user_id')['generated_when'].apply(
        lambda x: sorted(set(x.tolist()))  # Usar set para remover duplicatas
    ).to_dict()
    
    print(f"✓ Usuários com sessões: {len(user_sessions):,}")
    
    # Construir checkpoints
    checkpoints = []
    total_checkpoints = 0
    checkpoints_with_pool = 0
    
    for user_id, timestamps in user_sessions.items():
        for i, t_rec in enumerate(timestamps):
            # Determinar t_next_rec
            if i + 1 < len(timestamps):
                t_next_rec = timestamps[i + 1]
            else:
                # Último checkpoint do usuário: usar fim do dataset
                t_next_rec = df_interactions['rating_when'].max()
            
            # Verificar se existe candidate pool para este checkpoint
            key = (user_id, t_rec)
            if key in pool_dict:
                pool_info = pool_dict[key]
                candidate_news_ids = pool_info['news_ids']
                candidate_size = pool_info['list_size']
                has_candidate_pool = True
                checkpoints_with_pool += 1
            else:
                # Checkpoint sem candidate pool
                candidate_news_ids = []
                candidate_size = 0
                has_candidate_pool = False
            
            checkpoints.append({
                'user_id': user_id,
                't_rec': t_rec,
                't_next_rec': t_next_rec,
                'candidate_news_ids': candidate_news_ids,
                'candidate_size': candidate_size,
                'has_candidate_pool': has_candidate_pool
            })
            total_checkpoints += 1
    
    df_checkpoints = pd.DataFrame(checkpoints)
    
    print(f"\n✓ Checkpoints construídos: {len(df_checkpoints):,} registros")
    print(f"  → Com candidate pool: {checkpoints_with_pool:,} ({checkpoints_with_pool/len(df_checkpoints)*100:.1f}%)")
    print(f"  → Sem candidate pool: {len(df_checkpoints) - checkpoints_with_pool:,} ({(len(df_checkpoints) - checkpoints_with_pool)/len(df_checkpoints)*100:.1f}%)")
    print(f"  → Usuários únicos: {df_checkpoints['user_id'].nunique():,}")
    
    return df_checkpoints


def generate_replay_report(
    df_checkpoints: pd.DataFrame,
    df_rec_sessions: pd.DataFrame,
    df_interactions: pd.DataFrame,
    output_path: str = 'outputs/reports/replay_report.md'
):
    """
    Gera relatório detalhado sobre checkpoints de replay.
    
    Args:
        df_checkpoints: DataFrame com checkpoints
        df_rec_sessions: DataFrame original com sessões
        df_interactions: DataFrame com interações
        output_path: Caminho do arquivo de saída
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Replay Temporal\n\n")
        f.write(f"**Data de geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # ========================================
        # 1. VISÃO GERAL DOS CHECKPOINTS
        # ========================================
        f.write("## 1. Visão Geral dos Checkpoints\n\n")
        
        total_checkpoints = len(df_checkpoints)
        checkpoints_with_pool = df_checkpoints['has_candidate_pool'].sum()
        checkpoints_without_pool = total_checkpoints - checkpoints_with_pool
        unique_users = df_checkpoints['user_id'].nunique()
        
        f.write(f"- **Total de checkpoints:** {total_checkpoints:,}\n")
        f.write(f"- **Checkpoints com candidate pool:** {checkpoints_with_pool:,} ({checkpoints_with_pool/total_checkpoints*100:.1f}%)\n")
        f.write(f"- **Checkpoints sem candidate pool:** {checkpoints_without_pool:,} ({checkpoints_without_pool/total_checkpoints*100:.1f}%)\n")
        f.write(f"- **Usuários únicos:** {unique_users:,}\n\n")
        
        # Período temporal
        t_min = df_checkpoints['t_rec'].min()
        t_max = df_checkpoints['t_rec'].max()
        duration = (t_max - t_min).days
        
        f.write(f"**Período temporal:**\n")
        f.write(f"- Início: {t_min}\n")
        f.write(f"- Fim: {t_max}\n")
        f.write(f"- Duração: {duration} dias\n\n")
        
        # ========================================
        # 2. DISTRIBUIÇÃO DE CANDIDATE POOLS
        # ========================================
        f.write("## 2. Distribuição de Candidate Pools\n\n")
        
        # Filtrar checkpoints com pools
        with_pools = df_checkpoints[df_checkpoints['has_candidate_pool'] == True]
        
        if len(with_pools) > 0:
            f.write("### 2.1 Distribuição de candidate_size\n\n")
            size_dist = with_pools['candidate_size'].value_counts().sort_index(ascending=False)
            
            f.write("| Candidate Size | Checkpoints | Percentual |\n")
            f.write("|----------------|-------------|------------|\n")
            for size, count in size_dist.items():
                pct = count / len(with_pools) * 100
                bar = '█' * int(pct / 2)
                f.write(f"| {size:3d} itens | {count:5,} | {pct:5.1f}% {bar} |\n")
            
            f.write(f"\n**Estatísticas:**\n")
            f.write(f"- Média: {with_pools['candidate_size'].mean():.1f} itens\n")
            f.write(f"- Mediana: {with_pools['candidate_size'].median():.0f} itens\n")
            f.write(f"- Mínimo: {with_pools['candidate_size'].min()} itens\n")
            f.write(f"- Máximo: {with_pools['candidate_size'].max()} itens\n\n")
        else:
            f.write("⚠️ Nenhum checkpoint com candidate pool encontrado.\n\n")
        
        # ========================================
        # 3. CHECKPOINTS POR USUÁRIO
        # ========================================
        f.write("## 3. Checkpoints por Usuário\n\n")
        
        checkpoints_per_user = df_checkpoints.groupby('user_id').size()
        checkpoints_with_pool_per_user = df_checkpoints[df_checkpoints['has_candidate_pool'] == True].groupby('user_id').size()
        
        f.write(f"**Total de checkpoints por usuário:**\n")
        f.write(f"- Média: {checkpoints_per_user.mean():.1f} checkpoints/usuário\n")
        f.write(f"- Mediana: {checkpoints_per_user.median():.0f} checkpoints/usuário\n")
        f.write(f"- Mínimo: {checkpoints_per_user.min()} checkpoints\n")
        f.write(f"- Máximo: {checkpoints_per_user.max()} checkpoints\n\n")
        
        f.write(f"**Checkpoints com pool por usuário:**\n")
        users_with_pools = len(checkpoints_with_pool_per_user)
        f.write(f"- Usuários com ≥1 checkpoint com pool: {users_with_pools:,} ({users_with_pools/unique_users*100:.1f}%)\n")
        if users_with_pools > 0:
            f.write(f"- Média: {checkpoints_with_pool_per_user.mean():.1f} checkpoints/usuário\n")
            f.write(f"- Mediana: {checkpoints_with_pool_per_user.median():.0f} checkpoints/usuário\n\n")
        
        # Distribuição de checkpoints por usuário
        f.write("**Distribuição (top 10):**\n")
        dist = checkpoints_per_user.value_counts().sort_index(ascending=False).head(10)
        for n_checkpoints, n_users in dist.items():
            pct = n_users / unique_users * 100
            f.write(f"- {n_checkpoints:3d} checkpoints: {n_users:3,} usuários ({pct:5.1f}%)\n")
        f.write("\n")
        
        # ========================================
        # 4. COBERTURA DE CANDIDATE POOLS
        # ========================================
        f.write("## 4. Cobertura de Candidate Pools\n\n")
        
        # Verificar quais usuários têm pelo menos um checkpoint com pool
        users_with_pool = df_checkpoints[df_checkpoints['has_candidate_pool'] == True]['user_id'].unique()
        users_without_pool = set(df_checkpoints['user_id'].unique()) - set(users_with_pool)
        
        f.write(f"- **Usuários com ≥1 candidate pool:** {len(users_with_pool):,} ({len(users_with_pool)/unique_users*100:.1f}%)\n")
        f.write(f"- **Usuários sem candidate pool:** {len(users_without_pool):,} ({len(users_without_pool)/unique_users*100:.1f}%)\n\n")
        
        # ========================================
        # 5. JANELAS DE AVALIAÇÃO
        # ========================================
        f.write("## 5. Janelas de Avaliação (t_rec → t_next_rec)\n\n")
        
        # Calcular duração das janelas em dias
        df_checkpoints['window_duration'] = (df_checkpoints['t_next_rec'] - df_checkpoints['t_rec']).dt.total_seconds() / 86400
        
        f.write(f"**Duração das janelas:**\n")
        f.write(f"- Média: {df_checkpoints['window_duration'].mean():.1f} dias\n")
        f.write(f"- Mediana: {df_checkpoints['window_duration'].median():.1f} dias\n")
        f.write(f"- Mínimo: {df_checkpoints['window_duration'].min():.1f} dias\n")
        f.write(f"- Máximo: {df_checkpoints['window_duration'].max():.1f} dias\n\n")
        
        # Distribuição de durações
        f.write("**Distribuição de durações:**\n")
        bins = [0, 1, 7, 14, 30, 60, 999]
        labels = ['<1 dia', '1-7 dias', '7-14 dias', '14-30 dias', '30-60 dias', '>60 dias']
        df_checkpoints['duration_bin'] = pd.cut(df_checkpoints['window_duration'], bins=bins, labels=labels)
        
        for label in labels:
            count = (df_checkpoints['duration_bin'] == label).sum()
            pct = count / len(df_checkpoints) * 100
            bar = '█' * int(pct / 2)
            f.write(f"- {label:12s}: {count:5,} checkpoints ({pct:5.1f}%) {bar}\n")
        f.write("\n")
        
        # ========================================
        # 6. VALIDAÇÕES
        # ========================================
        f.write("## 6. Validações\n\n")
        
        # Verificar ordenação temporal
        temporal_errors = 0
        for _, row in df_checkpoints.iterrows():
            if row['t_next_rec'] <= row['t_rec']:
                temporal_errors += 1
        
        if temporal_errors == 0:
            f.write("✓ **Ordenação temporal:** Todos os checkpoints têm t_next_rec > t_rec\n\n")
        else:
            f.write(f"⚠️ **Ordenação temporal:** {temporal_errors} checkpoints com t_next_rec ≤ t_rec\n\n")
        
        # Verificar consistência de candidate_size
        size_errors = 0
        for _, row in df_checkpoints.iterrows():
            if row['has_candidate_pool']:
                expected_size = len(row['candidate_news_ids'])
                if row['candidate_size'] != expected_size:
                    size_errors += 1
        
        if size_errors == 0:
            f.write("✓ **Consistência de candidate_size:** Todos os tamanhos estão corretos\n\n")
        else:
            f.write(f"⚠️ **Consistência de candidate_size:** {size_errors} checkpoints com inconsistência\n\n")
        
        # Verificar duplicatas
        duplicates = df_checkpoints.duplicated(subset=['user_id', 't_rec']).sum()
        if duplicates == 0:
            f.write("✓ **Duplicatas:** Nenhuma duplicata encontrada\n\n")
        else:
            f.write(f"⚠️ **Duplicatas:** {duplicates} checkpoints duplicados\n\n")
        
        # ========================================
        # 7. RECOMENDAÇÕES PARA USO
        # ========================================
        f.write("## 7. Recomendações para Uso\n\n")
        
        f.write("### 7.1 Regra de Treino/Teste\n\n")
        f.write("Para cada checkpoint (user_id, t_rec):\n\n")
        f.write("**TREINO:**\n")
        f.write("```python\n")
        f.write("train_data = df_interactions[df_interactions['rating_when'] < t_rec]\n")
        f.write("```\n")
        f.write("- Todas as interações de TODOS os usuários antes de t_rec\n")
        f.write("- Representa o conhecimento global do sistema naquele momento\n\n")
        
        f.write("**TESTE:**\n")
        f.write("```python\n")
        f.write("test_data = df_interactions[\n")
        f.write("    (df_interactions['user_id'] == user_id) &\n")
        f.write("    (df_interactions['rating_when'] > t_rec) &\n")
        f.write("    (df_interactions['rating_when'] <= t_next_rec)\n")
        f.write("]\n")
        f.write("```\n")
        f.write("- Avaliações do usuário no intervalo (t_rec, t_next_rec]\n\n")
        
        f.write("**EXPOSIÇÃO (para RMSE):**\n")
        f.write("```python\n")
        f.write("exposed_news = checkpoint['candidate_news_ids'][:20]  # top-20\n")
        f.write("observable_test = test_data[test_data['news_id'].isin(exposed_news)]\n")
        f.write("```\n")
        f.write("- Apenas avaliações de itens que estavam na lista recomendada\n\n")
        
        f.write("### 7.2 Filtros Recomendados\n\n")
        
        if checkpoints_with_pool / total_checkpoints < 0.5:
            f.write(f"⚠️ **Atenção:** Apenas {checkpoints_with_pool/total_checkpoints*100:.1f}% dos checkpoints têm candidate pool.\n")
            f.write("- Considere filtrar apenas checkpoints com `has_candidate_pool == True`\n")
            f.write("- Ou implementar estratégia alternativa para checkpoints sem pool\n\n")
        
        # Identificar usuários com poucos checkpoints
        users_few_checkpoints = (checkpoints_per_user < 3).sum()
        if users_few_checkpoints > unique_users * 0.3:
            f.write(f"⚠️ **Atenção:** {users_few_checkpoints} usuários ({users_few_checkpoints/unique_users*100:.1f}%) têm <3 checkpoints.\n")
            f.write("- Considere filtrar usuários com ≥3 checkpoints para treino/validação consistente\n\n")
        
        f.write("### 7.3 Próximos Passos\n\n")
        f.write("1. **Implementar baseline de recomendação** (popularity, item-based CF)\n")
        f.write("2. **Calcular métricas por checkpoint** (RMSE, NDCG@20, etc.)\n")
        f.write("3. **Agregar métricas** (média, mediana) por usuário e global\n")
        f.write("4. **Analisar evolução temporal** das métricas ao longo do tempo\n\n")
        
        f.write("---\n\n")
        f.write("*Relatório gerado automaticamente pelo pipeline de replay temporal*\n")
    
    print(f"\n✓ Relatório de replay salvo em: {output_path}")


def save_replay_checkpoints(df_checkpoints: pd.DataFrame, output_path='outputs/replay_checkpoints.parquet'):
    """
    Salva checkpoints em formato Parquet.
    
    Args:
        df_checkpoints: DataFrame com checkpoints
        output_path: Caminho do arquivo de saída
    """
    print("\n" + "="*70)
    print("SALVANDO REPLAY CHECKPOINTS")
    print("="*70)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_checkpoints.to_parquet(output_path, index=False, engine='pyarrow')
    
    file_size = output_path.stat().st_size / 1024  # KB
    print(f"✓ Salvo {output_path.name}: {len(df_checkpoints):,} checkpoints, {file_size:.1f} KB")
    
    print("\n" + "="*70)
    print("SALVAMENTO CONCLUÍDO")
    print("="*70)


def main():
    """
    Função principal: carrega dados, constrói checkpoints e gera relatórios.
    """
    print("\n" + "█"*70)
    print(" "*15 + "PIPELINE: BUILD REPLAY CHECKPOINTS")
    print("█"*70)
    
    # Carregar dados canônicos
    df_rec_sessions, df_interactions = load_canonical_data()
    
    # Construir checkpoints de replay
    df_checkpoints = build_replay_checkpoints(df_rec_sessions, df_interactions)
    
    # Salvar checkpoints
    save_replay_checkpoints(df_checkpoints)
    
    # Gerar relatório
    generate_replay_report(df_checkpoints, df_rec_sessions, df_interactions)
    
    print("\n" + "█"*70)
    print(" "*20 + "PIPELINE CONCLUÍDO COM SUCESSO!")
    print("█"*70 + "\n")


if __name__ == '__main__':
    main()
