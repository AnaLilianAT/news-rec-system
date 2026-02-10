"""
Módulo para formatar tabelas no formato exato da tese.

Usa interface padronizada de representações e similaridades.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings

from .representations import get_item_representation, prepare_item_vectors
from .similarity import compute_homogeneity, jaccard_similarity_sets

warnings.filterwarnings('ignore')


def normalize_algorithm_name(raw: str) -> str:
    """
    Normaliza nomes de algoritmos para o formato da tese.
    
    Mapeamento:
    - knn_user_user -> knnu
    - knn_item_item -> knni
    - svd -> svd
    Sufixos:
    - -topic-diversification ou _topic_diversification -> td
    - -mmr-diversification ou _mmr_diversification -> mmr
    
    Exemplos:
    - "knn_user_user" -> "knnu"
    - "knn_user_user-topic-diversification" -> "knnu td"
    - "svd_mmr_diversification" -> "svd mmr"
    """
    if pd.isna(raw):
        return "unknown"
    
    raw = str(raw).lower().strip()
    
    # Detectar sufixos (ordem importa: topic antes de mmr)
    suffix = ""
    if "topic" in raw:
        suffix = " td"
        # Remover todas as variações de topic-diversification
        raw = raw.replace("-topic-diversification", "").replace("_topic_diversification", "")
        raw = raw.replace("topic-diversification", "").replace("topic_diversification", "")
        raw = raw.replace("-topic-divers", "").replace("_topic_divers", "")
        raw = raw.replace("-topic", "").replace("_topic", "")
    elif "mmr" in raw:
        suffix = " mmr"
        # Remover todas as variações de mmr-diversification
        raw = raw.replace("-mmr-diversification", "").replace("_mmr_diversification", "")
        raw = raw.replace("mmr-diversification", "").replace("mmr_diversification", "")
        raw = raw.replace("-mmr-divers", "").replace("_mmr_divers", "")
        raw = raw.replace("-mmr", "").replace("_mmr", "")
    
    # Normalizar base (remover separadores restantes)
    raw = raw.replace("-", "_").strip("_")
    
    if "user" in raw and "knn" in raw:
        base = "knnu"
    elif "item" in raw and "knn" in raw:
        base = "knni"
    elif raw.startswith("svd"):
        base = "svd"
    else:
        base = raw.replace("_", " ").strip()
    
    return (base + suffix).strip()


def shapiro_p(values) -> float:
    """
    Calcula p-valor do teste Shapiro-Wilk.
    Retorna NaN se n < 3.
    """
    values = np.array(values)
    values = values[~np.isnan(values)]
    
    if len(values) < 3:
        return np.nan
    
    try:
        _, p_value = stats.shapiro(values)
        return p_value
    except Exception:
        return np.nan


def compute_jaccard(set_a, set_b):
    """
    Calcula similaridade de Jaccard entre dois conjuntos.
    
    NOTA: Mantida por compatibilidade. Para novos desenvolvimentos,
    usar jaccard_similarity_sets() do módulo similarity.
    """
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


def compute_GH_interaction_jaccard(
    eval_pairs: pd.DataFrame,
    topics: pd.DataFrame,
    representation_type: str = 'bin_topics',
    output_dir: str = 'outputs'
) -> pd.DataFrame:
    """
    Calcula GH (Jaccard) para itens recomendados e interagidos (Tabela 6.1).
    
    Para cada usuário:
    - Pegar itens expostos (do eval_pairs)
    - Calcular Jaccard entre todos os pares de itens (usando tópicos)
    - GH_user = soma(Jaccard) / |R| (normalização da Equação 4.3 da tese)
    
    IMPORTANTE: Usa normalização por |R| (não por #pares) para reproduzir escala da tese.
    
    Args:
        eval_pairs: DataFrame com itens expostos por usuário
        topics: DataFrame com tópicos (usado se representation_type='bin_topics')
        representation_type: Tipo de representação ('bin_topics' ou 'ae_topics')
        output_dir: Diretório base dos outputs
    
    Returns:
        DataFrame com colunas: [user_id, algorithm, gh_jaccard_interaction, n_items]
    """
    # Se usando representação binária (modo atual), usar código original
    if representation_type == 'bin_topics':
        return _compute_GH_interaction_jaccard_legacy(eval_pairs, topics)
    
    # Futuramente: suportar embeddings com cosine
    # elif representation_type == 'ae_topics':
    #     topics_rep = get_item_representation(representation_type, output_dir=output_dir)
    #     topic_vectors = prepare_item_vectors(topics_rep)
    #     return _compute_GH_interaction_cosine(eval_pairs, topic_vectors)
    
    else:
        raise ValueError(
            f"Representação não suportada: '{representation_type}'. "
            f"Use 'bin_topics' ou 'ae_topics' (futuro)"
        )


def _compute_GH_interaction_jaccard_legacy(eval_pairs: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    """
    Implementação original de GH com Jaccard sobre tópicos binários.
    
    Mantida intacta para garantir compatibilidade de resultados.
    
    Para cada usuário:
    - Pegar itens expostos (do eval_pairs)
    - Calcular Jaccard entre todos os pares de itens (usando tópicos)
    - GH_user = soma(Jaccard) / |R| (normalização da Equação 4.3 da tese)
    
    IMPORTANTE: Usa normalização por |R| (não por #pares) para reproduzir escala da tese.
    
    Returns:
        DataFrame com colunas: [user_id, algorithm, gh_jaccard_interaction, n_items]
    """
    # Preparar tópicos
    topics_dict = {}
    for _, row in topics.iterrows():
        item_id = row['news_id']
        # Topics são colunas Topic0, Topic1, etc.
        topic_cols = [c for c in topics.columns if c.startswith('Topic')]
        active_topics = [col for col in topic_cols if row[col] == 1]
        if item_id not in topics_dict:
            topics_dict[item_id] = set(active_topics)
    
    results = []
    
    # Agrupar por usuário e algoritmo
    for (user_id, algorithm), group in eval_pairs.groupby(['user_id', 'algorithm']):
        items = group['news_id'].unique()
        
        if len(items) < 2:
            continue  # Precisa de pelo menos 2 itens
        
        # Calcular Jaccard entre todos os pares
        jaccards = []
        for item_a, item_b in combinations(items, 2):
            topics_a = topics_dict.get(item_a, set())
            topics_b = topics_dict.get(item_b, set())
            if len(topics_a) > 0 or len(topics_b) > 0:
                jacc = compute_jaccard(topics_a, topics_b)
                jaccards.append(jacc)
        
        if len(jaccards) > 0:
            # CORREÇÃO: Normalizar por |R| (Equação 4.3 da tese), não por #pares
            # GH_user = (1/|R|) × Σ_{i<j} Jaccard(i,j)
            gh_user = np.sum(jaccards) / len(items)
            results.append({
                'user_id': user_id,
                'algorithm': algorithm,
                'gh_jaccard_interaction': gh_user,
                'n_items': len(items)
            })
    
    return pd.DataFrame(results)


def compute_cosine_similarity(vec_a, vec_b):
    """
    Calcula similaridade do cosseno entre dois vetores.
    
    NOTA: Mantida por compatibilidade. Para novos desenvolvimentos,
    usar cosine_similarity() do módulo similarity.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)


def compute_GH_lists_cosine(
    reclists: pd.DataFrame,
    features: pd.DataFrame,
    representation_type: str = 'bin_features',
    output_dir: str = 'outputs'
) -> pd.DataFrame:
    """
    Calcula GH (cosseno) para listas de recomendação (Tabela 6.6).
    
    Para cada lista top-20:
    - Calcular cosseno entre todos os pares de itens (usando features)
    - GH_list = média dos cossenos
    
    Depois agregar por usuário: GH_user = média das GH_list do usuário
    
    Args:
        reclists: DataFrame com listas top-20
        features: DataFrame com features (usado se representation_type='bin_features')
        representation_type: Tipo de representação ('bin_features' ou 'ae_features')
        output_dir: Diretório base dos outputs
    
    Returns:
        DataFrame com colunas: [user_id, algorithm, gh_cosine_lists, n_lists]
    """
    # Se usando representação binária (modo atual), usar código original
    if representation_type == 'bin_features':
        return _compute_GH_lists_cosine_legacy(reclists, features)
    
    # Futuramente: suportar embeddings
    # elif representation_type == 'ae_features':
    #     features_rep = get_item_representation(representation_type, output_dir=output_dir)
    #     feature_vectors = prepare_item_vectors(features_rep)
    #     return _compute_GH_lists_cosine_with_vectors(reclists, feature_vectors)
    
    else:
        raise ValueError(
            f"Representação não suportada: '{representation_type}'. "
            f"Use 'bin_features' ou 'ae_features' (futuro)"
        )


def _compute_GH_lists_cosine_legacy(reclists: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """
    Implementação original de GH com cosseno sobre features binárias.
    
    Mantida intacta para garantir compatibilidade de resultados.
    
    Returns:
        DataFrame com colunas: [user_id, algorithm, gh_cosine_lists, n_lists]
    """
    # Preparar features como vetores
    features_dict = {}
    feature_cols = [c for c in features.columns if c not in ['news_id']]
    
    for _, row in features.iterrows():
        item_id = row['news_id']
        vec = row[feature_cols].values.astype(float)
        features_dict[item_id] = vec
    
    results = []
    
    # Agrupar por (user_id, t_rec, algorithm) para formar cada lista
    list_results = []
    for (user_id, t_rec, algorithm), group in reclists.groupby(['user_id', 't_rec', 'algorithm']):
        items = group['news_id'].tolist()
        
        items = [item for item in items if item in features_dict]
        
        if len(items) < 2:
            continue
        
        # Calcular cosseno entre todos os pares
        cosines = []
        for item_a, item_b in combinations(items, 2):
            vec_a = features_dict[item_a]
            vec_b = features_dict[item_b]
            cos = compute_cosine_similarity(vec_a, vec_b)
            cosines.append(cos)
        
        if len(cosines) > 0:
            gh_list = np.mean(cosines)
            list_results.append({
                'user_id': user_id,
                'algorithm': algorithm,
                'gh_list': gh_list
            })
    
    df_lists = pd.DataFrame(list_results)
    
    # Agregar por usuário
    if len(df_lists) > 0:
        for (user_id, algorithm), group in df_lists.groupby(['user_id', 'algorithm']):
            gh_user = group['gh_list'].mean()
            n_lists = len(group)
            results.append({
                'user_id': user_id,
                'algorithm': algorithm,
                'gh_cosine_lists': gh_user,
                'n_lists': n_lists
            })
    
    return pd.DataFrame(results)


def compute_RMSE_user(eval_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula RMSE por usuário (Tabela 6.3).
    
    Para cada usuário:
    - RMSE = sqrt(mean((rating_real - score_pred)^2))
    
    Retorna DataFrame com colunas: [user_id, algorithm, rmse, n_pairs]
    """
    results = []
    
    for (user_id, algorithm), group in eval_pairs.groupby(['user_id', 'algorithm']):
        if len(group) < 2:
            continue
        
        real = group['rating_real'].values
        pred = group['score_pred'].values
        
        # Remover NaN
        mask = ~(np.isnan(real) | np.isnan(pred))
        real = real[mask]
        pred = pred[mask]
        
        if len(real) < 2:
            continue
        
        rmse = np.sqrt(np.mean((real - pred) ** 2))
        
        results.append({
            'user_id': user_id,
            'algorithm': algorithm,
            'rmse': rmse,
            'n_pairs': len(real)
        })
    
    return pd.DataFrame(results)


def aggregate_like_thesis(
    df_user_metric: pd.DataFrame,
    metric_col: str,
    include_users: bool = True,
    include_minmax: bool = False
) -> pd.DataFrame:
    """
    Agrega métricas por algoritmo no formato da tese.
    
    Args:
        df_user_metric: DataFrame com colunas [user_id, algorithm, metric_col]
        metric_col: Nome da coluna da métrica
        include_users: Se True, inclui coluna "Usuários"
        include_minmax: Se True, inclui colunas "Min." e "Máx."
    
    Retorna DataFrame com formato da tese.
    """
    results = []
    
    for algorithm, group in df_user_metric.groupby('algorithm'):
        values = group[metric_col].values
        values = values[~np.isnan(values)]
        
        if len(values) == 0:
            continue
        
        # IMPORTANTE: Não normalizar novamente - o algorithm já vem normalizado
        row = {'Algoritmo': algorithm}
        
        if include_users:
            row['Usuários'] = len(values)
        
        if include_minmax:
            row['Min.'] = np.min(values)
            row['Máx.'] = np.max(values)
        
        row['Média'] = np.mean(values)
        row['Mediana'] = np.median(values)
        
        if include_minmax:
            row['Desvio padrão'] = np.std(values, ddof=1)
        else:
            row['Desvio Padrão'] = np.std(values, ddof=1)
        
        row['p-valor'] = shapiro_p(values)
        
        results.append(row)
    
    df_result = pd.DataFrame(results)
    
    # Ordenar pelos algoritmos da tese
    order = ['knnu', 'knnu td', 'knnu mmr', 'knni', 'knni td', 'knni mmr', 'svd', 'svd td', 'svd mmr']
    df_result['_order'] = df_result['Algoritmo'].map({a: i for i, a in enumerate(order)})
    df_result = df_result.sort_values('_order').drop(columns=['_order'])
    df_result = df_result.reset_index(drop=True)
    
    return df_result


def format_table_for_export(df: pd.DataFrame, decimal_places: int = 3) -> pd.DataFrame:
    """
    Formata tabela para exportação com arredondamento apropriado.
    """
    df_formatted = df.copy()
    
    for col in df_formatted.columns:
        if col in ['Usuários']:
            df_formatted[col] = df_formatted[col].astype(int)
        elif col == 'p-valor':
            # Manter precisão científica se necessário
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f'{x:.4f}' if not np.isnan(x) else 'NaN'
            )
        elif col in ['Média', 'Mediana', 'Desvio Padrão', 'Desvio padrão', 'Min.', 'Máx.']:
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f'{x:.{decimal_places}f}' if not np.isnan(x) else 'NaN'
            )
    
    return df_formatted
