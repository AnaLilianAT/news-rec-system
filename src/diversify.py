"""
Algoritmos de diversificação para recomendações: MMR e TD (Topic Diversification).
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from .similarity import cosine_similarity, max_similarity_to_set


def mmr_select(
    ranked_items: List[Tuple[int, float]],
    feature_vectors: Dict[int, np.ndarray],
    lambda_mmr: float = 0.7,
    k: int = 20
) -> List[Tuple[int, float, int]]:
    """
    Seleciona top-k itens usando Maximal Marginal Relevance (MMR).
    
    MMR equilibra relevância (score) e diversidade (dissimilaridade).
    
    Args:
        ranked_items: Lista de (news_id, score_pred) ordenada por score
        feature_vectors: Dicionário {news_id: feature_vector}
        lambda_mmr: Peso para relevância (0.0=pura diversidade, 1.0=puro score)
        k: Número de itens a selecionar
    
    Returns:
        Lista de (news_id, score_pred, rank) com até k itens
    """
    # Filtrar apenas itens que têm features
    candidates = [(nid, score) for nid, score in ranked_items if nid in feature_vectors]
    
    if not candidates:
        # Fallback: retornar top-k sem diversificação
        return [(nid, score, rank + 1) for rank, (nid, score) in enumerate(ranked_items[:k])]
    
    if len(candidates) <= k:
        # Se temos menos ou igual a k candidatos, retornar todos
        return [(nid, score, rank + 1) for rank, (nid, score) in enumerate(candidates)]
    
    # Normalizar scores para [0, 1]
    scores = [score for _, score in candidates]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    if score_range == 0:
        normalized_scores = {nid: 0.5 for nid, _ in candidates}
    else:
        normalized_scores = {
            nid: (score - min_score) / score_range 
            for nid, score in candidates
        }
    
    # MMR selection
    selected = []
    selected_vectors = []
    remaining = list(candidates)
    
    for iteration in range(k):
        if not remaining:
            break
        
        best_mmr = -float('inf')
        best_idx = 0
        
        for idx, (nid, score) in enumerate(remaining):
            if nid not in feature_vectors:
                continue
            
            vec = feature_vectors[nid]
            relevance = normalized_scores[nid]
            
            if not selected_vectors:
                # Primeiro item: escolher o de maior score
                diversity = 0.0
            else:
                # Calcular diversidade: 1 - max_sim
                max_sim = max_similarity_to_set(vec, selected_vectors)
                diversity = 1.0 - max_sim
            
            # MMR = λ * relevance + (1-λ) * diversity
            mmr_score = lambda_mmr * relevance + (1 - lambda_mmr) * diversity
            
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx
        
        # Selecionar o melhor item
        nid, score = remaining.pop(best_idx)
        selected.append((nid, score, len(selected) + 1))
        selected_vectors.append(feature_vectors[nid])
    
    # Se não conseguimos k itens com features, completar com os próximos do ranking original
    if len(selected) < k:
        selected_ids = {nid for nid, _, _ in selected}
        for nid, score in ranked_items:
            if nid not in selected_ids:
                selected.append((nid, score, len(selected) + 1))
                if len(selected) >= k:
                    break
    
    return selected


def td_select(
    ranked_items: List[Tuple[int, float]],
    topic_vectors: Dict[int, np.ndarray],
    k: int = 20,
    fallback_features: Optional[Dict[int, np.ndarray]] = None
) -> List[Tuple[int, float, int]]:
    """
    Seleciona top-k itens usando Topic Diversification (TD).
    
    TD prefere itens que cobrem tópicos diversos (baseado em Topic0..Topic15).
    Usa heurística determinística: escolher item com menor sobreposição de tópicos dominantes.
    
    Args:
        ranked_items: Lista de (news_id, score_pred) ordenada por score
        topic_vectors: Dicionário {news_id: topic_vector (16 dims)}
        k: Número de itens a selecionar
        fallback_features: Dicionário opcional com features para fallback via cosseno
    
    Returns:
        Lista de (news_id, score_pred, rank) com até k itens
    """
    # Filtrar apenas itens que têm tópicos
    candidates_with_topics = [(nid, score) for nid, score in ranked_items if nid in topic_vectors]
    
    if not candidates_with_topics:
        # Fallback para MMR se tópicos não disponíveis
        if fallback_features:
            return mmr_select(ranked_items, fallback_features, lambda_mmr=0.7, k=k)
        else:
            # Último fallback: top-k por score
            return [(nid, score, rank + 1) for rank, (nid, score) in enumerate(ranked_items[:k])]
    
    if len(candidates_with_topics) <= k:
        return [(nid, score, rank + 1) for rank, (nid, score) in enumerate(candidates_with_topics)]
    
    # TD selection: heurística determinística
    # Para cada item, identificar tópicos dominantes (top-3) e preferir itens que cobrem tópicos não selecionados
    
    selected = []
    selected_dominant_topics = set()  # Conjunto de tópicos já cobertos
    remaining = list(candidates_with_topics)
    
    for iteration in range(k):
        if not remaining:
            break
        
        best_score = -float('inf')
        best_idx = 0
        
        for idx, (nid, score) in enumerate(remaining):
            if nid not in topic_vectors:
                continue
            
            topic_vec = topic_vectors[nid]
            
            # Identificar top-3 tópicos deste item
            top_topics_idx = np.argsort(topic_vec)[-3:][::-1]  # Top 3 índices
            item_topics = set(top_topics_idx)
            
            # Calcular diversidade: quantos tópicos novos este item traz
            new_topics = len(item_topics - selected_dominant_topics)
            
            # Score combinado: normalizar score original + bonificar novos tópicos
            # Usar ranking relativo como score de relevância
            relevance = len(remaining) - idx  # Quanto mais cedo no ranking, melhor
            
            # Diversidade: número de tópicos novos
            diversity_bonus = new_topics * 10  # Peso alto para novos tópicos
            
            combined_score = relevance + diversity_bonus
            
            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx
        
        # Selecionar o melhor item
        nid, score = remaining.pop(best_idx)
        selected.append((nid, score, len(selected) + 1))
        
        # Atualizar tópicos cobertos
        if nid in topic_vectors:
            topic_vec = topic_vectors[nid]
            top_topics_idx = np.argsort(topic_vec)[-3:][::-1]
            selected_dominant_topics.update(top_topics_idx)
    
    # Completar se necessário
    if len(selected) < k:
        selected_ids = {nid for nid, _, _ in selected}
        for nid, score in ranked_items:
            if nid not in selected_ids:
                selected.append((nid, score, len(selected) + 1))
                if len(selected) >= k:
                    break
    
    return selected


def apply_diversification(
    ranked_items: List[Tuple[int, float]],
    diversify: str,
    feature_vectors: Optional[Dict[int, np.ndarray]] = None,
    topic_vectors: Optional[Dict[int, np.ndarray]] = None,
    k: int = 20
) -> List[Tuple[int, float, int]]:
    """
    Aplica estratégia de diversificação apropriada.
    
    Args:
        ranked_items: Lista de (news_id, score_pred) ordenada por score
        diversify: Estratégia ('none', 'mmr', 'td')
        feature_vectors: Features para MMR/fallback
        topic_vectors: Tópicos para TD
        k: Número de itens a retornar
    
    Returns:
        Lista de (news_id, score_pred, rank) com até k itens
    """
    if diversify == 'none' or diversify is None:
        # Sem diversificação: top-k por score
        return [(nid, score, rank + 1) for rank, (nid, score) in enumerate(ranked_items[:k])]
    
    elif diversify == 'mmr':
        if feature_vectors is None:
            # Fallback para none
            return [(nid, score, rank + 1) for rank, (nid, score) in enumerate(ranked_items[:k])]
        return mmr_select(ranked_items, feature_vectors, lambda_mmr=0.7, k=k)
    
    elif diversify == 'td':
        if topic_vectors is None:
            # Fallback para MMR ou none
            if feature_vectors is not None:
                return mmr_select(ranked_items, feature_vectors, lambda_mmr=0.7, k=k)
            else:
                return [(nid, score, rank + 1) for rank, (nid, score) in enumerate(ranked_items[:k])]
        return td_select(ranked_items, topic_vectors, k=k, fallback_features=feature_vectors)
    
    else:
        # Estratégia desconhecida: fallback para none
        return [(nid, score, rank + 1) for rank, (nid, score) in enumerate(ranked_items[:k])]
