"""
Funções de similaridade para diversificação de recomendações.
"""
import numpy as np
from typing import List, Optional


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calcula a similaridade de cosseno entre dois vetores.
    
    Args:
        vec_a: Vetor numpy (1D)
        vec_b: Vetor numpy (1D)
    
    Returns:
        Similaridade de cosseno no intervalo [-1, 1]
    """
    # Normalizar vetores
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)


def pairwise_cosine_similarity(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Calcula matriz de similaridade cosseno entre múltiplos vetores.
    
    Args:
        vectors: Lista de vetores numpy
    
    Returns:
        Matriz numpy (n x n) com similaridades
    """
    n = len(vectors)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            sim = cosine_similarity(vectors[i], vectors[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    
    return sim_matrix


def max_similarity_to_set(
    candidate_vec: np.ndarray,
    selected_vecs: List[np.ndarray]
) -> float:
    """
    Calcula a similaridade máxima de um candidato com um conjunto já selecionado.
    
    Args:
        candidate_vec: Vetor do item candidato
        selected_vecs: Lista de vetores já selecionados
    
    Returns:
        Similaridade máxima (maior similaridade com qualquer item do conjunto)
    """
    if not selected_vecs:
        return 0.0
    
    similarities = [cosine_similarity(candidate_vec, sel_vec) for sel_vec in selected_vecs]
    return max(similarities)


def vector_diversity(vectors: List[np.ndarray]) -> float:
    """
    Calcula a diversidade média de um conjunto de vetores.
    
    Args:
        vectors: Lista de vetores numpy
    
    Returns:
        Diversidade média (1 - similaridade média)
    """
    if len(vectors) <= 1:
        return 1.0
    
    similarities = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            similarities.append(cosine_similarity(vectors[i], vectors[j]))
    
    avg_similarity = np.mean(similarities) if similarities else 0.0
    return 1.0 - avg_similarity


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normaliza um vetor para ter norma L2 = 1.
    
    Args:
        vec: Vetor numpy
    
    Returns:
        Vetor normalizado
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm
