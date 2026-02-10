"""
Funções de similaridade para diversificação de recomendações.

Inclui:
- Similaridade de cosseno (para vetores contínuos)
- Similaridade de Jaccard (para conjuntos/vetores binários)
- Interface padronizada para cálculo de similaridades
"""
import numpy as np
from typing import List, Optional, Dict, Callable, Tuple, Set
from itertools import combinations


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


def jaccard_similarity(vec_a: np.ndarray, vec_b: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calcula similaridade de Jaccard entre dois vetores binários.
    
    Para vetores binários: Jaccard = |A ∩ B| / |A ∪ B|
    Para vetores contínuos: usa threshold para binarizar
    
    Args:
        vec_a: Vetor numpy (1D)
        vec_b: Vetor numpy (1D)
        threshold: Limiar para considerar feature ativa (default: 0.5)
    
    Returns:
        Similaridade de Jaccard no intervalo [0, 1]
    """
    # Binarizar vetores (>= threshold)
    bin_a = (vec_a >= threshold).astype(int)
    bin_b = (vec_b >= threshold).astype(int)
    
    # Calcular interseção e união
    intersection = np.sum(bin_a & bin_b)
    union = np.sum(bin_a | bin_b)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def jaccard_similarity_sets(set_a: Set, set_b: Set) -> float:
    """
    Calcula similaridade de Jaccard entre dois conjuntos.
    
    Args:
        set_a: Conjunto de elementos (ex: {Topic0, Topic3, Topic5})
        set_b: Conjunto de elementos
    
    Returns:
        Similaridade de Jaccard no intervalo [0, 1]
    """
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_pairwise_similarity(
    vectors: Dict[int, np.ndarray],
    metric: str = 'cosine',
    **kwargs
) -> Dict[Tuple[int, int], float]:
    """
    Calcula similaridades par a par entre itens.
    
    Args:
        vectors: Dicionário {item_id: vector}
        metric: Métrica de similaridade ('cosine', 'jaccard')
        **kwargs: Argumentos adicionais para a métrica (ex: threshold para jaccard)
    
    Returns:
        Dicionário {(item_i, item_j): similarity}
    """
    metric_fn = get_similarity_function(metric)
    
    item_ids = list(vectors.keys())
    similarities = {}
    
    for item_i, item_j in combinations(item_ids, 2):
        vec_i = vectors[item_i]
        vec_j = vectors[item_j]
        sim = metric_fn(vec_i, vec_j, **kwargs)
        similarities[(item_i, item_j)] = sim
        similarities[(item_j, item_i)] = sim  # Simétrica
    
    # Diagonal (self-similarity)
    for item_id in item_ids:
        similarities[(item_id, item_id)] = 1.0
    
    return similarities


def get_similarity_function(metric: str) -> Callable:
    """
    Retorna função de similaridade de acordo com a métrica.
    
    Args:
        metric: Nome da métrica ('cosine', 'jaccard')
    
    Returns:
        Função de similaridade (vec_a, vec_b, **kwargs) -> float
    
    Raises:
        ValueError: Se métrica não for reconhecida
    """
    metrics = {
        'cosine': cosine_similarity,
        'jaccard': jaccard_similarity
    }
    
    if metric not in metrics:
        raise ValueError(
            f"Métrica desconhecida: '{metric}'. "
            f"Métricas disponíveis: {list(metrics.keys())}"
        )
    
    return metrics[metric]


def compute_homogeneity(
    item_ids: List[int],
    vectors: Dict[int, np.ndarray],
    metric: str = 'cosine',
    normalize_by: str = 'n_items',
    **kwargs
) -> float:
    """
    Calcula homogeneidade (GH) de um conjunto de itens.
    
    GH = (1/|R|) × Σ_{i<j} similarity(i,j)
    
    Args:
        item_ids: Lista de item_ids do conjunto
        vectors: Dicionário {item_id: vector}
        metric: Métrica de similaridade ('cosine', 'jaccard')
        normalize_by: Como normalizar ('n_items' ou 'n_pairs')
        **kwargs: Argumentos adicionais para a métrica
    
    Returns:
        Homogeneidade (GH) no intervalo [0, 1]
    """
    if len(item_ids) < 2:
        return np.nan
    
    # Filtrar apenas itens disponíveis
    available_items = [item_id for item_id in item_ids if item_id in vectors]
    
    if len(available_items) < 2:
        return np.nan
    
    # Calcular similaridades entre pares
    metric_fn = get_similarity_function(metric)
    similarities = []
    
    for item_i, item_j in combinations(available_items, 2):
        vec_i = vectors[item_i]
        vec_j = vectors[item_j]
        sim = metric_fn(vec_i, vec_j, **kwargs)
        similarities.append(sim)
    
    if len(similarities) == 0:
        return np.nan
    
    # Normalizar de acordo com o método
    if normalize_by == 'n_items':
        # Equação 4.3 da tese: normalizar por |R|
        return np.sum(similarities) / len(available_items)
    elif normalize_by == 'n_pairs':
        # Alternativa: média das similaridades
        return np.mean(similarities)
    else:
        raise ValueError(f"Método de normalização desconhecido: '{normalize_by}'")


def get_metric_for_representation(representation_type: str) -> str:
    """
    Retorna a métrica apropriada para um tipo de representação.
    
    Args:
        representation_type: Tipo da representação
            - 'bin_features' ou 'bin_topics': usa 'cosine' (para compatibilidade)
            - 'ae_features' ou 'ae_topics': usa 'cosine' (embeddings são L2-normalizados)
    
    Returns:
        Nome da métrica ('cosine' ou 'jaccard')
    """
    # Para embeddings, sempre cosine (já são L2-normalizados)
    if representation_type in ['ae_features', 'ae_topics']:
        return 'cosine'
    
    # Para binários, manter comportamento atual (cosine)
    # Nota: o código original usa cosine, não jaccard
    if representation_type in ['bin_features', 'bin_topics']:
        return 'cosine'
    
    # Default: cosine
    return 'cosine'
