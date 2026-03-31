"""
Funções de similaridade para diversificação de recomendações.

Inclui:
- Similaridade de cosseno (para vetores contínuos)
- Similaridade de Jaccard (para conjuntos/vetores binários)
- Interface padronizada para cálculo de similaridades
"""
import numpy as np
from typing import List, Optional, Dict, Callable, Tuple, Set, Union
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def compute_similarity_from_embeddings(
    ids: List[int],
    embeddings: Dict[int, np.ndarray],
    metric: str = "cosine",
    top_k: Optional[int] = None
) -> Tuple[List[int], Union[np.ndarray, Callable]]:
    """
    Calcula matriz de similaridade baseada em embeddings.
    
    Usa sklearn para eficiência. Embeddings devem estar L2-normalizados
    para que cosine_similarity funcione corretamente.
    
    Args:
        ids: Lista de item IDs
        embeddings: Dict {item_id: embedding_vector}
        metric: Métrica de similaridade ('cosine' é a única suportada)
        top_k: Se especificado, retorna apenas top-k vizinhos mais similares (sparse).
                Se None, retorna matriz cheia.
    
    Returns:
        (ordered_ids, similarity_matrix ou similarity_callable)
        - ordered_ids: Lista de IDs na ordem da matriz
        - similarity_matrix: Matriz numpy (n x n) se top_k=None
        - similarity_callable: função (i, j) -> float se top_k especificado
    
    Raises:
        ValueError: Se métrica não for 'cosine'
    
    Example:
        >>> ids = [10, 20, 30]
        >>> emb = {10: np.array([0.1, 0.2]), 20: np.array([0.3, 0.4]), 30: np.array([0.5, 0.6])}
        >>> ordered_ids, sim_matrix = compute_similarity_from_embeddings(ids, emb)
        >>> sim_matrix[0, 1]  # similaridade entre ids[0] e ids[1]
        0.998
    """
    if metric != "cosine":
        raise ValueError(
            f"Métrica '{metric}' não suportada. "
            "compute_similarity_from_embeddings suporta apenas 'cosine'."
        )
    
    # Filtrar IDs que têm embeddings disponíveis
    available_ids = [item_id for item_id in ids if item_id in embeddings]
    
    if len(available_ids) == 0:
        # Retornar matriz vazia
        return [], np.array([[]])
    
    # Construir matriz de embeddings (n_items x embedding_dim)
    embedding_matrix = np.vstack([embeddings[item_id] for item_id in available_ids])
    
    # Calcular similaridade cosseno usando sklearn (eficiente)
    # Nota: sklearn já assume that embeddings are L2-normalized
    sim_matrix = sklearn_cosine_similarity(embedding_matrix)
    
    if top_k is None:
        # Retornar matriz cheia
        return available_ids, sim_matrix
    else:
        # Implementação sparse: retornar callable que faz lookup
        # Construir dicionário de top-k vizinhos para cada item
        top_k_neighbors = {}
        for i, item_id in enumerate(available_ids):
            # Pegar top-k mais similares (excluindo o próprio item)
            similarities = sim_matrix[i, :]
            # Zerar self-similarity para não incluir na seleção
            similarities_copy = similarities.copy()
            similarities_copy[i] = -1
            
            # Top-k índices
            top_k_indices = np.argsort(similarities_copy)[-top_k:][::-1]
            
            # Armazenar como dict {neighbor_id: similarity}
            top_k_neighbors[item_id] = {
                available_ids[j]: similarities[j] 
                for j in top_k_indices
            }
        
        # Criar callable que faz lookup
        def similarity_lookup(item_i: int, item_j: int) -> float:
            """Lookup de similaridade (sparse)."""
            if item_i == item_j:
                return 1.0
            
            # Tentar lookup nos vizinhos de item_i
            if item_i in top_k_neighbors and item_j in top_k_neighbors[item_i]:
                return top_k_neighbors[item_i][item_j]
            
            # Tentar lookup nos vizinhos de item_j (simétrico)
            if item_j in top_k_neighbors and item_i in top_k_neighbors[item_j]:
                return top_k_neighbors[item_j][item_i]
            
            # Não está nos top-k vizinhos
            return 0.0
        
        return available_ids, similarity_lookup


class SimilarityProvider:
    """
    Provedor centralizado de similaridade para algoritmos de diversificação.
    
    Suporta:
    - Matriz de similaridade pré-computada (eficiente para embeddings)
    - Cálculo on-demand (para vetores binários ou quando matriz não disponível)
    - Interface unificada sim(i,j) e max_sim_to_set(candidate, selected)
    
    Para embeddings L2-normalizados, cosine similarity = dot product (muito eficiente).
    """
    
    def __init__(
        self, 
        ids: List[int],
        vectors_dict: Dict[int, np.ndarray],
        similarity_matrix: Optional[np.ndarray] = None,
        metric: str = 'cosine'
    ):
        """
        Args:
            ids: Lista ordenada de IDs dos itens
            vectors_dict: Dicionário {item_id: vector}
            similarity_matrix: Matriz de similaridade pré-computada (opcional)
            metric: Métrica de similaridade ('cosine' por padrão)
        """
        self.ids = ids
        self.id_to_idx = {item_id: idx for idx, item_id in enumerate(ids)}
        self.vectors = vectors_dict
        self.similarity_matrix = similarity_matrix
        self.metric = metric
        self._is_precomputed = similarity_matrix is not None
    
    @classmethod
    def from_embeddings(
        cls,
        ids: List[int],
        embedding_dict: Dict[int, np.ndarray],
        metric: str = 'cosine',
        precompute: bool = True,
        top_k: Optional[int] = None
    ):
        """
        Cria SimilarityProvider a partir de embeddings.
        
        Args:
            ids: Lista de IDs dos itens
            embedding_dict: Dicionário {item_id: embedding_vector}
            metric: Métrica ('cosine' recomendado para embeddings L2-normalizados)
            precompute: Se True, pré-computa matriz de similaridade
            top_k: Se especificado, usa computação esparsa (top-k vizinhos)
        
        Returns:
            SimilarityProvider configurado
        """
        if precompute:
            # Usar função existente para pré-computar matriz
            ordered_ids, sim_result = compute_similarity_from_embeddings(
                ids=ids,
                embeddings=embedding_dict,
                metric=metric,
                top_k=top_k
            )
            
            # Se top_k foi usado, sim_result é uma função callable
            if callable(sim_result):
                # Não temos matriz, então criar sem pré-computação
                return cls(ordered_ids, embedding_dict, None, metric)
            else:
                # sim_result é a matriz
                return cls(ordered_ids, embedding_dict, sim_result, metric)
        else:
            # Modo on-demand
            return cls(ids, embedding_dict, None, metric)
    
    @classmethod
    def from_vectors(
        cls,
        vectors_dict: Dict[int, np.ndarray],
        metric: str = 'cosine'
    ):
        """
        Cria SimilarityProvider a partir de vetores arbitrários (sem pré-computação).
        
        Útil para vetores binários ou quando não vale a pena pré-computar.
        
        Args:
            vectors_dict: Dicionário {item_id: vector}
            metric: Métrica de similaridade
        
        Returns:
            SimilarityProvider configurado para cálculo on-demand
        """
        ids = list(vectors_dict.keys())
        return cls(ids, vectors_dict, None, metric)
    
    def sim(self, id_i: int, id_j: int) -> float:
        """
        Calcula similaridade entre dois itens.
        
        Args:
            id_i: ID do primeiro item
            id_j: ID do segundo item
        
        Returns:
            Similaridade no intervalo [-1, 1] (ou [0, 1] para algumas métricas)
        """
        # Tentar usar matriz pré-computada primeiro
        if self._is_precomputed:
            idx_i = self.id_to_idx.get(id_i)
            idx_j = self.id_to_idx.get(id_j)
            
            if idx_i is not None and idx_j is not None:
                return float(self.similarity_matrix[idx_i, idx_j])
        
        # Fallback: computar on-demand
        vec_i = self.vectors.get(id_i)
        vec_j = self.vectors.get(id_j)
        
        if vec_i is None or vec_j is None:
            return 0.0
        
        # Usar função cosine_similarity existente
        return cosine_similarity(vec_i, vec_j)
    
    def max_sim_to_set(self, candidate_id: int, selected_ids: List[int]) -> float:
        """
        Calcula similaridade máxima de um candidato com um conjunto selecionado.
        
        Essencial para MMR (Maximal Marginal Relevance).
        
        Args:
            candidate_id: ID do item candidato
            selected_ids: Lista de IDs dos itens já selecionados
        
        Returns:
            Similaridade máxima (maior similaridade com qualquer item selecionado)
        """
        if not selected_ids:
            return 0.0
        
        similarities = [self.sim(candidate_id, sel_id) for sel_id in selected_ids]
        return max(similarities) if similarities else 0.0
    
    def get_vector(self, item_id: int) -> Optional[np.ndarray]:
        """
        Obtém vetor de um item.
        
        Útil para TD (Topic Diversification) que precisa dos vetores reais.
        
        Args:
            item_id: ID do item
        
        Returns:
            Vetor numpy ou None se não encontrado
        """
        return self.vectors.get(item_id)
    
    def has_item(self, item_id: int) -> bool:
        """Verifica se um item está disponível."""
        return item_id in self.vectors
    
    def is_precomputed(self) -> bool:
        """Retorna True se similaridades estão pré-computadas."""
        return self._is_precomputed
    
    def __repr__(self) -> str:
        mode = "precomputed" if self._is_precomputed else "on-demand"
        return f"SimilarityProvider(items={len(self.vectors)}, metric={self.metric}, mode={mode})"


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
    vectors: Optional[Dict[int, np.ndarray]] = None,
    metric: str = 'cosine',
    normalize_by: str = 'n_items',
    similarity_matrix: Optional[np.ndarray] = None,
    ordered_ids: Optional[List[int]] = None,
    similarity_fn: Optional[Callable] = None,
    **kwargs
) -> float:
    """
    Calcula homogeneidade (ILS) de um conjunto de itens.
    
    ILS = (1/|R|) × Σ_{i<j} similarity(i,j)
    
    Suporta três modos:
    1. Vetores: calcula similaridade on-the-fly (modo original)
    2. Matriz precalculada: usa similarity_matrix e ordered_ids
    3. Callable: usa similarity_fn(item_i, item_j)
    
    Args:
        item_ids: Lista de item_ids do conjunto
        vectors: Dicionário {item_id: vector} (modo 1)
        metric: Métrica de similaridade ('cosine', 'jaccard') (modo 1)
        normalize_by: Como normalizar ('n_items' ou 'n_pairs')
        similarity_matrix: Matriz de similaridade precalculada (modo 2)
        ordered_ids: IDs na ordem da matriz (modo 2)
        similarity_fn: Função (item_i, item_j) -> float (modo 3)
        **kwargs: Argumentos adicionais para a métrica (modo 1)
    
    Returns:
        Homogeneidade (ILS) no intervalo [0, 1]
    
    Raises:
        ValueError: Se nenhum modo for especificado corretamente
    """
    if len(item_ids) < 2:
        return np.nan
    
    # Determinar qual modo usar
    if similarity_matrix is not None and ordered_ids is not None:
        # MODO 2: Matriz precalculada
        # Criar mapeamento ID -> índice na matriz
        id_to_idx = {item_id: idx for idx, item_id in enumerate(ordered_ids)}
        
        # Filtrar apenas itens disponíveis na matriz
        available_items = [item_id for item_id in item_ids if item_id in id_to_idx]
        
        if len(available_items) < 2:
            return np.nan
        
        # Calcular similaridades usando matriz
        similarities = []
        for item_i, item_j in combinations(available_items, 2):
            idx_i = id_to_idx[item_i]
            idx_j = id_to_idx[item_j]
            sim = similarity_matrix[idx_i, idx_j]
            similarities.append(sim)
    
    elif similarity_fn is not None:
        # MODO 3: Callable
        # Assumir que todos os itens têm similaridade disponível via callable
        if len(item_ids) < 2:
            return np.nan
        
        similarities = []
        for item_i, item_j in combinations(item_ids, 2):
            sim = similarity_fn(item_i, item_j)
            similarities.append(sim)
        
        available_items = item_ids
    
    elif vectors is not None:
        # MODO 1: Vetores (modo original)
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
    
    else:
        raise ValueError(
            "Você deve fornecer um dos seguintes:\n"
            "  1. vectors (modo original)\n"
            "  2. similarity_matrix + ordered_ids (matriz precalculada)\n"
            "  3. similarity_fn (callable)"
        )
    
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
            - 'svd_features' ou 'svd_topics': usa 'cosine' (embeddings são L2-normalizados)
    
    Returns:
        Nome da métrica ('cosine' ou 'jaccard')
    """
    # Para embeddings densos (AE ou SVD), sempre cosine (já são L2-normalizados)
    if representation_type in ['ae_features', 'ae_topics', 'svd_features', 'svd_topics']:
        return 'cosine'
    
    # Para binários, manter comportamento atual (cosine)
    # Nota: o código original usa cosine, não jaccard
    if representation_type in ['bin_features', 'bin_topics']:
        return 'cosine'
    
    # Default: cosine
    return 'cosine'
