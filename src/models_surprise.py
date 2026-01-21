"""
Wrappers para algoritmos da biblioteca Surprise para uso no replay temporal.
"""
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .config import SURPRISE_PARAMS, RANDOM_SEED


class SurpriseModelWrapper:
    """
    Wrapper base para modelos Surprise.
    """
    
    def __init__(self, algo_name: str):
        self.algo_name = algo_name
        self.model = None
        self.trainset = None
        self.trained = False
    
    def fit(self, train_df: pd.DataFrame):
        """
        Treina o modelo com dados de treino.
        
        Args:
            train_df: DataFrame com colunas [user_id, news_id, rating]
        """
        # Preparar dados no formato Surprise
        reader = Reader(rating_scale=(-1, 1))  # Ratings: -1, 0, 1
        
        # Criar dataset
        data = Dataset.load_from_df(
            train_df[['user_id', 'news_id', 'rating']], 
            reader
        )
        
        # Construir trainset completo
        self.trainset = data.build_full_trainset()
        
        # Treinar modelo
        self.model.fit(self.trainset)
        self.trained = True
    
    def predict_score(self, user_id: int, item_id: int) -> float:
        """
        Prediz score para um par (user, item).
        
        Args:
            user_id: ID do usuário
            item_id: ID do item
        
        Returns:
            Score predito (float)
        """
        if not self.trained:
            raise ValueError("Modelo não foi treinado. Chame fit() primeiro.")
        
        # Fazer predição
        prediction = self.model.predict(user_id, item_id, verbose=False)
        
        return prediction.est
    
    def predict_batch(self, user_id: int, item_ids: List[int]) -> Dict[int, float]:
        """
        Prediz scores para múltiplos itens de um usuário.
        
        Args:
            user_id: ID do usuário
            item_ids: Lista de IDs de itens
        
        Returns:
            Dicionário {item_id: score}
        """
        scores = {}
        for item_id in item_ids:
            try:
                scores[item_id] = self.predict_score(user_id, item_id)
            except:
                # Se falhar (e.g., item/user não visto), usar score padrão
                scores[item_id] = 0.0
        
        return scores


class KNNUserModel(SurpriseModelWrapper):
    """
    Wrapper para KNN User-based (knnu).
    """
    
    def __init__(self):
        super().__init__('knnu')
        params = SURPRISE_PARAMS['knnu']
        self.model = KNNBasic(
            k=params['k'],
            min_k=params['min_k'],
            sim_options=params['sim_options']
        )


class KNNItemModel(SurpriseModelWrapper):
    """
    Wrapper para KNN Item-based (knni).
    """
    
    def __init__(self):
        super().__init__('knni')
        params = SURPRISE_PARAMS['knni']
        self.model = KNNBasic(
            k=params['k'],
            min_k=params['min_k'],
            sim_options=params['sim_options']
        )


class SVDModel(SurpriseModelWrapper):
    """
    Wrapper para SVD.
    """
    
    def __init__(self):
        super().__init__('svd')
        params = SURPRISE_PARAMS['svd']
        self.model = SVD(
            n_factors=params['n_factors'],
            n_epochs=params['n_epochs'],
            lr_all=params['lr_all'],
            reg_all=params['reg_all'],
            random_state=params['random_state']
        )


def create_model(algo_name: str) -> SurpriseModelWrapper:
    """
    Factory para criar modelos Surprise.
    
    Args:
        algo_name: Nome do algoritmo ('knnu', 'knni', 'svd')
    
    Returns:
        Instância do modelo wrapper
    """
    if algo_name == 'knnu':
        return KNNUserModel()
    elif algo_name == 'knni':
        return KNNItemModel()
    elif algo_name == 'svd':
        return SVDModel()
    else:
        raise ValueError(f"Algoritmo desconhecido: {algo_name}. Use: knnu, knni, svd")


def parse_user_algorithm(algo_text: str) -> Tuple[str, str]:
    """
    Parseia o texto do algoritmo do usuário para extrair base_algo e diversify.
    
    Args:
        algo_text: Texto do campo 'algoritmo' em users.csv
    
    Returns:
        Tupla (base_algo, diversify)
        - base_algo: 'knnu', 'knni', ou 'svd'
        - diversify: 'none', 'td', ou 'mmr'
    """
    algo_text_lower = str(algo_text).lower()
    
    # Identificar base_algo (ordem importa: checar user antes de item)
    base_algo = 'svd'  # default
    if 'knn_user' in algo_text_lower or 'user-based' in algo_text_lower or 'user_based' in algo_text_lower:
        base_algo = 'knnu'
    elif 'knn_item' in algo_text_lower or 'item-based' in algo_text_lower or 'item_based' in algo_text_lower:
        base_algo = 'knni'
    elif 'svd' in algo_text_lower:
        base_algo = 'svd'
    
    # Identificar diversify
    diversify = 'none'  # default
    if 'topic' in algo_text_lower or 'diversification' in algo_text_lower:
        # Checar se é mmr ou topic-based
        if 'mmr' in algo_text_lower:
            diversify = 'mmr'
        else:
            diversify = 'td'
    elif 'mmr' in algo_text_lower:
        diversify = 'mmr'
    
    return base_algo, diversify


def rank_candidates_by_score(
    scores: Dict[int, float], 
    max_items: int = 100
) -> List[Tuple[int, float, int]]:
    """
    Ranqueia candidatos por score.
    
    Args:
        scores: Dicionário {item_id: score}
        max_items: Número máximo de itens a retornar
    
    Returns:
        Lista de tuplas (item_id, score, rank) ordenada por score desc
    """
    # Ordenar por score descendente
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Limitar a max_items
    sorted_items = sorted_items[:max_items]
    
    # Adicionar rank (1-based)
    ranked = [(item_id, score, rank + 1) for rank, (item_id, score) in enumerate(sorted_items)]
    
    return ranked


if __name__ == '__main__':
    # Teste básico
    print("Testando parse_user_algorithm:")
    test_cases = [
        "knnu",
        "knni-td",
        "svd-mmr",
        "user-based",
        "item-based-topic-diversity"
    ]
    
    for case in test_cases:
        base_algo, diversify = parse_user_algorithm(case)
        print(f"  '{case}' -> base_algo='{base_algo}', diversify='{diversify}'")
