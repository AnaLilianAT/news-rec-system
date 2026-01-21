"""
Configurações globais do pipeline de replay temporal.
"""

# Seed aleatória para reprodutibilidade
RANDOM_SEED = 42

# Tamanho mínimo de candidate pool para processar checkpoint
MIN_CANDIDATE_SIZE = 97

# Tamanho de batch para processamento incremental
BATCH_SIZE = 100

# Cache de modelos por t_rec (para evitar retreinar se possível)
CACHE_BY_TREC = True

# Hiperparâmetros dos modelos Surprise
SURPRISE_PARAMS = {
    'knnu': {  # User-based KNN
        'k': 5,
        'min_k': 1,
        'sim_options': {
            'name': 'pearson',
            'user_based': True,
            'shrinkage': 100  # Shrinkage para reduzir overfitting
        }
    },
    'knni': {  # Item-based KNN
        'k': 5,
        'min_k': 1,
        'sim_options': {
            'name': 'pearson',
            'user_based': False,
            'shrinkage': 100  # Shrinkage para reduzir overfitting
        }
    },
    'svd': {  # SVD
        'n_factors': 100,
        'n_epochs': 20,
        'lr_all': 0.005,
        'reg_all': 0.02,
        'random_state': RANDOM_SEED
    }
}

# Mapeamento de algoritmos no formato do arquivo users.csv
ALGORITHM_MAPPING = {
    'knnu': ['knnu', 'knn_user', 'user-based'],
    'knni': ['knni', 'knn_item', 'item-based'],
    'svd': ['svd', 'matrix_factorization']
}

# Mapeamento de estratégias de diversificação
DIVERSIFY_MAPPING = {
    'td': ['td', 'topic-diversity', 'topic_diversity'],
    'mmr': ['mmr', 'maximal-marginal-relevance']
}

# Número máximo de itens a ranquear (candidate pool)
MAX_ITEMS_TO_RANK = 100

# Verbose para debug
VERBOSE = True
