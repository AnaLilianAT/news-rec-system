"""
Configurações globais do pipeline de replay temporal.
"""

# ============================================================================
# REPRODUTIBILIDADE
# ============================================================================

# Seed aleatória para reprodutibilidade
RANDOM_SEED = 42

# ============================================================================
# AUTOENCODER (EMBEDDINGS)
# ============================================================================

# Configurações centralizadas do autoencoder para reprodutibilidade
# Otimizado para entrada binária esparsa (~85% zeros), N~2k itens
AUTOENCODER_CONFIG = {
    # Arquitetura
    'embedding_dim': 28,           # Dimensão final do embedding (camada do meio)
    'hidden_dim': 52,              # Dimensão oculta (64 é adequado para D~81 esparso)
    
    # Treinamento
    'epochs': 300,                 # Número de épocas de treinamento
    'batch_size': 32,             # Tamanho do batch (maior para dados esparsos)
    'learning_rate': 0.001,        # Taxa de aprendizado (Adam optimizer)
    'weight_decay': 1e-5,          # Weight decay para regularização L2
    
    # Regularização
    'dropout_rate': 0.2,           # Dropout entre camadas (0.1 para evitar overfitting)
    'denoising_prob': 0.2,         # Probabilidade de corrupção para denoising AE
    
    # Early stopping
    'early_stopping_patience': 15, # Paciência para early stopping (0 = desabilitado)
    
    # Tratamento de desbalanceamento
    'pos_weight_mode': 'sqrt',     # 'auto', 'sqrt', ou valor float (ex: 2.0)
    'pos_weight_clip': [1.5, 8.0],# Clipping de pos_weight [min, max]
    
    # Tratamento de features contínuas
    'include_continuous': False,   # Se True, inclui features contínuas no treino
    'continuous_cols': ['polaridade', 'subjetividade'],  # Colunas contínuas
    'concat_continuous_after': True,  # Se True, concatena após embedding
    
    # Pós-processamento
    'l2_normalize': True,          # Normalizar embeddings (L2) após extração
    
    # Reprodutibilidade
    'seed': RANDOM_SEED,           # Seed para torch, numpy e random
    
    # Cache
    'use_cache': True,             # Habilitar cache de embeddings
    'invalidate_on_data_change': True,  # Invalidar cache se dados mudarem
    'invalidate_on_config_change': True  # Invalidar cache se config mudar
}

# ============================================================================
# PIPELINE
# ============================================================================

# Tamanho mínimo de candidate pool para processar checkpoint
MIN_CANDIDATE_SIZE = 20

# Tamanho de batch para processamento incremental
BATCH_SIZE = 100

# Cache de modelos por t_rec (para evitar retreinar se possível)
CACHE_BY_TREC = True

# ============================================================================
# HIPERPARÂMETROS DOS MODELOS SURPRISE
# ============================================================================

SURPRISE_PARAMS = {
    'knnu': {  # User-based KNN
        'k': 5,
        'min_k': 1,
        'sim_options': {
            'name': 'pearson_baseline',
            'user_based': True,
            'shrinkage': 100  # Shrinkage para reduzir overfitting
        }
    },
    'knni': {  # Item-based KNN
        'k': 5,
        'min_k': 1,
        'sim_options': {
            'name': 'pearson_baseline',
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

# ============================================================================
# MAPEAMENTO DE ALGORITMOS
# ============================================================================

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
