"""
Módulo para gerenciar representações de itens (notícias).

Suporta:
- Representações binárias (features, tópicos)
- Embeddings densos (a ser implementado)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ItemRepresentation:
    """
    Classe para encapsular representações de itens.
    
    Attributes:
        matrix: DataFrame com news_id e colunas de features/embeddings
        item_ids: Lista de news_id presentes
        feature_names: Lista de nomes das features/dimensões
        representation_type: Tipo da representação ('bin_features', 'bin_topics', 'ae_features', 'ae_topics')
        metadata: Metadados adicionais (dimensionalidade, origem, etc.)
    """
    matrix: pd.DataFrame
    item_ids: List[int]
    feature_names: List[str]
    representation_type: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[int, np.ndarray]:
        """
        Converte representação para dicionário {news_id: vector}.
        
        Returns:
            Dicionário {news_id: np.ndarray} com vetores por item
        """
        vectors = {}
        for _, row in self.matrix.iterrows():
            news_id = row['news_id']
            vec = row[self.feature_names].values.astype(float)
            vectors[news_id] = vec
        return vectors
    
    def __len__(self) -> int:
        return len(self.item_ids)
    
    def __repr__(self) -> str:
        return (f"ItemRepresentation(type='{self.representation_type}', "
                f"items={len(self.item_ids)}, dims={len(self.feature_names)})")


def get_item_representation(
    kind: str,
    data_path: Optional[Path] = None,
    output_dir: str = 'outputs',
    embedding_dim: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs
) -> ItemRepresentation:
    """
    Carrega representação de itens de acordo com o tipo especificado.
    
    Args:
        kind: Tipo de representação
              - 'bin_features': Features binárias + numéricas (canonical_features.parquet)
              - 'bin_topics': Tópicos binários (canonical_topics.parquet)
              - 'ae_features': Embeddings de features
              - 'ae_topics': Embeddings de tópicos
        data_path: Caminho explícito para o arquivo (opcional)
        output_dir: Diretório base dos outputs (default: 'outputs')
        embedding_dim: Dimensão dos embeddings (apenas para ae_features/ae_topics).
                      Se None, usa 32 como padrão ou detecta automaticamente.
        seed: Seed usada no treinamento do embedding (apenas para ae_features/ae_topics).
              Se None, tenta encontrar arquivo mais recente ou usa busca por padrão.
        **kwargs: Argumentos adicionais (para futura compatibilidade)
    
    Returns:
        ItemRepresentation com matriz, ids e metadados
    
    Raises:
        ValueError: Se kind não for reconhecido
        FileNotFoundError: Se arquivo não existir
    """
    output_path = Path(output_dir)
    
    # Para representações binárias, usar caminhos fixos
    if kind in ['bin_features', 'bin_topics']:
        default_paths = {
            'bin_features': output_path / 'canonical_features.parquet',
            'bin_topics': output_path / 'canonical_topics.parquet'
        }
        file_path = data_path if data_path else default_paths[kind]
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Arquivo de representação não encontrado: {file_path}\n"
                f"Certifique-se de que o pipeline gerou '{file_path.name}'"
            )
        
        return _load_binary_representation(file_path, kind)
    
    # Para embeddings, buscar por (d, seed) usando find_cached_embedding
    elif kind in ['ae_features', 'ae_topics']:
        # Se data_path explícito foi fornecido, usar diretamente
        if data_path:
            file_path = data_path
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Arquivo de embedding não encontrado: {file_path}"
                )
            return _load_embedding_representation(file_path, kind)
        
        # Caso contrário, buscar por (d, seed)
        # Se seed não foi fornecida, tentar buscar qualquer embedding com d
        from ..embeddings.cache_utils import find_cached_embedding
        
        # Usar embedding_dim se fornecido, caso contrário 32
        d = embedding_dim if embedding_dim is not None else 32
        
        # Se seed fornecida, buscar exatamente
        if seed is not None:
            parquet_path, json_path = find_cached_embedding(
                base_dir=output_path,
                representation_type=kind,
                d=d,
                seed=seed
            )
            
            if parquet_path and parquet_path.exists():
                return _load_embedding_representation(parquet_path, kind)
            else:
                raise FileNotFoundError(
                    f"Embedding não encontrado: {kind} com d={d}, seed={seed}\n"
                    f"Execute o treinamento primeiro com: python -m src.embeddings.train_embeddings "
                    f"--embedding-dim {d} --seed {seed}"
                )
        else:
            # Seed não fornecida: buscar qualquer embedding com d (fallback para compatibilidade)
            # Buscar padrão: {kind}_d{d}_seed*_*.parquet
            embeddings_dir = output_path / "embeddings"
            if embeddings_dir.exists():
                pattern = f"{kind}_d{d}_seed*.parquet"
                matches = sorted(embeddings_dir.glob(pattern))
                
                if matches:
                    # Usar primeiro match encontrado (mais antigo)
                    parquet_path = matches[0]
                    print(f"[AVISO] Seed não especificada. Usando: {parquet_path.name}")
                    return _load_embedding_representation(parquet_path, kind)
            
            # Fallback antigo: buscar arquivo sem seed (compatibilidade)
            old_path = output_path / 'embeddings' / f'{kind}_dim{d}.parquet'
            if old_path.exists():
                print(f"[AVISO] Usando arquivo legado sem seed: {old_path.name}")
                return _load_embedding_representation(old_path, kind)
            
            raise FileNotFoundError(
                f"Nenhum embedding encontrado para {kind} com d={d}\n"
                f"Execute o treinamento primeiro com: python -m src.embeddings.train_embeddings "
                f"--embedding-dim {d} --seed <SEED>"
            )
    
    else:
        raise ValueError(
            f"Tipo de representação desconhecido: '{kind}'. "
            f"Tipos válidos: ['bin_features', 'bin_topics', 'ae_features', 'ae_topics']"
        )


def _load_binary_representation(file_path: Path, kind: str) -> ItemRepresentation:
    """
    Carrega representação binária (features ou tópicos).
    
    Args:
        file_path: Caminho para arquivo parquet
        kind: 'bin_features' ou 'bin_topics'
    
    Returns:
        ItemRepresentation com matriz binária
    """
    # Carregar parquet
    df = pd.read_parquet(file_path)
    
    # Validar estrutura
    if 'news_id' not in df.columns:
        raise ValueError(f"Coluna 'news_id' não encontrada em {file_path}")
    
    # Extrair colunas de features
    feature_cols = [col for col in df.columns if col != 'news_id']
    
    if len(feature_cols) == 0:
        raise ValueError(f"Nenhuma coluna de feature encontrada em {file_path}")
    
    # Extrair item_ids
    item_ids = df['news_id'].tolist()
    
    # Metadados
    metadata = {
        'source_file': str(file_path),
        'n_items': len(item_ids),
        'n_features': len(feature_cols),
        'is_binary': True,
        'is_dense': False
    }
    
    if kind == 'bin_topics':
        # Para tópicos, verificar que são Topic0..TopicN
        topic_cols = [c for c in feature_cols if c.startswith('Topic')]
        metadata['n_topics'] = len(topic_cols)
    
    return ItemRepresentation(
        matrix=df,
        item_ids=item_ids,
        feature_names=feature_cols,
        representation_type=kind,
        metadata=metadata
    )


def _load_embedding_representation(file_path: Path, kind: str) -> ItemRepresentation:
    """
    Carrega representação densa (embeddings de autoencoder).
    
    Args:
        file_path: Caminho para arquivo parquet de embeddings
        kind: 'ae_features' ou 'ae_topics'
    
    Returns:
        ItemRepresentation com embeddings densos
    
    Raises:
        FileNotFoundError: Se arquivo de embeddings não existir
        ValueError: Se estrutura do arquivo estiver incorreta
    """
    import json
    
    # Carregar parquet de embeddings
    df = pd.read_parquet(file_path)
    
    # Validar estrutura
    if 'news_id' not in df.columns:
        raise ValueError(f"Coluna 'news_id' não encontrada em {file_path}")
    
    # Extrair colunas de embeddings (emb_0, emb_1, ...)
    emb_cols = [col for col in df.columns if col.startswith('emb_')]
    
    if len(emb_cols) == 0:
        raise ValueError(f"Nenhuma coluna de embedding encontrada em {file_path}")
    
    # Extrair item_ids
    item_ids = df['news_id'].tolist()
    
    # Tentar carregar metadados do JSON associado
    json_path = file_path.with_suffix('.json')
    json_metadata = {}
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                json_metadata = json.load(f)
        except Exception:
            pass  # Ignorar se JSON não puder ser lido
    
    # Metadados
    metadata = {
        'source_file': str(file_path),
        'n_items': len(item_ids),
        'n_features': len(emb_cols),
        'embedding_dim': len(emb_cols),
        'is_binary': False,
        'is_dense': True,
        'is_normalized': True  # Embeddings são L2-normalizados por padrão
    }
    
    # Adicionar metadados do treino se disponíveis
    if json_metadata:
        metadata.update({
            'training_config': {
                'epochs': json_metadata.get('epochs'),
                'batch_size': json_metadata.get('batch_size'),
                'learning_rate': json_metadata.get('learning_rate'),
                'dropout_rate': json_metadata.get('dropout_rate'),
                'seed': json_metadata.get('seed')
            },
            'data_hash': json_metadata.get('data_hash')
        })
    
    return ItemRepresentation(
        matrix=df,
        item_ids=item_ids,
        feature_names=emb_cols,
        representation_type=kind,
        metadata=metadata
    )


def prepare_item_vectors(
    representation: ItemRepresentation,
    item_ids: Optional[List[int]] = None
) -> Dict[int, np.ndarray]:
    """
    Converte representação para dicionário de vetores.
    
    Args:
        representation: ItemRepresentation carregada
        item_ids: Lista opcional de item_ids para filtrar (None = todos)
    
    Returns:
        Dicionário {news_id: np.ndarray}
    """
    vectors = representation.to_dict()
    
    if item_ids is not None:
        # Filtrar apenas item_ids solicitados
        vectors = {nid: vec for nid, vec in vectors.items() if nid in item_ids}
    
    return vectors


def get_representation_metadata(kind: str, output_dir: str = 'outputs') -> Dict[str, Any]:
    """
    Retorna metadados de uma representação sem carregar a matriz completa.
    
    Args:
        kind: Tipo de representação
        output_dir: Diretório base dos outputs
    
    Returns:
        Dicionário com metadados (n_items, n_features, source_file, etc.)
    """
    try:
        rep = get_item_representation(kind, output_dir=output_dir)
        return rep.metadata
    except Exception as e:
        return {
            'error': str(e),
            'kind': kind,
            'available': False
        }
