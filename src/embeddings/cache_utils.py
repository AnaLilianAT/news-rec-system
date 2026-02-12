"""
Utilitários para gerenciar cache de embeddings com seed.

Sistema de cache que considera (d, seed, ae_config, datahash) para garantir
que cada combinação de parâmetros gere embeddings distintos e reproduzíveis.
"""

import hashlib
import json
from typing import Dict, Any, Tuple
from pathlib import Path


def compute_config_hash(ae_config: Dict[str, Any]) -> str:
    """
    Calcula hash MD5 da configuração do autoencoder.
    
    Args:
        ae_config: Dicionário com configurações do AE (hidden_dim, dropout_rate, etc.)
    
    Returns:
        String com hash hexadecimal (8 caracteres)
    """
    # Ordenar chaves para garantir consistência
    config_str = json.dumps(ae_config, sort_keys=True)
    hash_full = hashlib.md5(config_str.encode()).hexdigest()
    return hash_full[:8]  # Usar apenas 8 caracteres para nomes mais curtos


def make_embedding_cache_key(
    d: int,
    seed: int,
    ae_config: Dict[str, Any],
    datahash: str
) -> str:
    """
    Gera chave de cache única para embeddings.
    
    A chave combina todos os fatores que influenciam o embedding:
    - d: Dimensão do embedding
    - seed: Seed para treinamento
    - ae_config: Configuração do autoencoder (hidden_dim, dropout, epochs, etc.)
    - datahash: Hash dos dados de entrada
    
    Args:
        d: Dimensão do embedding
        seed: Seed usada no treinamento
        ae_config: Dict com config do AE (excluindo d e seed, que são explícitos)
        datahash: Hash MD5 dos dados binários de entrada
    
    Returns:
        String formatada: "d{d}_seed{seed}_{confighash[:8]}_{datahash[:8]}"
        Exemplo: "d32_seed100_a3f2b8c1_9e4d7a2f"
    """
    config_hash = compute_config_hash(ae_config)
    datahash_short = datahash[:8] if len(datahash) >= 8 else datahash
    
    cache_key = f"d{d}_seed{seed}_{config_hash}_{datahash_short}"
    return cache_key


def get_embedding_paths(
    base_dir: Path,
    representation_type: str,
    cache_key: str
) -> Tuple[Path, Path]:
    """
    Constrói caminhos para parquet e JSON de embeddings.
    
    Args:
        base_dir: Diretório base (ex: outputs/)
        representation_type: 'ae_features' ou 'ae_topics'
        cache_key: Chave de cache gerada por make_embedding_cache_key()
    
    Returns:
        Tupla (parquet_path, json_path)
        Exemplo:
            (outputs/embeddings/ae_features_d32_seed100_a3f2b8c1_9e4d7a2f.parquet,
             outputs/embeddings/ae_features_d32_seed100_a3f2b8c1_9e4d7a2f.json)
    """
    embeddings_dir = base_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_path = embeddings_dir / f"{representation_type}_{cache_key}.parquet"
    json_path = embeddings_dir / f"{representation_type}_{cache_key}.json"
    
    return parquet_path, json_path


def build_cache_metadata(
    d: int,
    seed: int,
    ae_config: Dict[str, Any],
    datahash: str,
    representation_type: str,
    cache_key: str,
    shape: Tuple[int, int],
    item_ids_hash: str = None
) -> Dict[str, Any]:
    """
    Constrói dicionário completo de metadados para salvar com embeddings.
    
    Args:
        d: Dimensão do embedding
        seed: Seed usada no treinamento
        ae_config: Configuração completa do AE
        datahash: Hash dos dados binários
        representation_type: 'ae_features' ou 'ae_topics'
        cache_key: Chave de cache gerada
        shape: Shape da matriz de embeddings (n_items, d)
        item_ids_hash: Hash opcional dos item_ids (para validação adicional)
    
    Returns:
        Dicionário com metadados completos
    """
    from datetime import datetime
    
    metadata = {
        # Identificação do cache
        'cache_key': cache_key,
        'representation_type': representation_type,
        
        # Parâmetros principais
        'embedding_dim': d,
        'seed': seed,
        
        # Configuração do AE
        'ae_config': ae_config,
        
        # Dados
        'data_hash': datahash,
        'shape': shape,
        
        # Timestamps
        'created_at': datetime.now().isoformat(),
        
        # Validação adicional
        'item_ids_hash': item_ids_hash
    }
    
    return metadata


def validate_cache_metadata(
    cached_metadata: Dict[str, Any],
    expected_d: int,
    expected_seed: int,
    expected_ae_config: Dict[str, Any],
    expected_datahash: str
) -> bool:
    """
    Valida se metadados do cache correspondem aos parâmetros esperados.
    
    Args:
        cached_metadata: Metadados carregados do JSON
        expected_d: Dimensão esperada
        expected_seed: Seed esperada
        expected_ae_config: Config do AE esperada
        expected_datahash: Hash dos dados esperado
    
    Returns:
        True se cache é válido, False caso contrário
    """
    # Validar dimensão
    if cached_metadata.get('embedding_dim') != expected_d:
        return False
    
    # Validar seed
    if cached_metadata.get('seed') != expected_seed:
        return False
    
    # Validar hash dos dados
    if cached_metadata.get('data_hash') != expected_datahash:
        return False
    
    # Validar configuração do AE
    cached_config = cached_metadata.get('ae_config', {})
    for key, value in expected_ae_config.items():
        if cached_config.get(key) != value:
            return False
    
    return True


def find_cached_embedding(
    base_dir: Path,
    representation_type: str,
    d: int,
    seed: int
) -> Tuple[Path, Path]:
    """
    Busca embedding em cache baseado em (representation_type, d, seed).
    
    Útil quando não sabemos o cache_key completo, mas queremos encontrar
    um embedding que corresponda aos parâmetros principais.
    
    Args:
        base_dir: Diretório base
        representation_type: 'ae_features' ou 'ae_topics'
        d: Dimensão do embedding
        seed: Seed do treinamento
    
    Returns:
        Tupla (parquet_path, json_path) do arquivo encontrado, ou (None, None)
    """
    embeddings_dir = base_dir / "embeddings"
    if not embeddings_dir.exists():
        return None, None
    
    # Padrão de busca: {type}_d{d}_seed{seed}_*.parquet
    pattern = f"{representation_type}_d{d}_seed{seed}_*.parquet"
    matches = list(embeddings_dir.glob(pattern))
    
    if not matches:
        return None, None
    
    # Retornar primeiro match (deveria haver apenas um)
    parquet_path = matches[0]
    json_path = parquet_path.with_suffix('.json')
    
    if json_path.exists():
        return parquet_path, json_path
    else:
        return None, None


def compute_item_ids_hash(item_ids) -> str:
    """
    Calcula hash dos item_ids para validação adicional.
    
    Args:
        item_ids: Array ou lista de news_id
    
    Returns:
        Hash MD5 hexadecimal (8 caracteres)
    """
    import numpy as np
    
    if not isinstance(item_ids, np.ndarray):
        item_ids = np.array(item_ids)
    
    ids_bytes = item_ids.tobytes()
    hash_full = hashlib.md5(ids_bytes).hexdigest()
    return hash_full[:8]


if __name__ == '__main__':
    # Testes básicos
    print("="*70)
    print("TESTES DO MÓDULO cache_utils")
    print("="*70)
    
    # Teste 1: Gerar cache key
    print("\nTeste 1: make_embedding_cache_key()")
    ae_config = {
        'hidden_dim': 64,
        'dropout_rate': 0.2,
        'epochs': 100,
        'batch_size': 512,
        'learning_rate': 0.001
    }
    datahash = "a1b2c3d4e5f6g7h8"
    
    key1 = make_embedding_cache_key(d=32, seed=100, ae_config=ae_config, datahash=datahash)
    print(f"  key1 (d=32, seed=100): {key1}")
    
    key2 = make_embedding_cache_key(d=32, seed=200, ae_config=ae_config, datahash=datahash)
    print(f"  key2 (d=32, seed=200): {key2}")
    
    key3 = make_embedding_cache_key(d=64, seed=100, ae_config=ae_config, datahash=datahash)
    print(f"  key3 (d=64, seed=100): {key3}")
    
    assert key1 != key2, "Seeds diferentes devem gerar keys diferentes"
    assert key1 != key3, "Dimensões diferentes devem gerar keys diferentes"
    print("  ✓ Keys são distintas para diferentes parâmetros")
    
    # Teste 2: Paths
    print("\nTeste 2: get_embedding_paths()")
    base_dir = Path("outputs")
    parquet_path, json_path = get_embedding_paths(base_dir, "ae_features", key1)
    print(f"  Parquet: {parquet_path}")
    print(f"  JSON: {json_path}")
    assert "d32_seed100" in str(parquet_path), "Path deve conter d e seed"
    print("  ✓ Paths gerados corretamente")
    
    # Teste 3: Metadados
    print("\nTeste 3: build_cache_metadata()")
    metadata = build_cache_metadata(
        d=32,
        seed=100,
        ae_config=ae_config,
        datahash=datahash,
        representation_type='ae_features',
        cache_key=key1,
        shape=(1000, 32),
        item_ids_hash="abc123"
    )
    print(f"  Embedding dim: {metadata['embedding_dim']}")
    print(f"  Seed: {metadata['seed']}")
    print(f"  Cache key: {metadata['cache_key']}")
    print(f"  Shape: {metadata['shape']}")
    assert metadata['embedding_dim'] == 32
    assert metadata['seed'] == 100
    print("  ✓ Metadados construídos corretamente")
    
    # Teste 4: Validação
    print("\nTeste 4: validate_cache_metadata()")
    is_valid = validate_cache_metadata(metadata, 32, 100, ae_config, datahash)
    print(f"  Validação (mesmos params): {is_valid}")
    assert is_valid, "Deve validar com mesmos parâmetros"
    
    is_invalid = validate_cache_metadata(metadata, 32, 200, ae_config, datahash)
    print(f"  Validação (seed diferente): {is_invalid}")
    assert not is_invalid, "Não deve validar com seed diferente"
    print("  ✓ Validação funcionando corretamente")
    
    print("\n" + "="*70)
    print("TODOS OS TESTES PASSARAM ✓")
    print("="*70)
