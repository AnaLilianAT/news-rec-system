"""
Utilitários para gerenciamento de seeds de experimentos.

Este módulo provê funções para gerar e carregar seeds de forma determinística
e reprodutível, garantindo que experimentos possam ser replicados.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional


def generate_seeds(master_seed: int, n_seeds: int) -> List[int]:
    """
    Gera lista de seeds de forma determinística a partir de uma master seed.
    
    Usa numpy.random.default_rng() para gerar inteiros únicos no intervalo
    [0, 2**31-1), garantindo reprodutibilidade perfeita.
    
    Args:
        master_seed: Seed mestre para geração determinística
        n_seeds: Número de seeds a gerar
    
    Returns:
        Lista de n_seeds inteiros únicos
    
    Examples:
        >>> seeds1 = generate_seeds(20260211, 5)
        >>> seeds2 = generate_seeds(20260211, 5)
        >>> seeds1 == seeds2
        True
        >>> len(seeds1)
        5
        >>> len(set(seeds1))  # Todas únicas
        5
    """
    rng = np.random.default_rng(master_seed)
    
    # Gerar n_seeds inteiros únicos no intervalo [0, 2**31-1)
    # Usamos choice sem replacement para garantir unicidade
    max_int = 2**31 - 1
    seeds = rng.choice(max_int, size=n_seeds, replace=False)
    
    # Converter para int nativo do Python (para serialização JSON)
    return [int(seed) for seed in seeds]


def load_seeds_from_file(seeds_file: Path) -> List[int]:
    """
    Carrega seeds de um arquivo JSON.
    
    Args:
        seeds_file: Caminho para arquivo JSON com seeds
    
    Returns:
        Lista de seeds
    
    Raises:
        FileNotFoundError: Se arquivo não existir
        ValueError: Se formato do arquivo for inválido
    """
    if not seeds_file.exists():
        raise FileNotFoundError(f"Arquivo de seeds não encontrado: {seeds_file}")
    
    try:
        with open(seeds_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validar estrutura
        if not isinstance(data, dict):
            raise ValueError(f"Formato inválido: esperado dict, recebido {type(data)}")
        
        if 'seeds' not in data:
            raise ValueError("Chave 'seeds' não encontrada no arquivo")
        
        seeds = data['seeds']
        
        if not isinstance(seeds, list):
            raise ValueError(f"'seeds' deve ser uma lista, recebido {type(seeds)}")
        
        if not all(isinstance(s, int) for s in seeds):
            raise ValueError("Todas as seeds devem ser inteiros")
        
        return seeds
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Erro ao parsear JSON: {e}")


def save_seeds_to_file(
    seeds: List[int],
    seeds_file: Path,
    master_seed: int,
    n_seeds: int
) -> None:
    """
    Salva seeds em arquivo JSON com metadados.
    
    Args:
        seeds: Lista de seeds a salvar
        seeds_file: Caminho para arquivo JSON
        master_seed: Seed mestre usada para geração
        n_seeds: Número de seeds geradas
    """
    seeds_file.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'master_seed': master_seed,
        'n_seeds': n_seeds,
        'seeds': seeds,
        'info': {
            'description': 'Seeds fixas para experimentos de reprodutibilidade',
            'generation_method': 'numpy.random.default_rng() with choice(replace=False)',
            'range': '[0, 2**31-1)',
            'uniqueness': 'guaranteed'
        }
    }
    
    with open(seeds_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_or_create_seeds(
    seeds_file: str = 'configs/experiment_seeds.json',
    master_seed: int = 20260211,
    n_seeds: int = 20
) -> List[int]:
    """
    Carrega seeds de arquivo ou cria automaticamente se não existir.
    
    Esta é a função principal do módulo. Garante reprodutibilidade:
    - Se o arquivo existir: carrega e retorna as seeds salvas
    - Se não existir: gera seeds deterministicamente e salva no arquivo
    
    Args:
        seeds_file: Caminho para arquivo JSON (relativo ao workspace root)
        master_seed: Seed mestre para geração (default: 20260211)
        n_seeds: Número de seeds a gerar (default: 20)
    
    Returns:
        Lista de seeds inteiras
    
    Examples:
        >>> # Primeira chamada: cria arquivo
        >>> seeds1 = load_or_create_seeds()
        >>> # Segunda chamada: lê do arquivo
        >>> seeds2 = load_or_create_seeds()
        >>> seeds1 == seeds2
        True
    
    Notes:
        - O arquivo é sempre relativo ao workspace root
        - Seeds são geradas de forma determinística (mesma master_seed = mesmas seeds)
        - O arquivo contém metadados para rastreabilidade
    """
    seeds_path = Path(seeds_file)
    
    # Tentar carregar seeds existentes
    if seeds_path.exists():
        print(f"[SEEDS] Carregando seeds existentes de: {seeds_path}")
        try:
            seeds = load_seeds_from_file(seeds_path)
            print(f"[SEEDS] {len(seeds)} seeds carregadas com sucesso")
            
            # Validar que temos seeds suficientes
            if len(seeds) < n_seeds:
                print(f"[SEEDS] AVISO: Arquivo tem apenas {len(seeds)} seeds, mas {n_seeds} foram solicitadas")
                print(f"[SEEDS] Usando as primeiras {len(seeds)} seeds disponíveis")
            
            return seeds[:n_seeds]  # Retornar apenas o que foi solicitado
        
        except (FileNotFoundError, ValueError) as e:
            print(f"[SEEDS] ERRO ao carregar arquivo: {e}")
            print(f"[SEEDS] Gerando novas seeds...")
    
    # Arquivo não existe ou houve erro: gerar novas seeds
    print(f"[SEEDS] Arquivo não encontrado: {seeds_path}")
    print(f"[SEEDS] Gerando {n_seeds} seeds deterministicamente (master_seed={master_seed})")
    
    seeds = generate_seeds(master_seed, n_seeds)
    
    # Salvar para uso futuro
    save_seeds_to_file(seeds, seeds_path, master_seed, n_seeds)
    print(f"[SEEDS] Seeds salvas em: {seeds_path}")
    print(f"[SEEDS] Seeds geradas: {seeds[:5]}... (mostrando 5 primeiras)")
    
    return seeds


def validate_seeds_reproducibility(
    master_seed: int = 20260211,
    n_seeds: int = 20,
    n_trials: int = 3
) -> bool:
    """
    Valida que a geração de seeds é reprodutível.
    
    Args:
        master_seed: Seed mestre a testar
        n_seeds: Número de seeds a gerar
        n_trials: Número de tentativas para validar reprodutibilidade
    
    Returns:
        True se todas as tentativas produzirem seeds idênticas
    
    Examples:
        >>> validate_seeds_reproducibility()
        True
    """
    reference_seeds = generate_seeds(master_seed, n_seeds)
    
    for trial in range(n_trials):
        trial_seeds = generate_seeds(master_seed, n_seeds)
        if trial_seeds != reference_seeds:
            print(f"FALHA: Trial {trial+1} produziu seeds diferentes!")
            return False
    
    print(f"SUCESSO: {n_trials} tentativas produziram seeds idênticas")
    return True


# CLI para teste standalone
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Gerenciador de seeds para experimentos'
    )
    
    parser.add_argument(
        '--master-seed',
        type=int,
        default=20260211,
        help='Seed mestre para geração determinística (default: 20260211)'
    )
    
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=20,
        help='Número de seeds a gerar (default: 20)'
    )
    
    parser.add_argument(
        '--seeds-file',
        type=str,
        default='configs/experiment_seeds.json',
        help='Caminho para arquivo JSON de seeds (default: configs/experiment_seeds.json)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validar reprodutibilidade da geração de seeds'
    )
    
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Forçar regeneração mesmo se arquivo existir'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("  GERENCIADOR DE SEEDS PARA EXPERIMENTOS")
    print("="*70)
    
    if args.validate:
        print("\nValidando reprodutibilidade...")
        is_valid = validate_seeds_reproducibility(
            master_seed=args.master_seed,
            n_seeds=args.n_seeds,
            n_trials=5
        )
        
        if is_valid:
            print("\n✓ Validação PASSOU: Seeds são reprodutíveis")
        else:
            print("\n✗ Validação FALHOU: Seeds não são reprodutíveis")
            exit(1)
    
    # Forçar regeneração se solicitado
    if args.force_regenerate:
        seeds_path = Path(args.seeds_file)
        if seeds_path.exists():
            print(f"\nRemovendo arquivo existente: {seeds_path}")
            seeds_path.unlink()
    
    # Carregar ou criar seeds
    print(f"\nConfigurações:")
    print(f"  Master seed: {args.master_seed}")
    print(f"  Número de seeds: {args.n_seeds}")
    print(f"  Arquivo: {args.seeds_file}")
    
    seeds = load_or_create_seeds(
        seeds_file=args.seeds_file,
        master_seed=args.master_seed,
        n_seeds=args.n_seeds
    )
    
    print("\n" + "="*70)
    print("  SEEDS GERADAS/CARREGADAS")
    print("="*70)
    print(f"\nTotal: {len(seeds)} seeds")
    print(f"\nPrimeiras 10 seeds:")
    for i, seed in enumerate(seeds[:10], 1):
        print(f"  [{i:2d}] {seed:10d}")
    
    if len(seeds) > 10:
        print(f"  ... ({len(seeds) - 10} seeds adicionais)")
    
    print(f"\nArquivo salvo em: {args.seeds_file}")
    print("\n✓ Operação concluída com sucesso!")
