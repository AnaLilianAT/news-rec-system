"""
Utilitários para geração de grade de dimensões de embeddings.

Este módulo provê funções para construir listas de dimensões para sweeps,
garantindo que d_max está sempre incluído e validando propriedades da grade.
"""

from typing import List, Optional


def build_dims(
    d_min: int,
    d_max: int,
    step: int = 5,
    dims_list: Optional[List[int]] = None
) -> List[int]:
    """
    Constrói lista de dimensões para sweep de embeddings.
    
    Regras:
    - Se dims_list for fornecida, usa ela (ignorando d_min/d_max/step)
    - Caso contrário, gera range(d_min, d_max+1, step)
    - SEMPRE garante que d_max está incluído (adiciona ao final se necessário)
    - Retorna lista ordenada e sem duplicatas
    
    Args:
        d_min: Dimensão mínima
        d_max: Dimensão máxima (sempre incluída no resultado)
        step: Passo entre dimensões (default: 5)
        dims_list: Lista manual de dimensões (se fornecida, ignora outros params)
    
    Returns:
        Lista de dimensões inteiras, ordenadas, únicas, com d_max garantido
    
    Raises:
        ValueError: Se parâmetros forem inválidos
    
    Examples:
        >>> # Caso 1: step=5, d_max cai exatamente no passo
        >>> build_dims(d_min=10, d_max=30, step=5)
        [10, 15, 20, 25, 30]
        
        >>> # Caso 2: step=5, d_max NÃO cai no passo (adiciona ao final)
        >>> build_dims(d_min=10, d_max=32, step=5)
        [10, 15, 20, 25, 30, 32]
        
        >>> # Caso 3: step=1 (equivalente a range completo)
        >>> build_dims(d_min=13, d_max=16, step=1)
        [13, 14, 15, 16]
        
        >>> # Caso 4: Lista manual (ignora outros params)
        >>> build_dims(d_min=10, d_max=50, step=5, dims_list=[8, 16, 32, 64])
        [8, 16, 32, 64]
        
        >>> # Caso 5: d_min == d_max
        >>> build_dims(d_min=32, d_max=32, step=5)
        [32]
    """
    # Caso 1: Lista manual fornecida
    if dims_list is not None:
        if not dims_list:
            raise ValueError("dims_list não pode ser vazia")
        
        # Validar e ordenar
        validated_dims = validate_dims_list(dims_list, d_max=None)
        return validated_dims
    
    # Caso 2: Gerar a partir de range
    # Validações básicas
    if d_min <= 0:
        raise ValueError(f"d_min deve ser positivo, recebido: {d_min}")
    
    if d_max <= 0:
        raise ValueError(f"d_max deve ser positivo, recebido: {d_max}")
    
    if d_min > d_max:
        raise ValueError(f"d_min ({d_min}) não pode ser maior que d_max ({d_max})")
    
    if step <= 0:
        raise ValueError(f"step deve ser positivo, recebido: {step}")
    
    # Gerar range básico
    dims = list(range(d_min, d_max + 1, step))
    
    # Garantir que d_max está incluído
    if dims[-1] != d_max:
        dims.append(d_max)
    
    # Validar resultado
    validated_dims = validate_dims_list(dims, d_max=d_max)
    
    return validated_dims


def validate_dims_list(dims: List[int], d_max: Optional[int] = None) -> List[int]:
    """
    Valida e normaliza lista de dimensões.
    
    Validações:
    - Todos os valores devem ser inteiros positivos
    - Lista deve ser ordenada (ordena automaticamente se necessário)
    - Não pode haver duplicatas (remove automaticamente)
    - Se d_max fornecido, garante que está incluído
    
    Args:
        dims: Lista de dimensões a validar
        d_max: Dimensão máxima esperada (opcional)
    
    Returns:
        Lista validada e normalizada
    
    Raises:
        ValueError: Se validação falhar
    
    Examples:
        >>> validate_dims_list([10, 20, 30])
        [10, 20, 30]
        
        >>> validate_dims_list([30, 10, 20])  # Ordena automaticamente
        [10, 20, 30]
        
        >>> validate_dims_list([10, 20, 20, 30])  # Remove duplicatas
        [10, 20, 30]
        
        >>> validate_dims_list([10, 20, 30], d_max=40)  # Adiciona d_max
        [10, 20, 30, 40]
    """
    if not dims:
        raise ValueError("Lista de dimensões não pode ser vazia")
    
    # Validar tipos
    if not all(isinstance(d, int) for d in dims):
        non_ints = [d for d in dims if not isinstance(d, int)]
        raise ValueError(f"Todas as dimensões devem ser inteiros, encontrados: {non_ints}")
    
    # Validar positividade
    if not all(d > 0 for d in dims):
        non_positive = [d for d in dims if d <= 0]
        raise ValueError(f"Todas as dimensões devem ser positivas, encontradas: {non_positive}")
    
    # Remover duplicatas e ordenar
    unique_sorted_dims = sorted(set(dims))
    
    # Se d_max fornecido, garantir que está incluído
    if d_max is not None:
        if d_max not in unique_sorted_dims:
            unique_sorted_dims.append(d_max)
            unique_sorted_dims.sort()
        
        # Validar que nenhuma dimensão excede d_max
        exceeding = [d for d in unique_sorted_dims if d > d_max]
        if exceeding:
            raise ValueError(
                f"Dimensões não podem exceder d_max ({d_max}), "
                f"encontradas: {exceeding}"
            )
    
    return unique_sorted_dims


def compute_d_min_heuristic(d_bin: int) -> int:
    """
    Calcula d_min baseado em heurística informacional.
    
    Heurística: d_min = max(4, round(log2(D_bin) * 2))
    
    Justificativa:
    - log2(D_bin) representa a "entropia informacional" aproximada
    - Multiplicar por 2 dá margem para capturar relações não-lineares
    - Mínimo de 4 evita embeddings triviais demais
    
    Args:
        d_bin: Dimensão total do vetor binário (features + topics)
    
    Returns:
        Dimensão mínima recomendada
    
    Examples:
        >>> compute_d_min_heuristic(99)  # 83 features + 16 topics
        13
        >>> compute_d_min_heuristic(200)
        16
        >>> compute_d_min_heuristic(10)
        7
    """
    import numpy as np
    
    if d_bin <= 0:
        raise ValueError(f"d_bin deve ser positivo, recebido: {d_bin}")
    
    d_min = max(4, round(np.log2(d_bin) * 2))
    return d_min


# CLI para teste standalone
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Gerador de grade de dimensões para sweeps'
    )
    
    parser.add_argument(
        '--d-min',
        type=int,
        default=13,
        help='Dimensão mínima (default: 13)'
    )
    
    parser.add_argument(
        '--d-max',
        type=int,
        default=99,
        help='Dimensão máxima (default: 99)'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        default=5,
        help='Passo entre dimensões (default: 5)'
    )
    
    parser.add_argument(
        '--dims',
        type=int,
        nargs='+',
        help='Lista manual de dimensões (ignora --d-min, --d-max, --step se fornecida)'
    )
    
    parser.add_argument(
        '--d-bin',
        type=int,
        help='Dimensão binária para calcular d_min heurístico (sobrescreve --d-min)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("  GERADOR DE GRADE DE DIMENSÕES")
    print("="*70)
    
    # Se d_bin fornecido, calcular d_min heurístico
    d_min = args.d_min
    if args.d_bin is not None:
        d_min = compute_d_min_heuristic(args.d_bin)
        print(f"\nUsando d_min heurístico baseado em d_bin={args.d_bin}:")
        print(f"  d_min = max(4, round(log2({args.d_bin}) * 2)) = {d_min}")
    
    # Gerar dimensões
    if args.dims:
        print(f"\nModo: Lista manual")
        print(f"  Dimensões fornecidas: {args.dims}")
        dims = build_dims(d_min=d_min, d_max=args.d_max, dims_list=args.dims)
    else:
        print(f"\nModo: Range com step")
        print(f"  d_min: {d_min}")
        print(f"  d_max: {args.d_max}")
        print(f"  step: {args.step}")
        dims = build_dims(d_min=d_min, d_max=args.d_max, step=args.step)
    
    print("\n" + "="*70)
    print("  DIMENSÕES GERADAS")
    print("="*70)
    print(f"\nTotal: {len(dims)} dimensões")
    print(f"Range: {dims[0]} até {dims[-1]}")
    
    # Mostrar todas se poucas, ou amostra se muitas
    if len(dims) <= 20:
        print(f"\nLista completa:")
        for i, d in enumerate(dims, 1):
            print(f"  [{i:2d}] d={d}")
    else:
        print(f"\nPrimeiras 10:")
        for i, d in enumerate(dims[:10], 1):
            print(f"  [{i:2d}] d={d}")
        print(f"  ...")
        print(f"\nÚltimas 10:")
        for i, d in enumerate(dims[-10:], len(dims)-9):
            print(f"  [{i:2d}] d={d}")
    
    # Validações
    print("\n" + "="*70)
    print("  VALIDAÇÕES")
    print("="*70)
    print(f"✓ Todas positivas: {all(d > 0 for d in dims)}")
    print(f"✓ Todas únicas: {len(dims) == len(set(dims))}")
    print(f"✓ Ordenadas: {dims == sorted(dims)}")
    
    if not args.dims:  # Apenas se gerado por range
        print(f"✓ d_max incluído: {args.d_max in dims}")
        print(f"✓ Última dimensão é d_max: {dims[-1] == args.d_max}")
    
    print("\n✓ Operação concluída com sucesso!")
