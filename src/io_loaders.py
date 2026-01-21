"""
Módulo para carregar dados de arquivos CSV/TSV com diferentes formatos.
"""
import pandas as pd
from pathlib import Path
from typing import Optional


def load_tsv(path: str, encoding: str = 'utf-8', sep: str = '\t') -> pd.DataFrame:
    """
    Carrega arquivo TSV com encoding e separador especificados.
    
    Args:
        path: Caminho do arquivo
        encoding: Encoding do arquivo (padrão: utf-8)
        sep: Separador (padrão: \t)
    
    Returns:
        DataFrame carregado
    """
    try:
        df = pd.read_csv(path, sep=sep, encoding=encoding)
        print(f"✓ Carregado {path}: {len(df)} linhas, {len(df.columns)} colunas")
        return df
    except Exception as e:
        print(f"✗ Erro ao carregar {path}: {e}")
        raise


def load_news_tsv_latin1(path: str = 'dataset/news.csv') -> pd.DataFrame:
    """
    Carrega news.csv com encoding latin1 (ISO-8859-1).
    
    Args:
        path: Caminho do arquivo news.csv
    
    Returns:
        DataFrame com as notícias
    """
    df = load_tsv(path, encoding='latin1', sep='\t')
    
    # Remove coluna de índice se existir
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        print(f"  → Removida coluna 'Unnamed: 0'")
    
    return df


def load_features_semicolon(path: str = 'features/join_features_selecionadas.csv') -> pd.DataFrame:
    """
    Carrega join_features_selecionadas.csv com separador ';'.
    
    Args:
        path: Caminho do arquivo de features
    
    Returns:
        DataFrame com as features
    """
    df = pd.read_csv(path, sep=';', encoding='utf-8')
    print(f"✓ Carregado {path}: {len(df)} linhas, {len(df.columns)} colunas")
    return df


def load_all_datasets(base_path: str = '.') -> dict:
    """
    Carrega todos os datasets do projeto.
    
    Args:
        base_path: Diretório base do projeto
    
    Returns:
        Dicionário com todos os DataFrames carregados
    """
    base_path = Path(base_path)
    dataset_path = base_path / 'dataset'
    features_path = base_path / 'features'
    
    print("\n" + "="*70)
    print("CARREGANDO DATASETS")
    print("="*70 + "\n")
    
    datasets = {}
    
    # Carregar arquivos TSV da pasta dataset
    datasets['users'] = load_tsv(dataset_path / 'users.csv')
    datasets['ratings'] = load_tsv(dataset_path / 'ratings.csv')
    datasets['news_ratings'] = load_tsv(dataset_path / 'news_ratings.csv')
    datasets['news'] = load_news_tsv_latin1(dataset_path / 'news.csv')
    datasets['recLists'] = load_tsv(dataset_path / 'recLists.csv')
    datasets['clicks'] = load_tsv(dataset_path / 'clicks.csv')
    datasets['news_clicks'] = load_tsv(dataset_path / 'news_clicks.csv')
    
    # Carregar features
    datasets['features'] = load_features_semicolon(features_path / 'join_features_selecionadas.csv')
    
    print("\n" + "="*70)
    print("CARREGAMENTO CONCLUÍDO")
    print("="*70 + "\n")
    
    return datasets


if __name__ == '__main__':
    # Teste básico dos loaders
    datasets = load_all_datasets()
    
    print("\nResumo dos datasets:")
    for name, df in datasets.items():
        print(f"{name:20s}: {len(df):6d} linhas, {len(df.columns):3d} colunas")
