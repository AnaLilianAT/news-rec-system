"""
Autoencoder para compressão de vetores binários multi-label.

Implementa um autoencoder com PyTorch para gerar embeddings densos
a partir de representações binárias (features ou tópicos).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from typing import Tuple, Optional, List, Dict, Any
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


class BinaryAutoencoder(nn.Module):
    """
    Autoencoder para vetores binários esparsos (multi-label).
    
    Arquitetura:
    - Encoder: input_dim -> hidden_dim -> embedding_dim (bottleneck)
    - Decoder: embedding_dim -> hidden_dim -> input_dim
    - Loss: BCEWithLogitsLoss (apropriada para multi-label)
    - Opção de denoising via dropout na entrada
    
    Args:
        input_dim: Dimensão do vetor de entrada (número de features binárias)
        embedding_dim: Dimensão do bottleneck (embeddings)
        hidden_dim: Dimensão da camada oculta (padrão: média entre input e embedding)
        dropout_rate: Taxa de dropout para denoising (padrão: 0.0 = sem denoising)
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
        dropout_rate: float = 0.0
    ):
        super(BinaryAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim or (input_dim + embedding_dim) // 2
        self.dropout_rate = dropout_rate
        
        # Encoder (com dropout entre camadas)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, embedding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, input_dim)
            # Nota: sem sigmoid aqui pois BCEWithLogitsLoss já aplica sigmoid internamente
        )
        
        # Dropout removido daqui - denoising será aplicado manualmente no treino
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai embeddings do encoder.
        
        Args:
            x: Tensor com shape (batch_size, input_dim)
        
        Returns:
            Embeddings com shape (batch_size, embedding_dim)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstrói entrada a partir de embeddings.
        
        Args:
            z: Embeddings com shape (batch_size, embedding_dim)
        
        Returns:
            Logits reconstruídos com shape (batch_size, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass completo: encoding + decoding.
        
        Args:
            x: Tensor com shape (batch_size, input_dim)
        
        Returns:
            Tupla (embeddings, reconstruction_logits)
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return z, x_reconstructed


def train_autoencoder(
    X: np.ndarray,
    embedding_dim: int,
    hidden_dim: Optional[int] = None,
    dropout_rate: float = 0.1,
    pos_weight_mode: str = 'auto',
    pos_weight_clip: Optional[List[float]] = None,
    denoising_prob: float = 0.0,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 10,
    val_split: float = 0.2,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[BinaryAutoencoder, Dict[str, Any]]:
    """
    Treina um autoencoder para vetores binários esparsos.
    
    Args:
        X: Matriz numpy com shape (n_samples, n_features) com valores binários
        embedding_dim: Dimensão dos embeddings (bottleneck)
        hidden_dim: Dimensão da camada oculta (None = calculado automaticamente)
        dropout_rate: Taxa de dropout no encoder
        pos_weight_mode: 'auto' (neg/pos por feature), 'sqrt' (sqrt do auto), ou float
        pos_weight_clip: [min, max] para clipping de pos_weight (None = sem clip)
        denoising_prob: Probabilidade de zerar 1s durante treino (0.0 = desabilitado)
        epochs: Número de épocas de treinamento
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        weight_decay: Weight decay para regularização L2
        early_stopping_patience: Paciência para early stopping (0 = desabilitado)
        val_split: Fração dos dados para validação (0.0-1.0)
        seed: Seed para reprodutibilidade
        verbose: Se True, imprime progresso
    
    Returns:
        Tupla (modelo_treinado, metadata_treino)
    """
    # Configurar seed para reprodutibilidade completa
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Split treino/validação
    if val_split > 0:
        X_train, X_val = train_test_split(
            X, test_size=val_split, random_state=seed, shuffle=True
        )
    else:
        X_train = X
        X_val = None
    
    # Calcular pos_weight por feature (baseado nos dados de treino)
    eps = 1e-8
    pos = X_train.sum(axis=0)  # Contagem de 1s por feature
    neg = len(X_train) - pos    # Contagem de 0s por feature
    
    if pos_weight_mode == 'auto':
        # pos_weight = neg / (pos + eps)
        pos_weight_per_feature = neg / (pos + eps)
    elif pos_weight_mode == 'sqrt':
        # Modo mais suave: raiz quadrada do auto
        pos_weight_per_feature = np.sqrt(neg / (pos + eps))
    else:
        # Valor fixo fornecido
        try:
            fixed_weight = float(pos_weight_mode)
            pos_weight_per_feature = np.full(X_train.shape[1], fixed_weight)
        except (ValueError, TypeError):
            raise ValueError(
                f"pos_weight_mode inválido: '{pos_weight_mode}'. "
                f"Use 'auto', 'sqrt' ou um valor float."
            )
    
    # Aplicar clipping se especificado
    if pos_weight_clip is not None:
        min_weight, max_weight = pos_weight_clip
        pos_weight_per_feature = np.clip(pos_weight_per_feature, min_weight, max_weight)
    
    # Estatísticas de pos_weight
    pw_mean = pos_weight_per_feature.mean()
    pw_median = np.median(pos_weight_per_feature)
    pw_min = pos_weight_per_feature.min()
    pw_max = pos_weight_per_feature.max()
    
    if verbose:
        print(f"Treinando autoencoder:")
        print(f"  - Input dim: {X.shape[1]}")
        print(f"  - Samples: {len(X)} (treino: {len(X_train)}, val: {len(X_val) if X_val is not None else 0})")
        print(f"  - Embedding dim: {embedding_dim}")
        print(f"  - Hidden dim: {hidden_dim or (X.shape[1] + embedding_dim) // 2}")
        print(f"  - Dropout rate: {dropout_rate}")
        print(f"  - Denoising prob: {denoising_prob}")
        print(f"  - Pos weight mode: {pos_weight_mode}")
        print(f"    * Mean: {pw_mean:.2f}, Median: {pw_median:.2f}")
        print(f"    * Min: {pw_min:.2f}, Max: {pw_max:.2f}")
        if pos_weight_clip:
            print(f"    * Clip: {pos_weight_clip}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Early stopping patience: {early_stopping_patience if early_stopping_patience > 0 else 'Desabilitado'}")
    
    # Criar modelo
    input_dim = X.shape[1]
    model = BinaryAutoencoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    )
    
    # Configurar loss com pos_weight por feature
    pos_weight_tensor = torch.tensor(pos_weight_per_feature, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # Optimizer com weight_decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Converter dados para tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Dataloader de validação (se houver)
    val_loader = None
    if X_val is not None:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    
    # Early stopping setup
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    
    # Loop de treinamento
    for epoch in range(epochs):
        # === TREINO ===
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0
        
        for (batch_X,) in train_loader:
            # Aplicar denoising: zerar aleatoriamente uma fração dos 1s
            if denoising_prob > 0:
                batch_X_noisy = batch_X.clone()
                # Criar máscara: onde está 1 E rand < denoise_p, vira 0
                noise_mask = (batch_X == 1) & (torch.rand_like(batch_X) < denoising_prob)
                batch_X_noisy[noise_mask] = 0
            else:
                batch_X_noisy = batch_X
            
            optimizer.zero_grad()
            
            # Forward (entrada ruidosa, alvo original)
            _, reconstructed = model(batch_X_noisy)
            loss = criterion(reconstructed, batch_X)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            n_train_batches += 1
        
        avg_train_loss = total_train_loss / n_train_batches
        train_losses.append(avg_train_loss)
        
        # === VALIDAÇÃO ===
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for (batch_X,) in val_loader:
                    _, reconstructed = model(batch_X)
                    loss = criterion(reconstructed, batch_X)
                    total_val_loss += loss.item()
                    n_val_batches += 1
            
            avg_val_loss = total_val_loss / n_val_batches
            val_losses.append(avg_val_loss)
            
            # Early stopping check
            if early_stopping_patience > 0:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    
                    if epochs_without_improvement >= early_stopping_patience:
                        if verbose:
                            print(f"\n  Early stopping na época {epoch + 1}/{epochs}")
                            print(f"  Melhor val_loss: {best_val_loss:.4f} (época {epoch + 1 - early_stopping_patience})")
                        break
        
        # Log de progresso
        if verbose and (epoch + 1) % 10 == 0:
            if avg_val_loss is not None:
                print(f"  Época {epoch + 1}/{epochs}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}")
            else:
                print(f"  Época {epoch + 1}/{epochs}: train_loss = {avg_train_loss:.4f}")
    
    # Restaurar melhor modelo se early stopping foi usado
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\nModelo restaurado com melhor val_loss: {best_val_loss:.4f}")
    elif verbose:
        print(f"\nTreinamento concluído. Loss final: {avg_train_loss:.4f}")
    
    # Metadata do treinamento
    metadata = {
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'best_val_loss': best_val_loss if best_model_state is not None else None,
        'total_epochs': len(train_losses),
        'early_stopped': best_model_state is not None and len(train_losses) < epochs,
        'pos_weight_stats': {
            'mean': float(pw_mean),
            'median': float(pw_median),
            'min': float(pw_min),
            'max': float(pw_max),
            'mode': pos_weight_mode,
            'clip': pos_weight_clip
        }
    }
    
    return model, metadata


def extract_embeddings(
    model: BinaryAutoencoder,
    X: np.ndarray,
    normalize_l2: bool = True
) -> np.ndarray:
    """
    Extrai embeddings de um autoencoder treinado.
    
    IMPORTANTE: Sempre usa model.eval() para desabilitar dropout e garantir
    inferência determinística (sem ruído de denoising).
    
    Args:
        model: Modelo BinaryAutoencoder treinado
        X: Matriz numpy com shape (n_samples, n_features) com features BINÁRIAS
        normalize_l2: Se True, aplica normalização L2 nos embeddings (linha a linha)
    
    Returns:
        Embeddings com shape (n_samples, embedding_dim)
    """
    # Garantir modo de inferência (desabilita dropout)
    model.eval()
    
    # Inferência sem gradientes
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        embeddings = model.encode(X_tensor)
        embeddings_np = embeddings.numpy()
    
    # L2 normalization por linha (preparar para cosine similarity)
    if normalize_l2:
        eps = 1e-8
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        embeddings_np = embeddings_np / (norms + eps)
    
    return embeddings_np


def extract_embeddings_with_continuous(
    model: BinaryAutoencoder,
    X_binary: np.ndarray,
    X_continuous: Optional[np.ndarray] = None,
    normalize_l2: bool = True,
    concat_continuous_after: bool = True
) -> np.ndarray:
    """
    Extrai embeddings e opcionalmente concatena features contínuas.
    
    Fluxo:
    1. Treinar AE apenas com features binárias (X_binary)
    2. Extrair embeddings: z = encoder(X_binary)
    3. L2 normalize z (se normalize_l2=True)
    4. Concatenar X_continuous ao final de z (se fornecido e concat_continuous_after=True)
    
    Isso permite usar features contínuas (polaridade, subjetividade) sem
    misturá-las no treino binário do AE.
    
    Args:
        model: Modelo BinaryAutoencoder treinado com X_binary
        X_binary: Matriz numpy (n_samples, n_binary_features) com features binárias
        X_continuous: Matriz numpy (n_samples, n_continuous_features) ou None
        normalize_l2: Se True, normaliza embeddings antes de concatenar contínuas
        concat_continuous_after: Se True e X_continuous fornecido, concatena
    
    Returns:
        Embeddings com shape:
        - (n_samples, embedding_dim) se X_continuous=None
        - (n_samples, embedding_dim + n_continuous_features) se concatenado
    """
    # Extrair embeddings das features binárias
    Z = extract_embeddings(
        model=model,
        X=X_binary,
        normalize_l2=normalize_l2
    )
    
    # Concatenar features contínuas se fornecidas
    if X_continuous is not None and concat_continuous_after:
        # Verificar alinhamento
        if X_continuous.shape[0] != Z.shape[0]:
            raise ValueError(
                f"X_continuous tem {X_continuous.shape[0]} amostras, "
                f"mas Z tem {Z.shape[0]} amostras"
            )
        
        # Concatenar ao final
        Z_final = np.concatenate([Z, X_continuous], axis=1)
        return Z_final
    
    return Z
