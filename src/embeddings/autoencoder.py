"""
Autoencoder para compressão de vetores binários multi-label.

Implementa um autoencoder com PyTorch para gerar embeddings densos
a partir de representações binárias (features ou tópicos).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import normalize


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
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
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
        
        # Dropout para denoising (aplicado na entrada durante treino)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
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
        # Aplicar denoising dropout apenas durante treino
        if self.training and self.dropout is not None:
            x_noisy = self.dropout(x)
        else:
            x_noisy = x
        
        z = self.encode(x_noisy)
        x_reconstructed = self.decode(z)
        return z, x_reconstructed


def train_autoencoder(
    X: np.ndarray,
    embedding_dim: int,
    hidden_dim: Optional[int] = None,
    dropout_rate: float = 0.1,
    pos_weight: Optional[float] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 42,
    verbose: bool = True
) -> BinaryAutoencoder:
    """
    Treina um autoencoder para vetores binários.
    
    Args:
        X: Matriz numpy com shape (n_samples, n_features) com valores binários
        embedding_dim: Dimensão dos embeddings (bottleneck)
        hidden_dim: Dimensão da camada oculta (None = calculado automaticamente)
        dropout_rate: Taxa de dropout para denoising
        pos_weight: Peso para classe positiva na loss (None = calculado automaticamente)
        epochs: Número de épocas de treinamento
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        seed: Seed para reprodutibilidade
        verbose: Se True, imprime progresso
    
    Returns:
        Modelo BinaryAutoencoder treinado
    """
    # Configurar seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Calcular pos_weight automaticamente se não fornecido
    if pos_weight is None:
        n_zeros = np.sum(X == 0)
        n_ones = np.sum(X == 1)
        if n_ones > 0:
            pos_weight = n_zeros / n_ones
        else:
            pos_weight = 1.0
    
    if verbose:
        print(f"Treinando autoencoder:")
        print(f"  - Input dim: {X.shape[1]}")
        print(f"  - Embedding dim: {embedding_dim}")
        print(f"  - Hidden dim: {hidden_dim or (X.shape[1] + embedding_dim) // 2}")
        print(f"  - Dropout rate: {dropout_rate}")
        print(f"  - Pos weight: {pos_weight:.2f}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {learning_rate}")
    
    # Criar modelo
    input_dim = X.shape[1]
    model = BinaryAutoencoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    )
    
    # Configurar loss com pos_weight
    pos_weight_tensor = torch.tensor([pos_weight] * input_dim, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Converter dados para tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Loop de treinamento
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        
        for (batch_X,) in dataloader:
            optimizer.zero_grad()
            
            # Forward
            _, reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_X)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Época {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")
    
    if verbose:
        print(f"Treinamento concluído. Loss final: {avg_loss:.4f}")
    
    return model


def extract_embeddings(
    model: BinaryAutoencoder,
    X: np.ndarray,
    normalize_l2: bool = True
) -> np.ndarray:
    """
    Extrai embeddings de um autoencoder treinado.
    
    Args:
        model: Modelo BinaryAutoencoder treinado
        X: Matriz numpy com shape (n_samples, n_features)
        normalize_l2: Se True, aplica normalização L2 nos embeddings
    
    Returns:
        Embeddings com shape (n_samples, embedding_dim)
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        embeddings = model.encode(X_tensor)
        embeddings_np = embeddings.numpy()
    
    if normalize_l2:
        embeddings_np = normalize(embeddings_np, norm='l2', axis=1)
    
    return embeddings_np
