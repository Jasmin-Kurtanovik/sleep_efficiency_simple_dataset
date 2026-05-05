"""Transformer model definition for time series forecasting."""

import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """
    Transformer-based neural network model for time series prediction.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Transformer embedding dimension.
        num_layers (int): Number of Transformer encoder layers.
        output_size (int): Number of output features.
        dropout (float): Dropout probability for regularization.
        num_users (int): Number of users for optional user embeddings.
        user_emb_dim (int): Dimension of user embeddings.
        num_heads (int): Number of attention heads.
        ff_dim (int): Feed-forward dimension inside the encoder.
        max_seq_len (int): Maximum sequence length supported by positional embeddings.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
        num_users: int = None,
        user_emb_dim: int = 16,
        num_heads: int = 4,
        ff_dim: int = 128,
        max_seq_len: int = 32,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for the Transformer model")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_users = num_users
        self.user_emb_dim = user_emb_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

        self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.sequence_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

        if num_users is not None and num_users > 0:
            self.user_emb = nn.Embedding(num_users, user_emb_dim)
            self.user_proj = nn.Linear(user_emb_dim, hidden_size)
        else:
            self.user_emb = None
            self.user_proj = None

    def forward(self, x: torch.Tensor, user_idx: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the Transformer model."""
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds the configured max_seq_len {self.max_seq_len}"
            )

        x = self.input_projection(x)
        x = x + self.positional_embedding[:, :seq_len, :]

        if self.user_emb is not None and user_idx is not None:
            user_context = self.user_proj(self.user_emb(user_idx)).unsqueeze(1)
            x = x + user_context

        encoded = self.transformer(x)
        context = encoded.mean(dim=1)
        context = self.sequence_norm(context)

        return self.fc(context)