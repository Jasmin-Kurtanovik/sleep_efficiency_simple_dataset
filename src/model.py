"""
LSTM model definition for time series forecasting.
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM-based neural network model for time series prediction.
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units in LSTM layers
        num_layers (int): Number of LSTM layers
        output_size (int): Number of output features
        dropout (float): Dropout probability for regularization
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
        num_users: int = None,
        user_emb_dim: int = 16
    ):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_users = num_users
        self.user_emb_dim = user_emb_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Temporal attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Layer normalization after attention
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

        # Optional user embedding to condition LSTM hidden state
        if num_users is not None and num_users > 0:
            self.user_emb = nn.Embedding(num_users, user_emb_dim)
            # Project embedding to initial hidden+cell states: 2 * num_layers * hidden_size
            self.user_proj = nn.Linear(user_emb_dim, 2 * num_layers * hidden_size)
        else:
            self.user_emb = None
            self.user_proj = None
    
    def forward(self, x: torch.Tensor, user_idx: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the model with temporal attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            user_idx (torch.Tensor): Optional user indices of shape (batch,)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        # Initialize hidden state from user embedding if provided
        if self.user_emb is not None and user_idx is not None:
            # user_idx: (batch,)
            emb = self.user_emb(user_idx)  # (batch, emb_dim)
            proj = self.user_proj(emb)     # (batch, 2 * num_layers * hidden_size)
            batch = x.size(0)
            # reshape to (2, num_layers, batch, hidden_size)
            proj = proj.view(batch, 2, self.num_layers, self.hidden_size).permute(1,2,0,3)
            h0 = proj[0].contiguous()  # (num_layers, batch, hidden_size)
            c0 = proj[1].contiguous()  # (num_layers, batch, hidden_size)
            lstm_out, _ = self.lstm(x, (h0, c0))
        else:
            lstm_out, _ = self.lstm(x)
        
        # Temporal attention
        # lstm_out: (batch_size, seq_len, hidden_size)
        # Compute attention scores for each timestep
        attention_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Compute weighted sum of LSTM outputs
        context = (lstm_out * attention_weights).sum(dim=1)  # (batch_size, hidden_size)
        
        # Apply layer normalization
        context = self.attention_norm(context)
        
        # Pass through fully connected layers
        output = self.fc(context)
        
        return output
