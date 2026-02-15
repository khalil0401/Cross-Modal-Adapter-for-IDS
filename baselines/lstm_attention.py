"""
LSTM with Attention Baseline

Bidirectional LSTM with attention mechanism for time-series classification.

Reference: Paper baselines (Table II)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention mechanism for LSTM hidden states.
    """
    
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        """
        Args:
            lstm_output (torch.Tensor): LSTM outputs (batch, seq_len, hidden_dim)
        
        Returns:
            torch.Tensor: Context vector (batch, hidden_dim)
        """
        # Compute attention scores
        scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(scores, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_dim)
        
        return context


class LSTMAttention(nn.Module):
    """
    Bidirectional LSTM with attention for time-series classification.
    
    From paper baselines:
    "LSTM with attention mechanism provides temporal modeling
    and achieves competitive results on sequential data."
    """
    
    def __init__(self, input_channels, num_classes=1, task='binary',
                 hidden_dim=128, num_layers=2, dropout=0.3):
        """
        Args:
            input_channels (int): Number of input features
            num_classes (int): Number of output classes
            task (str): 'binary' or 'multiclass'
            hidden_dim (int): LSTM hidden dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
        """
        super(LSTMAttention, self).__init__()
        
        self.task = task
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = Attention(hidden_dim * 2)  # *2 for bidirectional
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.fc = nn.Linear(hidden_dim * 2, num_classes if task == 'multiclass' else 1)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input time-series (batch, channels, time_steps)
        
        Returns:
            torch.Tensor: Logits
        """
        # Transpose for LSTM: (batch, time_steps, channels)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, time_steps, hidden_dim*2)
        
        # Attention
        context = self.attention(lstm_out)  # (batch, hidden_dim*2)
        
        # Dropout
        context = self.dropout(context)
        
        # Classification
        out = self.fc(context)
        
        return out


if __name__ == "__main__":
    # Test LSTM with Attention
    batch_size = 4
    input_channels = 10
    time_steps = 288
    
    model = LSTMAttention(input_channels, num_classes=1, task='binary')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LSTM-Attention parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(batch_size, input_channels, time_steps)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output: ({batch_size}, 1)")
