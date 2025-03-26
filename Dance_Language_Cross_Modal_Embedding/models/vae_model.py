import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual block with layer normalization and dropout.
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
    def forward(self, x):
        return x + self.block(x)

class LightweightEncoderLSTM(nn.Module):
    """
    Lightweight LSTM encoder with temporal attention for VAE.
    """
    def __init__(self, n_joints, n_dims, hidden_dim, latent_dim, n_layers=2, dropout=0.3):
        super(LightweightEncoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_joints = n_joints
        self.n_dims = n_dims
        
        self.input_dim = n_joints * n_dims
        
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False
        )
        
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        batch_size, seq_length, n_joints, n_dims = x.size()
        
        x = x.reshape(batch_size, seq_length, n_joints * n_dims)
        
        x = self.input_projection(x)
        
        lstm_out, (hidden, _) = self.lstm(x)
        
        temporal_weights = self.temporal_attention(lstm_out)
        context_vector = torch.sum(lstm_out * temporal_weights, dim=1)
        
        combined = context_vector + hidden[-1]
        
        hidden_mlp = self.mlp(combined)
        
        mu = self.mu_proj(hidden_mlp)
        logvar = self.logvar_proj(hidden_mlp)
        
        return mu, logvar

class EfficientDecoderGRU(nn.Module):
    """
    Efficient GRU decoder for reconstructing motion sequences from latent vectors.
    """
    def __init__(self, latent_dim, hidden_dim, n_joints, n_dims, n_layers=2, dropout=0.3):
        super(EfficientDecoderGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_joints = n_joints
        self.n_dims = n_dims
        
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_joints * n_dims)
        )
    
    def forward(self, z, seq_length):
        batch_size = z.size(0)
        
        hidden = self.latent_projection(z)
        
        h = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)
        
        decoder_input = hidden.unsqueeze(1).repeat(1, seq_length, 1)
        
        gru_out, _ = self.gru(decoder_input, h)
        
        output = self.output_projection(gru_out)
        output = output.reshape(batch_size, seq_length, self.n_joints, self.n_dims)
        
        return output

class DanceVAE(nn.Module):
    """
    Variational Autoencoder for dance motion sequences.
    
    Uses LSTM encoder with temporal attention and GRU decoder.
    """
    def __init__(self, n_joints, n_dims, hidden_dim=384, latent_dim=256, n_layers=2, dropout=0.3):
        super(DanceVAE, self).__init__()
        self.n_joints = n_joints
        self.n_dims = n_dims
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = LightweightEncoderLSTM(n_joints, n_dims, hidden_dim, latent_dim, n_layers, dropout)
        self.decoder = EfficientDecoderGRU(latent_dim, hidden_dim, n_joints, n_dims, n_layers, dropout)
    
    def encode(self, x):
        """Encode input sequence to latent distribution parameters"""
        return self.encoder(x)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_length):
        """Decode latent vector to motion sequence"""
        return self.decoder(z, seq_length)
    
    def forward(self, x):
        """Forward pass through the full VAE"""
        batch_size, seq_length, n_joints, n_dims = x.shape
        
        mu, logvar = self.encode(x)
        
        z = self.reparameterize(mu, logvar)
        
        reconstructed = self.decode(z, seq_length)
        
        return reconstructed, mu, logvar
    
    def get_embedding(self, x):
        """Extract latent embedding for a motion sequence"""
        mu, _ = self.encode(x)
        return mu
