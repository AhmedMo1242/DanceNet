import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
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
    def __init__(self, n_joints, n_dims, hidden_dim, latent_dim, n_layers=2, dropout=0.3):
        super(LightweightEncoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_joints = n_joints
        self.n_dims = n_dims
        
        # Calculate input size (flattened joints and dimensions)
        self.input_dim = n_joints * n_dims
        
        # Simplified input projection
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Simplified LSTM encoder - unidirectional to reduce complexity
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False  # Changed to unidirectional
        )
        
        # Simplified temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Simplified MLP for latent space projection
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Latent space projections
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        batch_size, seq_length, n_joints, n_dims = x.size()
        
        # Reshape to (batch_size, seq_length, n_joints * n_dims)
        x = x.reshape(batch_size, seq_length, n_joints * n_dims)
        
        # Apply input projection
        x = self.input_projection(x)
        
        # LSTM encoding
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Apply temporal attention
        temporal_weights = self.temporal_attention(lstm_out)
        context_vector = torch.sum(lstm_out * temporal_weights, dim=1)
        
        # Combine context vector with final hidden state
        combined = context_vector + hidden[-1]
        
        # Apply MLP
        hidden_mlp = self.mlp(combined)
        
        # Project to latent space parameters
        mu = self.mu_proj(hidden_mlp)
        logvar = self.logvar_proj(hidden_mlp)
        
        return mu, logvar

class EfficientDecoderGRU(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_joints, n_dims, n_layers=2, dropout=0.3, use_text=False):
        super(EfficientDecoderGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_joints = n_joints
        self.n_dims = n_dims
        self.use_text = use_text
        
        # Input dimension for latent projection
        input_dim = latent_dim
        if use_text:
            # If using text, augment input with text embedding
            # Assuming text embedding dimension is also latent_dim for simplicity
            input_dim = latent_dim * 2
        
        # Simplified latent vector projection
        self.latent_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Single GRU decoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Simplified output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_joints * n_dims)
        )
    
    def forward(self, z, seq_length, text_embedding=None):
        batch_size = z.size(0)
        
        # Combine latent vector with text embedding if available
        if self.use_text and text_embedding is not None:
            z = torch.cat([z, text_embedding], dim=1)
        
        # Project latent vector to hidden dimension
        hidden = self.latent_projection(z)
        
        # Initialize GRU hidden state
        h = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)
        
        # Initial decoder input sequence
        decoder_input = hidden.unsqueeze(1).repeat(1, seq_length, 1)
        
        # GRU decoding
        gru_out, _ = self.gru(decoder_input, h)
        
        # Project to output space and reshape
        output = self.output_projection(gru_out)
        output = output.reshape(batch_size, seq_length, self.n_joints, self.n_dims)
        
        return output

class OptimizedDanceVAE(nn.Module):
    def __init__(self, n_joints, n_dims, hidden_dim=384, latent_dim=256, n_layers=2, dropout=0.3, use_text=False):
        """
        Optimized VAE for dance motion sequences with reduced complexity
        Args:
            n_joints: Number of joints in the motion data
            n_dims: Number of dimensions per joint (typically 3 for x,y,z)
            hidden_dim: Hidden dimension size (reduced from 768 to 384)
            latent_dim: Latent dimension size (reduced from 384 to 256)
            n_layers: Number of layers in RNNs (reduced from 4 to 2)
            dropout: Dropout rate
            use_text: Whether to use text embeddings
        """
        super(OptimizedDanceVAE, self).__init__()
        self.n_joints = n_joints
        self.n_dims = n_dims
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_text = use_text
        
        # Optimized encoder and decoder
        self.encoder = LightweightEncoderLSTM(n_joints, n_dims, hidden_dim, latent_dim, n_layers, dropout)
        self.decoder = EfficientDecoderGRU(latent_dim, hidden_dim, n_joints, n_dims, n_layers, dropout, use_text)
    
    def encode(self, x):
        """
        Encode input sequence to latent distribution parameters
        """
        return self.encoder(x)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_length, text_embedding=None):
        """
        Decode latent vector into sequence
        """
        return self.decoder(z, seq_length, text_embedding)
    
    def forward(self, x, text_embedding=None):
        """
        Forward pass
        """
        batch_size, seq_length, n_joints, n_dims = x.shape
        
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decode(z, seq_length, text_embedding)
        
        return reconstructed, mu, logvar
    
    def get_embedding(self, x):
        """
        Get the embedding vector for a sequence
        """
        mu, _ = self.encode(x)
        return mu

# Loss functions
def vae_loss(reconstructed, x, mu, logvar, beta=1.0):
    """
    Compute VAE loss: reconstruction loss + beta * KL divergence
    Args:
        reconstructed: Reconstructed motion sequence
        x: Original motion sequence
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

def motion_loss(reconstructed, x, alpha=1.0):
    """
    Compute motion-specific loss with additional terms for motion quality
    Args:
        reconstructed: Reconstructed motion sequence
        x: Original motion sequence
        alpha: Weight for velocity term
    """
    # Base reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
    
    # Velocity loss (temporal smoothness)
    vel_orig = x[:, 1:] - x[:, :-1]
    vel_recon = reconstructed[:, 1:] - reconstructed[:, :-1]
    vel_loss = F.mse_loss(vel_recon, vel_orig, reduction='mean')
    
    # Total motion loss
    total_loss = recon_loss + alpha * vel_loss
    
    return total_loss, recon_loss, vel_loss
