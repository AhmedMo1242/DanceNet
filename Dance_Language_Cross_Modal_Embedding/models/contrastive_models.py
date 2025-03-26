import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block.
    
    Implements a simple multi-head attention mechanism that can be used
    to model complex dependencies in embeddings.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to include bias in QKV projection
        attn_drop: Dropout rate for attention weights
    """
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)
        
    def forward(self, x):
        """
        Forward pass for attention block.
        
        Args:
            x: Input tensor of shape [batch_size, embedding_dim]
            
        Returns:
            Attention-weighted output of same shape as input
        """
        B, N = x.shape
        # Reshape for multi-head attention
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)  # [3, B, heads, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, heads, head_dim]

        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, head_dim, head_dim]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Output projection
        x = (attn @ v)  # [B, heads, head_dim]
        x = x.transpose(1, 2).reshape(B, N)  # [B, N]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization.
    
    Implements a residual connection with two linear layers and batch normalization.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        dropout: Dropout probability
    """
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.downsample = None
        if in_dim != out_dim:
            self.downsample = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
    
    def forward(self, x):
        """
        Forward pass for residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after applying residual connections
        """
        identity = x
        
        # First block
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        # Second block
        out = self.linear2(out)
        out = self.bn2(out)
        
        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = F.gelu(out)
        return out

class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    
    Implements a multi-layer perceptron that projects embeddings to a space
    suitable for contrastive learning. The projection head follows the design
    principles from SimCLR and related contrastive learning approaches.
    
    Args:
        input_dim: Dimension of input embeddings
        hidden_dim: Dimension of hidden layer
        output_dim: Dimension of output embeddings
        dropout: Dropout probability
    """
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=256, dropout=0.0):
        super(ProjectionHead, self).__init__()
        
        # Simple 3-layer MLP with ReLU activations
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Initialize weights to small values
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using uniform distribution with small values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use simple uniform initialization with small values
                nn.init.uniform_(m.weight, -0.05, 0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass for projection head.
        
        Args:
            x: Input embeddings
            
        Returns:
            Projected embeddings
        """
        return self.projection(x)

class AdaptiveTemperature(nn.Module):
    """
    Learnable temperature parameter for contrastive loss.
    
    Implements a temperature parameter that can be optimized during training
    to control the sharpness of the probability distribution in contrastive loss.
    
    Args:
        init_temp: Initial temperature value
        min_temp: Minimum allowed temperature
        max_temp: Maximum allowed temperature
    """
    def __init__(self, init_temp=0.07, min_temp=0.03, max_temp=0.2):
        super(AdaptiveTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp))
        self.min_temp = min_temp
        self.max_temp = max_temp
    
    def forward(self):
        """
        Get the current temperature value, clamped to allowed range.
        
        Returns:
            Current temperature parameter value
        """
        # Clamp temperature to prevent extreme values
        return torch.clamp(self.temperature, self.min_temp, self.max_temp)

class ClusterContrastiveLoss(nn.Module):
    """
    Contrastive loss with cluster-based positive pairs and stability measures.
    
    Implements a contrastive loss that considers samples from the same cluster
    as positive pairs and samples from different clusters as negative pairs.
    Includes numerical stability measures to prevent common training issues.
    
    Args:
        temperature: Temperature parameter for loss scaling
        margin: Margin for triplet-style contrastive loss
        class_weights: Optional dictionary mapping class IDs to weights for handling imbalanced data
    """
    def __init__(self, temperature=0.5, margin=1.0, class_weights=None):
        super(ClusterContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.class_weights = class_weights
    
    def forward(self, features, labels):
        """
        Compute contrastive loss for given features and labels.
        
        Args:
            features: Embedding features of shape [batch_size, feature_dim]
            labels: Cluster/class labels of shape [batch_size]
            
        Returns:
            Tuple of (loss, temperature) values
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Basic sanity check
        if torch.isnan(features).any():
            print("WARNING: NaN detected in features input!")
            features = torch.nan_to_num(features, nan=0.0)
        
        # Simple L2 normalization with safe epsilon
        features_norm = torch.norm(features, p=2, dim=1, keepdim=True)
        features = features / (features_norm + 1e-5)
        
        # Compute pairwise distances instead of similarities
        # This is more stable numerically than dot products
        dist_matrix = torch.cdist(features, features, p=2)  # Euclidean distance
        
        # Create mask for positive and negative pairs
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-comparisons
        identity_mask = torch.eye(batch_size, device=device)
        pos_mask = pos_mask * (1 - identity_mask)
        
        # Basic triplet loss formulation
        # For each anchor, find the hardest positive and negative
        pos_dist = dist_matrix * pos_mask
        pos_dist[pos_mask == 0] = 0
        
        # If an anchor has no positives, assign it a zero loss
        hardest_pos_dist = pos_dist.max(dim=1)[0]
        has_pos = (pos_mask.sum(dim=1) > 0).float()
        
        # For negatives, use all examples from different classes
        neg_mask = 1.0 - pos_mask - identity_mask
        neg_dist = dist_matrix * neg_mask
        neg_dist[neg_mask == 0] = 1e6  # Large value for non-negatives
        hardest_neg_dist = neg_dist.min(dim=1)[0]
        
        # Basic triplet loss with margin
        # We want positives to be close and negatives to be far
        # So we want: pos_dist - neg_dist < -margin
        # Or: neg_dist - pos_dist > margin
        basic_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            class_weights = torch.tensor([self.class_weights.get(label.item(), 1.0) 
                                         for label in labels.view(-1)], device=device)
            basic_loss = basic_loss * class_weights
        
        # Only include anchors that have positive examples
        final_loss = (basic_loss * has_pos).sum() / (has_pos.sum() + 1e-6)
        
        # Check for NaN
        if torch.isnan(final_loss):
            print("WARNING: NaN detected in loss! Using default loss value.")
            return torch.tensor(0.1, device=device, requires_grad=True), self.temperature
        
        return final_loss, self.temperature
