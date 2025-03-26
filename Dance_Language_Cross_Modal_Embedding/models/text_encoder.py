import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    """
    TextEncoder model that leverages a pre-trained BERT model to encode text into fixed-size embeddings.
    
    The model takes text input, processes it through BERT, and reduces dimensions to create
    a compact representation suitable for similarity-based applications.
    """
    
    def __init__(self, bert_model_name="bert-base-uncased", freeze_bert=True, output_dim=256):
        """
        Initialize the TextEncoder model.
        
        Args:
            bert_model_name (str): Name of the pre-trained BERT model to use
            freeze_bert (bool): Whether to freeze BERT parameters during training
            output_dim (int): Dimension of the output embeddings
        """
        super(TextEncoder, self).__init__()
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT parameters if needed
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Dimension reduction model (768 -> output_dim)
        self.dimension_reducer = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, text_list):
        """
        Forward pass of the TextEncoder model.
        
        Args:
            text_list (list): List of text strings to encode
            
        Returns:
            torch.Tensor: Encoded text embeddings with shape [batch_size, output_dim]
        """
        # Tokenize the input texts
        encoded_inputs = self.tokenizer(text_list, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=128)
        
        # Move to the same device as the model
        input_ids = encoded_inputs['input_ids'].to(next(self.parameters()).device)
        attention_mask = encoded_inputs['attention_mask'].to(next(self.parameters()).device)
        
        # Get BERT embeddings
        with torch.no_grad() if not self.bert.training else torch.enable_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            
        # Use the [CLS] token embedding as sentence representation
        bert_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, 768]
        
        # Reduce dimensions to output_dim
        reduced_embeddings = self.dimension_reducer(bert_embeddings)  # Shape: [batch_size, output_dim]
        
        return reduced_embeddings

def cosine_similarity_loss(predicted, target, margin=0.0):
    """
    Calculate cosine similarity loss between predicted and target embeddings.
    
    Args:
        predicted (torch.Tensor): Predicted embeddings
        target (torch.Tensor): Target embeddings
        margin (float): Margin for the loss calculation
        
    Returns:
        torch.Tensor: Computed loss
    """
    # Normalize both vectors
    predicted_norm = F.normalize(predicted, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    
    # Compute cosine similarity (1 means identical, -1 means opposite)
    cosine_sim = torch.sum(predicted_norm * target_norm, dim=1)
    
    # Convert to a loss (1 - similarity) so that 0 is the minimum loss
    loss = torch.clamp(1 - cosine_sim, min=0.0)
    
    return loss.mean()

def mean_squared_error_loss(predicted, target):
    """
    Calculate mean squared error loss between predicted and target embeddings.
    
    Args:
        predicted (torch.Tensor): Predicted embeddings
        target (torch.Tensor): Target embeddings
        
    Returns:
        torch.Tensor: Computed loss
    """
    return F.mse_loss(predicted, target)

def combined_loss(predicted, target, alpha=0.5):
    """
    Calculate combined loss using cosine similarity and mean squared error.
    
    Args:
        predicted (torch.Tensor): Predicted embeddings
        target (torch.Tensor): Target embeddings
        alpha (float): Weight for cosine similarity loss (1-alpha for MSE)
        
    Returns:
        torch.Tensor: Combined loss
    """
    cosine_loss = cosine_similarity_loss(predicted, target)
    mse_loss = mean_squared_error_loss(predicted, target)
    
    return alpha * cosine_loss + (1 - alpha) * mse_loss
