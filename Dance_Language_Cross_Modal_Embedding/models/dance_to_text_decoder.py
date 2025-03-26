import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel

class DanceToTextDecoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, bert_model_name="bert-base-uncased"):
        super(DanceToTextDecoder, self).__init__()
        
        # Projection network to map dance embeddings to BERT-compatible space
        self.projection_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 768),  # BERT's dimension
            nn.LayerNorm(768)
        )
        
        # Load BERT tokenizer and model for generating embeddings
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.bert_model.eval()  # Set to evaluation mode
        
        # Initialize mapping dictionary (to be filled during training)
        self.text_embeddings_dict = {}
        self.texts_by_cluster = {}
        
    def forward(self, dance_embedding):
        # Project dance embedding to BERT-compatible space
        projected_embedding = self.projection_network(dance_embedding)
        return projected_embedding
    
    def build_text_embedding_database(self, text_encoder, unique_texts, cluster_ids):
        """Build a database of text embeddings for inference lookup using raw BERT embeddings"""
        # Store the encoder just in case it's needed for reference
        self.text_encoder = text_encoder
        
        # Create mapping of texts by cluster for faster lookup
        for text, cluster_id in zip(unique_texts, cluster_ids):
            # Convert numpy array to hashable type (assuming cluster IDs are integers)
            cluster = int(cluster_id.item()) if hasattr(cluster_id, 'item') else int(cluster_id)
            
            if cluster not in self.texts_by_cluster:
                self.texts_by_cluster[cluster] = []
            self.texts_by_cluster[cluster].append(text)
    
        # Create BERT embeddings for all unique texts
        device = next(self.parameters()).device
        with torch.no_grad():
            for text in unique_texts:
                # Tokenize the text
                encoded_input = self.tokenizer(text, return_tensors='pt', 
                                              padding=True, truncation=True, max_length=128)
                encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                
                # Get BERT embedding (using [CLS] token representation)
                outputs = self.bert_model(**encoded_input)
                bert_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
                
                # Store the embedding
                self.text_embeddings_dict[text] = bert_embedding.cpu()
    
    def find_closest_text(self, dance_embedding, top_k=5, target_cluster=None):
        """Find the closest text given a dance embedding"""
        # Ensure dance_embedding has the right shape
        if dance_embedding.dim() == 1:
            dance_embedding = dance_embedding.unsqueeze(0)
            
        # Project dance embedding to BERT space
        projected_embedding = self.forward(dance_embedding)
        
        texts_to_search = []
        if target_cluster is not None and target_cluster in self.texts_by_cluster:
            # Only search texts from the target cluster if specified
            texts_to_search = self.texts_by_cluster[target_cluster]
        else:
            # Otherwise search all texts
            texts_to_search = list(self.text_embeddings_dict.keys())
        
        similarities = []
        for text in texts_to_search:
            text_embedding = self.text_embeddings_dict[text].to(dance_embedding.device)
            similarity = torch.cosine_similarity(projected_embedding, text_embedding, dim=1)
            similarities.append((text, similarity.item()))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        return similarities[:top_k]
