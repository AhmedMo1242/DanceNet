import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DanceToTextDataset(Dataset):
    def __init__(self, data_path, text_encoder=None, device='cpu'):
        """
        Dataset for Dance-to-Text training
        
        Args:
            data_path: Path to .npy file with dance data
            text_encoder: Optional text encoder to precompute text embeddings
            device: Device to use for tensor operations
        """
        self.data = np.load(data_path, allow_pickle=True)
        self.device = device
        
        # Extract all unique text descriptions
        unique_texts = set()
        for i in range(len(self.data)):
            text = self.data[i][3][0]  # x[i][3][0] contains text description
            unique_texts.add(text)
        
        self.unique_texts = list(unique_texts)
        print(f"Found {len(self.unique_texts)} unique text descriptions")
        
        # Precompute text embeddings if encoder is provided
        self.text_embeddings_cache = {}
        if text_encoder is not None:
            text_encoder.eval()
            with torch.no_grad():
                for text in self.unique_texts:
                    embedding = text_encoder([text])[0].cpu()
                    self.text_embeddings_cache[text] = embedding
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get dance embedding (256d)
        dance_embedding = torch.tensor(self.data[idx][5], dtype=torch.float32).to(self.device)
        
        # Get corresponding text and cluster ID
        text = self.data[idx][3][0]
        cluster_id = self.data[idx][2]
        
        # Use cached embedding if available, otherwise return a zero tensor
        if text in self.text_embeddings_cache:
            text_embedding = self.text_embeddings_cache[text].to(self.device)
        else:
            text_embedding = torch.zeros(768, dtype=torch.float32).to(self.device)  
        
        return {
            'dance_embedding': dance_embedding,
            'text': text,
            'cluster_id': cluster_id,
            'text_embedding': text_embedding
        }
    
    def get_unique_texts_with_clusters(self):
        """Return a mapping of unique texts to their cluster IDs"""
        text_to_cluster = {}
        for i in range(len(self.data)):
            text = self.data[i][3][0]
            cluster_id = self.data[i][2]
            if text not in text_to_cluster:
                text_to_cluster[text] = cluster_id
        return text_to_cluster

def create_dance_to_text_dataloader(data_path, batch_size=32, text_encoder=None, device='cpu', shuffle=True):
    """Create a dataloader for dance-to-text training"""
    dataset = DanceToTextDataset(data_path, text_encoder, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Get unique texts and their cluster IDs
    texts_with_clusters = dataset.get_unique_texts_with_clusters()
    unique_texts = list(texts_with_clusters.keys())
    cluster_ids = [texts_with_clusters[text] for text in unique_texts]
    
    return dataloader, unique_texts, cluster_ids
