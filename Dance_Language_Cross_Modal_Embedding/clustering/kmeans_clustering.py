import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def extract_embeddings(data):
    """
    Extract 256D embeddings from data points.
    
    Args:
        data (numpy.ndarray): Array of data points
        
    Returns:
        numpy.ndarray: Array of embeddings (shape: n_samples x 256)
    """
    embeddings = []
    
    for item in data:
        # Check if item has at least 2 elements and the second element is the 256D embedding
        if len(item) >= 2 and hasattr(item[1], 'shape'):
            # Make sure the embedding is 1D and has the expected dimension
            if len(item[1].shape) == 1 and item[1].shape[0] == 256:
                embeddings.append(item[1])
            else:
                # Try to flatten or reshape if needed
                try:
                    embedding = item[1].flatten()
                    if len(embedding) >= 256:
                        embeddings.append(embedding[:256])  # Truncate if too long
                    else:
                        # Pad if too short
                        padded = np.pad(embedding, (0, 256 - len(embedding)))
                        embeddings.append(padded)
                except:
                    print(f"Warning: Could not process embedding with shape {item[1].shape}")
        else:
            print(f"Warning: Item has unexpected format, skipping")
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    
    return embeddings

def perform_kmeans_clustering(embeddings, n_clusters, random_state=42):
    """
    Perform K-means clustering on embeddings.
    
    Args:
        embeddings (numpy.ndarray): Array of embeddings
        n_clusters (int): Number of clusters
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: 
            - cluster_labels (numpy.ndarray): Cluster labels for each data point
            - cluster_centers (numpy.ndarray): Cluster centroids
    """
    # Create and fit K-means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(embeddings, cluster_labels)
    print(f"K-means clustering with {n_clusters} clusters, silhouette score: {silhouette_avg:.3f}")
    
    # Count instances per cluster
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique_clusters, counts):
        print(f"Cluster {cluster_id}: {count} instances")
    
    return cluster_labels, kmeans.cluster_centers_
