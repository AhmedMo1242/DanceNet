import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def apply_pca(embeddings, n_components=2):
    """
    Apply PCA to reduce dimensionality of embeddings.
    
    Args:
        embeddings (numpy.ndarray): Array of embeddings
        n_components (int): Number of components to keep
        
    Returns:
        tuple: 
            - reduced_embeddings (numpy.ndarray): Reduced dimensionality embeddings
            - pca (sklearn.decomposition.PCA): Fitted PCA object
    """
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    explained_var = pca.explained_variance_ratio_
    print(f"PCA explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}")
    
    return reduced_embeddings, pca

def plot_clusters(reduced_embeddings, cluster_labels, output_path, 
                 figsize=(12, 10), dpi=300, colors=None, title=None):
    """
    Create scatter plot of clusters.
    
    Args:
        reduced_embeddings (numpy.ndarray): Reduced embeddings (2D)
        cluster_labels (numpy.ndarray): Cluster labels for each data point
        output_path (str): Path to save the plot
        figsize (tuple): Figure size
        dpi (int): DPI for the saved figure
        colors (list): Colors for each cluster
        title (str): Title for the plot
    """
    plt.figure(figsize=figsize)
    
    # Get number of clusters
    n_clusters = len(np.unique(cluster_labels))
    
    # Set colors if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    elif len(colors) < n_clusters:
        # Extend colors if not enough provided
        colors = colors + list(plt.cm.tab10(np.linspace(0, 1, n_clusters - len(colors))))
    
    # Plot each cluster with its own color
    for i in range(n_clusters):
        mask = cluster_labels == i
        plt.scatter(
            reduced_embeddings[mask, 0], 
            reduced_embeddings[mask, 1],
            s=80, 
            color=colors[i] if i < len(colors) else plt.cm.tab10(i / n_clusters),
            label=f'Cluster {i} ({np.sum(mask)} items)'
        )
    
    # Set title and labels
    if title:
        plt.title(title)
    else:
        plt.title(f'Cluster visualization (k={n_clusters})')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the plot
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    
    print(f"Saved cluster visualization to {output_path}")

def plot_silhouette_scores(scores, n_clusters_range, output_path, figsize=(10, 6), dpi=300):
    """
    Plot silhouette scores for different numbers of clusters.
    
    Args:
        scores (list): List of silhouette scores
        n_clusters_range (list): Range of number of clusters
        output_path (str): Path to save the plot
        figsize (tuple): Figure size
        dpi (int): DPI for the saved figure
    """
    plt.figure(figsize=figsize)
    plt.plot(n_clusters_range, scores, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    
    print(f"Saved silhouette score plot to {output_path}")

def visualize_clusters_with_text(reduced_embeddings, cluster_labels, text_descriptions, 
                                output_path, figsize=(14, 10), dpi=300, colors=None):
    """
    Create scatter plot of clusters with text descriptions.
    
    Args:
        reduced_embeddings (numpy.ndarray): Reduced embeddings (2D)
        cluster_labels (numpy.ndarray): Cluster labels for each data point
        text_descriptions (list): Text descriptions for each data point
        output_path (str): Path to save the plot
        figsize (tuple): Figure size
        dpi (int): DPI for the saved figure
        colors (list): Colors for each cluster
    """
    plt.figure(figsize=figsize)
    
    # Get number of clusters
    n_clusters = len(np.unique(cluster_labels))
    
    # Set colors if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    elif len(colors) < n_clusters:
        # Extend colors if not enough provided
        colors = colors + list(plt.cm.tab10(np.linspace(0, 1, n_clusters - len(colors))))
    
    # Plot each point with its text description
    for i, (x, y) in enumerate(reduced_embeddings):
        cluster_id = cluster_labels[i]
        plt.scatter(x, y, color=colors[cluster_id] if cluster_id < len(colors) else plt.cm.tab10(cluster_id / n_clusters), 
                   s=70, alpha=0.7)
        
        # Add text description as annotation
        if i < len(text_descriptions):
            plt.annotate(
                text_descriptions[i], 
                (x, y),
                fontsize=7,
                alpha=0.8,
                xytext=(5, 5),
                textcoords='offset points'
            )
    
    # Add legend for clusters
    for i in range(n_clusters):
        plt.scatter([], [], 
                   color=colors[i] if i < len(colors) else plt.cm.tab10(i / n_clusters),
                   label=f'Cluster {i}')
    
    plt.title('Visualization with text descriptions')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    
    print(f"Saved text visualization to {output_path}")

