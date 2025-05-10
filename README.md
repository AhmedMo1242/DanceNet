# DanceNet: AI-Enabled Choreography System

## Project Overview

This project explores the intersection of artificial intelligence and dance, developing two interconnected components:

1. **HathorSeq**: A specialized tool for labeling and segmenting motion capture sequences, enabling efficient annotation and visualization of dance data.

2. **Dance-Language Cross-Modal Embedding**: A comprehensive pipeline that bridges dance movements and natural language through contrastive learning, allowing bidirectional translation between these modalities.

The core approach involves first learning a latent representation of dance movements using a Variational Autoencoder, then aligning this representation with text embeddings through contrastive learning. This creates a unified semantic space where similar movements and descriptions are positioned closely, enabling both dance-to-text and text-to-dance translation.

## Project Structure

This repository is organized into two main components, each with detailed documentation:

- **[HathorSeq](hathorseq/README.md)**: Tools for motion capture sequence labeling and visualization.

- **[Dance-Language Cross-Modal Embedding](Dance_Language_Cross_Modal_Embedding/README.md)**: Implementation of the complete pipeline for dance-language embedding.

Each folder contains comprehensive documentation with architecture diagrams, implementation details, and visual examples. The Dance-Language Cross-Modal Embedding folder includes a complete technical report that follows IEEE conference paper formatting.

## Highlights

- Complete pipeline connecting motion capture data with natural language
- VAE-based dance representation learning system
- Unsupervised clustering approach that requires minimal manual labeling
- Contrastive learning framework for cross-modal alignment
- Bidirectional conversion between movement and text
- Detailed documentation and visualizations throughout
