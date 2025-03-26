# AI-Enabled Choreography: GSoC 2025 Test Submission

## Project Overview

Hi! I'm Ahmed Hassan, and this is my submission for the HumanAI Foundation's "AI-Enabled Choreography: Dance Beyond Music" GSoC 2025 assessment.

For this test, I developed two interconnected components:

1. **HathorSeq**: A specialized tool for labeling and segmenting motion capture sequences, enabling efficient annotation and visualization of dance data.

2. **Dance-Language Cross-Modal Embedding**: A comprehensive pipeline that bridges dance movements and natural language through contrastive learning, allowing bidirectional translation between these modalities.

The core of my approach involves first learning a latent representation of dance movements using a Variational Autoencoder, then aligning this representation with text embeddings through contrastive learning. This creates a unified semantic space where similar movements and descriptions are positioned closely, enabling both dance-to-text and text-to-dance translation.

## Project Structure

This repository is organized into two main components, each with detailed documentation:

- **[HathorSeq](hathorseq/README.md)**: My solution for Task 1, providing tools for motion capture sequence labeling and visualization.

- **[Dance-Language Cross-Modal Embedding](Dance_Language_Cross_Modal_Embedding/README.md)**: My solution for Task 2, implementing the complete pipeline for dance-language embedding.

Each folder contains comprehensive documentation with architecture diagrams, implementation details, and visual examples. I've also included a complete technical report in the Dance-Language Cross-Modal Embedding folder that follows IEEE conference paper formatting.

## Personal Motivation

In [why_this_project.md](why_this_project.md), I've shared my personal motivation for applying to this project, my background in creative arts, and thoughts on the intersection of AI and artistic expression. I've also outlined my preferred collaboration style and some ideas I'm excited to explore further.

## Highlights

- Developed a complete pipeline connecting motion capture data with natural language
- Implemented a VAE-based dance representation learning system
- Created an unsupervised clustering approach that requires minimal manual labeling
- Built a contrastive learning framework for cross-modal alignment
- Demonstrated bidirectional conversion between movement and text
- Provided detailed documentation and visualizations throughout

I'm excited about the potential of this work and would be thrilled to continue developing it as part of the GSoC program with the HumanAI Foundation.
