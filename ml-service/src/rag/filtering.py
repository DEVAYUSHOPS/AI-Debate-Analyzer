# src/rag/filtering.py

import numpy as np
from sentence_transformers import SentenceTransformer

# Load once (shared across project ideally)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# 1. Semantic Similarity Filtering
# =========================
def filter_chunks(query, chunks, top_k=5):
    """
    Ranks and filters chunks based on semantic similarity with query
    """

    if not chunks:
        return []

    # Encode query + chunks
    query_emb = embedder.encode([query])[0]
    chunk_embs = embedder.encode(chunks)

    # Compute similarity (dot product)
    scores = np.dot(chunk_embs, query_emb)

    # Sort indices by similarity (descending)
    ranked_indices = np.argsort(scores)[::-1]

    # Select top-k
    top_chunks = [chunks[i] for i in ranked_indices[:top_k]]

    return top_chunks


# =========================
# 2. Optional: Threshold Filtering (stricter)
# =========================
def filter_with_threshold(query, chunks, threshold=0.4):
    """
    Filters chunks using similarity threshold
    """

    if not chunks:
        return []

    query_emb = embedder.encode([query])[0]
    chunk_embs = embedder.encode(chunks)

    scores = np.dot(chunk_embs, query_emb)

    filtered = [
        chunks[i]
        for i in range(len(chunks))
        if scores[i] > threshold
    ]

    return filtered


# =========================
# 3. Hybrid Filtering (BEST)
# =========================
def hybrid_filter(query, chunks, top_k=5, threshold=0.3):
    """
    Combines threshold + ranking (recommended)
    """

    if not chunks:
        return []

    query_emb = embedder.encode([query])[0]
    chunk_embs = embedder.encode(chunks)

    scores = np.dot(chunk_embs, query_emb)

    # Step 1: threshold filter
    valid_indices = [i for i in range(len(chunks)) if scores[i] > threshold]

    if not valid_indices:
        return []

    # Step 2: rank remaining
    valid_scores = [(i, scores[i]) for i in valid_indices]
    sorted_scores = sorted(valid_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i for i, _ in sorted_scores[:top_k]]

    return [chunks[i] for i in top_indices]