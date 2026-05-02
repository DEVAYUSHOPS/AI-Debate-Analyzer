# src/rag/pipeline.py

from .filtering import hybrid_filter
from .retriever import dynamic_retrieve, retrieve_with_debug
from sentence_transformers import SentenceTransformer
import numpy as np

# Shared embedder (reuse for filtering)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# 1. Chunk Filtering (IMPORTANT)
# =========================
def filter_chunks(query, chunks, top_k=5):
    """
    Filters and ranks retrieved chunks based on semantic similarity
    """

    if len(chunks) == 0:
        return []

    query_emb = embedder.encode([query])[0]
    chunk_embs = embedder.encode(chunks)

    scores = np.dot(chunk_embs, query_emb)

    # Sort by similarity
    ranked_indices = np.argsort(scores)[::-1]

    filtered = [chunks[i] for i in ranked_indices[:top_k]]

    return filtered


# =========================
# 2. Context Builder
# =========================
cache = {}

def build_context(query, top_k=5, topic=None):
    """
    Main retrieval + filtering pipeline
    """
    context, _ = build_context_with_debug(query, top_k=top_k, topic=topic)
    return context


def build_context_with_debug(query, top_k=5, topic=None):
    """
    Main retrieval + filtering pipeline with debug metadata for API inspection.
    """
    cache_key = (topic, query)
    if cache_key in cache:
        cached_context, cached_debug = cache[cache_key]
        cached_debug = {**cached_debug, "context_cache_hit": True}
        return cached_context, cached_debug

    raw_chunks, debug = retrieve_with_debug(query, k=10, topic=topic)
    debug["context_cache_hit"] = False

    if len(raw_chunks) == 0:
        context = "No relevant context found."
        cache[cache_key] = (context, debug)
        return context, debug

    filtered_chunks = hybrid_filter(query, raw_chunks, top_k=5)
    debug["filtered_chunk_count"] = len(filtered_chunks)

    # With a single fetched summary, the threshold filter can be too strict.
    # Falling back to raw chunks is better than sending an empty context.
    if not filtered_chunks:
        filtered_chunks = raw_chunks
        debug["filter_fallback"] = "raw_chunks"

    context = "\n\n".join(filtered_chunks)

    cache[cache_key] = (context, debug)

    return context, debug


# =========================
# 3. Final RAG Pipeline
# =========================
def rag_pipeline(argument, topic=None):
    """
    Returns enriched input with context + argument
    """

    context = build_context(argument, topic=topic)

    enriched_input = f"""
[CONTEXT]
{context}

[ARGUMENT]
{argument}
"""

    return enriched_input


# =========================
# 4. Fact Check Helper
# =========================
def simple_fact_check(argument):
    """
    Lightweight heuristic fact check (no LLM)
    """

    context = build_context(argument)

    if "No relevant context" in context:
        return "❌ No evidence found"

    overlap = sum(word in context.lower() for word in argument.lower().split())

    if overlap > 5:
        return "✅ Likely supported"

    elif overlap > 2:
        return "⚠️ Partially supported"

    else:
        return "❓ Needs verification"
