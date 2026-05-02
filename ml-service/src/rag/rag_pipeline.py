# src/rag/pipeline.py

from .filtering import hybrid_filter
from .retriever import dynamic_retrieve, retrieve_with_debug
from .academic_retriever import search_semantic_scholar
from .query_rewriter import rewrite_query
from sentence_transformers import SentenceTransformer
import numpy as np

# Shared embedder (reuse for filtering)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# 1. Research Query Detection
# =========================
def is_research_query(text: str) -> bool:
    triggers = [
        "research", "study", "studies", "data",
        "evidence", "statistics", "report", "analysis"
    ]
    text = text.lower()
    return any(t in text for t in triggers)


# =========================
# 2. Chunk Filtering (legacy, not used directly now)
# =========================
def filter_chunks(query, chunks, top_k=5):
    if len(chunks) == 0:
        return []

    query_emb = embedder.encode([query])[0]
    chunk_embs = embedder.encode(chunks)

    scores = np.dot(chunk_embs, query_emb)
    ranked_indices = np.argsort(scores)[::-1]

    return [chunks[i] for i in ranked_indices[:top_k]]


# =========================
# 3. Context Builder
# =========================
cache = {}

def build_context(query, top_k=5, topic=None):
    context, _ = build_context_with_debug(query, top_k=top_k, topic=topic)
    return context


def build_context_with_debug(query, top_k=5, topic=None):
    """
    Enhanced RAG pipeline:
    - Multi-query retrieval
    - Academic retrieval (Semantic Scholar)
    - Deduplication
    - Diversity filtering (MMR-lite)
    - Structured context
    """

    from .query_expansion import expand_query

    cache_key = (topic, query)
    if cache_key in cache:
        cached_context, cached_debug = cache[cache_key]
        cached_debug = {**cached_debug, "context_cache_hit": True}
        return cached_context, cached_debug

    # =========================
    # 1. Query Expansion
    # =========================
    # queries = expand_query(query, topic=topic)
    # Step 1: Rewrite query (NEW 🔥)
    rewritten_query = rewrite_query(query, topic)

    # Step 2: Expand rewritten query
    queries = expand_query(rewritten_query, topic=topic)

    all_chunks = []
    retrieval_traces = []

    # =========================
    # 2. Multi-source Retrieval
    # =========================
    for q in queries:

        # --- Wikipedia / Base ---
        chunks, dbg = retrieve_with_debug(q, k=5, topic=topic)
        all_chunks.extend(chunks)

        retrieval_traces.append({
            "query": q,
            "source": "wikipedia",
            "num_chunks": len(chunks)
        })

        # --- Academic (Semantic Scholar) ---
        if is_research_query(query):
            academic_chunks = search_semantic_scholar(q, limit=3)

            all_chunks.extend(academic_chunks)

            retrieval_traces.append({
                "query": q,
                "source": "semantic_scholar",
                "num_chunks": len(academic_chunks)
            })

    # =========================
    # 3. Deduplicate
    # =========================
    unique_chunks = list(set([c.strip() for c in all_chunks if c.strip()]))

    if len(unique_chunks) == 0:
        context = "No supporting evidence retrieved from knowledge base."
        debug = {
            "queries_used": queries,
            "retrieval_traces": retrieval_traces,
            "final_chunk_count": 0,
            "context_cache_hit": False
        }
        cache[cache_key] = (context, debug)
        return context, debug

    # =========================
    # 4. Ranking
    # =========================
    query_emb = embedder.encode([query])[0]
    chunk_embs = embedder.encode(unique_chunks)

    scores = np.dot(chunk_embs, query_emb)

    ranked_indices = np.argsort(scores)[::-1]
    ranked_chunks = [unique_chunks[i] for i in ranked_indices]

    # =========================
    # 5. Diversity Filtering (MMR-lite)
    # =========================
    selected = []
    selected_embs = []

    for chunk in ranked_chunks:
        chunk_emb = embedder.encode([chunk])[0]

        if not selected:
            selected.append(chunk)
            selected_embs.append(chunk_emb)
            continue

        relevance = np.dot(chunk_emb, query_emb)
        diversity = max(np.dot(chunk_emb, emb) for emb in selected_embs)

        mmr_score = 0.7 * relevance - 0.3 * diversity

        if len(selected) < top_k:
            selected.append(chunk)
            selected_embs.append(chunk_emb)
        else:
            break

    filtered_chunks = selected

    # =========================
    # 6. Structured Context
    # =========================
    structured_context = ""

    for i, chunk in enumerate(filtered_chunks):

        # Label type of evidence
        if "(" in chunk and len(chunk) > 100:
            label = "Research Evidence"
        else:
            label = "General Evidence"

        structured_context += f"[{label} {i+1}]\n{chunk}\n\n"

    # =========================
    # 7. Debug Info
    # =========================
    debug = {
        "queries_used": queries,
        "retrieval_traces": retrieval_traces,
        "unique_chunks": len(unique_chunks),
        "final_chunks": len(filtered_chunks),
        "context_cache_hit": False
    }

    cache[cache_key] = (structured_context, debug)

    return structured_context, debug


# =========================
# 4. Final RAG Pipeline
# =========================
def rag_pipeline(argument, topic=None):
    context = build_context(argument, topic=topic)

    enriched_input = f"""
[CONTEXT]
{context}

[ARGUMENT]
{argument}
"""

    return enriched_input


# =========================
# 5. Fact Check Helper
# =========================
def simple_fact_check(argument):
    context = build_context(argument)

    if "No supporting evidence retrieved" in context:
        return "❌ No evidence found"

    overlap = sum(word in context.lower() for word in argument.lower().split())

    if overlap > 5:
        return "✅ Likely supported"
    elif overlap > 2:
        return "⚠️ Partially supported"
    else:
        return "❓ Needs verification"