# src/rag/retriever.py

import uuid
import re
from typing import Dict, List, Optional, Tuple

import chromadb
import wikipedia
from chromadb.utils import embedding_functions

from .query_expansion import (
    build_retrieval_query,
    build_wikipedia_queries,
    extract_keywords,
)


PLACEHOLDER_MARKERS = (
    "no factual context found",
    "no relevant context found",
    "no specific factual context found",
)

CACHE_SOFT_TERMS = {
    "academic", "argument", "attention", "class", "classes", "classroom",
    "education", "government", "governments", "lesson", "lessons", "policy",
    "school", "schools", "student", "students",
}

REQUIRED_TERM_GROUPS = [
    {"smartphone", "phone", "mobile", "cellphone"},
    {"transport", "transit", "bus", "train"},
    {"renewable", "energy"},
    {"fossil", "fuel"},
    {"ai", "artificial", "intelligence"},
]


# =========================
# 1. Initialize Vector Database
# =========================
chroma_client = chromadb.PersistentClient(path="./chroma_db")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)

collection = chroma_client.get_or_create_collection(
    name="debate_knowledge_base",
    embedding_function=sentence_transformer_ef
)


def is_placeholder_context(text: str) -> bool:
    if not text:
        return True

    normalized = text.strip().lower()
    return any(marker in normalized for marker in PLACEHOLDER_MARKERS)


def normalize_term(term: str) -> str:
    if term.endswith("ies") and len(term) > 4:
        return term[:-3] + "y"
    if term.endswith("sses"):
        return term[:-2]
    if term.endswith("s") and len(term) > 3:
        return term[:-1]
    return term


def document_terms(text: str) -> set:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", text.lower())
    return {
        normalize_term(token.strip("'"))
        for token in tokens
        if len(token.strip("'")) > 2
    }


def cache_relevance(query: str, document: str) -> Tuple[bool, Dict]:
    """
    Embedding distance alone can confuse broad school/education topics. Require
    the cached text to share at least one specific factual term from the query.
    """
    query_terms = {
        normalize_term(term)
        for term in extract_keywords(query, max_keywords=14)
    }

    important_query_terms = query_terms - CACHE_SOFT_TERMS
    doc_terms = document_terms(document)
    shared_terms = sorted(important_query_terms & doc_terms)

    debug = {
        "cache_query_terms": sorted(important_query_terms),
        "cache_overlap_terms": shared_terms,
    }

    for group in REQUIRED_TERM_GROUPS:
        if important_query_terms & group and not doc_terms & group:
            debug["cache_relevance_reason"] = (
                f"missing_required_topic_terms:{','.join(sorted(group))}"
            )
            return False, debug

    if not important_query_terms:
        debug["cache_relevance_reason"] = "no_specific_query_terms"
        return True, debug

    if shared_terms:
        debug["cache_relevance_reason"] = "specific_term_overlap"
        return True, debug

    debug["cache_relevance_reason"] = "no_specific_term_overlap"
    return False, debug


def cache_wikipedia_summary(summary: str, title: str, query: str) -> None:
    if is_placeholder_context(summary):
        return

    fact_id = f"wiki_{uuid.uuid4().hex[:8]}"

    collection.add(
        ids=[fact_id],
        documents=[summary],
        metadatas=[{
            "source": "wikipedia_auto_cache",
            "title": title,
            "query": query,
        }]
    )


# =========================
# 2. Wikipedia Fallback & Flywheel
# =========================
def fetch_and_cache_from_wikipedia(queries: List[str]) -> Tuple[str, Dict]:
    """
    Searches Wikipedia with multiple concise candidates, caches the first useful
    summary in ChromaDB, and returns both the text and debug metadata.
    """
    debug = {
        "source": "wikipedia",
        "wikipedia_queries": queries,
        "wikipedia_search_results": {},
        "wikipedia_title": None,
        "errors": [],
    }

    for query in queries:
        print(f"Searching Wikipedia for '{query}'...")

        page_titles = [query]

        try:
            search_results = wikipedia.search(query, results=5)
            debug["wikipedia_search_results"][query] = search_results
            page_titles.extend(search_results)
        except Exception as exc:
            debug["errors"].append(f"search:{query}:{exc}")

        for title in dict.fromkeys(page_titles):
            try:
                summary = wikipedia.summary(
                    title,
                    sentences=4,
                    auto_suggest=False
                )

                if is_placeholder_context(summary):
                    continue

                cache_wikipedia_summary(summary, title=title, query=query)
                debug["wikipedia_title"] = title
                print(f"Cached Wikipedia page '{title}'.")
                return summary, debug

            except wikipedia.exceptions.DisambiguationError as exc:
                debug["errors"].append(f"disambiguation:{title}")
                for option in exc.options[:3]:
                    try:
                        summary = wikipedia.summary(
                            option,
                            sentences=4,
                            auto_suggest=False
                        )
                        if is_placeholder_context(summary):
                            continue

                        cache_wikipedia_summary(summary, title=option, query=query)
                        debug["wikipedia_title"] = option
                        return summary, debug

                    except Exception as option_exc:
                        debug["errors"].append(f"option:{option}:{option_exc}")

            except Exception as exc:
                debug["errors"].append(f"summary:{title}:{exc}")

    debug["source"] = "none"
    return "", debug


# =========================
# 3. Hybrid Retrieval Function
# =========================
def retrieve_with_debug(
    query: str,
    k: int = 1,
    topic: Optional[str] = None
) -> Tuple[List[str], Dict]:
    """
    Searches local ChromaDB first. If confidence is low or a cached placeholder
    is found, falls back to Wikipedia search.
    """
    retrieval_query = build_retrieval_query(query, topic=topic)
    wikipedia_queries = build_wikipedia_queries(query, topic=topic)

    debug = {
        "retrieval_query": retrieval_query,
        "wikipedia_queries": wikipedia_queries,
        "source": None,
        "cache_distance": None,
        "cache_hit": False,
        "errors": [],
    }

    try:
        results = collection.query(
            query_texts=[retrieval_query],
            n_results=k
        )

        if results and "documents" in results and results["documents"][0]:
            best_fact = results["documents"][0][0]
            distance_score = results["distances"][0][0]
            debug["cache_distance"] = round(float(distance_score), 4)
            is_relevant, relevance_debug = cache_relevance(
                retrieval_query,
                best_fact
            )
            debug.update(relevance_debug)

            if (
                distance_score < 1.3
                and not is_placeholder_context(best_fact)
                and is_relevant
            ):
                debug["source"] = "chroma_cache"
                debug["cache_hit"] = True
                print(f"Local cache hit. Distance: {distance_score:.2f}")
                return [best_fact], debug

            debug["cache_rejected"] = True
            print(f"Local cache rejected or weak match. Distance: {distance_score:.2f}")

    except Exception as exc:
        debug["errors"].append(f"chroma:{exc}")

    summary, wiki_debug = fetch_and_cache_from_wikipedia(wikipedia_queries)
    debug.update(wiki_debug)

    if is_placeholder_context(summary):
        return [], debug

    return [summary], debug


def dynamic_retrieve(query: str, k: int = 1, topic: Optional[str] = None) -> List[str]:
    chunks, _ = retrieve_with_debug(query, k=k, topic=topic)
    return chunks
