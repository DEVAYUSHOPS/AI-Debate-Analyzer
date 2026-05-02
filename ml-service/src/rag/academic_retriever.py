import requests


def search_semantic_scholar(query, limit=3):
    """
    Fetch research paper abstracts from Semantic Scholar API
    """

    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year"
    }

    try:
        response = requests.get(url, params=params, timeout=5)

        if response.status_code != 200:
            return []

        data = response.json()
        papers = data.get("data", [])

        results = []

        for p in papers:
            title = p.get("title", "")
            abstract = p.get("abstract", "")
            year = p.get("year", "")

            if abstract:
                text = f"{title} ({year}): {abstract}"
                results.append(text)

        return results

    except Exception:
        return []