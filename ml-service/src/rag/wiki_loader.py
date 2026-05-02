import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

def fetch_wikipedia_page(title):
    page = wiki.page(title)
    
    if not page.exists():
        return None
    
    return page.text