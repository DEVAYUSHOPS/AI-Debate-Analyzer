import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("wiki_knowledge")

def add_documents(chunks):
    ids = [str(i) for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        ids=ids
    )