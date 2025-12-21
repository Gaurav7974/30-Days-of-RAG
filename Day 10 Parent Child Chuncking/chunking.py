from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from docs import DOCUMENTS, PARENT_CHILDREN


def setup_chromadb():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.EphemeralClient()
    
    try:
        client.delete_collection("parent_child_chunks")
    except Exception:
        pass
    
    collection = client.get_or_create_collection(
        name="parent_child_chunks",
        embedding_function=embedding_fn
    )
    
    return collection


def index_documents(collection):
    parent_storage = {}
    child_to_parent = {}
    all_children = []
    
    for parent_id, child_chunks in PARENT_CHILDREN.items():
        parent_storage[parent_id] = DOCUMENTS[parent_id]
        
        for i, chunk in enumerate(child_chunks):
            child_id = f"{parent_id}_child_{i}"
            child_to_parent[child_id] = parent_id
            all_children.append({
                "id": child_id,
                "content": chunk,
                "parent_id": parent_id
            })
    
    collection.add(
        ids=[c["id"] for c in all_children],
        documents=[c["content"] for c in all_children],
        metadatas=[{"parent_id": c["parent_id"]} for c in all_children]
    )
    
    return parent_storage, child_to_parent


def retrieve(query, collection, parent_storage, child_to_parent, k=5, threshold=1.0):
    results = collection.query(query_texts=[query], n_results=k)
    
    child_ids = results["ids"][0]
    distances = results["distances"][0]
    child_contents = results["documents"][0]
    
    # Check relevance threshold
    if not child_ids or distances[0] > threshold:
        raise ValueError(f"No relevant results found. Try a different query.")
    
    seen = set()
    parents = []
    
    for child_id, distance, child_content in zip(child_ids, distances, child_contents):
        parent_id = child_to_parent.get(child_id)
        
        if parent_id and parent_id not in seen:
            parents.append({
                "content": parent_storage[parent_id],
                "score": round(1 - distance, 4),  # Convert distance to similarity score
                "matched_child": child_content
            })
            seen.add(parent_id)
    
    return parents


def search(query, collection, parent_storage, child_to_parent, top_k=3):
    try:
        results = retrieve(query, collection, parent_storage, child_to_parent, k=top_k, threshold=1.0)
        
        
        for i, result in enumerate(results, 1):
            print(f"[{i}] Score: {result['score']} Matched: \"{result['matched_child']}\"")
            print(f"{result['content'].strip()}\n")
        
    except ValueError as e:
        print(f"\nError: {e}\n")


def main():
    collection = setup_chromadb()
    parent_storage, child_to_parent = index_documents(collection)

    #user input
    while True:
        user_query = input("Enter your query: ").strip()
        
        search(user_query, collection, parent_storage, child_to_parent, top_k=3)


if __name__ == "__main__":
    main()