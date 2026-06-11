import os
import requests
import chromadb

# 1. Setup local persistence directory
# This mimics how Home Assistant will store it locally on its storage drive
current_dir = os.path.dirname(os.path.abspath(__file__))
db_storage_path = os.path.join(current_dir, "local_test_chroma")

print(f"Initializing local ChromaDB in: {db_storage_path}")
chroma_client = chromadb.PersistentClient(path=db_storage_path)

# 2. Define collection using Cosine similarity
# Distance of 0.0 means perfect match, 1.0 means completely unrelated
collection = chroma_client.get_or_create_collection(
    name="home_assistant_test_memory",
    metadata={"hnsw:space": "cosine"}
)

def get_embedding(text: str) -> list[float]:
    """Fetches text coordinates from your Ollama container."""
    payload = {
        "model": "nomic-embed-text",
        "prompt": text
    }
    try:
        # Update this IP if your Unraid server address shifts
        response = requests.post("http://192.168.4.23:11434/api/embeddings", json=payload, timeout=5)
        response.raise_for_status()
        return response.json().get("embedding")
    except Exception as e:
        print(f"[-] Ollama Embedding Connection Failed: {e}")
        return []

def add_test_memories():
    """Simulates loading structured data into new vector space."""
    # A mix of categories and topics to test filtering accuracy
    sample_facts = [
        {"id": "mem_1", "category": "schedule", "key": "flight_to_ny", "value": "2026-06-05 06:00:00"},
        {"id": "mem_2", "category": "schedule", "key": "dentist_appointment", "value": "2026-06-15 14:00:00"},
        {"id": "mem_3", "category": "food", "key": "favorite_drink", "value": "Water"},
        {"id": "mem_4", "category": "food", "key": "allergic_to", "value": "Peanuts"},
        {"id": "mem_5", "category": "general", "key": "wifi_password", "value": "dragon123"},
    ]

    ids = []
    embeddings = []
    metadatas = []
    documents = []

    print("\n[+] Generating embeddings via Ollama...")
    for fact in sample_facts:
        combined_text = f"{fact['key']}: {fact['value']}"
        vector = get_embedding(combined_text)
        
        if vector:
            ids.append(fact['id'])
            embeddings.append(vector)
            metadatas.append({"category": fact['category']})
            documents.append(combined_text)

    if ids:
        # This writes data cleanly into your local directory database
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"[+] Successfully database-seeded {len(ids)} items locally.")

def query_memory(user_query: str, category_filter: str = None, top_k: int = 2, min_confidence: float = 0.60):
    """Executes a strict hybrid query against the local vector database."""
    print(f"\nEvaluating User Voice Query: '{user_query}' (Filter: {category_filter})")
    
    query_vector = get_embedding(user_query)
    if not query_vector:
        print("[-] Query failed: Could not fetch vector from Ollama.")
        return

    # Build standard metadata parameters
    search_args = {
        "query_embeddings": [query_vector],
        "n_results": top_k * 2 # Pull extra entries to filter against thresholds
    }
    
    if category_filter:
        search_args["where"] = {"category": category_filter}

    # Execute DB query
    results = collection.query(**search_args)
    
    valid_results = []
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            
            # Map cosine distance directly to an absolute confidence score
            confidence = 1.0 - distance
            
            # Apply confidence guardrail
            if confidence >= min_confidence:
                valid_results.append({
                    "content": results['documents'][0][i],
                    "category": results['metadatas'][0][i]['category'],
                    "confidence": round(confidence, 4)
                })

    # Return strict top-k slice
    final_output = valid_results[:top_k]
    
    if not final_output:
        print("   -> No matched memories cleared the confidence barrier.")
    for rank, match in enumerate(final_output, start=1):
        print(f"   [{rank}] MATCH: {match['content']} | Category: {match['category']} | Confidence: {match['confidence']}")

# ==========================================
# EXECUTION ENGINE
# ==========================================
if __name__ == "__main__":
    # 1. Seed the local DB file folder
    add_test_memories()
    
    # TEST 1: Semantic Lookup (Using 'plane trip' instead of 'flight')
    # Should catch the New York flight details perfectly
    query_memory(
        user_query="When is my plane trip?", 
        category_filter="schedule", 
        top_k=1, 
        min_confidence=0.60
    )
    
    # TEST 2: Intent Isolation Cross-Check
    # Looking up 'water' but specifying 'schedule'. 
    # The database metadata filter should prevent food data from leaking through.
    query_memory(
        user_query="Tell me about water", 
        category_filter="food", 
        top_k=1, 
        min_confidence=0.60
    )

    # TEST 3: Pure Semantic Matching without hard category filtering
    query_memory(
        user_query="What's the internet password?", 
        category_filter="general", 
        top_k=1, 
        min_confidence=0.55
    )