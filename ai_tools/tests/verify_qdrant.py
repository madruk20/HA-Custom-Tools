import time
import requests
from qdrant_client import QdrantClient

# Configuration
QDRANT_URL = "http://192.168.4.23:6333"
OLLAMA_URL = "http://192.168.4.23:11434/api/embeddings"
OLLAMA_MODEL = "qwen-embed-2k:latest"

client = QdrantClient(url=QDRANT_URL)

def get_embedding_with_time(text):
    start = time.perf_counter()
    payload = {"model": OLLAMA_MODEL, "prompt": text}
    response = requests.post(OLLAMA_URL, json=payload).json()
    end = time.perf_counter()
    return response["embedding"], (end - start)

def search_qdrant_with_time(query_vec):
    start = time.perf_counter()
    search_res = client.query_points(
        collection_name="tools_collection", 
        query=query_vec, 
        limit=5
    )
    end = time.perf_counter()
    return search_res, (end - start)

# --- Execution ---
print("Running latency test...")

# 1. Test Embedding
query = "What is the current stock price of VTI?"
vec, embed_time = get_embedding_with_time(query)
print(f"Embedding generation took: {embed_time:.4f} seconds")

# 2. Test Qdrant Search
res, search_time = search_qdrant_with_time(vec)
print(f"Qdrant search took: {search_time:.4f} seconds")

print("\n--- Results ---")
for hit in res.points:
    print(f"Tool: {hit.payload.get('tool_id')} | Score: {hit.score:.4f}")