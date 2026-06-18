import hashlib
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance, 
    SparseVectorParams, Modifier, Document
)

# --- Configuration ---
QDRANT_URL = "http://192.168.4.23:6333"
COLLECTION_NAME = "memories_collection"
OLLAMA_MODEL = "qwen-embed-2k:latest"
OLLAMA_URL = "http://192.168.4.23:11434/api/embeddings"
qdrant = QdrantClient(url=QDRANT_URL)

def get_dense_embedding(text):
    payload = {"model": OLLAMA_MODEL, "prompt": text}
    response = requests.post(OLLAMA_URL, json=payload).json()
    return response.get("embedding")

def get_consistent_id(name):
    return hashlib.md5(name.encode()).hexdigest()

# --- Initialize Collection ---
if qdrant.collection_exists(collection_name=COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' already exists. Deleting to rebuild with Hybrid Indexes...")
    qdrant.delete_collection(collection_name=COLLECTION_NAME)

print(f"Creating Collection '{COLLECTION_NAME}' with Dense and Sparse indexes...")
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "qwen_dense": VectorParams(size=1024, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "keyword_sparse": SparseVectorParams(modifier=Modifier.IDF),
    }
)

# --- Memories Data ---
memories_to_add = [
    {"content": "Chuck's birthday is January 15th 1978."},
    {"content": "Rhonda Pardridge is a family member."},
    {"content": "Madruk prefers pepperoni pizza with extra sauce and extra cheese."},
    {"content": "The home theater system uses a 9.2-channel network AV receiver."}
]

print(f"Starting ingestion of {len(memories_to_add)} memories...")

for mem in memories_to_add:
    # 1. Get Dense embedding (Concepts) from Ollama
    dense_vec = get_dense_embedding(mem['content'])
    if not dense_vec: 
        print(f"Error: Failed to get dense embedding for {mem['content']}. Skipping.")
        continue

    # 2. Upsert using Qdrant's Native BM25 Inference for the sparse vector
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=get_consistent_id(mem["content"]),
                vector={
                    "qwen_dense": dense_vec,
                    "keyword_sparse": Document(text=mem["content"], model="qdrant/bm25")
                },
                payload={
                    "content": mem["content"],
                    "tags": ["personal_memory"]
                }
            )
        ]
    )

print("Ingestion complete!")

