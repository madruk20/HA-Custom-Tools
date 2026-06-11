import hashlib
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# --- Configuration ---
QDRANT_URL = "http://192.168.4.23:6333"
COLLECTION_NAME = "memories_collection"
qdrant = QdrantClient(url=QDRANT_URL)

def get_embedding(text):
    payload = {"model": "qwen-embed-2k:latest", "prompt": text}
    response = requests.post("http://192.168.4.23:11434/api/embeddings", json=payload).json()
    return response.get("embedding")

def get_consistent_id(text):
    return hashlib.md5(text.encode()).hexdigest()

# --- Initialize Collection ---
if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
    print(f"Creating missing collection: {COLLECTION_NAME}")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
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
    vec = get_embedding(mem["content"])
    if not vec: continue

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=get_consistent_id(mem["content"]),
                vector=vec,
                payload={"content": mem["content"], "tags": ["personal_memory"]}
            )
        ]
    )
    print(f"Indexed: {mem['content'][:40]}...")

print("Memory ingestion complete!")