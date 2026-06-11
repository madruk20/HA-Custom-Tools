
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
import requests

# --- TOOL CONFIGURATION (MATCHING YOUR price_lookup.py) ---
TOOL_NAME = "music_player"
TOOL_DESCRIPTION = "Control music playback, stream audio, and browse your media library. Use this to play specific artists, albums, songs, or playlists, and for transport actions like pause, stop, or skip."

# Storing the schema here allows your LLM to 'read' the tool requirements from memory
TOOL_SCHEMA = {
    "parameters": {
        "query": "The media control action to perform. Choose from: 'play', 'pause', 'next', or 'previous'.",
        "api": "Must be 'play', 'pause', 'next', 'previous'"
    }
}

# --- EMBEDDING BRIDGE ---
def get_embedding(text: str):
    api_url = "http://192.168.4.23:8150/v1/embeddings"
    response = requests.post(api_url, json={"input": text})
    return response.json()["data"][0]["embedding"]

# --- QDRANT OPS ---
client = QdrantClient(url="http://192.168.4.23:6333")
COLLECTION_NAME = "tools_collection"

# client.create_collection(
#    collection_name=COLLECTION_NAME,
#    vectors_config=models.VectorParams(size=2048, distance=models.Distance.COSINE),
#    quantization_config=models.ScalarQuantization(
#        scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
#    )
#)

# 1. Generate the semantic vector from the ACTUAL description
print(f"Embedding tool: {TOOL_NAME}...")
vector = get_embedding(TOOL_DESCRIPTION)

# 2. Add to Qdrant with full schema metadata
client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "tool_id": TOOL_NAME,
                "description": TOOL_DESCRIPTION,
                "schema": TOOL_SCHEMA,
                "domain": "music and audio",
                "sector": "tools_schema"
            }
        )
    ]
)
print("✅ Indexed tool with full parameter schema and semantic vector.")