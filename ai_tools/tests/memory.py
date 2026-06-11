import os
import time
import requests
import chromadb
from chromadb.errors import InternalError

# ==========================================
# 1. DATABASE SETUP & SMB LOCK HANDLING
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
db_storage_path = os.path.join(current_dir, "local_test_chroma")

def get_chroma_client(retries=5, delay=1.0):
    """Safely connects to ChromaDB, handling SMB file locks on Unraid network shares."""
    print(f"Connecting to ChromaDB at: {db_storage_path}")
    for attempt in range(retries):
        try:
            return chromadb.PersistentClient(path=db_storage_path)
        except InternalError as e:
            if "526" in str(e) or "unable to open database" in str(e):
                print(f"   [!] Database locked by network (Attempt {attempt+1}/{retries}). Waiting {delay}s...")
                time.sleep(delay)
            else:
                raise e
    raise Exception("Failed to connect to ChromaDB after multiple retries. Is the file permanently locked?")

chroma_client = get_chroma_client()
collection = chroma_client.get_or_create_collection(
    name="home_assistant_master_memory",
    metadata={"hnsw:space": "cosine"}
)

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================
def get_embedding(text: str) -> list[float]:
    """Fetches text coordinates from local Ollama container."""
    payload = {"model": "mxbai-embed-large", "prompt": text}
    try:
        response = requests.post("http://192.168.4.23:11434/api/embeddings", json=payload, timeout=5)
        response.raise_for_status()
        return response.json().get("embedding")
    except Exception as e:
        print(f"[-] Ollama Embedding Failed: {e}")
        return []

def chunk_text_by_words(text: str, max_words: int = 40, overlap_words: int = 10) -> list[str]:
    """Slices long logs into overlapping chunks."""
    words = text.split()
    chunks = []
    if len(words) <= max_words:
        return [text]
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        start += (max_words - overlap_words)
    return chunks

# ==========================================
# 3. CRUD OPERATIONS (CREATE, UPDATE, DELETE)
# ==========================================
def upsert_memory(memory_id: str, index_text: str, domain: str, storage_mode: str = "standard", full_payload: str = None):
    """
    UPSERT means 'Update or Insert'. If the memory_id doesn't exist, it creates it.
    If it DOES exist, it completely overwrites the old memory with the new data.
    """
    vector = get_embedding(index_text)
    if not vector: return

    metadata = {
        "domain": domain,
        "storage_mode": storage_mode
    }
    
    # Attach payload if provided
    if full_payload and storage_mode == "payload_attached":
        metadata["full_text"] = full_payload

    # Using .upsert() instead of .add() allows us to overwrite data cleanly
    collection.upsert(
        ids=[memory_id],
        embeddings=[vector],
        metadatas=[metadata],
        documents=[index_text]
    )
    print(f"   [+] Upserted Memory ID: {memory_id} (Domain: {domain})")

def delete_memory(memory_id: str):
    """Permanently deletes a memory from the database by its ID."""
    collection.delete(ids=[memory_id])
    print(f"   [-] Deleted Memory ID: {memory_id}")

def add_chunked_log(log_prefix: str, log_text: str, domain: str):
    """Breaks a massive log into chunks and upserts them."""
    chunks = chunk_text_by_words(log_text)
    for index, chunk in enumerate(chunks):
        chunk_id = f"{log_prefix}_chunk_{index}"
        upsert_memory(memory_id=chunk_id, index_text=chunk, domain=domain, storage_mode="chunked")

# ==========================================
# 4. QUERY / RETRIEVAL ENGINE
# ==========================================
def query_memory(user_query: str, target_domain: str, top_k: int = 1, min_confidence: float = 0.60):
    """Executes a hard-filtered query and dynamically parses payloads/chunks."""
    print(f"\nEvaluating Voice Query: '{user_query}' | Filter: [{target_domain}]")
    
    query_vector = get_embedding(user_query)
    if not query_vector: return

    # Hard Metadata Pre-Filter
    search_args = {
        "query_embeddings": [query_vector],
        "n_results": top_k,
        "where": {"domain": target_domain} 
    }

    results = collection.query(**search_args)
    
    if results['ids'] and results['ids'][0]:
        distance = results['distances'][0][0]
        confidence = 1.0 - distance # Dynamic Confidence calculation
        
        if confidence >= min_confidence:
            metadata = results['metadatas'][0][0]
            print(f"   -> [Match Found] Confidence: {round(confidence, 4)}")
            
            if metadata.get("storage_mode") == "payload_attached":
                print(f"   -> Unpacking Payload: \n      {metadata.get('full_text')}")
            else:
                print(f"   -> Returned Text: \"{results['documents'][0][0]}\"")
        else:
            print(f"   -> Match fell below the safety confidence barrier ({round(confidence, 4)}).")
    else:
        print("   -> No records found in this domain.")

# ==========================================
# 5. TEST SCENARIOS
# ==========================================
if __name__ == "__main__":
    print("\n--- PHASE 1: SEEDING THE DATABASE ---")
    
    # 1. Standard Info
    upsert_memory("wifi_pass", "The guest wifi password is dragon123", "infrastructure")
    
    # 2. Preference (To be updated later)
    upsert_memory("pref_drink", "My favorite drink is iced water.", "lifestyle")
    
    # 3. Payload Attachment (Massive Recipe)
    upsert_memory(
        memory_id="recipe_brownies", 
        index_text="Fudgy Chocolate Brownie Recipe with walnuts and cocoa powder dessert baking", 
        domain="lifestyle",
        storage_mode="payload_attached",
        full_payload="[MASSIVE 1000 WORD RECIPE TEXT HIDDEN IN METADATA...]"
    )
    
    # 4. Chunking (Massive Log)
    add_chunked_log(
        log_prefix="sec_log_june5",
        log_text="At 08:14 AM Front Door Camera detected the mailman dropping off envelopes. At 11:32 AM Driveway Camera flagged a stray white dog running past the gate. At 03:45 PM Backyard Camera recorded a delivery courier placing a large cardboard package behind the patio chairs.",
        domain="security"
    )

    print("\n--- PHASE 2: INITIAL QUERIES ---")
    
    # Test A: Should return the massive hidden payload based on the tiny vector index
    query_memory("How do I bake those chocolate brownies?", target_domain="lifestyle")
    
    # Test B: Should find the exact slice of the chunked log
    query_memory("Where did the delivery guy leave the package?", target_domain="security")
    
    # Test C: Verify the current preference
    query_memory("What is my favorite drink?", target_domain="lifestyle")

    print("\n--- PHASE 3: UPDATING AND DELETING ---")
    
    # Update: The user changes their mind. We use upsert on the exact same ID.
    print("\n[Action] Updating favorite drink to Soda...")
    upsert_memory("pref_drink", "My favorite drink is Coca-Cola.", "lifestyle")
    
    # Delete: The user wants to forget the wifi password.
    print("\n[Action] Forgetting Wi-Fi password...")
    delete_memory("wifi_pass")

    print("\n--- PHASE 4: VERIFYING CHANGES ---")
    
    # Test D: Should now return Coca-Cola, completely overwriting the water memory.
    query_memory("What is my favorite drink?", target_domain="lifestyle")
    
    # Test E: Should fail because the wifi password was successfully wiped.
    query_memory("What is the guest wifi password?", target_domain="infrastructure")