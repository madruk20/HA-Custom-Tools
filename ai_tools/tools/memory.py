import uuid
import logging
import asyncio
import aiohttp
import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

_LOGGER = logging.getLogger(__name__)

class MemoryTool(llm.Tool):
    """Vector Database Tool for Long-Term Context and Memory."""
    name = "smart_home_memory"
    description = (
        "Save or retrieve long-term memories, facts, schedules, logs, or large documents like recipes. "
        "Use 'save' to store new information. Use 'retrieve' to search for answers."
    )
    
    parameters = vol.Schema({
        vol.Required(
            "action", 
            description="The operation to perform: 'save' to store a memory, 'retrieve' to search for one, 'forget' to delete a memory."
        ): vol.In(["save", "retrieve", "forget"]),
        
        vol.Required(
            "category", 
            description=(
                "The domain of the memory. "
                "'infrastructure' (IT, servers, IPs, energy, solar, passwords), "
                "'security' (cameras, doors, locks, motion, deliveries), "
                "'lifestyle' (recipes, preferences, schedules, hobbies)."
            )
        ): vol.In(["infrastructure", "security", "lifestyle"]),
        
        vol.Required(
            "payload", 
            description="The exact text to save, delete, or the question/keyword to search for."
        ): str
    })

    def __init__(self):
        self._client = None
        self._collection = None

    def _init_chroma(self, hass: HomeAssistant):
        """Synchronous setup for ChromaDB, run in an executor to avoid blocking the event loop."""

        logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

        # 1. Trick Python into thinking onnxruntime is loaded and functional
        import sys
        from unittest.mock import MagicMock
        sys.modules['onnxruntime'] = MagicMock()

        # --- THE NUMPY 2.0 MONKEY PATCH ---
        import numpy as np
        if not hasattr(np, 'float_'): np.float_ = np.float64
        if not hasattr(np, 'int_'): np.int_ = np.int64
        if not hasattr(np, 'complex_'): np.complex_ = np.complex128
        np.NaN = np.nan

        import chromadb

        if self._collection is None:
            _LOGGER.debug("Initializing ChromaDB connection...")
            try:
                db_path = hass.config.path("custom_components/ai_tools/chroma_db")
                self._client = chromadb.PersistentClient(
                    path=db_path,
                    settings=chromadb.Settings(anonymized_telemetry=False)
                )
                self._collection = self._client.get_or_create_collection(
                    name="ha_master_memory",
                    metadata={"hnsw:space": "cosine"}
                )
                _LOGGER.info(f"ChromaDB successfully initialized at {db_path}")
            except Exception as e:
                _LOGGER.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
                raise

    async def _get_embedding(self, text: str) -> list[float]:
        """Fetches vector coordinates asynchronously from your local Unraid Ollama container."""
        url = "http://192.168.4.23:11434/api/embeddings"
        payload = {"model": "mxbai-embed-large", "prompt": text}
        
        _LOGGER.debug(f"Requesting embedding for text (length: {len(text)})")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10.0) as response:
                    if response.status == 200:
                        data = await response.json()
                        _LOGGER.debug("Embedding successfully received from Ollama.")
                        return data.get("embedding", [])
                    else:
                        error_text = await response.text()
                        _LOGGER.error(f"Ollama HTTP Error {response.status}: {error_text}")
                        return []
        except asyncio.TimeoutError:
            _LOGGER.error("Ollama Embedding request timed out after 10 seconds.")
            return []
        except Exception as e:
            _LOGGER.error(f"Ollama Embedding Connection Failed: {e}", exc_info=True)
            return []

    def _chunk_text(self, text: str, max_words: int = 250, overlap: int = 50) -> list[str]:
        """Slices large text payloads into overlapping blocks."""
        words = text.split()
        if len(words) <= max_words:
            return [text]
        
        chunks, start = [], 0
        while start < len(words):
            end = start + max_words
            chunks.append(" ".join(words[start:end]))
            start += (max_words - overlap)
        return chunks

    def _sync_save_memory(self, parent_id: str, payload: str, category: str, vectors: dict):
        """Saves memory with Search-and-Replace deduplication."""
        word_count = len(payload.split())
        
        try:
            # 1. Search for existing memory in this category
            existing = self._collection.query(
                query_embeddings=[vectors["parent"]],
                n_results=1,
                where={"domain": category}
            )
            
            # 2. SAFE CHECK: Ensure the inner lists actually contain data
            # Check if outer list exists AND inner list has at least one item
            is_match = False
            if (existing['ids'] and len(existing['ids'][0]) > 0 and 
                existing['distances'] and len(existing['distances'][0]) > 0):
                
                # Now it is safe to access the distance
                if existing['distances'][0][0] < 0.15:
                    target_id = existing['ids'][0][0]
                    _LOGGER.info(f"Existing memory found for '{category}'. Updating ID: {target_id}")
                    is_match = True
            
            if not is_match:
                target_id = parent_id
                _LOGGER.info(f"No existing memory found. Creating new ID: {target_id}")

            # 3. Save Logic
            if word_count <= 300:
                self._collection.upsert(
                    ids=[target_id],
                    embeddings=[vectors["parent"]],
                    metadatas=[{"domain": category, "storage_mode": "standard"}],
                    documents=[payload]
                )
            else:
                self._collection.upsert(
                    ids=[target_id],
                    embeddings=[vectors["parent"]],
                    metadatas=[{"domain": category, "storage_mode": "payload_attached", "full_text": payload}],
                    documents=[f"Anchor: {payload[:100]}..."]
                )
                chunks = self._chunk_text(payload)
                for index, chunk in enumerate(chunks):
                    chunk_id = f"{target_id}_chunk_{index}"
                    chunk_vec = vectors["chunks"].get(f"{parent_id}_chunk_{index}")
                    if chunk_vec:
                        self._collection.upsert(
                            ids=[chunk_id],
                            embeddings=[chunk_vec],
                            metadatas=[{"domain": category, "storage_mode": "chunked", "parent_id": target_id}],
                            documents=[chunk]
                        )
            return f"Successfully saved memory to {category} domain."
            
        except Exception as e:
            _LOGGER.error(f"Database write failed: {e}", exc_info=True)
            return f"Error: {str(e)}"
        
    def _sync_forget_memory(self, query_vector: list[float], category: str):
        """Surgically delete only the single best match."""
        try:
            # Change n_results to 1 so we only delete the exact memory match
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=1, 
                where={"domain": category}
            )
            
            if results['ids'] and results['ids'][0]:
                target_id = results['ids'][0][0]
                dist = results['distances'][0][0]
                
                # Verify confidence
                if (1.0 - dist) > 0.60:
                    self._collection.delete(ids=[target_id])
                    _LOGGER.info(f"Surgically deleted memory {target_id} from {category}.")
                    return f"I have forgotten that specific memory."
            
            return "I couldn't find a memory that matches that well enough to delete it."
        except Exception as e:
            return f"Error: {str(e)}"

    def _sync_query_memory(self, query_vector: list[float], category: str, top_k: int = 1, min_confidence: float = 0.60):
        """Synchronous database search."""
        _LOGGER.info(f"Executing database search in '{category}' domain.")
        
        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where={"domain": category}
            )
            
            # SAFE CHECK: Ensure inner list exists
            if results['ids'] and results['ids'][0] and results['distances'] and results['distances'][0]:
                distance = results['distances'][0][0]

            if results['ids'] and results['ids'][0]:
                distance = results['distances'][0][0]
                confidence = 1.0 - distance
                _LOGGER.debug(f"Top match retrieved: Distance={distance:.4f}, Confidence={confidence:.4f}")
                
                if confidence >= min_confidence:
                    metadata = results['metadatas'][0][0]
                    mode = metadata.get("storage_mode", "standard")
                    
                    _LOGGER.info(f"Match accepted. Confidence: {confidence:.2f}, Storage Mode: {mode}")
                    
                    # Unpack hidden payloads automatically if it hit an anchor
                    if mode == "payload_attached":
                        _LOGGER.debug("Unpacking hidden payload attached to anchor vector.")
                        return f"[Match Found: {confidence:.2f}] {metadata.get('full_text')}"
                    else:
                        return f"[Match Found: {confidence:.2f}] {results['documents'][0][0]}"
                else:
                    _LOGGER.info(f"Match rejected. Highest confidence ({confidence:.2f}) was below threshold ({min_confidence}).")
                    return f"No memories found in '{category}' related to the query. (Best match confidence was too low: {confidence:.2f})"
                    
            _LOGGER.info(f"Database returned empty results for domain '{category}'.")
            return f"Database is currently empty for the '{category}' category."
            
        except Exception as e:
            _LOGGER.error(f"Database read failed during query operation: {e}", exc_info=True)
            return f"Database error occurred while querying: {str(e)}"

    async def async_call(self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext) -> dict:
        """Main entry point for the LLM."""
        args = tool_input.tool_args
        action = args.get("action")
        category = args.get("category")
        payload = args.get("payload")
        
        _LOGGER.info(f"MemoryTool triggered | Action: '{action}' | Category: '{category}'")
        _LOGGER.debug(f"Payload Content: {payload}")
        
        # Ensure ChromaDB is initialized securely in a background thread
        await hass.async_add_executor_job(self._init_chroma, hass)

        if action == "save":
            parent_id = f"mem_{uuid.uuid4().hex[:8]}"
            vectors = {"chunks": {}}
            
            # Fetch vectors async so we don't block HA
            vectors["parent"] = await self._get_embedding(payload)
            if not vectors["parent"]:
                _LOGGER.error("Save aborted: Could not retrieve parent embedding.")
                return {"error": "Failed to connect to Ollama embedding model."}
                
            word_count = len(payload.split())
            if word_count > 300:
                _LOGGER.debug(f"Pre-fetching {word_count} words worth of chunk embeddings...")
                chunks = self._chunk_text(payload)
                for index, chunk in enumerate(chunks):
                    chunk_vector = await self._get_embedding(chunk)
                    if chunk_vector:
                        vectors["chunks"][f"{parent_id}_chunk_{index}"] = chunk_vector

            # Execute the database write in a background thread
            result_msg = await hass.async_add_executor_job(
                self._sync_save_memory, parent_id, payload, category, vectors
            )
            return {"result": result_msg}

        elif action == "retrieve":
            query_vector = await self._get_embedding(payload)
            if not query_vector:
                _LOGGER.error("Retrieve aborted: Could not retrieve query embedding.")
                return {"error": "Failed to connect to Ollama embedding model."}
                
            # Execute the database search in a background thread
            result_msg = await hass.async_add_executor_job(
                self._sync_query_memory, query_vector, category
            )
            return {"result": result_msg}
        
        elif action == "forget":
            vec = await self._get_embedding(payload)
            result = await hass.async_add_executor_job(self._sync_forget_memory, vec, category)
            return {"result": result}
            
        _LOGGER.warning(f"Invalid action received from LLM: {action}")
        return {"error": "Invalid action. Must be 'save' or 'retrieve'."}