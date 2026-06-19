# Custom Conversation Agent for Home Assistant

A customizable conversational AI agentfor Home Assistant. This integration focuses on local LLM orchestration, dynamic tool injection via Vector RAG, and fast semantic caching. 

## ✨ Core Features

* **Dynamic Tool Injection (RAG):** Home Assistant has dozens of native tools, which easily overwhelm local models and bloat the context window. This agent uses Qdrant and dense/sparse embeddings (Hybrid RRF) to search and inject *only* the tools necessary for the user's specific request.
* **Semantic Routine Caching:** Frequently used commands bypass the vector search entirely. The agent hashes the query, remembers exactly which tools were needed last time, and executes with near-zero retrieval latency.
* **Drop-in Custom Tools:** Write your own Python tools (like custom music players or web searchers) and drop them into the `tools` folder. They are dynamically loaded and injected into the AI's toolchain on boot.
* **Multimodal Streaming:** Full support for real-time streaming responses and vision models (processing images attached in the chat).
* **Background Automation Engine:** Exposes a custom `AITaskEntity` allowing you to trigger background data generation or JSON structuring via standard Home Assistant automations.

---

## ⚙️ Configuration Menu Guide

The configuration is split into an easy-to-navigate menu within Home Assistant's UI.

### 1. LLM Endpoint & Models
* **Backend Type:** Choose between local Ollama or an OpenAI-compatible proxy (note: using the compatible proxy setting allows you to use standard API formats locally without relying on external cloud services).
* **URL & API Key:** Point this to your inference server.
* **Model Selection:** Dynamically fetches and lists the active models currently loaded on your server (ollama only).

### 2. Context & Parameters
* **System Instructions:** The base prompt.
* **Generation Tweaks:** Sliders to adjust `temperature`, `top_p`, `num_predict`, and the total context window size (`num_ctx`).
* **Keep Alive:** Configure how long the model stays loaded in VRAM after a request.

### 3. Vector Database & Embeddings
* **Embed Backend & Model:** Select the embedding model used to map user queries (e.g., `qwen-embed` or `nomic-embed-text`).
* **Qdrant URL:** The address of your local Qdrant vector database used for storing tool schemas and personal memory facts.

### 4. Tool Management & Caching
* **Tool Blacklists:** Select any native or custom tools you want to strictly hide from the AI.
* **Injection Limits & Thresholds:** Fine-tune the cosine similarity thresholds. This dictates how confident the vector search must be before injecting a tool or memory into the prompt.
* **Clear Semantic Cache:** A toggle to instantly wipe the saved RAM/disk cache if the AI learns a bad tool routine.

---

## 🧠 How the Code Works (Architecture)

### The Execution Pipeline
When a user speaks to the agent, the request goes through a strict pipeline:

1. **The Hash Check:** The user's query is hashed and checked against the internal cache. If it's a routine command (e.g., "Turn off the lights"), the agent skips the vector database and immediately loads the known required tools.
2. **The Vector Search (RAG):** If it's a new command, the query is embedded and sent to Qdrant. Qdrant performs a hybrid search (BM25 keyword + dense vector) across the `tools_collection` and `memories_collection`.
3. **Prompt Assembly:** The system prompt is built dynamically. It injects the base instructions, filters the live Home Assistant device state based on the user's current room, and attaches the RAG-retrieved personal memories.
4. **The LLM Loop:** The AI is given the prompt and the optimized list of tools. It can execute multiple tools in a loop (up to 5 default iterations, adjustable). 
5. **Tool Compression:** Native Home Assistant tool responses are notoriously verbose. The code strips out the junk data and passes clean, minimal JSON back to the LLM to save tokens and prevent local models from getting confused.
6. **Streaming:** The final response is streamed back to the UI or voice satellite in real-time.

---

### Dynamic Overrides
The integration patches Home Assistant's core prompt mechanisms. It blanks out HA's native system prompts (e.g. `DEFAULT_INSTRUCTIONS_PROMPT = ""`) so that the agent has 100% complete control over the context window, ensuring local models aren't distracted by boilerplate text.
