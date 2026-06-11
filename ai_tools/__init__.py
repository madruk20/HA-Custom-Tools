import aiohttp
import asyncio
import json
import ollama
import os
import logging
import re
from pathlib import Path
import time # Added for execution timing

# Centralized Home Assistant Imports
import homeassistant.util.dt as dt_util
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
import homeassistant.helpers.device_registry as dr
import homeassistant.helpers.area_registry as ar

# Import custom tools
from .tools.web_search_brave import WebSearchTool
from .tools.alarms import AlarmManagerTool
from .tools.price_lookup import PriceLookupTool
# from .tools.memory import MemoryTool
from .tools.music import MusicPlayerTool, register_media_service

from qdrant_client import QdrantClient
qdrant_client = QdrantClient(url="http://192.168.4.23:6333")

_LOGGER = logging.getLogger(__name__)

# ==========================================
# SELF-CONTAINED .ENV PARSER
# ==========================================
CURRENT_DIR = Path(__file__).parent
ENV_FILE_PATH = CURRENT_DIR / ".env"

if ENV_FILE_PATH.exists():
    _LOGGER.info(f"Loading local environment variables from {ENV_FILE_PATH}")
    with open(ENV_FILE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                clean_key = key.strip()
                clean_value = value.strip().strip('"').strip("'")
                os.environ[clean_key] = clean_value
                _LOGGER.debug(f"Loaded ENV VAR: {clean_key}")
else:
    _LOGGER.warning(f"No local .env file found at {ENV_FILE_PATH}")

# Global state to hold our parallel DB fetches
_CURRENT_TURN_CONTEXT = {
    "dynamic_tools": [],
    "personal_memories": "",
    "recently_used": [] # Add this line if it isn't there
}

# --- IN-MEMORY CACHE ---
_QUERY_CACHE = {}

async def stream_wrapper(async_gen):
    async for chunk in async_gen:
        try:
            p_tokens = chunk.get("prompt_eval_count") if isinstance(chunk, dict) else getattr(chunk, "prompt_eval_count", None)
            if p_tokens: 
                p_time = (chunk.get("prompt_eval_duration", 0) if isinstance(chunk, dict) else getattr(chunk, "prompt_eval_duration", 0)) / 1e9
                gen_tokens = chunk.get("eval_count", 0) if isinstance(chunk, dict) else getattr(chunk, "eval_count", 0)
                total_time = (chunk.get("total_duration", 0) if isinstance(chunk, dict) else getattr(chunk, "total_duration", 0)) / 1e9
                speed = f"{(gen_tokens / (total_time - p_time)):.2f} t/s" if (total_time - p_time) > 0 else "N/A"
                _LOGGER.info(f"🤖 OLLAMA VERBOSE: Prompt: {p_tokens}t | Generated: {gen_tokens}t | Speed: {speed} | Total: {total_time:.2f}s")
        except Exception:
            pass
        yield chunk 

# ==========================================
# 1. PARALLEL VECTOR DATABASE ENGINE
# ==========================================
async def fetch_context(user_query: str):
    """
    1. Fetches embedding vector from local Ollama container.
    2. Uses that vector to query Qdrant via the universal query_points() method.
    """
    global _CURRENT_TURN_CONTEXT, _QUERY_CACHE
    
    # Safely clear per-turn variables without wiping cross-turn sticky memory
    _CURRENT_TURN_CONTEXT["dynamic_tools"] = []
    _CURRENT_TURN_CONTEXT["personal_memories"] = ""
    
    # Normalize the string for caching (lowercase, strip whitespace)
    cache_key = user_query.strip().lower()
    
    # --- CACHE INTERCEPT ---
    if cache_key in _QUERY_CACHE:
        _LOGGER.info(f"⚡ [CACHE HIT] Skipping DB fetch. Instantly loading tools for: '{cache_key}'")
        _CURRENT_TURN_CONTEXT["dynamic_tools"] = _QUERY_CACHE[cache_key].copy()
        # Note: If you want to cache memories too, you can expand the dictionary to store both.
        # For now, this just caches the tool routing for maximum speed.
        return
    # -----------------------

    # Configuration for local Ollama
    ollama_url = "http://192.168.4.23:11434/api/embeddings"
    ollama_model = "qwen-embed-2k:latest"
    
    # 1. Generate the embedding using LOCAL Ollama
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"model": ollama_model, "prompt": user_query}
            async with session.post(ollama_url, json=payload) as resp:
                if resp.status != 200:
                    _LOGGER.error(f"Ollama Embedding API failed: {resp.status}")
                    return
                data = await resp.json()
                query_vector = data.get("embedding")
                
                if not query_vector:
                    _LOGGER.error("Ollama returned an empty embedding.")
                    return
    except Exception as e:
        _LOGGER.error(f"❌ Failed to fetch embedding from local Ollama: {e}")
        return

    # 2. Perform Parallel Search
    async def search_collection(col_name, limit):
        return qdrant_client.query_points(
            collection_name=col_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )

    results = await asyncio.gather(
        search_collection("tools_collection", 3),
        search_collection("memories_collection", 5)
    )
    
    tool_results, fact_results = results
    
    # 3. Whitelist Tools
    for hit in tool_results.points:
        if hit.score > 0.50:  
            tool_name = hit.payload.get("tool_id") or hit.payload.get("tool_name")
            
            if tool_name and tool_name not in _CURRENT_TURN_CONTEXT["dynamic_tools"]:
                _CURRENT_TURN_CONTEXT["dynamic_tools"].append(tool_name)
                _LOGGER.info(f"🔓 [UNLOCKED] Tool '{tool_name}' added (Score: {hit.score:.2f})")
    
    # 4. Extract Memories
    found_facts = []
    for hit in fact_results.points:
        if hit.score > 0.50:
            found_facts.append(f"- {hit.payload.get('content', '')}")
            
    if found_facts:
        _CURRENT_TURN_CONTEXT["personal_memories"] = "\n".join(found_facts)

    # --- SAVE TO CACHE ---
    # Store the successful DB lookup for next time
    if _CURRENT_TURN_CONTEXT["dynamic_tools"]:
        _QUERY_CACHE[cache_key] = _CURRENT_TURN_CONTEXT["dynamic_tools"].copy()
        
# ==========================================
# 2. THE MASTER OLLAMA INTERCEPTOR
# ==========================================
def apply_ollama_client_patch():
    if ollama is None:
        return

    try:
        if not hasattr(ollama.AsyncClient, "chat"):
            return
        
        original_chat = ollama.AsyncClient.chat
        
        async def patched_chat(self, model, messages=None, **kwargs):
            
            # 1. NORMALIZE MESSAGES FIRST
            working_messages = []
            if messages:
                for msg in messages:
                    if hasattr(msg, "model_dump"): m_dict = msg.model_dump()
                    elif hasattr(msg, "__dict__"): m_dict = {k: v for k, v in msg.__dict__.items() if v is not None}
                    elif isinstance(msg, dict): m_dict = msg.copy()
                    else: m_dict = dict(msg)
                    working_messages.append(m_dict)
            
            # --- TURN STATE AWARENESS ---
            is_tool_return_turn = False
            user_query = ""
            
            if working_messages:
                if working_messages[-1].get("role") in ["tool", "tool_result"]:
                    is_tool_return_turn = True
                    _LOGGER.info("⏭️ [BYPASS] Tool result turn detected. Skipping DB fetch.")
                
                for msg in reversed(working_messages):
                    if msg.get("role") == "user":
                        user_query = msg.get("content", "")
                        break

            # --- DB FETCH ---
            if user_query and not is_tool_return_turn:
                try:
                    await fetch_context(user_query)
                except Exception as e:
                    _LOGGER.error(f"❌ Could not fetch embedding for vector search: {e}")

            # 2. CONTEXT INJECTOR & SYSTEM SCRUBBER
            for m_dict in working_messages:
                if m_dict.get("role") == "system" and isinstance(m_dict.get("content"), str):
                    sys_text = m_dict["content"]
                    from homeassistant.helpers import llm as ha_llm
                    if hasattr(ha_llm, "DEVICE_CONTROL_TOOL_USAGE_PROMPT"): sys_text = sys_text.replace(ha_llm.DEVICE_CONTROL_TOOL_USAGE_PROMPT, "")
                    if hasattr(ha_llm, "DYNAMIC_CONTEXT_PROMPT"): sys_text = sys_text.replace(ha_llm.DYNAMIC_CONTEXT_PROMPT, "")
                    if hasattr(ha_llm, "DEFAULT_INSTRUCTIONS_PROMPT"): sys_text = sys_text.replace(ha_llm.DEFAULT_INSTRUCTIONS_PROMPT, "")
                    # Strip the rendered time/date injected by Home Assistant
                    sys_text = re.sub(r"\s*Current time is \d{2}:\d{2}:\d{2}\. Today's date is \d{4}-\d{2}-\d{2}\.", "", sys_text)

                    if _CURRENT_TURN_CONTEXT.get("personal_memories"):
                        sys_text += f"\n\nPersonal Facts/Memories:\n{_CURRENT_TURN_CONTEXT['personal_memories']}"

                    m_dict["content"] = sys_text.strip()

                if m_dict.get("role") == "assistant" and isinstance(m_dict.get("content"), str):
                    m_dict.pop("thinking", None)
                    m_dict.pop("thinking_content", None)
                    m_dict["content"] = re.sub(r'<think>.*?</think>', '', m_dict["content"], flags=re.DOTALL).strip()
                            
            # 3. DYNAMIC TOOL FILTER
            if "tools" in kwargs and kwargs["tools"]:
                # Define tools that should ALWAYS bypass the vector DB and go to the LLM
                always_allow_tools = [
                    "smart_web_search", "GetLiveContext"
                ] 
                
                # --- TOOL BUNDLING LOGIC ---
                # Group related tools together. If Qdrant finds ONE, we unlock ALL of them.
                music_bundle = ["music_player", "MusicPause", "MusicNext", "MusicPrevious"]
                
                # If any tool in the music bundle was picked up by the database...
                if any(tool in _CURRENT_TURN_CONTEXT["dynamic_tools"] for tool in music_bundle):
                    # ...inject the rest of the bundle into the allowed list
                    for mt in music_bundle:
                        if mt not in _CURRENT_TURN_CONTEXT["dynamic_tools"]:
                            _CURRENT_TURN_CONTEXT["dynamic_tools"].append(mt)
                            _LOGGER.debug(f"🔗 [BUNDLE] Auto-unlocked related music tool: {mt}")
                # --------------------------------
                
                filtered_tools = []
                
                for t in kwargs["tools"]:
                    t_name = t.get("function", {}).get("name", "")
                    
                    # Keep the tool ONLY if it is permanent OR it is in our dynamic list
                    if t_name in always_allow_tools or t_name in _CURRENT_TURN_CONTEXT["dynamic_tools"]:
                        filtered_tools.append(t)
                    else:
                        _LOGGER.debug(f"🔒 [HIDDEN] Withholding unrequested tool: {t_name}")
                
                kwargs["tools"] = filtered_tools
                tool_names = [t.get("function", {}).get("name") for t in filtered_tools]
                _LOGGER.info(f"🛠️ [VERIFY TOOLS] The LLM is receiving these {len(filtered_tools)} tools: {tool_names}")

            # ==========================================
            # --- DEBUG LOGGING: FULL LLM PAYLOAD ---
            # ==========================================
            try:
                debug_payload = {
                    "model": model,
                    "messages": working_messages,
                }
                # "tools": kwargs.get("tools", []) - Add to debug_payload to see tool schemas sent
                # Use default=str to prevent crashes if HA passes a non-standard object
                formatted_payload = json.dumps(debug_payload, default=str, indent=2)
                _LOGGER.debug(f"📤 FULL PAYLOAD SENT TO OLLAMA:\n{formatted_payload}")
            except Exception as e:
                _LOGGER.error(f"⚠️ Could not format payload for logging: {e}")

            # --- TRANSMIT REQUEST ---
            is_stream = kwargs.get("stream", False)
            response = await original_chat(self, model=model, messages=working_messages, **kwargs)
            
            if not is_stream:
                try:
                    p_tokens = response.get("prompt_eval_count", 0) if isinstance(response, dict) else getattr(response, "prompt_eval_count", 0)
                    gen_tokens = response.get("eval_count", 0) if isinstance(response, dict) else getattr(response, "eval_count", 0)
                    _LOGGER.info(f"🤖 OLLAMA VERBOSE: Prompt: {p_tokens}t | Generated: {gen_tokens}t")
                except Exception: pass 
                return response
            else:
                return stream_wrapper(response)
            
        ollama.AsyncClient.chat = patched_chat
        
    except Exception as e:
        _LOGGER.error(f"❌ Failed to patch Ollama client: {e}")

apply_ollama_client_patch()

# ==========================================
# 3. HA TOOL REGISTRATION (Registers ALL to HA)
# ==========================================
def apply_tool_filter_patch():
    """Registers custom tools to Home Assistant so the Conversation Agent can execute them."""
    try:
        if not hasattr(llm.AssistAPI, "_async_get_tools"):
            return

        original_get_tools = llm.AssistAPI._async_get_tools

        def patched_get_tools(self, llm_context, exposed_entities):
            tools = original_get_tools(self, llm_context, exposed_entities)

            all_tool_names = [t.name for t in tools]
            _LOGGER.info(f"DEBUG: Tools provided by HA to filter: {all_tool_names}")
            
            built_in_blacklist = [
                "HassMediaSearchAndPlay", "HassMediaPause", "HassMediaUnpause",       
                "HassMediaNext", "HassMediaPrevious", "HassSetVolume",          
                "HassSetVolumeRelative", "HassMediaPlayerMute", "HassMediaPlayerUnmute",
                "HassCancelAllTimers", "HassIncreaseTimer", "HassDecreaseTimer", 
                "HassPauseTimer", "HassUnpauseTimer", "GetDateTime" 
            ]
            
            pruned_tools = [t for t in tools if t.name not in built_in_blacklist]
            _LOGGER.debug(f"🧹 Pruned {len(tools) - len(pruned_tools)} native HA media/timer tools.")
            
            # REGISTER ALL CUSTOM TOOLS HERE.
            pruned_tools.extend([
                WebSearchTool(),    
                AlarmManagerTool(), 
                PriceLookupTool(), 
                MusicPlayerTool()   
            ])
            
            return pruned_tools

        llm.AssistAPI._async_get_tools = patched_get_tools
            
    except Exception as e:
        _LOGGER.error(f"❌ Critical failure patching AssistAPI tools: {e}")

apply_tool_filter_patch()

# ==========================================
# 4. LOCATION & PHYSICAL ROOM ISOLATION
# ==========================================
def apply_prompt_text_filter_patch():
    """Strips non-local room entity lines based on HA's registry context."""
    try:
        if not hasattr(llm.AssistAPI, "_async_get_api_prompt"):
            return

        original_get_api_prompt = llm.AssistAPI._async_get_api_prompt

        def patched_get_api_prompt(self, llm_context, exposed_entities):
            base_prompt = original_get_api_prompt(self, llm_context, exposed_entities)
            if not base_prompt or not isinstance(base_prompt, str):
                return base_prompt
            
            base_prompt = base_prompt.replace(
                "Static Context: An overview of the areas and the devices in this smart home:",
                "Static Context: An overview of the areas and the devices in this room:"
            )

            if llm_context and llm_context.device_id:
                try:
                    hass = self.hass if hasattr(self, 'hass') else None
                    if hass:
                        dev_reg = dr.async_get(hass)
                        area_reg = ar.async_get(hass)
                        device = dev_reg.async_get(llm_context.device_id)
                        if device and device.area_id:
                            active_area = area_reg.async_get_area(device.area_id)
                            active_area_name = active_area.name.lower().strip() if active_area else ""
                            
                            if active_area_name:
                                _LOGGER.debug(f"📍 Active area context detected: {active_area_name}")
                                pattern = r'-\s+names:[^\n]+\n\s+domain:[^\n]+\n\s+areas:\s+([^\n]+)\n'
                                def room_evaluator(match):
                                    mentioned_area = match.group(1).lower().strip()
                                    if mentioned_area != active_area_name: return ""
                                    return match.group(0)
                                
                                cleaned_prompt = re.sub(pattern, room_evaluator, base_prompt)
                                cleaned_prompt = re.sub(r'\n\s*\n', '\n', cleaned_prompt)
                                _LOGGER.debug("✂️ Stripped non-local entities from context prompt.")
                                return cleaned_prompt
                except Exception as err:
                    _LOGGER.error(f"❌ Text-based context filtering routine failed: {err}")
            
            return base_prompt

        llm.AssistAPI._async_get_api_prompt = patched_get_api_prompt
            
    except Exception as e:
        _LOGGER.error(f"❌ Critical failure patching AssistAPI prompt string layout: {e}")

apply_prompt_text_filter_patch()


def apply_static_prompt_patch():
    """Replaces voice satellite prompt with contextual location mapping."""
    try:
        if not hasattr(llm.AssistAPI, "_async_get_voice_satellite_area_prompt"):
            return
            
        def patched_area_prompt(self, llm_context):
            location_name = "Unknown"
            if llm_context.device_id:
                try:
                    dev_reg = dr.async_get(self.hass)
                    area_reg = ar.async_get(self.hass)
                    device = dev_reg.async_get(llm_context.device_id)
                    if device and device.area_id:
                        area = area_reg.async_get_area(device.area_id)
                        if area: location_name = area.name
                except Exception:
                    pass

            now = dt_util.now()
            day = now.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            day_str = f"{now.strftime('%B')} {day}{suffix}, {now.strftime('%Y')}"

            return (
                f"Physical Location: You are physically located in the {location_name}. If the request does not state a room and one is required, default to this room.\n"
                f"If the user requests context or operations for an entity outside of the {location_name}, you MUST call `assist__GetLiveContext` for that specific area first to discover it.\n"
                f"Current Context: Today is {now.strftime('%A')}, {day_str} and the current time is {now.strftime('%-I:%M %p')}."
            )

        llm.AssistAPI._async_get_voice_satellite_area_prompt = patched_area_prompt
            
    except Exception:
        pass

apply_static_prompt_patch()


async def async_setup(hass: HomeAssistant, config: dict):
    register_media_service(hass)
    _LOGGER.info("✅ Custom Custom Component Setup Complete.")
    return True