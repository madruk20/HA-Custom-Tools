import logging
import aiohttp
import asyncio
from datetime import datetime
import hashlib
import json
import os
import re
import ollama
from voluptuous_openapi import convert
import importlib
import pkgutil
import inspect
from pathlib import Path
from qdrant_client import models

import homeassistant.util.dt as dt_util
from homeassistant.core import HomeAssistant
from homeassistant.components import conversation
from homeassistant.helpers import llm
from homeassistant.util.ssl import get_default_context
from homeassistant.config_entries import ConfigEntry
import homeassistant.helpers.device_registry as dr
import homeassistant.helpers.area_registry as ar

from urllib.parse import urlparse



_LOGGER = logging.getLogger(__name__)

DOMAIN = "ai_tools"

class CustomAIAgent(conversation.ConversationEntity):
    """Custom Conversation Agent mirroring HA's Native Prompt Organization."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        self.hass = hass
        self.entry = entry

        # Setup permanent semantic cache
        self.cache_file = Path(__file__).parent / "semantic_cache.json"
        self.semantic_cache = {}
        
        # Tell the UI what to call this agent in the dropdown menu
        self._attr_name = "Custom AI Agent"

        # Unique ID is required for the gear icon link to work
        self._attr_unique_id = entry.entry_id
        self._attr_supports_streaming = True

        # Sparse embed model
        self.sparse_model = None

        # LLM Models and Connection Settings
        self.ollama_client = None
        self.llm_backend_type = self.entry.options.get("llm_backend_type", "local_ollama")
        self.llm_url = self.entry.options.get("llm_url", "http://localhost:11434")
        self.llm_model = self.entry.options.get("llm_model", "qwen2.5:latest")
        self.llm_api_key = entry.options.get("llm_api_key", "")
        self.max_iterations = int(self.entry.options.get("max_tool_iterations", 5)) # Maxium tool calls before giving up

        # Embedding Models and Connection Settings
        self.embed_backend_type = self.entry.options.get("embed_backend_type", "None")
        self.embed_url = self.entry.options.get("embed_url", self.llm_url)
        self.embed_url_base = self.entry.options.get("embed_url", self.llm_url)
        self.embedding_model = entry.options.get("embedding_model", "nomic-embed-text:latest")
        self.embed_api_key = self.entry.options.get("embed_api_key", self.llm_api_key)
        
        # Vector DB Settings
        self.vector_db_client = None
        self.vector_db_url = self.entry.options.get("vector_db_url", "http://localhost:6333")
        self.vector_db_api = self.entry.options.get("vector_db_api_key", "")
        self.use_rag = (   # Disable RAG if model or embed backend is set to None
            self.embedding_model.lower() != "none" and 
            self.embed_backend_type.lower() != "none"
        )
        
        # Determine model paths
        if self.llm_backend_type == "openai_official":
            self.base_api_url = "https://api.openai.com/v1"
            self.embeddings_url = "https://api.openai.com/v1/embeddings"
        elif self.llm_backend_type == "openai_compatible":
            self.base_api_url = self.llm_url.rstrip("/")
            self.embeddings_url = f"{self.embed_url_base.rstrip('/')}/embeddings"
        else:
            # Standard Local Ollama Native paths
            self.base_api_url = self.llm_url.rstrip("/")
            self.embeddings_url = f"{self.embed_url_base.rstrip('/')}/api/embeddings"
        
        # Tool and State Management
        self._query_cache = {}
        self.history = {}
        self.session_tool_cache = {}
        self.session_timeouts = {}
        self.blacklisted_tools = self.entry.options.get("blacklisted_tools", [])

        # Custom Tool Instantiation
        self.custom_tools = {}

        # Vector Database limits
        self.TOOL_LIMIT = int(self.entry.options.get("tool_injection_limit", 3))
        self.TOOL_COSINE_LIMIT = self.entry.options.get("tool_cosine_threshold", 0.50)
        self.MEMORY_LIMIT = int(self.entry.options.get("memory_injection_limit", 3))
        self.MEMORY_COSINE_LIMIT = self.entry.options.get("memory_cosine_threshold", 0.50)
        self.MEMORY_COLLECTIONS = self.entry.options.get("memory_collections", ["memories_collection"])
        self.MEMORY_ENABLED = self.entry.options.get("enable_memory_injection", True)

    @property
    def supported_languages(self) -> list[str]:
        return ["en"]
        
    @property
    def name(self) -> str:
        return "Custom AI Agent"
    
    def _sync_load_cache(self):
        """Synchronously load the cache (called by executor job)."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self.semantic_cache = json.load(f)
                _LOGGER.info(f"💾 Loaded {len(self.semantic_cache)} cached routines from disk.")
            except Exception as e:
                _LOGGER.warning(f"⚠️ Failed to load semantic cache: {e}")


    def _save_cache(self):
        """Saves the RAM cache to disk so it survives reboots."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.semantic_cache, f)
        except Exception as e:
            _LOGGER.error(f"❌ Failed to save semantic cache to disk: {e}")


    def _sync_load_custom_tools(self):
        """Synchronously import tools (MUST be run in executor thread)."""
        tools_dir = Path(__file__).parent / "tools"
        if not tools_dir.exists():
            _LOGGER.warning("⚠️ Tools directory not found. No custom tools loaded.")
            return

        # Iterate through every Python file in the tools folder dynamically
        for _, module_name, _ in pkgutil.iter_modules([str(tools_dir)]):
            try:
                module = importlib.import_module(f".tools.{module_name}", package=__package__)
                
                # Scan the file for any class that inherits from llm.Tool
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, llm.Tool) and obj is not llm.Tool:
                        tool_instance = obj()
                        self.custom_tools[tool_instance.name] = tool_instance
                        _LOGGER.debug(f"✅ Dynamically loaded custom tool: {tool_instance.name}")
                        
            except Exception as e:
                _LOGGER.error(f"❌ Failed to load custom tool module '{module_name}': {e}")


    async def async_initialize_tools(self):
        """Async wrapper to offload initialization to background threads."""
        
        # 1. ALWAYS load custom tools (independent of cache state)
        await self.hass.async_add_executor_job(self._sync_load_custom_tools)
        
        # 2. Safely attempt to load cache if it exists
        if self.cache_file.exists():
            await self.hass.async_add_executor_job(self._sync_load_cache)
        else:
            # Explicitly reset to empty if file is missing
            self.semantic_cache = {}
            _LOGGER.info("ℹ️ No cache file found. Starting with an empty RAM cache.")
            
    
    def _trim_history_safely(self, history: list, max_messages: int = 40) -> list:
        """Trims history safely without breaking tool call/response chains."""
        if len(history) <= max_messages:
            return history
            
        slice_index = len(history) - max_messages
        # Walk forward to ensure we start at a 'user' message
        while slice_index < len(history) and history[slice_index].get("role") != "user":
            slice_index += 1
            
        return history[slice_index:]
    
    def _fix_invalid_arguments(self, arguments) -> tuple[dict, str]:
        """Sanitize JSON arguments dynamically and return (cleaned_dict, warning_string)."""
        parsed_args = {}
        warning_msg = ""

        if isinstance(arguments, dict):
            parsed_args = arguments
        elif isinstance(arguments, str):
            arguments = arguments.strip().removeprefix("```json").removesuffix("```").strip()
            if arguments.startswith('"') and arguments.endswith('"'):
                try:
                    unquoted = json.loads(arguments)
                    if isinstance(unquoted, dict):
                        parsed_args = unquoted
                except Exception:
                    pass

            if not parsed_args:
                try:
                    match = re.search(r"\{.*\}", arguments, re.DOTALL)
                    if match:
                        parsed_args = json.loads(match.group(0))
                except Exception:
                    _LOGGER.warning(f"Failed to repair JSON arguments: {arguments}")

        if isinstance(parsed_args, dict):
            cleaned_args = {}
            # Generic catch-all for placeholder words local LLMs emit when trying to leave a field blank
            generic_nulls = ["null", "none", "undefined", "true", "false", "na", "n/a", "blank", "empty"]
            captured_warnings = []

            for k, v in parsed_args.items():
                if isinstance(v, str):
                    v_clean = v.lower().strip()
                    # Catch generic string placeholders or if it passes boolean flags as strings
                    if v_clean in generic_nulls:
                        captured_warnings.append(f"Tool parameter '{k}' was called with an incorrect value '{v}'")
                        cleaned_args[k] = "" # Blank it out so the executor loop drops it
                        continue
                cleaned_args[k] = v

            if captured_warnings:
                # Keep the feedback strict, generic, and completely dynamic
                warning_msg = " (System Notice: " + " and ".join(captured_warnings) + ". This parameter has been automatically pruned. Do not pass placeholder strings for optional arguments.)"

            return cleaned_args, warning_msg

        return {}, ""

    def _assemble_and_filter_tools(self, ha_api_instance, unlocked_tool_names):
        master_dict = {}
        for ha_tool in ha_api_instance.tools:
            clean_name = ha_tool.name.replace("assist__", "")
            # Check against the dynamic blacklist
            if clean_name not in self.blacklisted_tools:
                master_dict[ha_tool.name] = ha_tool

        for name, tool_obj in self.custom_tools.items():
            if name not in self.blacklisted_tools:
                master_dict[name] = tool_obj

        # Process bundles only if we are actually filtering tools (Qdrant is active)
        if unlocked_tool_names is not None:
            music_bundle = ["music_player", "MusicPause", "MusicNext", "MusicPrevious"]
            if any(tool in unlocked_tool_names for tool in music_bundle):
                for mt in music_bundle:
                    if mt not in unlocked_tool_names:
                        unlocked_tool_names.append(mt)
                        _LOGGER.debug(f"🔗 [BUNDLE] Auto-unlocked music cluster tool: {mt}")

        active_tools = {}
        tool_schemas = []
        for t_name, t_obj in master_dict.items():
            clean_name = t_name.replace("assist__", "")
            
            # If unlocked_tool_names is None (bypassed), or the tool is in the list, include it
            if unlocked_tool_names is None or clean_name in unlocked_tool_names:
                active_tools[t_name] = t_obj
                ha_schema = convert(t_obj.parameters, custom_serializer=ha_api_instance.custom_serializer)
                tool_schemas.append({
                    "type": "function",
                    "function": {"name": t_name, "description": t_obj.description, "parameters": ha_schema}
                })

        return active_tools, tool_schemas
    

    def _build_system_prompt(self, device_id: str, personal_memories: str, ha_base_prompt: str) -> str:
        """Assembles prompt mirroring HA's native organization based on UI Injection Strategy."""
        location_name = "Unknown"
        active_area_name = ""
        
        dev_reg = dr.async_get(self.hass)
        area_reg = ar.async_get(self.hass)

        # Always try to figure out where the user is physically located
        if device_id:
            if device := dev_reg.async_get(device_id):
                if device.area_id and (area := area_reg.async_get_area(device.area_id)):
                    location_name = area.name
                    active_area_name = area.name.lower().strip()

        # Fetch the device injection strategy from Config Flow
        strategy = self.entry.options.get("device_injection_strategy", "current_room")
        allowed_area_names = []

        if strategy == "current_room" and active_area_name:
            allowed_area_names = [active_area_name]
            ha_base_prompt = ha_base_prompt.replace(
                "Static Context: An overview of the areas and the devices in this smart home:",
                f"Static Context: An overview of the areas and the devices in the {location_name}:"
            )
            
        elif strategy == "specific_rooms":
            specific_room_ids = self.entry.options.get("injection_specific_rooms", [])
            for a_id in specific_room_ids:
                if area := area_reg.async_get_area(a_id):
                    allowed_area_names.append(area.name.lower().strip())
            
            # Make the prompt header reflect that this is a curated list
            ha_base_prompt = ha_base_prompt.replace(
                "Static Context: An overview of the areas and the devices in this smart home:",
                "Static Context: An overview of the devices in the requested rooms:"
            )

        # Apply the Regex filter ONLY if we are using a restrictive strategy and have target areas
        if strategy in ["current_room", "specific_rooms"]:
            pattern = r'-\s+names:[^\n]+\n\s+domain:[^\n]+\n\s+areas:\s+([^\n]+)\n'
            def room_evaluator(match):
                # If the device's area is not in our allowed list, wipe it from the prompt
                if match.group(1).lower().strip() not in allowed_area_names: 
                    return ""
                return match.group(0)
            
            # Only run the filter if there are actually areas to filter by
            if allowed_area_names or strategy == "specific_rooms":
                ha_base_prompt = re.sub(pattern, room_evaluator, ha_base_prompt)
                ha_base_prompt = re.sub(r'\n\s+areas:\s+[^\n]+', '', ha_base_prompt)
                _LOGGER.debug(f"✂️ Applied Regex Room Filter for areas: {allowed_area_names}")

        # Assemble the final static prompt
        static_prefix = self.entry.options.get(
            "Instructions", 
            "You are the conversational brain of a smart home..."
        )
        
        # Assemble final device list
        ha_context = f"\n### HOME ASSISTANT ENTITIES\n{ha_base_prompt}\n"

        # Calculate time variables for formatting
        now = dt_util.now()
        day = now.day
        suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        day_str = f"{now.strftime('%B')} {day}{suffix}, {now.strftime('%Y')}"

        # Fetch the user's custom template from configuration options
        suffix_template = self.entry.options.get("dynamic_suffix", "")

        if suffix_template.strip():
            try:
                # Safely map keys to template placeholders
                rendered_suffix = suffix_template.format(
                    location_name=location_name,
                    day_of_week=now.strftime('%A'),
                    date_str=day_str,
                    current_time=now.strftime('%-I:%M %p')
                )
                dynamic_suffix = f"\n{rendered_suffix}\n"
            except KeyError as err:
                _LOGGER.error(f"❌ Custom dynamic_suffix prompt template contains an invalid placeholder key: {err}. Falling back to default layout.")
                dynamic_suffix = (
                    f"\n### TIME AND LOCATION CONTEXT\n"
                    f"Physical Location: You are physically located in the {location_name}.\n"
                    f"Current Context: Today is {now.strftime('%A')}, {day_str} and the current time is {now.strftime('%-I:%M %p')}.\n"
                )
            except Exception as err:
                _LOGGER.error(f"❌ Failed rendering custom dynamic_suffix template: {err}")
                dynamic_suffix = ""
        else:
            dynamic_suffix = ""

        # Add any relevant memories to the dynamic prompt
        if personal_memories: 
            dynamic_suffix += f"\n### PERSONAL MEMORIES & FACTS\n{personal_memories}\n"
        
        # Final prompt: Static prompt - devices - dynamic prompt elements
        return f"{static_prefix}{ha_context}{dynamic_suffix}"

    def _compress_tool_response(self, result, tool_name: str) -> str:
        """Compresses HA intent responses to save tokens and maintain fine-tuning patterns."""
        # Standardize native Home Assistant intent response objects into standard dictionaries
        if hasattr(result, "as_dict"):
            result_dict = result.as_dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {}

        # Automatically fixes 24-hour time strings for ANY tool returning a 'result' string
        if "result" in result_dict and isinstance(result_dict["result"], str):
            speech_text = result_dict["result"]
            time_matches = re.findall(r"'\d{2}:\d{2}:\d{2}'|\b\d{2}:\d{2}:\d{2}\b", speech_text)
            for raw_timestamp in time_matches:
                clean_ts = raw_timestamp.replace("'", "")
                try:
                    friendly_time = datetime.strptime(clean_ts, "%H:%M:%S").strftime("%I:%M %p")
                    speech_text = speech_text.replace(raw_timestamp, f"{raw_timestamp} ({friendly_time})")
                except ValueError:
                    continue
            
            # Write it back safely to whichever object structure came in
            if hasattr(result, "as_dict"):
                if hasattr(result, "speech") and "plain" in result.speech:
                    result.speech["plain"]["speech"] = speech_text
            else:
                result["result"] = speech_text
                
            _LOGGER.debug(f"⏰ Converted tool response times cleanly: {speech_text}")

        if tool_name in self.custom_tools:
            return json.dumps(result_dict, default=str)
            
        try:
            if "response_type" in result_dict:
                response_type = result_dict.get("response_type")
                
                # Compress Action Responses
                if response_type == "action_done":
                    success_entities = result_dict.get("data", {}).get("success", [])
                    if success_entities:
                        names = [e.get("name", "Unknown") for e in success_entities]
                        return f"Success. Action executed on: {', '.join(names)}"
                    return "Success. Action executed."
                
                # Compress Errors
                elif response_type == "error":
                    return f"Failed. Error code: {result_dict.get('data', {}).get('code', 'unknown')}"
                
                # Extract clean speech from Native Query Responses
                elif response_type == "query_answer":
                    speech = result_dict.get("speech", {}).get("plain", {}).get("speech", "")
                    return speech if speech else json.dumps(result_dict, default=str)
                        
            return json.dumps(result_dict, default=str)
        except Exception as err:
            _LOGGER.error(f"⚠️ Error inside tool compression utility block: {err}")
            return json.dumps(result_dict, default=str)


    async def _ensure_models_loaded(self):
        """Sequentially warm up local ollama models to prevent VRAM spikes."""
        async with aiohttp.ClientSession() as session:
            # Check currently loaded models
            async with session.get(f"{self.llm_url.rstrip('/')}/api/ps") as resp:
                data = await resp.json()
                loaded_models = [m["name"] for m in data.get("models", [])]

                # 1. Warm up LLM if not loaded
                if self.llm_model not in loaded_models:
                    _LOGGER.info(f"⏳ LLM {self.llm_model} not loaded. Warming up...")
                    await self.ollama_client.generate(model=self.llm_model, prompt="hi")
                    await asyncio.sleep(2)  # let it stabilize

                # 2. Warm up Embed Model if not loaded
                if self.embedding_model != "None" and self.embedding_model not in loaded_models:
                    _LOGGER.info(f"⏳ Embed Model {self.embedding_model} not loaded. Warming up...")
                    headers = {"Authorization": f"Bearer {self.embed_api_key}"} if self.embed_api_key else {}
                    payload = {"model": self.embedding_model, "prompt": "warmup"}
                    async with aiohttp.ClientSession() as session:
                        await session.post(self.embeddings_url, json=payload, headers=headers)
                    await asyncio.sleep(1)


    async def _get_vector_db_client(self):
        if self.vector_db_client is not None:
            return self.vector_db_client

        backend = self.entry.options.get("vector_db_backend", "qdrant")
        url = self.entry.options.get("vector_db_url", "http://localhost:6333")
        api_key = self.entry.options.get("qdrant_api_key", "")

        def _create_client():
            if backend == "qdrant":
                from qdrant_client import AsyncQdrantClient
                return AsyncQdrantClient(url=url, api_key=api_key if api_key else None)
            # Add new vector database backends here
            return None

        self.vector_db_client = await self.hass.async_add_executor_job(_create_client)
        return self.vector_db_client


    async def _get_ollama_client(self):
        if self.ollama_client is None:
            _LOGGER.info("Initializing Ollama Client...")
            
            def _create_client():
                return ollama.AsyncClient(
                    host=self.llm_url,
                    headers={"Authorization": f"Bearer {self.llm_api_key}"} if self.llm_api_key else None,
                    verify=get_default_context()
                )
            
            self.ollama_client = await self.hass.async_add_executor_job(_create_client)
            
        return self.ollama_client
    

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """The main conversation turn logic with streaming and vision support."""

        user_query = user_input.text
        session_id = user_input.conversation_id or "default"

        # =================================================================
        # GLOBAL GARBAGE COLLECTOR & IDLE TIMEOUT
        # =================================================================
        current_time = dt_util.utcnow().timestamp()
        
        expired_sessions = []
        for stored_session, last_active in self.session_timeouts.items():
            if (current_time - last_active) > 300: # 5 minutes (300 seconds)
                expired_sessions.append(stored_session)
                
        for exp_session in expired_sessions:
            _LOGGER.info(f"🧹 Sweeping orphaned memory for expired session: {exp_session}")
            self.history.pop(exp_session, None)
            self.session_tool_cache.pop(exp_session, None)
            self.session_timeouts.pop(exp_session, None)
            
        self.session_timeouts[session_id] = current_time
        
        # 1. Fetch API Instance, Tools, and the unique Hash for this query
        ha_api_instance = await self._get_ha_api_instance(user_input)
        
        # Unpack values
        unlocked_tool_names, personal_memories, query_hash = await self._fetch_context(user_query)
        
        # Ensure our session tool cache exists for this conversation
        if session_id not in self.session_tool_cache:
            self.session_tool_cache[session_id] = set()

        # If we have tools stored from the last turn, inject them
        if unlocked_tool_names is not None and self.session_tool_cache[session_id]:
            _LOGGER.info(f"🧠 [SESSION MEMORY] Injecting tools from previous turn: {list(self.session_tool_cache[session_id])}")
            for cached_tool in self.session_tool_cache[session_id]:
                if cached_tool not in unlocked_tool_names:
                    unlocked_tool_names.append(cached_tool)

        # Assemble active execution tools
        active_tools, active_tool_schemas = self._assemble_and_filter_tools(ha_api_instance, unlocked_tool_names)

        # 2. Build System Prompt
        system_prompt = self._build_system_prompt(
            user_input.device_id, 
            personal_memories, 
            ha_api_instance.api_prompt
        )

        # 3. Manage local history
        if session_id not in self.history:
            self.history[session_id] = []

        images = []
        for content in reversed(chat_log.content):
            if isinstance(content, conversation.UserContent):
                for attachment in content.attachments or ():
                    if attachment.mime_type.startswith("image/"):
                        images.append(attachment.path)
                break 

        user_msg = {"role": "user", "content": user_query}
        if images:
            user_msg["images"] = images

        self.history[session_id].append(user_msg)

        max_history = self.entry.options.get("max_history", 40)
        if len(self.history[session_id]) > max_history:
            self.history[session_id] = self.history[session_id][-max_history:]

        messages = [{"role": "system", "content": system_prompt}] + self.history[session_id]

        # 4. Execute the Tool Loop (Passing query_hash and memories for the new Outcome-Based Cache)
        await self._execute_tool_loop_streaming(
            messages, 
            active_tools, 
            active_tool_schemas, 
            ha_api_instance, 
            chat_log, 
            session_id,
            query_hash,
            personal_memories
        )

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
    

    async def _execute_tool_loop_streaming(
            self, 
            messages, 
            active_tools, 
            tool_schemas, 
            ha_api_instance, 
            chat_log: conversation.ChatLog, 
            session_id: str, 
            query_hash: str, 
            personal_memories: str
    ):
        max_iterations = self.max_iterations
        ollama_client = await self._get_ollama_client()
        
        # Track ONLY tools that execute successfully during this turn for caching
        successful_tools = []
        
        # Set Ollama options
        ollama_options = {
            "num_ctx": self.entry.options.get("num_ctx", 32768), 
            "temperature": self.entry.options.get("temperature", 0.5),
            "top_k": self.entry.options.get("top_k", 40),
            "top_p": self.entry.options.get("top_p", 0.9),
            "repeat_penalty": self.entry.options.get("repeat_penalty", 1.1),
            "num_predict": self.entry.options.get("num_predict", 512),
            "mirostat": int(self.entry.options.get("mirostat", 0))
        }   

        selected_model = self.llm_model
        use_thinking = self.entry.options.get("thinking", False)    
        raw_keep_alive = self.entry.options.get("keep_alive", -1)
        keep_alive_val = -1 if raw_keep_alive == -1 else f"{raw_keep_alive}m"

        _LOGGER.info(f"🚀 Attempting Ollama Chat with model: {selected_model}")

        for iteration in range(max_iterations):
            _LOGGER.info(
                f"\n{'='*50}\n"
                f"📤 LLM INFERENCE PAYLOAD DUMP (Iteration {iteration + 1} | Backend: {self.llm_backend_type})\n"
                f"{'='*50}\n"
                f"TOOLS PROVIDED:\n{json.dumps(tool_schemas, indent=2)}\n\n"
                f"MESSAGES PAYLOAD:\n{json.dumps(messages, indent=2)}\n"
                f"{'='*50}\n"
            )

            full_content_out = []
            full_thinking_out = []
            tool_calls_buffer = []
            final_metadata = {}

            # =================================================================
            # BRANCH A: Cloud Providers & OpenAI Compatible Frameworks
            # =================================================================
            if self.llm_backend_type in ["openai_official", "openai_compatible"]:
                # Translate parameters to standard OpenAI endpoint formatting keys
                cloud_payload = {
                    "model": selected_model,
                    "messages": messages,
                    "stream": True,
                    "temperature": self.entry.options.get("temperature", 0.5),
                    "top_p": self.entry.options.get("top_p", 0.9),
                    "max_tokens": self.entry.options.get("num_predict", 512)
                }
                if tool_schemas:
                    cloud_payload["tools"] = tool_schemas

                headers = {"Content-Type": "application/json"}
                if self.llm_api_key:
                    headers["Authorization"] = f"Bearer {self.llm_api_key}"

                ssl_context = get_default_context() if self.chat_url.startswith("https") else False

                async def _transform_cloud_stream():
                    timeout = aiohttp.ClientTimeout(total=60.0)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(self.chat_url, json=cloud_payload, headers=headers, ssl=ssl_context) as resp:
                            if resp.status != 200:
                                err_text = await resp.text()
                                _LOGGER.error(f"❌ Cloud Backend Error Response ({resp.status}): {err_text}")
                                yield {"content": f"Connection error to backend provider: Status {resp.status}"}
                                return

                            # Process Server-Sent Events (SSE) data stream chunks
                            async for line in resp.content:
                                line = line.decode("utf-8").strip()
                                if not line or not line.startswith("data: "):
                                    continue
                                
                                data_str = line.removeprefix("data: ").strip()
                                if data_str == "[DONE]":
                                    break

                                try:
                                    chunk_json = json.loads(data_str)
                                    choices = chunk_json.get("choices", [])
                                    if not choices:
                                        continue
                                        
                                    delta = choices[0].get("delta", {})
                                    out_delta = {}

                                    if raw_reasoning := delta.get("reasoning_content"):
                                        out_delta["thinking_content"] = raw_reasoning
                                        full_thinking_out.append(raw_reasoning)

                                    if raw_content := delta.get("content"):
                                        out_delta["content"] = raw_content
                                        full_content_out.append(raw_content)

                                    if cloud_tools := delta.get("tool_calls"):
                                        for ct in cloud_tools:
                                            idx = ct.get("index", 0)
                                            while len(tool_calls_buffer) <= idx:
                                                tool_calls_buffer.append({"function": {"name": "", "arguments": ""}})
                                            
                                            func_delta = ct.get("function", {})
                                            if name_chunk := func_delta.get("name"):
                                                tool_calls_buffer[idx]["function"]["name"] += name_chunk
                                            if args_chunk := func_delta.get("arguments"):
                                                tool_calls_buffer[idx]["function"]["arguments"] += args_chunk

                                    if out_delta:
                                        yield out_delta

                                except Exception as parse_err:
                                    _LOGGER.debug(f"Skipping unparseable cloud chunk line: {parse_err}")
                                    yield {"content": "I am sorry, but I encountered a connection error with my cloud AI provider."}

                async for _ in chat_log.async_add_delta_content_stream(self.entity_id, _transform_cloud_stream()):
                    pass

            # =================================================================
            # BRANCH B: Native Local Ollama Client Module Routine
            # =================================================================
            else:
                try:
                    response_generator = await ollama_client.chat(
                        model=selected_model, 
                        messages=messages, 
                        tools=tool_schemas if tool_schemas else None, 
                        stream=True,
                        think=use_thinking,
                        options=ollama_options,
                        keep_alive=keep_alive_val          
                    )
                except Exception as e:
                    await chat_log.async_add_delta_content(
                        self.entity_id, 
                        {"role": "assistant", "content": "I am sorry, but I encountered a connection error with my AI engine."}
                    )
                    _LOGGER.error(f"❌ Ollama connection error: {e}")
                    return

                async def _transform_ollama_stream():
                    new_msg = True
                    async for chunk in response_generator:
                        if "eval_count" in chunk or "prompt_eval_count" in chunk:
                            final_metadata.update(chunk)

                        msg = chunk.get("message", {})
                        delta = {}
                        
                        if new_msg:
                            delta["role"] = "assistant"
                            new_msg = False
                        
                        if thinking_trace := msg.get("thinking"):
                            delta["thinking_content"] = thinking_trace
                            full_thinking_out.append(thinking_trace)
                        
                        if content := msg.get("content"):
                            delta["content"] = content
                            full_content_out.append(content)
                            
                        if raw_tool_calls := msg.get("tool_calls"):
                            for tc in raw_tool_calls:
                                tool_calls_buffer.append(tc)

                        if delta and ("content" in delta or "thinking_content" in delta):
                            yield delta

                async for _ in chat_log.async_add_delta_content_stream(self.entity_id, _transform_ollama_stream()):
                    pass

                p_tokens = final_metadata.get("prompt_eval_count", 0)
                p_time = final_metadata.get("prompt_eval_duration", 0) / 1e9
                gen_tokens = final_metadata.get("eval_count", 0)
                total_time = final_metadata.get("total_duration", 0) / 1e9
                speed = f"{(gen_tokens / (total_time - p_time)):.2f} t/s" if (total_time - p_time) > 0 else "N/A"
                _LOGGER.info(f"🤖 OLLAMA METRICS: Prompt: {p_tokens}t | Generated: {gen_tokens}t | Speed: {speed} | Total: {total_time:.2f}s")

            # =================================================================
            # POST-STREAM SYNCHRONIZATION
            # =================================================================
            if full_thinking_out:
                _LOGGER.info(f"🧠 AI THINKING TRACE:\n{''.join(full_thinking_out)}")

            full_content = "".join(full_content_out)
            safe_message = {"role": "assistant", "content": full_content}
            
            if full_thinking_out:
                safe_message["thinking"] = "".join(full_thinking_out)

            if tool_calls_buffer:
                safe_message["tool_calls"] = []
                for tc in tool_calls_buffer:
                    func = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                    name = func.get("name", "") if isinstance(func, dict) else getattr(func, "name", "")
                    args = func.get("arguments", {}) if isinstance(func, dict) else getattr(func, "arguments", {})
                    
                    parsed_args = json.loads(args) if isinstance(args, str) and args.strip() else args
                    
                    safe_message["tool_calls"].append({
                        "function": {
                            "name": name,
                            "arguments": parsed_args
                        }
                    })

            messages.append(safe_message)
            self.history[session_id].append(safe_message)

            if not tool_calls_buffer:
                break
                        
            # Execute tools manually
            for tool_call in safe_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                raw_args_payload = tool_call["function"]["arguments"]
                
                repaired_args, argument_warning = self._fix_invalid_arguments(raw_args_payload)
                args = {k: v for k, v in repaired_args.items() if v is not None and v != ""}

                # Save to session cache for follow-ups
                clean_cache_name = tool_name.replace("assist__", "")
                self.session_tool_cache[session_id].add(clean_cache_name)
                
                _LOGGER.info(f"⚙️ Executing Tool: {tool_name} with args {args}")
                try:
                    ha_tool_input = llm.ToolInput(tool_name=tool_name, tool_args=args)
                    
                    if tool_name in self.custom_tools:
                        result = await self.custom_tools[tool_name].async_call(self.hass, ha_tool_input, ha_api_instance.llm_context)
                    else:
                        result = await ha_api_instance.async_call_tool(ha_tool_input)
                    
                    # TRACK SUCCESS: If we reach here, the tool executed successfully
                    successful_tools.append(clean_cache_name)
                    
                    compressed_content = self._compress_tool_response(result, tool_name)
                    if argument_warning:
                        compressed_content += argument_warning

                    tool_msg = {"role": "tool", "content": compressed_content, "name": tool_name}
                    messages.append(tool_msg)
                    self.history[session_id].append(tool_msg)
                    
                except Exception as e:
                    _LOGGER.error(f"❌ Tool execution exception {tool_name}: {e}")
                    error_msg = {"role": "tool", "content": json.dumps({"error": str(e)}), "name": tool_name}
                    messages.append(error_msg)
                    self.history[session_id].append(error_msg)

        # --- COMMIT TO TOOL CACHE ON SUCCESS ---
        if successful_tools:
            # We use a set to ensure unique tools are saved for this routine
            unique_tools = list(set(successful_tools))
            
            _LOGGER.info(f"💾 [CACHE SUCCESS] Verified routine '{query_hash}'. Saving tools: {unique_tools}")
            
            self.semantic_cache[query_hash] = {
                "tools": unique_tools,
                "memories": personal_memories
            }
            self.hass.async_add_executor_job(self._save_cache)


    async def _fetch_context(self, query: str) -> tuple[list, str, str]:
        """Fetch embeddings and query Qdrant, using a persistent cache for exact matches."""
  
        # --- 1. CHECK THE SEMANTIC CACHE FIRST ---
        clean_query = query.lower().strip()
        query_hash = hashlib.md5(clean_query.encode()).hexdigest()

        # RAG disabled, exit
        if not self.use_rag:
            _LOGGER.info("ℹ️ Embedding model set to 'None'. Bypassing vector search and unlocking all tools.")
            return None, "", query_hash

        # CACHE HIT
        if query_hash in self.semantic_cache:
            cached_data = self.semantic_cache[query_hash]
            _LOGGER.info(f"⚡ [CACHE HIT] Bypassing Vector Search! 0ms Retrieval for routine command.")
            _LOGGER.info(f"⚡ [CACHE HITS] Injected Cached Tools: {cached_data['tools']}")
            return cached_data["tools"], cached_data["memories"], query_hash

        # --- 2. RUN OLLAMA EMBEDDINGS (CACHE MISS) ---
        _LOGGER.info(f"🔍 [VECTOR SEARCH] Requesting embedding for: '{query}'")
        try:
            timeout = aiohttp.ClientTimeout(total=30.0)
            headers = {"Authorization": f"Bearer {self.embed_api_key}"} if self.embed_api_key else {}
            ssl_context = get_default_context() if self.embed_url_base.startswith("https") else False
            raw_keep_alive = self.entry.options.get("keep_alive", -1)
            keep_alive_val = f"{raw_keep_alive}m" if raw_keep_alive != -1 else -1

            async with aiohttp.ClientSession(timeout=timeout) as session:
                if self.embed_backend_type in ["openai_official", "openai_compatible"]:
                    payload = {"model": self.embedding_model, "input": query}
                else: 
                    payload = {"model": self.embedding_model, "prompt": query}
                    if self.embed_backend_type == "local_ollama" and keep_alive_val != -1:
                        payload["keep_alive"] = keep_alive_val

                async with session.post(
                    self.embeddings_url, 
                    json=payload, 
                    headers=headers,
                    ssl=ssl_context
                ) as resp:
                    if resp.status != 200: 
                        _LOGGER.error(f"Embedding request failed with status {resp.status}: {await resp.text()}")
                        return [], "", query_hash
                    data = await resp.json()
                    query_vector = data.get("embedding") or (data.get("data", [{}])[0].get("embedding") if "data" in data else None)
                    if not query_vector: return [], "", query_hash

        except Exception as e:
            _LOGGER.error(f"❌ Failed to connect to Ollama Embeddings: {e}")
            return [], "", query_hash

        client = await self._get_vector_db_client()

        always_allow = ["smart_web_search", "GetLiveContext"]
        unlocked_tools = list(always_allow)

        _LOGGER.info("🔍 [VECTOR SEARCH] Querying Qdrant Database...")
        padded_limit = self.TOOL_LIMIT + len(self.blacklisted_tools)

        dense_query_vector = query_vector

        def build_hybrid_task(collection, limit_val):
            return client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(
                        query=models.Document(text=query, model="qdrant/bm25"),
                        using="keyword_sparse",
                        limit=limit_val,
                    ),
                    models.Prefetch(
                        query=dense_query_vector,
                        using="qwen_dense",
                        limit=limit_val,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit_val,
                with_payload=True
            )

        qdrant_tasks = [
            build_hybrid_task("tools_collection", padded_limit)
        ]
        
        if self.MEMORY_ENABLED:
            for col_name in self.MEMORY_COLLECTIONS:
                qdrant_tasks.append(
                    client.query_points(
                        collection_name=col_name, 
                        query=dense_query_vector, 
                        using="qwen_dense", 
                        limit=self.MEMORY_LIMIT, 
                        with_payload=True
                    )
                )

        try:
            async with asyncio.timeout(5.0):
                db_results = await asyncio.gather(*qdrant_tasks, return_exceptions=True)
        except TimeoutError:
            _LOGGER.error("❌ Qdrant vector search timed out after 5 seconds.")
            return unlocked_tools, "", query_hash

        tool_results = db_results[0]
        memory_results_list = db_results[1:] if self.MEMORY_ENABLED else []

        _LOGGER.info("--- QDRANT TOOL SEARCH SCORES (HYBRID RRF) ---")
        
        if isinstance(tool_results, Exception):
            _LOGGER.error(f"❌ CRITICAL: Failed to query tools_collection: {tool_results}")
        else:
            top_score = tool_results.points[0].score if tool_results.points else 1.0
            dynamic_threshold = top_score * self.TOOL_COSINE_LIMIT 

            valid_tools_injected = 0
            for hit in tool_results.points:
                tool_name = hit.payload.get("tool_id") or hit.payload.get("tool_name")
                clean_name = tool_name.replace("assist__", "")
                
                if clean_name in self.blacklisted_tools:
                    continue
                
                passes_threshold = hit.score >= dynamic_threshold
                _LOGGER.info(f"🛠️ Tool: {tool_name:<25} | RRF Score: {hit.score:.4f} | Pass: {passes_threshold}")
                
                if passes_threshold:
                    if tool_name not in unlocked_tools:
                        unlocked_tools.append(tool_name)
                    valid_tools_injected += 1
                    
                if valid_tools_injected >= self.TOOL_LIMIT:
                    break
        
        found_facts = []
        if self.MEMORY_ENABLED:
            raw_facts = []
            for idx, col_name in enumerate(self.MEMORY_COLLECTIONS):
                col_results = memory_results_list[idx]
                if isinstance(col_results, Exception): continue
                for hit in col_results.points:
                    content = hit.payload.get('content', '')
                    if hit.score >= self.MEMORY_COSINE_LIMIT:
                        raw_facts.append((hit.score, f"- {content}"))

            raw_facts.sort(key=lambda x: x[0], reverse=True)
            found_facts = [fact[1] for fact in raw_facts[:self.MEMORY_LIMIT]]

        memories = "\n".join(found_facts) if found_facts else ""
        final_tools = list(set(unlocked_tools))

        _LOGGER.info(f"💾 [CACHE MISS] This is a new command: '{clean_query}'")
        
        return final_tools, memories, query_hash
    

    async def _execute_tool_loop(
            self, 
            messages, 
            tool_schemas, 
            ha_api_instance, 
            session_id: str, 
            query_hash: str, 
            memories: str):

        max_iterations = self.max_iterations
        # Track ONLY tools that execute successfully during this turn for caching
        successful_tools = []
        
        for iteration in range(max_iterations):
            _LOGGER.info(
                f"\n{'='*50}\n"
                f"📤 OLLAMA PAYLOAD DUMP (Iteration {iteration + 1})\n"
                f"{'='*50}\n"
                f"TOOLS PROVIDED:\n{json.dumps(tool_schemas, indent=2)}\n\n"
                f"MESSAGES PAYLOAD:\n{json.dumps(messages, indent=2)}\n"
                f"{'='*50}\n"
            )

            # Pull settings directly from the UI widget configuration
            ollama_options = {
                "num_ctx": self.entry.options.get("num_ctx", 32768), 
                "temperature": self.entry.options.get("temperature", 0.5),
                "top_k": self.entry.options.get("top_k", 40),
                "top_p": self.entry.options.get("top_p", 0.9),
                "repeat_penalty": self.entry.options.get("repeat_penalty", 1.1),
                "num_predict": self.entry.options.get("num_predict", 512),
                "mirostat": int(self.entry.options.get("mirostat", 0))
            }
            selected_model = self.llm_model
            use_thinking = self.entry.options.get("thinking", False)
            
            raw_keep_alive = self.entry.options.get("keep_alive", -1)
            keep_alive_val = -1 if raw_keep_alive == -1 else f"{raw_keep_alive}m"

            ollama_client = await self._get_ollama_client()

            # 5. Execute
            response = await ollama_client.chat(
                model=selected_model, 
                messages=messages, 
                tools=tool_schemas, 
                stream=True,
                think=use_thinking,
                options=ollama_options,
                keep_alive=keep_alive_val          
            )

            p_tokens = response.get("prompt_eval_count", 0)
            p_time = response.get("prompt_eval_duration", 0) / 1e9
            gen_tokens = response.get("eval_count", 0)
            total_time = response.get("total_duration", 0) / 1e9
            speed = f"{(gen_tokens / (total_time - p_time)):.2f} t/s" if (total_time - p_time) > 0 else "N/A"
            _LOGGER.info(f"🤖 OLLAMA VERBOSE: Prompt: {p_tokens}t | Generated: {gen_tokens}t | Speed: {speed} | Total: {total_time:.2f}s")
            
            raw_message = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
            raw_text = raw_message.get("content", "") if isinstance(raw_message, dict) else getattr(raw_message, "content", "")
            raw_tool_calls = raw_message.get("tool_calls", []) if isinstance(raw_message, dict) else getattr(raw_message, "tool_calls", [])
            
            _LOGGER.info(f"📥 [RAW UNFILTERED OLLAMA RESPONSE]:\n{raw_text}")
            final_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip() if raw_text else ""
            
            safe_message = {"role": "assistant", "content": final_text}

            if raw_tool_calls:
                safe_message["tool_calls"] = []
                for tc in raw_tool_calls:
                    func = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                    name = func.get("name", "") if isinstance(func, dict) else getattr(func, "name", "")
                    args = func.get("arguments", {}) if isinstance(func, dict) else getattr(func, "arguments", {})
                    safe_message["tool_calls"].append({
                        "function": {
                            "name": name,
                            "arguments": args
                        }
                    })
            
            if not safe_message.get("tool_calls"):
                if not final_text and raw_text:
                    _LOGGER.warning("⚠️ Warning: Thinking scrub wiped the entire response. Reverting to safe answer.")
                    safe_message["content"] = "I heard you, but I couldn't process a clear response."
                
                messages.append(safe_message)
                return safe_message["content"]

            messages.append(safe_message)
            
            # Execute tools manually
            for tool_call in safe_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                raw_args_payload = tool_call["function"]["arguments"]
                
                repaired_args, argument_warning = self._fix_invalid_arguments(raw_args_payload)
                args = {k: v for k, v in repaired_args.items() if v is not None and v != ""}
                
                clean_cache_name = tool_name.replace("assist__", "")
                self.session_tool_cache[session_id].add(clean_cache_name)
                
                _LOGGER.info(f"⚙️ Executing Tool: {tool_name} with args {args}")
                
                try:
                    ha_tool_input = llm.ToolInput(tool_name=tool_name, tool_args=args)
                    
                    if tool_name in self.custom_tools:
                        result = await self.custom_tools[tool_name].async_call(self.hass, ha_tool_input, ha_api_instance.llm_context)
                    else:
                        result = await ha_api_instance.async_call_tool(ha_tool_input)
                    
                    # TRACK SUCCESS: If we reach here, the tool executed successfully
                    successful_tools.append(clean_cache_name)
                    
                    try:
                        compressed_content = self._compress_tool_response(result, tool_name)
                    except Exception as compress_err:
                        _LOGGER.error(f"⚠️ Compression routine error inside loop: {compress_err}")
                        compressed_content = str(result)

                    if argument_warning:
                        compressed_content += argument_warning

                    messages.append({"role": "tool", "content": compressed_content, "name": tool_name})
                    
                except Exception as e:
                    _LOGGER.error(f"❌ Tool execution exception {tool_name}: {e}")
                    messages.append({"role": "tool", "content": json.dumps({"error": str(e)}), "name": tool_name})

        # --- COMMIT TO CACHE ON SUCCESS ---
        if successful_tools:
            # We use a set to ensure unique tools are saved for this routine
            unique_tools = list(set(successful_tools))
            _LOGGER.info(f"💾 [CACHE SUCCESS] Verified routine '{query_hash}'. Saving tools: {unique_tools}")
            
            self.semantic_cache[query_hash] = {
                "tools": unique_tools,
                "memories": memories
            }
            # Schedule the save to disk safely
            self.hass.async_add_executor_job(self._save_cache)
            
        return "I had to stop thinking because I used too many tools."


    async def _get_ha_api_instance(self, user_input):
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            language=user_input.language,
            assistant="conversation",
            device_id=user_input.device_id,
        )
        return await llm.async_get_api(self.hass, llm.LLM_API_ASSIST, llm_context)