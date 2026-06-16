import logging
import aiohttp
import asyncio
from datetime import datetime
import json
import re
import ollama
from voluptuous_openapi import convert

import homeassistant.util.dt as dt_util
from homeassistant.core import HomeAssistant
from homeassistant.components import conversation
from homeassistant.helpers import llm
from homeassistant.util.ssl import get_default_context
from homeassistant.config_entries import ConfigEntry
import homeassistant.helpers.device_registry as dr
import homeassistant.helpers.area_registry as ar

from qdrant_client import AsyncQdrantClient

# Import your custom tools
from .tools.web_search_brave import WebSearchTool
from .tools.alarms import AlarmManagerTool
from .tools.price_lookup import PriceLookupTool
from .tools.music import MusicPlayerTool
from .tools.camera_stream import CameraStreamTool

_LOGGER = logging.getLogger(__name__)

DOMAIN = "ai_tools"

BUILT_IN_BLACKLIST = [
    "HassGetState","HassMediaSearchAndPlay", "HassMediaPause", "HassMediaUnpause",       
    "HassMediaNext", "HassMediaPrevious", "HassSetVolume",          
    "HassSetVolumeRelative", "HassMediaPlayerMute", "HassMediaPlayerUnmute",
    "HassCancelAllTimers", "HassIncreaseTimer", "HassDecreaseTimer", 
    "HassPauseTimer", "HassUnpauseTimer", "GetDateTime"
]

class CustomAIAgent(conversation.ConversationEntity):
    """Custom Conversation Agent mirroring HA's Native Prompt Organization."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        self.hass = hass
        self.entry = entry
        
        # Tell the UI what to call this agent in the dropdown menu
        self._attr_name = "Custom AI Agent"

        # Unique ID is required for the gear icon link to work
        self._attr_unique_id = entry.entry_id
        self._attr_supports_streaming = True

        # LLM Models and Connection Settings
        self.ollama_client = None
        self.ollama_url = self.entry.options.get("ollama_url", "http://192.168.4.23:11434")
        self.ollama_model = self.entry.options.get("ollama_model", "qwen2.5:latest")
        self.llm_api_key = entry.options.get("llm_api_key", "")
        self.ollama_url_embeddings = f"{self.ollama_url.rstrip('/')}/api/embeddings"
        self.max_iterations = 5 # Maxium tool calls before giving up

        # Embedding Models and Connection Settings
        self.qdrant_client = None
        self.embedding_model = entry.options.get("embedding_model", "None")
        self.qdrant_api_key = entry.options.get("qdrant_api_key", "")
        self.use_rag = self.embedding_model.lower() != "none"
        self.qdrant_url = entry.options.get("qdrant_url", "http://192.168.4.23:6333")

        # Route endpoints dynamically
        self.backend_type = self.entry.options.get("backend_type", "local_ollama")
        if self.backend_type == "openai_official":
            self.base_api_url = "https://api.openai.com/v1"
            self.embeddings_url = f"{self.base_api_url}/embeddings"
        elif self.backend_type == "openai_compatible":
            self.base_api_url = self.ollama_url.rstrip("/")
            self.embeddings_url = f"{self.base_api_url}/embeddings"
        else:
            # Standard Local Ollama Native paths
            self.base_api_url = self.ollama_url.rstrip("/")
            self.embeddings_url = f"{self.base_api_url}/api/embeddings"
        
        # State Management
        self._query_cache = {}
        self.history = {}
        self.session_tool_cache = {}

        # Custom Tool Instantiation
        self.custom_tools = {
            "smart_web_search": WebSearchTool(),
            "alarm_manager": AlarmManagerTool(),
            "stock_and_retail_price_lookup": PriceLookupTool(),
            "music_player": MusicPlayerTool(),
            "stream_camera_to_tv": CameraStreamTool(),
        }

        # Vector Database limits
        self.TOOL_LIMIT = 3
        self.MEMORY_LIMIT = 3

    @property
    def supported_languages(self) -> list[str]:
        return ["en"]
        
    @property
    def name(self) -> str:
        return "Custom AI (Local RAG)"
    
    def _trim_history_safely(self, history: list, max_messages: int = 40) -> list:
        """Trims history safely without breaking tool call/response chains."""
        if len(history) <= max_messages:
            return history
            
        slice_index = len(history) - max_messages
        # Walk forward to ensure we start at a 'user' message
        while slice_index < len(history) and history[slice_index].get("role") != "user":
            slice_index += 1
            
        return history[slice_index:]
    

    async def _get_qdrant_client(self):
        if self.qdrant_client is None:
            _LOGGER.info("Initializing Qdrant Client...")
            
            def _create_client():
                # Pass the api_key parameter (it safely ignores it if it's an empty string)
                return AsyncQdrantClient(
                    url=self.qdrant_url, 
                    api_key=self.qdrant_api_key if self.qdrant_api_key else None,
                    timeout=10.0
                )
            
            self.qdrant_client = await self.hass.async_add_executor_job(_create_client)
            
        return self.qdrant_client

    async def _get_ollama_client(self):
        if self.ollama_client is None:
            _LOGGER.info("Initializing Ollama Client...")
            
            def _create_client():
                return ollama.AsyncClient(
                    host=self.ollama_url,
                    headers={"Authorization": f"Bearer {self.llm_api_key}"} if self.llm_api_key else None,
                    verify=get_default_context()
                )
            
            self.ollama_client = await self.hass.async_add_executor_job(_create_client)
            
        return self.ollama_client
    

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


    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """The main conversation turn logic with streaming and vision support."""
        user_query = user_input.text
        session_id = user_input.conversation_id or "default"
        
        # 1. Fetch API Instance and Tools
        ha_api_instance = await self._get_ha_api_instance(user_input)
        unlocked_tool_names, personal_memories = await self._fetch_context(user_query)
        
        # Ensure our session tool cache exists for this conversation
        if session_id not in self.session_tool_cache:
            self.session_tool_cache[session_id] = set()

        # If we have tools stored from the last turn, inject them into our unlocked tool list
        if unlocked_tool_names is not None and self.session_tool_cache[session_id]:
            _LOGGER.info(f"🧠 [SESSION MEMORY] Injecting tools from previous turn: {list(self.session_tool_cache[session_id])}")
            for cached_tool in self.session_tool_cache[session_id]:
                if cached_tool not in unlocked_tool_names:
                    unlocked_tool_names.append(cached_tool)

        # Assemble active execution tools using the combined list
        active_tools, active_tool_schemas = self._assemble_and_filter_tools(ha_api_instance, unlocked_tool_names)

        # 2. Build System Prompt
        system_prompt = self._build_system_prompt(
            user_input.device_id, 
            personal_memories, 
            ha_api_instance.api_prompt
        )

        # 3. Manage local history for the conversation thread
        if session_id not in self.history:
            self.history[session_id] = []

        # Find any images attached to the current user input in the HA chat log
        images = []
        for content in reversed(chat_log.content):
            if isinstance(content, conversation.UserContent):
                for attachment in content.attachments or ():
                    if attachment.mime_type.startswith("image/"):
                        images.append(attachment.path)
                break  # Stop after the most recent user message

        user_msg = {"role": "user", "content": user_query}

        if images:
            user_msg["images"] = images

        self.history[session_id].append(user_msg)

        # Truncate history if it gets too long
        max_history = self.entry.options.get("max_history", 40)
        if len(self.history[session_id]) > max_history:
            self.history[session_id] = self.history[session_id][-max_history:]

        messages = [{"role": "system", "content": system_prompt}] + self.history[session_id]

        # 4. Execute the Tool Loop with Streaming
        await self._execute_tool_loop_streaming(messages, active_tools, active_tool_schemas, ha_api_instance, chat_log, session_id)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
    

    async def _execute_tool_loop_streaming(self, messages, active_tools, tool_schemas, ha_api_instance, chat_log: conversation.ChatLog, session_id: str):
        max_iterations = 5
        ollama_client = await self._get_ollama_client()
        
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

        selected_model = self.ollama_model
        use_thinking = self.entry.options.get("thinking", False)    
        raw_keep_alive = self.entry.options.get("keep_alive", -1)
        keep_alive_val = -1 if raw_keep_alive == -1 else f"{raw_keep_alive}m"

        _LOGGER.info(f"🚀 Attempting Ollama Chat with model: {selected_model}")

        for iteration in range(max_iterations):
            _LOGGER.info(
                f"\n{'='*50}\n"
                f"📤 OLLAMA PAYLOAD DUMP (Iteration {iteration + 1})\n"
                f"{'='*50}\n"
                f"TOOLS PROVIDED:\n{json.dumps(tool_schemas, indent=2)}\n\n"
                f"MESSAGES PAYLOAD:\n{json.dumps(messages, indent=2)}\n"
                f"{'='*50}\n"
            )

            try:
                response_generator = await ollama_client.chat(
                    model=selected_model, 
                    messages=messages, 
                    tools=tool_schemas, 
                    stream=True,
                    think=use_thinking,
                    options=ollama_options,
                    keep_alive=keep_alive_val          
                )
            except Exception as e:
                _LOGGER.error(f"❌ Ollama connection error: {e}")
                return

            full_content_out = []
            tool_calls_buffer = []
            full_thinking_out = []
            
            # Placeholder dictionary to capture final metadata from the stream
            final_metadata = {}

            # Define a tracking flag outside the generator to catch reasoning states
            self.in_thinking_block = False

            async def _transform_stream():
                new_msg = True
                async for chunk in response_generator:
                    if "eval_count" in chunk or "prompt_eval_count" in chunk:
                        final_metadata.update(chunk)

                    msg = chunk.get("message", {})
                    delta = {}
                    
                    if new_msg:
                        delta["role"] = "assistant"
                        new_msg = False
                    
                    # 1. Route the thinking trace to Home Assistant's native parameter
                    if thinking_trace := msg.get("thinking"):
                        delta["thinking_content"] = thinking_trace
                        full_thinking_out.append(thinking_trace)
                    
                    # 2. Route the clean verbal content 
                    if content := msg.get("content"):
                        delta["content"] = content
                        full_content_out.append(content)
                        
                    if raw_tool_calls := msg.get("tool_calls"):
                        for tc in raw_tool_calls:
                            tool_calls_buffer.append(tc)

                    # Yield the delta if it contains either speech or thoughts
                    if delta and ("content" in delta or "thinking_content" in delta):
                        yield delta

            # Stream text to the UI
            async for _ in chat_log.async_add_delta_content_stream(self.entity_id, _transform_stream()):
                pass

            # 4. Log the thinking trace so you can actually read what it thought!
            if full_thinking_out:
                _LOGGER.info(f"🧠 AI THINKING TRACE:\n{''.join(full_thinking_out)}")

            # --- LLM TOKEN METRICS ---
            p_tokens = final_metadata.get("prompt_eval_count", 0)
            p_time = final_metadata.get("prompt_eval_duration", 0) / 1e9
            gen_tokens = final_metadata.get("eval_count", 0)
            total_time = final_metadata.get("total_duration", 0) / 1e9
            speed = f"{(gen_tokens / (total_time - p_time)):.2f} t/s" if (total_time - p_time) > 0 else "N/A"
            _LOGGER.info(f"🤖 OLLAMA VERBOSE: Prompt: {p_tokens}t | Generated: {gen_tokens}t | Speed: {speed} | Total: {total_time:.2f}s")

            full_content = "".join(full_content_out)
            
            # Reconstruct the safe message payload for the LLM history
            safe_message = {"role": "assistant", "content": full_content}
            if tool_calls_buffer:
                safe_message["tool_calls"] = []
                for tc in tool_calls_buffer:
                    func = tc.get("function", {})
                    safe_message["tool_calls"].append({
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", {})
                        }
                    })
            
            # Save assistant response to both our current request payload and persistent history
            messages.append(safe_message)
            self.history[session_id].append(safe_message)

            # If no tools were called, the conversation turn is complete
            if not tool_calls_buffer:
                # If the AI didn't use tools on this turn, we clear the cache so 
                # unrelated follow-ups don't hold stale tool permissions forever.
                self.session_tool_cache[session_id].clear()
                break
            
            # Execute tools manually
            # Clear previous cache to make room for ONLY the tools executed in this active turn
            self.session_tool_cache[session_id].clear()
            
            # Execute tools manually
            for tool_call in safe_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                raw_args = tool_call["function"]["arguments"]
                
                # Unpack both the repaired dictionary and the dynamic warning string
                repaired_args, argument_warning = self._fix_invalid_arguments(raw_args)
                # Prune out the keys that _fix_invalid_arguments just blanked out
                args = {k: v for k, v in repaired_args.items() if v is not None and v != ""}

                # --- SAVE TO SESSION CACHE ---
                # Strip 'assist__' if it exists so it maps back cleanly into vector database namespaces
                clean_cache_name = tool_name.replace("assist__", "")
                self.session_tool_cache[session_id].add(clean_cache_name)
                
                _LOGGER.info(f"⚙️ Executing Tool: {tool_name} with args {args}")
                try:
                    ha_tool_input = llm.ToolInput(tool_name=tool_name, tool_args=args)
                    
                    if tool_name in self.custom_tools:
                        result = await self.custom_tools[tool_name].async_call(self.hass, ha_tool_input, ha_api_instance.llm_context)
                    else:
                        result = await ha_api_instance.async_call_tool(ha_tool_input)
                    
                    try:
                        compressed_content = self._compress_tool_response(result, tool_name)
                    except Exception as compress_err:
                        _LOGGER.error(f"⚠️ Compression routine error: {compress_err}")
                        compressed_content = str(result)
                    
                    # If an argument was stripped, stitch the warning directly to the tool results text
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


    async def _fetch_context(self, query: str) -> tuple[list, str]:
        """Fetch embeddings and query Qdrant dynamically on every request."""

        # --- BYPASS RAG SEARCH ---
        if not self.use_rag:
            _LOGGER.info("ℹ️ Embedding model set to 'None'. Bypassing vector search and unlocking all tools.")
            return None, ""
        
        _LOGGER.info(f"🔍 [VECTOR SEARCH] Requesting embedding for: '{query}'")
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            headers = {"Authorization": f"Bearer {self.llm_api_key}"} if self.llm_api_key else {}
            ssl_context = get_default_context() if self.ollama_url.startswith("https") else False

            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Format payload based on backend architecture
                if self.backend_type in ["openai_official", "openai_compatible"]:
                    payload = {"model": self.embedding_model, "input": query}
                else:
                    payload = {"model": self.embedding_model, "prompt": query}

                async with session.post(
                    self.embeddings_url, # dynamic routing variable
                    json=payload, 
                    headers=headers,
                    ssl=ssl_context
                ) as resp:
                    if resp.status != 200: return [], ""
                    data = await resp.json()
                    query_vector = data.get("embedding")
                    if not query_vector: return [], ""
        except Exception as e:
            _LOGGER.error(f"❌ Failed to connect to Ollama Embeddings: {e}")
            return [], ""

        client = await self._get_qdrant_client()

        # Always inject these fallback tools regardless of the search
        always_allow = ["smart_web_search", "GetLiveContext"]
        unlocked_tools = list(always_allow)

        _LOGGER.info("🔍 [VECTOR SEARCH] Querying Qdrant Database...")
        try:
            async with asyncio.timeout(5.0):
                tool_results, fact_results = await asyncio.gather(
                    client.query_points(collection_name="tools_collection", query=query_vector, limit=self.TOOL_LIMIT, with_payload=True),
                    client.query_points(collection_name="memories_collection", query=query_vector, limit=self.MEMORY_LIMIT, with_payload=True)
                )
        except TimeoutError:
            _LOGGER.error("❌ Qdrant vector search timed out after 5 seconds.")
            return unlocked_tools, "" # Return default tools so the AI can still attempt an answer

        _LOGGER.info("--- QDRANT TOOL SEARCH SCORES ---")
        
        # Add the number of highest embed ranked tools based on tool limit set. 
        for hit in tool_results.points:
            tool_name = hit.payload.get("tool_id") or hit.payload.get("tool_name")
            passes_threshold = hit.score > 0.50 # Check tool ranking score for debugging tool unlocks
            _LOGGER.info(f"🛠️ Tool: {tool_name:<25} | Cosine Score: {hit.score:.4f} | Pass: {passes_threshold}")
            
            unlocked_tools.append(tool_name)

        _LOGGER.info("--- QDRANT MEMORY SEARCH SCORES ---")
        found_facts = []
        for hit in fact_results.points:
            content = hit.payload.get('content', '')
            short_content = (content[:60] + '...') if len(content) > 60 else content
            passes_threshold = hit.score > 0.50
            _LOGGER.info(f"🧠 Memory: {short_content:<25} | Cosine Score: {hit.score:.4f} | Pass: {passes_threshold}")
            
            if passes_threshold:
                found_facts.append(f"- {content}")

        memories = "\n".join(found_facts) if found_facts else ""

        _LOGGER.info(f"🔓 [FINAL UNLOCKED TOOLS]: {unlocked_tools}")
        _LOGGER.info(f"🧠 [FINAL RETRIEVED MEMORIES]: {len(found_facts)} facts injection ready.")

        return unlocked_tools, memories

    def _build_system_prompt(self, device_id: str, personal_memories: str, ha_base_prompt: str) -> str:
        """Assembles prompt mirroring HA's native organization."""
        location_name = "Unknown"
        active_area_name = ""
        
        if device_id:
            dev_reg = dr.async_get(self.hass)
            area_reg = ar.async_get(self.hass)
            if device := dev_reg.async_get(device_id):
                if device.area_id and (area := area_reg.async_get_area(device.area_id)):
                    location_name = area.name
                    active_area_name = area.name.lower().strip()

        if active_area_name:
            ha_base_prompt = ha_base_prompt.replace(
                "Static Context: An overview of the areas and the devices in this smart home:",
                f"Static Context: An overview of the areas and the devices in the {location_name}:"
            )
            pattern = r'-\s+names:[^\n]+\n\s+domain:[^\n]+\n\s+areas:\s+([^\n]+)\n'
            def room_evaluator(match):
                if match.group(1).lower().strip() != active_area_name: return ""
                return match.group(0)
            
            ha_base_prompt = re.sub(pattern, room_evaluator, ha_base_prompt)
            ha_base_prompt = re.sub(r'\n\s+areas:\s+[^\n]+', '', ha_base_prompt)
            _LOGGER.debug(f"✂️ Applied Regex Room Filter for area: {active_area_name}")

        static_prefix = self.entry.options.get(
            "Instructions", 
            "You are the conversational brain of a smart home..."
        )
        
        ha_context = f"\n### HOME ASSISTANT ENTITIES\n{ha_base_prompt}\n"

        now = dt_util.now()
        day = now.day
        suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        day_str = f"{now.strftime('%B')} {day}{suffix}, {now.strftime('%Y')}"

        dynamic_suffix = (
            f"\n### TIME AND LOCATION CONTEXT\n"
            f"Physical Location: You are physically located in the {location_name}.\n"
            f"If the user requests context or operations for an entity outside of the {location_name}, you MUST call `GetLiveContext` for that specific area first to discover it.\n"
            f"Current Context: Today is {now.strftime('%A')}, {day_str} and the current time is {now.strftime('%-I:%M %p')}.\n"
        )

        if personal_memories: dynamic_suffix += f"\n### PERSONAL MEMORIES & FACTS\n{personal_memories}\n"
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

    async def _execute_tool_loop(self, messages, tool_schemas, ha_api_instance):
        max_iterations = self.max_iterations
        
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
            selected_model = self.entry.options.get("model", "qwen2.5:latest")
            use_thinking = self.entry.options.get("thinking", False)
            
            # 4. Format the keep_alive variable for Ollama
            raw_keep_alive = self.entry.options.get("keep_alive", -1)
            if raw_keep_alive == -1:
                keep_alive_val = -1
            else:
                keep_alive_val = f"{raw_keep_alive}m"  # Converts 5 to "5m"

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
            
            # Deconstruct Ollama's custom response object into plain Python primitives
            raw_message = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
            raw_text = raw_message.get("content", "") if isinstance(raw_message, dict) else getattr(raw_message, "content", "")
            raw_tool_calls = raw_message.get("tool_calls", []) if isinstance(raw_message, dict) else getattr(raw_message, "tool_calls", [])
            
            _LOGGER.info(f"📥 [RAW UNFILTERED OLLAMA RESPONSE]:\n{raw_text}")
            final_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip() if raw_text else ""
            
            # 100% Clean Dictionary for json.dumps() compatibility
            safe_message = {"role": "assistant", "content": final_text}

            # Safely rebuild tool calls into native dictionaries
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
            
            for tool_call in safe_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                raw_args_payload = tool_call["function"]["arguments"]
                
                # Unpack both the repaired dictionary and the dynamic warning string
                repaired_args, argument_warning = self._fix_invalid_arguments(raw_args_payload)
                # Prune out the keys that _fix_invalid_arguments just blanked out
                args = {k: v for k, v in repaired_args.items() if v is not None and v != ""}
                
                _LOGGER.info(f"⚙️ Executing Tool: {tool_name} with args {args}")
                
                try:
                    ha_tool_input = llm.ToolInput(tool_name=tool_name, tool_args=args)
                    
                    if tool_name in self.custom_tools:
                        result = await self.custom_tools[tool_name].async_call(self.hass, ha_tool_input, ha_api_instance.llm_context)
                    else:
                        result = await ha_api_instance.async_call_tool(ha_tool_input)
                    
                    try:
                        compressed_content = self._compress_tool_response(result, tool_name)
                    except Exception as compress_err:
                        _LOGGER.error(f"⚠️ Compression routine error inside fallback loop: {compress_err}")
                        compressed_content = str(result)

                    # If an argument was stripped, stitch the warning directly to the tool results text
                    if argument_warning:
                        compressed_content += argument_warning

                    messages.append({"role": "tool", "content": compressed_content, "name": tool_name})
                    
                except Exception as e:
                    _LOGGER.error(f"❌ Tool execution exception {tool_name}: {e}")
                    messages.append({"role": "tool", "content": json.dumps({"error": str(e)}), "name": tool_name})

        return "I had to stop thinking because I used too many tools."

    def _assemble_and_filter_tools(self, ha_api_instance, unlocked_tool_names):
        master_dict = {}
        for ha_tool in ha_api_instance.tools:
            if ha_tool.name.replace("assist__", "") not in BUILT_IN_BLACKLIST:
                master_dict[ha_tool.name] = ha_tool

        for name, tool_obj in self.custom_tools.items():
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

    async def _get_ha_api_instance(self, user_input):
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            language=user_input.language,
            assistant="conversation",
            device_id=user_input.device_id,
        )
        return await llm.async_get_api(self.hass, llm.LLM_API_ASSIST, llm_context)