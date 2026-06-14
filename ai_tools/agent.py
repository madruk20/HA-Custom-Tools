import logging
import aiohttp
import asyncio
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

_LOGGER = logging.getLogger(__name__)

DOMAIN = "ai_tools"

BUILT_IN_BLACKLIST = [
    "HassMediaSearchAndPlay", "HassMediaPause", "HassMediaUnpause",       
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
        self.api_key = entry.options.get("api_key", "")
        self.ollama_url_embeddings = f"{self.ollama_url.rstrip('/')}/api/embeddings"

        # Embedding Models and Connection Settings
        self.qdrant_client = None
        self.embedding_model = entry.options.get("embedding_model", "qwen-embed-2k:latest")
        self.qdrant_url = entry.options.get("qdrant_url", "http://192.168.4.23:6333")
        
        # State Management
        self._query_cache = {}
        self.history = {}

        # Custom Tool Instantiation
        self.custom_tools = {
            "smart_web_search": WebSearchTool(),
            "alarm_manager": AlarmManagerTool(),
            "stock_and_retail_price_lookup": PriceLookupTool(),
            "music_player": MusicPlayerTool(),
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
        # Check AND set the same variable
        if self.qdrant_client is None:
            _LOGGER.info("Initializing Qdrant Client...")
            
            # Wrap the blocking constructor in a standard synchronous function
            def _create_client():
                # Explicitly set the connection timeout and use the dynamic URL
                return AsyncQdrantClient(url=self.qdrant_url, timeout=10.0)
            
            # Offload the creation to a background thread to protect the event loop
            self.qdrant_client = await self.hass.async_add_executor_job(_create_client)
            
        return self.qdrant_client


    async def _get_ollama_client(self):
        if self.ollama_client is None:
            _LOGGER.info("Initializing Ollama Client...")
            
            def _create_client():
                return ollama.AsyncClient(
                    host=self.ollama_url,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None,
                    verify=get_default_context()
                )
            
            self.ollama_client = await self.hass.async_add_executor_job(_create_client)
            
        return self.ollama_client
    

    def _fix_invalid_arguments(self, arguments) -> dict:
        """Sanitize and repair malformed JSON arguments from smaller LLMs."""
        if isinstance(arguments, dict):
            return arguments

        if not isinstance(arguments, str):
            return {}

        # Strip whitespace and potential markdown code blocks
        arguments = arguments.strip().removeprefix("```json").removesuffix("```").strip()

        # Repair double-serialized strings (e.g., "{\"key\": \"value\"}")
        if arguments.startswith('"') and arguments.endswith('"'):
            try:
                import json
                unquoted = json.loads(arguments)
                if isinstance(unquoted, dict):
                    return unquoted
            except Exception:
                pass

        # Extract the outermost JSON object if the model wrapped it in text
        try:
            match = re.search(r"\{.*\}", arguments, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            _LOGGER.warning(f"Failed to repair JSON arguments: {arguments}")

        return {}


    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """The main conversation turn logic with streaming and vision support."""
        user_query = user_input.text
        
        # 1. Fetch API Instance and Tools
        ha_api_instance = await self._get_ha_api_instance(user_input)
        unlocked_tool_names, personal_memories = await self._fetch_context(user_query)
        active_tools, active_tool_schemas = self._assemble_and_filter_tools(ha_api_instance, unlocked_tool_names)

        # 2. Build System Prompt
        system_prompt = self._build_system_prompt(
            user_input.device_id, 
            personal_memories, 
            ha_api_instance.api_prompt
        )

        # 3. Manage local history for the conversation thread
        session_id = user_input.conversation_id or "default"
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
            "top_k": self.entry.options.get("top_k", 40)
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
            
            # Placeholder dictionary to capture final metadata from the stream
            final_metadata = {}

            async def _transform_stream():
                new_msg = True
                async for chunk in response_generator:
                    # Capture token statistics if they exist in the chunk (usually the final chunk)
                    if "eval_count" in chunk or "prompt_eval_count" in chunk:
                        final_metadata.update(chunk)

                    msg = chunk.get("message", {})
                    delta = {}
                    
                    if new_msg:
                        delta["role"] = "assistant"
                        new_msg = False
                    
                    if content := msg.get("content"):
                        delta["content"] = content
                        full_content_out.append(content)
                        
                    if raw_tool_calls := msg.get("tool_calls"):
                        for tc in raw_tool_calls:
                            tool_calls_buffer.append(tc)

                    if delta:
                        yield delta

            # Stream text to the UI
            async for _ in chat_log.async_add_delta_content_stream(self.entity_id, _transform_stream()):
                pass

            # --- LOG TOKEN METRICS HERE ---
            p_tokens = final_metadata.get("prompt_eval_count", 0)
            p_time = final_metadata.get("prompt_eval_duration", 0) / 1e9
            gen_tokens = final_metadata.get("eval_count", 0)
            total_time = final_metadata.get("total_duration", 0) / 1e9
            speed = f"{(gen_tokens / (total_time - p_time)):.2f} t/s" if (total_time - p_time) > 0 else "N/A"
            _LOGGER.info(f"🤖 OLLAMA VERBOSE: Prompt: {p_tokens}t | Generated: {gen_tokens}t | Speed: {speed} | Total: {total_time:.2f}s")
            # ------------------------------

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
                break
            
            # Execute tools manually
            for tool_call in safe_message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                raw_args = tool_call["function"]["arguments"]
                repaired_args = self._fix_invalid_arguments(raw_args)
                args = {k: v for k, v in repaired_args.items() if v is not None and v != ""}
                
                _LOGGER.info(f"⚙️ Executing Tool: {tool_name} with args {args}")
                try:
                    ha_tool_input = llm.ToolInput(tool_name=tool_name, tool_args=args)
                    
                    if tool_name in self.custom_tools:
                        result = await self.custom_tools[tool_name].async_call(self.hass, ha_tool_input, ha_api_instance.llm_context)
                    else:
                        result = await ha_api_instance.async_call_tool(ha_tool_input)
                    
                    compressed_content = self._compress_tool_response(result, tool_name)
                    
                    # Append tool result to both current payload and persistent history
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
        _LOGGER.info(f"🔍 [VECTOR SEARCH] Requesting embedding for: '{query}'")
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            ssl_context = get_default_context() if self.ollama_url.startswith("https") else False

            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {"model": self.embedding_model, "prompt": query}
                async with session.post(
                    self.ollama_url_embeddings, 
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
                "Static Context: An overview of the areas and the devices in this room:"
            )
            pattern = r'-\s+names:[^\n]+\n\s+domain:[^\n]+\n\s+areas:\s+([^\n]+)\n'
            def room_evaluator(match):
                if match.group(1).lower().strip() != active_area_name: return ""
                return match.group(0)
            
            ha_base_prompt = re.sub(pattern, room_evaluator, ha_base_prompt)
            ha_base_prompt = re.sub(r'\n\s*\n', '\n', ha_base_prompt)
            _LOGGER.debug(f"✂️ Applied Regex Room Filter for area: {active_area_name}")

        static_prefix = self.entry.options.get(
            "Instructions", 
            "You are the conversational brain of a smart home in Pacific Palisades..."
        )
        
        ha_context = f"\n### HOME ASSISTANT ENTITIES\n{ha_base_prompt}\n"

        now = dt_util.now()
        day = now.day
        suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        day_str = f"{now.strftime('%B')} {day}{suffix}, {now.strftime('%Y')}"

        dynamic_suffix = (
            f"\n### TIME AND LOCATION CONTEXT\n"
            f"Physical Location: You are physically located in the {location_name}. If the request does not state a room and one is required, default to the {location_name}.\n"
            f"If the user requests context or operations for an entity outside of the {location_name}, you MUST call `GetLiveContext` for that specific area first to discover it.\n"
            f"Current Context: Today is {now.strftime('%A')}, {day_str} and the current time is {now.strftime('%-I:%M %p')}.\n"
        )

        if personal_memories: dynamic_suffix += f"\n### PERSONAL MEMORIES & FACTS\n{personal_memories}\n"
        return f"{static_prefix}{ha_context}{dynamic_suffix}"

    def _compress_tool_response(self, result: dict, tool_name: str) -> str:
        """Compresses HA intent responses to save tokens and maintain fine-tuning patterns."""
        if tool_name in self.custom_tools:
            # Leave custom tools (search, music) untouched so the LLM can read the full text results
            return json.dumps(result, default=str) if isinstance(result, dict) else str(result)
            
        try:
            if isinstance(result, dict) and "response_type" in result:
                response_type = result.get("response_type")
                
                # Compress Action Responses
                if response_type == "action_done":
                    success_entities = result.get("data", {}).get("success", [])
                    if success_entities:
                        names = [e.get("name", "Unknown") for e in success_entities]
                        return f"Success. Action executed on: {', '.join(names)}"
                    return "Success. Action executed."
                
                # Compress Errors
                elif response_type == "error":
                    return f"Failed. Error code: {result.get('data', {}).get('code', 'unknown')}"
                
                # Extract clean speech from Query Responses (e.g., GetLiveContext)
                elif response_type == "query_answer":
                    speech = result.get("speech", {}).get("plain", {}).get("speech", "")
                    if speech:
                        return speech
                        
            return json.dumps(result, default=str) if isinstance(result, dict) else str(result)
        except Exception:
            return json.dumps(result, default=str) if isinstance(result, dict) else str(result)

    async def _execute_tool_loop(self, messages, active_tools, tool_schemas, ha_api_instance):
        max_iterations = 5
        
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
            "top_k": self.entry.options.get("top_k", 40)
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
                
                # 1. Grab the raw arguments (which might be a broken string)
                raw_args_payload = tool_call["function"]["arguments"]
                
                # 2. Pass it through our new repair function to guarantee it is a dictionary
                repaired_args = self._fix_invalid_arguments(raw_args_payload)
                
                # 3. Prune empty or null arguments
                args = {k: v for k, v in repaired_args.items() if v is not None and v != ""}
                
                _LOGGER.info(f"⚙️ Executing Tool: {tool_name} with args {args}")
                
                try:
                    # Pass the cleaned 'args' dictionary instead of the 'raw_args'
                    ha_tool_input = llm.ToolInput(tool_name=tool_name, tool_args=args)
                    
                    if tool_name in self.custom_tools:
                        result = await self.custom_tools[tool_name].async_call(self.hass, ha_tool_input, ha_api_instance.llm_context)
                    else:
                        result = await ha_api_instance.async_call_tool(ha_tool_input)
                    
                    # Apply your compression logic
                    compressed_content = self._compress_tool_response(result, tool_name)
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
            if clean_name in unlocked_tool_names:
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