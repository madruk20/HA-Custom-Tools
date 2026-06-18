import logging
import re
import os
import voluptuous as vol
from pathlib import Path

from homeassistant import config_entries
from homeassistant.helpers import selector
from homeassistant.helpers.selector import NumberSelectorMode
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util.ssl import get_default_context

DOMAIN = "ai_tools"
_LOGGER = logging.getLogger(__name__)

ALL_NATIVE_HA_TOOLS = [
    "HassTurnOn", "HassTurnOff", "HassStartTimer", "HassCancelTimer", 
    "HassTimerStatus", "HassLightSet", "HassBroadcast", "HassListAddItem", 
    "HassListCompleteItem", "HassListRemoveItem", "todo_get_items", 
    "HassGetState", "HassMediaSearchAndPlay", "HassMediaPause", 
    "HassMediaUnpause", "HassMediaNext", "HassMediaPrevious", "HassSetVolume", 
    "HassSetVolumeRelative", "HassMediaPlayerMute", "HassMediaPlayerUnmute", 
    "HassCancelAllTimers", "HassIncreaseTimer", "HassDecreaseTimer", 
    "HassPauseTimer", "HassUnpauseTimer", "GetDateTime"
]

DEFAULT_SYSTEM_PROMPT = (
    "You are the conversational brain of a smart home in Pacific Palisades.\n\n"
    "### OPERATIONAL RULES\n"
    "- VERBAL RESPONSES: Respond in clean, unformatted plain text ONLY. Never use Markdown (*, #, bolds, lists) in final verbal answers.\n"
    "- DEVICE CONTROL: Select and call the exact tool name from the available tool list. Pass ONLY target 'name' and 'domain' or 'area' and 'domain' to the tool arguments. Never fill in unrequested optional parameters.\n"
    "- REAL-TIME STATES: Never guess or rely on conversation history to determine the current state of any device, sensor, or alarm times. You MUST call the `GetLiveContext` tool immediately to verify current truth. (Exception: The current time and date are dynamically provided below under Time and Location Context and can be answered immediately without tool calls).\n\n"
    "### TOOL ENFORCEMENT RULES\n"
    "- KNOWLEDGE/FACTS: Your internal data for world facts, current events, and politics is empty. To answer these, you MUST call `smart_web_search` tool.\n"
    "- RETRIEVAL: Use `stock_and_retail_price_lookup` tool for market tickers/investments, electronics, shopping, or product costs.\n"
    "- ALARMS: Use `alarm_manager` tool for all requests to set or cancel an alarm.\n"
    "- MUSIC: Use `music_player` tool to control playing music on any media player or speaker.\n"
    "- CAMERAS: Use `stream_camera_to_tv` tool to view any security cameras. Leave TV blank if user does not specify a TV."
)

CLOUD_LLM_MODELS = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "llama3-70b-8192"]
CLOUD_EMBED_MODELS = ["text-embedding-3-small", "text-embedding-3-large", "None"]

class AIToolsConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")
        
        if user_input is None:
            return self.async_show_form(step_id="user")
            
        return self.async_create_entry(title="Custom AI", data={})

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return AIToolsOptionsFlowHandler()

class AIToolsOptionsFlowHandler(config_entries.OptionsFlow):
    def __init__(self) -> None:
        """Initialize options flow."""
        self._options = {}

    async def async_step_init(self, user_input=None):
        options = self.config_entry.options
        errors = {}
        
        # Get current backend type or default to local Ollama
        llm_backend_type = user_input.get("llm_backend_type") if user_input else options.get("llm_backend_type", "local_ollama")
        llm_url = user_input.get("llm_url") if user_input else options.get("llm_url", "http://192.168.4.23:11434")
        llm_api_key = user_input.get("llm_api_key") if user_input else options.get("llm_api_key", "")

        # Embed backend settings (default to whatever the LLM is doing so it doesn't break existing setups)
        embed_backend_type = user_input.get("embed_backend_type") if user_input else options.get("embed_backend_type", llm_backend_type)
        embed_url = user_input.get("embed_url") if user_input else options.get("embed_url", llm_url)
        embed_api_key = user_input.get("embed_api_key") if user_input else options.get("embed_api_key", llm_api_key)


        # 1. Conditional Connection Validation
        if user_input is not None:
            # ONLY validate tags if they are actually trying to use a local Ollama instance
            if llm_backend_type == "local_ollama":
                try:
                    session = async_get_clientsession(self.hass)
                    headers = {"Authorization": f"Bearer {llm_api_key}"} if llm_api_key else {}
                    ssl_context = get_default_context() if llm_url.startswith("https") else False
                    
                    async with session.get(f"{llm_url.rstrip('/')}/api/tags", headers=headers, ssl=ssl_context, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            available_models = [m["name"] for m in data.get("models", [])]
                            if user_input.get("llm_model") not in available_models:
                                errors["llm_model"] = "model_not_found"
                        else:
                            errors["base"] = "cannot_connect"
                except Exception as e:
                    _LOGGER.error(f"Ollama Validation failed: {e}")
                    errors["base"] = "cannot_connect"
            
            if not errors:
                # Save first page options to the class instance
                self._options.update(user_input)
                
                # Check if user wants to open the secondary tool config window
                if user_input.get("configure_tools"):
                    return await self.async_step_tools()
                    
                # If they didn't check the box, finalize the entry immediately
                # Ensure we carry over any existing blacklisted tools if they skip the menu
                if "blacklisted_tools" not in self._options:
                    self._options["blacklisted_tools"] = options.get("blacklisted_tools", [])
                return self.async_create_entry(title="", data=self._options)

        # 2. Populate Dropdown Lists dynamically based on current backend selection
        llm_models = []
        embed_models = []

        if llm_backend_type == "local_ollama":
            try:
                session = async_get_clientsession(self.hass)
                headers = {"Authorization": f"Bearer {llm_api_key}"} if llm_api_key else {}
                async with session.get(f"{llm_url.rstrip('/')}/api/tags", headers=headers, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        embedding_model_families = {"bert", "roberta", "distilbert", "embedding"}
                        for m in data.get("models", []):
                            families = [f.lower() for f in m.get("details", {}).get("families", [])]
                            name = m["name"].lower()
                            is_embed = any(fam in embedding_model_families for fam in families) or "embed" in name
                            if is_embed:
                                embed_models.append(m["name"])
                            else:
                                llm_models.append(m["name"])
            except Exception as e:
                _LOGGER.error(f"Could not refresh local Ollama list: {e}")
            
            if not llm_models: llm_models = ["offline_fallback:latest"]
            embed_models.insert(0, "None")
        else:
            # Fallback to standard cloud arrays if they selected OpenAI or a Generic Proxy
            llm_models = CLOUD_LLM_MODELS
            embed_models = CLOUD_EMBED_MODELS

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required("llm_backend_type", default=llm_backend_type): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": "local_ollama", "label": "Ollama (Local Server)"},
                            {"value": "openai_official", "label": "OpenAI (Official Cloud)"},
                            {"value": "openai_compatible", "label": "OpenAI Compatible (Groq, OpenRouter, LM Studio)"}
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                ),

                vol.Required(
                    "llm_url", 
                    default=llm_url
                ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.URL)),

                vol.Optional(
                    "llm_api_key", 
                    default=llm_api_key
                ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)),

                vol.Required(
                    "llm_model", 
                    default=user_input.get("llm_model") if user_input else options.get("llm_model", llm_models[0])
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=llm_models, 
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        custom_value=True
                    )
                ),

                vol.Required(
                    "embed_backend_type", 
                    default=embed_backend_type
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": "none", "label": "None (Disable RAG / Vector Database)"},
                            {"value": "local_ollama", "label": "Ollama (Local Embeddings)"},
                            {"value": "openai_official", "label": "OpenAI (Official Cloud Embeddings)"},
                            {"value": "openai_compatible", "label": "OpenAI Compatible Embeddings"}
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                ),

                vol.Required(
                    "embed_url", 
                    default=embed_url
                ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.URL)),

                vol.Optional(
                    "embed_api_key", 
                    default=embed_api_key
                ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)),

                vol.Required(
                    "embedding_model", 
                    default=user_input.get("embedding_model") if user_input else options.get("embedding_model", embed_models[0] if embed_models else "qwen-embed-2k:latest")
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=embed_models, 
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        custom_value=True
                    )
                ),

                vol.Required("vector_db_backend", default="qdrant"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": "none", "label": "None (Disable RAG / Vector Database)"},
                            {"value": "qdrant", "label": "Qdrant"},
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                ),

                vol.Required(
                    "vector_db_url", 
                    default=options.get("vector_db_url", "http://localhost:6333")
                ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.URL)),

                vol.Optional(
                    "vector_db_api_key", 
                    default=options.get("vector_db_api_key", "")
                ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)),

                vol.Optional(
                    "Instructions", 
                    default=options.get("Instructions", DEFAULT_SYSTEM_PROMPT)
                ): selector.TemplateSelector(),
                
                vol.Optional(
                    "thinking", 
                    default=options.get("thinking", False)
                ): selector.BooleanSelector(),
                
                vol.Optional(
                    "temperature", 
                    default=options.get("temperature", 0.5)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0, max=1, step=0.1, mode=NumberSelectorMode.SLIDER)
                ),

                vol.Optional(
                    "repeat_penalty", 
                    default=options.get("repeat_penalty", 1.1)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=1.0, max=2.0, step=0.05, mode=NumberSelectorMode.SLIDER)
                ),

                vol.Optional(
                    "top_p", 
                    default=options.get("top_p", 0.9)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode=NumberSelectorMode.SLIDER)
                ),

                vol.Optional(
                    "top_k", 
                    default=options.get("top_k", 40)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=1, max=100, mode=NumberSelectorMode.BOX)
                ),

                vol.Optional(
                    "max_history", 
                    default=options.get("max_history", 40)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0, max=100, mode=NumberSelectorMode.BOX)
                ),
                
                vol.Optional(
                    "num_ctx", 
                    default=options.get("num_ctx", 32768)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=2048, max=32768, mode=NumberSelectorMode.BOX)
                ),

                vol.Optional(
                    "num_predict", 
                    default=options.get("num_predict", 512)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=128, max=4096, mode=NumberSelectorMode.BOX)
                ),

                vol.Optional(
                    "mirostat", 
                    default=options.get("mirostat", "0")
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": "0", "label": "Disabled"},
                            {"value": "1", "label": "Mirostat 1.0"},
                            {"value": "2", "label": "Mirostat 2.0"}
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                ),

                vol.Optional(
                    "keep_alive", 
                    default=options.get("keep_alive", -1)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=-1, max=1440, mode=NumberSelectorMode.BOX)
                ),

                vol.Optional("configure_tools", default=False): selector.BooleanSelector(),
            }),
            errors=errors
        )
    

    async def async_step_tools(self, user_input=None):
        """Second step to handle the tool blacklist."""
        if user_input is not None:
            # --- CLEAR CACHE LOGIC ---
            if user_input.get("clear_cache_now"):
                cache_path = Path(__file__).parent / "semantic_cache.json"
                if cache_path.exists():
                    _LOGGER.info("🗑️ Clearing semantic cache via Config Flow...")
                    await self.hass.async_add_executor_job(os.remove, cache_path)
                
                # Reset the checkbox to False so it doesn't try to delete it every time you save settings
                user_input["clear_cache_now"] = False
            
            self._options.update(user_input)
            return self.async_create_entry(title="", data=self._options)
        
        def get_custom_tools() -> list[str]:
            """Dynamically scan the tools directory for custom tool names."""
            tools_dir = Path(__file__).parent / "tools"
            found_tools = []

            if not tools_dir.exists():
                return found_tools

            for file_path in tools_dir.glob("*.py"):
                if file_path.name == "__init__.py":
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            # Search for the class attribute `name = "tool_name"`
                            if line.startswith("name =") or line.startswith("name="):
                                match = re.search(r'["\']([^"\']+)["\']', line)
                                if match:
                                    found_tools.append(match.group(1))
                                    break # Found the tool name, move to next file
                except Exception:
                    pass
                    
            return found_tools

        # Current States
        options = self.config_entry.options
        current_blacklist = options.get("blacklisted_tools", [])
        current_limit = options.get("tool_injection_limit", 3)
        current_threshold = options.get("tool_cosine_threshold", 0.30)

        current_memory_enabled = options.get("enable_memory_injection", True)
        current_mem_limit = options.get("memory_injection_limit", 3)
        current_mem_threshold = options.get("memory_cosine_threshold", 0.50)
        current_collections = options.get("memory_collections", ["memories_collection"])

        # Create dropdown options for Tools
        native_options = [{"value": tool, "label": f"{tool} (Native)"} for tool in ALL_NATIVE_HA_TOOLS]
        custom_tools = await self.hass.async_add_executor_job(get_custom_tools)
        custom_options = [{"value": tool, "label": f"{tool} (Custom)"} for tool in custom_tools]
        all_options = native_options + custom_options

        # Create dynamically populated chips for Memory Collections
        memory_options = [{"value": c, "label": c} for c in current_collections]
        # Ensure the default is always visibly selectable even if the list is cleared
        if not any(opt["value"] == "memories_collection" for opt in memory_options):
            memory_options.append({"value": "memories_collection", "label": "memories_collection"})

        return self.async_show_form(
            step_id="tools",
            data_schema=vol.Schema({
                vol.Optional("blacklisted_tools", default=current_blacklist): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=all_options,
                        multiple=True, 
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        custom_value=True 
                    )
                ),
                vol.Optional("tool_injection_limit", default=current_limit): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1,
                        max=25,
                        mode=NumberSelectorMode.BOX
                    )
                ),
                vol.Optional("tool_cosine_threshold", default=current_threshold): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        mode=NumberSelectorMode.SLIDER or NumberSelectorMode.BOX
                    )
                ),
                vol.Optional("enable_memory_injection", default=current_memory_enabled): selector.BooleanSelector(),
                vol.Optional("memory_collections", default=current_collections): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=memory_options,
                        multiple=True, 
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        custom_value=True
                    )
                ),
                vol.Optional("memory_injection_limit", default=current_mem_limit): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0, max=50, mode=NumberSelectorMode.BOX)
                ),
                vol.Optional("memory_cosine_threshold", default=current_mem_threshold): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode=NumberSelectorMode.SLIDER)
                ),
                vol.Optional("clear_cache_now", default=False): selector.BooleanSelector()
            })
        )
