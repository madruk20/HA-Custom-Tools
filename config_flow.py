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
from homeassistant.helpers import area_registry as ar

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
        return self.async_create_entry(title="Custom AI Agent", data={})

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return AIToolsOptionsFlowHandler(config_entry)

class AIToolsOptionsFlowHandler(config_entries.OptionsFlow):
    def __init__(self, config_entry) -> None:
        self._options = dict(config_entry.options)

    async def async_step_init(self, user_input=None):
        """The Main Dashboard Menu."""
        return self.async_show_menu(
            step_id="init",
            menu_options=["provider_settings", "tuning_settings", "tool_settings", "memory_settings", "finish"]
        )

    async def async_step_provider_settings(self, user_input=None):
        """Step 1: Connection and Model Providers."""
        if user_input is not None:
            self._options.update(user_input)
            return await self.async_step_init()

        options = self._options
        llm_backend = options.get("llm_backend_type", "local_ollama")
        llm_url = options.get("llm_url", "http://localhost:11434")
        llm_api_key = options.get("llm_api_key", "")
        
        llm_models = []
        embed_models = []

        # Dynamic Model Fetching
        if llm_backend == "local_ollama":
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
            llm_models = CLOUD_LLM_MODELS
            embed_models = CLOUD_EMBED_MODELS

        return self.async_show_form(
            step_id="provider_settings",
            data_schema=vol.Schema({
                vol.Required("llm_backend_type", default=llm_backend, description="Primary AI inference backend."): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": "local_ollama", "label": "Ollama (Local Server)"},
                            {"value": "openai_official", "label": "OpenAI (Official Cloud)"},
                            {"value": "openai_compatible", "label": "OpenAI Compatible"}
                        ], mode=selector.SelectSelectorMode.DROPDOWN)
                ),
                vol.Required("llm_url", default=llm_url, description="Address of the LLM API."): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.URL)),
                vol.Optional("llm_api_key", default=llm_api_key, description="Required for Cloud or restricted local APIs."): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)),
                vol.Required("llm_model", default=options.get("llm_model", llm_models[0]), description="The specific AI model to load."): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=llm_models, mode=selector.SelectSelectorMode.DROPDOWN, custom_value=True)
                ),
                vol.Required("embed_backend_type", default=options.get("embed_backend_type", llm_backend), description="Backend used specifically for embeddings/RAG."): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": "none", "label": "None (Disable RAG)"},
                            {"value": "local_ollama", "label": "Ollama (Local Embeddings)"},
                            {"value": "openai_official", "label": "OpenAI (Cloud Embeddings)"},
                            {"value": "openai_compatible", "label": "OpenAI Compatible"}
                        ], mode=selector.SelectSelectorMode.DROPDOWN)
                ),
                vol.Required("embed_url", default=options.get("embed_url", llm_url), description="Address for the embedding API."): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.URL)),
                vol.Optional("embed_api_key", default=options.get("embed_api_key", llm_api_key), description="API Key for embeddings."): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)),
                vol.Required("embedding_model", default=options.get("embedding_model", embed_models[0] if embed_models else "qwen-embed-2k:latest"), description="Model used for vectorizing text."): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=embed_models, mode=selector.SelectSelectorMode.DROPDOWN, custom_value=True)
                ),
                vol.Required("vector_db_backend", default=options.get("vector_db_backend", "qdrant"), description="Vector Database software."): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=[{"value": "none", "label": "None"}, {"value": "qdrant", "label": "Qdrant"}], mode=selector.SelectSelectorMode.DROPDOWN)
                ),
                vol.Required("vector_db_url", default=options.get("vector_db_url", "http://localhost:6333"), description="Address of the Vector DB."): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.URL)),
                vol.Optional("vector_db_api_key", default=options.get("vector_db_api_key", ""), description="Vector DB Auth Key."): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)),
            })
        )

    async def async_step_tuning_settings(self, user_input=None):
        """Step 2: Prompts and Inference Sliders."""
        if user_input is not None:
            self._options.update(user_input)
            return await self.async_step_init()

        options = self._options
        return self.async_show_form(
            step_id="tuning_settings",
            data_schema=vol.Schema({
                vol.Optional("Instructions", default=options.get("Instructions", DEFAULT_SYSTEM_PROMPT), description="The core System Prompt defining agent behavior."): selector.TemplateSelector(),
                vol.Optional("thinking", default=options.get("thinking", False), description="Enable <think> tags for supported reasoning models."): selector.BooleanSelector(),
                vol.Optional("temperature", default=options.get("temperature", 0.5), description="Creativity/Randomness of the response."): selector.NumberSelector(selector.NumberSelectorConfig(min=0, max=1, step=0.1, mode=NumberSelectorMode.SLIDER)),
                vol.Optional("top_p", default=options.get("top_p", 0.9), description="Nucleus sampling probability."): selector.NumberSelector(selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode=NumberSelectorMode.SLIDER)),
                vol.Optional("repeat_penalty", default=options.get("repeat_penalty", 1.1), description="Prevents the AI from repeating itself."): selector.NumberSelector(selector.NumberSelectorConfig(min=1.0, max=2.0, step=0.05, mode=NumberSelectorMode.SLIDER)),
                vol.Optional("top_k", default=options.get("top_k", 40), description="Limit token selection to top K choices."): selector.NumberSelector(selector.NumberSelectorConfig(min=1, max=100, mode=NumberSelectorMode.BOX)),
                vol.Optional("max_history", default=options.get("max_history", 10), description="Max messages retained in session memory."): selector.NumberSelector(selector.NumberSelectorConfig(min=0, max=100, mode=NumberSelectorMode.BOX)),
                vol.Optional("num_ctx", default=options.get("num_ctx", 8192), description="Maximum context window size."): selector.NumberSelector(selector.NumberSelectorConfig(min=2048, max=32768, mode=NumberSelectorMode.BOX)),
                vol.Optional("num_predict", default=options.get("num_predict", 512), description="Max tokens to generate per response."): selector.NumberSelector(selector.NumberSelectorConfig(min=128, max=4096, mode=NumberSelectorMode.BOX)),
                vol.Optional("mirostat", default=options.get("mirostat", "0"), description="Alternative to Temperature/Top P tuning."): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=[{"value": "0", "label": "Disabled"}, {"value": "1", "label": "Mirostat 1.0"}, {"value": "2", "label": "Mirostat 2.0"}], mode=selector.SelectSelectorMode.DROPDOWN)
                ),
                vol.Optional("keep_alive", default=options.get("keep_alive", -1), description="Time in minutes to keep model loaded in VRAM (-1 for infinite)."): selector.NumberSelector(selector.NumberSelectorConfig(min=-1, max=1440, mode=NumberSelectorMode.BOX)),
            })
        )

    async def async_step_tool_settings(self, user_input=None):
        """Handle tool blacklists, iterations, caching, and device injection strategies."""
        config = self._options

        if user_input is not None:
            if user_input.get("clear_cache_now"):
                cache_path = Path(__file__).parent / "semantic_cache.json"
                if cache_path.exists():
                    _LOGGER.info("🗑️ Clearing semantic cache via Config Flow...")
                    await self.hass.async_add_executor_job(os.remove, cache_path)
                user_input["clear_cache_now"] = False
            
            self._options.update(user_input)
            return await self.async_step_init()
        
        # 1. Fetch Custom Tools
        def get_custom_tools() -> list[str]:
            tools_dir = Path(__file__).parent / "tools"
            found_tools = []
            if not tools_dir.exists(): return found_tools
            for file_path in tools_dir.glob("*.py"):
                if file_path.name == "__init__.py": continue
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("name =") or line.startswith("name="):
                                match = re.search(r'["\']([^"\']+)["\']', line)
                                if match:
                                    found_tools.append(match.group(1))
                                    break 
                except Exception: pass
            return found_tools

        custom_tools = await self.hass.async_add_executor_job(get_custom_tools)
        all_tools_options = [{"value": t, "label": f"{t} (Native)"} for t in ALL_NATIVE_HA_TOOLS] + [{"value": t, "label": f"{t} (Custom)"} for t in custom_tools]

        # 2. Fetch Home Assistant Areas/Rooms dynamically
        area_reg = ar.async_get(self.hass)
        ha_areas = area_reg.async_list_areas()
        area_options = [{"value": area.id, "label": area.name} for area in ha_areas]

        return self.async_show_form(
            step_id="tool_settings",
            data_schema=vol.Schema({
                vol.Optional("max_tool_iterations", default=config.get("max_tool_iterations", 5)): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=1, max=15, step=1, mode=NumberSelectorMode.BOX)
                ),
                vol.Required("device_injection_strategy", default=config.get("device_injection_strategy", "current_room")): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": "all", "label": "Entire Home (All Exposed Devices)"},
                            {"value": "current_room", "label": "Dynamic (Current Room Only)"},
                            {"value": "specific_rooms", "label": "Static (Only Select Rooms)"}
                        ], mode=selector.SelectSelectorMode.DROPDOWN
                    )
                ),
                vol.Optional("injection_specific_rooms", default=config.get("injection_specific_rooms", [])): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=area_options, multiple=True, mode=selector.SelectSelectorMode.DROPDOWN)
                ),
                vol.Optional("blacklisted_tools", default=config.get("blacklisted_tools", [])): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=all_tools_options, multiple=True, mode=selector.SelectSelectorMode.DROPDOWN, custom_value=True)
                ),
                vol.Optional("tool_injection_limit", default=config.get("tool_injection_limit", 3)): selector.NumberSelector(selector.NumberSelectorConfig(min=1, max=25, mode=NumberSelectorMode.BOX)),
                vol.Optional("tool_cosine_threshold", default=config.get("tool_cosine_threshold", 0.30)): selector.NumberSelector(selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode=NumberSelectorMode.SLIDER)),
                vol.Optional("clear_cache_now", default=False): selector.BooleanSelector()
            })
        )

    async def async_step_memory_settings(self, user_input=None):
        """Step 4: Fact and Memory Injection."""
        if user_input is not None:
            self._options.update(user_input)
            return await self.async_step_init()

        options = self._options
        current_collections = options.get("memory_collections", ["memories_collection"])
        memory_options = [{"value": c, "label": c} for c in current_collections]
        if not any(opt["value"] == "memories_collection" for opt in memory_options):
            memory_options.append({"value": "memories_collection", "label": "memories_collection"})

        return self.async_show_form(
            step_id="memory_settings",
            data_schema=vol.Schema({
                vol.Optional("enable_memory_injection", default=options.get("enable_memory_injection", False), description="Enable dynamic fact and memory retrieval."): selector.BooleanSelector(),
                vol.Optional("memory_collections", default=current_collections, description="Qdrant collections to search for personal memories."): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=memory_options, multiple=True, mode=selector.SelectSelectorMode.DROPDOWN, custom_value=True)
                ),
                vol.Optional("memory_injection_limit", default=options.get("memory_injection_limit", 3), description="Max facts to inject per turn."): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0, max=50, mode=NumberSelectorMode.BOX)
                ),
                vol.Optional("memory_cosine_threshold", default=options.get("memory_cosine_threshold", 0.50), description="Sensitivity for memory semantic search match."): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode=NumberSelectorMode.SLIDER)
                ),
            })
        )

    async def async_step_finish(self, user_input=None):
        """Finalize and Save."""
        return self.async_create_entry(title="Custom AI Options", data=self._options)