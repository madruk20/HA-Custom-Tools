import logging
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.helpers import selector
from homeassistant.helpers.selector import NumberSelectorMode
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util.ssl import get_default_context

DOMAIN = "ai_tools"
_LOGGER = logging.getLogger(__name__)

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

    async def async_step_init(self, user_input=None):
        options = self.config_entry.options
        errors = {}
        
        # Get current backend type or default to local Ollama
        backend_type = user_input.get("backend_type") if user_input else options.get("backend_type", "local_ollama")
        ollama_url = user_input.get("ollama_url") if user_input else options.get("ollama_url", "http://192.168.4.23:11434")
        llm_api_key = user_input.get("llm_api_key") if user_input else options.get("llm_api_key", "")

        # 1. Conditional Connection Validation
        if user_input is not None:
            # ONLY validate tags if they are actually trying to use a local Ollama instance
            if backend_type == "local_ollama":
                try:
                    session = async_get_clientsession(self.hass)
                    headers = {"Authorization": f"Bearer {llm_api_key}"} if llm_api_key else {}
                    ssl_context = get_default_context() if ollama_url.startswith("https") else False
                    
                    async with session.get(f"{ollama_url.rstrip('/')}/api/tags", headers=headers, ssl=ssl_context, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            available_models = [m["name"] for m in data.get("models", [])]
                            if user_input.get("ollama_model") not in available_models:
                                errors["ollama_model"] = "model_not_found"
                        else:
                            errors["base"] = "cannot_connect"
                except Exception as e:
                    _LOGGER.error(f"Ollama Validation failed: {e}")
                    errors["base"] = "cannot_connect"
            
            if not errors:
                return self.async_create_entry(title="", data=user_input)

        # 2. Populate Dropdown Lists dynamically based on current backend selection
        llm_models = []
        embed_models = []

        if backend_type == "local_ollama":
            try:
                session = async_get_clientsession(self.hass)
                headers = {"Authorization": f"Bearer {llm_api_key}"} if llm_api_key else {}
                async with session.get(f"{ollama_url.rstrip('/')}/api/tags", headers=headers, timeout=5) as response:
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
                vol.Required("backend_type", default=backend_type): selector.SelectSelector(
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
                    "ollama_url", 
                    default=ollama_url
                ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.URL)),

                vol.Optional(
                    "llm_api_key", 
                    default=llm_api_key
                ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)),

                vol.Optional(
                    "qdrant_url", 
                    default=options.get("qdrant_url", "http://192.168.4.23:6333")
                ): selector.TextSelector(
                    selector.TextSelectorConfig(type=selector.TextSelectorType.URL)),

                vol.Optional(
                    "qdrant_api_key", 
                    default=options.get("qdrant_api_key", "")
                ): selector.TextSelector(
                    selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)), 

                vol.Required(
                    "ollama_model", 
                    default=user_input.get("ollama_model") if user_input else options.get("ollama_model", llm_models[0])
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=llm_models, 
                        mode=selector.SelectSelectorMode.DROPDOWN,
                        custom_value=True
                    )
                ),

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
            })
        )