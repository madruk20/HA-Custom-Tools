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
    "- MUSIC: Use `music_player` tool to control playing music on any media player or speaker."
)

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
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        options = self.config_entry.options
        
        # Read URL and API Key from options (with fallbacks)
        ollama_url = options.get("url", "http://192.168.4.23:11434")
        api_key = options.get("api_key", "")
        
        available_models = []
        try:
            session = async_get_clientsession(self.hass)
            
            # Setup Authentication Headers and SSL Context
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            ssl_context = get_default_context() if ollama_url.startswith("https") else False
            
            # Query the dynamic URL securely
            async with session.get(f"{ollama_url.rstrip('/')}/api/tags", headers=headers, ssl=ssl_context, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    available_models = [m["name"] for m in data.get("models", [])]
                else:
                    _LOGGER.warning(f"Ollama API returned status: {response.status}")
        except Exception as e:
            _LOGGER.error(f"Failed to fetch Ollama models for dropdown: {str(e)}")

        # Fallback if the server is offline or empty
        if not available_models:
            available_models = ["offline_fallback:latest"]

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required(
                    "ollama_url", 
                    default=ollama_url
                ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.URL)),

                vol.Optional(
                    "qdrant_url", 
                    default=options.get("qdrant_url", "http://192.168.4.23:6333")
                ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.URL)),
                
                vol.Optional(
                    "api_key", 
                    default=api_key
                ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)),

                vol.Required(
                    "ollama_model", 
                    default=options.get("model", available_models[0])
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=available_models, mode=selector.SelectSelectorMode.DROPDOWN)
                ),

                vol.Optional(
                    "embedding_model", 
                    default=options.get("embedding_model", "qwen-embed-2k:latest")
                ): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.TEXT)),

                vol.Optional(
                    "Instructions", 
                    default=options.get("Instructions", DEFAULT_SYSTEM_PROMPT)
                ): selector.TemplateSelector(),
                
                vol.Optional(
                    "Thinking", 
                    default=options.get("thinking", False)
                ): selector.BooleanSelector(),
                
                vol.Optional(
                    "Temperature", 
                    default=options.get("temperature", 0.5)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0, max=1, step=0.1, mode=NumberSelectorMode.SLIDER)
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
                # Keep Alive restricted to numbers (minutes). Default is -1.
                vol.Optional(
                    "keep_alive", 
                    default=options.get("keep_alive", -1)
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=-1, max=1440, mode=NumberSelectorMode.BOX)
                ),
            })
        )