import logging
import json
import ollama
from homeassistant.components.ai_task import AITaskEntity, AITaskEntityFeature, GenDataTask, GenDataTaskResult
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util.ssl import get_default_context

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the AI Task entity from a config entry."""
    async_add_entities([CustomAITaskEntity(hass, entry)])

class CustomAITaskEntity(AITaskEntity):
    """Exposes the Local Custom AI to background automation scripts."""

    # Tells Home Assistant this entity supports generating structured data
    _attr_supported_features = AITaskEntityFeature.GENERATE_DATA

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self._attr_name = "Custom AI Task Engine"
        self._attr_unique_id = f"{entry.entry_id}_ai_task"
        
        # Pull your settings from the UI just like agent.py does
        self.ollama_url = entry.options.get("url", "http://192.168.4.23:11434")
        self.api_key = entry.options.get("api_key", "")
        self.model = entry.options.get("model", "qwen2.5:latest")

    async def _async_generate_data(self, task: GenDataTask, chat_log) -> GenDataTaskResult:
        """Handle a background task from an automation service call."""
        
        # 1. Grab the user's prompt from the automation payload
        user_prompt = chat_log.content[-1].content
        
        _LOGGER.info(f"🤖 [AI Task] Received background data generation request: {user_prompt}")

        # 2. Build a fresh, isolated Ollama client
        client = ollama.AsyncClient(
            host=self.ollama_url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None,
            verify=get_default_context()
        )

        # 3. Request the data from Ollama (using the JSON schema if the automation provided one)
        try:
            response = await client.chat(
                model=self.model,
                messages=[{"role": "user", "content": user_prompt}],
                format=task.structure, # Forces Ollama to output matching the requested JSON schema
                stream=False
            )
            
            raw_text = response.get("message", {}).get("content", "")
            
            # 4. If the automation asked for structured data, parse it back into JSON
            if task.structure:
                try:
                    final_data = json.loads(raw_text)
                except json.JSONDecodeError as err:
                    _LOGGER.error(f"❌ Failed to parse Ollama JSON task response: {raw_text}")
                    final_data = {}
            else:
                final_data = raw_text

            return GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=final_data,
            )

        except Exception as e:
            _LOGGER.error(f"❌ AI Task Execution failed: {e}")
            return GenDataTaskResult(conversation_id=chat_log.conversation_id, data={"error": str(e)})