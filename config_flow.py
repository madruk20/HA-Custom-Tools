# Standard Library Imports
import logging
import os
from pathlib import Path
import re

# Third-Party Imports
import voluptuous as vol

# Home Assistant Imports
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import NumberSelectorMode

from .const import (
    DOMAIN, 
    DEFAULT_SYSTEM_PROMPT, 
    DEFAULT_DYNAMIC_SUFFIX, 
    CLOUD_LLM_MODELS, 
    CLOUD_EMBED_MODELS, 
    ALL_NATIVE_HA_TOOLS
)

_LOGGER = logging.getLogger(__name__)

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
            menu_options=[
                "provider_settings", 
                "tuning_settings", 
                "tool_settings", 
                "memory_settings", 
                "alarm_settings",
                "finish",
            ]
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
                vol.Required(
                    "llm_backend_type", 
                    default=llm_backend, 
                    description="Primary AI inference backend."): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": "local_ollama", "label": "Ollama (Local Server)"},
                            {"value": "openai_official", "label": "OpenAI (Official Cloud)"},
                            {"value": "openai_compatible", "label": "OpenAI Compatible"}
                        ], mode=selector.SelectSelectorMode.DROPDOWN)
                ),
                vol.Required(
                    "llm_url", 
                    default=llm_url, 
                    description="Address of the LLM API."
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.URL
                        )
                    ),
                vol.Optional(
                    "llm_api_key", 
                    default=llm_api_key, 
                    description="Required for Cloud or restricted local APIs."
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.PASSWORD
                        )
                    ),
                vol.Required(
                    "llm_model", 
                    default=options.get("llm_model", llm_models[0]), 
                    description="The specific AI model to load."
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=llm_models, 
                            mode=selector.SelectSelectorMode.DROPDOWN, 
                            custom_value=True
                        )
                    ),
                vol.Required(
                    "embed_backend_type", 
                    default=options.get("embed_backend_type", llm_backend), 
                    description="Backend used specifically for embeddings/RAG."
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"value": "none", "label": "None (Disable RAG)"},
                                {"value": "local_ollama", "label": "Ollama (Local Embeddings)"},
                                {"value": "openai_official", "label": "OpenAI (Cloud Embeddings)"},
                                {"value": "openai_compatible", "label": "OpenAI Compatible"}
                            ], mode=selector.SelectSelectorMode.DROPDOWN
                        )
                    ),
                vol.Required(
                    "embed_url", 
                    default=options.get("embed_url", llm_url), 
                    description="Address for the embedding API."
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.URL
                        )
                    ),
                vol.Optional(
                    "embed_api_key", 
                    default=options.get("embed_api_key", llm_api_key), 
                    description="API Key for embeddings."
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.PASSWORD
                        )
                    ),
                vol.Required(
                    "embedding_model", 
                    default=options.get("embedding_model", embed_models[0] if embed_models else "qwen-embed-2k:latest"), 
                    description="Model used for vectorizing text."
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=embed_models, 
                            mode=selector.SelectSelectorMode.DROPDOWN, 
                            custom_value=True
                        )
                    ),
                vol.Required(
                    "vector_db_backend", 
                    default=options.get("vector_db_backend", "qdrant"), 
                    description="Vector Database software."
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[{"value": "none", "label": "None"}, 
                                     {"value": "qdrant", "label": "Qdrant"}], 
                                     mode=selector.SelectSelectorMode.DROPDOWN
                        )
                    ),
                vol.Required(
                    "vector_db_url", 
                    default=options.get("vector_db_url", "http://localhost:6333"), 
                    description="Address of the Vector DB."): selector.TextSelector(selector.TextSelectorConfig(type=selector.TextSelectorType.URL)),
                vol.Optional(
                    "vector_db_api_key", 
                    default=options.get("vector_db_api_key", ""), 
                    description="Vector DB Auth Key."
                    ): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.PASSWORD
                        )
                    )
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
                vol.Optional(
                    "Instructions", 
                    default=options.get("Instructions", DEFAULT_SYSTEM_PROMPT), 
                    description="The core System Prompt defining agent behavior."
                    ): selector.TemplateSelector(),
                vol.Optional(
                    "dynamic_suffix", 
                    default=options.get("dynamic_suffix", DEFAULT_DYNAMIC_SUFFIX)
                    ): selector.TemplateSelector(),
                vol.Optional(
                    "thinking", 
                    default=options.get("thinking", False), 
                    description="Enable <think> tags for supported reasoning models."
                    ): selector.BooleanSelector(),
                vol.Optional(
                    "enable_streaming", 
                    default=options.get("enable_streaming", True), 
                    description="Enable text streaming (Disable for incompatible models)"
                    ): selector.BooleanSelector(),
                vol.Optional(
                    "enable_parallel_tools", 
                    default=options.get("enable_parallel_tools", True), 
                    description="Enable parallel tool execution"
                    ): selector.BooleanSelector(),
                vol.Optional(
                    "draft_num_predict", 
                    default=options.get("draft_num_predict", 2), 
                    description="Number of speculative tokens to draft. Set to 0 to disable MTP."
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, 
                        max=6, 
                        step=1, 
                        mode=NumberSelectorMode.SLIDER
                    )
                ),   
                vol.Optional(
                    "temperature", 
                    default=options.get("temperature", 0.5), 
                    description="Creativity/Randomness of the response."
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, 
                        max=1, 
                        step=0.1, 
                        mode=NumberSelectorMode.SLIDER
                    )
                ), 
                vol.Optional(
                    "top_p", default=options.get("top_p", 0.9), 
                    description="Nucleus sampling probability."
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0, 
                        max=1.0, 
                        step=0.05, 
                        mode=NumberSelectorMode.SLIDER
                    )
                ),
                vol.Optional(
                    "repeat_penalty", 
                    default=options.get("repeat_penalty", 1.1), 
                    description="Prevents the AI from repeating itself."
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1.0, 
                            max=2.0, 
                            step=0.05, 
                            mode=NumberSelectorMode.SLIDER
                        )
                ),
                vol.Optional(
                    "top_k", default=options.get("top_k", 40), 
                    description="Limit token selection to top K choices."
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1, 
                            max=100, 
                            mode=NumberSelectorMode.BOX
                        )
                    ),
                vol.Optional(
                    "max_history", 
                    default=options.get("max_history", 10), 
                    description="Max messages retained in session memory."
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0, 
                            max=100, 
                            mode=NumberSelectorMode.BOX
                        )
                    ),
                vol.Optional(
                    "num_ctx", 
                    default=options.get("num_ctx", 8192), 
                    description="Maximum context window size."
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=2048, 
                            max=32768, 
                            mode=NumberSelectorMode.BOX
                        )
                    ),
                vol.Optional(
                    "num_predict", 
                    default=options.get("num_predict", 512), 
                    description="Max tokens to generate per response."
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=128, 
                            max=4096, 
                            mode=NumberSelectorMode.BOX
                        )
                    ),
                vol.Optional(
                    "keep_alive", 
                    default=options.get("keep_alive", -1), 
                    description="Time in minutes to keep model loaded in VRAM (-1 for infinite)."
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=-1, 
                            max=1440, 
                            mode=NumberSelectorMode.BOX
                        )
                    )
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
                vol.Optional(
                    "max_tool_iterations", 
                    default=config.get("max_tool_iterations", 5)
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1, 
                            max=15, 
                            step=1, 
                            mode=NumberSelectorMode.BOX
                        )
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
                vol.Optional(
                    "injection_specific_rooms", 
                    default=config.get("injection_specific_rooms", [])
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=area_options, 
                            multiple=True, 
                            mode=selector.SelectSelectorMode.DROPDOWN
                        )
                ),
                vol.Optional(
                    "blacklisted_tools", 
                    default=config.get("blacklisted_tools", [])
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=all_tools_options, 
                            multiple=True, 
                            mode=selector.SelectSelectorMode.DROPDOWN, 
                            custom_value=True
                        )
                ),
                vol.Optional(
                    "tool_injection_limit", 
                    default=config.get("tool_injection_limit", 3)
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1, 
                            max=25, 
                            mode=NumberSelectorMode.BOX
                        )
                ),
                vol.Optional(
                    "tool_cosine_threshold", 
                    default=config.get("tool_cosine_threshold", 0.30)
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0, 
                            max=1.0, 
                            step=0.05, 
                            mode=NumberSelectorMode.SLIDER
                        )
                ),
                vol.Optional(
                    "clear_cache_now", 
                    default=False
                    ): selector.BooleanSelector()
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
                vol.Optional(
                    "enable_memory_injection", 
                    default=options.get("enable_memory_injection", False), 
                    description="Enable dynamic fact and memory retrieval."
                    ): selector.BooleanSelector(),
                vol.Optional(
                    "memory_collections", 
                    default=current_collections, 
                    description="Qdrant collections to search for personal memories."
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=memory_options, 
                            multiple=True, 
                            mode=selector.SelectSelectorMode.DROPDOWN, 
                            custom_value=True
                        )
                ),
                vol.Optional(
                    "memory_injection_limit", 
                    default=options.get("memory_injection_limit", 3),
                    description="Max facts to inject per turn."
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0,
                            max=50, 
                            mode=NumberSelectorMode.BOX
                        )
                ),
                vol.Optional(
                    "memory_cosine_threshold", 
                    default=options.get("memory_cosine_threshold", 0.50), 
                    description="Sensitivity for memory semantic search match."
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0, 
                            max=1.0, 
                            step=0.05, 
                            mode=NumberSelectorMode.SLIDER
                        )
                ),
            })
        )

    async def async_step_alarm_settings(self, user_input=None):
        """Dynamic room selection menu."""
        if user_input is not None:
            self.selected_alarm_room = user_input["room"]
            return await self.async_step_alarm_configure_room()

        area_reg = ar.async_get(self.hass)
        ha_areas = area_reg.async_list_areas()
        
        # Build the dropdown dynamically based on actual hardware layout
        room_options = [{"value": area.name.lower().strip(), "label": area.name} for area in ha_areas]

        return self.async_show_form(
            step_id="alarm_settings",
            data_schema=vol.Schema({
                vol.Required("room"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=room_options,
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                )
            })
        )

    async def async_step_alarm_configure_room(self, user_input=None):
        room = self.selected_alarm_room
        alarms_config = {k: dict(v) for k, v in self._options.get("alarms_config", {}).items()}
        room_data = alarms_config.get(room, {})

        if user_input is not None:
            # INTERCEPT "NONE" SELECTION AND WIPE IT BLANK
            if user_input.get("fallback_notify") == "none":
                user_input["fallback_notify"] = ""
                
            alarms_config[room] = user_input
            new_options = dict(self._options)
            new_options["alarms_config"] = alarms_config
            self._options = new_options
            return await self.async_step_init()

        # --- DYNAMIC MEDIA PLAYER FILTERING ---
        area_reg = ar.async_get(self.hass)
        dev_reg = dr.async_get(self.hass)
        ent_reg = er.async_get(self.hass)

        target_area_id = None
        for area in area_reg.async_list_areas():
            if area.name.lower().strip() == room:
                target_area_id = area.id
                break

        raw_media_players = []
        if target_area_id:
            for entity in er.async_entries_for_area(ent_reg, target_area_id):
                # Ignore hidden entities or background helpers
                if entity.domain == "media_player" and not entity.hidden and not entity.entity_category:
                    raw_media_players.append(entity)
                    
            for device in dr.async_entries_for_area(dev_reg, target_area_id):
                for entity in er.async_entries_for_device(ent_reg, device.id):
                    if entity.domain == "media_player" and not entity.hidden and not entity.entity_category:
                        raw_media_players.append(entity)

        # Remove duplicates
        unique_mps = {e.entity_id: e for e in raw_media_players}.values()

        # Isolate Music Assistant Players
        ma_players = []
        for e in unique_mps:
            # Check native platform registration or custom attributes
            if e.platform == "mass":
                ma_players.append(e.entity_id)
            else:
                state = self.hass.states.get(e.entity_id)
                if state and "mass_player_id" in state.attributes:
                    ma_players.append(e.entity_id)

        # Build dropdown options using friendly names and tag MA players
        mp_options = []
        for e in unique_mps:
            state = self.hass.states.get(e.entity_id)
            name = state.name if state else e.entity_id
            
            # Tag Music Assistant players so the user knows which is which
            is_ma = e.platform == "mass" or (state and "mass_player_id" in state.attributes)
            if is_ma:
                name = f"{name} (Music Assistant)"
                
            mp_options.append({"value": e.entity_id, "label": name})

        if not mp_options:
            mp_options = [{"value": "", "label": "No media players found in this room"}]
            
        # --- DYNAMIC NOTIFY FILTERING ---
        notify_options = [{"value": "none", "label": "🚫 None (Disable Fallback)"}]
        current_fallback = room_data.get("fallback_notify", "")
        
        # Correctly iterate through the registry values to filter by domain
        for entity in ent_reg.entities.values():
            if entity.domain == "notify" and not entity.hidden:
                state = self.hass.states.get(entity.entity_id)
                name = state.name if state else (entity.name or entity.original_name or entity.entity_id)
                notify_options.append({"value": entity.entity_id, "label": name})
                
        # Inject the current fallback if it's a legacy service not caught by the entity registry
        if current_fallback and current_fallback != "none" and not any(opt["value"] == current_fallback for opt in notify_options):
            notify_options.append({"value": current_fallback, "label": current_fallback})
            
        fallback_default = current_fallback if current_fallback else "none"

        return self.async_show_form(
            step_id="alarm_configure_room",
            data_schema=vol.Schema({
                vol.Required("enabled", default=room_data.get("enabled", False)): selector.BooleanSelector(),
                vol.Required("time", default=room_data.get("time", "07:00:00")): selector.TimeSelector(),
                vol.Required("volume", default=room_data.get("volume", 0.7)): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.1, mode=selector.NumberSelectorMode.SLIDER)
                ),
                vol.Required("media_player", default=room_data.get("media_player", mp_options[0]["value"])): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=mp_options, mode=selector.SelectSelectorMode.DROPDOWN, custom_value=True)
                ),
                vol.Required("song", default=room_data.get("song", "")): selector.TextSelector(),
                
                vol.Optional("wake_entity", description={"suggested_value": room_data.get("wake_entity", "")}): selector.EntitySelector(),
                
                vol.Required("fallback_notify", default=fallback_default): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=notify_options, mode=selector.SelectSelectorMode.DROPDOWN, custom_value=True)
                )
            }),
            description_placeholders={"room_name": room.title()}
        )

    async def async_step_finish(self, user_input=None):
        """Finalize and Save."""
        return self.async_create_entry(title="Custom AI Options", data=self._options)