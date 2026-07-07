import logging
import os
from pathlib import Path

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
import homeassistant.helpers.llm as ha_llm

from .tools.music import register_media_service
from .alarm_system import AlarmSystem

_LOGGER = logging.getLogger(__name__)

DOMAIN = "ai_tools"
CURRENT_DIR = Path(__file__).parent
ENV_FILE_PATH = CURRENT_DIR / ".env"

def _load_env_vars():
    """Load environment variables from local .env file in background thread."""
    if ENV_FILE_PATH.exists():
        _LOGGER.info(f"Loading local environment variables from {ENV_FILE_PATH}")
        with open(ENV_FILE_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

# =====================================================================
# GLOBAL BASE PROMPT OVERRIDES
# =====================================================================
def _apply_global_prompt_patches():
    """Wipe the native instructions from the core source module directly."""
    _LOGGER.info("✂️ Dynamic Overrides: Blanking out core Home Assistant instructions...")
    
    # Force override the module-level globals to empty strings
    ha_llm.DEVICE_CONTROL_TOOL_USAGE_PROMPT = ""
    ha_llm.DYNAMIC_CONTEXT_PROMPT = ""
    ha_llm.DEFAULT_INSTRUCTIONS_PROMPT = ""
    ha_llm.NO_ENTITIES_PROMPT = ""
    ha_llm.DATE_TIME_PROMPT = ""

    # Replace the satellite area prompt generator
    # We redefine it to return an empty string so Home Assistant drops the text entirely.
    def _empty_area_prompt(*args, **kwargs):
        return ""

    ha_llm.AssistAPI._async_get_voice_satellite_area_prompt = _empty_area_prompt


async def async_setup(hass: HomeAssistant, config: dict):
    """Initial boot setup."""
    await hass.async_add_executor_job(_load_env_vars)
    _apply_global_prompt_patches()
    register_media_service(hass) 
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Set up the Custom AI Agent from a UI Config Entry."""
    await hass.config_entries.async_forward_entry_setups(entry, ["conversation", "ai_task"])
    
    hass.data.setdefault(DOMAIN, {})
    alarm_system = AlarmSystem(hass, entry)
    await alarm_system.async_start()
    
    hass.data[DOMAIN][entry.entry_id] = {"alarm_system": alarm_system}
    
    entry.async_on_unload(entry.add_update_listener(async_update_options))
    return True

async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reloads the integration when the user changes options in the UI."""
    await hass.config_entries.async_reload(entry.entry_id)

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry and kill background tasks."""
    if entry.entry_id in hass.data.get(DOMAIN, {}):
        alarm_system = hass.data[DOMAIN][entry.entry_id]["alarm_system"]
        await alarm_system.async_unload()
        
    return await hass.config_entries.async_unload_platforms(entry, ["conversation", "ai_task"])