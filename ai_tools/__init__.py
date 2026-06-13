import logging
import os
from pathlib import Path

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry

from .tools.music import register_media_service

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

async def async_setup(hass: HomeAssistant, config: dict):
    """Initial boot setup."""
    await hass.async_add_executor_job(_load_env_vars)
    register_media_service(hass) 
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Set up the Custom AI Agent from a UI Config Entry."""
    
    # 1. Forward the setup to conversation.py to register it as an official entity
    await hass.config_entries.async_forward_entry_setups(entry, ["conversation", "ai_task"])
    
    # 2. Listen for options updates from the config flow UI
    entry.async_on_unload(entry.add_update_listener(async_update_options))
    
    _LOGGER.info("✅ AI Tools Agent UI Setup Complete.")
    return True

async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reloads the integration when the user changes options in the UI."""
    await hass.config_entries.async_reload(entry.entry_id)

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, ["conversation", "ai_task"])