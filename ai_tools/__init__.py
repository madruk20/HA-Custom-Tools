import os
import json
import logging
import re
from pathlib import Path

# Centralized Home Assistant Imports
import homeassistant.util.dt as dt_util
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
import homeassistant.helpers.device_registry as dr
import homeassistant.helpers.area_registry as ar

# Import custom tools
from .tools.web_search_brave import WebSearchTool
from .tools.alarms import AlarmManagerTool
from .tools.price_lookup import PriceLookupTool
from .tools.music import MusicPlayerTool, register_media_service

_LOGGER = logging.getLogger(__name__)


# ==========================================
# 0.2 OLLAMA HTTP PAYLOAD PATCH (STREAM AWARE)
# ==========================================
try:
    import ollama
except ImportError:
    ollama = None

def apply_ollama_client_patch():
    """Intercepts outgoing payload, normalizes to dicts, scrubs HA redundancies, and kills thinking tokens."""
    if ollama is None:
        _LOGGER.debug("Ollama python client not found, skipping patch.")
        return

    try:
        if not hasattr(ollama.AsyncClient, "chat"):
            _LOGGER.warning("Ollama AsyncClient does not have 'chat' method. Skipping patch.")
            return
        
        original_chat = ollama.AsyncClient.chat
        
        async def patched_chat(self, model, messages=None, **kwargs):
            
            # --- THE OUTGOING PAYLOAD SCRUBBER ---
            if messages:
                normalized_messages = []
                for msg in messages:
                    # Force conversion of Pydantic objects or custom classes to clean dicts
                    if hasattr(msg, "model_dump"):
                        m_dict = msg.model_dump()
                    elif hasattr(msg, "__dict__"):
                        m_dict = {k: v for k, v in msg.__dict__.items() if v is not None}
                    elif isinstance(msg, dict):
                        m_dict = msg.copy()
                    else:
                        m_dict = dict(msg)
                    
                    # 1. SYSTEM PROMPT SCRUBBER (Dynamic HA removal + Clock Erasure)
                    if m_dict.get("role") == "system" and isinstance(m_dict.get("content"), str):
                        sys_text = m_dict["content"]
                        
                        # A. Future-proof removal of redundant core instructions
                        from homeassistant.helpers import llm as ha_llm
                        # Remove tool usage rules
                        if hasattr(ha_llm, "DEVICE_CONTROL_TOOL_USAGE_PROMPT"):
                            sys_text = sys_text.replace(ha_llm.DEVICE_CONTROL_TOOL_USAGE_PROMPT, "")
                        # Remove live context instructions
                        if hasattr(ha_llm, "DYNAMIC_CONTEXT_PROMPT"):
                            sys_text = sys_text.replace(ha_llm.DYNAMIC_CONTEXT_PROMPT, "")
                        
                        # B. Erase the exact-second volatile clock
                        sys_text = re.sub(
                            r"\s*Current time is \d{2}:\d{2}:\d{2}\. Today's date is \d{4}-\d{2}-\d{2}\.", 
                            "", 
                            sys_text
                        )
                        
                        m_dict["content"] = sys_text.strip()
                    # 2. ASSISTANT THINKING SCRUBBER (Removes raw <think> tags)
                    if m_dict.get("role") == "assistant" and isinstance(m_dict.get("content"), str):
                        m_dict.pop("thinking", None)
                        m_dict.pop("thinking_content", None)
                        m_dict["content"] = re.sub(r'<think>.*?</think>', '', m_dict["content"], flags=re.DOTALL).strip()
                        
                    normalized_messages.append(m_dict)
                
                # Override the outgoing messages payload with our cleanly scrubbed dictionaries
                messages = normalized_messages
                            
            # --- TRANSMIT REQUEST TO API ENGINE ---
            is_stream = kwargs.get("stream", False)
            # Log the system prompt sent to ollama
            #try:
            #    for msg in messages:
            #        if msg.get("role") == "system":
            #            _LOGGER.info(f"🔍 FINAL OLLAMA SYSTEM PROMPT:\n{msg.get('content')}")
            #            break
            #except Exception:
            #    pass
            
            response = await original_chat(self, model=model, messages=messages, **kwargs)
            
            # --- INCOMING METRICS CAPTURE ---
            if not is_stream:
                try:
                    p_tokens = response.get("prompt_eval_count", 0) if isinstance(response, dict) else getattr(response, "prompt_eval_count", 0)
                    p_time = (response.get("prompt_eval_duration", 0) if isinstance(response, dict) else getattr(response, "prompt_eval_duration", 0)) / 1e9
                    gen_tokens = response.get("eval_count", 0) if isinstance(response, dict) else getattr(response, "eval_count", 0)
                    total_time = (response.get("total_duration", 0) if isinstance(response, dict) else getattr(response, "total_duration", 0)) / 1e9
                    
                    # Visual Cache Hit Detector
                    cache_status = f"✅ CACHED ({p_time:.2f}s)" if p_time < 0.15 else f"❌ MISSED ({p_time:.2f}s)"
                    speed = f"{(gen_tokens / (total_time - p_time)):.2f} t/s" if (total_time - p_time) > 0 else "N/A"
                    
                    # === UPDATED: ADDED TOTAL TIME BACK ===
                    _LOGGER.info(f"🤖 OLLAMA VERBOSE: Prompt: {p_tokens}t {cache_status} | Generated: {gen_tokens}t | Speed: {speed} | Total: {total_time:.2f}s")
                except Exception:
                    pass 
                return response
                
            else:
                async def stream_wrapper(async_gen):
                    async for chunk in async_gen:
                        try:
                            p_tokens = chunk.get("prompt_eval_count") if isinstance(chunk, dict) else getattr(chunk, "prompt_eval_count", None)
                            if p_tokens: 
                                p_time = (chunk.get("prompt_eval_duration", 0) if isinstance(chunk, dict) else getattr(chunk, "prompt_eval_duration", 0)) / 1e9
                                gen_tokens = chunk.get("eval_count", 0) if isinstance(chunk, dict) else getattr(chunk, "eval_count", 0)
                                total_time = (chunk.get("total_duration", 0) if isinstance(chunk, dict) else getattr(chunk, "total_duration", 0)) / 1e9
                                
                                cache_status = f"✅ CACHED ({p_time:.2f}s)" if p_time < 0.15 else f"❌ MISSED ({p_time:.2f}s)"
                                speed = f"{(gen_tokens / (total_time - p_time)):.2f} t/s" if (total_time - p_time) > 0 else "N/A"
                                
                                # === UPDATED: ADDED TOTAL TIME BACK ===
                                _LOGGER.info(f"🤖 OLLAMA VERBOSE: Prompt: {p_tokens}t {cache_status} | Generated: {gen_tokens}t | Speed: {speed} | Total: {total_time:.2f}s")
                        except Exception:
                            pass
                        yield chunk 
                return stream_wrapper(response)
        ollama.AsyncClient.chat = patched_chat
        _LOGGER.info("Successfully patched Ollama client for Token Scrubbing, Redundancy Erasure, and Volatile Clocks!")
        
    except Exception as e:
        _LOGGER.error(f"Failed to patch Ollama client: {e}")

apply_ollama_client_patch()

# ==========================================
# 0.3 ASSIST PROMPT STRUCTURAL FILTER PATCH
# ==========================================
def apply_prompt_text_filter_patch():
    """Intercepts and physically strips non-local room entity lines from the final system prompt string."""
    try:
        if not hasattr(llm.AssistAPI, "_async_get_api_prompt"):
            _LOGGER.warning("AssistAPI._async_get_api_prompt missing! Skipping patch.")
            return

        original_get_api_prompt = llm.AssistAPI._async_get_api_prompt

        def patched_get_api_prompt(self, llm_context, exposed_entities):
            # 1. Let the native engine assemble the complete text payload first
            base_prompt = original_get_api_prompt(self, llm_context, exposed_entities)
            if not base_prompt or not isinstance(base_prompt, str):
                return base_prompt
            
            base_prompt = re.sub(r"Current time is \d{2}:\d{2}:\d{2}\. Today's date is \d{4}-\d{2}-\d{2}\.", "", base_prompt)

            # 2. Only proceed with stripping if we have a valid microphone context
            if llm_context and llm_context.device_id:
                try:
                    hass = self.hass if hasattr(self, 'hass') else None
                    if hass:
                        dev_reg = dr.async_get(hass)
                        area_reg = ar.async_get(hass)
                        
                        device = dev_reg.async_get(llm_context.device_id)
                        if device and device.area_id:
                            active_area = area_reg.async_get_area(device.area_id)
                            active_area_name = active_area.name.lower().strip() if active_area else ""
                            
                            if active_area_name:
                                # Home Assistant builds lists formatted as:
                                # - names: Device Name
                                #   domain: light
                                #   areas: Room Name
                                # This regex captures that entire 3-line block cleanly.
                                pattern = r'-\s+names:[^\n]+\n\s+domain:[^\n]+\n\s+areas:\s+([^\n]+)\n'
                                
                                def room_evaluator(match):
                                    mentioned_area = match.group(1).lower().strip()
                                    # If the device's area doesn't match our active room, return an empty string to erase it
                                    if mentioned_area != active_area_name:
                                        return ""
                                    return match.group(0) # Keep it if it matches
                                
                                # Process the text block through the evaluator
                                cleaned_prompt = re.sub(pattern, room_evaluator, base_prompt)
                                
                                # Clean up extra newline spacing artifacts left behind by erased lines
                                cleaned_prompt = re.sub(r'\n\s*\n', '\n', cleaned_prompt)
                                
                                _LOGGER.info(f"✂️ Physical Prompt Pruning Active! Stripped all non-{active_area_name} entities from raw context string.")
                                return cleaned_prompt
                                
                except Exception as err:
                    _LOGGER.error(f"Text-based context filtering routine failed: {err}")
            
            return base_prompt

        llm.AssistAPI._async_get_api_prompt = patched_get_api_prompt
        _LOGGER.info("Successfully deployed high-performance Text Prompt Pruning to AssistAPI!")
            
    except Exception as e:
        _LOGGER.error(f"Critical failure patching AssistAPI prompt string layout: {e}")

apply_prompt_text_filter_patch()


# ==========================================
# 0.4 ASSIST TOOL COGNITIVE SLIMMING PATCH
# ==========================================
def apply_tool_filter_patch():
    """Globally removes redundant native tools to keep the model's focus optimized."""
    try:
        if not hasattr(llm.AssistAPI, "_async_get_tools"):
            _LOGGER.warning("AssistAPI._async_get_tools missing! Skipping tool patch.")
            return

        original_get_tools = llm.AssistAPI._async_get_tools

        def patched_get_tools(self, llm_context, exposed_entities):
            # 1. Let the native engine assemble the core tool list first
            tools = original_get_tools(self, llm_context, exposed_entities)
            
            # 2. List of redundant native tool names 
            built_in_blacklist = [
                "HassMediaSearchAndPlay",
                "HassMediaPause",         
                "HassMediaUnpause",       
                "HassMediaNext",         
                "HassMediaPrevious",      
                "HassSetVolume",          
                "HassSetVolumeRelative",
                "HassMediaPlayerMute", 
                "HassMediaPlayerUnmute",
                "HassCancelAllTimers", 
                "HassIncreaseTimer",
                "HassDecreaseTimer", 
                "HassPauseTimer",
                "HassUnpauseTimer",
                "GetDateTime" 
            ]
            
            # 3. Filter out any matching tools from the active toolbelt
            pruned_tools = [t for t in tools if t.name not in built_in_blacklist]
            
            _LOGGER.info(f"🧹 Tool Belt Slimmed: Hidden {len(built_in_blacklist)} redundant native tools.")
            return pruned_tools

        llm.AssistAPI._async_get_tools = patched_get_tools
        _LOGGER.info("Successfully deployed tool selection pruning to AssistAPI!")
            
    except Exception as e:
        _LOGGER.error(f"Critical failure patching AssistAPI tools: {e}")

apply_tool_filter_patch()


# ==========================================
# SELF-CONTAINED .ENV PARSER
# ==========================================
CURRENT_DIR = Path(__file__).parent
ENV_FILE_PATH = CURRENT_DIR / ".env"

if ENV_FILE_PATH.exists():
    _LOGGER.info(f"Loading local environment variables from {ENV_FILE_PATH}")
    with open(ENV_FILE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                clean_key = key.strip()
                clean_value = value.strip().strip('"').strip("'")
                os.environ[clean_key] = clean_value
else:
    _LOGGER.warning(f"No local .env file found at {ENV_FILE_PATH}")


# ==========================================
# API WORKSPACE WRAPPER
# ==========================================
class AiToolsAPI(llm.API):
    """
    Finalizes context adding custom tools, speaker location, and dynamic
    eleements to the end of the prompt for cache efficiency
    """
    id = "custom"
    name = "Custom API"

    def __init__(self, hass: HomeAssistant):
        self.hass = hass

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
            """Return the API instance with custom tools and defensive context lookups."""
            # Define custom tools
            tools = [
                WebSearchTool(), 
                AlarmManagerTool(), 
                PriceLookupTool(), 
                MusicPlayerTool()
            ]

            location_name = "Unknown"
            
            # Find speaker room location
            # Filter devices for only the room the user is in
            if llm_context.device_id:
                try:
                    dev_reg = dr.async_get(self.hass)
                    area_reg = ar.async_get(self.hass)
                    device = dev_reg.async_get(llm_context.device_id)
                    
                    if device and device.area_id:
                        area = area_reg.async_get_area(device.area_id)
                        if area:
                            location_name = area.name
                except Exception as e:
                    _LOGGER.error("Error retrieving registry context: %s", e)

            # Filter date for proper pronunciation
            now = dt_util.now()
            day = now.day
            if 11 <= day <= 13:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

            day_str = f"{now.strftime('%B')} {day}{suffix}, {now.strftime('%Y')}"

            # Add final prompt with dynamic elements for cache efficiency
            static_prompt = (
                f"Physical Location: You are physically located in the {location_name}. If the request does not state a room and one is required, default to this room. "
                f"If the user requests context or operations for an entity outside of the {location_name}, you MUST call `assist__GetLiveContext` for that specific area first to discover it. "
                f"Current Context: Today is {now.strftime('%A')}, {day_str} and the current time is {now.strftime('%-I:%M %p')}. "
            )
            
            return llm.APIInstance(
                api=self, 
                api_prompt=static_prompt,
                llm_context=llm_context,
                tools=tools
            )

async def async_setup(hass: HomeAssistant, config: dict):
    # Register custom AI tools and intents
    llm.async_register_api(hass, AiToolsAPI(hass))
    register_media_service(hass)
    return True