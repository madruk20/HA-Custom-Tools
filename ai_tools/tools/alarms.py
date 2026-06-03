import logging
import re
from datetime import datetime
import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
import homeassistant.helpers.device_registry as dr
import homeassistant.helpers.area_registry as ar

_LOGGER = logging.getLogger(__name__)

class AlarmManagerTool(llm.Tool):
    """Tool to set and cancel alarms"""
    name = "alarm_manager"
    description = "Manage alarms. Set a new alarm or cancel existing ones. Leave 'room' blank if unspecified."
    
    parameters = vol.Schema({
        vol.Required(
            "action", 
            description="The type of modification to make. Must be 'set' or 'cancel'."
        ): vol.In(["set", "cancel"]),
        vol.Optional(
            "time", 
            description="The time requested (e.g. '7:30 AM', '18:45'). Pass exactly as user spoke. Only required if action is 'set'."
        ): str,
        vol.Optional(
            "room", 
            description="The room containing the alarm entity ('bedroom' or 'playroom'). Leave blank if unnamed."
        ): str  
    })

    def _sanitize_time(self, time_str: str) -> str:
        """Deep cleans whisper-transcribed text to a strict HH:MM:00 format."""
        if not time_str:
            return ""
        
        # Normalize 'a.m.' / 'p.m.' variants -> 'am' / 'pm'
        clean = re.sub(r'([ap])\.?m\.?', r'\1m', time_str.lower().strip())
        
        # Fix mashed numbers from STT (e.g., "109 am" -> "1:09 am", "1115" -> "11:15")
        if ":" not in clean:
            clean = re.sub(r'\b(\d{1,2})(\d{2})\b', r'\1:\2', clean)
        
        # Normalize dots/spaces (e.g., '10.29' or '10 29') to colons ('10:29')
        clean = re.sub(r'(\d+)[\.\s](\d+)', r'\1:\2', clean)
        
        # Try to parse with common variations
        for fmt in ("%I:%M %p", "%I:%M:%S %p", "%H:%M", "%H:%M:%S", "%I:%M"):
            try:
                # Force 24-hour HH:MM:00 return
                return datetime.strptime(clean, fmt).strftime("%H:%M:00")
            except ValueError:
                continue
        
        return time_str  # If all else fails, return raw

    async def async_call(self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext) -> dict:
        action = tool_input.tool_args.get("action")
        raw_room = tool_input.tool_args.get("room", "").strip().lower()
        
        valid_rooms = ["bedroom", "playroom"]
        target_room = None

        # Check explicit LLM Input
        if raw_room in valid_rooms:
            target_room = raw_room
        else:
            # Resolve via Hardware Context
            if llm_context.device_id:
                dev_reg = dr.async_get(hass)
                area_reg = ar.async_get(hass)
                
                device = dev_reg.async_get(llm_context.device_id)
                if device and device.area_id:
                    area = area_reg.async_get_area(device.area_id)
                    if area and area.name:
                        area_clean = area.name.lower().strip()
                        if area_clean in valid_rooms:
                            target_room = area_clean
            
            # Final Safe Fallback
            if not target_room:
                target_room = "bedroom"

        # Execute action with resolved room
        if action == "set":
            raw_time = tool_input.tool_args.get("time")
            if not raw_time: 
                return {"error": "Time is required to set an alarm."}
            
            clean_time = self._sanitize_time(raw_time)
            try:
                # Update the datetime helper
                await hass.services.async_call("input_datetime", "set_datetime", 
                    {"entity_id": f"input_datetime.voice_alarm_{target_room}", "time": clean_time}, 
                    blocking=True, context=llm_context.context)
                
                # Enable the alarm
                await hass.services.async_call("input_boolean", "turn_on", 
                    {"entity_id": f"input_boolean.voice_alarm_enabled_{target_room}"}, 
                    blocking=True, context=llm_context.context)
                
                return {"result": f"Successfully set and enabled {target_room} alarm for {clean_time}."}
            except Exception as e:
                _LOGGER.error(f"Set Time or Enable Alarm service call failed: {e}")
                return {"error": f"Set Time or Enable Alarm service call failed: {e}"}
        
        elif action == "cancel":
            try:
                await hass.services.async_call("script", "cancel_active_alarms", 
                    {"room": target_room}, 
                    blocking=True, context=llm_context.context)
                return {"result": f"Successfully cancelled alarm for {target_room}."}
            except Exception as e:
                _LOGGER.error(f"Cancel Service call failed: {e}")
                return {"error": f"Cancel Service call failed: {e}"}                
            
        return {"error": "Invalid action specified."}