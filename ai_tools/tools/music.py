import logging
import re
import asyncio
import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
import homeassistant.helpers.device_registry as dr
import homeassistant.helpers.area_registry as ar
import homeassistant.helpers.entity_registry as er
from homeassistant.core import ServiceCall, ServiceResponse, SupportsResponse

_LOGGER = logging.getLogger(__name__)

# =====================================================================
# UNIFIED MEDIA CORE STATE MACHINE
# =====================================================================
async def async_execute_media_action(
    hass: HomeAssistant, 
    action: str, 
    query: str = None, 
    target_room: str = None, 
    device_name: str = None, 
    device_id: str = None,
    context = None
) -> dict:
    """Single source of truth for routing and executing media transport controls."""
    _LOGGER.info(f"[MusicCore] Action: {action} | Query: {query} | Room: {target_room} | Device: {device_name} | DeviceID: {device_id}")

    dev_reg = dr.async_get(hass)
    area_reg = ar.async_get(hass)
    ent_reg = er.async_get(hass)
    
    def clean_name(name):
        return re.sub(r'[^a-z0-9]', '', str(name).lower())
        
    target_area_id = None
    
    # --- STAGE 1: Determine Target Area ---
    if target_room:
        for area in area_reg.areas.values():
            if clean_name(area.name) == clean_name(target_room) or clean_name(area.normalized_name) == clean_name(target_room):
                target_area_id = area.id
                break
    elif device_id:
        source_device = dev_reg.async_get(device_id)
        if source_device and source_device.area_id:
            target_area_id = source_device.area_id
            
    possible_players = []
    
    # --- STAGE 2: Find ALL Music Assistant Players ---
    for entity in ent_reg.entities.values():
        if entity.domain == "media_player":
            state_obj = hass.states.get(entity.entity_id)
            if getattr(entity, "platform", None) != "music_assistant":
                continue
                
            if state_obj and state_obj.state not in ["unavailable", "unknown"]:
                ent_area = entity.area_id
                if not ent_area and entity.device_id:
                    dev = dev_reg.async_get(entity.device_id)
                    if dev: ent_area = dev.area_id
                
                in_target_area = (target_area_id is None) or (ent_area == target_area_id)
                if in_target_area:
                    possible_players.append(entity)

    # --- STAGE 3: Filter or Default to Target Speaker ---
    target_media_player = None
    if device_name:
        clean_target = clean_name(device_name)
        for player in possible_players:
            player_strings = clean_name(player.name or "") + clean_name(player.original_name or "") + clean_name(player.entity_id)
            if clean_target in player_strings:
                target_media_player = player.entity_id
                break
    else:
        if device_id:
            source_device = dev_reg.async_get(device_id)
            if source_device:
                source_clean = clean_name(source_device.name_by_user or source_device.name or "")
                for player in possible_players:
                    player_strings = clean_name(player.name or "") + clean_name(player.original_name or "") + clean_name(player.entity_id)
                    if source_clean and (source_clean in player_strings or player_strings in source_clean):
                        target_media_player = player.entity_id
                        break
        
        if not target_media_player:
            for player in possible_players:
                if "voice" in player.entity_id or "speaker" in player.entity_id or "soundbar" in player.entity_id:
                    target_media_player = player.entity_id
                    break

        if not target_media_player and possible_players:
            target_media_player = possible_players[0].entity_id

    if not target_media_player:
        failed_target = device_name if device_name else "the requested area"
        return {"error": f"I could not find any active Music Assistant speakers to control for '{failed_target}'."}
            
    resolved_entity = ent_reg.async_get(target_media_player)
    resolved_name = resolved_entity.name or resolved_entity.original_name or target_media_player

    # =====================================================================
    # STICKY STREAM INTERCEPT LAYER
    # =====================================================================
    if action in ["next", "previous", "pause"]:
        state_obj = hass.states.get(target_media_player)
        if not state_obj or state_obj.state not in ["playing", "buffering"]:
            for player in possible_players:
                p_state = hass.states.get(player.entity_id)
                if p_state and p_state.state in ["playing", "buffering"]:
                    _LOGGER.warning(f"🚏 [MusicCore] STICKY INTERCEPT: Redirecting payload to running stream: '{player.entity_id}'")
                    target_media_player = player.entity_id
                    resolved_entity = ent_reg.async_get(target_media_player)
                    resolved_name = resolved_entity.name or resolved_entity.original_name or target_media_player
                    break

    # =====================================================================
    # TRANSPORT EXECUTION LOGIC
    # =====================================================================
    if action == "play":
        try:
            await hass.services.async_call(
                "music_assistant", "play_media",
                {
                    "entity_id": target_media_player, 
                    "media_id": query, 
                    "enqueue": "replace"
                },
                blocking=True, context=context
            )
            return {"result": f"Successfully started playing '{query}' on {resolved_name}."}
        except Exception as e:
            return {"error": f"Playback initialization failed: {str(e)}"}

    elif action == "pause":
        try:
            await hass.services.async_call(
                "media_player", "media_pause", {"entity_id": target_media_player},
                blocking=True, context=context
            )
            return {"result": f"Successfully paused playback stream on {resolved_name}."}
        except Exception as e:
            return {"error": f"Failed to execute pause transport: {str(e)}"}

    elif action == "next":
        try:
            await hass.services.async_call(
                "media_player", "media_next_track", {"entity_id": target_media_player},
                blocking=True, context=context
            )
            return {"result": f"Successfully skipped to the next track on {resolved_name}."}
        except Exception as e:
            return {"error": f"Failed to skip track: {str(e)}"}

    elif action == "previous":
        try:
            state_obj = hass.states.get(target_media_player)
            media_position = state_obj.attributes.get("media_position", 0) if state_obj else 0
            
            await hass.services.async_call(
                "media_player", "media_previous_track", {"entity_id": target_media_player},
                blocking=True, context=context
            )
            
            if media_position > 3.0:
                _LOGGER.info(f"⏱️ [MusicCore] Track was at {media_position}s. Executing smart second tap.")
                await asyncio.sleep(0.5)
                await hass.services.async_call(
                    "media_player", "media_previous_track", {"entity_id": target_media_player},
                    blocking=True, context=context
                )
            return {"result": f"Successfully reverted to the previous track on {resolved_name}."}
        except Exception as e:
            return {"error": f"Failed to return to previous track: {str(e)}"}
    
# =====================================================================
# LLM TOOL WRAPPING DECORATOR
# =====================================================================
class MusicPlayerTool(llm.Tool):
    """Tool to play music on specified devices/room"""
    name = "music_player"
    description = "Control music playback via Music Assistant. Can target specific rooms or devices."
    
    parameters = vol.Schema({
        vol.Required(
            "action", 
            description="The media control action to perform. Choose from: 'play', 'pause', 'next', or 'previous'."
        ): vol.In(["play", "pause", "next", "previous"]),
        vol.Required(
            "query",
            description=(
                "If the user mentions an artist, format strictly as 'Artist - Track' or 'Artist - Album'. "
                "If the user ONLY states a song or album title, pass JUST that title text. "
                "CRITICAL: Never guess, invent, or assume an artist name if the user did not explicitly say it. "
                "Leave empty if action is 'pause', 'next', or 'previous'."
            )
        ): str,
        vol.Optional(
            "room",
            description="Target room name ONLY if explicitly named by the user (e.g., 'playroom'). Otherwise, leave blank."
        ): str,
        vol.Optional(
            "device_name",
            description="Target device name ONLY if explicitly named by the user (e.g., 'Bedroom TV'). Otherwise, leave blank."
        ): str
    })

    async def async_call(self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext) -> dict:
        return await async_execute_media_action(
            hass=hass,
            action=tool_input.tool_args.get("action", ""),
            query=tool_input.tool_args.get("query"),
            target_room=tool_input.tool_args.get("room"),
            device_name=tool_input.tool_args.get("device_name"),
            device_id=llm_context.device_id,
            context=llm_context.context
        )


# =====================================================================
# HOME ASSISTANT NATIVE INTENT OVERRIDES
# =====================================================================

def register_media_service(hass: HomeAssistant):
    """Registers the unified media core as a native Home Assistant service."""
    async def handle_media_command(call: ServiceCall) -> ServiceResponse:

        try:

            res = await async_execute_media_action(
                    hass, 
                    action=call.data.get("action"), 
                    query=call.data.get("query"), 
                    target_room=call.data.get("room"), 
                    device_name=call.data.get("device_name"), 
                    context=call.context
                )
            
            return {"result": res.get("result", res.get("error", "Action completed"))}
            
        except Exception as e:
                # If the code crashes, we return the error string as a 'result' 
                # so the template engine doesn't crash on an 'undefined' variable.
                _LOGGER.error(f"[MusicCore] SERVICE CRASH: {str(e)}")
                return {"result": f"Internal Error: {str(e)}"}        
            # Services that return responses MUST return a dictionary

    hass.services.async_register(
        "ai_tools", 
        "execute_media",
        handle_media_command
    )