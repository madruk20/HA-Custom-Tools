import logging
import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er

_LOGGER = logging.getLogger(__name__)

class CameraStreamTool(llm.Tool):
    """Tool to stream security cameras or cast dashboards to media players."""
    name = "stream_camera_to_tv"
    description = "Streams a security camera feed or casts a multi-camera dashboard to a TV. Use this to 'play' or 'stop' camera feeds."
    
    parameters = vol.Schema({
        vol.Required(
            "action", 
            description="Must be 'play' to view a camera, or 'stop' to cancel/close an active feed."
        ): str,
        vol.Optional(
            "camera_name", 
            description="The location or name of the camera (e.g., 'backyard'). Use 'all' if the user wants to see all cameras at once. Required if action is 'play'."
        ): str,
        vol.Optional(
            "target_tv", 
            description="The specific TV to display or stop the feed on. Leave empty to use the TV in the user's current room."
        ): str
    })

    async def async_call(self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext) -> dict:
        args = tool_input.tool_args
        action = args.get("action", "play").lower()
        camera_query = args.get("camera_name", "").lower()
        target_tv_query = args.get("target_tv", "").lower()

        ent_reg = er.async_get(hass)
        dev_reg = dr.async_get(hass)

        def _get_entity_area(entity):
            if entity.area_id: return entity.area_id
            if entity.device_id:
                dev = dev_reg.async_get(entity.device_id)
                if dev and dev.area_id: return dev.area_id
            return None

        # 1. Resolve Target Media Players (Strictly Google Cast)
        target_media_players = []
        
        if target_tv_query:
            for state in hass.states.async_all("media_player"):
                friendly_name = state.attributes.get("friendly_name", "").lower()
                entity_id = state.entity_id.lower()
                if target_tv_query in friendly_name or target_tv_query in entity_id:
                    if "cast" in entity_id or "chromecast" in entity_id:
                        target_media_players.append(state.entity_id)
        else:
            if llm_context.device_id:
                source_device = dev_reg.async_get(llm_context.device_id)
                if source_device and source_device.area_id:
                    for entity in ent_reg.entities.values():
                        if entity.domain == "media_player" and _get_entity_area(entity) == source_device.area_id:
                            if "cast" in entity.entity_id.lower() or "chromecast" in entity.entity_id.lower():
                                target_media_players.append(entity.entity_id)

        if not target_media_players:
            return {"error": "Could not identify a Google Cast-enabled TV in the requested room."}

        # --- ACTION: STOP ---
        if action == "stop":
            success = False
            for player_id in target_media_players:
                try:
                    await hass.services.async_call("media_player", "media_stop", {"entity_id": player_id}, blocking=True)
                    # Also try turning off the cast entity which helps clear dashboards
                    await hass.services.async_call("media_player", "turn_off", {"entity_id": player_id}, blocking=True)
                    success = True
                except Exception as e:
                    _LOGGER.warning(f"Failed to stop {player_id}: {e}")
            
            if success:
                tv_friendly = hass.states.get(target_media_players[0]).attributes.get("friendly_name", "TV")
                return {"result": f"Successfully closed the camera feed on the {tv_friendly}."}
            return {"error": "Attempted to stop the camera feed, but the media player rejected the command."}

        # --- ACTION: PLAY GRID / ALL CAMERAS ---
        if camera_query in ["all", "all cameras", "grid", "split screen", "everything", "dashboard"]:
            success = False
            successful_player = None
            for player_id in target_media_players:
                try:
                    _LOGGER.info(f"Casting camera dashboard to {player_id}...")
                    await hass.services.async_call(
                        "cast",
                        "show_lovelace_view",
                        {
                            "entity_id": player_id,
                            "dashboard_path": "lovelace",
                            "view_path": "cameras" # This must match the URL path you set in Step 1 ######################################################
                        },
                        blocking=True
                    )
                    success = True
                    successful_player = player_id
                    break 
                except Exception as e:
                    _LOGGER.warning(f"Dashboard cast failed on {player_id}: {e}")

            if success:
                tv_friendly = hass.states.get(successful_player).attributes.get("friendly_name", "TV")
                return {"result": f"Successfully cast the multi-camera grid dashboard to the {tv_friendly}."}
            return {"error": "Failed to cast the camera dashboard to the TV."}

        # --- ACTION: PLAY SINGLE CAMERA (HLS Stream) ---
        target_camera_id = None
        for state in hass.states.async_all("camera"):
            friendly_name = state.attributes.get("friendly_name", "").lower()
            entity_id = state.entity_id.lower()
            
            if camera_query in friendly_name or camera_query in entity_id:
                target_camera_id = state.entity_id
                if "fluent" not in entity_id and "fluent" not in friendly_name and "sub" not in entity_id and "sub" not in friendly_name:
                    break

        if not target_camera_id:
            return {"error": f"Could not find a camera matching '{camera_query}'."}

        media_uri = f"media-source://camera/{target_camera_id}"
        success = False
        successful_player = None

        for player_id in target_media_players:
            try:
                _LOGGER.info(f"Attempting to cast {target_camera_id} to {player_id}...")
                await hass.services.async_call(
                    "media_player",
                    "play_media",
                    {
                        "entity_id": player_id,
                        "media_content_id": media_uri,
                        "media_content_type": "application/vnd.apple.mpegurl",
                    },
                    blocking=True
                )
                success = True
                successful_player = player_id
                break 
            except Exception as e:
                _LOGGER.warning(f"Cast failed on {player_id}: {e}")

        if success:
            cam_friendly = hass.states.get(target_camera_id).attributes.get("friendly_name", "camera")
            tv_friendly = hass.states.get(successful_player).attributes.get("friendly_name", "TV")
            return {"result": f"Successfully cast the {cam_friendly} to the {tv_friendly}."}
        return {"error": "Failed to cast the single camera feed to the TV."}