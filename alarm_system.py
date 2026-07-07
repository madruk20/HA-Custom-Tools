import asyncio
import logging
from datetime import datetime

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import device_registry as dr
from homeassistant.util import slugify
from homeassistant.components.tts import async_default_engine
from homeassistant.components import assist_pipeline
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

class AlarmSystem:
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        self.hass = hass
        self.entry = entry
        self.active_alarms = {} 
        self.original_volumes = {}
        self._unsub = None

    def _get_greeting(self) -> str:
        """Returns dynamic greeting based on current time."""
        hour = datetime.now().hour
        if 5 <= hour < 12: return "Good morning"
        if 12 <= hour < 18: return "Good afternoon"
        return "Good evening"

    async def async_start(self):
        """Start the background scheduler to check time every minute."""
        self._unsub = async_track_time_change(self.hass, self._check_alarms, second=0)
        _LOGGER.info("✅ Alarm System background scheduler active.")

    async def async_unload(self):
        """Clean up the listener if the integration is unloaded."""
        if self._unsub:
            self._unsub()

    async def _check_alarms(self, now: datetime):
        """Evaluates all alarms triggered exactly at the current minute."""
        current_time = now.strftime("%H:%M:00")
        alarms_config = self.entry.options.get("alarms_config", {})

        for room, data in alarms_config.items():
            if data.get("enabled") and data.get("time") == current_time:
                _LOGGER.info(f"⏰ ALARM SYSTEM: 🔔 ALARM TRIGGERED for '{room}' at {current_time}!")
                await self.async_trigger_alarm(room, data)

    async def async_restore_volume(self, player: str):
        """Restores the volume to the level it was at before the alarm."""
        if player in self.original_volumes:
            old_vol = self.original_volumes.pop(player)
            _LOGGER.info(f"⏰ ALARM SYSTEM: Restoring volume for {player} to {old_vol}")
            await self.hass.services.async_call("media_player", "volume_set", {
                "entity_id": player, 
                "volume_level": old_vol
            })

    async def async_trigger_alarm(self, room: str, data: dict):
        player = data.get("media_player")
        song = data.get("song")
        fallback = data.get("fallback_notify")
        volume = data.get("volume", 0.7)
        success = False

        player_state = self.hass.states.get(player) if player else None
        
        if player_state:
            try:
                # --- THE GENERIC BOOT SEQUENCE ---
                if player_state.state in ["off", "idle", "standby", "unavailable", "unknown"]:
                    wake_entity = data.get("wake_entity")
                    woke_something = False
                    
                    if wake_entity:
                        # 1. User-Defined Generic Wake Action
                        _LOGGER.info(f"⏰ ALARM SYSTEM: Triggering user-defined wake entity '{wake_entity}'...")
                        try:
                            # Using the universal 'homeassistant.turn_on' means the user can select 
                            # a Script, a Scene, a Switch, or a Media Player
                            await self.hass.services.async_call("homeassistant", "turn_on", {"entity_id": wake_entity})
                            woke_something = True
                        except Exception as e:
                            _LOGGER.warning(f"⚠️ ALARM SYSTEM: Failed to trigger wake entity: {e}")
                            
                    else:
                        # 2. Automatic Fallback: Look for physical hardware siblings
                        ent_reg = er.async_get(self.hass)
                        entity = ent_reg.async_get(player)
                        
                        if entity and entity.platform == "mass" and entity.device_id:
                            for sibling in er.async_entries_for_device(ent_reg, entity.device_id):
                                if sibling.domain == "media_player" and sibling.platform != "mass":
                                    _LOGGER.info(f"🔍 ALARM SYSTEM: Attempting to wake hardware sibling -> {sibling.entity_id}")
                                    try:
                                        await self.hass.services.async_call("media_player", "turn_on", {"entity_id": sibling.entity_id})
                                        woke_something = True
                                    except Exception:
                                        # Catch silently so unsupported devices don't crash the script
                                        pass
                                        
                    if woke_something:
                        # Give the hardware 10 seconds to boot and connect to Wi-Fi
                        _LOGGER.info("⏰ ALARM SYSTEM: Waiting 10 seconds for hardware to boot...")
                        await asyncio.sleep(10)
                        
                    # Refresh the state object so our snapshot uses the fresh, awake data
                    player_state = self.hass.states.get(player)

                # --- 1. SNAPSHOT BEFORE DOING ANYTHING ---
                # We use the refreshed player_state for our snapshot
                pre_state_obj = player_state
                pre_state = pre_state_obj.state if pre_state_obj else None
                pre_title = pre_state_obj.attributes.get("media_title") if pre_state_obj else None
                
                # Save the original volume so we can restore it later
                if pre_state_obj and "volume_level" in pre_state_obj.attributes:
                    self.original_volumes[player] = pre_state_obj.attributes.get("volume_level")
                elif player not in self.original_volumes:
                    # Fallback to a safe normal volume if the speaker doesn't report it while idle
                    self.original_volumes[player] = 0.4

                # 2. Set Alarm Volume
                await self.hass.services.async_call("media_player", "volume_set", {"entity_id": player, "volume_level": volume})
                
                # 3. Attempt Music Assistant Playback
                _LOGGER.info(f"⏰ ALARM SYSTEM: Triggering Music Assistant on '{player}' with item: {song}")
                await self.hass.services.async_call("music_assistant", "play_media", {
                    "entity_id": player,
                    "media_id": song,
                    "enqueue": "replace"
                })
                
                # VERIFICATION: Wait 2 seconds for the player to react
                await asyncio.sleep(2)
                
                # --- SNAPSHOT AFTER PLAYBACK ---
                post_state_obj = self.hass.states.get(player)
                post_state = post_state_obj.state if post_state_obj else None
                post_title = post_state_obj.attributes.get("media_title") if post_state_obj else None
                
                # Check if it is currently playing
                if post_state in ["playing", "buffering"]:
                    
                    # Verify it is playing OUR song (State changed OR Title changed)
                    if pre_state != post_state or pre_title != post_title:
                        success = True
                        self.active_alarms[room] = player
                        _LOGGER.info(f"✅ ALARM SYSTEM: Music playback verified as active on '{player}'.")
                    else:
                        # It was already playing, and the title didn't change (e.g. TV watching Netflix)
                        _LOGGER.error(f"❌ '{player}' is active, but the media did not change. MA likely failed.")
                        raise Exception("Media failed to override current playback")
                else:
                    _LOGGER.error(f"❌ Music Assistant failed to start '{song}'. Verifying playback failed.")
                    raise Exception("Media failed to play")

            except Exception as e:
                # MUSIC ASSISTANT FAILED - Trigger TTS Backup
                _LOGGER.error(f"❌ Music Assistant failed to resolve '{song}': {e}")
                now = dt_util.now()
                # Grab the local Home Assistant time and format it to 12-hour clock (e.g., "07:30 AM")
                raw_time = now.strftime("%I:%M %p")
                current_time = raw_time.lstrip("0")
                # Format date to "Monday, July 06" then replace the leading zero
                current_date = now.strftime("%A, %B %d").replace(" 0", " ")
                greet = self._get_greeting()
                error_msg = f"{greet}! It is now {current_time} on {current_date}. Your currently set music file for your alarm is unavailable."
                
                # Background Discovery: Get default Assist Pipeline
                pipeline = assist_pipeline.async_get_pipeline(self.hass)
                primary_tts_engine = pipeline.tts_engine if pipeline else async_default_engine(self.hass)
                primary_tts_voice = pipeline.tts_voice if pipeline else None
                
                # Apply the specific voice ID only if the pipeline has one configured
                tts_options = {"voice": primary_tts_voice} if primary_tts_voice else {}
                
                try:
                    # Attempt 1: Sync with Voice Assistant Pipeline
                    await self.hass.services.async_call(
                        "tts", 
                        "speak", 
                        {
                            "message": error_msg,
                            "media_player_entity_id": player,
                            "options": tts_options
                        },
                        target={"entity_id": primary_tts_engine}
                    )
                    _LOGGER.info(f"✅ ALARM SYSTEM: TTS fallback triggered using Pipeline Engine ({primary_tts_engine}).")
                
                except Exception as primary_tts_err:
                    _LOGGER.warning(f"⚠️ Pipeline TTS failed ({primary_tts_err}). Attempting Google fallback...")
                    
                    try:
                        # Attempt 2: Emergency Google Cloud Fallback
                        # Replace this string with Google TTS entity ID if it differs
                        fallback_engine = "tts.google_translate_en_com" 
                        
                        await self.hass.services.async_call(
                            "tts", 
                            "speak", 
                            {
                                "message": error_msg,
                                "media_player_entity_id": player
                            },
                            target={"entity_id": fallback_engine}
                        )
                        _LOGGER.info(f"✅ ALARM SYSTEM: Emergency Google TTS fallback successful.")
                    
                    except Exception as backup_tts_err:
                        _LOGGER.error(f"❌ ALARM SYSTEM: All TTS fallbacks failed: {backup_tts_err}")
                
                finally:
                    # Volume is ALWAYS restored, regardless of which attempt succeeded or failed
                    await self.async_restore_volume(player)
                    _LOGGER.info(f"✅ ALARM SYSTEM: Volume restored on '{player}'.")
        else:
            _LOGGER.warning(f"⚠️ ALARM SYSTEM: Player '{player}' is offline.")

        # --- FALLBACK NOTIFICATION SYSTEM ---
        if not success:
            if fallback:
                _LOGGER.info(f"🚨 ALARM SYSTEM: Triggering backup mobile notification via '{fallback}'.")
                
                notify_service = None
                ent_reg = er.async_get(self.hass)
                entity = ent_reg.async_get(fallback)
                
                if entity and entity.device_id:
                    dev_reg = dr.async_get(self.hass)
                    device = dev_reg.async_get(entity.device_id)
                    if device:
                        for entry_id in device.config_entries:
                            entry = self.hass.config_entries.async_get_entry(entry_id)
                            if entry and entry.domain == "mobile_app":
                                device_name = entry.data.get("device_name")
                                if device_name:
                                    notify_service = f"mobile_app_{slugify(device_name)}"
                                    break
                
                if not notify_service:
                    clean_name = fallback.replace("notify.", "")
                    notify_service = f"mobile_app_{clean_name}"
                    if clean_name.startswith("mobile_app_"):
                        notify_service = clean_name
                        
                try:
                    await self.hass.services.async_call("notify", notify_service, {
                        "title": "🚨 WAKE UP ALARM",
                        "message": f"Wake up! The speaker in the {room} failed or was offline.",
                        "data": {
                            "channel": "Voice_Alarm",
                            "ttl": 0,
                            "priority": "high",
                            "push": {
                                "sound": {
                                    "name": "default",
                                    "critical": 1,
                                    "volume": 1.0
                                }
                            }
                        }
                    })
                    _LOGGER.info(f"✅ ALARM SYSTEM: Fallback notification successfully sent via {notify_service}.")
                except Exception as e:
                    _LOGGER.error(f"❌ ALARM SYSTEM: Failed to route mobile app fallback notification: {e}")
            
            else:
                _LOGGER.error(f"❌ ALARM SYSTEM: Alarm failed for {room}, but no fallback mobile app was configured!")