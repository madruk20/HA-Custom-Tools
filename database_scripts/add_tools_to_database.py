import hashlib
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams, Distance, SparseVectorParams, 
    Modifier, PointStruct, SparseVector, Document
)

# --- Configuration ---
QDRANT_URL = "http://192.168.4.23:6333"
OLLAMA_URL = "http://192.168.4.23:11434/api/embeddings"
OLLAMA_MODEL = "qwen-embed-2k:latest"
COLLECTION_NAME = "tools_collection"
DIMENSIONS = 1024

qdrant = QdrantClient(url=QDRANT_URL)

def get_dense_embedding(text):
    payload = {"model": OLLAMA_MODEL, "prompt": text}
    response = requests.post(OLLAMA_URL, json=payload).json()
    return response.get("embedding")

def get_consistent_id(name):
    return hashlib.md5(name.encode()).hexdigest()

# --- Initialize Collection (Updated Logic) ---
if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' does not exist. Creating for Hybrid Search...")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "qwen_dense": VectorParams(size=DIMENSIONS, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "keyword_sparse": SparseVectorParams(modifier=Modifier.IDF),
        }
    )
    print("Hybrid Collection created successfully.")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists. Proceeding to upsert data...")


# --- MASTER TOOL LIST ---
tools_to_add = [
    {
        "name": "HassTurnOn",
        "desc": (
            "HassTurnOn - Use this tool to activate, power up, boot, turn on, enable, or start smart home devices such as lights, televisions, electronics, appliances, and entities. "
            "Applies universally to lights, switches, smart plugs, screens, television sets, monitors, climate zones, and media playback targets. "
            "CRITICAL SECURITY RULE: Always call this specific tool to LOCK smart locks, secure motorized deadbolts, or lock security doors. "
            "Keywords: turn on, start, power on, activate, switch on, lock, secure, enable, boot, engage, open up, fire up, toggle on. "
            "Examples: 'Turn on the bedroom TV', 'Lock the front door', 'Switch on the fan', 'Turn on the lights', 'Activate movie scene', 'Secure the house deadbolts'."
            "EXCLUSIONS: Do NOT use this tool to alter illumination hues, adjust percentage levels, or initiate audio/video playback."
        )
    },
    {
        "name": "HassTurnOff",
        "desc": (
            "HassTurnOff - Use this tool to deactivate, shut down, power down, turn off, terminate, or disable smart home hardware devices such as lights, televisions, electronics, appliances, and entities. "
            "Applies universally to lights, switches, smart plugs, screens, television sets, monitors, climate zones, and media playback targets. "
            "CRITICAL SECURITY RULE: Always call this specific tool to UNLOCK smart locks, release entryways, open deadbolts, or open doors. "
            "Keywords: turn off, TV, television, stop, power off, deactivate, switch off, shut down, unlock, open lock, release door, kill power, disable, toggle off. "
            "Examples: 'Turn off the lights', 'Unlock the door', 'Shut off the arcade TV', 'Turn off the TV', 'Deactivate the fan', 'Power down the theater system'."
            "EXCLUSIONS: Do NOT use this tool to suspend audio/video playback, halt movies, or quiet speakers."
        )
    },
    {
        "name": "HassStartTimer",
        "desc": (
            "HassStartTimer - Use this tool to initialize, create, or launch a new numeric countdown helper timer for tracking a specified duration of time. "
            "Applies when running stopwatch helpers, cooking track clocks, chore counts, or reminder durations. "
            "Keywords: set a timer, start timer, count down, clock, tracking timer, minutes timer, duration. "
            "Examples: 'Set a timer for 10 minutes', 'Start a 5 minute timer for the laundry', 'Begin a kitchen countdown for twenty minutes'."
            "EXCLUSIONS: Do NOT use this tool for scheduling morning wake-up calls, recurring daily alerts, or setting a clock for a specific time of day."
        )
    },
    {
        "name": "HassCancelTimer",
        "desc": (
            "HassCancelTimer - Use this tool to immediately stop, reset, abort, delete, clear, or purge an active or running countdown timer helper instance. "
            "Keywords: cancel timer, stop timer, delete timer, abort countdown, reset timer, clear clock. "
            "Examples: 'Cancel the kitchen timer', 'Stop the 10 minute timer', 'Clear the laundry countdown clock', 'Reset the active timer'."
            "EXCLUSIONS: Do NOT use this tool for cancelling wake-up calls."
        )
    },
    {
        "name": "HassTimerStatus",
        "desc": (
            "HassTimerStatus - Use this tool to inspect, check, audit, read back, or query the remaining duration, elapsed runtime, or time left on active countdown helper timers. "
            "Keywords: timer status, time left, time remaining, how much time, check timer, countdown progress. "
            "Examples: 'How much time is left on the timer?', 'Check the laundry timer status', 'Is the kitchen countdown finished yet?'"
        )
    },
    {
        "name": "HassLightSet",
        "desc": (
            "HassLightSet - Use this tool to manipulate specific illuminating attributes, parameters, or states of dimmable or color-changing smart lights and illumination bulbs. "
            "Controls lighting brightness percentages, dim levels, transitions, hex values, RGB values, white color spectrum balances, or hue settings. "
            "Keywords: brightness, dim, color, temperature, percent, illuminate, change color, brighten, scale level, red, blue, green, warm white, cool white. "
            "Examples: 'Set the lights to 50%', 'Change the bedroom light to blue', 'Dim the lights', 'Brighten the playroom lamps', 'Set lighting color temperature to warm white'."
            "EXCLUSIONS: Do NOT use this tool for binary power commands. ONLY use this if the user explicitly dictates a percentage, dimming instruction, or a specific visual hue."
        )
    },
    {
        "name": "HassBroadcast",
        "desc": (
            "HassBroadcast - Use this tool to dispatch voice announcements, global audio alerts, text-to-speech (TTS) notifications, or whole-house audio broadcasts to networked smart speakers, media players, and intercom units. "
            "Keywords: broadcast, announce, tell everyone, say message, voice alert, text to speech, speak announcement, notify house, intercom. "
            "Examples: 'Broadcast that dinner is ready', 'Announce to the playroom that it is time for bed', 'Tell everyone we are leaving in ten minutes', 'Speak a message to the bedroom speaker'."
        )
    },
    {
        "name": "HassListAddItem",
        "desc": (
            "HassListAddItem - Use this tool to inject, append, write, add, or record a brand-new item, grocery asset, checkbox element, or task target onto an active shopping list, grocery list, task board, or to-do list entity. "
            "Keywords: add to list, remember to buy, put on grocery list, new task, append item, save to to-do list, list update. "
            "Examples: 'Add milk to the shopping list', 'Put take out the trash on my to do list', 'Add eggs to the grocery list', 'Put water filters on my shopping list'."
        )
    },
    {
        "name": "HassListCompleteItem",
        "desc": (
            "HassListCompleteItem - Use this tool to check off, strike through, mark as done, finish, or complete an existing item or task string inside an active shopping list, grocery organizer, or to-do list. "
            "Keywords: complete task, check off list, mark as done, cross off, finish task, resolved item. "
            "Examples: 'Cross milk off the grocery list', 'Mark take out the trash as complete', 'Check off bread from the shopping tracker', 'Finish the laundry task on my list'."
        )
    },
    {
        "name": "HassListRemoveItem",
        "desc": (
            "HassListRemoveItem - Use this tool to permanently erase, delete, scrub, clear, drop, or purge a specific text entry, task, or product item from a shopping list, grocery tracker, or to-do array. "
            "Keywords: delete from list, remove item, clear task, erase item, drop from list, purge element. "
            "Examples: 'Remove milk from the shopping list', 'Delete the trash task from my to do list', 'Clear apples from my grocery list'."
        )
    },
    {
        "name": "todo_get_items",
        "desc": (
            "todo_get_items - Use this tool to fetch, read back, inspect, list, verify, or review what textual data, tasks, checklist elements, or products are currently stored on a shopping list, grocery list, or to-do tracker. "
            "Keywords: read list, what is on my list, check shopping list, to do items, look up list, see list items, read items back. "
            "Examples: 'What is on the grocery list?', 'Read my to do list to me', 'Do I have eggs on the shopping list?', 'Check what items are on the shopping tracker'."
        )
    },
    {
        "name": "HassGetState",
        "desc": (
            "HassGetState - Use this tool to check, inspect, verify, or read the current status, state, temperature, or condition of any smart home device, sensor, door, window, lock, or appliance. "
            "Keywords: is the door open, what is the temperature, check the garage, status of the lights, is the TV on, read sensor. "
            "Examples: 'Is the front door locked?', 'What is the temperature in the living room?', 'Is the bedroom light on?', 'Did I leave the garage door open?'"
            "EXCLUSIONS: Do NOT use this tool to alter, change, or manipulate any device. This is strictly a read-only hardware query."
        )
    },
    {
        "name": "HassMediaSearchAndPlay",
        "desc": (
            "HassMediaSearchAndPlay - Use this tool to search for and initiate playback of native media content, movies, or audio on Home Assistant media players. "
            "Keywords: play media, watch movie, start video, cast video, media player playback. "
            "Examples: 'Play The Matrix on the living room TV', 'Watch a movie in the theater', 'Play content on the Chromecast'."
        )
    },
    {
        "name": "HassMediaPause",
        "desc": (
            "HassMediaPause - Use this tool to temporarily pause or halt media playback on native Home Assistant media players. "
            "Keywords: pause TV, stop playing, hold media, pause movie. "
            "Examples: 'Pause the living room TV', 'Pause the movie', 'Pause the Chromecast'."
        )
    },
    {
        "name": "HassMediaUnpause",
        "desc": (
            "HassMediaUnpause - Use this tool to resume, unpause, or continue media playback on native Home Assistant media players. "
            "Keywords: resume, unpause, continue playing, play video, resume movie. "
            "Examples: 'Resume the living room TV', 'Unpause the movie', 'Continue playing on the Chromecast'."
        )
    },
    {
        "name": "HassMediaNext",
        "desc": (
            "HassMediaNext - Use this tool to skip to the next media track, chapter, or video on a native media player. "
            "Keywords: next track, skip forward, next video, advance media. "
            "Examples: 'Next track on the living room TV', 'Skip chapter', 'Play the next video'."
        )
    },
    {
        "name": "HassMediaPrevious",
        "desc": (
            "HassMediaPrevious - Use this tool to return to the previous media track, chapter, or video on a native media player. "
            "Keywords: previous track, go back, last video, replay chapter. "
            "Examples: 'Previous track on the living room TV', 'Go back a chapter', 'Play the previous video'."
        )
    },
    {
        "name": "HassSetVolume",
        "desc": (
            "HassSetVolume - Use this tool to set the specific audio volume level of a media player, speaker, or television to an exact percentage. "
            "Keywords: set volume, change volume, volume 50 percent, exact volume level. "
            "Examples: 'Set the living room TV volume to 50%', 'Change the soundbar volume to 30', 'Set speaker to 100%'."
        )
    },
    {
        "name": "HassSetVolumeRelative",
        "desc": (
            "HassSetVolumeRelative - Use this tool to increase or decrease the audio volume of a media player relative to its current level. "
            "Keywords: turn up the volume, turn it down, louder, quieter, increase volume, lower volume. "
            "Examples: 'Turn up the volume on the TV', 'Make the music quieter', 'Turn it down', 'Increase the soundbar volume'."
        )
    },
    {
        "name": "HassMediaPlayerMute",
        "desc": (
            "HassMediaPlayerMute - Use this tool to mute or silence the audio output of a media player, television, or speaker. "
            "Keywords: mute, silence, quiet, turn off sound, cut the volume. "
            "Examples: 'Mute the living room TV', 'Silence the music', 'Mute the soundbar'."
        )
    },
    {
        "name": "HassMediaPlayerUnmute",
        "desc": (
            "HassMediaPlayerUnmute - Use this tool to unmute or restore the audio output of a media player, television, or speaker. "
            "Keywords: unmute, restore sound, turn sound back on. "
            "Examples: 'Unmute the living room TV', 'Restore audio on the speaker', 'Unmute the soundbar'."
        )
    },
    {
        "name": "HassCancelAllTimers",
        "desc": (
            "HassCancelAllTimers - Use this tool to simultaneously cancel, stop, or clear all currently active countdown timers across the entire smart home. "
            "Keywords: cancel all timers, stop all timers, clear all countdowns, wipe timers. "
            "Examples: 'Cancel all timers', 'Stop every timer', 'Clear all countdowns in the house'."
        )
    },
    {
        "name": "HassIncreaseTimer",
        "desc": (
            "HassIncreaseTimer - Use this tool to add more time, extend, or increase the duration of an actively running countdown timer. "
            "Keywords: add time, extend timer, increase countdown, more time. "
            "Examples: 'Add 5 minutes to the kitchen timer', 'Extend the laundry timer', 'Put 2 more minutes on the clock'."
        )
    },
    {
        "name": "HassDecreaseTimer",
        "desc": (
            "HassDecreaseTimer - Use this tool to subtract time, reduce, or decrease the duration of an actively running countdown timer. "
            "Keywords: remove time, decrease timer, subtract minutes, less time. "
            "Examples: 'Remove 2 minutes from the kitchen timer', 'Decrease the laundry timer', 'Take a minute off the clock'."
        )
    },
    {
        "name": "HassPauseTimer",
        "desc": (
            "HassPauseTimer - Use this tool to temporarily pause or freeze an actively running countdown timer without canceling it. "
            "Keywords: pause timer, freeze countdown, hold timer, stop the clock temporarily. "
            "Examples: 'Pause the kitchen timer', 'Hold the laundry countdown', 'Pause the clock'."
        )
    },
    {
        "name": "HassUnpauseTimer",
        "desc": (
            "HassUnpauseTimer - Use this tool to resume or continue a previously paused countdown timer. "
            "Keywords: resume timer, unpause countdown, continue timer, start the clock again. "
            "Examples: 'Resume the kitchen timer', 'Unpause the laundry countdown', 'Continue the clock'."
        )
    },
    {
        "name": "GetDateTime",
        "desc": (
            "GetDateTime - Use this tool to retrieve the current physical time, date, day of the week, or year. "
            "Keywords: what time is it, current date, what day is today, check the time, get current day. "
            "Examples: 'What time is it right now?', 'What is today's date?', 'What day of the week is it?', 'What year is this?'"
        )
    },
    {
        "name": "stock_and_retail_price_lookup",
        "desc": (
            "stock_and_retail_price_lookup - Use this tool to retrieve real-time financial market tickers, equity data, public corporate tracking, and retail product prices. "
            "Applies when searching for financial performance indices, stock charts, investment valuation, corporate ticker symbols, market capitalization, or consumer goods costs. "
            "Keywords: price, cost, stock, market, equity, shares, value, ticker, chart, retail, trade, ticker symbol, investments, worth, how much does it cost, what is the price. "
            "Examples: 'What is the current stock price of Apple?', 'Show me the latest market charts for NVIDIA', 'How much does this TV cost?', 'What is the retail price of an Xbox?'"
        )
    },
    {
        "name": "alarm_manager",
        "desc": (
            "alarm_manager - Use this tool to create, configure, modify, check status, disable, or cancel wake-up alarms and schedule daily reminders across any area or device. "
            "Directly controls input_datetime configurations, state flags, and tracking helpers related to alarm entities. "
            "Keywords: alarm, wake up, set alarm, cancel alarm, wake me up, turn off alarm, stop alarm, disable alarm, change alarm time, alarm status, alarm time. "
            "Examples: 'Set my alarm for 4PM', 'Cancel my alarm', 'Turn off the bedroom alarm', 'Wake me up at 7 AM tomorrow', 'Disable the playroom alarm', 'What time is my alarm set to?', 'Is my alarm enabled?'"
            "EXCLUSIONS: Do NOT use this tool for short culinary countdowns, laundry tracking, or stopwatch intervals."
        )
    },
    {
        "name": "stream_camera_to_tv",
        "desc": (
            "stream_camera_to_tv - Use this tool to stream, display, cast, view, or STOP security camera video feeds directly onto smart televisions, media players, or displays. "
            "Applies when the user wants to visually monitor, watch, pull up, bring up, close, or cancel security cameras, surveillance streams, or CCTV monitors. "
            "Can handle single cameras OR cast a grid/split-screen of ALL cameras at once. "
            "Keywords: camera, stream, view, cast, security camera, show me the camera, pull up the camera, stop camera, close feed, cancel camera, display camera on TV, monitor, all cameras, grid, split screen. "
            "Examples: 'Bring up the backyard camera on my bedroom TV', 'Show me all the cameras', 'Put the camera grid on the television', 'Stop the backyard camera feed'."
            "EXCLUSIONS: Do NOT use this tool for regular television shows, cinematic films, or musical playback."
        )
    },
    {
        "name": "LoadGame",
        "desc": (
            "LoadGame - Use this tool to boot, start, execute, launch, or change video games, simulation roms, or software configurations on cabinet arcade installations, gaming systems, or display configurations. "
            "Keywords: load, play game, start game, launch emulator, boot game, retro arcade, load cabinet game. "
            "Examples: 'Load game Street Fighter II', 'Start game Donkey Kong', 'Boot up Pac-Man on the arcade machine', 'Launch Marvel vs Capcom'."
        )
    },
    {
        "name": "music_player",
        "desc": (
            "music_player - PLAY_MEDIA_ACTION: Music control tool. Use this tool to control media playback, music streaming, audio tracks, and playlists via Music Assistant. "
            "Handles commands to play specific tracks, search for music, stream audio, or choose songs, artists, albums, genres, and media queues. "
            "Can target specific entertainment hardware, speakers, soundbars, smart displays, media players, or media zones globally or in designated areas. "
            "Keywords: play, stream, listen to, music, song, artist, album, track, audio, sound, queue, playlist, media, band, tune, soundtrack. "
            "Examples: 'Play the song titled Enter Sandman from Metallica', 'Play the album Core by Stone Temple Pilots', 'Play the album Bad by Michael Jackson', "
            "'Stream some music on the Arcade Soundbar', 'Play my favorite playlist', 'Put on some tunes in the playroom'."
            "EXCLUSIONS: Do NOT use this tool for cinematic films, television shows, or casting video content to screens."
        )
    },
    {
        "name": "MusicPause",
        "desc": (
            "MusicPause - Use this tool to immediately pause, freeze, halt, suspend, or temporarily interrupt currently playing audio streams, music tracks, video files, television sets, or active media streams. "
            "Keywords: pause, pause music, stop playback, halt media, freeze track, suspend stream, pause audio. "
            "Examples: 'Pause the music', 'Pause the bedroom TV', 'Halt media playback on the soundbar', 'Pause the arcade television'."
            "EXCLUSIONS: Do NOT use this tool for cinematic films, television shows, or casting video content to screens."
        )
    },
    {
        "name": "MusicNext",
        "desc": (
            "MusicNext - Use this tool to advance, skip forward, jump to, or play the subsequent track, song, clip, chapter, video, or media entry inside an active queue or media playlist. "
            "Keywords: next, skip, skip track, forward, skip song, next track, move forward, subsequent track. "
            "Examples: 'Play the next song', 'Skip this track', 'Next track', 'Go to the next item in the queue'."
            "EXCLUSIONS: Do NOT use this tool for cinematic films, television shows, or casting video content to screens."
        )
    },
    {
        "name": "MusicPrevious",
        "desc": (
            "MusicPrevious - Use this tool to backtrack, step backward, return to, or replay the preceding track, song, chapter, clip, or media asset in an active playback queue. "
            "Keywords: previous, back, go back, last track, previous song, restart track, return backward. "
            "Examples: 'Play the last song again', 'Go back to the previous track', 'Previous track', 'Restart this song from the beginning'."
            "EXCLUSIONS: Do NOT use this tool for cinematic films, television shows, or casting video content to screens."
        )
    }
]

print(f"Starting hybrid ingestion of {len(tools_to_add)} tools...")

for tool in tools_to_add:
    # 1. Get Dense embedding (Concepts) from Ollama
    dense_vec = get_dense_embedding(tool['desc'])
    if not dense_vec: continue

    # 2. Upsert using Qdrant's Native BM25 Inference for the sparse vector
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                # Using the tool name as the hash seed means updating a tool description
                # will cleanly overwrite the old entry instead of creating a duplicate.
                id=get_consistent_id(tool["name"]),
                vector={
                    "qwen_dense": dense_vec,
                    "keyword_sparse": Document(text=tool["desc"], model="qdrant/bm25")
                },
                payload={
                    "tool_id": tool["name"],
                    "description": tool["desc"]
                }
            )
        ]
    )
print("Ingestion complete!")