import requests
import uuid
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.http.models import PointStruct

# --- Configuration ---
QDRANT_URL = "http://192.168.4.23:6333"
OLLAMA_URL = "http://192.168.4.23:11434/api/embeddings"
OLLAMA_MODEL = "qwen-embed-2k:latest"
COLLECTION_NAME = "tools_collection"

qdrant = QdrantClient(url=QDRANT_URL)

# 1. Check if the collection exists
if qdrant.collection_exists(collection_name=COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
else:
    # 2. Create if it doesn't exist
    print(f"Collection '{COLLECTION_NAME}' not found. Creating now...")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print("Collection created successfully.")

# --- MASTER TOOL LIST ---
# Simply add or edit tools here. 
# The script will automatically handle creation or updates
tools_to_add = [
    {
        "name": "music_player",
        "desc": (
            "music_player - This tool controls audio playback on smart speakers and media devices. "
            "Use this tool when the user wants to play, pause, resume, or skip music. "
            "It handles requests for specific song titles, album names, artist names, "
            "or general playback commands like 'play the song' or 'listen to album.' "
            "Capable of routing audio to specific rooms like the Bedroom TV or Theater speakers. "
            "Keywords: play the song, play music, play song, play album, play from artist. "
            "Examples: 'Play the song titled Enter Sandman from Metallica', 'Play the album Core by Stone Temple Pilots', 'Pause the music', 'Play the next track', 'Play the previous track'."
        )
    },
    {
        "name": "stock_and_retail_price_lookup",
        "desc": (
            "stock_and_retail_price_lookup - Tool to lookup retail prices, stock market data, and corporate ticker symbols. "
            "Use when the user asks about the current cost of a product, stock market performance, "
            "or corporate financial metrics. Examples include 'How much does this TV cost?', "
            "Note: Returns real-time or near-real-time data for major retail outlets and global stock exchanges."
            "Keywords: current price, price, stock, retail, retail price, what is the price. "
            "Examples: 'What is the current stock price of Apple?', or 'Show me the latest market charts for NVIDIA.'"
            
        )
    },
    {
        "name": "alarm_manager",
        "desc": (
            "alarm_manager - A tool to set a new alarm or cancel existing ones. Use when the user wants to manage alarms. "
            "User can set alarms in different rooms. "
            "Keywords: set my alarm, cancel my alarm"
            "Examples: 'Set my alarm for 4PM', 'Cancel my alarm'. "
        )
    },
    {
        "name": "HassTurnOn",
        "desc": (
            "HassTurnOn - Use this tool to turn on devices, activate scenes, or power up electronics. "
            "Keywords: turn on, start, power on, activate, switch on. "
            "CRITICAL RULE: Always use this tool to LOCK smart locks or doors. "
            "Examples: 'Turn on the bedroom TV', 'Lock the front door', 'Switch on the fan'."
        )
    },
    {
        "name": "HassTurnOff",
        "desc": (
            "HassTurnOff - Use this tool to turn off devices or power down electronics. "
            "Keywords: turn off, stop, power off, deactivate, switch off, shut down. "
            "CRITICAL RULE: Always use this tool to UNLOCK smart locks or doors. "
            "Examples: 'Turn off the lights', 'Unlock the door', 'Shut off the arcade TV'."
        )
    },
    {
        "name": "HassStartTimer",
        "desc": (
            "HassStartTimer - Use this tool to start a new countdown timer. "
            "Keywords: set a timer, start timer, count down. "
            "Examples: 'Set a timer for 10 minutes', 'Start a 5 minute timer for the laundry'."
        )
    },
    {
        "name": "HassCancelTimer",
        "desc": (
            "HassCancelTimer - Use this tool to stop, cancel, or delete an active countdown timer. "
            "Keywords: cancel timer, stop timer, delete timer. "
            "Examples: 'Cancel the kitchen timer', 'Stop the 10 minute timer'."
        )
    },
    {
        "name": "HassTimerStatus",
        "desc": (
            "HassTimerStatus - Use this tool to check how much time is remaining on an active timer. "
            "Keywords: timer status, time left, time remaining, how much time. "
            "Examples: 'How much time is left on the timer?', 'Check the laundry timer status'."
        )
    },
    {
        "name": "HassLightSet",
        "desc": (
            "HassLightSet - Use this tool to change the specific attributes of smart lights. "
            "Keywords: brightness, dim, color, temperature, percent. "
            "Examples: 'Set the lights to 50%', 'Change the bedroom light to blue', 'Dim the lights'."
        )
    },
    {
        "name": "HassBroadcast",
        "desc": (
            "HassBroadcast - Use this tool to send voice announcements, broadcasts, or text-to-speech messages to smart speakers around the house. "
            "Keywords: broadcast, announce, tell everyone, say message. "
            "Examples: 'Broadcast that dinner is ready', 'Announce to the playroom that it is time for bed'."
        )
    },
    {
        "name": "LoadGame",
        "desc": (
            "LoadGame - Use this tool to load arcade video games. "
            "Keywords: load, play game, start game. "
            "Examples: 'Load game Street Fighter II.  Start game Donkey Kong."
        )
    },
    {
        "name": "MusicPause",
        "desc": (
            "MusicPause - Use this tool to pause currently playing music, media, or TV audio. "
            "Keywords: pause music, stop playback, halt media. "
            "Examples: 'Pause the music', 'Pause the bedroom TV'."
        )
    },
    {
        "name": "MusicNext",
        "desc": (
            "MusicNext - Use this tool to skip forward to the next track, song, or media item in the queue. "
            "Keywords: next song, skip track, go forward, next track. "
            "Examples: 'Play the next song', 'Skip this track', 'Next track'."
        )
    },
    {
        "name": "MusicPrevious",
        "desc": (
            "MusicPrevious - Use this tool to go back to the previous track, song, or media item in the queue. "
            "Keywords: previous song, last track, go back, restart song. "
            "Examples: 'Play the last song again', 'Go back to the previous track', 'previous track'."
        )
    },
    {
        "name": "HassListAddItem",
        "desc": (
            "HassListAddItem - Use this tool to add a new item or task to a shopping list, grocery list, or to-do list. "
            "Keywords: add to list, remember to buy, put on grocery list, new task. "
            "Examples: 'Add milk to the shopping list', 'Put take out the trash on my to do list'."
        )
    },
    {
        "name": "HassListCompleteItem",
        "desc": (
            "HassListCompleteItem - Use this tool to check off, mark as done, or complete an item on a shopping or to-do list. "
            "Keywords: complete task, check off list, mark as done. "
            "Examples: 'Cross milk off the grocery list', 'Mark take out the trash as complete'."
        )
    },
    {
        "name": "HassListRemoveItem",
        "desc": (
            "HassListRemoveItem - Use this tool to completely delete or remove an item from a shopping list or to-do list. "
            "Keywords: delete from list, remove item, clear task. "
            "Examples: 'Remove milk from the shopping list', 'Delete the trash task from my to do list'."
        )
    },
    {
        "name": "todo_get_items",
        "desc": (
            "todo_get_items - Use this tool to read, list, or check what items are currently on a shopping list or to-do list. "
            "Keywords: read list, what is on my list, check shopping list, to do items. "
            "Examples: 'What is on the grocery list?', 'Read my to do list to me', 'Do I have eggs on the shopping list?'"
        )
    }
]

def get_embedding(text):
    """Generates 1024-dimension vector from Ollama."""
    payload = {"model": OLLAMA_MODEL, "prompt": text}
    response = requests.post(OLLAMA_URL, json=payload).json()
    return response.get("embedding")

def get_consistent_id(name):
    """Generates an MD5 hash of the name to ensure unique, deterministic IDs."""
    return hashlib.md5(name.encode()).hexdigest()

print(f"Starting ingestion of {len(tools_to_add)} tools into '{COLLECTION_NAME}'...")

for tool in tools_to_add:
    print(f"Indexing: {tool['name']}...")
    
    # Get embedding
    vec = get_embedding(tool['desc'])
    
    if not vec:
        print(f"Error: Failed to get embedding for {tool['name']}. Skipping.")
        continue

    # Upsert (If ID exists, it overwrites; if not, it adds)
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=get_consistent_id(tool["name"]),
                vector=vec,
                payload={
                    "tool_id": tool["name"],
                    "description": tool["desc"],
                    "tags": ["tool_schema", tool["name"]]
                }
            )
        ]
    )

print("Ingestion complete! All tools are synced.")