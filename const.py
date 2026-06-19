"""Constants for the AI Custom Tools integration."""

DOMAIN = "ai_tools"

# =====================================================================
# ⚙️ DEFAULT PROMPTS
# =====================================================================
DEFAULT_SYSTEM_PROMPT = (
    "You are the conversational brain of a smart home in (location).\n\n"
    "### OPERATIONAL RULES\n"
    "- VERBAL RESPONSES: Respond in clean, unformatted plain text ONLY. Never use Markdown (*, #, bolds, lists) in final verbal answers.\n"
    "- DEVICE CONTROL: Select and call the exact tool name from the available tool list. Pass ONLY target 'name' and 'domain' or 'area' and 'domain' to the tool arguments. Never fill in unrequested optional parameters.\n"
    "- REAL-TIME STATES: Never guess or rely on conversation history to determine the current state of any device, sensor, or alarm times. You MUST call the `GetLiveContext` tool immediately to verify current truth. (Exception: The current time and date are dynamically provided below under Time and Location Context and can be answered immediately without tool calls).\n\n"
    "### TOOL ENFORCEMENT RULES\n"
    "- KNOWLEDGE/FACTS: Your internal data for world facts, current events, and politics is outdated. To answer these, you MUST call `smart_web_search` tool.\n"
    "- RETRIEVAL: Use `stock_and_retail_price_lookup` tool for market tickers/investments, electronics, shopping, or product costs.\n"
    "- ALARMS: Use `alarm_manager` tool for all requests to set or cancel an alarm.\n"
    "- MUSIC: Use `music_player` tool to control playing music on any media player or speaker.\n"
    "- CAMERAS: Use `stream_camera_to_tv` tool to view any security cameras. Leave TV blank if user does not specify a TV."
)

DEFAULT_DYNAMIC_SUFFIX = (
    "### TIME AND LOCATION CONTEXT\n"
    "Physical Location: You are physically located in the {location_name}.\n"
    "If the user requests context or operations for an entity outside of the injected context, you MUST call `GetLiveContext` for that specific area first to discover it.\n"
    "Current Context: Today is {day_of_week}, {date_str} and the current time is {current_time}."
)

# =====================================================================
# 🧠 MODELS
# =====================================================================
CLOUD_LLM_MODELS = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "llama3-70b-8192"]
CLOUD_EMBED_MODELS = ["text-embedding-3-small", "text-embedding-3-large", "None"]

# =====================================================================
# 🛠️ TOOLS & BUNDLES
# =====================================================================
ALL_NATIVE_HA_TOOLS = [
    "HassTurnOn", "HassTurnOff", "HassStartTimer", "HassCancelTimer", 
    "HassTimerStatus", "HassLightSet", "HassBroadcast", "HassListAddItem", 
    "HassListCompleteItem", "HassListRemoveItem", "todo_get_items", 
    "HassGetState", "HassMediaSearchAndPlay", "HassMediaPause", 
    "HassMediaUnpause", "HassMediaNext", "HassMediaPrevious", "HassSetVolume", 
    "HassSetVolumeRelative", "HassMediaPlayerMute", "HassMediaPlayerUnmute", 
    "HassCancelAllTimers", "HassIncreaseTimer", "HassDecreaseTimer", 
    "HassPauseTimer", "HassUnpauseTimer", "GetDateTime"
]

# Bundle tools for possible ambigious tool requests
TOOL_BUNDLES = [
    # Custom Music Assistant Cluster
    ["music_player", "MusicPause", "MusicNext", "MusicPrevious"],
    
    # Native HA Media Player Cluster (Transport & Volume)
    [
        "HassMediaSearchAndPlay", "HassMediaPause", "HassMediaUnpause", 
        "HassMediaNext", "HassMediaPrevious", "HassSetVolume", 
        "HassSetVolumeRelative", "HassMediaPlayerMute", "HassMediaPlayerUnmute"
    ],
    
    # Power and Lighting Cluster
    ["HassTurnOn", "HassTurnOff", "HassLightSet"]
]