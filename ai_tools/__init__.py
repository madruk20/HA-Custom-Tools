import asyncio
import os
import logging
import re
import html
import aiohttp
import voluptuous as vol
from datetime import datetime
from pathlib import Path
import homeassistant.util.dt as dt_util
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.template import Template

_LOGGER = logging.getLogger(__name__)

# ==========================================
# 0.1 CORE PROMPT MEMORY PATCH
# ==========================================
def apply_prompt_patch():
    """Overwrites the entire Assist API preamble to remove all redundancies."""
    try:
        # We define a replacement function that returns ONLY your condensed text.
        # This completely bypasses the hardcoded HA intent and location instructions.
        def custom_preamble(self, llm_context):
            return []
        
        # Override the method directly on the AssistAPI class
        llm.AssistAPI._async_get_preable = custom_preamble
        _LOGGER.info("Successfully overwrote the entire AssistAPI preamble!")
            
    except Exception as e:
        _LOGGER.error(f"Failed to patch LLM prompt: {e}")

# Execute the patch during boot
apply_prompt_patch()

# ==========================================
# 0.2 OLLAMA HTTP PAYLOAD PATCH (STREAM AWARE)
# ==========================================
def apply_ollama_client_patch():
    """Intercepts outgoing payload to destroy thinking tokens, and formats incoming streaming metrics."""
    try:
        import ollama
        import re
        import logging
        
        _LOGGER = logging.getLogger(__name__)

        # Only patch if the attribute exists and is the one we expect
        if not hasattr(ollama.AsyncClient, "chat"):
            _LOGGER.warning("Ollama AsyncClient does not have 'chat' method. Skipping patch.")
            return
        
        original_chat = ollama.AsyncClient.chat
        
        async def patched_chat(self, model, messages=None, **kwargs):
            # --- 1. THE OUTGOING SCRUBBER ---
            if messages:
                for msg in messages:
                    # Handle Pydantic Objects
                    if not isinstance(msg, dict):
                        if getattr(msg, "role", None) == "assistant":
                            if hasattr(msg, "thinking"): setattr(msg, "thinking", None)
                            if hasattr(msg, "thinking_content"): setattr(msg, "thinking_content", None)
                            content = getattr(msg, "content", None)
                            if isinstance(content, str):
                                clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                                setattr(msg, "content", clean_content.strip())
                    # Handle Dictionaries
                    elif isinstance(msg, dict):
                        if msg.get("role") == "assistant":
                            msg.pop("thinking", None)
                            msg.pop("thinking_content", None)
                            if "content" in msg and isinstance(msg["content"], str):
                                clean_content = re.sub(r'<think>.*?</think>', '', msg["content"], flags=re.DOTALL)
                                msg["content"] = clean_content.strip()
                            
            # --- 2. FIRE THE REQUEST ---
            is_stream = kwargs.get("stream", False)
            response = await original_chat(self, model=model, messages=messages, **kwargs)
            
            # --- 3. CATCH VERBOSE METRICS (STREAM AWARE) ---
            if not is_stream:
                try:
                    p_tokens = response.get("prompt_eval_count", 0) if isinstance(response, dict) else getattr(response, "prompt_eval_count", 0)
                    gen_tokens = response.get("eval_count", 0) if isinstance(response, dict) else getattr(response, "eval_count", 0)
                    total_time = (response.get("total_duration", 0) if isinstance(response, dict) else getattr(response, "total_duration", 0)) / 1e9
                    speed = f"{(gen_tokens / total_time):.2f} t/s" if total_time > 0 else "N/A"
                    _LOGGER.info(f"🤖 OLLAMA VERBOSE: Prompt: {p_tokens} tokens | Generated: {gen_tokens} tokens | Time: {total_time:.2f}s | Speed: {speed}")
                except Exception:
                    pass
                return response
            else:
                # Wrap the active stream to catch the final chunk
                async def stream_wrapper(async_gen):
                    async for chunk in async_gen:
                        try:
                            # Ollama only sends eval metrics in the final streamed chunk
                            p_tokens = chunk.get("prompt_eval_count") if isinstance(chunk, dict) else getattr(chunk, "prompt_eval_count", None)
                            if p_tokens: 
                                gen_tokens = chunk.get("eval_count", 0) if isinstance(chunk, dict) else getattr(chunk, "eval_count", 0)
                                total_time = (chunk.get("total_duration", 0) if isinstance(chunk, dict) else getattr(chunk, "total_duration", 0)) / 1e9
                                speed = f"{(gen_tokens / total_time):.2f} t/s" if total_time > 0 else "N/A"
                                _LOGGER.info(f"🤖 OLLAMA VERBOSE: Prompt: {p_tokens} tokens | Generated: {gen_tokens} tokens | Time: {total_time:.2f}s | Speed: {speed}")
                        except Exception:
                            pass
                        yield chunk
                return stream_wrapper(response)

        ollama.AsyncClient.chat = patched_chat
        _LOGGER.info("Successfully patched Ollama client for Streaming Token Scrubbing and Verbose Logging!")
        
    except ImportError:
        logging.getLogger(__name__).debug("Ollama python client not found, skipping patch.")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to patch Ollama client: {e}")

# Execute the patch during boot
apply_ollama_client_patch()

# ==========================================
# 0.3 ASSIST TOOL SUPPRESSION PATCH
# ==========================================
def suppress_assist_tool():
    """Globally removes specific tools with safety checks."""
    try:
        from homeassistant.helpers import llm
        
        # 1. Safety Check: Does the method exist?
        if not hasattr(llm.AssistAPI, "_async_get_tools"):
            _LOGGER.warning("AssistAPI._async_get_tools missing! Skipping patch.")
            return

        original_get_tools = llm.AssistAPI._async_get_tools

        def patched_get_tools(self, llm_context, exposed_entities):
            # 2. Wrap the call in a try/except so if it fails, we still return original tools
            try:
                tools = original_get_tools(self, llm_context, exposed_entities)
                return [t for t in tools if t.name != "HassMediaSearchAndPlay"]
            except Exception as e:
                _LOGGER.error(f"Filter logic failed, returning all tools: {e}")
                return original_get_tools(self, llm_context, exposed_entities)

        llm.AssistAPI._async_get_tools = patched_get_tools
        _LOGGER.info("Successfully applied defensive patch to AssistAPI.")
            
    except Exception as e:
        # 3. This catches errors during the patch application itself
        _LOGGER.error(f"Critical failure patching AssistAPI: {e}")

suppress_assist_tool()

# ==========================================
# 1. SELF-CONTAINED .ENV PARSER
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
# 2. TOOL: LIVE STATES
# ==========================================
class LiveStatesTool(llm.Tool):
    name = "get_live_states"
    description = "CRITICAL STATUS TOOL. Call this immediately if the user asks for device status. Call with NO arguments."
    parameters = vol.Schema({}) 

    async def async_call(self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext) -> dict:
        now = dt_util.now()
        template_str = "{% for entity_id in label_entities('ai') -%} * {{ entity_id }}: {{ states(entity_id) }}\n{% endfor %}"
        device_states = Template(template_str, hass).async_render()
        return {
            "snapshot_type": "LIVE GROUND TRUTH",
            "device_states": device_states
        }


# ==========================================
# 3. TOOL: SMART WEB SEARCH
# ==========================================
class SmartWebSearchTool(llm.Tool):
    name = "smart_web_search"
    description = (
        "Search the internet to look up comprehensive details on current events or general facts."
    )
                   
    parameters = vol.Schema({
        vol.Required("query"): str
    })

    async def async_call(self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext) -> dict:
        query = tool_input.tool_args.get("query")
        
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
        brave_key = os.environ.get("BRAVE_API_KEY")
        if not brave_key: 
            return {"error": "Brave API key not found in server environment variables."}
            
        url = f"https://api.search.brave.com/res/v1/web/search?q={query}"
        headers["X-Subscription-Token"] = brave_key
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=5.0) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for item in data.get("web", {}).get("results", [])[:3]:
                            raw_title = html.unescape(item.get('title', ''))
                            raw_summary = html.unescape(item.get('description', ''))
                            clean_title = re.sub(r'<[^>]+>', '', raw_title)
                            clean_summary = re.sub(r'<[^>]+>', '', raw_summary)
                            results.append(f"Title: {clean_title} - Summary: {clean_summary}")
                        
                        if not results:
                            return {"result": "The search engine returned no results for that query."}
                            
                        return {"results": results}
                    return {"error": f"API failed with HTTP status {response.status}"}
        except asyncio.TimeoutError:
            return {"error": "The web search timed out."}
        except Exception as e:
            return {"error": f"An unexpected search failure occurred: {str(e)}"}


# ===========================================
# 4. ALARM MANAGER TOOL
# ===========================================

class AlarmManagerTool(llm.Tool):
    name = "alarm_manager"
    description = "Manage alarms. Set a new alarm or cancel existing ones. Leave 'room' blank if unspecified."
    parameters = vol.Schema({
        vol.Required("action"): vol.In(["set", "cancel"]),
        vol.Optional("time"): str, # Only needed for 'set'
        vol.Optional("room"): str  # Made optional and generic to handle blank/hallucinated inputs
    })

    def _sanitize_time(self, time_str: str) -> str:
        """Deep cleans whisper-transcribed text to a strict HH:MM:00 format."""
        if not time_str:
            return ""
        
        # 1. Normalize 'a.m.' / 'p.m.' variants -> 'am' / 'pm'
        clean = re.sub(r'([ap])\.?m\.?', r'\1m', time_str.lower().strip())
        
        # 2. Fix mashed numbers from STT (e.g., "109 am" -> "1:09 am", "1115" -> "11:15")
        if ":" not in clean:
            clean = re.sub(r'\b(\d{1,2})(\d{2})\b', r'\1:\2', clean)
        
        # 3. Normalize dots/spaces (e.g., '10.29' or '10 29') to colons ('10:29')
        clean = re.sub(r'(\d+)[\.\s](\d+)', r'\1:\2', clean)
        
        # 4. Try to parse with common variations
        for fmt in ("%I:%M %p", "%I:%M:%S %p", "%H:%M", "%H:%M:%S", "%I:%M"):
            try:
                # Force 24-hour HH:MM:00 return
                return datetime.strptime(clean, fmt).strftime("%H:%M:00")
            except ValueError:
                continue
        
        return time_str # If all else fails, return raw

    async def async_call(self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext) -> dict:
        action = tool_input.tool_args.get("action")
        raw_room = tool_input.tool_args.get("room", "").strip().lower()
        
        valid_rooms = ["bedroom", "playroom"]
        target_room = None

        # --- STAGE 1: Check LLM Input ---
        if raw_room in valid_rooms:
            target_room = raw_room
        else:
            # --- STAGE 2: Resolve via Hardware Context ---
            if llm_context.device_id:
                import homeassistant.helpers.device_registry as dr
                import homeassistant.helpers.area_registry as ar
                
                dev_reg = dr.async_get(hass)
                area_reg = ar.async_get(hass)
                
                device = dev_reg.async_get(llm_context.device_id)
                if device and device.area_id:
                    area = area_reg.async_get_area(device.area_id)
                    if area and area.name:
                        area_clean = area.name.lower().strip()
                        if area_clean in valid_rooms:
                            target_room = area_clean
            
            # --- STAGE 3: Final Safe Fallback ---
            if not target_room:
                target_room = "bedroom"  # Matches system prompt default

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


# ==========================================
# 5. TOOL: PRICE LOOKUP
# ==========================================
class UniversalPriceLookupTool(llm.Tool):
    name = "stock_and_retail_price_lookup" 
    description = "Look up the current retail prices/availability or real-time stock market data for an item or company."
    parameters = vol.Schema({
        vol.Required("query"): str,
        vol.Required("api"): vol.In([
            "retail", "stock", 
            "Retail", "Stock"
        ]) 
    })

    async def async_call(self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext) -> dict:
        query = tool_input.tool_args.get("query")
        api_target = str(tool_input.tool_args.get("api")).lower().replace(" ", "_")
        
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
        
        async with aiohttp.ClientSession() as session:
            
            # ------------------------------------------
            # PATH 1: ABSTRACT RETAIL ENGINE
            # ------------------------------------------
            if api_target == "retail":
                tasks = [
                    self._query_best_buy(session, query)
                ]
                
                # Execute all store lookups concurrently to minimize voice latency
                store_results = await asyncio.gather(*tasks)
                
                # Clean up and filter out any empty or None results from the store list
                active_results = [r for r in store_results if r is not None]
                
                if not active_results:
                    return {"result": f"The item '{query}' could not be located across any integrated retail networks."}
                
                return {
                    "search_query": query,
                    "retail_marketplace_offers": active_results
                }

            # ------------------------------------------
            # PATH 2: STOCK MARKET TICKER LOOKUP
            # ------------------------------------------
            elif api_target == "stock":
                try:
                    import urllib.parse
                    
                    # Strip hyphens and periods to fix "V-O-O" or "A.A.P.L."
                    clean_query = query.replace("-", "").replace(".", "")
                    
                    # URL-encode to safely handle spaces in company names (e.g., "Bank of America")
                    safe_query = urllib.parse.quote(clean_query)
                    
                    # Stage 1: Auto-convert Company Name to Ticker Symbol via Yahoo Search
                    search_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={safe_query}&lang=en-US&region=US"
                    async with session.get(search_url, headers=headers) as search_resp:
                        if search_resp.status != 200:
                            return {"error": f"Stock search catalog rejected with status {search_resp.status}"}
                        
                        search_data = await search_resp.json()
                        quotes = search_data.get("quotes", [])
                        if not quotes:
                            return {"result": f"Could not find an active stock ticker or company listing matching '{query}'."}
                        
                        ticker = quotes[0].get("symbol")
                        company_name = quotes[0].get("longname") or quotes[0].get("shortName") or query
                    
                    # Stage 2: Fetch Live Price Data for the Ticker
                    chart_url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d"
                    async with session.get(chart_url, headers=headers) as chart_resp:
                        if chart_resp.status != 200:
                            return {"error": f"Price lookup failed for ticker {ticker} with status {chart_resp.status}"}
                        
                        chart_data = await chart_resp.json()
                        meta = chart_data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                        current_price = meta.get("regularPrice")
                        currency = meta.get("currency", "USD")
                        
                        if current_price is None:
                            indicators = chart_data.get("chart", {}).get("result", [{}])[0].get("indicators", {}).get("quote", [{}])[0]
                            close_prices = indicators.get("close", [])
                            valid_prices = [p for p in close_prices if p is not None]
                            current_price = valid_prices[-1] if valid_prices else "Unknown"
                        
                        formatted_price = f"${current_price:.2f} {currency}" if isinstance(current_price, (int, float)) else str(current_price)
                        
                        return {
                            "market_context": "Yahoo Finance Live Data",
                            "company_name": company_name,
                            "ticker_symbol": ticker,
                            "current_price": formatted_price
                        }
                except Exception as e:
                    return {"error": f"Failed to query stock market indicators: {str(e)}"}

            return {"error": f"Unsupported lookup channel destination: {api_target}"}

    # ==========================================
    # SUB-FUNCTION VENDORS (The Factory Workers)
    # ==========================================
    async def _query_best_buy(self, session: aiohttp.ClientSession, query: str) -> dict:
        """Isolated sub-function handling Best Buy queries."""
        api_key = os.environ.get("BEST_BUY_API_KEY")
        zip_code = os.environ.get("HOME_ZIP_CODE", "90272")

        if not api_key:
            return {"merchant": "Best Buy", "status": "Error", "message": "API key missing in environment configuration."}

        try:
            # Step 1: Store Location Lookup
            store_url = f"https://api.bestbuy.com/v1/stores(area({zip_code},25))?format=json&apiKey={api_key}"
            store_name = "Local Best Buy"
            async with session.get(store_url) as store_resp:
                if store_resp.status == 200:
                    store_data = await store_resp.json()
                    stores = store_data.get("stores", [])
                    if stores:
                        store_name = f"{stores[0].get('name')} Best Buy ({stores[0].get('address')})"

            # Step 2: Product Pricing Lookup
            prod_url = f"https://api.bestbuy.com/v1/products(search={query})?format=json&show=name,regularPrice,salePrice,onSale,inStoreAvailability&pageSize=1&apiKey={api_key}"
            async with session.get(prod_url) as prod_resp:
                if prod_resp.status == 200:
                    prod_data = await prod_resp.json()
                    products = prod_data.get("products", [])
                    
                    if not products:
                        return None 
                    
                    product = products[0]
                    price = product.get("salePrice") if product.get("onSale") else product.get("regularPrice")
                    stock_note = "In Stock Locally" if product.get("inStoreAvailability") else "Ships to Home"
                    
                    return {
                        "merchant": "Best Buy",
                        "item_matched": product.get("name"),
                        "price": f"${price}",
                        "availability": f"{stock_note} at {store_name}"
                    }
        except Exception:
            return {"merchant": "Best Buy", "status": "Error", "message": "Network/API timeout extraction fail."}
        return None


# ==========================================
# 6. TOOL: MUSIC PLAYER
# ==========================================
class MusicPlayerTool(llm.Tool):
    name = "music_player"
    description = "Control music playback via Music Assistant. Can target specific rooms or devices."
    parameters = vol.Schema({
        vol.Required("action"): vol.In(["play", "pause"]),
        vol.Optional("query"): str,
        vol.Optional("room"): str,
        vol.Optional("device_name"): str
    })

    async def async_call(self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext) -> dict:
        action = tool_input.tool_args.get("action", "")
        query = tool_input.tool_args.get("query")
        target_room = tool_input.tool_args.get("room")
        device_name = tool_input.tool_args.get("device_name")
        
        _LOGGER.info(f"[MusicTool] Tool triggered! Action: {action} | Query: {query} | Room Arg: {target_room} | Device Arg: {device_name}")
        _LOGGER.info(f"[MusicTool] Incoming LLM Context Device ID: {llm_context.device_id}")
        
        import homeassistant.helpers.device_registry as dr
        import homeassistant.helpers.area_registry as ar
        import homeassistant.helpers.entity_registry as er
        import re
        
        dev_reg = dr.async_get(hass)
        area_reg = ar.async_get(hass)
        ent_reg = er.async_get(hass)
        
        def clean_name(name):
            return re.sub(r'[^a-z0-9]', '', str(name).lower())
            
        target_area_id = None
        target_area_obj = None
        
        # --- STAGE 1: Determine Target Area ---
        if target_room:
            for area in area_reg.areas.values():
                if clean_name(area.name) == clean_name(target_room) or clean_name(area.normalized_name) == clean_name(target_room):
                    target_area_id = area.id
                    target_area_obj = area
                    break
            _LOGGER.info(f"[MusicTool] Stage 1 (Explicit Room): Matched '{target_room}' to Area ID: {target_area_id}")
        elif llm_context.device_id:
            source_device = dev_reg.async_get(llm_context.device_id)
            if source_device and source_device.area_id:
                target_area_id = source_device.area_id
                target_area_obj = area_reg.async_get_area(target_area_id)
            _LOGGER.info(f"[MusicTool] Stage 1 (Context Room): Mic Device '{source_device.name if source_device else 'Unknown'}' mapped to Area: {target_area_obj.name if target_area_obj else 'None'}")
                
        # --- STAGE 2: Find ALL Music Assistant Players ---
        possible_players = []
        _LOGGER.info("[MusicTool] Stage 2: Scanning for Music Assistant players...")
        
        for entity in ent_reg.entities.values():
            if entity.domain == "media_player":
                state_obj = hass.states.get(entity.entity_id)
                
                # NEW STRICT FILTER: Only grab players provisioned by Music Assistant
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

        # SUMMARY LOG OF DETECTED DEVICES
        _LOGGER.info(f"[MusicTool] Stage 2 Complete. Found {len(possible_players)} valid Music Assistant players in the target area:")
        for p in possible_players:
            p_name = p.name or p.original_name or "Unknown Name"
            _LOGGER.info(f"[MusicTool] ---> VALID TARGET: {p_name} ({p.entity_id})")

        # --- STAGE 3: Filter or Default to Voice Satellite ---
        target_media_player = None
        if device_name:
            clean_target = clean_name(device_name)
            for player in possible_players:
                player_strings = clean_name(player.name or "") + clean_name(player.original_name or "") + clean_name(player.entity_id)
                if clean_target in player_strings:
                    target_media_player = player.entity_id
                    break
            _LOGGER.info(f"[MusicTool] Stage 3 (Explicit Filter): '{device_name}' matched to: {target_media_player}")
        else:
            # DEFAULTING LOGIC HIERARCHY
            if llm_context.device_id:
                source_device = dev_reg.async_get(llm_context.device_id)
                if source_device:
                    source_clean = clean_name(source_device.name_by_user or source_device.name or "")
                    
                    # Attempt 1: Match the voice satellite's name directly to a Music Assistant player
                    for player in possible_players:
                        player_strings = clean_name(player.name or "") + clean_name(player.original_name or "") + clean_name(player.entity_id)
                        if source_clean and (source_clean in player_strings or player_strings in source_clean):
                            target_media_player = player.entity_id
                            break
            
            # Attempt 2: Look for keywords indicating it's the primary room speaker
            if not target_media_player:
                for player in possible_players:
                    if "voice" in player.entity_id or "speaker" in player.entity_id or "soundbar" in player.entity_id:
                        target_media_player = player.entity_id
                        break

            # Attempt 3: Absolute fallback
            if not target_media_player and possible_players:
                target_media_player = possible_players[0].entity_id
                
            _LOGGER.info(f"[MusicTool] Stage 3 (Defaulting Logic): Selected Target Player -> {target_media_player}")

        # --- THE HARD STOP ---
        # If we reach this point and STILL have no target_media_player, abort gracefully.
        if not target_media_player:
            failed_target = device_name if device_name else target_room if target_room else "the requested area"
            _LOGGER.error(f"[MusicTool] Stage 3 failed to find ANY valid Music Assistant target for '{failed_target}'.")
            return {"error": f"I could not find any active Music Assistant speakers to play music on for '{failed_target}'."}
                
        # --- STAGE 4: Execute Action ---
        resolved_entity = ent_reg.async_get(target_media_player)
        resolved_name = resolved_entity.name or resolved_entity.original_name or target_media_player
        
        # ======================
        # PLAY MUSIC LOGIC
        # ======================
        if action == "play":
            if not query:
                return {"error": "A 'query' string must be provided when the action is 'play'."}
            
            _LOGGER.info(f"[MusicTool] Stage 4: Firing 'music_assistant.play_media' service call to Core. Target: {target_media_player}, Payload: {query}")
            try:
                # 1. Send the command
                await hass.services.async_call(
                    "music_assistant", "play_media",
                    {"entity_id": target_media_player, "media_id": query},
                    blocking=True,
                    context=llm_context.context
                )
                
                # 2. Give Music Assistant a moment to buffer and update the state
                await asyncio.sleep(2.5) 
                
                # 3. Check the actual state of the speaker
                current_state_obj = hass.states.get(target_media_player)
                current_state = current_state_obj.state if current_state_obj else "unknown"
                
                _LOGGER.info(f"[MusicTool] Stage 4: Post-command state for {target_media_player} is '{current_state}'")
                
                if current_state in ["playing", "buffering"]:
                    return {"result": f"Successfully started playing '{query}' on the {resolved_name}."}
                else:
                    return {"error": f"I tried to play '{query}' on the {resolved_name}, but it failed to start. The speaker state is currently '{current_state}'."}
            
            except Exception as e:
                _LOGGER.error(f"[MusicTool] Stage 4 CRASHED: Service call threw exception: {str(e)}")
                error_msg = str(e)
                if "Could not resolve" in error_msg:
                    return {"error": f"Music Assistant could not find '{query}' in the local library. Tell the user you don't have that music."}
                return {"error": f"Playback failed with error: {error_msg}"}
        
        # =======================
        # PAUSE MUSIC LOGIC
        # =======================
        elif action == "pause":
            targets_to_pause = []
            try:
                # --- SMART PAUSE LOGIC ---
                # 1. If the user named a specific device, stick to the target resolved in Stage 3
                if device_name:
                    targets_to_pause.append(target_media_player)
                
                # 2. If no device was named, scan the current room for ANY actively playing media
                else:
                    _LOGGER.info("[MusicTool] Stage 4: No specific device named for pause. Scanning room for active playback...")
                    for entity in ent_reg.entities.values():
                        if entity.domain == "media_player":
                            
                            # Match the room location
                            ent_area = entity.area_id
                            if not ent_area and entity.device_id:
                                dev = dev_reg.async_get(entity.device_id)
                                if dev: ent_area = dev.area_id
                            
                            in_target_area = (target_area_id is None) or (ent_area == target_area_id)
                            
                            if in_target_area:
                                state_obj = hass.states.get(entity.entity_id)
                                # Catch anything currently playing or buffering
                                if state_obj and state_obj.state in ["playing", "buffering"]:
                                    targets_to_pause.append(entity.entity_id)
                
                # 3. Absolute Fallback: If nothing is actively playing, just pause the default Stage 3 target
                if not targets_to_pause:
                    targets_to_pause = [target_media_player]

                _LOGGER.info(f"[MusicTool] Stage 4: Firing 'media_player.media_pause' to Core. Targets: {targets_to_pause}")
                
                # Fire the command to all identified targets
                for target in targets_to_pause:
                    await hass.services.async_call(
                        "media_player", "media_pause",
                        {"entity_id": target},
                        blocking=False,
                        context=llm_context.context
                    )
                
                # Return a context-aware response to the LLM
                if len(targets_to_pause) > 1 or (not device_name and targets_to_pause[0] != target_media_player):
                    return {"result": "Successfully paused all active media playback in the requested room."}
                else:
                    return {"result": f"Successfully sent pause command to the {resolved_name}."}
            except Exception as e:
                _LOGGER.error(f"[MusicTool] : Service call threw exception: {str(e)}")
                error_msg = str(e)
                return {"error": f"Pause of playback failed with error: {error_msg}"}
        

# ==========================================
# 7. API WORKSPACE WRAPPER
# ==========================================
class AiToolsAPI(llm.API):
    id = "custom"
    name = "Custom API"

    def __init__(self, hass: HomeAssistant):
        self.hass = hass

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
            """Return the API instance with your custom tools and defensive context lookups."""
            
            # 1. Define custom tools
            tools = [
                LiveStatesTool(), 
                SmartWebSearchTool(), 
                AlarmManagerTool(), 
                UniversalPriceLookupTool(), 
                MusicPlayerTool()
            ]

            location_name = "Unknown"
            
            if llm_context.device_id:
                try:
                    import homeassistant.helpers.device_registry as dr
                    import homeassistant.helpers.area_registry as ar
                    
                    dev_reg = dr.async_get(self.hass)
                    area_reg = ar.async_get(self.hass)
                    
                    device = dev_reg.async_get(llm_context.device_id)
                    
                    if not device:
                        _LOGGER.debug("Device ID %s not found in registry.", llm_context.device_id)
                    elif device.area_id:
                        area = area_reg.async_get_area(device.area_id)
                        if area:
                            location_name = area.name
                        else:
                            _LOGGER.debug("Area ID %s associated with device %s not found.", device.area_id, llm_context.device_id)
                    else:
                        _LOGGER.debug("Device %s exists but has no area assigned.", llm_context.device_id)
                        
                except Exception as e:
                    _LOGGER.error("Error retrieving registry context: %s", e)

            # 3. Time and Room Location Context
            now = dt_util.now()
            static_prompt = (
                f"Physical Location: You are physically located in the {location_name}. "
                f"Current Context: Today is {now.strftime('%A, %B %-d, %Y')} and the current time is {now.strftime('%-I:%M %p')}. "
            )
            
            return llm.APIInstance(
                api=self, 
                api_prompt=static_prompt,
                llm_context=llm_context,
                tools=tools
            )

async def async_setup(hass: HomeAssistant, config: dict):
    llm.async_register_api(hass, AiToolsAPI(hass))
    return True