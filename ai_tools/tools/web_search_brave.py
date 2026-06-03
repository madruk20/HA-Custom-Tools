import os
import re
import html
import aiohttp
import asyncio
import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

class WebSearchTool(llm.Tool):
    """Web Search Tool using the Brave API"""
    name = "smart_web_search"
    description = "Search the internet to look up comprehensive details on current events or general facts."
                   
    parameters = vol.Schema({
        vol.Required(
            "query", 
            description="The search engine text query. Keep keyword-focused."
        ): str
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