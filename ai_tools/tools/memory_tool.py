import os
import sqlite3
import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

class PersonalMemorySearchTool(llm.Tool):
    name = "search_personal_records"
    description = "Search personal archives, historical server logs, and household journals for specific past events or facts."
                   
    parameters = vol.Schema({
        vol.Required(
            "query", 
            description="Keywords to search the local database or log files for (e.g., 'NAS crash', 'estate records')."
        ): str
    })

    async def async_call(self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext) -> dict:
        query = tool_input.tool_args.get("query")
        
        # In a real setup, this path points to where you keep your home markdown files or database
        db_path = "/config/custom_components/ai_tools/data/personal_memory.db"
        
        if not os.path.exists(db_path):
            return {"error": "The personal memory database file is not initialized."}
            
        try:
            # Connect to your local database and run a text search query
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Simple SQL text search over a 'logs' or 'memories' table
            cursor.execute(
                "SELECT content, timestamp FROM memories WHERE content LIKE ? ORDER BY timestamp DESC LIMIT 3", 
                (f"%{query}%",)
            )
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {"result": f"No internal records or logs found matching the query: '{query}'."}
                
            results = [f"[{row[1]}] Record: {row[0]}" for row in rows]
            return {"matched_records": results}
            
        except Exception as e:
            return {"error": f"Failed to read local memory store: {str(e)}"}