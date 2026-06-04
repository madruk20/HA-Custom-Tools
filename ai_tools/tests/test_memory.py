import sqlite3
import json
import logging
import re
import requests
from datetime import datetime

# ==========================================
# 1. LOGGING & CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG so you can see exactly what the LLM outputs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

OLLAMA_URL = "http://192.168.4.23:11434/api/chat"
TARGET_MODEL = "qwen3.6:35b-a3b"
DB_PATH = "test_memory.db"

# ==========================================
# 2. DATABASE INITIALIZATION
# ==========================================
def init_db(db_connection):
    """Checks if the database and table exist, creating them if necessary."""
    cursor = db_connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            fact_key TEXT NOT NULL,
            fact_value TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(category, fact_key)
        );
    """)
    db_connection.commit()
    logging.info("Database schema verified/initialized.")

# ==========================================
# 3. LLM API CONNECTION (The Bypass Hack)
# ==========================================
def call_llm(prompt_text, pipeline_step_name):
    """Sends payload to Ollama. Uses Assistant Prefill to bypass thinking tokens."""
    payload = {
        "model": TARGET_MODEL, 
        "messages": [
            {
                "role": "system",
                "content": "You are a precise data extraction and optimization subroutine. Output ONLY the requested JSON array."
            },
            {
                "role": "user",
                "content": prompt_text
            },
            {
                # THE TRAP: We trick the model into believing its thinking phase is done
                "role": "assistant",
                "content": "<think>\n</think>\n["
            }
        ],
        "stream": False,
        "keep_alive": -1,  
        "options": {
            "temperature": 0.0,     # Strict, predictable output
            "num_predict": 1000     # Ample safety runway for massive JSON arrays
        }
    }
    
    logging.info(f"[{pipeline_step_name}] Sending context to LLM via Chat API ({TARGET_MODEL}).")
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120) 
        response.raise_for_status()
        
        response_data = response.json()
        response_text = response_data.get("message", {}).get("content", "")
        
        # RECONSTRUCTION: Manually stitch the bracket back onto the front
        full_output = "[" + response_text
        
        logging.debug(f"[{pipeline_step_name}] Reconstructed LLM Output:\n{full_output}\n")
        
        # Extract validated JSON array
        match = re.search(r'\[.*\]', full_output, re.DOTALL)
        if match:
            clean_json_string = match.group(0)
            return json.loads(clean_json_string)
        else:
            logging.error(f"[{pipeline_step_name}] Failed to find valid JSON array. Raw response: {response_text}")
            return []
            
    except Exception as e:
        logging.error(f"[{pipeline_step_name}] LLM invocation failed: {e}", exc_info=True)
        return []

# ==========================================
# 4. DETERMINISTIC CLEANUP (Python Regex)
# ==========================================
def cleanup_expired_schedule_items(db_cursor):
    """Pulls schedule items, checks bracketed timestamps, and deletes expired ones."""
    db_cursor.execute("SELECT id, fact_value FROM memory WHERE category = 'schedule'")
    rows = db_cursor.fetchall()
    
    ids_to_delete = []
    current_time = datetime.now()
    
    for row_id, fact_value in rows:
        match = re.search(r'\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\]', fact_value)
        if match:
            try:
                event_time = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                if event_time < current_time:
                    ids_to_delete.append(row_id)
            except ValueError:
                continue
                
    if ids_to_delete:
        placeholders = ','.join('?' for _ in ids_to_delete)
        db_cursor.execute(f"DELETE FROM memory WHERE id IN ({placeholders})", ids_to_delete)
        logging.info(f"Python parser identified and deleted {len(ids_to_delete)} expired schedule items.")

# ==========================================
# 5. DB UTILITIES & CONVERSATION CLEANER
# ==========================================
def clean_conversation_generic(raw_log_string):
    """Removes HA system messages and tool calls from the conversation history."""
    try:
        raw_history = json.loads(raw_log_string)
    except Exception as e:
        logging.error(f"Failed to parse raw conversation log JSON: {e}")
        return []

    cleaned = []
    skip_next = False
    
    for i in range(len(raw_history) - 1, -1, -1):
        item = raw_history[i]
        
        if item.get("agent_id") == "conversation.home_assistant":
            skip_next = True
            continue
            
        if item.get("role") == "tool_result" or "tool_calls" in item:
            continue
            
        if item.get("role") == "user" and skip_next:
            skip_next = False
            continue
            
        cleaned.insert(0, item) 
        
    logging.info(f"Filtered conversation history. Kept {len(cleaned)} out of {len(raw_history)} items.")
    return cleaned

def fetch_db_facts_by_categories(categories, db_cursor):
    """Fetches existing facts grouped by category for LLM context."""
    context_data = {cat: [] for cat in categories}
    placeholders = ",".join("?" for _ in categories)
    query = f"SELECT category, fact_key, fact_value FROM memory WHERE category IN ({placeholders})"
    
    db_cursor.execute(query, categories)
    for category, key, value in db_cursor.fetchall():
        context_data[category].append({"fact_key": key, "fact_value": value})
        
    return context_data

# ==========================================
# 6. PROMPT BUILDERS
# ==========================================
def build_extraction_prompt(conversation_history):
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_day_str = datetime.now().strftime('%A, %B %d, %Y')
    
    return f"""
You are a precise data extraction subroutine. 
Current Time Anchor: {current_time_str} ({current_day_str})

Analyze the following conversation history and extract any new, updated, or explicit facts about the user.
Valid Categories: ['food', 'general', 'schedule', 'locations', 'phone numbers', 'birthdays']

CRITICAL SCHEMA RULES:
1. 'schedule': Calculate the absolute date. The 'fact_key' MUST be the event description in snake_case. The 'fact_value' MUST be ONLY the timestamp ("YYYY-MM-DD HH:MM:SS"). 
   - TEMPORAL OVERRIDE: If the user says "Next [Day]", "This [Day]", or just a day name, it ALWAYS means the VERY FIRST upcoming occurrence of that day. For example, if today is Thursday, "Next Friday" must be calculated as tomorrow. Do not skip weeks.
2. 'birthdays': The 'fact_key' MUST be the person's name in snake_case. The 'fact_value' MUST be ONLY the date ("MM-DD").
3. All other categories: Invent a concise snake_case 'fact_key' and put the detail in 'fact_value'.

Conversation History:
{json.dumps(conversation_history, indent=2)}

Output JSON format ONLY:
[
  {{"category": "schedule", "fact_key": "dentist_appointment", "fact_value": "2026-06-08 14:00:00"}},
  {{"category": "birthdays", "fact_key": "uncle_bob", "fact_value": "11-05"}},
  {{"category": "food", "fact_key": "favorite_drink", "fact_value": "Water"}}
]
"""

def build_reconciliation_prompt(extracted_facts, db_context, conversation_history):
    current_time_str = datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')
    
    return f"""
Current Timestamp Context: {current_time_str}

Conversation History Reference:
{json.dumps(conversation_history, indent=2)}

Newly Discovered Conversational Facts:
{json.dumps(extracted_facts, indent=2)}

Existing Local Database Content:
{json.dumps(db_context, indent=2)}

Task Instruction:
Compare newly extracted facts against existing entries. Output a raw JSON array mapping out actions: "insert", "update", or "delete". 

CRITICAL RULES:
1. DO NOT recalculate dates or change the formatting from the Newly Discovered Facts. 
2. Copy the 'fact_key' and 'fact_value' EXACTLY as they appear in the Newly Discovered Facts.
3. Your only job is to append the correct "action".

Format:
[
  {{
    "category": "schedule",
    "fact_key": "flight_to_new_york",
    "fact_value": "2026-06-12 06:00:00",
    "action": "insert"
  }}
]
"""

# ==========================================
# 7. PIPELINE ORCHESTRATOR
# ==========================================
def reconcile_memory_pipeline(extracted_facts, conversation_history, db_connection):
    cursor = db_connection.cursor()
    
    target_categories = list(set(fact['category'] for fact in extracted_facts))
    if not target_categories:
        logging.info("No new conversational facts extracted. Exiting pipeline early.")
        return []
        
    db_context = fetch_db_facts_by_categories(target_categories, cursor)
    total_db_facts = sum(len(facts) for facts in db_context.values())
    
    final_transactions = []
    
    if total_db_facts > 150:
        logging.info(f"High context data mass ({total_db_facts} facts). Splitting operations by category.")
        for category in target_categories:
            single_cat_db = {category: db_context.get(category, [])}
            single_cat_extracted = [f for f in extracted_facts if f['category'] == category]
            
            if single_cat_extracted:
                prompt = build_reconciliation_prompt(single_cat_extracted, single_cat_db, conversation_history)
                cat_transactions = call_llm(prompt, f"RECONCILE: {category.upper()}")
                final_transactions.extend(cat_transactions)
    else:
        logging.info(f"Favorable context data profile ({total_db_facts} facts). Processing single batch payload.")
        prompt = build_reconciliation_prompt(extracted_facts, db_context, conversation_history)
        final_transactions = call_llm(prompt, "RECONCILE: BATCH")
        
    return final_transactions

def execute_sql_transactions(transactions, db_connection):
    cursor = db_connection.cursor()
    for tx in transactions:
        action = tx.get("action")
        category = tx.get("category")
        key = tx.get("fact_key")
        val = tx.get("fact_value")
        
        if not category or not key:
            continue
            
        if action == "insert":
            logging.info(f"Executing [INSERT] -> Category: {category} | Key: {key} | Value: {val}")
            cursor.execute(
                "INSERT OR REPLACE INTO memory (category, fact_key, fact_value, timestamp) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (category, key, val)
            )
        elif action == "update":
            logging.info(f"Executing [UPDATE] -> Category: {category} | Key: {key} | Value: {val}")
            cursor.execute(
                "UPDATE memory SET fact_value = ?, timestamp = CURRENT_TIMESTAMP WHERE category = ? AND fact_key = ?",
                (val, category, key)
            )
        elif action == "delete":
            logging.info(f"Executing [DELETE] -> Category: {category} | Key: {key}")
            cursor.execute("DELETE FROM memory WHERE category = ? AND fact_key = ?", (category, key))
            
    db_connection.commit()

# ==========================================
# 8. MAIN TEST EXECUTION
# ==========================================
def main():
    logging.info("--- STARTING LOCAL MEMORY STRESS TEST (200+ FACTS) ---")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Initialize DB and completely wipe it for a clean stress test
    init_db(conn)
    cursor.execute("DELETE FROM memory")
    conn.commit()
    
    # 2. Inject 240 Dummy Facts (40 per category)
    logging.info("Injecting 240 dummy facts into the database...")
    categories = ['food', 'general', 'schedule', 'locations', 'phone numbers', 'birthdays']
    dummy_facts = []
    
    for cat in categories:
        for i in range(1, 41):
            # Seed specific facts we plan to update or delete during the test
            if cat == 'food' and i == 1:
                dummy_facts.append((cat, 'favorite_drink', 'Tea'))
            elif cat == 'general' and i == 1:
                dummy_facts.append((cat, 'wifi_password', 'old_password_123'))
            elif cat == 'schedule' and i == 1:
                dummy_facts.append((cat, 'expired_meeting', '[2020-01-01 12:00:00] Old Team Meeting'))
            else:
                dummy_facts.append((cat, f'dummy_key_{i}', f'Dummy value {i} for {cat}'))

    cursor.executemany("""
        INSERT INTO memory (category, fact_key, fact_value)
        VALUES (?, ?, ?)
    """, dummy_facts)
    conn.commit()
    logging.info("Dummy data injection complete.")
    
    # 3. Run Native Python Schedule Cleanup
    cleanup_expired_schedule_items(cursor)
    conn.commit()
    
    # 4. Dummy Conversation History (Targets 4 categories: Food, General, Birthdays, Schedule)
    raw_history_string = json.dumps([
        {"role": "user", "content": "I don't like Tea anymore, my favorite drink is now Water."},
        {"role": "assistant", "content": "Got it. I updated your favorite drink."},
        {"role": "user", "content": "Also, change the wifi password to SuperSecretDragon99."},
        {"role": "assistant", "content": "Wifi password updated."},
        {"role": "user", "content": "Uncle Bob's birthday is on November 5th."},
        {"role": "assistant", "content": "I'll remember Uncle Bob's birthday."},
        {"role": "user", "content": "Remind me I have a flight to New York next Friday at 6 AM."}
    ])
    
    cleaned_history = clean_conversation_generic(raw_history_string)
    
    # 5. Phase 1: Extract Facts
    extraction_prompt = build_extraction_prompt(cleaned_history)
    extracted_facts = call_llm(extraction_prompt, "PHASE 1: EXTRACTION")
    
    if not extracted_facts:
        logging.info("No facts found by LLM in Phase 1.")
    else:
        logging.debug(f"Phase 1 Extracted Facts:\n{json.dumps(extracted_facts, indent=2)}")
        
        # 6. Phase 2: Orchestrate and Reconcile (This should trigger the split logic!)
        transactions = reconcile_memory_pipeline(extracted_facts, cleaned_history, conn)
        
        # 7. Execute SQL
        if transactions:
            execute_sql_transactions(transactions, conn)
        else:
            logging.info("No memory modifications to execute.")
            
    # Verify final state for the modified records
    logging.info("\n" + "="*40 + "\n      STRESS TEST VERIFICATION\n" + "="*40)
    
    # We only print the ones we care about to avoid flooding the console with 200 dummy facts
    cursor.execute("""
        SELECT category, fact_key, fact_value FROM memory 
        WHERE fact_key IN ('favorite_drink', 'wifi_password', 'uncle_bob') 
        OR category = 'schedule'
    """)
    rows = cursor.fetchall()
    for row in rows:
        print(f"[{row[0].upper()}] {row[1]}: {row[2]}")
        
    # Verify the expired meeting was deleted
    cursor.execute("SELECT COUNT(*) FROM memory WHERE fact_key = 'expired_meeting'")
    if cursor.fetchone()[0] == 0:
        print("[CLEANUP] expired_meeting successfully deleted.")
    else:
        print("[CLEANUP FAILED] expired_meeting still exists.")
        
    print("="*40 + "\n")
        
    conn.close()

if __name__ == "__main__":
    main()