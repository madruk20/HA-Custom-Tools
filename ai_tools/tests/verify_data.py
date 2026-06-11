import requests
import json

URL = "http://192.168.4.23:8085/query"
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer madruk-local-token"}

# This is the EXACT payload we are putting in __init__.py
payload = {
    "query": "stock market retail price lookup",
    "k": 3,
    "filters": {
        "sector": "procedural"
    }
}

print("========================================")
print(f"📡 SENDING REQUEST TO: {URL}")
print(f"📦 PAYLOAD: {json.dumps(payload)}")
print("========================================\n")

try:
    response = requests.post(URL, json=payload, headers=HEADERS)
    print(f"HTTP STATUS CODE: {response.status_code}\n")
    
    print("RAW DATABASE RESPONSE:")
    try:
        # Try to format it nicely if it's JSON
        parsed_json = response.json()
        print(json.dumps(parsed_json, indent=4))
    except json.JSONDecodeError:
        # If the server throws a weird text error, print the raw text
        print(response.text)
        
except Exception as e:
    print(f"🚨 CONNECTION ERROR: {e}")

print("\n========================================")