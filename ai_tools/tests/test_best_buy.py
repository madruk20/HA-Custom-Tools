import os
import re
import html
import asyncio
import aiohttp
from pathlib import Path

# ==========================================
# 1. LOCAL .ENV PARSER FOR STANDALONE RUNS
# ==========================================
CURRENT_DIR = Path(__file__).parent
ENV_FILE_PATH = CURRENT_DIR / ".env"

if ENV_FILE_PATH.exists():
    print(f"[INFO] Loading environment variables from {ENV_FILE_PATH}")
    with open(ENV_FILE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")
else:
    print(f"[WARNING] No .env file found at {ENV_FILE_PATH}")

# ==========================================
# 2. THE STANDALONE CORE FUNCTION
# ==========================================
async def check_best_buy(query: str) -> dict:
    """Standalone client to check Best Buy price and local availability."""
    api_key = os.environ.get("BEST_BUY_API_KEY")
    zip_code = os.environ.get("HOME_ZIP_CODE", "90272")

    if not api_key:
        return {"error": "Best Buy API key not found in environment/ .env file."}

    print(f"[DEBUG] Initiating search for: '{query}' around ZIP: {zip_code}...")

    try:
        async with aiohttp.ClientSession() as session:
            # Stage 1: Store Location Lookup
            store_url = f"https://api.bestbuy.com/v1/stores(area({zip_code},25))?format=json&apiKey={api_key}"
            store_name = "Local Best Buy"
            
            async with session.get(store_url) as store_resp:
                if store_resp.status == 200:
                    store_data = await store_resp.json()
                    stores = store_data.get("stores", [])
                    if stores:
                        store_name = f"{stores[0].get('name')} Best Buy ({stores[0].get('address')})"
                else:
                    print(f"[WARN] Store lookup failed with status: {store_resp.status}")

            # Stage 2: Product Inventory and Price Lookup
            prod_url = f"https://api.bestbuy.com/v1/products(search={query})?format=json&show=name,regularPrice,salePrice,onSale,onlineAvailability,inStoreAvailability&pageSize=1&apiKey={api_key}"
            
            async with session.get(prod_url) as prod_resp:
                if prod_resp.status == 200:
                    prod_data = await prod_resp.json()
                    products = prod_data.get("products", [])
                    
                    if not products:
                        return {
                            "status": "NOT_FOUND",
                            "message": f"Closest store: {store_name}. Product '{query}' did not match any active inventory."
                        }
                    
                    product = products[0]
                    name = product.get("name")
                    reg_price = product.get("regularPrice")
                    sale_price = product.get("salePrice")
                    on_sale = product.get("onSale")
                    is_in_stock_locally = product.get("inStoreAvailability")
                    
                    # Formatting text output
                    price_string = f"${sale_price}" if on_sale else f"${reg_price}"
                    sale_note = " (ON SALE!)" if on_sale else ""
                    stock_status = f"In Stock at {store_name}." if is_in_stock_locally else "Out of Stock locally (Online Only)."
                    
                    return {
                        "status": "SUCCESS",
                        "product_found": name,
                        "current_price": price_string + sale_note,
                        "availability": stock_status,
                        "raw_payload_sample": {
                            "regularPrice": reg_price,
                            "salePrice": sale_price,
                            "inStoreAvailability": is_in_stock_locally
                        }
                    }
                
                return {"error": f"Product API returned HTTP status {prod_resp.status}"}

    except Exception as e:
        return {"error": f"Network or Parsing Exception: {str(e)}"}

# ==========================================
# 3. INTERACTIVE EXECUTION WRAPPER
# ==========================================
async def main():
    print("\n=== Best Buy API Tool Test Runner ===")
    while True:
        item = input("\nEnter a product to look up (or type 'exit' to quit): ").strip()
        if item.lower() == 'exit':
            break
        if not item:
            continue
            
        response = await check_best_buy(item)
        
        print("\n--- RESULT RECIEVED ---")
        if "error" in response:
            print(f"❌ Error: {response['error']}")
        else:
            print(f"📦 Found: {response.get('product_found')}")
            print(f"💰 Price: {response.get('current_price')}")
            print(f"📍 Stock: {response.get('availability')}")
            print(f"🔍 Raw Metadata Check: {response.get('raw_payload_sample')}")

if __name__ == "__main__":
    # Standard Python async wrapper execution
    asyncio.run(main())