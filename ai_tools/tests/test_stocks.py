import asyncio
import aiohttp
import sys

# ==========================================
# YAHOO FINANCE MARKET RETRIEVAL TESTER
# ==========================================
async def check_stock_price(query: str):
    """Simulates the Home Assistant tool logic step-by-step with verbose logging."""
    
    # Crucial Header: Tells Yahoo's servers you are a real desktop web browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print(f"\n[STEP 1] Querying Yahoo Search directory for: '{query}'...")
    search_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&lang=en-US&region=US"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(search_url, headers=headers) as search_resp:
                print(f"[DEBUG] Search HTTP Response Code: {search_resp.status}")
                
                if search_resp.status != 200:
                    print(f"❌ Error: Search API returned status code {search_resp.status}")
                    return
                
                search_data = await search_resp.json()
                quotes = search_data.get("quotes", [])
                
                if not quotes:
                    print(f"⚠️ Notice: No stock ticker matches found for text '{query}'.")
                    return
                
                # Extract identity parameters from the first search row returned
                first_match = quotes[0]
                ticker = first_match.get("symbol")
                company_name = first_match.get("longname") or first_match.get("shortName") or query
                exchange = first_match.get("exchange")
                
                print(f"✅ Target Identified! Company: {company_name} | Ticker: {ticker} (Exchange: {exchange})")
            
            # ------------------------------------------
            # STAGE 2: LIVE METRIC EXTRACTION
            # ------------------------------------------
            print(f"\n[STEP 2] Fetching live trading profile charts for ticker: {ticker}...")
            chart_url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d"
            
            async with session.get(chart_url, headers=headers) as chart_resp:
                print(f"[DEBUG] Chart HTTP Response Code: {chart_resp.status}")
                
                if chart_resp.status != 200:
                    print(f"❌ Error: Chart API returned status code {chart_resp.status}")
                    return
                
                chart_data = await chart_resp.json()
                result_node = chart_data.get("chart", {}).get("result", [{}])[0]
                meta = result_node.get("meta", {})
                
                current_price = meta.get("regularPrice")
                currency = meta.get("currency", "USD")
                prev_close = meta.get("previousClose")
                
                # Dynamic fallback parsing logic if the default realtime node is empty (e.g., after-hours adjustments)
                if current_price is None:
                    print("[INFO] Live regularPrice node missing. Parsing historical indicators interval index...")
                    indicators = result_node.get("indicators", {}).get("quote", [{}])[0]
                    close_prices = indicators.get("close", [])
                    valid_prices = [p for p in close_prices if p is not None]
                    if valid_prices:
                        current_price = valid_prices[-1]
                
                # Render diagnostics output panel
                if current_price and current_price != "Unknown":
                    print("\n--- FINAL PARSED APP DICTIONARY PAYLOAD ---")
                    print(f"🏢 Company Name:  {company_name}")
                    print(f"🎫 Ticker Symbol: {ticker}")
                    print(f"💵 Current Price: {current_price:.2f} {currency}")
                    
                    if isinstance(prev_close, (int, float)):
                        net_change = current_price - prev_close
                        pct_change = (net_change / prev_close) * 100
                        sign = "+" if net_change >= 0 else ""
                        print(f"📊 Market Shift:  {sign}{net_change:.2f} ({sign}{pct_change:.2f}%)")
                    print("--------------------------------------------")
                else:
                    print("❌ Error: Unable to locate structural price coordinates in Yahoo JSON payload.")
                    
        except Exception as e:
            print(f"❌ Script Exception Encountered: {str(e)}")

# ==========================================
# INTERACTIVE CONSOLE EXECUTION LOOP
# ==========================================
async def main():
    print("====================================================")
    print("   Yahoo Finance Core Search Module Debug Runner   ")
    print("====================================================")
    while True:
        target = input("\nEnter a company name or stock ticker (or 'exit'): ").strip()
        if target.lower() == 'exit':
            sys.exit(0)
        if not target:
            continue
        await check_stock_price(target)

if __name__ == "__main__":
    # Handle event loop execution profile safely
    asyncio.run(main())