import asyncio
import os
import urllib.parse
import aiohttp
import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

class PriceLookupTool(llm.Tool):
    """Tool to find current stock and retail prices"""
    name = "stock_and_retail_price_lookup" 
    description = "Look up the current retail prices/availability or real-time stock market data for an item or company."
    
    parameters = vol.Schema({
        vol.Required(
            "query", 
            description="The product name, store item, corporate name, or market ticker symbol to search."
        ): str,
        vol.Required(
            "api", 
            description="Target API channel. Must be 'retail' (shopping/electronics) or 'stock' (market charts/tickers)."
        ): vol.In([
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
                store_results = await asyncio.gather(*tasks)
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
                    # Strip hyphens and periods to fix variations like "V-O-O" or "A.A.P.L."
                    clean_query = query.replace("-", "").replace(".", "")
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