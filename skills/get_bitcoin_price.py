import requests

def get_bitcoin_price() -> str:
    """
    Fetches the current Bitcoin price in USD.
    Tries CoinGecko first, then falls back to Coinbase public API.
    Returns the price as a clean formatted string.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # --- Primary: CoinGecko ---
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            data = response.json()
            price = data["bitcoin"]["usd"]
            return f"The current Bitcoin price is ${price:,.2f} USD."
    except Exception:
        pass

    # --- Fallback: Coinbase public API ---
    try:
        url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            data = response.json()
            price = float(data["data"]["amount"])
            return f"The current Bitcoin price is ${price:,.2f} USD (via Coinbase)."
    except Exception as e:
        return f"Failed to fetch Bitcoin price from all sources: {str(e)}"

    return "Unable to retrieve Bitcoin price at this time. Please try again later."