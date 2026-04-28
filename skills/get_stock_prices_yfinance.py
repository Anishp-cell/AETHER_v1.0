import yfinance as yf

def get_stock_prices_yfinance() -> str:
    """
    Fetches the current stock prices of Nvidia, Apple, and Google using yfinance.
    Returns a clean, human-readable string with each company's price.
    """
    try:
        ticker_map = {
            "NVDA": "Nvidia",
            "AAPL": "Apple",
            "GOOGL": "Google"
        }
        tickers = list(ticker_map.keys())
        
        data = yf.download(tickers=tickers, period='1d', progress=False)['Close']
        latest = data.iloc[-1]
        
        lines = []
        for ticker, company in ticker_map.items():
            price = latest.get(ticker, None)
            if price is not None:
                lines.append(f"{company} ({ticker}): ${float(price):,.2f}")
            else:
                lines.append(f"{company} ({ticker}): Price unavailable")
        
        return "Current stock prices — " + " | ".join(lines)
    
    except Exception as e:
        return f"An error occurred while fetching stock prices: {str(e)}"