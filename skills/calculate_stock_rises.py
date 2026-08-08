import yfinance as yf

def calculate_stock_rises() -> str:
    """
    Fetches current prices and percentage changes for NVDA, AAPL, and GOOGL using yfinance.
    """
    try:
        results = []
        for ticker_symbol in ["NVDA", "AAPL", "GOOGL"]:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="5d")
            if len(hist) >= 2:
                prev_close = hist["Close"].iloc[-2]
                curr_price = hist["Close"].iloc[-1]
                pct_change = ((curr_price - prev_close) / prev_close) * 100.0
                sign = "+" if pct_change >= 0 else ""
                results.append(f"{ticker_symbol}: ${curr_price:.2f} ({sign}{pct_change:.2f}%)")
            elif len(hist) == 1:
                curr_price = hist["Close"].iloc[-1]
                results.append(f"{ticker_symbol}: ${curr_price:.2f} (Latest)")
            else:
                results.append(f"{ticker_symbol}: Data unavailable")

        return "\n".join(results)
    except Exception as e:
        return f"[Stock Calculation Error]: {str(e)}"