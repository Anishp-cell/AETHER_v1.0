import requests

def get_cryptoprice() -> str:
    """
    Fetches the current price of Ethereum from CoinGecko.

    Returns:
        A string representing the current price of Ethereum in USD, formatted to two decimal places.
        If an error occurs during the fetch, returns an error message.
    """
    try:
        # API endpoint to fetch Ethereum price from CoinGecko
        url = "https://api.coingecko.com/v3/simple/price?ids=ethereum&vs_currencies=usd"
        
        # Send a GET request to the API
        response = requests.get(url)
        
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Extract and format the Ethereum price
            ethereum_price = data['ethereum']['usd']
            price_format = f"${ethereum_price:.2f}"
            
            return price_format
        else:
            return "Error: Failed to fetch cryptocurrency price"
    except requests.RequestException as e:
        return f"Error: {str(e)}"