import requests

def get_weather(query: str) -> str:
    """
    Scrape current and forecasted weather for a given location using an API like Dark Sky or AccuWeather.

    Parameters:
    query (str): The city name to retrieve the weather data for.

    Returns:
    str: A string describing the result of the execution. If successful, includes weather details.
         If an error occurs, returns an error message.
    """
    try:
        # Replace 'YOUR_API_KEY' with your actual API key from Dark Sky or AccuWeather
        api_key = 'YOUR_API_KEY'
        url = f'https://api.darksky.net/forecast/{api_key}/{query}'
        
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        
        data = response.json()
        
        current_weather = {
            "temperature": data['currently']['temperature'],
            "humidity": data['currently']['humidity']
        }
        
        forecast = []
        for day in data['daily']['data']:
            forecast.append({
                "date": day['time'],
                "summary": day['summary'],
                "temperatureHigh": day['temperatureHigh'],
                "temperatureLow": day['temperatureLow']
            })
        
        result = f"Current Temperature: {current_weather['temperature']}°F, Humidity: {current_weather['humidity']}%"
        result += "\nForecast:\n"
        for i, forecast_day in enumerate(forecast):
            result += f"{i+1}. Date: {forecast_day['date']}, Summary: {forecast_day['summary']}\n"
            result += f"   High: {forecast_day['temperatureHigh']}°F, Low: {forecast_day['temperatureLow']}°F\n"
        
        return result
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {e}"