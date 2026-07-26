import requests
import json

def get_top_credit_posts() -> str:
    """
    Fetches the top 5 credit card deals, sign-up bonuses, and offers from r/CreditCards on Reddit.

    Returns:
        A JSON-formatted string containing the titles and URLs of the top credit card deal posts.
    """
    try:
        # Search r/CreditCards for deal/offer/bonus posts
        url = "https://api.reddit.com/r/CreditCards/search.json"
        params = {
            "q": "deal OR offer OR bonus",
            "restrict_sr": "1",
            "sort": "relevance",
            "t": "week",
            "limit": "5"
        }
        
        # Set up headers for the API request with a professional User-Agent to avoid Reddit blocks
        headers = {
            "User-Agent": "AETHER Skill Factory/1.0 (by /u/AetherSkillFactory)"
        }
        
        # Send a GET request to the Reddit API
        response = requests.get(url, params=params, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            posts = data.get('data', {}).get('children', [])
            
            # Extract top 5 posts
            result = {
                "posts": [
                    {
                        "title": post['data']['title'],
                        "url": f"https://reddit.com{post['data']['permalink']}" if post['data'].get('permalink') else post['data']['url']
                    }
                    for post in posts
                ]
            }
            
            return json.dumps(result, indent=2)
        else:
            return f"Failed to retrieve data from Reddit. Status code: {response.status_code}"
    
    except Exception as e:
        return f"An error occurred while fetching Reddit deals: {str(e)}"