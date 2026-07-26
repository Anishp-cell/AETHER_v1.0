import requests
import json

def get_credit_card_deals() -> str:
    """
    Fetches the top 5 credit card deals from Reddit's /r/creditcardsdeals subreddit.

    Returns:
        A string describing the result of the execution.
    """
    try:
        url = 'https://www.reddit.com/r/creditcardsdeals/search/.json'
        
        # Replace with actual access token if needed
        headers = {'Authorization': 'Bearer YOUR_ACCESS_TOKEN'}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = json.loads(response.text)
            results = data['data']['children']
            
            deals = []
            for result in results[:5]:  # Fetch top 5 deals
                title = result['data']['title']
                url = result['data']['url']
                deals.append({'title': title, 'url': url})
            
            return json.dumps(deals)
        else:
            return f"Failed to fetch data: {response.status_code}"
    except requests.RequestException as e:
        return f"An error occurred: {str(e)}"