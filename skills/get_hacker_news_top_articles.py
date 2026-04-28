import requests
from bs4 import BeautifulSoup

def get_hacker_news_top_articles() -> str:
    """
    Fetches the top 5 stories from Hacker News and returns a summary of each article in a clean string format.

    Returns:
        A string containing the titles and summaries of the top 5 Hacker News articles.
    """
    try:
        # Make an API call to Hacker News
        response = requests.get('https://news.ycombinator.com/')
        
        if response.status_code != 200:
            return f"Failed to retrieve data. Status code: {response.status_code}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the top stories section
        top_stories_section = soup.find('div', class_='athing')
        
        if not top_stories_section:
            return "No top stories found."
        
        articles = []
        
        # Extract the titles and summaries of the top 5 stories
        for i in range(5):
            title = top_stories_section.find_all('a', class_='storylink')[i].text.strip()
            summary = top_stories_section.find_all('span', class_='sitebit comhead')[i].text.strip().split('|')[0].strip()
            articles.append(f"- {title}: {summary}")
        
        # Join the articles into a single string
        result = "Top 5 Hacker News stories:\n" + "\n".join(articles)
        
        return result
    
    except requests.RequestException as e:
        return f"An error occurred: {e}"