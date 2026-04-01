from bs4 import BeautifulSoup
import requests
import json

def search_and_read_web(query: str) -> str:
    """
    Called by the Swarm LLM router to silently search the web for an answer using direct HTML scraping.
    """
    try:
        import urllib.parse
        print(f"\n🌐 [AetherWeb] Silently indexing the web for: '{query}'...")
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # DuckDuckGo's result class is 'result__snippet'
        snippets = soup.find_all('a', class_='result__snippet')
        titles = soup.find_all('h2', class_='result__title')
        
        results = []
        for i in range(min(3, len(snippets))):
            title = titles[i].get_text(strip=True) if i < len(titles) else "Result"
            desc = snippets[i].get_text(strip=True)
            results.append(f"- {title}: {desc}")
            
        if not results:
            return "[AetherWeb] No snippets found on the web for that query."
            
        summary = "\n".join(results)
        return f"[AetherWeb Silent Search Results]:\n{summary}\n\n(Synthesize the above information into a concise spoken answer to the user)."
        
    except Exception as e:
        return f"[AetherWeb Engine Error] {e}"

def read_specific_url(url: str) -> str:
    """
    Used when the Agent needs to read a specific website URL completely.
    """
    try:
        print(f"\n🌐 [AetherWeb] Silently fetching webpage: '{url}'...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove massive structural elements that clutter text
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Restrict to ~4000 chars to avoid blowing up the LLM 1.5b Context Window
        return f"[AetherWeb Page Excerpt ({url})]:\n{text[:4000]}\n\n(Summarize the key information above for the user)."
    except Exception as e:
        return f"[AetherWeb Engine Error] Failed to read {url}: {e}"
