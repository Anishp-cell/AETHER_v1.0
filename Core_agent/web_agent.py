from bs4 import BeautifulSoup
import requests
import json

def _clean_search_query(raw_query: str) -> str:
    """Strip conversational filler from user speech to produce a clean DuckDuckGo query."""
    import re
    q = raw_query.strip()
    # Remove common voice assistant filler phrases
    filler_patterns = [
        r"^(hey |ok |okay |please |can you |could you |would you |i want you to |i need you to )",
        r"^(give me |get me |tell me |show me |find me |fetch me |grab me |bring me )",
        r"^(the |a |an )",
        r"^(as of today[,.]?\s*)",
        r"^(as of \d{1,2}(st|nd|rd|th)?\s+\w+[,.]?\s*\d{0,4}[,.]?\s*)",
        r"(on google|on the web|on the internet|from the internet|from google)$",
        r"(right now|for me|please|asap)$",
    ]
    for pattern in filler_patterns:
        q = re.sub(pattern, "", q, flags=re.IGNORECASE).strip()
    # Second pass: catch prefixes revealed after date/filler stripping
    for pattern in filler_patterns:
        q = re.sub(pattern, "", q, flags=re.IGNORECASE).strip()
    # Remove leading/trailing punctuation
    q = q.strip(".,!?;: ")
    return q if len(q) > 3 else raw_query.strip()

def search_and_read_web(query: str) -> str:
    """
    Called by the Swarm LLM router to silently search the web for an answer using direct HTML scraping.
    Returns up to 5 result snippets with titles and descriptions.
    """
    try:
        import urllib.parse
        clean_query = _clean_search_query(query)
        print(f"\n🌐 [AetherWeb] Silently indexing the web for: '{clean_query}' (raw: '{query[:60]}...')")
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(clean_query)}"
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # DuckDuckGo's result class is 'result__snippet'
        snippets = soup.find_all('a', class_='result__snippet')
        titles = soup.find_all('h2', class_='result__title')
        links = soup.find_all('a', class_='result__url')
        
        results = []
        for i in range(min(5, len(snippets))):
            title = titles[i].get_text(strip=True) if i < len(titles) else "Result"
            desc = snippets[i].get_text(strip=True)
            link = links[i].get_text(strip=True) if i < len(links) else ""
            results.append(f"[{i+1}] {title}\n    {desc}\n    Source: {link}")
            
        if not results:
            return "[AetherWeb] No snippets found on the web for that query."
            
        summary = "\n\n".join(results)
        return f"[AetherWeb Silent Search Results]:\n{summary}\n\n(Synthesize the above information into a concise spoken answer to the user. Cite specific facts from the results)."
        
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
