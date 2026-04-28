import requests
import json
import os

def startup_briefing() -> str:
    """
    Runs automatically when AETHER starts. Compiles a spoken briefing from:
    - Current weather (via wttr.in, no API key needed)
    - Top 3 Hacker News headlines
    - Current Bitcoin price
    - Any stored reminders from user_profile.json
    Returns a single clean string ready for TTS.
    """
    sections = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # (Bitcoin Price section removed as requested)

    # 2. Weather (wttr.in — returns JSON, no API key)
    try:
        city = "Pune"  # Change this to your city
        r = requests.get(f"https://wttr.in/{city}?format=j1", headers=headers, timeout=6)
        if r.status_code == 200:
            data = r.json()
            current = data["current_condition"][0]
            temp_c = current["temp_C"]
            desc = current["weatherDesc"][0]["value"]
            feels = current["FeelsLikeC"]
            sections.append(f"Weather in {city}: {desc}, {temp_c}°C, feels like {feels}°C.")
    except Exception:
        pass

    # 3. AI News from Hacker News
    try:
        top_ids = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=6).json()[:50]
        headlines = []
        ai_keywords = ["ai", "llm", "gpt", "openai", "machine learning", "deep learning", "anthropic", "claude", "gemini"]
        
        for story_id in top_ids:
            story = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json", timeout=4).json()
            if not story:
                continue
            title = story.get("title", "")
            title_lower = title.lower()
            
            # Check if any AI keyword is in the title
            if any(kw in title_lower for kw in ai_keywords):
                headlines.append(title)
            
            if len(headlines) >= 5: # Get top 5 AI news
                break
                
        if headlines:
            hn_str = " | ".join(headlines)
            sections.append(f"Top AI News today: {hn_str}.")
        else:
            sections.append("There are no major AI news trending on Hacker News right now.")
    except Exception:
        pass

    # 4. Stored Reminders
    try:
        profile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory", "user_profile.json")
        with open(profile_path, "r") as f:
            profile = json.load(f)
        reminders = profile.get("reminders", [])
        if reminders:
            sections.append(f"You have {len(reminders)} reminder(s): " + "; ".join(reminders) + ".")
    except Exception:
        pass

    if not sections:
        return "Good day, sir. All systems are online and I'm ready to assist."

    intro = "Here's your briefing, sir. "
    return intro + " ".join(sections)
