"""
AETHER Fast Web Driver — Channel 1 Deterministic Playwright Engine
Inspired by "The Tireless Hand" Hackathon Dual-Channel Architecture

Features:
- Sub-second deterministic execution for high-frequency web tasks
- Real-time visual cyber glow element highlighting:
    * #00ffcc Neon Cyan: Target focus / search inputs
    * #ffb300 Amber: Scanning / DOM inspection
    * #00ff88 Emerald: Confirmed action / successful result
    * #b026ff Purple: Audio / video media playback
- Zero LLM token cost for standard queries (YouTube, Web Search, Direct URLs)
- Seamless automatic fallback to Channel 2 (Autonomous browser-use) when tasks require multi-step reasoning
"""

import sys
import os
import time
import json
import asyncio
import urllib.parse
from pathlib import Path
from typing import Optional, Dict, Any, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Visual Glow Palette from The Tireless Hand
GLOW_CYAN = "#00ffcc"     # Target selection / search inputs
GLOW_AMBER = "#ffb300"    # In-progress processing / inspection
GLOW_EMERALD = "#00ff88"  # Success action / clicked result
GLOW_PURPLE = "#b026ff"   # Media play / video launch


async def highlight_element(element, color: str = GLOW_CYAN, duration_s: float = 0.4):
    """Injects cybernetic glowing outline and shadow directly into live DOM element."""
    try:
        await element.evaluate(f"""(el) => {{
            el.style.outline = '3px solid {color}';
            el.style.boxShadow = '0 0 20px {color}';
            el.style.transition = 'all 0.25s ease';
        }}""")
        if duration_s > 0:
            await asyncio.sleep(duration_s)
    except Exception:
        pass


async def clear_highlight(element):
    """Removes injected CSS glow."""
    try:
        await element.evaluate("""(el) => {
            el.style.outline = '';
            el.style.boxShadow = '';
        }""")
    except Exception:
        pass


class FastWebDriver:
    """Channel 1 Deterministic Playwright Automation."""

    def __init__(self, headless: bool = False):
        self.headless = headless

    async def _get_browser_context(self, playwright):
        """Launches Chromium with standard WebGL and anti-detection flags."""
        browser = await playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--start-maximized"
            ]
        )
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900} if self.headless else None,
            no_viewport=not self.headless,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        return browser, context, page

    async def play_youtube(self, query: str) -> Dict[str, Any]:
        """Fast-paths playing a video or song directly on YouTube with visual feedback."""
        from playwright.async_api import async_playwright
        print(f"\n⚡ [Channel 1 FastWeb] Launching YouTube deterministic flow: '{query}'")

        async with async_playwright() as p:
            browser, context, page = await self._get_browser_context(p)
            try:
                encoded = urllib.parse.quote_plus(query)
                await page.goto(f"https://www.youtube.com/results?search_query={encoded}", timeout=15000)
                await page.wait_for_selector("ytd-video-renderer, ytd-rich-item-renderer", timeout=8000)

                # Find first video title
                first_video = page.locator("ytd-video-renderer #video-title").first
                await first_video.wait_for(state="visible", timeout=5000)

                video_title = await first_video.inner_text()
                print(f"✨ [Visual Glow] Highlighting target video: '{video_title}'")
                await highlight_element(first_video, color=GLOW_PURPLE, duration_s=0.6)

                # Click to play
                await first_video.click()
                print(f"▶️ [Channel 1 FastWeb] Playback started: {video_title}")

                if not self.headless:
                    # Give it a moment to ensure video player starts streaming
                    await asyncio.sleep(2.0)
                    return {
                        "status": "success",
                        "title": video_title,
                        "url": page.url,
                        "message": f"Playing '{video_title}' on YouTube."
                    }

                await browser.close()
                return {
                    "status": "success",
                    "title": video_title,
                    "url": page.url,
                    "message": f"Found and played: '{video_title}'."
                }
            except Exception as e:
                try:
                    await browser.close()
                except Exception:
                    pass
                return {"status": "fallback_needed", "error": str(e)}

    async def search_web(self, query: str) -> Dict[str, Any]:
        """Fast-paths web search with cyber glow and instant result extraction."""
        from playwright.async_api import async_playwright
        print(f"\n⚡ [Channel 1 FastWeb] Launching deterministic web search: '{query}'")

        async with async_playwright() as p:
            browser, context, page = await self._get_browser_context(p)
            try:
                encoded = urllib.parse.quote_plus(query)
                search_url = f"https://search.yahoo.com/search?p={encoded}"
                await page.goto(search_url, timeout=12000)

                # Highlight search input with Cyan Glow
                search_box = page.locator("input[name='p'], #yschsp").first
                if await search_box.count() > 0:
                    await highlight_element(search_box, color=GLOW_CYAN, duration_s=0.3)

                # Wait for search results
                await page.wait_for_selector(".algo", timeout=8000)
                algos = page.locator(".algo")
                count = await algos.count()

                extracted = []
                for i in range(min(count, 4)):
                    item = algos.nth(i)
                    if i == 0:
                        await highlight_element(item, color=GLOW_EMERALD, duration_s=0.4)
                    else:
                        await highlight_element(item, color=GLOW_AMBER, duration_s=0.2)

                    title_loc = item.locator("h3 a, a.d-ib, a").first
                    raw_title = await title_loc.inner_text() if await title_loc.count() > 0 else "Result"
                    link = await title_loc.get_attribute("href") if await title_loc.count() > 0 else search_url
                    snippet_loc = item.locator(".compText").first
                    snippet = await snippet_loc.inner_text() if await snippet_loc.count() > 0 else ""

                    # Clean title lines
                    title_lines = [l.strip() for l in raw_title.split("\n") if l.strip()]
                    clean_title = title_lines[-1] if title_lines else raw_title

                    if clean_title and clean_title != "Result":
                        extracted.append({
                            "title": clean_title,
                            "snippet": snippet.strip()[:240],
                            "url": link
                        })

                if not extracted:
                    await browser.close()
                    return {"status": "fallback_needed", "error": "No structured results parsed"}

                await asyncio.sleep(0.5)
                await browser.close()

                formatted_lines = [f"{i+1}. {r['title']}\n   {r['snippet']}\n   Source: {r['url']}" for i, r in enumerate(extracted)]
                summary = "\n\n".join(formatted_lines)

                return {
                    "status": "success",
                    "results": extracted,
                    "url": search_url,
                    "message": f"Web search results for '{query}':\n\n{summary}"
                }
            except Exception as e:
                try:
                    await browser.close()
                except Exception:
                    pass
                return {"status": "fallback_needed", "error": str(e)}

    async def open_url(self, url: str) -> Dict[str, Any]:
        """Fast-paths directly opening a web URL with visual inspection and metadata extraction."""
        from playwright.async_api import async_playwright
        print(f"\n⚡ [Channel 1 FastWeb] Opening destination URL: {url}")

        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        async with async_playwright() as p:
            browser, context, page = await self._get_browser_context(p)
            try:
                await page.goto(url, timeout=15000, wait_until="domcontentloaded")
                title = await page.title()

                # Highlight main document area
                main_elem = page.locator("main, article, #content, body").first
                if await main_elem.count() > 0:
                    await highlight_element(main_elem, color=GLOW_CYAN, duration_s=0.5)

                # Extract first meaningful text
                paragraphs = page.locator("p, h1, h2")
                p_count = await paragraphs.count()
                snippets = []
                for i in range(min(p_count, 5)):
                    txt = await paragraphs.nth(i).inner_text()
                    txt = txt.strip()
                    if txt and len(txt) > 20:
                        snippets.append(txt)

                snippet_summary = " ".join(snippets)[:400]

                if not self.headless:
                    await asyncio.sleep(2.0)
                    return {
                        "status": "success",
                        "title": title,
                        "url": page.url,
                        "message": f"Opened '{title}' ({page.url}). Preview: {snippet_summary}"
                    }

                await browser.close()
                return {
                    "status": "success",
                    "title": title,
                    "url": page.url,
                    "message": f"Successfully loaded '{title}'. Preview: {snippet_summary}"
                }
            except Exception as e:
                try:
                    await browser.close()
                except Exception:
                    pass
                return {"status": "fallback_needed", "error": str(e)}


def run_fast_web_task(task_type: str, query: str, headless: bool = False) -> Dict[str, Any]:
    """Synchronous entry point for Channel 1 execution."""
    driver = FastWebDriver(headless=headless)
    if task_type == "youtube":
        res = asyncio.run(driver.play_youtube(query))
    elif task_type in ("search", "duckduckgo", "web_search", "google", "yahoo"):
        res = asyncio.run(driver.search_web(query))
    elif task_type in ("open_url", "url", "navigate"):
        res = asyncio.run(driver.open_url(query))
    else:
        res = {"status": "fallback_needed", "error": f"Unknown task_type '{task_type}'"}

    if res.get("status") == "success" and res.get("url"):
        try:
            from site_history_manager import SiteHistoryManager
            SiteHistoryManager().record_visit(res["url"])
        except Exception:
            pass

    return res
