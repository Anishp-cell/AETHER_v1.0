"""
AETHER Privacy-Safe Site History Manager
Tracks most-visited base domain names (e.g. 'youtube.com', 'github.com') EXCLUSIVELY
while AETHER is running. Strips all paths, query strings, and personal video links.

Privacy Guarantees:
1. Gated by session_state.is_active() — zero recording when AETHER is closed.
2. Extracts ONLY the base domain name (e.g. 'youtube.com'), never URLs like 'youtube.com/watch?v=xyz'.
3. Automatically ignores browser internals (about:blank, localhost, chrome://, data:).
4. Persisted in .aether/site_history.json with visit frequency and recency timestamps.
"""

import os
import sys
import re
import json
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "Core_agent"))

import session_state

HISTORY_FILE = WORKSPACE_ROOT / ".aether" / "site_history.json"


class SiteHistoryManager:
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or HISTORY_FILE
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _default_schema(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_visits_tracked": 0,
            "sites": {}
        }

    def _load(self) -> Dict[str, Any]:
        if not self.storage_path.exists():
            data = self._default_schema()
            self._save(data)
            return data

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "sites" not in data:
                data["sites"] = {}
            return data
        except Exception:
            return self._default_schema()

    def _save(self, data: Optional[Dict[str, Any]] = None):
        if data is None:
            data = self.data
        data["last_updated"] = datetime.now().isoformat()

        temp_path = self.storage_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.replace(self.storage_path)
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            print(f"[SiteHistoryManager Error] Could not save history file: {e}", file=sys.stderr)

    @staticmethod
    def extract_base_domain(url: str) -> Optional[str]:
        """
        Extracts ONLY the clean base domain name from any URL or string.
        Examples:
            'https://www.youtube.com/watch?v=4xDzrJKXOOY' -> 'youtube.com'
            'http://github.com/Anishp-cell/AETHER/pull/1' -> 'github.com'
            'https://search.yahoo.com/search?p=NVIDIA'     -> 'search.yahoo.com'
            'netflix.com/browse'                          -> 'netflix.com'
            'about:blank'                                 -> None
            'data:text/html,...'                          -> None
        """
        if not url or not isinstance(url, str):
            return None

        clean = url.strip()

        # Handle browser internal schemes
        if any(clean.startswith(prefix) for prefix in [
            "about:", "chrome:", "edge:", "brave:", "data:", "file:", "javascript:", "blob:"
        ]):
            return None

        # Add scheme if missing so urlparse works properly
        if "://" not in clean:
            clean = "https://" + clean

        try:
            parsed = urllib.parse.urlparse(clean)
            host = (parsed.hostname or parsed.netloc).lower().strip()

            # Remove port if present
            host = host.split(":")[0]

            # Remove common 'www.' prefix
            host = re.sub(r'^www\.', '', host)

            # Skip local addresses and invalid hosts
            if not host or host in ("localhost", "127.0.0.1", "0.0.0.0") or "." not in host:
                return None

            return host
        except Exception:
            return None

    def record_visit(self, url: str, force: bool = False) -> Optional[str]:
        """
        Records a visited site.
        Enforces privacy gate: ONLY records when AETHER is active (unless force=True).
        Extracts and stores strictly the base domain.
        """
        # Privacy gate: Only record if AETHER is running
        if not force and not session_state.is_active():
            return None

        base_domain = self.extract_base_domain(url)
        if not base_domain:
            return None

        now_iso = datetime.now().isoformat()
        sites = self.data["sites"]

        if base_domain not in sites:
            sites[base_domain] = {
                "visit_count": 0,
                "first_seen": now_iso,
                "last_seen": now_iso
            }

        sites[base_domain]["visit_count"] += 1
        sites[base_domain]["last_seen"] = now_iso
        self.data["total_visits_tracked"] += 1

        self._save()
        return base_domain

    def record_urls(self, urls: List[str], force: bool = False) -> List[str]:
        """
        Records a batch of visited URLs, returning the unique base domains recorded.
        """
        recorded = []
        for u in urls:
            dom = self.record_visit(u, force=force)
            if dom and dom not in recorded:
                recorded.append(dom)
        return recorded

    def get_top_sites(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns the most frequently visited domains sorted by visit count."""
        sites = self.data.get("sites", {})
        sorted_sites = sorted(
            [{"domain": d, **meta} for d, meta in sites.items()],
            key=lambda x: x.get("visit_count", 0),
            reverse=True
        )
        return sorted_sites[:limit]

    def get_most_visited_domains(self, limit: int = 10) -> List[str]:
        """Returns just the domain string names of the top sites."""
        return [s["domain"] for s in self.get_top_sites(limit=limit)]

    def get_sites_as_context(self, top_n: int = 5) -> str:
        """
        Returns a concise context string suitable for injection into system prompts:
        'User frequently visits: youtube.com (47 times), github.com (12 times), reddit.com (8 times)'
        """
        top = self.get_top_sites(limit=top_n)
        if not top:
            return ""
        items = [f"{s['domain']} ({s['visit_count']} times)" for s in top]
        return f"User frequently visits: {', '.join(items)}"

    def format_ascii_report(self) -> str:
        """Formats an ASCII dashboard table of visited domains."""
        top = self.get_top_sites(limit=15)
        active_status = "🟢 ACTIVE (Tracking Enabled)" if session_state.is_active() else "⚪ INACTIVE (Tracking Paused)"

        lines = [
            "╔══════════════════════════════════════════════════════════════════════╗",
            "║             AETHER PRIVACY-SAFE BASE DOMAIN TRACKER                  ║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            f"║  Session Gate:      {active_status:<48}║",
            f"║  Total Visited:     {self.data.get('total_visits_tracked', 0):<5} total visits across {len(self.data.get('sites', {}))} unique base domains{'':<4}║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  Rank   Base Domain Name               Visits   Last Seen             ║",
            "╟──────────────────────────────────────────────────────────────────────╢",
        ]

        if not top:
            lines.append("║  No domain visits recorded yet while AETHER is active.               ║")
        else:
            for idx, item in enumerate(top, 1):
                dom = item["domain"][:28]
                count = item["visit_count"]
                last_seen_raw = item.get("last_seen", "")
                last_seen = last_seen_raw.split("T")[0] if "T" in last_seen_raw else last_seen_raw[:10]
                lines.append(f"║  #{idx:<3}  {dom:<30} {count:<8} {last_seen:<18} ║")

        lines.append("╚══════════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


if __name__ == "__main__":
    manager = SiteHistoryManager()
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run self-test
        print("[Self-Test] Activating session and recording test URLs...")
        session_state.set_active(True)
        test_urls = [
            "https://www.youtube.com/watch?v=4xDzrJKXOOY",
            "https://github.com/Anishp-cell/AETHER/issues/12",
            "https://search.yahoo.com/search?p=NVIDIA+Cosmos",
            "https://www.youtube.com/watch?v=another_video",
            "about:blank",  # should be skipped
        ]
        manager.record_urls(test_urls)
        print("Recorded. Now reporting:\n")

    print(manager.format_ascii_report())
