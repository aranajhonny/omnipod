#!/usr/bin/env python3
"""Download all Lex Fridman Podcast transcripts. Single file, no deps beyond requests + bs4."""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class Episode:
    title: str = ""
    guest: str = ""
    youtube_url: Optional[str] = None
    episode_url: Optional[str] = None
    transcript_url: Optional[str] = None
    has_transcript_on_site: bool = False


def fetch_site(ep: Episode) -> Optional[str]:
    """Scrape transcript from lexfridman.com."""
    if not ep.transcript_url:
        return None
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(ep.transcript_url, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    content = soup.find("article") or soup.find("main") or soup.body
    for t in content.find_all(["script", "style", "nav", "header", "footer"]):
        t.decompose()
    text = "\n".join(
        line.strip()
        for line in content.get_text(separator="\n").split("\n")
        if len(line.strip()) >= 3
    )
    for m in [
        "episode highlight",
        "table of contents",
        "introduction",
        "00:00",
        "the following is",
    ]:
        p = text.lower().find(m)
        if p >= 0:
            text = text[p:]
            break
    return text.strip() if len(text) >= 500 else None


def fetch_youtube(url: str) -> Optional[str]:
    """Fetch YouTube transcript via youtubetranscript.pro (free, no key needed)."""
    m = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)", url or "")
    if not m:
        return None
    vid = m.group(1)
    try:
        r1 = requests.post(
            "https://youtubetranscript.pro/api/youtube/metadata",
            json={"videoID": vid},
            timeout=15,
        )
        if not r1.ok:
            return None
        r2 = requests.get(
            "https://youtubetranscript.pro/api/youtube/transcript",
            params={
                "videoId": vid,
                "userId": "",
                "sessionId": "cde9e447-a2b3-4266-9502-0098600315fe",
            },
            timeout=30,
        )
        if not r2.ok:
            return None
        data = r2.json()
        if not data.get("success"):
            return None
        items = data.get("data", {}).get("response", [])
        if not items:
            return None
        text = " ".join(i["text"] for i in items if "text" in i)
        return text.strip() if len(text.strip()) > 100 else None
    except Exception:
        return None


OUT = "data/transcripts"
os.makedirs(OUT, exist_ok=True)

with open("lex_podcast/data/episodes.json") as f:
    episodes = json.load(f)

ok = len([f for f in os.listdir(OUT) if f.endswith(".txt")])
fail = 0
skip = 0
total = len(episodes)
print(f"Total: {total}, already have: {ok}\n")

for i, d in enumerate(episodes, 1):
    title = d.get("title", f"ep_{i}")
    safe = re.sub(r"[^\w\s\-]", "", title).strip()[:100]
    fpath = os.path.join(OUT, f"{safe}___YouTube.txt")
    if os.path.exists(fpath):
        skip += 1
        continue

    ep = Episode(
        title=d["title"],
        guest=d.get("guest", ""),
        youtube_url=d.get("youtube_url"),
        episode_url=d.get("episode_url"),
        transcript_url=d.get("transcript_url"),
        has_transcript_on_site=d.get("has_transcript_on_site", False),
    )

    text = fetch_site(ep) or fetch_youtube(ep.youtube_url or "")
    if text:
        with open(fpath, "w") as f:
            f.write(text)
        ok += 1
        print(f"  [{i}/{total}] ✅ {safe[:50]} ({len(text)} chars)")
    else:
        fail += 1
        if i % 50 == 0:
            print(f"  [{i}/{total}] — {ok} OK, {fail} FAIL, {skip} SKIP")

    time.sleep(0.5)

print(f"\n✅ Done: {ok} OK, {fail} FAIL, {skip} SKIP / {total}")
