"""
Fetches transcripts for Lex Fridman Podcast episodes.
- From lexfridman.com (official transcript pages)
- From YouTube (via youtube-transcript-api, free, no API key needed)
"""

import html
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup
from config import REQUEST_TIMEOUT
from scraper import Episode

_TRANSCRIPT_SESSION = "cde9e447-a2b3-4266-9502-0098600315fe"
_METADATA_API = "https://youtubetranscript.pro/api/youtube/metadata"
_TRANSCRIPT_API = "https://youtubetranscript.pro/api/youtube/transcript"


def fetch_site_transcript(episode: Episode) -> Optional[str]:
    """
    Fetch transcript from lexfridman.com transcript page.
    The transcript pages have the full text with timestamps in <p> tags.
    """
    if not episode.transcript_url:
        return None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(
            episode.transcript_url, headers=headers, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # The transcript is inside the main content area
    content = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_="entry-content")
    )

    if not content:
        content = soup.body

    # Remove unwanted elements
    for tag in content.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Get text
    text = content.get_text(separator="\n")

    # Clean up the text
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        line = html.unescape(line)
        if len(line) < 3:
            continue
        if any(
            skip in line.lower()
            for skip in [
                "skip to content",
                "lex fridman",
                "research scientist",
                "podcast",
                "click to play",
                "subscribe",
                "support this podcast",
                "thank you",
                "proudly powered",
                "share this",
                "comments",
                "tags:",
                "about lex",
                "view all posts",
            ]
        ):
            continue
        lines.append(line)

    full_text = "\n".join(lines)

    markers = [
        "episode highlight",
        "table of contents",
        "introduction",
        "00:00",
        "the following is",
    ]
    start_pos = 0
    for marker in markers:
        pos = full_text.lower().find(marker)
        if pos != -1:
            start_pos = max(start_pos, pos)

    if start_pos > 0:
        full_text = full_text[start_pos:]

    if len(full_text) < 500:
        return None

    return full_text.strip()


def extract_video_id(youtube_url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    if not youtube_url:
        return None

    patterns = [
        r"(?:youtube\.com/watch\?v=)([\w-]+)",
        r"(?:youtu\.be/)([\w-]+)",
        r"(?:youtube\.com/embed/)([\w-]+)",
        r"(?:youtube\.com/shorts/)([\w-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)

    return None


def fetch_youtube_transcript(episode: Episode) -> Optional[str]:
    """
    Fetch transcript from YouTube via youtubetranscript.pro API.
    Calls metadata endpoint first to register the video, then fetches transcript.
    """
    urls_to_try = []
    if episode.youtube_url:
        urls_to_try.append(episode.youtube_url)
    if episode.episode_url and "youtube" in episode.episode_url.lower():
        urls_to_try.append(episode.episode_url)

    for url in urls_to_try:
        video_id = extract_video_id(url)
        if not video_id:
            continue

        try:
            # Step 1: POST metadata to register the video with the session
            metadata_resp = requests.post(
                _METADATA_API,
                json={"videoID": video_id},
                timeout=15,
            )
            if not metadata_resp.ok:
                continue

            # Step 2: GET transcript
            transcript_resp = requests.get(
                _TRANSCRIPT_API,
                params={
                    "videoId": video_id,
                    "userId": "",
                    "sessionId": _TRANSCRIPT_SESSION,
                },
                timeout=30,
            )

            if not transcript_resp.ok:
                continue

            data = transcript_resp.json()
            if not data.get("success"):
                continue

            items = data.get("data", {}).get("response", [])
            if not items:
                continue

            text = " ".join(item["text"] for item in items if "text" in item)
            if len(text.strip()) > 100:
                return text.strip()

        except Exception:
            continue

    return None


def fetch_transcript(episode: Episode) -> Optional[str]:
    """
    Fetch transcript for an episode.
    Prefers site transcript, falls back to YouTube.
    """
    # Try site transcript first
    if episode.has_transcript_on_site:
        transcript = fetch_site_transcript(episode)
        if transcript:
            return transcript

    # Fallback to YouTube
    transcript = fetch_youtube_transcript(episode)
    return transcript
