"""Parallel transcript downloader for Lex Fridman Podcast."""

import concurrent.futures
import json
import os
import re
import sys
import time

import requests

METADATA_API = "https://youtubetranscript.pro/api/youtube/metadata"
TRANSCRIPT_API = "https://youtubetranscript.pro/api/youtube/transcript"
SESSION_ID = "cde9e447-a2b3-4266-9502-0098600315fe"
OUTPUT_DIR = "data/transcripts"
MAX_WORKERS = 10


def extract_video_id(url: str) -> str | None:
    if not url:
        return None
    m = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)", url)
    return m.group(1) if m else None


def download_one(ep: dict) -> tuple[str, bool]:
    url = ep.get("youtube_url") or ep.get("episode_url", "")
    video_id = extract_video_id(url)
    if not video_id:
        return ep.get("title", "?"), False

    title = ep.get("title", f"episode_{video_id}")
    safe_name = re.sub(r"[^\w\s\-]", "", title).strip()[:100]
    filename = os.path.join(OUTPUT_DIR, f"{safe_name}___YouTube.txt")

    if os.path.exists(filename):
        return safe_name, True  # already exists

    try:
        # Register
        r1 = requests.post(METADATA_API, json={"videoID": video_id}, timeout=15)
        if not r1.ok:
            return safe_name, False

        # Fetch
        r2 = requests.get(
            TRANSCRIPT_API,
            params={
                "videoId": video_id,
                "userId": "",
                "sessionId": SESSION_ID,
            },
            timeout=30,
        )
        if not r2.ok:
            return safe_name, False

        data = r2.json()
        if not data.get("success"):
            return safe_name, False

        items = data.get("data", {}).get("response", [])
        if not items:
            return safe_name, False

        text = " ".join(item["text"] for item in items if "text" in item)
        if len(text.strip()) < 100:
            return safe_name, False

        with open(filename, "w", encoding="utf-8") as f:
            f.write(text.strip())

        return safe_name, True

    except Exception:
        return safe_name, False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open("lex_podcast/data/episodes.json") as f:
        episodes = json.load(f)

    print(f"Total episodes: {len(episodes)}")
    print(f"Downloading to {OUTPUT_DIR}/ with {MAX_WORKERS} workers\n")

    done = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")])
    total = len(episodes)
    success = done
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, ep): ep for ep in episodes}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            if i % 25 == 0 or i == total:
                print(f"  [{i}/{total}] OK={success} FAIL={failed}")

    print(f"\nDone: {success} OK, {failed} FAIL ({(success / total) * 100:.0f}%)")


if __name__ == "__main__":
    main()
