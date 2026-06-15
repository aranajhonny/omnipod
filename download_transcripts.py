"""Download all transcripts sequentially (avoids rate limiting)."""

import json
import os
import re
import time

import requests

METADATA_API = "https://youtubetranscript.pro/api/youtube/metadata"
TRANSCRIPT_API = "https://youtubetranscript.pro/api/youtube/transcript"
SESSION_ID = "cde9e447-a2b3-4266-9502-0098600315fe"
OUTPUT_DIR = "data/transcripts"


def extract_video_id(url: str) -> str | None:
    if not url:
        return None
    m = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)", url)
    return m.group(1) if m else None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open("lex_podcast/data/episodes.json") as f:
        episodes = json.load(f)

    existing = set(os.listdir(OUTPUT_DIR))
    success = 0
    fail = 0
    skip = 0

    for i, ep in enumerate(episodes, 1):
        url = ep.get("youtube_url") or ep.get("episode_url", "")
        video_id = extract_video_id(url)
        if not video_id:
            fail += 1
            continue

        title = ep.get("title", f"ep_{video_id}")
        safe = re.sub(r"[^\w\s\-]", "", title).strip()[:100]
        fname = f"{safe}___YouTube.txt"

        if fname in existing:
            skip += 1
            continue

        time.sleep(1.0)

        try:
            r1 = requests.post(METADATA_API, json={"videoID": video_id}, timeout=15)
            if not r1.ok:
                fail += 1
                print(f"  [{i}/{len(episodes)}] FAIL metadata {video_id}")
                continue

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
                fail += 1
                print(f"  [{i}/{len(episodes)}] FAIL transcript {video_id}")
                continue

            data = r2.json()
            if not data.get("success"):
                fail += 1
                print(f"  [{i}/{len(episodes)}] FAIL no success {video_id}")
                continue

            items = data.get("data", {}).get("response", [])
            if not items:
                fail += 1
                continue

            text = " ".join(item["text"] for item in items if "text" in item)
            if len(text.strip()) < 100:
                fail += 1
                continue

            with open(os.path.join(OUTPUT_DIR, fname), "w", encoding="utf-8") as f:
                f.write(text.strip())

            success += 1
            if success % 25 == 0:
                print(f"  [{i}/{len(episodes)}] OK={success} FAIL={fail} SKIP={skip}")

        except Exception as e:
            fail += 1
            print(f"  [{i}/{len(episodes)}] ERROR {video_id}: {e}")

    print(f"\n✅ Done: {success} OK, {fail} FAIL, {skip} SKIP / {len(episodes)} total")


if __name__ == "__main__":
    main()
