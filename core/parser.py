"""YouTube filename parser.

Parses chaotic filenames exported from YouTube into structured metadata:

    Input (full):  "Andrew Huberman - Neuroscientist___YouTube.txt"
    Output: {"guest": "Andrew Huberman", "title": "Neuroscientist", "type": "full"}

    Input (clip):  "Andrej Karpathy_ Tesla AI_ Self-Driving_ Optimus_ Aliens_ and AGI Andrej Karpathy - AI Researcher___YouTube.txt"
    Output: {"guest": "Andrej Karpathy", "title": "Tesla AI: Self-Driving, Optimus, Aliens, and AGI", "type": "clip"}
"""

import re


def parse_youtube_title(filename: str) -> dict:
    """Parse a YouTube transcript filename into structured metadata.

    Returns a dict with keys: guest, title, type, source_file.
    """
    source_file = filename

    # 1. Strip _YouTube.txt suffix (and any trailing underscores before it)
    name = filename
    name = re.sub(r"_+YouTube\.txt$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_+YouTube$", "", name, flags=re.IGNORECASE)

    # 2. Split on " - " (space-hyphen-space)
    if " - " not in name:
        # No structure to parse
        cleaned = name.replace("_", " ").strip()
        return {
            "guest": cleaned,
            "title": cleaned,
            "type": "full",
            "source_file": source_file,
        }

    parts = name.split(" - ", 1)
    before_hyphen = parts[0].strip()
    after_hyphen = parts[1].strip() if len(parts) > 1 else ""

    # Clean trailing underscores from after_hyphen
    after_hyphen = after_hyphen.rstrip("_").strip()

    # 3. Detect if it's a clip (guest name repeats)
    cleaned_before = before_hyphen.replace("_", " ").strip()
    words = cleaned_before.split()

    # Try to find a repeating guest name at the beginning
    best_guest = ""
    best_guest_len = 0

    for i in range(1, min(len(words) + 1, 8)):
        candidate = " ".join(words[:i])
        rest = " ".join(words[i:])
        if candidate.lower() in rest.lower():
            if len(candidate) > best_guest_len:
                best_guest = candidate
                best_guest_len = len(candidate)

    if best_guest:
        # ── Clip format ──
        guest = best_guest
        # Title is everything between the first and second occurrence of guest
        title_text = cleaned_before[len(guest) :].strip()
        # Remove leading punctuation/underscores
        title_text = re.sub(r"^[,:\s_]+", "", title_text).strip()
        # Clean up
        title_text = title_text.replace("_", " ").strip()
        title_text = re.sub(r"\s+", " ", title_text)
        doc_type = "clip"

        # Clean trailing repeated guest name from title
        # (e.g., "Topic ... Guest" → just "Topic ...")
        lower_title = title_text.lower()
        lower_guest = guest.lower()
        if lower_title.endswith(lower_guest):
            title_text = title_text[: -len(guest)].strip()
        elif lower_title.endswith(lower_guest.rstrip(".")):
            title_text = title_text[: -(len(guest.rstrip(".")))].strip()

        title_text = re.sub(r"\s+", " ", title_text).strip()
    else:
        # ── Full podcast format ──
        # Guest is everything before " - "
        guest = cleaned_before
        # Title is everything after " - " (the category/description)
        title_text = after_hyphen if after_hyphen else guest
        doc_type = "full"

    # Clean up: replace underscores in title with proper punctuation
    # Replace _ that look like separators with ", "
    # Pattern: word_word → word: word (if it looks like a topic separator)
    title_text = re.sub(r"\s*_+\s*", ", ", title_text)
    title_text = re.sub(r"\s+", " ", title_text).strip()

    # Clean guest
    guest = guest.rstrip("_,;:").strip()

    return {
        "guest": guest,
        "title": title_text,
        "type": doc_type,
        "source_file": source_file,
    }


def clean_text(text: str) -> str:
    """Basic cleaning of transcript text."""
    # Remove multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Remove timestamp-like patterns [00:00:00] or (00:00)
    text = re.sub(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?", "", text)
    return text.strip()
