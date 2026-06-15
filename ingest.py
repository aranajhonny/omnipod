#!/usr/bin/env python3
"""Ingest: dense embeddings, file by file (low memory)."""

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient, models

from core.config import COLLECTION_NAME, EMBED_MODEL, QDRANT_URL
from core.parser import clean_text, parse_youtube_title
from core.vectorstore import get_splitter


def ingest(rebuild: bool = False):
    start = time.time()
    client = QdrantClient(url=QDRANT_URL)

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("🗑️  Deleted")
        except Exception:
            pass

    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384, distance=models.Distance.COSINE
            ),
        )
        print("✅ Collection created (384d)")
    else:
        print("ℹ️  Collection exists")

    files = sorted(
        glob.glob("data/transcripts/*_YouTube.txt")
        or glob.glob("data/transcripts/*.txt")
    )
    print(f"📂 {len(files)} files\n")

    # Load model once
    print("🧠 Loading model...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="mps")
    print("   ✅ Model loaded\n")

    splitter = get_splitter()
    total_chunks = 0
    errors = 0

    for idx, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
        except Exception:
            errors += 1
            continue
        if not raw.strip():
            continue

        meta = parse_youtube_title(fname)
        cleaned = clean_text(raw)
        chunks = splitter.split_text(cleaned)
        if not chunks:
            continue

        # Embed this file
        t0 = time.time()
        vecs = model.encode(chunks, batch_size=256, show_progress_bar=False)
        embed_time = time.time() - t0

        # Upload
        points = []
        for ci, (chunk, vec) in enumerate(zip(chunks, vecs)):
            pid = (
                hash(f"{meta['source_file']}_{total_chunks + ci}") & 0x7FFFFFFFFFFFFFFF
            )
            points.append(
                models.PointStruct(
                    id=pid,
                    vector=vec.tolist(),
                    payload={
                        "guest": meta["guest"],
                        "title": meta["title"],
                        "type": meta["type"],
                        "source_file": meta["source_file"],
                        "text": chunk,
                    },
                )
            )

        t0 = time.time()
        for i in range(0, len(points), 256):
            client.upload_points(COLLECTION_NAME, points[i : i + 256], wait=True)
        upload_time = time.time() - t0

        total_chunks += len(chunks)
        print(
            f"  [{idx}/{len(files)}] {fname[:50]:50s} {len(chunks):4d} chunks | "
            f"embed {embed_time:.1f}s | upload {upload_time:.1f}s | total {total_chunks:,}"
        )

    print(f"\n{'=' * 50}")
    print(
        f"✅ COMPLETE: {total_chunks:,} chunks from {len(files)} files in {time.time() - start:.1f}s"
    )
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    ingest(rebuild=args.rebuild)
