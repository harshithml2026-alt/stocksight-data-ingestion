"""
Ingest SEC filings from the SEC/ folder into a Pinecone index.

Reads all .htm/.html files, extracts plain text, chunks it, embeds via
OpenAI text-embedding-3-small, and upserts into Pinecone.

Required environment variables:
  OPENAI_API_KEY
  PINECONE_API_KEY
  PINECONE_INDEX_NAME

Usage:
  python3 ingest_to_pinecone.py
  python3 ingest_to_pinecone.py --sec-dir ./SEC --batch-size 100
"""

import os
import re
import time
import hashlib
import argparse
from pathlib import Path

from bs4 import BeautifulSoup
from openai import OpenAI, RateLimitError
from pinecone import Pinecone
from tqdm import tqdm

EMBEDDING_MODEL  = "text-embedding-3-small"

COMPANY_NAMES = {
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "TSLA": "Tesla",
}
CHUNK_SIZE       = 1500   # chars (~375 tokens)
CHUNK_OVERLAP    = 150    # chars
EMBED_SLEEP      = 0.5    # seconds between embedding calls (rate limit buffer)
MAX_RETRIES      = 6      # max retries on rate limit errors


def extract_text(file_path: Path) -> str:
    """Strip HTML tags and collapse whitespace."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of CHUNK_SIZE characters."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Embed texts with exponential backoff on rate limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            time.sleep(EMBED_SLEEP)
            return [item.embedding for item in response.data]
        except RateLimitError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt  # 1, 2, 4, 8, 16, 32 seconds
            print(f"\n  Rate limit hit — retrying in {wait}s...")
            time.sleep(wait)


def file_metadata(file_path: Path) -> dict | None:
    """Extract ticker/year/form_type from SEC/<TICKER>/<year>/<form>/<file> structure."""
    parts = file_path.parts
    try:
        sec_idx = next(i for i, p in enumerate(parts) if p == "SEC")
    except StopIteration:
        return None
    if len(parts) < sec_idx + 5:
        return None
    return {
        "ticker":    parts[sec_idx + 1],
        "year":      parts[sec_idx + 2],
        "form_type": parts[sec_idx + 3],
        "filename":  file_path.name,
    }


def vector_id(file_path: Path, chunk_index: int) -> str:
    """Deterministic ID so reruns overwrite existing vectors."""
    return hashlib.md5(f"{file_path}::{chunk_index}".encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Ingest SEC filings into Pinecone.")
    parser.add_argument("--sec-dir",     default="./SEC", metavar="DIR")
    parser.add_argument("--batch-size",  type=int, default=100,
                        help="Vectors per Pinecone upsert batch (default: 100)")
    parser.add_argument("--embed-batch", type=int, default=50,
                        help="Chunks per OpenAI embedding call (default: 50)")
    parser.add_argument("--max-files",  type=int, default=None,
                        help="Limit number of files processed (useful for testing)")
    args = parser.parse_args()

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.environ["PINECONE_INDEX_NAME"]
    )

    files = sorted(Path(args.sec_dir).rglob("*.htm")) + \
            sorted(Path(args.sec_dir).rglob("*.html"))
    if args.max_files:
        files = files[:args.max_files]
    print(f"Found {len(files)} files in {args.sec_dir}\n")

    pending: list[dict] = []
    total_vectors = 0

    def flush(batch: list[dict]) -> list[dict]:
        nonlocal total_vectors
        index.upsert(vectors=batch)
        total_vectors += len(batch)
        return []

    for file_path in tqdm(files, desc="Ingesting"):
        meta = file_metadata(file_path)
        if not meta:
            continue

        text = extract_text(file_path)
        if not text:
            continue

        chunks = chunk_text(text)
        company = COMPANY_NAMES.get(meta['ticker'], meta['ticker'])
        prefix = f"[{company} ({meta['ticker']}) | {meta['form_type']} | {meta['year']}] "

        for i in range(0, len(chunks), args.embed_batch):
            batch_chunks = chunks[i : i + args.embed_batch]
            # prepend metadata prefix so year/ticker signal is baked into the vector
            prefixed = [prefix + c for c in batch_chunks]
            embeddings = embed(prefixed, openai_client)

            for j, (chunk, emb) in enumerate(zip(batch_chunks, embeddings)):
                pending.append({
                    "id":     vector_id(file_path, i + j),
                    "values": emb,
                    "metadata": {
                        **meta,
                        "chunk_index": i + j,
                        "text": chunk[:1500],  # stored for retrieval context
                    },
                })

            while len(pending) >= args.batch_size:
                pending = flush(pending[:args.batch_size]) + pending[args.batch_size:]

    if pending:
        flush(pending)

    print(f"\nDone! {total_vectors} vectors upserted to '{os.environ['PINECONE_INDEX_NAME']}'.")


if __name__ == "__main__":
    main()
