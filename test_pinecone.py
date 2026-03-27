"""
Quick test to verify Pinecone ingestion worked.

Usage:
  export $(cat .env | xargs) && python3 test_pinecone.py
  python3 test_pinecone.py --query "NVIDIA revenue growth 2024"
  python3 test_pinecone.py --query "Microsoft cloud revenue" --ticker MSFT --top-k 5
"""

import os
import argparse
from openai import OpenAI
from pinecone import Pinecone

EMBEDDING_MODEL = "text-embedding-3-small"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",  default="quarterly revenue and earnings", metavar="TEXT")
    parser.add_argument("--ticker", default=None, help="Filter by ticker (e.g. NVDA, MSFT)")
    parser.add_argument("--top-k",  type=int, default=3)
    args = parser.parse_args()

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.environ["PINECONE_INDEX_NAME"]
    )

    # -- Index stats ----------------------------------------------------------
    stats = index.describe_index_stats()
    print(f"Index stats:")
    print(f"  Total vectors : {stats['total_vector_count']}")
    print(f"  Dimension     : {stats['dimension']}")
    print()

    # -- Semantic search ------------------------------------------------------
    print(f"Query : \"{args.query}\"")
    if args.ticker:
        print(f"Filter: ticker = {args.ticker}")
    print()

    embedding = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=[args.query]
    ).data[0].embedding

    filter_ = {"ticker": args.ticker} if args.ticker else None
    results = index.query(
        vector=embedding,
        top_k=args.top_k,
        include_metadata=True,
        filter=filter_,
    )

    for i, match in enumerate(results["matches"], 1):
        m = match["metadata"]
        print(f"Result {i} — score: {match['score']:.4f}")
        print(f"  Ticker    : {m.get('ticker')} | {m.get('form_type')} | {m.get('year')}")
        print(f"  File      : {m.get('filename')}")
        print(f"  Excerpt   : {m.get('text', '')[:200]}...")
        print()


if __name__ == "__main__":
    main()
