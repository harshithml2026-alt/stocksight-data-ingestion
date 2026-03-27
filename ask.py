"""
Ask questions about SEC filings using RAG (Pinecone + OpenAI).

Usage:
  export $(cat .env | xargs)
  python3 ask.py "What was NVIDIA's revenue in 2024?"
  python3 ask.py "How is Microsoft investing in AI?" --ticker MSFT
  python3 ask.py "What are the main risk factors?" --ticker NVDA --year 2024
"""

import os
import argparse
from openai import OpenAI
from pinecone import Pinecone

EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL    = "gpt-4o-mini"
TOP_K           = 6  # number of chunks to retrieve


def retrieve(query: str, index, client: OpenAI, ticker: str = None, year: str = None) -> list[dict]:
    """Embed query and fetch top-k matching chunks from Pinecone."""
    embedding = client.embeddings.create(
        model=EMBEDDING_MODEL, input=[query]
    ).data[0].embedding

    filter_ = {}
    if ticker:
        filter_["ticker"] = ticker
    if year:
        filter_["year"] = str(year)

    return index.query(
        vector=embedding,
        top_k=TOP_K,
        include_metadata=True,
        filter=filter_ or None,
    )["matches"]


def answer(question: str, chunks: list[dict], client: OpenAI) -> str:
    """Send retrieved chunks + question to GPT and return the answer."""
    context = "\n\n---\n\n".join([
        f"[{m['metadata'].get('ticker')} | {m['metadata'].get('form_type')} | "
        f"{m['metadata'].get('year')} | score: {m['score']:.3f}]\n"
        f"{m['metadata'].get('text', '')}"
        for m in chunks
    ])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst assistant. Answer questions based strictly "
                "on the provided SEC filing excerpts. Cite the form type and year when "
                "referencing specific data. If the answer is not in the context, say so."
            ),
        },
        {
            "role": "user",
            "content": f"Context from SEC filings:\n\n{context}\n\n---\n\nQuestion: {question}",
        },
    ]

    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Ask questions about SEC filings.")
    parser.add_argument("question", help="Your question (wrap in quotes)")
    parser.add_argument("--ticker", default=None, help="Filter by ticker (e.g. NVDA, MSFT)")
    parser.add_argument("--year",   default=None, help="Filter by year (e.g. 2024)")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    index  = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(
        os.environ["PINECONE_INDEX_NAME"]
    )

    print(f"\nQuestion: {args.question}")
    if args.ticker:
        print(f"Filter  : {args.ticker}", end="")
        if args.year:
            print(f" | {args.year}", end="")
        print()
    print()

    chunks = retrieve(args.question, index, client, args.ticker, args.year)

    if not chunks:
        print("No relevant chunks found. Try a different query or remove filters.")
        return

    print(f"Retrieved {len(chunks)} chunks — top sources:")
    for c in chunks:
        m = c["metadata"]
        print(f"  {m.get('ticker')} | {m.get('form_type')} | {m.get('year')} | score: {c['score']:.3f}")
    print()

    print("Answer:")
    print("-" * 60)
    print(answer(args.question, chunks, client))
    print("-" * 60)


if __name__ == "__main__":
    main()
