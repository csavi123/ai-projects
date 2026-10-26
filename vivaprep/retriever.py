from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from vivaprep.chunker import Chunk, read_chunks
from vivaprep.config import SIMILARITY_THRESHOLD, TOP_K
from vivaprep.embedder import embed_query, read_embeddings

# SEARCH stage: embed the question with the same model as the chunks, compare it
# with every chunk vector (cosine similarity), and return the k closest chunks.
# search() is the one function a vector DB (Chroma, Pinecone) would replace later.


NOT_COVERED = "Not covered in the Notes"  # reply when a question isn't answered by the notes


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity: higher = closer in meaning


_index = None  # (chunks in the same order as the vector rows, vectors)


def _load_index() -> tuple[list[Chunk], np.ndarray]:
    """Read the stored vectors and chunks once, and check they belong together."""
    global _index
    if _index is None:
        ids, vectors = read_embeddings()
        chunks = {chunk.id: chunk for chunk in read_chunks()}
        if set(ids) != set(chunks):
            raise SystemExit("Chunks changed since they were embedded. Run: python -m vivaprep.embedder")
        _index = ([chunks[chunk_id] for chunk_id in ids], vectors)
    return _index


def search(question: str, k: int = TOP_K) -> list[Hit]:
    chunks, vectors = _load_index()
    # Dot product per chunk: multiply matching numbers, add them up. Equals cosine
    # similarity because all vectors have length 1. (Written out instead of
    # `vectors @ query` to avoid false matmul warnings from numpy 2.0 on macOS.)
    scores = (vectors * embed_query(question)).sum(axis=1)
    best = np.argsort(-scores)[:k]            # row numbers of the k highest scores
    return [Hit(chunks[i], float(scores[i])) for i in best]


def is_in_notes(hits: list[Hit]) -> bool:
    """Search always returns k chunks, even for off-topic questions; the threshold decides if they count."""
    return bool(hits) and hits[0].score >= SIMILARITY_THRESHOLD


def _print_hits(question: str) -> None:
    print(f"\nQ: {question}")
    hits = search(question)
    if not is_in_notes(hits):
        print(f"  {NOT_COVERED}  (best {hits[0].score:.3f} < {SIMILARITY_THRESHOLD})")
        return
    for hit in hits:
        draft = "  [⚠ draft note]" if hit.chunk.metadata["draft"] else ""
        text = hit.chunk.text.replace("\n", " ")  # own-words chunks have a line break between Q and A
        print(f"  {hit.score:.3f}  {hit.chunk.id:<20}{draft}  {text[:70]}...")


# Questions for choosing the "not in my notes" threshold: compare best scores of the two groups.
IN_NOTES = [
    "Why is generating output slower and pricier than reading input?",
    "What does lowering the temperature do?",
    "Why can't a better prompt fully fix hallucination?",
    "Why does a long chat cost more per message?",
    "How does attention link 'it' to 'trophy'?",
]
NOT_IN_NOTES = [
    "How does LoRA fine-tuning work?",
    "What is Kubernetes used for?",
    "How do I bake sourdough bread?",
]


if __name__ == "__main__":
    # python -m vivaprep.retriever "your question"  → top chunks for that question
    # python -m vivaprep.retriever                  → best score per test question, to pick a threshold
    if len(sys.argv) > 1:
        _print_hits(" ".join(sys.argv[1:]))
    else:
        for label, questions in (("IN NOTES", IN_NOTES), ("NOT IN NOTES", NOT_IN_NOTES)):
            print(f"\n{label}")
            for question in questions:
                best = search(question, k=1)[0]
                verdict = "pass" if best.score >= SIMILARITY_THRESHOLD else "refuse"
                print(f"  best {best.score:.3f}  {verdict:<6}  {best.chunk.id:<20}  {question}")
