from __future__ import annotations

import json

import numpy as np
from fastembed import TextEmbedding

from vivaprep.chunker import Chunk, read_chunks
from vivaprep.config import DATA_DIR, EMBEDDING_MODEL, MODEL_CACHE_DIR, PROJECT_ROOT

# EMBED stage: turn every chunk into a vector (384 numbers) that captures its
# meaning, and save them. Questions are embedded later with the SAME model,
# so both live on the same "map" and can be compared.

#- get_model() loads bge-small once. The first time, it downloads 64 MB into .cache/.
#- chunk_embedding_text() builds the text to embed as lesson title + chunk text, e.g. "Tokens, not words: Everything is counted in tokens…".
#- embed_chunks() turns all chunks into vectors.
#- embed_query() turns a question into a vector, with bge's search instruction in front. The search step will use it.
#- save_embeddings() writes two files:
#  - data/embeddings.npy: the 53 × 384 numbers.
# - data/embeddings.json: which chunk each row belongs to, plus the model's name.
#- read_embeddings() loads both files. It refuses if the vectors came from a different model than the current one. That's the "same model for chunks and questions" rule, enforced in code.

VECTORS_FILE = DATA_DIR / "embeddings.npy"   # one row per chunk
INDEX_FILE = DATA_DIR / "embeddings.json"    # which chunk each row is, and which model made it

# bge models were trained with this instruction in front of search questions (not passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def get_model() -> TextEmbedding:
    """Load the embedding model once (downloads ~64 MB on first use)."""
    global _model
    if _model is None:
        _model = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(MODEL_CACHE_DIR))
    return _model


def chunk_embedding_text(chunk: Chunk) -> str:
    """The lesson title gives short chunks their topic, e.g. 'Tokens, not words: Everything is counted...'."""
    return f"{chunk.metadata['lesson_title']}: {chunk.text}"


def embed_chunks(chunks: list[Chunk]) -> np.ndarray:
    texts = [chunk_embedding_text(chunk) for chunk in chunks]
    return np.array(list(get_model().passage_embed(texts)))


def embed_query(question: str) -> np.ndarray:
    return next(iter(get_model().query_embed(QUERY_PREFIX + question)))


def save_embeddings(chunks: list[Chunk], vectors: np.ndarray) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    np.save(VECTORS_FILE, vectors)
    index = {"model": EMBEDDING_MODEL, "chunk_ids": [chunk.id for chunk in chunks]}
    INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def read_embeddings() -> tuple[list[str], np.ndarray]:
    """Read the saved vectors (used by SEARCH). Refuses vectors made by a different model."""
    if not VECTORS_FILE.exists() or not INDEX_FILE.exists():
        raise SystemExit("No embeddings in data/. Run: python -m vivaprep.embedder")
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    if index["model"] != EMBEDDING_MODEL:
        raise SystemExit(
            f"Embeddings were made with {index['model']}, but the current model is {EMBEDDING_MODEL}. "
            "Run: python -m vivaprep.embedder"
        )
    return index["chunk_ids"], np.load(VECTORS_FILE)


if __name__ == "__main__":  # stage 3: python -m vivaprep.embedder
    chunks = read_chunks()
    vectors = embed_chunks(chunks)
    save_embeddings(chunks, vectors)
    print(f"wrote {VECTORS_FILE.relative_to(PROJECT_ROOT)}  {vectors.shape[0]} chunks × {vectors.shape[1]} numbers")
    print(f"wrote {INDEX_FILE.relative_to(PROJECT_ROOT)}  (model: {EMBEDDING_MODEL})")
    first = chunks[0]
    print(f"\nexample: {first.id}")
    print(f"  text:   {chunk_embedding_text(first)[:80]}...")
    print(f"  vector: [{', '.join(f'{x:.3f}' for x in vectors[0][:5])}, ...]  (length {np.linalg.norm(vectors[0]):.2f})")
