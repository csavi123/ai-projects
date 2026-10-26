from __future__ import annotations

from vivaprep.config import SIMILARITY_THRESHOLD, TOP_K
from vivaprep.loader import read_loaded
from vivaprep.retriever import is_in_notes, search

# RETRIEVAL EVAL: score search() on its own, without calling Claude (free, and
# the same result every run). The quiz questions are the labelled test set: each
# belongs to a known lesson, and the quiz is never chunked, so nothing leaks.
# Rerun after changing chunking, the embedding model, TOP_K or the threshold.

# Questions the notes don't cover: search should refuse all of them.
OFF_NOTES = [
    "How does LoRA fine-tuning work?",
    "What is Kubernetes used for?",
    "How do I bake sourdough bread?",
    "Who is Cinny Savitri?",
    "What is ABC model?",
]


def quiz_questions() -> list[tuple[str, str]]:
    """(question, lesson it belongs to) for every quiz question in the loaded notes."""
    return [
        (item["q"], lesson.full_id)
        for chapter in read_loaded()
        for lesson in chapter.lessons
        for item in lesson.quiz
    ]


def run() -> None:
    questions = quiz_questions()
    top1 = in_top_k = refused = 0
    misses = []
    for question, expected in questions:
        hits = search(question)
        lessons = [f"{hit.chunk.metadata['chapter_id']}.{hit.chunk.metadata['lesson_id']}" for hit in hits]
        top1 += lessons[0] == expected
        in_top_k += expected in lessons
        if not is_in_notes(hits):
            refused += 1
            misses.append(f"  refused  {expected}  (best {hits[0].score:.3f})  {question}")
        elif lessons[0] != expected:
            misses.append(f"  {expected} → {lessons[0]}  ({hits[0].score:.3f})  {question}")

    passed = []
    for question in OFF_NOTES:
        hits = search(question)
        if is_in_notes(hits):
            passed.append(f"  passed  ({hits[0].score:.3f}, {hits[0].chunk.id})  {question}")

    n = len(questions)
    print(f"\nRetrieval eval: {n} quiz questions, {len(OFF_NOTES)} off-notes questions, "
          f"top {TOP_K}, threshold {SIMILARITY_THRESHOLD}\n")
    print(f"  Hit@1 (right lesson ranked first):        {top1}/{n}")
    print(f"  Recall@{TOP_K} (right lesson in top {TOP_K}):        {in_top_k}/{n}")
    print(f"  False refusals (real question blocked):    {refused}/{n}")
    print(f"  False passes (off-notes question let in):  {len(passed)}/{len(OFF_NOTES)}")
    if misses:
        print("\nQuiz misses (expected → got):")
        print("\n".join(misses))
    if passed:
        print("\nOff-notes questions that got through:")
        print("\n".join(passed))


if __name__ == "__main__":  # python -m vivaprep.eval_retrieval
    run()
