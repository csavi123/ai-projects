from __future__ import annotations

import html
import re
import sys
from dataclasses import dataclass, field

from vivaprep.llm import ask_claude
from vivaprep.retriever import NOT_COVERED, Hit, is_in_notes, search

# ASK: the full RAG flow. Retrieve the closest chunks, refuse in code if they're
# not close enough (layer 1, free), otherwise let Claude answer ONLY from them.
# Code then decides (layer 2) whether to show Claude's answer: only if Claude
# didn't signal "no answer" and every citation is a chunk we sent. Otherwise the
# user sees the NOT_COVERED constant, so the wording never comes from the model.

SENTINEL = "NOT_COVERED"  # Claude's "no answer" signal; code checks for it, the user never sees it

SYSTEM_PROMPT = f"""You answer questions using only the study notes given in <notes>.
Rules:
1. Use only information from the notes. Do not add outside knowledge.
2. After each sentence, cite the chunk id(s) it came from in square brackets, e.g. [ch01.l01.kp2].
3. If the notes do not answer the question, reply with only the word {SENTINEL}
4. Keep the answer to 2-5 sentences."""

BRACKETS = re.compile(r"\[([^\]]+)\]")         # anything in [ ... ]
CHUNK_ID = re.compile(r"^\w+\.\w+\.\w+$")      # ch01.l01.kp2, ch01.l03.ownwords, ...


@dataclass
class Answer:
    question: str
    hits: list[Hit]                 # the chunks retrieved (and, if in notes, sent to Claude)
    covered: bool                   # code's final decision: show the answer, or show NOT_COVERED
    reason: str = ""                # why it's not covered (shown in brackets, for debugging)
    text: str = ""                  # Claude's reply, or "" if refused before calling it
    sources: list[Hit] = field(default_factory=list)  # retrieved chunks Claude actually cited


def build_notes(hits: list[Hit]) -> str:
    """The retrieved chunks as tagged text, so notes (data) are kept apart from instructions."""
    lines = ["<notes>"]
    for hit in hits:
        title = html.escape(hit.chunk.metadata["lesson_title"], quote=True)
        lines.append(f'<chunk id="{hit.chunk.id}" lesson="{title}">{hit.chunk.text}</chunk>')
    lines.append("</notes>")
    return "\n".join(lines)


def cited_ids(text: str) -> list[str]:
    """Chunk ids in the answer, in order, without repeats; handles [a, b] as well as [a][b]."""
    ids = []
    for group in BRACKETS.findall(text):
        for part in re.split(r"[,;]", group):
            part = part.strip()
            if CHUNK_ID.match(part) and part not in ids:
                ids.append(part)
    return ids


def ask(question: str) -> Answer:
    hits = search(question)
    if not is_in_notes(hits):  # layer 1: Claude is never called
        return Answer(question, hits, covered=False,
                      reason=f"best match {hits[0].score:.3f}, below the threshold")
    text = ask_claude(SYSTEM_PROMPT, f"{build_notes(hits)}\n\nQuestion: {question}")
    sent = {hit.chunk.id: hit for hit in hits}
    ids = cited_ids(text)
    unknown = [i for i in ids if i not in sent]

    # Layer 2, decided by code: trust the answer only if it cites at least one
    # chunk we sent and nothing we didn't send.
    if SENTINEL in text:
        reason = "Claude found no answer in the retrieved notes"
    elif unknown:
        reason = f"cited chunks that weren't sent: {', '.join(unknown)}"
    elif not ids:
        reason = "the answer cited no notes"
    else:
        return Answer(question, hits, covered=True, text=text, sources=[sent[i] for i in ids])
    return Answer(question, hits, covered=False, reason=reason, text=text)


def strip_citations(text: str) -> str:
    """Remove the [chunk-id] brackets: they're for code to check, not for the reader."""
    text = BRACKETS.sub(lambda m: "" if cited_ids(m.group(0)) else m.group(0), text)
    return re.sub(r"\s+([.,;:])", r"\1", text)  # "token [id]." became "token ." → "token."


def lessons_cited(answer: Answer) -> dict[str, bool]:
    """Lesson title → draft?, each lesson once, in citation order (what the user sees as sources)."""
    lessons = {}
    for hit in answer.sources:
        lessons[hit.chunk.metadata["lesson_title"]] = hit.chunk.metadata["draft"]
    return lessons


def print_answer(answer: Answer, debug: bool = False) -> None:
    print(f"\nQ: {answer.question}\n")
    if not answer.covered:
        print(NOT_COVERED + (f"  ({answer.reason})" if debug else ""))
        return
    if debug:  # everything code used to decide: raw answer, cited chunks
        print(answer.text)
        print("\nSources:")
        for hit in answer.sources:
            draft = "  ⚠ draft note" if hit.chunk.metadata["draft"] else ""
            print(f"  {hit.chunk.id:<20} {hit.chunk.metadata['lesson_title']}{draft}")
        return
    print(strip_citations(answer.text))
    print()
    for title, draft in lessons_cited(answer).items():
        print(f"From your notes: {title}" + ("  ⚠ draft note" if draft else ""))


if __name__ == "__main__":  # python -m vivaprep.ask [--debug] "your question"
    args = sys.argv[1:]
    debug = "--debug" in args
    args = [a for a in args if a != "--debug"]
    if not args:
        raise SystemExit('Usage: python -m vivaprep.ask [--debug] "your question"')
    print_answer(ask(" ".join(args)), debug)
