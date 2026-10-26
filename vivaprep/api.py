from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vivaprep.ask import ask, lessons_cited, strip_citations
from vivaprep.loader import read_loaded
from vivaprep.questions import generate_chapter_quiz
from vivaprep.retriever import NOT_COVERED

# BACKEND: the same ask() as the command line, offered over HTTP so a frontend
# (Streamlit now, maybe Lovable later) can call it from another machine.
# Run: uvicorn vivaprep.api:app --reload   → try it at http://127.0.0.1:8000/docs

app = FastAPI(title="VivaPrep")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class Source(BaseModel):
    lesson: str
    draft: bool


class Debug(BaseModel):
    reason: str                   # why it's not covered ("" if covered)
    raw_answer: str               # Claude's reply with [chunk-id] citations
    retrieved: dict[str, float]   # chunk id → similarity score, best first
    cited: list[str]              # chunk ids Claude cited


class AskResponse(BaseModel):
    question: str
    covered: bool
    answer: str              # clean text, or exactly NOT_COVERED
    sources: list[Source]    # each cited lesson once
    debug: Debug             # the frontend decides whether to show it


@app.get("/health")
def health() -> dict[str, str]:
    """For checking the server is up (Render pings this after deploying)."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    try:
        answer = ask(request.question.strip())
    except SystemExit as e:  # our clear error messages (missing index, API key, ...) → HTTP error, not a crash
        raise HTTPException(status_code=503, detail=str(e))
    return AskResponse(
        question=answer.question,
        covered=answer.covered,
        answer=strip_citations(answer.text) if answer.covered else NOT_COVERED,
        sources=[Source(lesson=t, draft=d) for t, d in lessons_cited(answer).items()],
        debug=Debug(
            reason=answer.reason,
            raw_answer=answer.text,
            retrieved={hit.chunk.id: round(hit.score, 3) for hit in answer.hits},
            cited=[hit.chunk.id for hit in answer.sources],
        ),
    )


# ---- v2: Question mode ----

class ChapterInfo(BaseModel):
    id: str             # "ch01": what POST /questions takes
    title: str
    lessons: int


class QuestionsRequest(BaseModel):
    chapter_id: str = Field(examples=["ch01"])


class QuestionOut(BaseModel):
    type: str            # definition | why | scenario
    question: str
    lesson_id: str       # "ch01.l03": where the question comes from (v3 grades against this lesson)
    lesson_title: str
    draft: bool
    tests: list[str]     # key point ids, e.g. ["kp1", "kp3"]: shown only in debug
    attempts: int        # 2 means the first reply failed the checks


class QuestionsResponse(BaseModel):
    chapter_id: str
    chapter_title: str
    questions: list[QuestionOut]   # one per lesson, in lesson order
    problems: list[str]            # lessons whose question was dropped after two tries, and why


@app.get("/chapters", response_model=list[ChapterInfo])
def chapters_endpoint() -> list[ChapterInfo]:
    """Every chapter, for the frontend's dropdown."""
    try:
        chapters = read_loaded()
    except SystemExit as e:
        raise HTTPException(status_code=503, detail=str(e))
    return [ChapterInfo(id=c.id, title=c.title, lessons=len(c.lessons)) for c in chapters]


@app.post("/questions", response_model=QuestionsResponse)
def questions_endpoint(request: QuestionsRequest) -> QuestionsResponse:
    """One question per lesson of the chapter (the lessons are sent to Claude in parallel)."""
    if request.chapter_id not in {chapter.id for chapter in chapters_endpoint()}:
        raise HTTPException(status_code=404, detail=f"No chapter {request.chapter_id!r}. See GET /chapters.")
    try:
        quizzes = generate_chapter_quiz(request.chapter_id)
    except SystemExit as e:  # e.g. bad API key → HTTP error, not a crash
        raise HTTPException(status_code=503, detail=str(e))
    chapter = quizzes[0].chapter
    return QuestionsResponse(
        chapter_id=chapter.id,
        chapter_title=chapter.title,
        questions=[
            QuestionOut(type=q.type, question=q.text, lesson_id=quiz.lesson.full_id,
                        lesson_title=quiz.lesson.title, draft=quiz.lesson.draft,
                        tests=q.tests, attempts=quiz.attempts)
            for quiz in quizzes
            for q in quiz.questions
        ],
        problems=[f"{quiz.lesson.full_id}: {p}" for quiz in quizzes for p in quiz.problems],
    )
