import os

import requests
import streamlit as st

# FRONTEND: a web page that sends the question to the FastAPI backend (POST /ask)
# and shows the reply. It only talks HTTP, never imports vivaprep, so it can be
# deployed separately (Streamlit Cloud) without the embedding model or API key.
# Run: streamlit run app.py   (with the backend running: uvicorn vivaprep.api:app --port 8001)

API_URL = os.getenv("VIVAPREP_API_URL", "http://127.0.0.1:8001")

# VOICE INPUT: Chrome's built-in Web Speech API turns speech into text inside the
# browser (Chrome sends the audio to Google; it's free and needs no key). The
# backend still only ever receives text. Works in Chrome and Edge, not Firefox.
VOICE_JS = """
export default function(component) {
  const { setTriggerValue, parentElement } = component;
  const button = parentElement.querySelector("button");
  const status = parentElement.querySelector(".status");
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    button.disabled = true;
    status.textContent = "Voice input needs Chrome or Edge.";
    return;
  }
  const recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;  // only the final text
  let listening = false;

  // Every step writes to the status text, so a failure is visible instead of silent.
  button.onclick = () => {
    status.textContent = "Starting...";
    try {
      listening ? recognition.stop() : recognition.start();
    } catch (error) {
      status.textContent = `Voice error: ${error.message}`;
    }
  };
  recognition.onstart = () => { listening = true; button.textContent = "⏹ Stop"; status.textContent = "Listening..."; };
  recognition.onend = () => {
    listening = false;
    button.textContent = "🎤 Speak";
    if (status.textContent === "Listening...") status.textContent = "Didn't catch anything. Try again.";
  };
  recognition.onresult = (event) => {
    const text = event.results[0][0].transcript;
    status.textContent = `Heard: "${text}"`;
    setTriggerValue("transcript", text);
  };
  recognition.onerror = (event) => { status.textContent = `Voice error: ${event.error}`; };
  return () => recognition.abort();  // stop listening if the component is removed
}
"""
voice_input = st.components.v2.component(
    "voice_input",
    html='<button>🎤 Speak</button> <span class="status"></span>',
    css="button { padding: 0.3rem 0.8rem; cursor: pointer; } .status { margin-left: 0.5rem; opacity: 0.7; }",
    js=VOICE_JS,
)


def call_api(method: str, path: str, payload: dict | None = None) -> dict | list:
    """The backend's JSON reply, or {"error": message} if it can't be reached or fails."""
    try:
        response = requests.request(method, f"{API_URL}{path}", json=payload, timeout=60)
    except requests.ConnectionError:
        return {"error": f"Can't reach the backend at {API_URL}. Is uvicorn running?"}
    if response.status_code != 200:
        return {"error": f"Backend error {response.status_code}: {response.json().get('detail')}"}
    return response.json()


def is_error(reply) -> bool:
    return isinstance(reply, dict) and "error" in reply


def draft_label(draft: bool) -> str:
    return "  ⚠ draft note" if draft else ""


# Look of the tabs: dark "pill" tabs with a blurred background, styled after a
# React animated-tabs component. Plain CSS on Streamlit's own tab elements.
TABS_CSS = """
<style>
[data-testid="stTabs"] [role="tablist"] {
  gap: 0.25rem;
  width: fit-content;
  padding: 0.25rem;
  border-radius: 0.75rem;
  background: rgba(17, 17, 17, 0.6);
  backdrop-filter: blur(4px);
}
[data-testid="stTab"] {
  padding: 0.35rem 0.9rem;
  border-radius: 0.5rem;
  border: none;
  color: rgba(255, 255, 255, 0.75);
  transition: background 0.3s ease, color 0.3s ease;  /* the soft slide between tabs */
}
[data-testid="stTab"]:hover { color: #fff; }
[data-testid="stTab"][aria-selected="true"] {
  color: #fff;
  background: rgba(17, 17, 17, 0.82);
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);
}
[data-testid="stTab"] > div:last-child:empty { display: none; }  /* Streamlit's red underline */
[data-testid="stTab"] p { font-weight: 500; }
[data-testid="InputInstructions"] { display: none; }  /* the "Press ⌘+Enter to submit" hint over the text */
[data-testid="stTabPanel"] {
  margin-top: 0.75rem;
  padding: 1rem 1.25rem;
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 0.75rem;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.06);
}
</style>
"""

st.set_page_config(page_title="VivaPrep", page_icon="🎓")
st.html(TABS_CSS)
st.title("VivaPrep")
st.caption("Practise for interviews using only your own notes.")
show_debug = st.sidebar.checkbox("Show debug")  # one switch for both tabs

# Streamlit reruns this whole file on every click; session_state keeps the last
# reply in each tab, so clicking elsewhere doesn't wipe it.
ask_tab, quiz_tab = st.tabs(["Ask", "Quiz me"])

# ---- v1: Ask a question about the notes (POST /ask) ----
with ask_tab:
    # The transcript fills the question box (it must be set before the box is
    # drawn); you can fix misheard words, then press Ask.
    voice = voice_input(key="voice", on_transcript_change=lambda: None)
    if voice.transcript:
        st.session_state.question = voice.transcript
        st.session_state.pop("result", None)  # a new question: clear the old answer

    with st.form("ask"):
        question = st.text_area("Your question", key="question", height=90)  # wraps long questions
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Searching your notes..."):
            st.session_state.result = call_api("POST", "/ask", {"question": question.strip()})

    result = st.session_state.get("result")
    if is_error(result):
        st.error(result["error"])
    elif result:
        st.markdown(f"**Q: {result['question']}**")  # which question this answer belongs to
        if result["covered"]:
            st.write(result["answer"])
            for source in result["sources"]:
                st.caption(f"From your notes: {source['lesson']}{draft_label(source['draft'])}")
        else:
            st.info(result["answer"])
        if show_debug:
            st.json(result["debug"])

# ---- v2: one question per lesson of a chapter (GET /chapters, POST /questions) ----
with quiz_tab:
    chapters = call_api("GET", "/chapters")
    if is_error(chapters):
        st.error(chapters["error"])
    else:
        labels = {c["id"]: f"{c['id']}  {c['title']}  ({c['lessons']} lessons)" for c in chapters}
        chapter_id = st.selectbox("Chapter", list(labels), format_func=labels.get)

        if st.button("Generate questions"):
            with st.spinner("Writing one question per lesson..."):
                st.session_state.quiz = call_api("POST", "/questions", {"chapter_id": chapter_id})

        quiz = st.session_state.get("quiz")
        if is_error(quiz):
            st.error(quiz["error"])
        elif quiz and quiz["chapter_id"] == chapter_id:  # hide questions from a previously chosen chapter
            for number, item in enumerate(quiz["questions"], start=1):
                st.markdown(f"{number}. *{item['type']}*: {item['question']}")
                if show_debug:  # where the question came from, and what v3 will grade against
                    st.caption(f"Lesson: {item['lesson_id']} {item['lesson_title']}{draft_label(item['draft'])}"
                               f" · tests: {', '.join(item['tests'])} · attempts: {item['attempts']}")
            for problem in quiz["problems"]:
                st.warning(f"Question dropped for {problem}")
