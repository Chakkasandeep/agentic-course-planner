"""
app.py — Streamlit Chat Interface
Agentic RAG Course Planning Assistant for UT Dallas Catalog.
"""

import os
import streamlit as st
from typing import List, Dict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UTD Course Planning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: radial-gradient(ellipse 120% 80% at 20% -10%, rgba(99, 102, 241, 0.14), transparent 50%),
                radial-gradient(ellipse 90% 60% at 100% 0%, rgba(6, 182, 212, 0.1), transparent 45%),
                linear-gradient(165deg, #0a0e1a 0%, #111827 45%, #0d1117 100%);
    color: #e2e8f0;
}

/* Main block: readable column */
.block-container {
    padding-top: 1.25rem !important;
    max-width: 1200px !important;
}

.hero-card {
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    background: rgba(15, 23, 42, 0.55);
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* Chat messages */
.stChatMessage {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px;
    margin-bottom: 12px;
    padding: 4px;
    backdrop-filter: blur(8px);
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(99, 102, 241, 0.12) !important;
    border-color: rgba(99, 102, 241, 0.3);
}

/* Input box */
.stChatInput textarea {
    background: rgba(30, 41, 59, 0.9) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1rem !important;
    transition: transform 0.16s ease, box-shadow 0.16s ease, opacity 0.16s ease;
}
.stButton > button:hover {
    transform: translateY(-1px) scale(1.01);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35) !important;
}
.stButton > button:disabled {
    opacity: 0.55 !important;
    transform: none !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 8px;
    padding: 12px;
}

/* Success / warning banners */
.stSuccess { background: rgba(16, 185, 129, 0.15) !important; border-color: #10b981 !important; }
.stWarning { background: rgba(245, 158, 11, 0.15) !important; border-color: #f59e0b !important; }
.stError   { background: rgba(239, 68, 68, 0.15)  !important; border-color: #ef4444 !important; }
.stInfo    { background: rgba(59, 130, 246, 0.15) !important; border-color: #3b82f6 !important; }

/* Headers */
h1, h2, h3, h4 { color: #e2e8f0 !important; }
h1 { 
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(30, 41, 59, 0.6) !important;
    border-radius: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
}
.stTabs [aria-selected="true"] {
    color: #6366f1 !important;
    border-bottom-color: #6366f1 !important;
}

/* Text inputs */
.stTextInput input, .stSelectbox select, .stNumberInput input {
    background: rgba(30, 41, 59, 0.8) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 6px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #4f46e5; border-radius: 3px; }

/* Citation badges */
.citation-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_courses_input(text: str) -> List[str]:
    """Parse UTD course codes like 'CS 3345' or 'MATH2414'."""
    import re
    raw = re.findall(r"\b([A-Z]{2,4})\s*-?\s*(\d[Vv]?\d{3,4})\b", text.upper())
    courses = [f"{dept} {num.upper()}" for dept, num in raw]
    return list(dict.fromkeys(courses))


def parse_grades_input(text: str) -> Dict[str, str]:
    """Parse UTD course-grade pairs like 'CS3345:A, MATH 2414:B+'."""
    import re
    grades: Dict[str, str] = {}
    results = re.findall(
        r"([A-Z]{2,4})\s*-?\s*(\d[Vv]?\d{3,4})\s*:?\s*([A-F][+-]?|\d+\.?\d*|pass|fail)",
        text.upper(),
        re.IGNORECASE,
    )
    for dept, num, grade in results:
        grades[f"{dept} {num.upper()}"] = grade.upper()
    return grades


@st.cache_resource(show_spinner=False)
def get_orchestrator():
    """Cached orchestrator (created once per session)."""
    from agents import CoursePlanningOrchestrator
    orch = CoursePlanningOrchestrator()
    return orch


# ── Initialize session state ──────────────────────────────────────────────────

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "profile_set" not in st.session_state:
        st.session_state.profile_set = False


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(orchestrator) -> bool:
    """Render sidebar and return True if profile is ready."""
    with st.sidebar:
        st.markdown("## 🎓 Course Planning Assistant")
        st.markdown("*UT Dallas Catalog — Grounded RAG*")
        st.divider()

        # Optional Groq key (used only when USE_GROQ=1). Local model works without it.
        api_key = os.getenv("GROQ_API_KEY", "").strip()

        data_dir = os.path.join(os.path.dirname(__file__), "data")
        faiss_dir = os.path.join(data_dir, "faiss_index")
        index_exists = os.path.exists(faiss_dir)

        if not index_exists:
            st.warning("Vector index not found. Run `python ingest.py` once in terminal.")
        elif not orchestrator.index_loaded:
            with st.spinner("Connecting assistant..."):
                try:
                    orchestrator.initialize(api_key=api_key)
                    st.success("Assistant ready.")
                except Exception as e:
                    st.error(f"Init error: {e}")

        st.divider()

        # ── Student Profile ───────────────────────────────────────────────────
        st.markdown("### 🧑‍🎓 Student Profile")

        program = st.selectbox(
            "Program",
            [
                "",
                "Computer Science (BS)",
                "Software Engineering (BS)",
                "Data Science (BS)",
                "Other",
            ],
            index=0,
            key="program_select",
        )

        completed_raw = st.text_area(
            "Completed Courses",
            placeholder="e.g. CS 1337, CS 2305, MATH 2413, PHYS 2325",
            height=80,
            key="completed_courses_input",
        )

        grades_raw = st.text_input(
            "Grades (optional)",
            placeholder="e.g. CS 1337:A, CS 2305:B+, MATH 2413:A-",
            key="grades_input",
        )

        target_term = st.selectbox(
            "Target Term",
            ["", "Fall 2025", "Spring 2026", "Summer 2026", "Fall 2026", "Spring 2027"],
            index=0,
            key="term_select",
        )

        max_credits = st.number_input(
            "Max Units / Credits",
            min_value=0, max_value=72, value=54, step=3,
            key="max_credits_input",
        )

        catalog_year = st.text_input(
            "Catalog year (optional)",
            placeholder="e.g. 2024–2025",
            key="catalog_year_input",
            help="Helps align degree-chart interpretation when policies differ by year.",
        )

        transfer_raw = st.text_area(
            "Transfer / AP credits (optional)",
            placeholder="e.g. MATH 2413, PHYS 2325 (list like completed courses)",
            height=60,
            key="transfer_input",
        )

        if st.button("💾 Save Profile", use_container_width=True):
            completed_courses = parse_courses_input(completed_raw)
            grades = parse_grades_input(grades_raw)
            transfer_courses = parse_courses_input(transfer_raw)
            if orchestrator.index_loaded:
                orchestrator.set_profile_from_sidebar(
                    completed_courses=completed_courses,
                    grades=grades,
                    program=program,
                    target_term=target_term,
                    max_credits=max_credits,
                    catalog_year=catalog_year.strip(),
                    transfer_credits=transfer_courses,
                )
            st.session_state.profile_set = True
            st.success(f"Profile saved! ({len(completed_courses)} courses)")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return bool(program or st.session_state.profile_set)


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    init_session()
    orchestrator = get_orchestrator()

    # ── Sidebar (first so state is ready) ─────────────────────────────────────
    render_sidebar(orchestrator)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hero-card">'
        "<h1 style='margin:0 0 0.35rem 0; font-size:1.85rem;'>🎓 UT Dallas Course Planning Assistant</h1>"
        "<p style='margin:0; color:#94a3b8; line-height:1.5;'>"
        "Agentic RAG with <strong style='color:#c7d2fe;'>Intake → Retrieve → Plan → Verify</strong>. "
        "Answers are grounded in catalog text with <strong style='color:#a5f3fc;'>citations</strong> "
        "and explicit <strong style='color:#fcd34d;'>abstention</strong> when policy is not in the corpus."
        "</p></div>",
        unsafe_allow_html=True,
    )

    if orchestrator.index_loaded:
        st.success("Assistant online.")
    else:
        st.info("Assistant will auto-connect using local lightweight model.")

    if not orchestrator.index_loaded:
        st.warning(
            "Chat is disabled until the index is available (`python ingest.py`)."
        )

    st.markdown("##### Starter prompts")
    c1, c2, c3, c4 = st.columns(4)
    quick_prompts = {
        "Plan my term": "I want to plan my courses for the upcoming term.",
        "Check prerequisite": "I want to check the prerequisites for a specific course. Please ask me which course I am looking for.",
        "Program requirements": "I want to review my degree program requirements. Please ask me what my major is.",
        "Policy question": "What does UTD policy say about general course registration and academic rules?",
    }
    for col, (label, prompt_text) in zip([c1, c2, c3, c4], quick_prompts.items()):
        with col:
            if st.button(label, use_container_width=True, disabled=not orchestrator.index_loaded):
                st.session_state.trigger_prompt = prompt_text

    st.markdown(
        "<p style='color:#64748b; font-size:0.9rem; margin:0 0 0.5rem 0;'>"
        "Responses follow required rubric: <em>Answer / Plan → Why → Citations → Clarifying questions → Assumptions</em>."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Chat history ──────────────────────────────────────────────────────────
    chat_container = st.container(border=True)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask about prerequisites, course plans, or UTD academic policies…",
        disabled=not orchestrator.index_loaded,
    )

    prompt = st.session_state.get("trigger_prompt") or user_input
    if prompt:
        if "trigger_prompt" in st.session_state:
            del st.session_state["trigger_prompt"]
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving from UTD catalog and reasoning..."):
                try:
                    result = orchestrator.process_message(prompt)
                    from rag_pipeline import format_response_for_display
                    display_text = format_response_for_display(result)

                    st.markdown(display_text)

                    # Store message with verification and sources
                    assistant_msg = {
                        "role": "assistant",
                        "content": display_text,
                        "verification": result.get("verification_display", ""),
                        "sources": result.get("retrieved_docs", []),
                    }
                    st.session_state.messages.append(assistant_msg)

                except Exception as e:
                    error_msg = f"⚠️ Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<p style='text-align:center; color:#64748b; font-size:0.8rem;'>"
        "UTD Course Planning Assistant • Sources: UT Dallas 2025 Undergraduate Catalog • "
        "Powered by LangChain + Groq (llama-3.3-70b-versatile) + FAISS + sentence-transformers/all-MiniLM-L6-v2"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
