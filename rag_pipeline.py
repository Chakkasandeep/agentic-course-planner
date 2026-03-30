"""
rag_pipeline.py — Retrieval-Augmented Generation Pipeline
Loads FAISS index, retrieves k=5 chunks, builds citation-enforcing chain.
LLM backend: Groq (llama-3.3-70b-versatile)
"""

import os
import re
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")

# ── Retriever config ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVER_K = 5

# ── Prompt template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a strict, citation-enforcing UT Dallas Course Planning Assistant.

RULES (MUST FOLLOW):
1. EVERY factual claim about prerequisites, requirements, or policies MUST include a citation
   in the format: [URL | section_heading OR chunk_id]
2. If a fact is NOT in the provided CONTEXT, do NOT state it as fact.
3. If the answer is not in CONTEXT, respond EXACTLY:
   "I don't have that information in the provided catalog/policies."
   Then suggest: advisor, department page (catalog.utdallas.edu), or schedule of classes.
4. NEVER hallucinate course numbers, prerequisites, or grade requirements.
5. If the QUESTION is casual conversation completely unrelated to UT Dallas or academics (e.g., "what's up", "how are you"), do NOT use the structured format. Instead, respond ONLY with: "I am a UT Dallas Course Planning Assistant. How can I help you with your classes today?"
6. For all academic queries, ALWAYS use the structured output format below (exact headings).

OUTPUT FORMAT (MANDATORY — exact headings, in this order):

Answer / Plan:
<direct answer, or a numbered proposed course list for the term with units if planning>

Why (requirements/prereqs satisfied):
- Decision: <Eligible / Not Eligible / Need More Info / N/A / Course Plan Generated>
- Evidence: <step-by-step reasoning; every rule or prereq claim MUST include an inline citation [URL | section_heading OR chunk_id]>
- Next step: <one concrete action for the student>

Citations:
<numbered list of every source used, each as [URL | section_heading or chunk_id]>

Clarifying questions (if needed):
<1–5 questions, or "None">

Assumptions / Not in catalog:
<risks and unknowns, e.g. whether a subject is offered this term — not fully specified in the catalog>

---

CONTEXT FROM UT DALLAS CATALOG:
{context}

STUDENT PROFILE / QUESTION:
{question}

ANSWER (follow the output format above strictly):"""

PREREQ_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT,
)


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a context string with citations."""
    if not docs:
        return "[No relevant documents retrieved from the catalog.]"
    parts = []
    for i, doc in enumerate(docs, 1):
        url = doc.metadata.get("source_url", "Unknown URL")
        section = doc.metadata.get("section_heading", "Unknown Section")
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
        parts.append(
            f"[{i}] SOURCE: {url}\n"
            f"    SECTION: {section}\n"
            f"    CHUNK_ID: {chunk_id}\n"
            f"    CONTENT: {doc.page_content}\n"
        )
    return "\n".join(parts)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class RAGPipeline:
    """Full RAG pipeline: load index → retrieve → generate → structure output."""

    def __init__(self, llm: Optional[Any] = None):
        self._vectorstore: Optional[FAISS] = None
        self._retriever = None
        self._llm = llm
        self._chain = None

    # ── Index management ──────────────────────────────────────────────────────

    def load_index(self):
        """Load FAISS index from disk."""
        if not os.path.exists(FAISS_DIR):
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_DIR}.\n"
                "Please run:  python ingest.py\n"
                "Or use the sidebar → 'Build Index' button."
            )
        try:
            embeddings = get_embeddings()
            self._vectorstore = FAISS.load_local(
                FAISS_DIR,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            self._retriever = self._vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVER_K},
            )
            print(f"RAG pipeline ready (k={RETRIEVER_K})")
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}") from e

    def set_llm(self, llm):
        self._llm = llm
        self._chain = None  # Reset so chain rebuilds with new LLM

    def _build_chain(self):
        """Lazily build the LangChain retrieval chain."""
        if self._llm is None:
            raise RuntimeError("LLM not set. Call set_llm() first.")
        if self._retriever is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")

        def _retrieve_and_format(query: str) -> str:
            try:
                docs = self._retriever.invoke(query)
                return format_docs(docs)
            except Exception as e:
                return f"[Retrieval error: {e}]"

        self._chain = (
            {
                "context": RunnableLambda(_retrieve_and_format),
                "question": RunnablePassthrough(),
            }
            | PREREQ_PROMPT
            | self._llm
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Document]:
        """Return top-k relevant documents for a query."""
        if self._retriever is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")
        try:
            return self._retriever.invoke(query)
        except Exception as e:
            raise RuntimeError(f"Retrieval failed for query '{query[:60]}': {e}") from e

    def retrieve_with_metadata(self, query: str) -> List[Dict]:
        """Return docs as dicts with metadata."""
        docs = self.retrieve(query)
        return [
            {
                "content": d.page_content,
                "source_url": d.metadata.get("source_url", ""),
                "section_heading": d.metadata.get("section_heading", ""),
                "chunk_id": d.metadata.get("chunk_id", ""),
                "category": d.metadata.get("category", ""),
                "content_preview": d.page_content[:200],
            }
            for d in docs
        ]

    # ── Generation ────────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        """Full RAG query: retrieve + generate. Returns structured dict."""
        if self._chain is None:
            self._build_chain()

        try:
            docs = self.retrieve(question)
        except Exception as e:
            return {
                "answer": f"Retrieval error: {e}",
                "decision": "N/A",
                "evidence": "",
                "next_step": "Please try again.",
                "citations": "",
                "clarifying_questions": "",
                "assumptions": f"Retrieval failed: {e}",
                "raw_response": f"Error: {e}",
                "retrieved_docs": [],
            }

        context = format_docs(docs)

        try:
            raw_output = self._chain.invoke(question)
            response_text = (
                raw_output.content
                if hasattr(raw_output, "content")
                else str(raw_output)
            )
        except Exception as e:
            response_text = (
                "Answer / Plan:\nI don't have that information in the provided catalog/policies.\n\n"
                "Why (requirements/prereqs satisfied):\n"
                "- Decision: N/A\n"
                f"- Evidence: LLM error: {e}\n"
                "- Next step: Please check your Groq API key or try again.\n\n"
                "Citations:\nNone\n\n"
                "Clarifying questions (if needed):\nNone\n\n"
                "Assumptions / Not in catalog:\nLLM call failed.\n"
            )

        parsed = parse_structured_output(response_text)
        parsed["retrieved_docs"] = [
            {
                "source_url": d.metadata.get("source_url", ""),
                "section_heading": d.metadata.get("section_heading", ""),
                "chunk_id": d.metadata.get("chunk_id", ""),
                "content_preview": d.page_content[:200],
            }
            for d in docs
        ]
        parsed["raw_response"] = response_text
        return parsed


# ── Output parsing ────────────────────────────────────────────────────────────

def _extract_from_why(why_text: str) -> Dict[str, str]:
    """Pull Decision / Evidence / Next step bullets from the Why section."""
    out = {"decision": "", "evidence": "", "next_step": ""}
    if not why_text:
        return out
    d = re.search(
        r"Decision:\s*(.*?)(?=\n\s*-\s*Evidence:|\nEvidence:|\Z)",
        why_text,
        re.DOTALL | re.IGNORECASE,
    )
    e = re.search(
        r"Evidence:\s*(.*?)(?=\n\s*-\s*Next step:|\nNext step:|\Z)",
        why_text,
        re.DOTALL | re.IGNORECASE,
    )
    n = re.search(r"Next [Ss]tep:\s*(.*?)\Z", why_text, re.DOTALL | re.IGNORECASE)
    if d:
        out["decision"] = d.group(1).strip()
    if e:
        out["evidence"] = e.group(1).strip()
    if n:
        out["next_step"] = n.group(1).strip()
    return out


def parse_structured_output(text: str) -> Dict[str, str]:
    """Parse structured LLM output into named sections (rubric + legacy fields)."""
    sections: Dict[str, str] = {
        "answer": "",
        "why": "",
        "decision": "",
        "evidence": "",
        "next_step": "",
        "citations": "",
        "clarifying_questions": "",
        "assumptions": "",
    }

    patterns = {
        "answer": (
            r"Answer\s*/\s*Plan:\s*(.*?)"
            r"(?=\nWhy\s*\(|\nCitations:|\Z)"
        ),
        "why": (
            r"Why\s*\([^)]*requirements/prereqs\s+satisfied[^)]*\):\s*(.*?)"
            r"(?=\nCitations:|\nClarifying questions|\Z)"
        ),
        "citations": (
            r"Citations:\s*(.*?)(?=\nClarifying questions|\nClarifying Questions|\Z)"
        ),
        "clarifying_questions": (
            r"Clarifying questions\s*\([^)]*\):\s*(.*?)(?=\nAssumptions\s*/\s*Not|\Z)"
        ),
        "assumptions": (
            r"Assumptions\s*/\s*Not in catalog:\s*(.*?)\Z"
        ),
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()

    # Legacy alternate headings (older prompts / model drift)
    if not sections["why"]:
        legacy = re.search(
            r"Evidence:\s*(.*?)(?=\nNext Step:|\nCitations:|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if legacy:
            sections["why"] = legacy.group(0).strip()
    if not sections["answer"]:
        m_ans = re.search(
            r"Answer\s*/\s*Plan:\s*(.*?)(?=\nWhy\s*\(|\nCitations:|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m_ans:
            sections["answer"] = m_ans.group(1).strip()

    sub = _extract_from_why(sections["why"])
    sections["decision"] = sub["decision"]
    sections["evidence"] = sub["evidence"] or sections["why"]
    sections["next_step"] = sub["next_step"]

    # Flat Decision:/Evidence:/Next Step: anywhere (fallback)
    if not sections["decision"]:
        m = re.search(
            r"(?:^|\n)Decision:\s*([^\n]+)",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if m:
            sections["decision"] = m.group(1).strip()
    if not sections["evidence"] or sections["evidence"] == sections["why"]:
        if not sub["evidence"] and sections["why"]:
            sections["evidence"] = sections["why"]
    if not sections["next_step"]:
        m = re.search(
            r"(?:^|\n)Next [Ss]tep:\s*(.*?)(?=\nCitations:|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            sections["next_step"] = m.group(1).strip()

    if not sections["citations"]:
        m = re.search(
            r"Citations:\s*(.*?)(?=\nClarifying|\nAssumptions|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            sections["citations"] = m.group(1).strip()

    if not sections["clarifying_questions"]:
        m = re.search(
            r"Clarifying questions[^:]*:\s*(.*?)(?=\nAssumptions|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            sections["clarifying_questions"] = m.group(1).strip()
    if not sections["clarifying_questions"]:
        m = re.search(
            r"Clarifying Questions[^:]*:\s*(.*?)(?=\nAssumptions|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            sections["clarifying_questions"] = m.group(1).strip()

    if not sections["assumptions"]:
        m = re.search(
            r"Assumptions\s*/\s*Not in [Cc]atalog:\s*(.*?)\Z",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            sections["assumptions"] = m.group(1).strip()

    # Fallback: unstructured response
    if not (sections["answer"] or sections["why"]):
        sections["answer"] = text.strip()

    return sections


# ── Display formatting ────────────────────────────────────────────────────────

def format_response_for_display(parsed: Dict[str, Any]) -> str:
    """Format parsed response for Streamlit markdown display."""
    lines = []

    if parsed.get("answer"):
        lines.append(f"### 📋 Answer / Plan\n{parsed['answer']}")

    why = (parsed.get("why") or "").strip()
    if why:
        lines.append(f"\n### 🧠 Why (requirements / prereqs satisfied)\n{why}")
    else:
        decision = (parsed.get("decision") or "").strip()
        evidence = (parsed.get("evidence") or "").strip()
        next_step = (parsed.get("next_step") or "").strip()
        parts = []
        if decision:
            dl = decision.lower()
            if "not eligible" in dl:
                emoji = "❌"
            elif "eligible" in dl and "not eligible" not in dl:
                emoji = "✅"
            elif "need more info" in dl:
                emoji = "❓"
            else:
                emoji = "ℹ️"
            parts.append(f"{emoji} **Decision:** {decision}")
        if evidence:
            parts.append(f"**Evidence:** {evidence}")
        if next_step:
            parts.append(f"**Next step:** {next_step}")
        if parts:
            lines.append(
                "\n### 🧠 Why (requirements / prereqs satisfied)\n"
                + "\n\n".join(parts)
            )

    if parsed.get("citations"):
        lines.append(f"\n### 📚 Citations\n{parsed['citations']}")

    if parsed.get("clarifying_questions"):
        lines.append(f"\n### ❓ Clarifying questions (if needed)\n{parsed['clarifying_questions']}")

    if parsed.get("assumptions"):
        lines.append(f"\n### ⚠️ Assumptions / not in catalog\n{parsed['assumptions']}")

    return "\n".join(lines) if lines else parsed.get("raw_response", "No response generated.")
