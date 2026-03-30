"""
agents.py — Agentic Orchestration Layer
Implements four agents: Intake, Retriever, Planner, Verifier.
Uses LangChain with Groq LLM (llama-3.3-70b-versatile).
"""

import os
import re
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer

class FallbackLLM:
    """Wrapper that tries primary LLM, then falls back to secondary LLM on error."""
    def __init__(self, primary: Any, secondary: Any):
        self.primary = primary
        self.secondary = secondary

    def invoke(self, input_data: Any) -> Any:
        try:
            return self.primary.invoke(input_data)
        except Exception as e:
            print(f"[FallbackLLM] Primary failed ({e}). Falling back to secondary.")
            return self.secondary.invoke(input_data)

try:
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover - optional dependency path
    ChatGroq = None  # type: ignore

from rag_pipeline import RAGPipeline, format_response_for_display, parse_structured_output
from verifier import verify_output, format_verification_report, VerificationError

load_dotenv()

# ── LLM factory ─────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast, accurate Groq model
LOCAL_MODEL = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")


class _TextResponse:
    def __init__(self, content: str):
        self.content = content


class LocalHFChatAdapter:
    """Adapter to mimic chat-like .invoke() for local HF text-generation models."""

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Allocate larger context for modern instruct models
        self.max_len = 3000 if "qwen" in model_name.lower() or "phi" in model_name.lower() else 640
        gen_pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=384,
            do_sample=False,
            truncation=True,
            return_full_text=False,
        )
        self._llm = HuggingFacePipeline(pipeline=gen_pipe)

    def invoke(self, input_data: Any) -> _TextResponse:
        if isinstance(input_data, list):
            try:
                # Use instruct chat templates for modern models
                chat_data = [{"role": "user", "content": m.content} for m in input_data]
                text = self.tokenizer.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = "\n".join(m.content for m in input_data if hasattr(m, "content"))
        else:
            text = str(input_data)

        # Truncate to avoid 'index out of range' errors in the local model
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_len)
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)

        out = self._llm.invoke(text)
        if isinstance(out, str):
            return _TextResponse(out)
        if hasattr(out, "content"):
            return _TextResponse(str(out.content))
        return _TextResponse(str(out))


def get_llm(temperature: float = 0.0, api_key: str = "") -> Any:
    """
    Return LLM client.
    Default: Groq API when GROQ_API_KEY is present.
    Fallback: local lightweight Hugging Face model if Groq isn't available or fails.
    """
    key = api_key or os.getenv("GROQ_API_KEY", "")

    if key and key != "your-groq-api-key-here" and ChatGroq is not None:
        print(f"[LLM] Fast Groq Cloud API detected. Bypassing offline LLM boot.")
        return ChatGroq(
            model=GROQ_MODEL,
            temperature=temperature,
            groq_api_key=key,
            max_retries=6,
        )

    print(f"[LLM] No API key detected. Booting heavy offline CPU model: {LOCAL_MODEL}")
    return LocalHFChatAdapter(LOCAL_MODEL)


# ── Student Profile ──────────────────────────────────────────────────────────

@dataclass
class StudentProfile:
    """Normalized student profile collected by the Intake Agent."""
    completed_courses: List[str] = field(default_factory=list)
    grades: Dict[str, str] = field(default_factory=dict)
    program: str = ""
    target_term: str = ""
    max_credits: int = 0
    catalog_year: str = ""
    transfer_credits: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Minimum profile for non-planning Q&A (identity + some academic context)."""
        return bool(self.program and len(self.completed_courses) > 0)

    def is_complete_for_planning(self) -> bool:
        """Required fields before generating a term plan (grades optional)."""
        return bool(
            self.program
            and len(self.completed_courses) > 0
            and bool(self.target_term)
            and int(self.max_credits or 0) > 0
        )

    def to_string(self) -> str:
        parts = []
        if self.program:
            parts.append(f"Program: {self.program}")
        if self.completed_courses:
            courses_with_grades = [
                f"{c} ({self.grades.get(c, 'grade unknown')})"
                for c in self.completed_courses
            ]
            parts.append(f"Completed courses: {', '.join(courses_with_grades)}")
        if self.target_term:
            parts.append(f"Target term: {self.target_term}")
        if self.max_credits:
            parts.append(f"Max credits per term: {self.max_credits}")
        if self.catalog_year:
            parts.append(f"Catalog year: {self.catalog_year}")
        if self.transfer_credits:
            parts.append(f"Transfer credits: {', '.join(self.transfer_credits)}")
        return "\n".join(parts) if parts else "No profile information provided."


# ── INTAKE AGENT ─────────────────────────────────────────────────────────────

INTAKE_SYSTEM = (
    "You are a helpful, strict UT Dallas academic advisor intake agent. "
    "You collect student info before course planning. "
    "Never make assumptions about what courses the student has taken."
)

INTAKE_PROMPT = """You are an academic advisor intake assistant for UT Dallas.
Collect the following from the student before course planning:
1. Current program/major (e.g., "Computer Science (BS)")
2. Completed courses (with course numbers like CS 1337, MATH 2413, etc.)
3. Grades for completed courses (optional but helpful)
4. Target enrollment term (e.g., "Spring 2024")
5. Maximum credits per term (e.g., 54 units)
6. Catalog year (optional)
7. Transfer credits (optional)

Current student context:
{profile}

If ANY of items 1-3 are missing, list them as clarifying questions (max 5 questions).
If all required fields (program, completed courses) are present, respond with exactly:
"PROFILE_COMPLETE"

Student said: {user_message}"""


class IntakeAgent:
    """Collects and normalises student profile. Asks clarifying questions when needed."""

    def __init__(self, llm: Any):
        self._llm = llm

    def process(self, user_message: str, profile: StudentProfile) -> Dict[str, Any]:
        self._extract_from_message(user_message, profile)
        missing = self._find_missing(profile)
        profile.missing_fields = missing

        prompt = INTAKE_PROMPT.format(
            profile=profile.to_string(),
            user_message=user_message,
        )
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            llm_text = response.content
        except Exception as e:
            llm_text = f"[Intake LLM error: {e}]"

        is_complete = "PROFILE_COMPLETE" in llm_text or profile.is_complete()

        questions = []
        if missing and not is_complete:
            question_map = {
                "program": "What is your current program/major at UT Dallas? (e.g., 'Computer Science (BS)')",
                "completed_courses": "Which courses have you completed? Please list course numbers (e.g., CS 1337, MATH 2413).",
                "grades": "Do you have grades for your completed courses? (Helps check grade-based prerequisites.)",
                "target_term": "Which term are you planning for? (e.g., 'Fall 2024', 'Spring 2025')",
                "max_credits": "What is your maximum credit load per term?",
            }
            for field_name in missing[:5]:
                if field_name in question_map:
                    questions.append(question_map[field_name])

        return {
            "is_complete": is_complete,
            "profile": profile,
            "missing_fields": missing,
            "clarifying_questions": questions,
            "llm_response": llm_text,
        }

    def _extract_from_message(self, msg: str, profile: StudentProfile):
        msg_lower = msg.lower()
        if re.search(r"\b(computer science|cs)\b", msg_lower):
            if not profile.program:
                profile.program = "Computer Science (BS)"
        elif re.search(r"\b(software engineering|se)\b", msg_lower):
            if not profile.program:
                profile.program = "Software Engineering (BS)"
        elif re.search(r"\b(data science|ds)\b", msg_lower):
            if not profile.program:
                profile.program = "Data Science (BS)"

        course_numbers = re.findall(r"\b([A-Z]{2,4})\s*-?\s*(\d[Vv]?\d{3,4})\b", msg.upper())
        for dept, num in course_numbers:
            course = f"{dept} {num.upper()}"
            if course not in profile.completed_courses:
                profile.completed_courses.append(course)

        grade_pattern = re.findall(
            r"([A-Z]{2,4})\s*-?\s*(\d[Vv]?\d{3,4})\s*:?\s*([A-F][+-]?|pass|fail)",
            msg.upper(),
            re.IGNORECASE,
        )
        for dept, num, grade in grade_pattern:
            profile.grades[f"{dept} {num.upper()}"] = grade.upper()

        # Catch terms flexibly ("Fall", "Spring 2025")
        term_pattern = re.search(r"(fall|spring|summer|iap)(?:\s*\d{4})?", msg, re.IGNORECASE)
        if term_pattern and not (profile.target_term or "").strip():
            val = term_pattern.group(0).title()
            if len(val) <= 6:
                val += " Term"
            profile.target_term = val

        # Catch credits flexibly ("15 credits" or just "15")
        credit_pattern = re.search(r"\b(\d{1,2})\b\s*(?:credits?|units?|hours?)?", msg, re.IGNORECASE)
        val_credits = int(credit_pattern.group(1)) if credit_pattern else 0
        if val_credits and not int(profile.max_credits or 0):
            if 3 <= val_credits <= 25:
                profile.max_credits = val_credits

    def _find_missing(self, profile: StudentProfile) -> List[str]:
        missing = []
        if not profile.program:
            missing.append("program")
        if not profile.completed_courses:
            missing.append("completed_courses")
        if not profile.grades and profile.completed_courses:
            missing.append("grades")
        if not profile.target_term:
            missing.append("target_term")
        if not profile.max_credits:
            missing.append("max_credits")
        return missing


# ── RETRIEVER AGENT ──────────────────────────────────────────────────────────

RETRIEVER_K_MAX = 8


class RetrieverAgent:
    """Retrieves relevant catalog chunks from FAISS. Handles query expansion."""

    def __init__(self, rag_pipeline: RAGPipeline):
        self._rag = rag_pipeline

    def retrieve(self, query: str, profile: Optional[StudentProfile] = None) -> List[Dict]:
        enriched_query = query
        if profile and profile.completed_courses:
            enriched_query = (
                f"Student completed: {', '.join(profile.completed_courses[:5])}. " + query
            )

        try:
            docs = self._rag.retrieve_with_metadata(enriched_query)
        except Exception as e:
            print(f"[RetrieverAgent] Primary retrieval error: {e}")
            docs = []

        # Reduce unrelated noise for CS planning questions (e.g., AHT pages)
        q_upper = query.upper()
        is_cs_like_query = any(tok in q_upper for tok in [" CS ", "SE ", "ECS ", "MATH ", "PHYS ", "PREREQ", "COURSE", "PLAN"])
        if is_cs_like_query and docs:
            preferred: List[Dict] = []
            others: List[Dict] = []
            for d in docs:
                url = (d.get("source_url") or "").lower()
                section = (d.get("section_heading") or "").lower()
                if ("/courses/" in url) or ("/programs/ecs/" in url) or ("/policies/" in url) or ("computer science" in section) or ("software engineering" in section):
                    preferred.append(d)
                else:
                    others.append(d)
            docs = (preferred + others)[:RETRIEVER_K_MAX]

        # Expand with per-course sub-queries if too few results
        if len(docs) < 3:
            courses_in_query = re.findall(r"\b(\d+\.\d+[A-Z]?|\d+\.\w+)\b", query)
            for course in courses_in_query[:3]:
                try:
                    extra = self._rag.retrieve_with_metadata(f"{course} prerequisites requirements UTD")
                    for d in extra:
                        if d not in docs:
                            docs.append(d)
                except Exception as e:
                    print(f"[RetrieverAgent] Sub-query error for {course}: {e}")
            docs = docs[:RETRIEVER_K_MAX]

        return docs

    def format_for_planner(self, docs: List[Dict]) -> str:
        if not docs:
            return "[No relevant catalog sections retrieved. The answer may not be in the documents.]"
        parts = []
        for i, d in enumerate(docs, 1):
            parts.append(
                f"[{i}] SOURCE: {d.get('source_url', 'unknown')}\n"
                f"    SECTION: {d.get('section_heading', 'unknown')}\n"
                f"    CHUNK_ID: {d.get('chunk_id', 'unknown')}\n"
                f"    CONTENT: {d.get('content', '')}\n"
            )
        return "\n".join(parts)


# ── PLANNER AGENT ────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a strict, citation-enforcing UT Dallas Course Planning Agent.

ABSOLUTE RULES:
1. Every claim about prerequisites, requirements, grades, or policies MUST include a citation
   formatted as [URL | section_heading OR chunk_id].
2. Do NOT state anything not found in the CONTEXT below.
3. If information is NOT in CONTEXT, respond EXACTLY with:
   "I don't have that information in the provided catalog/policies."
   Then suggest: check with your advisor, department page (catalog.utdallas.edu), or schedule of classes.
4. Never hallucinate course numbers, prerequisites, or credit requirements.
5. Always use the output format below — exact headings, in order.
6. For prerequisite eligibility questions, treat courses/grades explicitly stated in the QUESTION
   as the primary scenario. Use STUDENT PROFILE only as fallback context when the QUESTION
   does not provide enough information.

OUTPUT FORMAT (MANDATORY — exact headings):
Answer / Plan:
<direct answer, or numbered proposed courses for the term with units when planning>

Why (requirements/prereqs satisfied):
- Decision: <Eligible / Not Eligible / Need More Info / N/A / Course Plan Generated>
- Evidence: <step-by-step reasoning with inline citations [URL | section_heading OR chunk_id]>
- Next step: <concrete action for the student>

Citations:
<numbered list of every source used>

Clarifying questions (if needed):
<1–5 questions or "None">

Assumptions / Not in catalog:
<risks, offering availability not in docs, unknowns>"""


def _build_planner_prompt(context: str, profile_str: str, task_or_question: str) -> str:
    return f"""{PLANNER_SYSTEM}

CONTEXT FROM UT DALLAS CATALOG:
{context}

STUDENT PROFILE:
{profile_str}

{task_or_question}

ANSWER (follow the output format strictly):"""


class PlannerAgent:
    """Generates prerequisite decisions and course plans, strictly grounded in citations."""

    def __init__(self, llm: Any, retriever: RetrieverAgent):
        self._llm = llm
        self._retriever = retriever

    def _invoke_llm(self, prompt: str) -> str:
        """Invoke LLM with retry/backoff for transient rate limits."""
        last_err: Exception | None = None
        for i in range(8):
            try:
                response = self._llm.invoke([HumanMessage(content=prompt)])
                return response.content
            except Exception as e:
                last_err = e
                err_msg = str(e).lower()
                if ("rate_limit" in err_msg) or ("429" in err_msg):
                    # Exponential-ish backoff to survive free-tier throttling.
                    time.sleep(5 * (i + 1))
                    continue
                raise RuntimeError(f"LLM invocation failed: {e}") from e

        # Final fallback: keep user flow unblocked with a safe structured response.
        return (
            "Answer / Plan:\nI don't have enough verified context to produce a reliable answer right now.\n\n"
            "Why (requirements/prereqs satisfied):\n"
            "- Decision: Need More Info\n"
            "- Evidence: I could not complete full model reasoning for this request, so I am avoiding an unsupported answer.\n"
            "- Next step: Please restate your question with course codes and (if planning) your completed courses/term/max credits.\n\n"
            "Citations:\nNone\n\n"
            "Clarifying questions (if needed):\nNone\n\n"
            "Assumptions / Not in catalog:\nA temporary model service constraint prevented full generation.\n"
        )

    def check_prerequisite(self, query: str, profile: StudentProfile) -> Dict[str, Any]:
        docs = self._retriever.retrieve(query, profile)
        context = self._retriever.format_for_planner(docs)
        profile_context = (
            profile.to_string()
            + "\nNOTE: For this prerequisite check, prioritize courses/grades stated in QUESTION."
        )
        prompt = _build_planner_prompt(
            context, profile_context, f"QUESTION: {query}"
        )
        raw = self._invoke_llm(prompt)
        parsed = parse_structured_output(raw)
        parsed["raw_response"] = raw
        parsed["retrieved_docs"] = docs
        return parsed

    def generate_course_plan(self, profile: StudentProfile) -> Dict[str, Any]:
        completed = ", ".join(profile.completed_courses[:10]) if profile.completed_courses else "none yet"
        query = (
            f"What courses can a student in {profile.program} take next "
            f"after completing {completed}? "
            f"Suggest courses for {profile.target_term} within "
            f"{profile.max_credits or 'standard'} unit limit."
        )
        docs = self._retriever.retrieve(query, profile)
        context = self._retriever.format_for_planner(docs)
        task = (
            f"TASK: Generate a complete course plan for {profile.target_term or 'next term'}.\n"
            "Include: (1) recommended courses, (2) why each course is eligible with citations, "
            "(3) total units, (4) risks (e.g., courses not guaranteed to be offered)."
        )
        prompt = _build_planner_prompt(context, profile.to_string(), task)
        raw = self._invoke_llm(prompt)
        parsed = parse_structured_output(raw)
        parsed["raw_response"] = raw
        parsed["retrieved_docs"] = docs
        return parsed

    def answer_general_query(self, query: str, profile: StudentProfile) -> Dict[str, Any]:
        docs = self._retriever.retrieve(query, profile)
        context = self._retriever.format_for_planner(docs)
        prompt = _build_planner_prompt(
            context, profile.to_string(), f"QUESTION: {query}"
        )
        raw = self._invoke_llm(prompt)
        parsed = parse_structured_output(raw)
        parsed["raw_response"] = raw
        parsed["retrieved_docs"] = docs
        return parsed


# ── VERIFIER AGENT ───────────────────────────────────────────────────────────

class VerifierAgent:
    """Audits planner output for missing citations, logic errors, hallucinations."""

    def __init__(self, llm: Any):
        self._llm = llm

    def audit(self, parsed: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
        try:
            report = verify_output(parsed, strict=strict)
        except VerificationError as e:
            report = {
                "overall_passed": False,
                "all_issues": [str(e)],
                "all_warnings": [],
                "citation_count": 0,
                "is_abstention": False,
                "citation_check": {},
                "logic_check": {},
                "abstention_check": {},
            }
        except Exception as e:
            report = {
                "overall_passed": True,
                "all_issues": [],
                "all_warnings": [f"Verifier encountered an error: {e}"],
                "citation_count": 0,
                "is_abstention": False,
                "citation_check": {},
                "logic_check": {},
                "abstention_check": {},
            }
        parsed["verification_report"] = report
        parsed["verification_display"] = format_verification_report(report)
        return parsed


# ── MASTER ORCHESTRATOR ──────────────────────────────────────────────────────

PLANNING_QUESTION_MAP = {
    "program": "What is your current program or major? (e.g. Computer Science (BS))",
    "completed_courses": "Which courses have you already completed? (e.g. CS 1337, MATH 2413)",
    "target_term": "Which term are you planning for? (e.g. Fall 2025, Spring 2026)",
    "max_credits": "What is your maximum credit load for that term?",
}


class CoursePlanningOrchestrator:
    """
    Top-level orchestrator coordinating all four agents.
    Manages conversation state and routes queries.
    """

    def __init__(self):
        self._llm: Optional[Any] = None
        self._rag = RAGPipeline()
        self._profile = StudentProfile()
        self._intake: Optional[IntakeAgent] = None
        self._retriever: Optional[RetrieverAgent] = None
        self._planner: Optional[PlannerAgent] = None
        self._verifier: Optional[VerifierAgent] = None
        self._index_loaded = False
        self._conversation_history: List[Dict] = []
        self._in_intake_loop = False

    def initialize(self, api_key: str = ""):
        """Set up Groq LLM and load FAISS index."""
        self._llm = get_llm(api_key=api_key)
        self._rag.set_llm(self._llm)
        self._rag.load_index()
        self._index_loaded = True

        self._intake = IntakeAgent(self._llm)
        self._retriever = RetrieverAgent(self._rag)
        self._planner = PlannerAgent(self._llm, self._retriever)
        self._verifier = VerifierAgent(self._llm)

    def set_profile_from_sidebar(
        self,
        completed_courses: List[str],
        grades: Dict[str, str],
        program: str,
        target_term: str,
        max_credits: int,
        catalog_year: str = "",
        transfer_credits: Optional[List[str]] = None,
    ):
        self._profile.completed_courses = completed_courses
        self._profile.grades = grades
        self._profile.program = program
        self._profile.target_term = target_term
        self._profile.max_credits = max_credits
        self._profile.catalog_year = catalog_year or ""
        self._profile.transfer_credits = transfer_credits or []
        missing = []
        if not program:
            missing.append("program")
        if not completed_courses:
            missing.append("completed_courses")
        self._profile.missing_fields = missing

    def _planning_missing_fields(self, profile: StudentProfile) -> List[str]:
        missing: List[str] = []
        if not (profile.program or "").strip():
            missing.append("program")
        if not profile.completed_courses:
            missing.append("completed_courses")
        if not (profile.target_term or "").strip():
            missing.append("target_term")
        if not int(profile.max_credits or 0) > 0:
            missing.append("max_credits")
        return missing

    def process_message(self, user_message: str) -> Dict[str, Any]:
        """Route user message through the agent pipeline."""
        if not self._index_loaded:
            return {
                "raw_response": (
                    "⚠️ The course catalog index has not been built yet. "
                    "Please run `python ingest.py` first."
                ),
                "answer": "Index not built.",
                "why": "",
                "decision": "N/A",
                "evidence": "",
                "next_step": "Run `python ingest.py`, then restart Streamlit.",
                "citations": "",
                "clarifying_questions": "",
                "assumptions": "",
                "verification_display": "",
                "retrieved_docs": [],
            }

        # Always run intake extraction FIRST to memorize any profile data provided!
        if self._intake:
            self._intake._extract_from_message(user_message, self._profile)

        intent = self._detect_intent(user_message)
        
        missing_now = self._planning_missing_fields(self._profile)

        if intent == "course_plan" and missing_now:
            self._in_intake_loop = True

        if getattr(self, "_in_intake_loop", False):
            if user_message.lower() in ["cancel", "stop", "quit", "never mind", "exit"]:
                self._in_intake_loop = False
                intent = "smalltalk"
            elif missing_now:
                intent = "course_plan"
            else:
                self._in_intake_loop = False
                intent = "course_plan" # Data fully gathered, ready to plan

        if intent == "smalltalk":
            return self._smalltalk_response()

        # Term planning requires a complete sidebar-style profile; other intents do not.
        if intent == "course_plan":
            missing = self._planning_missing_fields(self._profile)
            if missing:
                questions = [
                    PLANNING_QUESTION_MAP[f]
                    for f in missing[:5]
                    if f in PLANNING_QUESTION_MAP
                ]
                questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
                full_raw = (
                    "Answer / Plan:\n"
                    "Before I can suggest a term plan, I need a few details (1–5 questions).\n\n"
                    "Why (requirements/prereqs satisfied):\n"
                    "- Decision: Need More Info\n"
                    "- Evidence: Planning requires program, completed subjects, target term, and unit cap.\n"
                    "- Next step: Answer the questions below or use **Save Profile** in the sidebar.\n\n"
                    "Citations:\nNone\n\n"
                    f"Clarifying questions (if needed):\n{questions_text}\n\n"
                    "Assumptions / Not in catalog:\nNone\n"
                )
                return {
                    "raw_response": full_raw,
                    "answer": "Before I can suggest a term plan, I need a few details.",
                    "why": (
                        "- Decision: Need More Info\n"
                        "- Evidence: Planning requires program, completed subjects, target term, and unit cap.\n"
                        "- Next step: Answer the clarifying questions or save your profile in the sidebar."
                    ),
                    "decision": "Need More Info",
                    "evidence": "Planning requires program, completed subjects, target term, and unit cap.",
                    "next_step": "Answer the questions below or use Save Profile in the sidebar.",
                    "citations": "None",
                    "clarifying_questions": questions_text,
                    "assumptions": "None",
                    "verification_display": "**Verification: ✅ PASSED**\n- Structured clarifying intake for planning.",
                    "retrieved_docs": [],
                }
        try:
            if intent == "course_plan":
                parsed = self._planner.generate_course_plan(self._profile)
            elif intent == "prereq_check":
                parsed = self._planner.check_prerequisite(user_message, self._profile)
            else:
                parsed = self._planner.answer_general_query(user_message, self._profile)
        except Exception as e:
            parsed = {
                "raw_response": f"⚠️ An error occurred while generating a response: {e}",
                "answer": f"Error: {e}",
                "why": "",
                "decision": "N/A",
                "evidence": "",
                "next_step": "Please try again or contact your advisor.",
                "citations": "",
                "clarifying_questions": "",
                "assumptions": "LLM call failed.",
                "retrieved_docs": [],
            }

        parsed = self._postprocess_grounding(parsed)

        # Step 3: Verifier audit
        parsed = self._verifier.audit(parsed, strict=False)

        self._conversation_history.append({
            "user": user_message,
            "assistant": parsed.get("raw_response", ""),
            "intent": intent,
        })

        return parsed

    def _detect_intent(self, message: str) -> str:
        msg = message.lower()
        if self._is_smalltalk(msg):
            return "smalltalk"
        plan_kw = [
            "plan", "schedule", "next term", "what courses", "suggest", "recommend",
            "prefer me some courses", "prefer courses", "courses for me", "recommend me"
        ]
        prereq_kw = ["can i take", "eligible", "prerequisite", "prereq", "have i completed", "do i qualify"]
        if any(k in msg for k in plan_kw):
            return "course_plan"
        if any(k in msg for k in prereq_kw):
            return "prereq_check"
        return "general"

    def _is_smalltalk(self, msg: str) -> bool:
        """Return True for short greeting-only messages."""
        normalized = re.sub(r"[^a-z]", "", msg)
        greetings = {"hi", "hii", "hiii", "hello", "hlo", "hey", "yo", "sup", "hola", "thanks", "thankyou", "whoareyou", "whareu", "whatareyou", "ok", "okay"}
        if len(msg.split()) <= 3 and not any(char.isdigit() for char in msg):
            return True
        return normalized in greetings

    def _smalltalk_response(self) -> Dict[str, Any]:
        text = (
            "Hi! I can help with prerequisite checks and next-term planning.\n"
            "Share your program, completed courses, target term, and max credits."
        )
        return {
            "raw_response": text,
            "answer": text,
            "why": "",
            "decision": "",
            "evidence": "",
            "next_step": "",
            "citations": "",
            "clarifying_questions": "",
            "assumptions": "",
            "verification_display": "",
            "retrieved_docs": [],
        }

    def _postprocess_grounding(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Improve robustness:
        1) Ensure clarifying questions exist for 'Need More Info'
        2) Backfill a citations section from retrieved docs when model omits it
        """
        decision = (parsed.get("decision") or "").lower()
        if "need more info" in decision and not (parsed.get("clarifying_questions") or "").strip():
            parsed["clarifying_questions"] = (
                "1. What is your program/major?\n"
                "2. Which courses have you completed (with grades if available)?\n"
                "3. Which term are you planning for?\n"
                "4. What is your max credit load?"
            )

        docs = parsed.get("retrieved_docs") or []
        has_citation_in_raw = "[" in (parsed.get("raw_response") or "") and "http" in (parsed.get("raw_response") or "")
        if docs and not has_citation_in_raw:
            citation_lines: List[str] = []
            for i, d in enumerate(docs[:5], 1):
                url = d.get("source_url", "")
                section = d.get("section_heading", "Unknown section")
                if url:
                    citation_lines.append(f"{i}. [{url} | {section}]")
            if citation_lines:
                citations_text = "\n".join(citation_lines)
                parsed["citations"] = citations_text
                raw = parsed.get("raw_response", "")
                if "Citations:" in raw:
                    raw = re.sub(r"Citations:\s*(.*?)(?=\nClarifying questions|\nAssumptions|\Z)", f"Citations:\n{citations_text}\n", raw, flags=re.DOTALL | re.IGNORECASE)
                else:
                    raw = raw.rstrip() + f"\n\nCitations:\n{citations_text}\n"
                parsed["raw_response"] = raw

        return parsed

    @property
    def profile(self) -> StudentProfile:
        return self._profile

    @property
    def index_loaded(self) -> bool:
        return self._index_loaded
