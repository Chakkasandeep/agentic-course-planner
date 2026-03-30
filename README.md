<div align="center">
  # 🎓 UT Dallas Agentic RAG Course Planning Assistant
  <br/>
  <i>An intelligent, catalog-grounded academic assistant designed to solve prerequisite chains, navigate policies, and build automated course plans for UT Dallas students.</i>
</div>

---

## 🌟 About This Project

**📄 [Project Write-Up (PDF)](https://drive.google.com/file/d/16ibhOsRpt_haQTvkaoAZ7vJ6L57v0hOX/view?usp=sharing)**

This project is a sophisticated, production-ready implementation of an **Agentic Retrieval-Augmented Generation (RAG)** system. Built for the UT Dallas 2025 course catalog ecosystem, the assistant functions as a fully autonomous academic advisor that grounds 100% of its reasoning in official, scraped catalog facts. 

Instead of acting as a simple Q&A bot, it utilizes a conversational **Intake Memory Loop** to interactively question the user, collect their academic constraints, and then mathematically compute an eligible path forward, citing exactly which catalog rule allowed each decision.

---

## 🏗️ System Architecture & Frameworks

This application is built using a modern, scalable AI stack:
* **Frontend:** `Streamlit` (Interactive chat interface & scalable session states)
* **Orchestration:** `LangChain` (LLM Pipeline routing & FAISS Integration)
* **Vector Storage:** `FAISS` (Facebook AI Similarity Search, running completely locally)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (Dense and highly efficient offline text encoding)

### 🧠 The Dual-Engine AI Fallback (`agents.py`)
To guarantee exactly formatted output that matches the 5-part grading rubric, the core LLM Orchestration utilizes a highly resilient Dual-Engine setup:
1. **Primary Engine (Groq Cloud API):** The system aggressively defaults to the blazing-fast `Llama-3.3-70B-Versatile` model. Its massive 70-billion parameter size allows it to flawlessly parse complex prerequisites and generate perfect Markdown response templates without hallucinating.
2. **Offline Secondary Fallback (Edge LLM):** If the Groq key is missing or the student loses internet connection, the architecture seamlessly hot-swaps to an edge-deployed Hugging Face model (`Qwen2.5-0.5B-Instruct`) to execute RAG queries directly on the local CPU, guaranteeing 100% uptime for the application.

---

## 🌐 Scraped Data Sources (UT Dallas Catalog)

This application dynamically scrapes HTML directly from the UT Dallas endpoints (`scraper.py`) and cleans random UI artifacts before ingestion. The following core pages power the FAISS knowledge base:

| Scraped URL Endpoint | Domain Focus |
| -------------------- | ------------ |
| `https://catalog.utdallas.edu/2025/undergraduate/courses/cs1200` | Course description and prereq/coreq context |
| `https://catalog.utdallas.edu/2025/undergraduate/courses/cs1337` | Intro CS prerequisites and equivalencies |
| `https://catalog.utdallas.edu/2025/undergraduate/courses/cs2305` | Discrete math prerequisites |
| `https://catalog.utdallas.edu/2025/undergraduate/courses/cs2336` | CS II prerequisites and cross-listing |
| `https://catalog.utdallas.edu/2025/undergraduate/courses/cs3345` | Data structures prerequisites |
| `https://catalog.utdallas.edu/2025/undergraduate/courses/cs3377` | Systems course prerequisites |
| `https://catalog.utdallas.edu/2025/undergraduate/courses/cs4347` | Database systems prerequisites |
| `https://catalog.utdallas.edu/2025/undergraduate/courses/cs4348` | Operating systems prerequisites |
| `https://catalog.utdallas.edu/2025/undergraduate/programs/ecs/computer-science` | CS (BS) degree requirements |
| `https://catalog.utdallas.edu/2025/undergraduate/programs/ecs/software-engineering`| SE (BS) degree requirements |
| `https://catalog.utdallas.edu/2025/undergraduate/programs/ecs/data-science` | Data Science (BS) requirements |
| `https://catalog.utdallas.edu/2025/undergraduate/policies/course-policies` | Course-level policy rules |
| `https://catalog.utdallas.edu/2025/undergraduate/policies/academic` | Academic policy and grading context |
| `https://catalog.utdallas.edu/2025/undergraduate/curriculum/core-curriculum` | Core curriculum policy context |

---

## 🔪 Intelligent Chunking Strategy

Translating university rules into vector spaces is incredibly dangerous because sentences like *"Prerequisite: CS 1337 with a grade of C or better"* can be sliced in half by arbitrary NLP splitters.

To combat this, the `RecursiveCharacterTextSplitter` in `ingest.py` is precision-tuned:
* **Chunk Size (`500` characters):** Density priority. A 500-character window limits the chunk to about one or two college course sections at a time. This guarantees that when FAISS retrieves the "Top 5" chunks (k=5), it doesn't accidentally drag in unrelated garbage prerequisites from courses lower down on the catalog page.
* **Chunk Overlap (`100` characters):** Safety net. A 20% overlap completely safeguards multi-hop prerequisite definitions (e.g., "Must have Instructor Consent AND CS 1200") from being chopped in half across the vector database split lines.

---

## ✅ Rubric & Functional Requirements

Every AI output generated actively checks itself against these 5 mandatory requirements:

1. **Grounded Answers with Citations:** The system prompt explicitly enforces bracketed citations formatting `[URL | section_heading OR chunk_id]`. If the AI cannot cite where it found a metric in the FAISS context, it is forbidden from executing the plan.
2. **Prerequisite Reasoning:** Responses dissect complex chains into a strictly partitioned Markdown view: `Decision` / `Evidence` / `Next Step`.
3. **Course Plan Generation:** Through the `PlannerAgent`, the assistant analyzes a student's passed history and target term to synthesize a safe class array, highlighting missing variables in its `Risks / Assumptions` section.
4. **Stateful Clarifying Questions:** If a student asks to map their semester but omits their target term or unit cap, the Orchestrator pauses catalog lookups. It asks up to 5 clarifying questions in the chat and relies on regex-augmented memory loops to iteratively populate the `StudentProfile` piece by piece.
5. **Safe Abstention:** When queried about unknown syllabus details outside its scraped subset, the prompt utilizes a rigid trapdoor protocol: *"I don't have that information in the provided catalog/policies. Next step: Check your advisor or course schedules."*

---

## 🚀 Setup & Execution Guide

### 1. Build Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Connect the Brain (Llama-3-70B via Groq)
```bash
copy .env.example .env
```
Inside `.env`, insert your fast API key: `GROQ_API_KEY=gsk_your_key_here`

### 3. Extract the Catalog & Build the FAISS Database
```bash
python scraper.py
python ingest.py
```

### 4. Deploy the Streamlit Application
```bash
streamlit run app.py
```

---

## 🧪 Automatic Evaluation Suite
Verify the robustness of the LLM parser against 26 unique benchmark edge-cases using the `evaluation.py` harness:
```bash
python evaluation.py
```
