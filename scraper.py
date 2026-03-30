"""
UTD 2025 Undergraduate Catalog Scraper
======================================
Scrapes course pages, program requirement pages, and policy pages from
catalog.utdallas.edu and writes a single flat text file (raw_catalog.txt)
plus a metadata.json summary — ready for the RAG ingest pipeline.

Run:
    pip install requests beautifulsoup4 lxml
    python scraper.py

Outputs (written to ./data/):
    raw_catalog.txt   — one document block per page, with structured headers
    metadata.json     — counts, word totals, validation results
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections import defaultdict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_CATALOG = os.path.join(DATA_DIR, "raw_catalog.txt")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# URL lists
# ---------------------------------------------------------------------------

# (url, short_label)  — 29 CS courses + MATH/PHYS prereq courses for chain completeness
COURSE_PAGE_URLS: list[tuple[str, str]] = [
    # ── Core CS sequence ──────────────────────────────────────────────────
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs1200",  "CS 1200"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs1334",  "CS 1334"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs1335",  "CS 1335"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs1336",  "CS 1336"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs1436",  "CS 1436"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs1337",  "CS 1337"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs2305",  "CS 2305"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs2336",  "CS 2336"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs2337",  "CS 2337"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs2340",  "CS 2340"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs2v95",  "CS 2V95"),
    # ── Upper-division CS ─────────────────────────────────────────────────
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs3305",  "CS 3305"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs3341",  "CS 3341"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs3345",  "CS 3345"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs3354",  "CS 3354"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs3162",  "CS 3162"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs3377",  "CS 3377"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs4141",  "CS 4141"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs4301",  "CS 4301"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs4337",  "CS 4337"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs4341",  "CS 4341"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs4347",  "CS 4347"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs4348",  "CS 4348"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs4349",  "CS 4349"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs4352",  "CS 4352"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs4375",  "CS 4375"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs4384",  "CS 4384"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs4390",  "CS 4390"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs4485",  "CS 4485"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/cs4v96",  "CS 4V96"),
    # ── MATH prereqs (needed for full prerequisite chain reasoning) ────────
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/math2413", "MATH 2413"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/math2414", "MATH 2414"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/math2415", "MATH 2415"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/math2418", "MATH 2418"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/math3310", "MATH 3310"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/math3379", "MATH 3379"),
    # ── Science/ECS prereqs ───────────────────────────────────────────────
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/phys2325", "PHYS 2325"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/phys2326", "PHYS 2326"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/ecs3390",  "ECS 3390"),
    ("https://catalog.utdallas.edu/2025/undergraduate/courses/ecs2390",  "ECS 2390"),
]

# Program requirement pages (degree plans)
PROGRAM_REQ_URLS: list[str] = [
    "https://catalog.utdallas.edu/2025/undergraduate/programs/ecs/computer-science",
    "https://catalog.utdallas.edu/2025/undergraduate/programs/ecs/software-engineering",
    "https://catalog.utdallas.edu/2025/undergraduate/programs/ecs/data-science",
    "https://catalog.utdallas.edu/2025/undergraduate/programs/ecs",          # ECS overview
    "https://catalog.utdallas.edu/2025/undergraduate/programs/ecs/computer-science/four-year",  # 4-year plan
    "https://catalog.utdallas.edu/2025/undergraduate/programs/ecs/software-engineering/four-year",
]

# Academic policy pages
POLICY_URLS: list[str] = [
    "https://catalog.utdallas.edu/2025/undergraduate/policies/course-policies",
    "https://catalog.utdallas.edu/2025/undergraduate/policies/academic",
    "https://catalog.utdallas.edu/2025/undergraduate/policies/disciplinary-actions",
    "https://catalog.utdallas.edu/2025/undergraduate/policies/degree-plans",
    "https://catalog.utdallas.edu/2025/undergraduate/policies/registration",
    "https://catalog.utdallas.edu/2025/undergraduate/policies/graduation",
    "https://catalog.utdallas.edu/2025/undergraduate/policies/graduate-courses/fasttrack",
    "https://catalog.utdallas.edu/2025/undergraduate/curriculum/core-curriculum",
    "https://catalog.utdallas.edu/2025/undergraduate/curriculum/other-degree-requirements",
]

# If cleanup reduces corpus size, auto-add relevant pages until target is met.
TARGET_WORD_COUNT = 30_000
EXTRA_PAGE_CAP = 20

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags whose content adds no value to RAG chunks
_STRIP_TAGS = {
    "nav", "footer", "header", "script", "style", "noscript", "svg",
    "form", "button", "aside", "meta", "link", "iframe", "picture",
}

# Exact noise strings to filter from extracted lines
_NOISE_LINES = {
    "download page as a pdf",
    "download page as a docx",
    "download page as a docx (ms word) file",
    "open page in a printable window",
    "compare versions",
    "send page to printer",
    "(null clip target)",
    "bugz",
    "ut dallas 2025 undergraduate catalog",
}


def fetch_html(url: str, retries: int = 4, base_delay: float = 2.0) -> str | None:
    """GET url with exponential backoff + jitter. Returns raw HTML or None."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt == retries - 1:
                print(f"      [ERROR] gave up on {url}: {exc}")
                return None
            wait = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"      [WARN] attempt {attempt + 1}/{retries} failed ({exc}); "
                  f"retry in {wait:.1f}s")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# HTML → clean text
# ---------------------------------------------------------------------------

def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove boilerplate tags in-place."""
    for tag_name in _STRIP_TAGS:
        for node in soup.find_all(tag_name):
            node.decompose()


def _content_root(soup: BeautifulSoup) -> Tag | None:
    """
    UTD catalog pages wrap the real content in <article id="article">.
    Fall back to common alternatives if that's absent.
    """
    for selector in [
        ("article", {"id": "article"}),
        ("div",     {"id": "bukku-page"}),
        ("div",     {"id": "page-content"}),
        ("main",    {}),
    ]:
        node = soup.find(selector[0], selector[1])
        if node:
            return node  # type: ignore[return-value]
    return soup.body  # type: ignore[return-value]


def _normalize(raw: str) -> str:
    """
    Collapse whitespace, drop empty / noise lines, de-duplicate blank lines.
    Returns clean multi-line string.
    """
    # Fix common parenthesis artifacts from line-by-line extraction:
    # "(( CE 2305 or CS 2305 ))" -> "( CE 2305 or CS 2305 )"
    raw = re.sub(r"\(\(\s*", "( ", raw)
    raw = re.sub(r"\s*\)\)", " )", raw)

    out: list[str] = []
    prev_blank = False
    for line in raw.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            if not prev_blank:
                out.append("")
            prev_blank = True
            continue
        prev_blank = False
        if line.lower() in _NOISE_LINES:
            continue
        # drop lines that are only punctuation / nav artifacts
        if re.fullmatch(r"[|\-•·]+", line):
            continue
        out.append(line)
    return "\n".join(out).strip()


def _drop_program_faculty_section(content: str) -> str:
    """
    Remove long faculty rosters from program pages; they are high-noise for planning.
    """
    lines = content.splitlines()
    out: list[str] = []
    skipping = False

    for line in lines:
        ll = line.lower()
        if not skipping and (
            ll == "faculty" or ll.endswith(" faculty") or ll.startswith("ut dallas affiliated faculty")
        ):
            skipping = True
            continue
        if skipping and (
            ll.startswith("degree requirements")
            or ll.startswith("minor")
            or ll.startswith("certificate")
            or ll.startswith("view an example of degree requirements by semester")
        ):
            skipping = False
        if not skipping:
            out.append(line)

    return _normalize("\n".join(out))


def html_to_text(html: str) -> tuple[str, str]:
    """
    Parse HTML, strip noise, extract heading + body text.
    Returns (heading, content).
    """
    soup = BeautifulSoup(html, "lxml")
    _strip_noise(soup)
    root = _content_root(soup)
    if not root:
        return "Unknown", ""

    # Best heading: first h1 inside the content root
    h = root.find(["h1", "h2"])
    heading = h.get_text(" ", strip=True) if h else "Main Content"

    text = root.get_text("\n", strip=True)
    content = _normalize(text)
    return heading, content


def _normalize_cross_listed(text: str) -> str:
    """Normalize '(Same as XYZ)' fragments for parsed field readability."""
    return re.sub(
        r"\(\s*Same as\s+([A-Z]{2,4}\s*\d[V]?\d{2,3}\s*)\)",
        r"(Cross-listed with \1)",
        text,
        flags=re.IGNORECASE,
    ).strip()


# ---------------------------------------------------------------------------
# Document ID + structured field extraction
# ---------------------------------------------------------------------------

def doc_id(url: str, category: str) -> str:
    slug = urlparse(url).path.rstrip("/")
    # keep last two path segments for uniqueness (e.g. ecs/computer-science)
    parts = [p for p in slug.split("/") if p]
    slug = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    slug = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    return f"{category}:{slug}"


# Compiled once for speed
_RE_PREREQ  = re.compile(
    r"Prerequisite(?:\(s\))?[:\s]+(.+?)(?=Corequisite|Prerequisite or Corequisite|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_RE_COREQ   = re.compile(
    r"(?:Corequisite|Co-requisite)(?:\(s\))?[:\s]+(.+?)(?=Prerequisite|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_RE_CREDITS = re.compile(r"\((\d)\s+semester\s+credit\s+hours?\)", re.IGNORECASE)
_RE_GRADE   = re.compile(
    r"(?:grade\s+of\s+[A-C][+-]?\s+or\s+better|minimum\s+grade\s+of\s+[A-C][+-]?)",
    re.IGNORECASE,
)
_RE_CONSENT = re.compile(r"instructor\s+consent", re.IGNORECASE)


def extract_course_fields(content: str) -> dict[str, str | bool]:
    """
    Pull structured fields from a course page's text content.
    All regex run against a single-line version of the text so newlines
    don't break multi-part prerequisite statements.
    """
    flat = " ".join(content.split())   # collapse all whitespace to single spaces

    prereq = coreq = credits = ""
    min_grades: list[str] = []

    m = _RE_PREREQ.search(flat)
    if m:
        prereq = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".")
        prereq = _normalize_cross_listed(prereq)

    m = _RE_COREQ.search(flat)
    if m:
        coreq = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".")
        coreq = _normalize_cross_listed(coreq)

    m = _RE_CREDITS.search(flat)
    if m:
        credits = m.group(1)

    min_grades = list(dict.fromkeys(  # preserve order, deduplicate
        g.strip() for g in _RE_GRADE.findall(flat)
    ))

    consent = bool(_RE_CONSENT.search(flat))

    return {
        "prereq":              prereq,
        "coreq":               coreq,
        "credit_hours":        credits,
        "min_grade":           "; ".join(min_grades),
        "instructor_consent":  consent,
    }


def _canonical_catalog_url(href: str, base_url: str) -> str | None:
    """Resolve a link and keep only 2025 undergraduate catalog pages."""
    url = urljoin(base_url, href).split("#", 1)[0].split("?", 1)[0].rstrip("/")
    if not url.startswith("https://catalog.utdallas.edu/2025/undergraduate/"):
        return None
    return url


def discover_extra_urls(seed_urls: list[str], excluded_urls: set[str], limit: int) -> list[str]:
    """Find additional high-value policy/curriculum/program pages from seed pages."""
    discovered: list[str] = []
    seen = set(excluded_urls)
    allowed_parts = ("/policies/", "/curriculum/", "/programs/")

    for seed in seed_urls:
        html = fetch_html(seed)
        if not html:
            continue
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            candidate = _canonical_catalog_url(a["href"], seed)
            if not candidate:
                continue
            if not any(part in candidate for part in allowed_parts):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            discovered.append(candidate)
            if len(discovered) >= limit:
                return discovered
    return discovered


# ---------------------------------------------------------------------------
# Scraping orchestration
# ---------------------------------------------------------------------------

def scrape_page(url: str, category: str) -> dict | None:
    """Fetch, clean, and return a document dict, or None on failure."""
    html = fetch_html(url)
    if not html:
        return None

    heading, content = html_to_text(html)
    if category == "program_requirement":
        content = _drop_program_faculty_section(content)
    if len(content) < 60:          # almost certainly a 404 / redirect page
        return None

    doc = {
        "doc_id":          doc_id(url, category),
        "source_url":      url,
        "section_heading": heading,
        "category":        category,
        "content":         content,
        "word_count":      len(content.split()),
    }

    # Attach parsed fields for course pages (speeds up downstream chunking)
    if category == "course_page":
        doc["parsed"] = extract_course_fields(content)

    return doc


def scrape_all() -> tuple[list[dict], dict[str, int]]:
    """
    Scrape all configured URLs in three phases.
    Returns (docs, category_stats).
    """
    docs: list[dict] = []
    stats: dict[str, int] = defaultdict(int)
    seen_ids: set[str] = set()

    def process(url: str, category: str, label: str) -> None:
        print(f"  [{category[:4].upper()}] {label}")
        doc = scrape_page(url, category)
        if not doc:
            print("         x no content")
            return
        if doc["doc_id"] in seen_ids:
            print(f"         x duplicate id: {doc['doc_id']}")
            return
        seen_ids.add(doc["doc_id"])
        docs.append(doc)
        stats[category] += 1
        print(f"         ok {doc['doc_id']}  ({doc['word_count']:,} words)")

    print("\n" + "=" * 64)
    print("PHASE 1 — Course pages")
    print("=" * 64)
    for url, label in COURSE_PAGE_URLS:
        process(url, "course_page", label)
        time.sleep(0.6 + random.uniform(0, 0.3))   # polite crawl rate

    print("\n" + "=" * 64)
    print("PHASE 2 — Program requirement pages")
    print("=" * 64)
    for url in PROGRAM_REQ_URLS:
        process(url, "program_requirement", url.split("/")[-1] or url.split("/")[-2])
        time.sleep(0.6 + random.uniform(0, 0.3))

    print("\n" + "=" * 64)
    print("PHASE 3 — Policy pages")
    print("=" * 64)
    for url in POLICY_URLS:
        process(url, "policy", url.split("/")[-1] or url.split("/")[-2])
        time.sleep(0.6 + random.uniform(0, 0.3))

    # Phase 4: auto-expand relevant pages when total words fall below target.
    total_words = sum(d["word_count"] for d in docs)
    if total_words < TARGET_WORD_COUNT:
        print("\n" + "=" * 64)
        print(f"PHASE 4 — Auto-expand corpus to reach {TARGET_WORD_COUNT:,} words")
        print("=" * 64)
        existing_urls = {d["source_url"] for d in docs}
        seeds = PROGRAM_REQ_URLS + POLICY_URLS
        extra_urls = discover_extra_urls(seeds, existing_urls, EXTRA_PAGE_CAP)
        if not extra_urls:
            print("  [INFO] no extra pages discovered")
        for url in extra_urls:
            process(url, "policy", url.split("/")[-1] or url.split("/")[-2])
            time.sleep(0.5 + random.uniform(0, 0.2))
            total_words = sum(d["word_count"] for d in docs)
            if total_words >= TARGET_WORD_COUNT:
                break
        print(f"  [INFO] total words now: {total_words:,}")

    return docs, dict(stats)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
_SEP = "=" * 64
_THIN = "-" * 64


def write_catalog(docs: list[dict]) -> int:
    """
    Write raw_catalog.txt.

    Format per document:
        ==============================
        DOC_ID:    <id>
        SOURCE:    <url>
        SECTION:   <heading>
        CATEGORY:  <category>
        PREREQ:    <...>          (course_page only, if present)
        COREQ:     <...>          (course_page only, if present)
        CREDITS:   <n>            (course_page only, if present)
        MIN_GRADE: <...>          (course_page only, if present)
        CONSENT:   yes            (course_page only, if applicable)
        CONTENT:
        <full page text>
        ------------------------------

    The structured header lines make it trivial for the chunker to
    split on '======' and parse metadata without re-running regexes.
    """
    with open(RAW_CATALOG, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(_SEP + "\n")
            f.write(f"DOC_ID:   {doc['doc_id']}\n")
            f.write(f"SOURCE:   {doc['source_url']}\n")
            f.write(f"SECTION:  {doc['section_heading']}\n")
            f.write(f"CATEGORY: {doc['category']}\n")

            if doc["category"] == "course_page" and "parsed" in doc:
                p = doc["parsed"]
                if p["prereq"]:
                    f.write(f"PREREQ:   {p['prereq']}\n")
                if p["coreq"]:
                    f.write(f"COREQ:    {p['coreq']}\n")
                if p["credit_hours"]:
                    f.write(f"CREDITS:  {p['credit_hours']}\n")
                if p["min_grade"]:
                    f.write(f"MIN_GRADE:{p['min_grade']}\n")
                if p["instructor_consent"]:
                    f.write("CONSENT:  yes\n")

            f.write("CONTENT:\n")
            f.write(doc["content"])
            f.write("\n" + _THIN + "\n\n")

    print(f"\nok wrote {len(docs)} documents -> {RAW_CATALOG}")
    return len(docs)


def write_metadata(docs: list[dict], stats: dict[str, int]) -> dict:
    meta = {
        "date_scraped":             time.strftime("%Y-%m-%d"),
        "total_documents":          len(docs),
        "unique_source_urls":       len({d["source_url"] for d in docs}),
        "total_word_count":         sum(d["word_count"] for d in docs),
        "category_breakdown":       stats,
        "avg_words_per_doc":        round(
            sum(d["word_count"] for d in docs) / max(len(docs), 1), 1
        ),
        "sources": [
            {"doc_id": d["doc_id"], "url": d["source_url"],
             "words": d["word_count"], "category": d["category"]}
            for d in docs
        ],
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"ok wrote metadata    -> {METADATA_FILE}")
    return meta


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(meta: dict, stats: dict[str, int]) -> None:
    errors:   list[str] = []
    warnings: list[str] = []

    # Hard minimums from the assignment rubric
    if stats.get("course_page", 0) < 20:
        errors.append(f"Need ≥20 course pages; got {stats.get('course_page', 0)}")
    if stats.get("program_requirement", 0) < 2:
        errors.append(f"Need ≥2 program pages; got {stats.get('program_requirement', 0)}")
    if stats.get("policy", 0) < 1:
        errors.append(f"Need ≥1 policy page; got {stats.get('policy', 0)}")
    if meta["total_documents"] < 25:
        errors.append(f"Need ≥25 distinct documents; got {meta['total_documents']}")
    if meta["unique_source_urls"] != meta["total_documents"]:
        errors.append("Duplicate URLs detected — each document must map to a unique URL.")

    if meta["total_word_count"] < 30_000:
        warnings.append(
            f"Word count is {meta['total_word_count']:,} (< 30,000). "
            "Assignment says '30,000 words OR 25+ distinct pages' — "
            "you satisfy the page criterion if total_documents ≥ 25."
        )

    print("\n" + "=" * 64)
    print("VALIDATION REPORT")
    print("=" * 64)

    if errors:
        for e in errors:
            print(f"  x {e}")
        raise SystemExit("Validation FAILED — fix errors above before proceeding.")

    print(f"  ok course pages      : {stats.get('course_page', 0)}")
    print(f"  ok program pages     : {stats.get('program_requirement', 0)}")
    print(f"  ok policy pages      : {stats.get('policy', 0)}")
    print(f"  ok total documents   : {meta['total_documents']}")
    print(f"  ok unique URLs       : {meta['unique_source_urls']}")
    print(f"  ok total words       : {meta['total_word_count']:,}")
    print(f"  ok avg words/doc     : {meta['avg_words_per_doc']}")
    for w in warnings:
        print(f"  WARN {w}")
    print("\n  All checks passed. Run: python ingest.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print(" UTD 2025 Catalog Scraper  —  RAG Optimized")
    print("=" * 64)

    docs, stats = scrape_all()
    if not docs:
        raise SystemExit("No pages scraped. Check network access.")

    write_catalog(docs)
    meta = write_metadata(docs, stats)
    validate(meta, stats)