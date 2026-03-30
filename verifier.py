"""
verifier.py — Verification & Auditing Module
Checks LLM outputs for missing citations, hallucinations, and prereq logic errors.
"""

import re
from typing import Dict, List, Any


class VerificationError(Exception):
    """Raised when the verifier detects a critical issue in the LLM output."""
    pass


# ── Patterns that signal incomplete/unsafe output ────────────────────────────
HALLUCINATION_PHRASES = [
    "based on my knowledge",
    "i believe",
    "i think",
    "as far as i know",
    "typically",
    "generally speaking",
    "usually",
    "in most cases",
    "from my understanding",
    "it is common for",
]

CITATION_PATTERN = re.compile(
    r"\[https?://[^\|\]]+\|[^\]]+\]|"     # [URL | section]
    r"\[https?://[^\]]+\]|"               # [URL]
    r"\[doc\d+_chunk\d+\]|"              # [doc0001_chunk002]
    r"\[chunk_\d+\]|"                     # [chunk_1]
    r"\[\d+\]",                           # [1], [2] (numbered citations)
    re.IGNORECASE,
)

SAFE_ABSTENTION_PHRASE = "i don't have that information in the provided catalog/policies"

# Factual claim indicators — lines with these should have citations
FACTUAL_CLAIM_INDICATORS = [
    "prerequisite",
    "prereq",
    "co-requisite",
    "corequisite",
    "required",
    "must complete",
    "must have",
    "grade of",
    "minimum grade",
    "credit",
    "subjects required",
    "units",
    "eligible",
    "not eligible",
    "permission",
    "instructor consent",
]


def check_citations_present(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that factual claims in evidence section contain citations.
    Returns a dict with pass/fail and issues list.
    """
    issues: List[str] = []
    warnings: List[str] = []

    evidence_text = parsed.get("evidence", "")
    why_text = parsed.get("why", "")
    answer_text = parsed.get("answer", "")
    reasoning_blob = f"{evidence_text}\n{why_text}"
    full_text = parsed.get("raw_response", "")

    # Check 1: If there are factual claims, there must be citations
    has_factual_claim = any(
        indicator in (reasoning_blob + answer_text).lower()
        for indicator in FACTUAL_CLAIM_INDICATORS
    )
    has_citations = bool(CITATION_PATTERN.search(full_text))
    citations_section = parsed.get("citations", "").strip()

    is_abstention = SAFE_ABSTENTION_PHRASE in full_text.lower()

    if has_factual_claim and not has_citations and not is_abstention:
        issues.append(
            "MISSING_CITATIONS: Factual claims present but no citations found. "
            "Every prerequisite/requirement claim must have a citation [URL | section]."
        )

    # Check 2: Citations section should not be empty if evidence exists
    if evidence_text and not citations_section and not is_abstention:
        warnings.append(
            "WEAK_CITATIONS: Evidence section exists but Citations section is empty."
        )

    # Check 3: Hallucination phrase detection
    for phrase in HALLUCINATION_PHRASES:
        if phrase in full_text.lower():
            issues.append(
                f"POTENTIAL_HALLUCINATION: Detected hedge phrase '{phrase}'. "
                "Only state facts that are explicitly in the retrieved context."
            )
            break

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "has_citations": has_citations,
        "is_abstention": is_abstention,
        "citation_count": len(CITATION_PATTERN.findall(full_text)),
    }


def check_prereq_logic(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate prerequisite reasoning logic.
    Checks for internal inconsistency (e.g. says Eligible but evidence says not).
    """
    issues: List[str] = []
    decision = parsed.get("decision", "").lower()
    evidence = (parsed.get("evidence", "") + "\n" + parsed.get("why", "")).lower()
    answer = parsed.get("answer", "").lower()

    # Check logical consistency between decision and evidence
    if "not eligible" in decision:
        if "is eligible" in evidence or "can take" in evidence:
            issues.append(
                "LOGIC_CONFLICT: Decision says 'Not Eligible' but evidence contains "
                "language suggesting eligibility. Check reasoning chain."
            )
    if "eligible" in decision and "not eligible" not in decision:
        if "cannot take" in evidence or "missing prerequisite" in evidence:
            issues.append(
                "LOGIC_CONFLICT: Decision says 'Eligible' but evidence mentions "
                "missing prerequisites. Check reasoning chain."
            )

    # Check: if need more info, clarifying questions should exist
    if "need more info" in decision:
        clarifying = parsed.get("clarifying_questions", "").strip()
        if not clarifying:
            issues.append(
                "MISSING_CLARIFYING_QUESTIONS: Decision is 'Need More Info' but "
                "no clarifying questions were provided."
            )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
    }


def check_safe_abstention(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that safe abstention is handled correctly.
    When abstaining, the system must use the exact phrase and suggest alternatives.
    """
    issues: List[str] = []
    full_text = parsed.get("raw_response", "").lower()

    is_abstention = SAFE_ABSTENTION_PHRASE in full_text

    if is_abstention:
        # Should suggest at least one alternative
        suggests_alternative = any(
            keyword in full_text
            for keyword in ["advisor", "department", "schedule of classes", "catalog.utdallas.edu"]
        )
        if not suggests_alternative:
            issues.append(
                "INCOMPLETE_ABSTENTION: Safe abstention phrase used but no alternative "
                "resources suggested (advisor, department page, schedule of classes)."
            )

    return {
        "is_abstention": is_abstention,
        "passed": len(issues) == 0,
        "issues": issues,
    }


def verify_output(parsed: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    """
    Master verification function.
    Runs all checks and returns a combined report.
    If strict=True, raises VerificationError on any issue.
    """
    citation_check = check_citations_present(parsed)
    logic_check = check_prereq_logic(parsed)
    abstention_check = check_safe_abstention(parsed)

    all_issues = (
        citation_check["issues"]
        + logic_check["issues"]
        + abstention_check["issues"]
    )
    all_warnings = citation_check.get("warnings", [])

    report = {
        "overall_passed": len(all_issues) == 0,
        "citation_check": citation_check,
        "logic_check": logic_check,
        "abstention_check": abstention_check,
        "all_issues": all_issues,
        "all_warnings": all_warnings,
        "citation_count": citation_check.get("citation_count", 0),
        "is_abstention": citation_check.get("is_abstention", False),
    }

    if strict and all_issues:
        raise VerificationError(
            "Output verification failed:\n" + "\n".join(f"  - {i}" for i in all_issues)
        )

    return report


def format_verification_report(report: Dict[str, Any]) -> str:
    """Format verification report for display."""
    status = "✅ PASSED" if report["overall_passed"] else "⚠️ ISSUES FOUND"
    lines = [f"**Verification: {status}**"]

    if report["citation_count"] > 0:
        lines.append(f"- Citations found: {report['citation_count']}")

    if report["is_abstention"]:
        lines.append("- Safe abstention: ✅")

    if report["all_issues"]:
        lines.append("\n**Issues:**")
        for issue in report["all_issues"]:
            lines.append(f"  - ⚠️ {issue}")

    if report["all_warnings"]:
        lines.append("\n**Warnings:**")
        for warning in report["all_warnings"]:
            lines.append(f"  - ℹ️ {warning}")

    return "\n".join(lines)
