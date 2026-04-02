"""
sec_interpreter/eval.py

LLM-as-judge evaluation of interpretation output against reference criteria.

For each (obligation, criterion) pair, asks a cheap LLM:
  "Does this interpretation adequately cover this concept?"
  -> PASS or FAIL + one-sentence explanation

Usage:
  python -m sec_interpreter.eval --run-id <run_id> --criteria tests/eval_criteria.json

Saves: artifacts/{run_id}/eval_report.json
Prints: coverage table per obligation
"""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage

from .utils import get_logger, parse_json_object

logger = get_logger("sec_interpreter.eval")

_JUDGE_PROMPT = """\
You are evaluating whether a set of SEC compliance interpretations collectively covers a specific concept.

The interpretations below may have been split into multiple numbered obligations by an automated
pipeline. Evaluate the FULL SET together -- if any interpretation covers the criterion, that is a PASS.

INTERPRETATIONS:
{interpretation_text}

CRITERION:
{criterion}

Does the collective set of interpretations adequately cover this criterion?
- PASS: at least one interpretation contains this concept, either explicitly or by clear implication
- FAIL: the criterion is missing from all interpretations or contradicted

Respond with JSON only:
{{"result": "PASS" or "FAIL", "explanation": "one sentence explaining why"}}
"""


def run_eval(
    run_id: str,
    criteria_path: str,
    llm: Any,
) -> dict:
    """
    Run LLM-as-judge evaluation for a completed interpret run.

    run_id         -- identifies artifact directory (artifacts/{run_id})
    criteria_path  -- path to eval_criteria.json
    llm            -- cheap LLM (gpt-4o-mini style)

    Returns eval report dict. Also saves artifacts/{run_id}/eval_report.json.
    """
    artifact_dir = os.path.join("artifacts", run_id)

    # Load interpretation output
    interp_path = os.path.join(artifact_dir, "interpretation.json")
    if not os.path.exists(interp_path):
        raise FileNotFoundError(
            f"No interpretation.json for run_id={run_id}. Run interpret first."
        )
    with open(interp_path, encoding="utf-8") as f:
        interp_data = json.load(f)

    raw_interpretations = interp_data.get("interpretations", [])

    # Build a single combined text from ALL interpretations -- eval is ID-agnostic.
    # Each interpretation is labelled by its obligation_id so the judge has context,
    # but criteria are checked against the full set, not just the ID-matched entry.
    all_interp_text = _build_all_interp_text(raw_interpretations)

    # Load criteria
    with open(criteria_path, encoding="utf-8") as f:
        criteria_data = json.load(f)

    obligation_criteria = criteria_data.get("obligations", {})
    sources = criteria_data.get("sources", [])

    results = {}
    total = 0
    passed = 0

    for obl_id, obl_spec in obligation_criteria.items():
        obl_results = []
        criteria_list = obl_spec.get("criteria", [])

        for criterion in criteria_list:
            result, explanation = _judge(all_interp_text, criterion, llm)
            total += 1
            if result == "PASS":
                passed += 1
            obl_results.append({
                "criterion": criterion,
                "result": result,
                "explanation": explanation,
            })
            status = "PASS" if result == "PASS" else "FAIL"
            logger.info("  %s  [%s]  %s", obl_id, status, criterion[:70])

        results[obl_id] = {
            "description": obl_spec.get("description", ""),
            "criteria_results": obl_results,
            "pass_count": sum(1 for r in obl_results if r["result"] == "PASS"),
            "total_count": len(obl_results),
        }

    report = {
        "run_id": run_id,
        "rule_title": interp_data.get("rule_title", ""),
        "sources": sources,
        "summary": {
            "total_criteria": total,
            "passed": passed,
            "failed": total - passed,
            "coverage_pct": round(100 * passed / total, 1) if total else 0,
        },
        "obligations": results,
    }

    # Save report
    out_path = os.path.join(artifact_dir, "eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def _build_all_interp_text(interpretations: list) -> str:
    """
    Combine all ObligationInterpretation dicts into one text block for the judge.
    Each obligation is labelled so the judge can see the full picture.
    """
    sections = []
    for interp in interpretations:
        obl_id = interp.get("obligation_id", "?")
        parents = interp.get("parent_obligation_ids", [])
        label = obl_id
        if parents:
            label += f" (subset of {', '.join(parents)})"
        body = _build_interp_text(interp)
        sections.append(f"[{label}]\n{body}")
    return "\n\n---\n\n".join(sections)


def _build_interp_text(interp: dict) -> str:
    """Flatten an ObligationInterpretation dict into a single text block for the judge."""
    parts = []
    if interp.get("primary_interpretation"):
        parts.append("Primary: " + interp["primary_interpretation"])
    if interp.get("key_details"):
        parts.append("Key details:\n" + "\n".join(f"- {d}" for d in interp["key_details"]))
    if interp.get("compliance_implication"):
        parts.append("Compliance implication: " + interp["compliance_implication"])
    if interp.get("alternative_interpretations"):
        for alt in interp["alternative_interpretations"]:
            parts.append("Alternative reading: " + alt)
    if interp.get("ambiguous_terms"):
        parts.append("Ambiguous terms: " + ", ".join(interp["ambiguous_terms"]))
    if interp.get("supporting_sections"):
        parts.append("Sources: " + ", ".join(interp["supporting_sections"]))
    return "\n\n".join(parts)


def _judge(interp_text: str, criterion: str, llm: Any) -> tuple[str, str]:
    """
    Call LLM judge for one criterion. Returns ("PASS"|"FAIL", explanation).
    Defaults to FAIL on any error.
    """
    prompt = _JUDGE_PROMPT.format(
        interpretation_text=interp_text,
        criterion=criterion,
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_json_object(raw)
        result = str(parsed.get("result", "FAIL")).upper()
        explanation = str(parsed.get("explanation", ""))
        if result not in ("PASS", "FAIL"):
            result = "FAIL"
        return result, explanation
    except Exception as e:
        logger.warning("judge call failed for criterion %r: %s", criterion[:50], e)
        return "FAIL", f"Judge error: {e}"


def print_report(report: dict) -> None:
    """Print a human-readable coverage table to stdout."""
    summary = report["summary"]
    print("=" * 72)
    print(f"EVAL REPORT  run_id={report['run_id']}")
    print(f"Rule: {report['rule_title']}")
    print(f"Coverage: {summary['passed']}/{summary['total_criteria']} criteria passed ({summary['coverage_pct']}%)")
    print("=" * 72)

    for obl_id, obl_data in report["obligations"].items():
        if obl_data.get("skipped"):
            print(f"\n{obl_id}  [SKIPPED] {obl_data['reason']}")
            continue

        p = obl_data["pass_count"]
        t = obl_data["total_count"]
        print(f"\n{obl_id}  {obl_data['description']}  ({p}/{t})")
        print("-" * 72)
        for r in obl_data["criteria_results"]:
            tag = "PASS" if r["result"] == "PASS" else "FAIL"
            print(f"  [{tag}] {r['criterion']}")
            if r["result"] == "FAIL":
                print(f"         -> {r['explanation']}")

    print()
