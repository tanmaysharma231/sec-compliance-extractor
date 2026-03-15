from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SEC Rule Extractor — two-stage compliance intelligence pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1 — ingest a PDF from SEC.gov (pages 5-30)
  python -m sec_interpreter.cli ingest --url https://sec.gov/rules/final/2023/33-11216.pdf --pages 5-30

  # Stage 2 — extract from the ingested chunks
  python -m sec_interpreter.cli extract --run-id <run_id> --output out.json

  # Combined run (ingest + extract in one step)
  python -m sec_interpreter.cli run --url https://sec.gov/.../rule.pdf --pages 5-30 --output out.json

  # Local file
  python -m sec_interpreter.cli ingest --input path/to/rule.pdf --pages 1-50
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ ingest
    ingest_parser = subparsers.add_parser("ingest", help="Fetch and chunk a regulatory document")
    _add_source_args(ingest_parser)

    # ------------------------------------------------------------------ extract
    extract_parser = subparsers.add_parser(
        "extract", help="Run LLM extraction on an already-ingested document"
    )
    extract_parser.add_argument("--run-id", required=True, help="run_id from a previous ingest run")
    extract_parser.add_argument("--output", required=True, help="Path to write output JSON")
    extract_parser.add_argument(
        "--strict", action="store_true", help="Require every obligation to cite a source chunk"
    )

    # ------------------------------------------------------------------ run (combined)
    run_parser = subparsers.add_parser("run", help="Ingest + extract in a single step")
    _add_source_args(run_parser)
    run_parser.add_argument("--output", required=True, help="Path to write output JSON")

    # ------------------------------------------------------------------ classify
    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify every section by content type (run once after ingest)",
    )
    classify_parser.add_argument(
        "--run-id", required=True, help="run_id from a previous ingest run"
    )

    # ------------------------------------------------------------------ interpret
    interpret_parser = subparsers.add_parser(
        "interpret",
        help="Interpret extracted obligations with context assembly and CFR lookup",
    )
    interpret_parser.add_argument("--run-id", required=True, help="run_id with validated_output.json")
    interpret_parser.add_argument("--output", required=True, help="Path to write interpretation JSON")

    # ------------------------------------------------------------------ gap
    gap_parser = subparsers.add_parser(
        "gap",
        help="Generate a gap analysis report from an extracted output",
    )
    gap_parser.add_argument("--run-id", required=True, help="run_id with a validated_output.json")
    gap_parser.add_argument("--output", default=None, help="Path to write the report (default: print to stdout)")
    gap_parser.add_argument("--company", default="", help="One sentence describing the company (optional)")

    # ------------------------------------------------------------------ report
    report_parser = subparsers.add_parser(
        "report",
        help="Format a markdown compliance report from extract + interpret outputs",
    )
    report_parser.add_argument("--run-id", required=True, help="run_id with validated_output.json")
    report_parser.add_argument("--output", required=True, help="Path to write the markdown report")

    # ------------------------------------------------------------------ comprehend
    comprehend_parser = subparsers.add_parser(
        "comprehend",
        help="Calibration tool: classify every chunk and compare against locator selection",
    )
    comprehend_parser.add_argument(
        "--run-id", required=True, help="run_id from a previous ingest run"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "extract":
        _cmd_extract(args)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "classify":
        _cmd_classify(args)
    elif args.command == "interpret":
        _cmd_interpret(args)
    elif args.command == "gap":
        _cmd_gap(args)
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "comprehend":
        _cmd_comprehend(args)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_ingest(args) -> None:
    from .module import IngestModule

    source, page_range, strict = _resolve_source(args)
    result = IngestModule().run(source, page_range=page_range, strict_citations=strict)
    print(f"Ingest complete.")
    print(f"  run_id:      {result.run_id}")
    print(f"  chunks:      {result.chunk_count}")
    print(f"  artifacts:   {result.artifact_dir}/")
    print(f"\nNext step:")
    print(f"  python -m sec_interpreter.cli extract --run-id {result.run_id} --output out.json")


def _cmd_extract(args) -> None:
    from .module import ExtractModule

    module = ExtractModule()
    output = module.run(args.run_id, strict_citations=args.strict)
    _write_output(output, args.output)


def _cmd_run(args) -> None:
    """Ingest + extract in one step — convenience wrapper."""
    from .module import IngestModule, ExtractModule

    source, page_range, strict = _resolve_source(args)

    print("Stage 1/2 — Ingesting document...")
    ingest_result = IngestModule().run(source, page_range=page_range, strict_citations=strict)
    print(f"  run_id={ingest_result.run_id}  chunks={ingest_result.chunk_count}")

    print("Stage 2/2 — Extracting compliance intelligence...")
    output = ExtractModule().run(ingest_result.run_id, strict_citations=strict)
    _write_output(output, args.output)


def _cmd_classify(args) -> None:
    """Classify every section by content type and save artifacts."""
    from .module import ClassifyModule

    print(f"Classifying sections for run_id={args.run_id} ...")
    result = ClassifyModule().run(args.run_id)
    print(f"Classification complete.")
    print(f"  run_id:              {result['run_id']}")
    print(f"  sections classified: {result['section_count']}")
    print(f"  compliance sections: {result['compliance_section_count']}")
    print(f"  rule_title:          {result['rule_title']}")
    if result.get("regulatory_objective"):
        print(f"  objective:           {result['regulatory_objective']}")
    print(f"\nContent type distribution:")
    for ct, count in sorted(result["type_counts"].items()):
        print(f"  {ct:<22} {count}")
    print(f"\nNext step:")
    print(f"  python -m sec_interpreter.cli extract --run-id {result['run_id']} --output out.json")


def _cmd_interpret(args) -> None:
    """Run interpretation pipeline on extracted obligations."""
    from .module import InterpretModule

    print(f"Interpreting obligations for run_id={args.run_id} ...")
    module = InterpretModule()
    output = module.run(args.run_id)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    print(f"Interpretation complete: {len(output.interpretations)} obligations")
    print(f"Output written to {out_path}")


def _cmd_gap(args) -> None:
    """Generate a plain-English gap analysis from a validated_output.json."""
    import os
    from langchain_core.messages import HumanMessage
    from .module import _load_env_llm
    from .prompts import build_gap_analysis_prompt

    output_path = os.path.join("artifacts", args.run_id, "validated_output.json")
    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"No validated_output.json found for run_id={args.run_id}. Run extract first."
        )

    with open(output_path, encoding="utf-8") as f:
        extracted = json.load(f)

    prompt = build_gap_analysis_prompt(extracted, company_context=args.company)
    llm = _load_env_llm()
    if llm is None:
        raise RuntimeError("No LLM configured. Set SEC_INTERPRETER_MODEL in your environment.")
    response = llm.invoke([HumanMessage(content=prompt)])
    report = response.content if hasattr(response, "content") else str(response)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Gap analysis written to {args.output}")
    else:
        print(report)


def _cmd_report(args) -> None:
    """Format a markdown compliance report from extract + interpret outputs."""
    from pathlib import Path
    from .report_formatter import format_report

    print(f"Generating report for run_id={args.run_id} ...")
    report = format_report(args.run_id)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")


def _cmd_comprehend(args) -> None:
    """Calibration tool: classify every chunk and compare against locator."""
    from .comprehend import run_comprehend
    run_comprehend(args.run_id)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_source_args(p: argparse.ArgumentParser) -> None:
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="URL of an SEC regulatory document (PDF or HTML)")
    group.add_argument("--input", help="Local file path (.pdf, .txt, or .json with rule_text key)")
    p.add_argument("--pages", help="Page range for PDFs, e.g. '1-50'", default=None)
    p.add_argument("--strict", action="store_true", help="Enable strict citation mode")


def _resolve_source(args) -> tuple[str, tuple[int, int] | None, bool]:
    """Return (source_str, page_range, strict_citations) from parsed args."""
    page_range = _parse_page_range(getattr(args, "pages", None))
    strict = getattr(args, "strict", False)

    if getattr(args, "url", None):
        return args.url, page_range, strict

    input_path = Path(args.input)
    suffix = input_path.suffix.lower()

    if suffix in (".pdf", ".txt"):
        return str(input_path), page_range, strict

    # JSON file with rule_text key — extract the text and write a temp txt
    raw = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if "rule_text" not in raw:
        raise ValueError("Input JSON must contain a 'rule_text' key")
    # Write to a temp file so IngestModule can read it via fetch_rule_text
    tmp = input_path.with_suffix(".tmp.txt")
    tmp.write_text(raw["rule_text"], encoding="utf-8")
    strict = raw.get("strict_citations", strict)
    return str(tmp), page_range, strict


def _write_output(output, path_str: str) -> None:
    from .schemas import RuleExtractorOutput
    output_path = Path(path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    print(f"Output written to {output_path}")


def _parse_page_range(pages_arg: str | None) -> tuple[int, int] | None:
    if not pages_arg:
        return None
    parts = pages_arg.split("-")
    if len(parts) != 2:
        raise ValueError(f"--pages must be 'start-end', e.g. '1-50'. Got: {pages_arg!r}")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"--pages values must be integers. Got: {pages_arg!r}")
    if start < 1 or end < start:
        raise ValueError(f"--pages range invalid: {pages_arg!r}")
    return (start, end)


if __name__ == "__main__":
    main()
