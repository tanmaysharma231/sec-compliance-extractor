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

    # ------------------------------------------------------------------ scan
    scan_parser = subparsers.add_parser(
        "scan",
        help="Run structure scan on an already-ingested document (no LLM)",
    )
    scan_parser.add_argument("--run-id", required=True, help="run_id from a previous ingest run")

    # ------------------------------------------------------------------ bin
    bin_parser = subparsers.add_parser(
        "bin",
        help="Run bin pass (secondary LLM scan over flagged remaining chunks)",
    )
    bin_parser.add_argument("--run-id", required=True, help="run_id with validated_output.json")

    # ------------------------------------------------------------------ brief
    brief_parser = subparsers.add_parser(
        "brief",
        help="Generate a case brief from extract + interpret + bin outputs",
    )
    brief_parser.add_argument("--run-id", required=True, help="run_id with validated_output.json")
    brief_parser.add_argument("--output", required=True, help="Path to write the case brief (.md)")

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
    elif args.command == "scan":
        _cmd_scan(args)
    elif args.command == "bin":
        _cmd_bin(args)
    elif args.command == "brief":
        _cmd_brief(args)
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
    """Full pipeline: ingest -> extract -> bin -> interpret -> brief."""
    import os
    from langchain_core.messages import HumanMessage
    from .module import IngestModule, ExtractModule, InterpretModule, _load_env_llm, _load_cheap_llm, DeterministicLLM
    from .bin_graph import run_bin_pass
    from .prompts import build_case_brief_prompt
    from .schemas import BinPassOutput
    from .utils import get_logger

    logger = get_logger("sec_interpreter")
    source, page_range, strict = _resolve_source(args)

    print("Stage 1/4 — Ingesting document...")
    ingest_result = IngestModule().run(source, page_range=page_range, strict_citations=strict)
    print(f"  run_id={ingest_result.run_id}  chunks={ingest_result.chunk_count}")

    print("Stage 2/4 — Extracting compliance intelligence (structure scan + LLM)...")
    extract_module = ExtractModule()
    output = extract_module.run(ingest_result.run_id, strict_citations=strict)
    _write_output(output, args.output)

    artifact_dir = os.path.join("artifacts", ingest_result.run_id)

    print("Stage 3/4 — Bin pass (secondary scan over flagged chunks)...")
    cheap_llm = _load_cheap_llm() or extract_module.llm
    extraction_dict = output.model_dump(mode="json")
    bin_result = run_bin_pass(ingest_result.run_id, extraction_dict, cheap_llm, logger)
    print(f"  bin findings: {len(bin_result.findings)}")

    print("Stage 4/4 — Interpreting obligations...")
    interp_module = InterpretModule(llm=extract_module.llm, cheap_llm=cheap_llm)
    interp_output = interp_module.run(ingest_result.run_id)
    print(f"  interpretations: {len(interp_output.interpretations)}")

    print("Generating case brief...")
    named_texts = _load_named_section_texts(artifact_dir)
    brief_prompt = build_case_brief_prompt(
        extraction_dict,
        bin_result.model_dump(mode="json")["findings"],
        interp_output.model_dump(mode="json"),
        named_texts,
    )
    llm = extract_module.llm
    response = llm.invoke([HumanMessage(content=brief_prompt)])
    brief_text = response.content if hasattr(response, "content") else str(response)
    brief_path = Path(args.output).with_suffix(".brief.md")
    brief_path.write_text(brief_text, encoding="utf-8")
    print(f"Case brief written to {brief_path}")
    print(f"\nArtifacts: {artifact_dir}/")


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


def _cmd_scan(args) -> None:
    """Run structure scan on an already-ingested document and print the result."""
    import os
    from .structure import structure_scan

    artifact_dir = os.path.join("artifacts", args.run_id)
    if not os.path.exists(os.path.join(artifact_dir, "chunks.json")):
        raise FileNotFoundError(
            f"No chunks.json found for run_id={args.run_id}. Run ingest first."
        )

    result = structure_scan(artifact_dir)
    print(f"Structure scan complete for run_id={args.run_id}")
    print(f"  obligation sections: {len(result.obligation_sections)}")
    print(f"  expected obligations: {result.expected_obligation_count}")
    print(f"  structured chunks: {len(result.structured_chunk_ids)}")
    print(f"  named section chunks: {len(result.named_section_chunk_ids)}")
    for sec in result.obligation_sections:
        cfr = ", ".join(sec.cfr_citations) if sec.cfr_citations else "none"
        print(f"    [{sec.section_letter}] {sec.heading[:60]}  cfr={cfr}")


def _cmd_bin(args) -> None:
    """Run bin pass over flagged chunks not sent to the main extractor."""
    import os
    from .bin_graph import run_bin_pass
    from .module import _load_cheap_llm, _load_env_llm, DeterministicLLM
    from .utils import get_logger

    logger = get_logger("sec_interpreter.bin")
    artifact_dir = os.path.join("artifacts", args.run_id)
    output_path = os.path.join(artifact_dir, "validated_output.json")
    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"No validated_output.json for run_id={args.run_id}. Run extract first."
        )

    with open(output_path, encoding="utf-8") as f:
        extraction_output = json.load(f)

    cheap_llm = _load_cheap_llm() or _load_env_llm() or DeterministicLLM()
    result = run_bin_pass(args.run_id, extraction_output, cheap_llm, logger)

    missed = sum(1 for f in result.findings if f.finding_type == "missed_obligation")
    print(f"Bin pass complete: {len(result.findings)} findings ({missed} missed obligations)")
    print(f"  bin_findings.json written to {artifact_dir}/")


def _cmd_brief(args) -> None:
    """Generate a case brief from extract + interpret + bin outputs."""
    import os
    from langchain_core.messages import HumanMessage
    from .module import _load_env_llm, DeterministicLLM
    from .prompts import build_case_brief_prompt
    from .schemas import BinPassOutput

    artifact_dir = os.path.join("artifacts", args.run_id)

    extract_path = os.path.join(artifact_dir, "validated_output.json")
    if not os.path.exists(extract_path):
        raise FileNotFoundError(
            f"No validated_output.json for run_id={args.run_id}. Run extract first."
        )
    with open(extract_path, encoding="utf-8") as f:
        extraction_output = json.load(f)

    bin_path = os.path.join(artifact_dir, "bin_findings.json")
    if os.path.exists(bin_path):
        with open(bin_path, encoding="utf-8") as f:
            bin_output = BinPassOutput.model_validate(json.load(f))
    else:
        bin_output = BinPassOutput(run_id=args.run_id)

    interp_path = os.path.join(artifact_dir, "interpretation.json")
    interpretation_output = {}
    if os.path.exists(interp_path):
        with open(interp_path, encoding="utf-8") as f:
            interpretation_output = json.load(f)

    named_texts = _load_named_section_texts(artifact_dir)
    prompt = build_case_brief_prompt(
        extraction_output,
        bin_output.model_dump(mode="json")["findings"],
        interpretation_output,
        named_texts,
    )

    llm = _load_env_llm() or DeterministicLLM()
    response = llm.invoke([HumanMessage(content=prompt)])
    brief_text = response.content if hasattr(response, "content") else str(response)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(brief_text, encoding="utf-8")
    print(f"Case brief written to {out_path}")


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


def _load_named_section_texts(artifact_dir: str) -> list:
    """Return text strings for named section chunks (effective dates, scope, exemptions).

    Loads structure_scan_result.json -> named_section_chunk_ids, then looks up
    each chunk's text in chunks.json. Returns empty list if artifacts missing.
    """
    import os

    scan_path = os.path.join(artifact_dir, "structure_scan_result.json")
    chunks_path = os.path.join(artifact_dir, "chunks.json")
    if not os.path.exists(scan_path) or not os.path.exists(chunks_path):
        return []

    with open(scan_path, encoding="utf-8") as f:
        scan_data = json.load(f)
    named_ids = set(scan_data.get("named_section_chunk_ids", []))
    if not named_ids:
        return []

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    texts = []
    for chunk in chunks:
        if chunk.get("src_id") in named_ids:
            heading = chunk.get("heading_path", [])
            heading_str = " > ".join(heading) if heading else ""
            texts.append(f"[{heading_str}]\n{chunk.get('text', '')}")
    return texts


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
